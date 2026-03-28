#!/usr/bin/env python3
"""
FastAPI backend for Legal RAG Assistant

Usage:
    cd src && uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import (
        Depends,
        FastAPI,
        HTTPException,
        Request,
        Security,
        UploadFile,
        File,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    from config import settings
    from rag_pipeline import RAGPipeline
    from utils.logger import api_logger
except ImportError:
    print("Install: pip install -r scripts/requirements-ai.txt")
    sys.exit(1)

# --- Auth dependency ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate API key. Skip in dev mode (no key configured)."""
    if not settings.api.secret_key:
        return  # Dev mode - no auth required
    if api_key != settings.api.secret_key:
        raise HTTPException(401, "Invalid API key")


# --- App setup ---
app = FastAPI(title=settings.legal.app_name)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.api.cors_origins.split(",")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Concurrency control
_semaphore = asyncio.Semaphore(settings.api.max_concurrent_requests)

# Initialize RAG pipeline once
rag = RAGPipeline(
    vector_backend=settings.vector.backend,
    llm_provider=settings.llm.provider,
    enable_reranking=settings.rag.enable_reranking,
)


class PlaygroundSettings(BaseModel):
    system_prompt: str | None = None
    intent_prompt: str | None = None
    guardrail_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=4000)
    top_k: int | None = Field(None, ge=1, le=20)
    enable_intent_detection: bool | None = None
    enable_guardrails: bool | None = None


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    history: list[dict] | None = (
        None  # [{"role": "user"|"assistant", "content": "..."}]
    )
    settings: PlaygroundSettings | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict | None = None


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict
    snippets: list[dict]
    page_num: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int
    retrieval_time_ms: int


@app.post("/api/chat")
@limiter.limit(settings.api.rate_limit_per_ip)
async def chat(req: ChatRequest, request: Request, _=Depends(verify_api_key)):
    start_time = time.time()

    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if len(req.question) > settings.api.max_question_length:
        raise HTTPException(
            400,
            f"Question too long (max {settings.api.max_question_length} chars)",
        )

    provider = rag.llm_client.provider
    model = rag.llm_client.config.get("model", "unknown")

    # Sanitize history: cap at 10 entries, strip to role+content only
    sanitized_history = None
    if req.history:
        sanitized_history = [
            {"role": h["role"], "content": h["content"][:500]}
            for h in req.history[-10:]
            if isinstance(h, dict)
            and h.get("role") in ("user", "assistant")
            and h.get("content")
        ]

    api_logger.info(
        f'POST /api/chat | q="{req.question}" | provider={provider} | model={model}'
        f" | history={len(sanitized_history) if sanitized_history else 0}"
    )

    async def generate():
        chunk_count = 0
        first_token_time = None
        try:
            async with _semaphore:
                gen = rag.safe_query_stream(
                    req.question,
                    history=sanitized_history,
                    override_settings=req.settings.model_dump()
                    if req.settings
                    else None,
                )
                sentinel = object()
                loop = asyncio.get_event_loop()
                while True:
                    event = await loop.run_in_executor(None, next, gen, sentinel)
                    if event is sentinel:
                        break
                    chunk_count += 1
                    elapsed = int((time.time() - start_time) * 1000)
                    if event.get("type") == "token":
                        if first_token_time is None:
                            first_token_time = elapsed
                            api_logger.debug(f"  [FIRST TOKEN] at {elapsed}ms")
                        api_logger.debug(
                            f"  chunk#{chunk_count} @{elapsed}ms | "
                            f"token[{len(event.get('content', ''))}]={event.get('content', '')[:50]!r}"
                        )
                    elif event.get("type") == "done":
                        api_logger.info(
                            f"  [DONE] stream done | chunks={chunk_count} | elapsed={elapsed}ms | "
                            f"first_token={first_token_time or 'N/A'}ms | "
                            f"confidence={event.get('confidence', 'N/A')} | "
                            f"sources={len(event.get('sources', []))}"
                        )
                    sse_line = f"data: {json.dumps(event)}\n\n"
                    api_logger.debug(f"  [YIELD] SSE: {sse_line[:100]!r}")
                    yield sse_line
        except Exception as e:
            api_logger.error(f"  [ERROR] stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        response_time = int((time.time() - start_time) * 1000)
        api_logger.info(
            f'  [COMPLETE] response complete | q="{req.question}" | '
            f"total_time={response_time}ms | chunks={chunk_count} | "
            f"provider={provider} | model={model}"
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/search")
@limiter.limit(settings.api.rate_limit_per_ip)
async def search(req: SearchRequest, request: Request, _=Depends(verify_api_key)):
    """Semantic search across legal documents without LLM generation."""
    from snippet_extractor import extract_snippet

    start_time = time.time()

    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    api_logger.info(f'POST /api/search | q="{req.query}" | top_k={req.top_k}')

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, rag.retrieve, req.query, req.top_k, None, req.filters
    )

    search_results = []
    for r in results:
        snippets = extract_snippet(r.text, req.query)
        search_results.append(
            SearchResult(
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                snippets=snippets,
                page_num=r.metadata.get("page_num"),
            )
        )

    retrieval_time = int((time.time() - start_time) * 1000)

    api_logger.info(
        f'POST /api/search | q="{req.query}" | results={len(results)} | time={retrieval_time}ms'
    )

    return SearchResponse(
        query=req.query,
        results=search_results,
        total_results=len(results),
        retrieval_time_ms=retrieval_time,
    )


class CaseResponse(BaseModel):
    cases: list
    total: int
    limit: int
    offset: int


@app.get("/api/cases")
@limiter.limit(settings.api.rate_limit_per_ip)
async def list_cases(
    request: Request,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    case_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    _=Depends(verify_api_key),
):
    """List all indexed cases with optional filtering and pagination."""
    api_logger.info(
        f"GET /api/cases | court={court} | year_from={year_from} | "
        f"year_to={year_to} | case_type={case_type} | limit={limit} | offset={offset}"
    )

    loop = asyncio.get_event_loop()
    all_docs = await loop.run_in_executor(None, rag.vector_store.get_all)

    unique_cases = {}
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        if source and source not in unique_cases:
            unique_cases[source] = {
                "case_name": doc.metadata.get("case_name", ""),
                "citation": doc.metadata.get("citation", ""),
                "court": doc.metadata.get("court", ""),
                "judge": doc.metadata.get("judge", ""),
                "year": doc.metadata.get("year", ""),
                "case_type": doc.metadata.get("case_type", ""),
            }

    cases = list(unique_cases.values())

    if court:
        cases = [c for c in cases if c["court"].lower() == court.lower()]
    if case_type:
        cases = [c for c in cases if c["case_type"].lower() == case_type.lower()]
    if year_from is not None:
        cases = [c for c in cases if c["year"] and int(c["year"]) >= year_from]
    if year_to is not None:
        cases = [c for c in cases if c["year"] and int(c["year"]) <= year_to]

    total = len(cases)
    paginated = cases[offset : offset + limit]

    api_logger.info(f"GET /api/cases | filtered={total} | returned={len(paginated)}")

    return CaseResponse(
        cases=paginated,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/filters")
@limiter.limit(settings.api.rate_limit_per_ip)
async def get_filter_options(request: Request, _=Depends(verify_api_key)):
    """Get available filter values from indexed data."""
    api_logger.info("GET /api/filters")

    loop = asyncio.get_event_loop()
    all_docs = await loop.run_in_executor(None, rag.vector_store.get_all)

    courts = set()
    judges = set()
    case_types = set()
    years = []

    for doc in all_docs:
        court = doc.metadata.get("court", "").strip()
        if court:
            courts.add(court)

        judge = doc.metadata.get("judge", "").strip()
        if judge:
            judges.add(judge)

        case_type = doc.metadata.get("case_type", "").strip()
        if case_type:
            case_types.add(case_type)

        year = doc.metadata.get("year", "")
        if year:
            try:
                years.append(int(year))
            except (ValueError, TypeError):
                pass

    year_range = {}
    if years:
        year_range = {"min": min(years), "max": max(years)}

    api_logger.info(
        f"GET /api/filters | courts={len(courts)} | judges={len(judges)} | "
        f"case_types={len(case_types)} | years={year_range}"
    )

    return {
        "courts": sorted(courts),
        "judges": sorted(judges),
        "case_types": sorted(case_types),
        "year_range": year_range,
    }


class CaseDetailResponse(BaseModel):
    case_id: str
    case_name: str
    citation: str
    court: str
    judge: str
    year: int | str | None
    case_type: str
    chunks: list
    full_text: str


@app.get("/api/cases/{case_id}")
@limiter.limit(settings.api.rate_limit_per_ip)
async def get_case(case_id: str, request: Request, _=Depends(verify_api_key)):
    """Get full case document with all chunks."""
    api_logger.info(f"GET /api/cases/{case_id}")

    loop = asyncio.get_event_loop()
    all_docs = await loop.run_in_executor(None, rag.vector_store.get_all)

    case_chunks = []
    case_metadata = {}

    for doc in all_docs:
        source = doc.metadata.get("source", "")
        if source == case_id:
            case_chunks.append(
                {
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "page_num": doc.metadata.get("page_num"),
                    "text": doc.text,
                }
            )
            if not case_metadata:
                case_metadata = {
                    "case_name": doc.metadata.get("case_name", ""),
                    "citation": doc.metadata.get("citation", ""),
                    "court": doc.metadata.get("court", ""),
                    "judge": doc.metadata.get("judge", ""),
                    "year": doc.metadata.get("year", ""),
                    "case_type": doc.metadata.get("case_type", ""),
                }

    if not case_chunks:
        raise HTTPException(404, f"Case '{case_id}' not found")

    full_text = "\n\n".join(c["text"] for c in case_chunks)

    api_logger.info(f"GET /api/cases/{case_id} | chunks={len(case_chunks)}")

    return CaseDetailResponse(
        case_id=case_id,
        case_name=case_metadata.get("case_name", ""),
        citation=case_metadata.get("citation", ""),
        court=case_metadata.get("court", ""),
        judge=case_metadata.get("judge", ""),
        year=case_metadata.get("year", ""),
        case_type=case_metadata.get("case_type", ""),
        chunks=case_chunks,
        full_text=full_text,
    )


# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "legal-docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".xml"}


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_indexed: int
    message: str


@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request, file: UploadFile = File(...), _=Depends(verify_api_key)
):
    """Upload a legal document (PDF or TXT), parse, embed, and index it live."""
    from pdf_ingest import (
        parse_legal_pdf,
        parse_legal_xml,
        parse_text_file,
        extract_legal_metadata,
        extract_xml_metadata,
        chunk_pdf,
    )
    from embedding_client import EmbeddingClient

    # Validate file type
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save file to disk
    doc_id = str(uuid.uuid4())[:8]
    safe_name = f"{doc_id}_{filename}"
    file_path = UPLOAD_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    api_logger.info(f"Upload saved: {filename} -> {file_path}")

    # Parse, chunk, embed, and index
    try:
        if suffix == ".pdf":
            parsed = parse_legal_pdf(file_path)
            legal_meta = extract_legal_metadata(parsed["full_text"], filename)
        elif suffix == ".xml":
            parsed = parse_legal_xml(file_path)
            legal_meta = extract_xml_metadata(parsed, filename)
        else:
            parsed = parse_text_file(file_path)
            legal_meta = extract_legal_metadata(parsed["full_text"], filename)
        doc_metadata = {
            "type": "legal_case",
            "source": filename,
            "file_path": str(file_path),
            "page_count": parsed["metadata"]["page_count"],
            **legal_meta,
        }

        chunks = chunk_pdf(
            parsed["pages"],
            chunk_size=400,
            overlap=80,
            doc_metadata=doc_metadata,
        )

        if not chunks:
            raise ValueError("No text content extracted from file")

        # Embed chunks
        embedder = EmbeddingClient()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        # Build metadata for each chunk
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "page_num": c["page_num"],
                "word_count": c["word_count"],
                **doc_metadata,
            }
            for c in chunks
        ]

        # Upsert into the live vector store
        rag.vector_store.upsert(texts, embeddings, metadatas, id_prefix=filename)

        api_logger.info(
            f"Ingested: {filename} | chunks={len(chunks)} | doc_id={doc_id}"
        )

        return UploadResponse(
            document_id=doc_id,
            filename=filename,
            status="indexed",
            chunks_indexed=len(chunks),
            message=f"Document uploaded and indexed ({len(chunks)} chunks). Ready for search.",
        )

    except Exception as e:
        api_logger.error(f"Ingest failed for {filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = int((time.time() - start_time) * 1000)

    client_ip = request.client.host if request.client else "unknown"
    api_logger.info(
        f"{request.method} {request.url.path} | ip={client_ip} | status={response.status_code} | time={process_time}ms"
    )

    return response


@app.get("/api/health")
async def health():
    try:
        # Check backend is responsive
        if settings.vector.backend == "zvec":
            _ = rag.vector_store.count()
        else:
            _ = rag.client.get_collection(rag.collection)

        return {"status": "ok"}
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        return {"status": "error"}


@app.get("/")
async def root():
    return {
        "message": f"{settings.legal.app_name} API",
        "version": "1.0.0",
    }


# Static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/demo")
async def demo():
    """Serve the demo HTML page."""
    demo_file = static_dir / "demo.html"
    if demo_file.exists():
        return FileResponse(str(demo_file))
    return {"message": "Demo page not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
