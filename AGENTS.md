# FastAPI RAG Engine - AI Agent Documentation

## Project Overview

**Purpose:** RAG Engine Backend Demo
**Type:** FastAPI backend with RAG
**Primary Use:** Search-first legal document retrieval with optional AI chat
**Status:** ✅ Fully working - search, browse, filter, view case details, AI chat

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| Vector Database | Zvec (local, primary) |
| Embeddings | BAAI/bge-small-en-v1.5 (384 dimensions) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Cerebras (llama3.1-8b) with fallback to Gemini/OpenRouter |
| PDF Processing | PyMuPDF |
| Package Manager | UV |

---

## Project Structure

```
legal-rag/
├── src/
│   ├── api_server.py          # FastAPI endpoints
│   ├── rag_pipeline.py        # RAG orchestration + retrieval
│   ├── pdf_ingest.py          # PDF + TXT parsing & chunking
│   ├── reranker.py            # Cross-encoder reranking
│   ├── snippet_extractor.py   # Keyword highlighting
│   ├── sync_knowledge_base.py # Batch document ingestion
│   ├── config.py              # Settings (Pydantic)
│   ├── llm_client.py         # LLM provider abstraction
│   ├── embedding_client.py    # Embedding generation
│   └── vector_store.py       # Zvec backend
├── static/
│   └── demo.html             # Demo UI (search-first casebase interface)
├── data/
│   └── legal-docs/           # PDF & TXT documents
├── docs/
│   └── plans/                # Implementation plans
└── pyproject.toml            # Dependencies
```

---

## Key Files

### `src/api_server.py`
- FastAPI application with rate limiting, CORS, API key auth
- Endpoints: `/api/chat`, `/api/search`, `/api/upload`, `/api/health`, `/api/cases`, `/api/cases/{id}`, `/api/filters`

### `src/rag_pipeline.py`
- `RAGPipeline` class handles retrieval + generation
- Methods: `retrieve()`, `query()`, `query_stream()`, `safe_query_stream()`
- Supports metadata filters in retrieval

---

## API Endpoints

### GET /api/health
Health check - verifies vector store connectivity.

```bash
curl http://localhost:8000/api/health
# {"status":"ok"}
```

### GET /api/filters
Get available filter options for the UI.

```bash
curl http://localhost:8000/api/filters
# {"courts": ["FCA", "NSWSC"], "judges": [...], "case_types": [], "year_range": {"min": 2022, "max": 2024}}
```

### GET /api/cases
List all indexed cases with pagination and filtering.

```bash
curl "http://localhost:8000/api/cases?limit=10&offset=0"
# {"cases": [...], "total": 3, "limit": 10, "offset": 0}
```

### GET /api/cases/{case_id}
Get full case document with all chunks.

```bash
curl "http://localhost:8000/api/cases/contract_case_001.pdf"
# {"case_id": "...", "case_name": "...", "citation": "...", "court": "...", "chunks": [...], "full_text": "..."}
```

### POST /api/search
Semantic search with metadata filtering.

**Request:**
```json
{
  "query": "breach of contract damages",
  "top_k": 5,
  "filters": {
    "court": "High Court",
    "year_from": 2020,
    "year_to": 2024,
    "case_type": "contract"
  }
}
```

### POST /api/chat
Conversational Q&A with streaming response (SSE).

**Request:**
```json
{
  "question": "What did the court decide about damages?",
  "history": [],
  "settings": {
    "temperature": 0.3,
    "max_tokens": 500,
    "top_k": 5
  }
}
```

### POST /api/upload
Upload and auto-index a document (PDF or TXT).

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@case.pdf"
```

---

## Demo Interface

Visit http://localhost:8000/demo

### Layout
```
[Filters ◀] | [Search + Results] | [Chat ▶]
```

- **Left sidebar** (collapsible): Court, case type, year range filters
- **Center panel**: Search bar + case cards with relevance scores + snippets
- **Right sidebar** (collapsible): AI chat, opens on search

### Tabs
- **Search** - Semantic search with filters
- **Browse Cases** - Grid view of all indexed cases with pagination
- **Playground** - Configure prompts, temperature, tokens, safety features

### Features
- Case cards with metadata badges (court, judge, year, citation)
- Color-coded relevance score (green >70%, yellow 40-70%, red <40%)
- "View Full Case" modal with complete document text
- "Ask AI" button per case for contextual Q&A
- Keyboard shortcuts: `/` search, `s` search tab, `b` browse, `c` chat, `?` help
- Loading skeletons and toast notifications

---

## Running the Project

### 1. Install Dependencies
```bash
cd legal-rag
uv sync
```

### 2. Generate Test PDFs
```bash
cd src
uv run python -m pdf_ingest --generate-test-pdfs
```

### 3. Sync Knowledge Base
```bash
cd src
uv run python sync_knowledge_base.py
```

### 4. Start Server
```bash
cd src
uv run uvicorn api_server:app --reload --port 8000
```

### 5. Open Demo
Visit http://localhost:8000/demo

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_BACKEND` | zvec | Backend: zvec or qdrant |
| `VECTOR_STORE_PATH` | .vector_store | Local vector store directory |
| `RAG_TOP_K` | 5 | Number of results to retrieve |
| `RAG_ENABLE_RERANKING` | true | Enable cross-encoder reranking |
| `LLM_PROVIDER` | cerebras | LLM: cerebras, gemini, openrouter |
| `API_SECRET_KEY` | (empty) | API key for authentication |

---

## Dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
qdrant-client[fastembed]>=1.7.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
openai>=1.6.0
slowapi>=0.1.9
sentence-transformers>=2.2.0
pymupdf>=1.24.0
python-magic>=0.4.27
reportlab>=4.0.0
zvec>=0.1.0
python-multipart>=0.0.6
```

---

## Deployment (Railway)

```toml
# railway.toml
[build]
builder = "NIXPACKS"
buildCommand = "uv sync"

[deploy]
startCommand = "cd src && uvicorn api_server:app --host 0.0.0.0 --port $PORT"
```

Required env vars for Railway:
- `CEREBRAS_API_KEY` (or Gemini/OpenRouter)
- `VECTOR_BACKEND=zvec`
- `API_SECRET_KEY` (set for production)
```
