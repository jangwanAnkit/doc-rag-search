#!/usr/bin/env python3
"""
RAG Pipeline for Ankit's Portfolio Assistant
Uses Qdrant FastEmbed for retrieval, Gemini for generation.

Usage:
    python scripts/rag_pipeline.py "What projects has Ankit built?"
    python scripts/rag_pipeline.py --interactive
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generator, List, Optional

from config import settings
from llm_client import LLMClient
from utils.logger import qdrant_logger, llm_logger

intent_logger = logging.getLogger("intent")
intent_logger.setLevel(logging.DEBUG)

INTENT_SYSTEM_PROMPT = """You are a legal casebase intent classifier. Determine if the user's question is related to legal research, case law, legal documents, or legal information in the loaded casebase.

Legal-related topics include:
- Questions about court cases, judgments, rulings
- Questions about legal principles, doctrines, statutes
- Questions about contracts, agreements, legal terms
- Questions about legal procedures, litigation, trials
- Questions about jurisdiction, applicable law
- Questions about legal rights, obligations, liabilities
- Questions about evidence, witnesses, testimony
- Questions about damages, remedies, sentencing

NOT legal-related:
- General conversation or greetings
- Questions about non-legal topics (weather, sports, etc.)
- Questions about personal advice not related to law
- Questions about the assistant itself

Classify as YES if the question is about legal topics and the casebase can potentially help.
Classify as NO if it's not related to legal research.

Respond ONLY with valid JSON in this format:
{"should_proceed": true/false, "reason": "brief reason", "confidence": 0.0-1.0}"""

GUARDRAIL_SYSTEM_PROMPT = """You are a legal query guardrail. Analyze the user's question for potential issues.

Check for:
1. Potential PII (personally identifiable information) in the query
2. Harmful or inappropriate content
3. Queries trying to extract specific personal information about individuals from the casebase
4. Queries seeking privileged or confidential information

Return JSON with:
{"is_safe": true/false, "reason": "brief reason", "requires_disclaimer": true/false}

Only block if the query is clearly malicious, seeks to extract personal info about non-public figures from cases, or contains harmful content."""

LEGAL_SYSTEM_PROMPT = """You are a legal research assistant helping lawyers and legal professionals find relevant case law and legal information.

**Your role:**
- Answer questions based ONLY on the provided legal documents
- Cite specific cases, sections, and page numbers when referencing information
- Quote relevant passages verbatim when appropriate
- Clearly distinguish between holdings, dicta, and procedural history
- Flag jurisdictional differences when relevant
- If information is not in the provided context, say so explicitly

**Response format:**
- Start with a direct answer to the question
- Support with specific case citations and quotes
- Reference page numbers when available
- Note any limitations or caveats

**Do not:**
- Provide legal advice
- Make assumptions beyond the provided documents
- Cite cases not in the retrieval context
- Summarize without attribution"""

# Qdrant imports are deferred to __init__ — only loaded when backend=qdrant


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: dict


@dataclass
class RAGResponse:
    answer: str
    sources: List[RetrievalResult]
    confidence: float


class RAGPipeline:
    def __init__(
        self,
        qdrant_url: str = settings.qdrant.url,
        qdrant_api_key: str = settings.qdrant.api_key,
        collection_name: str = settings.rag.collection_name,
        llm_provider: str = settings.llm.provider,
        skip_llm: bool = False,
        hybrid_search: bool = settings.rag.hybrid_search,
        vector_backend: str = settings.vector.backend,
        enable_reranking: bool = True,
    ):
        self.vector_backend = vector_backend
        self.collection = collection_name
        self.hybrid_search = hybrid_search
        self.enable_reranking = enable_reranking and (vector_backend == "zvec")

        if self.vector_backend == "zvec":
            from embedding_client import EmbeddingClient
            from vector_store import ZvecVectorStore

            self.vector_store = ZvecVectorStore()
            self.embedding_client = EmbeddingClient()
        else:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client not installed. Run: pip install 'qdrant-client[fastembed]' "
                    "or set VECTOR_BACKEND=zvec in .env"
                )

            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            if self.hybrid_search:
                self.client.set_sparse_model(settings.rag.sparse_model)

        if not skip_llm:
            self.llm_client = LLMClient(
                provider=llm_provider,
                fallback_provider=settings.llm.fallback_provider,
            )
        else:
            self.llm_client = None

        self.reranker = None
        if self.enable_reranking:
            try:
                from reranker import Reranker

                self.reranker = Reranker()
            except Exception as e:
                print(f"Warning: Failed to load reranker: {e}")

        self.system_prompt = LEGAL_SYSTEM_PROMPT
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.top_k = settings.rag.top_k
        self.enable_intent_detection = settings.legal.enable_intent_detection
        self.enable_guardrails = settings.legal.enable_guardrails
        self._override_system_prompt = None
        self._default_temperature = settings.llm.temperature
        self._default_max_tokens = settings.llm.max_tokens
        self._default_top_k = settings.rag.top_k
        self._default_max_sources = getattr(settings.api, "max_sources", 3)

    def _check_legal_intent(self, query: str) -> dict:
        """Check if query is legal-related using LLM with JSON mode."""
        if not settings.legal.enable_intent_detection:
            return {"should_proceed": True, "reason": "disabled", "confidence": 1.0}

        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}"},
        ]

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.get("model", "llama-3.3-70b"),
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200,
            )

            result = json.loads(response.choices[0].message.content)

            intent_logger.info(
                f'intent_check | query="{query[:50]}" | '
                f"should_proceed={result.get('should_proceed')} | "
                f"confidence={result.get('confidence')} | "
                f"reason={result.get('reason')}"
            )

            return result

        except Exception as e:
            intent_logger.warning(
                f"Intent detection failed: {e}, proceeding by default"
            )
            return {
                "should_proceed": True,
                "reason": "intent_detection_failed",
                "confidence": 0.5,
            }

    def _check_guardrails(self, query: str) -> dict:
        """Check query against guardrails for safety and compliance."""
        if not settings.legal.enable_guardrails:
            return {"is_safe": True, "reason": "disabled", "requires_disclaimer": False}

        messages = [
            {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ]

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.get("model", "llama-3.3-70b"),
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200,
            )

            result = json.loads(response.choices[0].message.content)

            intent_logger.debug(
                f'guardrail_check | query="{query[:50]}" | '
                f"is_safe={result.get('is_safe')} | "
                f"requires_disclaimer={result.get('requires_disclaimer')}"
            )

            return result

        except Exception as e:
            intent_logger.warning(f"Guardrail check failed: {e}, allowing query")
            return {
                "is_safe": True,
                "reason": "check_failed",
                "requires_disclaimer": False,
            }

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        filters: dict = None,
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents"""
        if top_k is None:
            top_k = settings.rag.top_k
        if score_threshold is None:
            if self.vector_backend == "zvec":
                score_threshold = settings.vector.score_threshold
            else:
                score_threshold = settings.rag.score_threshold

        initial_k = top_k * 4 if self.reranker else top_k

        if self.vector_backend == "zvec":
            initial_results = self._retrieve_zvec(
                query, initial_k, score_threshold, filters
            )
        else:
            initial_results = self._retrieve_qdrant(query, initial_k, score_threshold)

        if self.reranker and initial_results:
            reranked = self.reranker.rerank(query, initial_results, top_k=top_k)
            return [
                RetrievalResult(
                    text=r.text,
                    score=r.rerank_score,
                    metadata={**r.metadata, "original_score": r.score},
                )
                for r in reranked
            ]

        return initial_results[:top_k]

    def _retrieve_zvec(
        self,
        query: str,
        top_k: int,
        score_threshold: float = None,
        filters: dict = None,
    ) -> List[RetrievalResult]:
        """Retrieve using Zvec backend."""
        if score_threshold is None:
            score_threshold = settings.vector.score_threshold
        start_time = time.time()
        try:
            query_embedding = self.embedding_client.embed(query)
            results = self.vector_store.search(
                query_embedding, top_k, score_threshold, filters
            )
            retrieval_results = [
                RetrievalResult(
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ]

            response_time = int((time.time() - start_time) * 1000)
            scores = [round(r.score, 3) for r in results]

            qdrant_logger.debug(
                f'backend=zvec | query="{query}" | results={len(results)} | scores={scores} | time={response_time}ms'
            )

            return retrieval_results
        except Exception as e:
            qdrant_logger.error(f"Zvec retrieval error: {e}")
            return []

    def _retrieve_qdrant(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[RetrievalResult]:
        """Retrieve using Qdrant backend."""
        from qdrant_client.models import Document, Prefetch, FusionQuery, Fusion

        start_time = time.time()
        try:
            if self.hybrid_search:
                # Hybrid Search: Dense + Sparse with RRF Fusion
                dense_model = settings.rag.embedding_model
                sparse_model = settings.rag.sparse_model
                dense_using = "fast-bge-small-en"
                sparse_using = f"fast-sparse-{sparse_model.split('/')[-1].lower().replace('_', '-')}"
                prefetch = [
                    Prefetch(
                        query=Document(text=query, model=dense_model),
                        using=dense_using,
                        limit=top_k,
                    ),
                    Prefetch(
                        query=Document(text=query, model=sparse_model),
                        using=sparse_using,
                        limit=top_k,
                    ),
                ]

                hits = self.client.query_points(
                    collection_name=self.collection,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                    # Score threshold might behave differently with fusion scores, removing it for hybrid or adjusting?
                    # RRF scores are usually lower/different. Let's keep it optional or lower it.
                    # For now, let's accept all RRF results as they constitute top_k
                ).points
            else:
                # Standard Semantic Search
                dense_model = settings.rag.embedding_model
                dense_using = "fast-bge-small-en"
                hits = self.client.query_points(
                    collection_name=self.collection,
                    query=Document(text=query, model=dense_model),
                    limit=top_k,
                    with_payload=True,
                    score_threshold=score_threshold,
                    using=dense_using,
                ).points

            results = []
            for hit in hits:
                # Qdrant with FastEmbed stores text in the vector-associated document or we put it in payload
                # In sync_knowledge_base.py, we now explicitly put it in payload['document']
                text = hit.payload.get("document", "")
                if not text:
                    # Fallback: try to reconstruct from other fields if 'document' is missing
                    text = f"{hit.payload.get('source', '')}: {hit.payload.get('section_title', '')}"

                results.append(
                    RetrievalResult(
                        text=text,
                        metadata=hit.payload,
                        score=hit.score,
                    )
                )

            response_time = int((time.time() - start_time) * 1000)
            scores = [round(hit.score, 3) for hit in hits]  # Use 'hits' here

            qdrant_logger.debug(
                f'backend=qdrant | query="{query}" | results={len(hits)} | scores={scores} | time={response_time}ms'
            )
            qdrant_logger.debug(f"Retrieved documents: {results}")

            return results  # Return 'results' list
        except Exception as e:
            qdrant_logger.error(f"Retrieval error: {e}")
            return []

    def _build_prompt(
        self,
        query: str,
        context: List[RetrievalResult],
        history: Optional[List[dict]] = None,
    ) -> str:
        context_text = "\n\n".join(
            [
                f"[Source: {r.metadata.get('source', 'unknown')} | Page {r.metadata.get('page_num', '?')}] {r.text}"
                for r in context
            ]
        )

        # Build conversation history block (last 5 exchanges max)
        history_block = ""
        if history:
            sanitized = [
                h
                for h in history[-10:]
                if h.get("role") in ("user", "assistant") and h.get("content")
            ]
            if sanitized:
                lines = []
                for h in sanitized:
                    role = "User" if h["role"] == "user" else "Assistant"
                    lines.append(f"{role}: {h['content'][:500]}")
                history_block = f"\n\nPrevious conversation:\n" + "\n".join(lines)

        system_prompt = (
            getattr(self, "_override_system_prompt", None) or self.system_prompt
        )

        return f"""{system_prompt}
{history_block}

Context:
{context_text}

Question: {query}

Answer:"""

    def _generate_llm(self, prompt: str) -> str:
        """Generate response using configured LLM provider"""
        start_time = time.time()
        try:
            answer = self.llm_client.generate(
                prompt,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

            response_time = int((time.time() - start_time) * 1000)
            model = self.llm_client.config.get("model", "unknown")
            prompt_preview = prompt[:200].replace("\n", " ")

            llm_logger.info(
                f'model={model} | prompt="{prompt_preview}..." | time={response_time}ms | answer_len={len(answer)}'
            )

            return answer
        except Exception as e:
            llm_logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"

    def query(self, question: str) -> RAGResponse:
        # 0. Check intent first
        if self.llm_client and settings.legal.enable_intent_detection:
            intent_result = self._check_legal_intent(question)

            if not intent_result.get("should_proceed", True):
                fallback_answer = (
                    f"I'm a legal research assistant that can help you find relevant case law and legal information. "
                    f"Please ask questions about legal topics, court cases, or legal principles. "
                    f"{settings.legal.disclaimer}"
                )
                return RAGResponse(
                    answer=fallback_answer,
                    sources=[],
                    confidence=0.0,
                )

        # 1. Retrieve relevant documents
        results = self.retrieve(question)

        if not results:
            return RAGResponse(
                answer="I don't have information about that in the current legal document collection. "
                f"{settings.legal.disclaimer}",
                sources=[],
                confidence=0.0,
            )

        # 2. Build prompt with context
        prompt = self._build_prompt(question, results)

        # 3. Generate with LLM
        answer = self._generate_llm(prompt)

        # 4. Calculate confidence (average retrieval score)
        confidence = sum(r.score for r in results) / len(results)

        return RAGResponse(answer=answer, sources=results, confidence=confidence)

    def query_stream(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        override_settings: Optional[dict] = None,
    ) -> Generator[dict, None, None]:
        """Stream response with metadata at the end.

        Runs suggestion generation in parallel with the main answer stream.
        """
        if override_settings:
            if override_settings.get("system_prompt"):
                self._override_system_prompt = override_settings["system_prompt"]
            if override_settings.get("temperature"):
                self.temperature = override_settings["temperature"]
            if override_settings.get("max_tokens"):
                self.max_tokens = override_settings["max_tokens"]
            if override_settings.get("top_k"):
                self.top_k = override_settings["top_k"]
            if "enable_intent_detection" in override_settings:
                self.enable_intent_detection = override_settings[
                    "enable_intent_detection"
                ]
            if "enable_guardrails" in override_settings:
                self.enable_guardrails = override_settings["enable_guardrails"]

        # Check intent first
        if self.llm_client and self.enable_intent_detection:
            intent_result = self._check_legal_intent(question)

            if not intent_result.get("should_proceed", True):
                yield {
                    "type": "done",
                    "answer": (
                        f"I'm a legal research assistant that can help you find relevant case law and legal information. "
                        f"Please ask questions about legal topics, court cases, or legal principles. "
                        f"{settings.legal.disclaimer}"
                    ),
                    "sources": [],
                    "confidence": 0.0,
                    "suggested_questions": [],
                }
                return

        results = self.retrieve(question)

        if not results:
            yield {
                "type": "done",
                "answer": f"I don't have information about that in the current legal document collection. {settings.legal.disclaimer}",
                "sources": [],
                "confidence": 0.0,
                "suggested_questions": [],
            }
            return

        prompt = self._build_prompt(question, results, history=history)
        confidence = sum(r.score for r in results) / len(results)

        # Fire suggestion call in parallel (runs while main answer streams)
        suggestions_future = None
        if self.llm_client:
            context_snippets = [r.text for r in results[:3]]
            executor = ThreadPoolExecutor(max_workers=1)
            suggestions_future = executor.submit(
                self.llm_client.get_suggestions,
                question,
                context_snippets,
                history,
            )

        # Stream the main answer
        for token in self.llm_client.generate_stream(prompt):
            yield {"type": "token", "content": token}

        # Collect suggestions (should be done by now or very soon)
        suggested_questions = []
        if suggestions_future:
            try:
                suggested_questions = suggestions_future.result(timeout=10)
            except Exception as e:
                llm_logger.warning(f"Suggestions timed out or failed: {e}")
            finally:
                executor.shutdown(wait=False)

        yield {
            "type": "done",
            "sources": [
                {
                    "type": s.metadata.get("type", "legal_case"),
                    "source": s.metadata.get("source", ""),
                    "case_name": s.metadata.get("case_name", ""),
                    "citation": s.metadata.get("citation", ""),
                    "court": s.metadata.get("court", ""),
                    "page_num": s.metadata.get("page_num"),
                    "score": round(s.score, 3),
                }
                for s in results[: self._default_max_sources]
            ],
            "confidence": round(confidence, 3),
            "suggested_questions": suggested_questions,
        }

        # Reset to defaults after request
        self.temperature = self._default_temperature
        self.max_tokens = self._default_max_tokens
        self.top_k = self._default_top_k
        if hasattr(self, "_override_system_prompt"):
            del self._override_system_prompt

    def safe_query_stream(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        override_settings: Optional[dict] = None,
    ) -> Generator[dict, None, None]:
        """Query with safety checks, streaming version."""
        is_safe, block_msg = self.is_safe_query(question)
        if not is_safe:
            yield {
                "type": "done",
                "answer": block_msg,
                "sources": [],
                "confidence": 1.0,
                "suggested_questions": [],
            }
            return
        yield from self.query_stream(
            question, history=history, override_settings=override_settings
        )

    def is_safe_query(self, query: str) -> tuple[bool, str]:
        """Block sensitive queries using word boundaries + LLM guardrails."""
        import re

        if not settings.legal.enable_guardrails:
            return True, ""

        blocked_patterns = {
            "social security": f"This query references sensitive identification. {settings.legal.disclaimer}",
            "ssn": f"This query references sensitive identification. {settings.legal.disclaimer}",
            "driver license": f"This query references sensitive identification. {settings.legal.disclaimer}",
            "credit card": f"This query references sensitive financial information. {settings.legal.disclaimer}",
            "bank account": f"This query references sensitive financial information. {settings.legal.disclaimer}",
        }

        query_lower = query.lower()
        for pattern, response in blocked_patterns.items():
            if re.search(r"\b" + re.escape(pattern) + r"\b", query_lower):
                return False, response

        if self.llm_client:
            llm_check = self._check_guardrails(query)
            if not llm_check.get("is_safe", True):
                reason = llm_check.get("reason", "blocked_by_guardrail")
                return False, f"Query blocked: {reason}. {settings.legal.disclaimer}"

        return True, ""

    def safe_query(self, question: str) -> RAGResponse:
        """Query with safety checks"""
        is_safe, block_msg = self.is_safe_query(question)
        if not is_safe:
            return RAGResponse(answer=block_msg, sources=[], confidence=1.0)
        return self.query(question)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--qdrant-url", default=settings.qdrant.url)
    parser.add_argument("--provider", default=settings.llm.provider)
    parser.add_argument(
        "--retrieve-only", action="store_true", help="Only retrieve documents, skip LLM"
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Use Hybrid Search (Dense + Sparse)"
    )
    args = parser.parse_args()

    rag = RAGPipeline(
        qdrant_url=args.qdrant_url,
        llm_provider=args.provider,
        skip_llm=args.retrieve_only,
        hybrid_search=args.hybrid,
    )

    if args.interactive:
        print("[BOT] Legal Casebase Assistant (type 'quit' to exit)\n")
        print(f"Disclaimer: {settings.legal.disclaimer}\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break

            if args.retrieve_only:
                sources = rag.retrieve(q)
                print(f"[Retrieval Only] Found {len(sources)} sources:")
                for i, s in enumerate(sources, 1):
                    source_type = s.metadata.get("type", "?")
                    source_name = s.metadata.get("source", "?")
                    section = s.metadata.get("section_title", "")

                    print(f"  {i}. [{s.score:.3f}] {source_type}: {source_name}")
                    if section:
                        print(f"     Section: {section}")
                    print(f"     Preview: {s.text[:500]}...")  # Show more text
                    print("-" * 50)  # Separator
            else:
                result = rag.safe_query(q)
                print(f"\nAssistant: {result.answer}")
                print(f"  [Confidence: {result.confidence:.2f}]")
                print(f"  [Sources: {len(result.sources)}]")
                for s in result.sources[:3]:
                    print(
                        f"    - {s.metadata.get('source', '?')} (score: {s.score:.3f})"
                    )
                print()
    elif args.question:
        if args.retrieve_only:
            sources = rag.retrieve(args.question)
            print(f"[Retrieval Only] Found {len(sources)} sources:")
            for i, s in enumerate(sources, 1):
                source_type = s.metadata.get("type", "?")
                source_name = s.metadata.get("source", "?")
                section = s.metadata.get("section_title", "")

                print(f"  {i}. [{s.score:.3f}] {source_type}: {source_name}")
                if section:
                    print(f"     Section: {section}")
                print(f"     Preview: {s.text[:500]}...")  # Show more text
                print("-" * 50)  # Separator
        else:
            result = rag.safe_query(args.question)
            print(f"\n{result.answer}\n")
            print(f"Confidence: {result.confidence:.2f}")
            for s in result.sources:
                print(
                    f"  [{s.score:.3f}] {s.metadata.get('type')}: {s.metadata.get('source')}"
                )
