#!/usr/bin/env python3
"""
Knowledge Base Sync for AI Portfolio Assistant
Supports Qdrant (with FastEmbed) and Zvec (local ONNX) backends.

Usage:
    python scripts/sync_knowledge_base.py
    python scripts/sync_knowledge_base.py --qdrant-url https://your-cloud.qdrant.io
"""

import json
import os
import re
import sys
from pathlib import Path

from config import settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Document, PointStruct
except ImportError:
    QdrantClient = None
    Document = None
    PointStruct = None

try:
    from embedding_client import EmbeddingClient
except ImportError:
    EmbeddingClient = None

try:
    from vector_store import ZvecVectorStore
except ImportError:
    ZvecVectorStore = None

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEGAL_DOCS_DIR = DATA_DIR / "legal-docs"
PROJECT_DOCS_DIR = DATA_DIR / "project-docs"

COLLECTION_NAME = settings.rag.collection_name


class KnowledgeBaseSync:
    def __init__(self, qdrant_url=None, qdrant_api_key=None):
        self.documents = []
        self.metadata = []
        self.handled_files = set()

        if settings.vector.backend == "qdrant":
            self.client = QdrantClient(
                url=qdrant_url or settings.qdrant.url,
                api_key=qdrant_api_key or settings.qdrant.api_key,
            )
        else:
            self.client = None

    def _load_json(self, filepath: Path) -> dict | list | None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Skipping {filepath}: {e}")
            return None

    def _chunk_markdown(self, markdown: str) -> list[dict]:
        chunks = []
        sections = re.split(r"\n## ", markdown)
        for section in sections:
            if not section.strip():
                continue
            lines = section.strip().split("\n")
            title = lines[0].strip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            if body:
                chunks.append({"title": title, "body": body})
        return chunks

    def ingest_profile(self):
        profile = self._load_json(DATA_DIR / "profile.json")
        if not profile:
            return 0
        self.handled_files.add(DATA_DIR / "profile.json")
        text = f"{profile['name']}, {profile.get('jobTitle', '')}."
        self.documents.append(text)
        self.metadata.append({"type": "profile", "source": "profile.json"})
        print(f"Profile ingested")
        return 1

    def ingest_experience(self):
        data = self._load_json(DATA_DIR / "experience.json")
        if not data or "experience" not in data:
            return 0
        self.handled_files.add(DATA_DIR / "experience.json")
        count = 0
        for exp in data["experience"]:
            company = exp.get("company", "")
            role = exp.get("role", "")
            start = exp.get("startDate", "")
            end = exp.get("endDate", "Present")
            for detail in exp.get("details", []):
                text = f"At {company} as {role} ({start} to {end}): {detail}"
                self.documents.append(text)
                self.metadata.append(
                    {
                        "type": "experience",
                        "company": company,
                        "role": role,
                        "source": "experience.json",
                    }
                )
                count += 1
        print(f"Experience: {count} entries")
        return count

    def ingest_projects(self):
        data = self._load_json(DATA_DIR / "projects.json")
        if not data or "projects" not in data:
            return 0
        self.handled_files.add(DATA_DIR / "projects.json")
        count = 0
        for proj in data["projects"]:
            title = proj.get("title", "")
            desc = proj.get("description", "")
            techs = [
                t["name"] if isinstance(t, dict) else t
                for t in proj.get("technologies", [])
            ]
            slug = proj.get("slug", "")
            text = f"Project: {title}. {desc} Technologies: {', '.join(techs)}"
            self.documents.append(text)
            self.metadata.append(
                {
                    "type": "project",
                    "project_slug": slug,
                    "project_title": title,
                    "tech_stack": techs,
                    "source": "projects.json",
                }
            )
            count += 1
        print(f"Projects: {count} entries")
        return count

    def ingest_case_studies(self):
        if not PROJECT_DOCS_DIR.exists():
            return 0
        count = 0
        for project_dir in PROJECT_DOCS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            case_study_path = project_dir / "case-study.md"
            if not case_study_path.exists():
                continue
            slug = project_dir.name
            try:
                content = case_study_path.read_text(encoding="utf-8")
                self.handled_files.add(case_study_path)
                chunks = self._chunk_markdown(content)
                for chunk in chunks:
                    text = f"Case Study - {slug} - {chunk['title']}: {chunk['body']}"
                    words = text.split()
                    if len(words) > 500:
                        text = " ".join(words[:500])
                    self.documents.append(text)
                    self.metadata.append(
                        {
                            "type": "case_study",
                            "project_slug": slug,
                            "section_title": chunk["title"],
                            "source": f"project-docs/{slug}/case-study.md",
                            "full_case_study_url": f"https://ankitjang.one/case-studies/{slug}",
                        }
                    )
                    count += 1
            except Exception as e:
                print(f"  [!] Error reading {case_study_path}: {e}")
        print(f"  [+] Case Studies: {count} sections")
        return count

    def ingest_education(self):
        data = self._load_json(DATA_DIR / "education.json")
        if not data or "education" not in data:
            return 0
        self.handled_files.add(DATA_DIR / "education.json")
        count = 0
        for edu in data["education"]:
            text = (
                f"Education: {edu.get('degree', '')} from "
                f"{edu.get('institution', '')}, {edu.get('location', '')} "
                f"({edu.get('duration', '')})"
            )
            self.documents.append(text)
            self.metadata.append(
                {
                    "type": "education",
                    "source": "education.json",
                }
            )
            count += 1
        print(f"  [+] Education: {count} entries")
        return count

    def ingest_legal_pdfs(self):
        """Ingest PDF documents from legal-docs directory."""
        from pdf_ingest import parse_legal_pdf, extract_legal_metadata, chunk_pdf

        if not LEGAL_DOCS_DIR.exists():
            print(f"  [*] Legal docs directory not found: {LEGAL_DOCS_DIR}")
            return 0

        pdf_files = list(LEGAL_DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"  [*] No PDF files found in {LEGAL_DOCS_DIR}")
            return 0

        count = 0
        for pdf_path in pdf_files:
            print(f"  [*] Processing {pdf_path.name}...")

            try:
                parsed = parse_legal_pdf(pdf_path)

                legal_meta = extract_legal_metadata(parsed["full_text"], pdf_path.name)

                doc_metadata = {
                    "type": "legal_case",
                    "source": pdf_path.name,
                    "file_path": str(pdf_path),
                    "page_count": parsed["metadata"]["page_count"],
                    **legal_meta,
                }

                chunks = chunk_pdf(
                    parsed["pages"],
                    chunk_size=400,
                    overlap=80,
                    doc_metadata=doc_metadata,
                )

                for chunk in chunks:
                    self.documents.append(chunk["text"])
                    self.metadata.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "page_num": chunk["page_num"],
                            "word_count": chunk["word_count"],
                            **doc_metadata,
                        }
                    )

                count += 1
                self.handled_files.add(pdf_path)
                print(f"      + {len(chunks)} chunks from {pdf_path.name}")

            except Exception as e:
                print(f"  [!] Error processing {pdf_path.name}: {e}")

        print(f"  [+] Legal PDFs: {count} files processed")
        return count

    def ingest_generic_data(self):
        """Recursively ingest any file in DATA_DIR not already handled."""
        print(f"[*] Scanning for additional data in {DATA_DIR}...")
        count = 0
        skipped = 0

        ignored_extensions = {
            ".tex",
            ".j2",
            ".py",
            ".pyc",
            ".DS_Store",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
        }

        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                filepath = Path(root) / file

                if filepath.suffix in ignored_extensions:
                    continue
                if filepath in self.handled_files:
                    continue

                if any(
                    part.startswith(".")
                    for part in filepath.relative_to(DATA_DIR).parts
                ):
                    continue

                try:
                    relative_path = filepath.relative_to(DATA_DIR)

                    if filepath.suffix == ".md":
                        content = filepath.read_text(encoding="utf-8")
                        chunks = self._chunk_markdown(content)
                        for chunk in chunks:
                            text = f"Doc: {relative_path} - {chunk['title']}: {chunk['body']}"
                            self.documents.append(text)
                            self.metadata.append(
                                {
                                    "type": "generic_markdown",
                                    "source": str(relative_path),
                                    "section_title": chunk["title"],
                                }
                            )
                            count += 1

                    elif filepath.suffix == ".json":
                        data = self._load_json(filepath)
                        if data:
                            text = (
                                f"Data: {relative_path}: {json.dumps(data, indent=2)}"
                            )
                            self.documents.append(text)
                            self.metadata.append(
                                {"type": "generic_json", "source": str(relative_path)}
                            )
                            count += 1
                    else:
                        try:
                            content = filepath.read_text(encoding="utf-8")
                            if content.strip():
                                text = f"File: {relative_path}: {content}"
                                self.documents.append(text)
                                self.metadata.append(
                                    {
                                        "type": "generic_text",
                                        "source": str(relative_path),
                                    }
                                )
                                count += 1
                        except UnicodeDecodeError:
                            skipped += 1
                            continue

                    self.handled_files.add(filepath)

                except Exception as e:
                    print(f"  [!] Error processing {filepath}: {e}")

        print(
            f"  [+] Generic Data: {count} additional entries (skipped {skipped} binary/ignored)"
        )
        return count

    def _load_documents(self):
        """Load all documents from data sources."""
        print(f"[*] Loading documents...")
        print(f"   Data dir: {DATA_DIR}")
        print(f"   Legal docs: {LEGAL_DOCS_DIR}")
        print(f"   Project docs: {PROJECT_DOCS_DIR}")
        print()

        # Legal PDFs (main demo data)
        self.ingest_legal_pdfs()

        # Portfolio data (disabled for legal demo)
        # self.ingest_profile()
        # self.ingest_experience()
        # self.ingest_projects()
        # self.ingest_case_studies()
        # self.ingest_education()
        # self.ingest_generic_data()

        if not self.documents:
            print("\nNo documents to sync")
            return False
        return True

    def sync(self):
        """Main sync method - dispatches to appropriate backend."""
        print("[*] Syncing knowledge base...")
        print(f"   Backend: {settings.vector.backend}")

        if settings.vector.backend == "zvec":
            return self._sync_zvec()
        else:
            return self._sync_qdrant()

    def _sync_zvec(self):
        """Sync documents to Zvec with local ONNX embeddings."""
        print("[*] Syncing knowledge base to Zvec with local ONNX embeddings...")

        if not self._load_documents():
            return 0

        print(f"\nEmbedding {len(self.documents)} documents locally (ONNX)...")
        embedder = EmbeddingClient()

        batch_size = 32
        all_embeddings = []
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i : i + batch_size]
            all_embeddings.extend(embedder.embed_batch(batch))
            print(
                f"  Embedded {min(i + batch_size, len(self.documents))}/{len(self.documents)}"
            )

        print(f"Uploading to Zvec ({settings.vector.store_path})...")
        store = ZvecVectorStore()
        store.recreate()
        store.upsert(self.documents, all_embeddings, self.metadata)
        print(f"\n[+] Synced {len(self.documents)} documents to Zvec")
        return len(self.documents)

    def _sync_qdrant(self):
        """Sync documents to Qdrant with FastEmbed."""
        print("[*] Syncing knowledge base to Qdrant with FastEmbed...")

        if not self._load_documents():
            return 0

        print(f"\nUploading {len(self.documents)} documents to Qdrant...")

        import uuid

        print(f"Recreating collection '{settings.rag.collection_name}'...")

        self.client.set_sparse_model(settings.rag.sparse_model)

        if self.client.collection_exists(settings.rag.collection_name):
            self.client.delete_collection(settings.rag.collection_name)

        self.client.create_collection(
            collection_name=settings.rag.collection_name,
            vectors_config=self.client.get_fastembed_vector_params(),
            sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
        )

        points = []
        vector_params = self.client.get_fastembed_vector_params()
        sparse_params = self.client.get_fastembed_sparse_vector_params()
        vector_name = list(vector_params.keys())[0]
        sparse_vector_name = list(sparse_params.keys())[0]
        model_name = settings.rag.embedding_model

        for doc, meta in zip(self.documents, self.metadata):
            payload = meta.copy()
            payload["document"] = doc

            points.append(
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector={
                        vector_name: Document(text=doc, model=model_name),
                        sparse_vector_name: Document(
                            text=doc, model=settings.rag.sparse_model
                        ),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=settings.rag.collection_name,
            points=points,
        )

        print(
            f"\n[+] Synced {len(self.documents)} documents to '{settings.rag.collection_name}'"
        )
        print("   (Qdrant automatically embedded them using FastEmbed)")
        return len(self.documents)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync portfolio to Qdrant")
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant.url,
    )
    parser.add_argument("--qdrant-api-key", default=settings.qdrant.api_key)
    args = parser.parse_args()
    syncer = KnowledgeBaseSync(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )
    syncer.sync()
