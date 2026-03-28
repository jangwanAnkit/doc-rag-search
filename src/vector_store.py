import os
from pathlib import Path
from typing import List, Optional

import zvec
from zvec import CollectionOption
from pydantic import BaseModel

from config import settings


class SearchResult(BaseModel):
    text: str
    metadata: dict
    score: float


class ZvecVectorStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        store_path: Optional[str] = None,
    ):
        self.collection_name = collection_name or settings.rag.collection_name
        self.dimensions = dimensions or settings.vector.dimensions
        self.store_path = store_path or settings.vector.store_path
        self._schema = self._build_schema()

    def _build_schema(self) -> zvec.CollectionSchema:
        return zvec.CollectionSchema(
            name=self.collection_name,
            fields=[
                zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="source", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="doc_type", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="court", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="judge", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="year", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="case_type", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="case_name", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="citation", data_type=zvec.DataType.STRING),
            ],
            vectors=[
                zvec.VectorSchema(
                    name="default",
                    dimension=self.dimensions,
                    data_type=zvec.DataType.VECTOR_FP32,
                    index_param=zvec.HnswIndexParam(
                        metric_type=zvec.MetricType.COSINE,
                    ),
                )
            ],
        )

    def _get_or_create_collection(self) -> zvec.Collection:
        if os.path.exists(self.store_path):
            option = CollectionOption(read_only=0)
            return zvec.open(path=self.store_path, option=option)

        return zvec.create_and_open(
            path=self.store_path,
            schema=self._schema,
        )

    def _open_collection(self, read_only: bool = False) -> zvec.Collection:
        option = CollectionOption(read_only=1 if read_only else 0)
        return zvec.open(path=self.store_path, option=option)

    def upsert(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        id_prefix: Optional[str] = None,
    ) -> None:
        """Insert documents with their embeddings.

        Doc IDs are derived from source metadata to avoid collisions
        when uploading documents incrementally.
        """
        metadatas = metadatas or [{}] * len(documents)

        docs = []
        for i, (doc_text, emb, meta) in enumerate(
            zip(documents, embeddings, metadatas)
        ):
            # Build a stable, unique ID from source filename + chunk index
            source = meta.get("source", id_prefix or "doc")
            doc_id = f"{source}_{i}"

            docs.append(
                zvec.Doc(
                    id=doc_id,
                    vectors={"default": emb},
                    fields={
                        "text": doc_text,
                        "source": meta.get("source", ""),
                        "doc_type": meta.get("type", ""),
                        "court": meta.get("court", ""),
                        "judge": meta.get("judge", ""),
                        "year": str(meta.get("year", "")),
                        "case_type": meta.get("case_type", ""),
                        "case_name": meta.get("case_name", ""),
                        "citation": meta.get("citation", ""),
                    },
                )
            )

        collection = self._get_or_create_collection()
        collection.insert(docs)
        collection.flush()
        del collection

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        collection = self._open_collection(read_only=True)
        try:
            results = collection.query(
                vectors=zvec.VectorQuery(
                    field_name="default",
                    vector=query_embedding,
                ),
                topk=top_k * 4 if filters else top_k,
            )
        finally:
            del collection
        search_results = []
        for result in results:
            if score_threshold is not None and result.score < score_threshold:
                continue
            fields = result.fields
            if filters:
                match = True
                for key, value in filters.items():
                    if fields.get(key, "").lower() != value.lower():
                        match = False
                        break
                if not match:
                    continue
            metadata = {
                "source": fields.get("source", ""),
                "type": fields.get("doc_type", ""),
                "court": fields.get("court", ""),
                "judge": fields.get("judge", ""),
                "year": fields.get("year", ""),
                "case_type": fields.get("case_type", ""),
                "case_name": fields.get("case_name", ""),
                "citation": fields.get("citation", ""),
            }
            search_results.append(
                SearchResult(
                    text=fields.get("text", ""),
                    metadata=metadata,
                    score=result.score,
                )
            )
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]

    def get_all(self) -> List[SearchResult]:
        """Retrieve all documents from the vector store using a high top_k query."""
        from embedding_client import EmbeddingClient

        embedder = EmbeddingClient()
        query_emb = embedder.embed("the")

        return self.search(query_emb, top_k=1000)

    def count(self) -> int:
        collection = self._open_collection(read_only=True)
        try:
            stats = collection.stats
            return stats.doc_count
        finally:
            del collection

    def recreate(self) -> None:
        """Delete and recreate the collection."""
        import shutil

        # Remove the entire store path to clean up any stale lock files
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        collection = zvec.create_and_open(path=self.store_path, schema=self._schema)
        del collection
