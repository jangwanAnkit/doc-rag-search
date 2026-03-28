from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

_ENV_FILE = str(Path(__file__).parent.parent / ".env")
_ENV_COMMON = {"env_file": _ENV_FILE, "env_file_encoding": "utf-8"}


class LLMConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "LLM_", "extra": "ignore"}

    provider: str = "cerebras"
    fallback_provider: Optional[str] = None
    model_override: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 500
    timeout: int = 300
    retry_delay: int = 1


class VectorConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "VECTOR_", "extra": "ignore"}

    backend: str = "zvec"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    store_path: str = ".vector_store"
    score_threshold: float = 0.25  # Lower for Zvec (scores are 0.25-0.40)


class RAGConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "RAG_", "extra": "ignore"}

    collection_name: str = "legal_casebase"
    top_k: int = 5
    score_threshold: float = 0.80
    hybrid_search: bool = False
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class APIConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "API_", "extra": "ignore"}

    secret_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    max_question_length: int = 500
    max_sources: int = 3
    rate_limit_per_ip: str = "20/minute"
    rate_limit_global: str = "200/minute"
    max_concurrent_requests: int = 10
    uvicorn_workers: int = 2
    debug: bool = False


class QdrantConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "QDRANT_", "extra": "ignore"}

    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30


class LogConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "LOG_", "extra": "ignore"}

    level: str = "INFO"
    dir: str = "logs"
    backup_count: int = 7


class LegalConfig(BaseSettings):
    model_config = {**_ENV_COMMON, "env_prefix": "LEGAL_", "extra": "ignore"}

    app_name: str = "Legal Casebase Search"
    owner_name: str = "Legal Team"
    owner_email: str = "legal@example.com"
    disclaimer: str = "This is a legal research tool. Not legal advice."
    enable_guardrails: bool = True
    enable_intent_detection: bool = True


class Settings(BaseSettings):
    model_config = {**_ENV_COMMON, "extra": "ignore"}

    cerebras_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    public_ai_api_url: Optional[str] = None

    llm: LLMConfig = LLMConfig()
    vector: VectorConfig = VectorConfig()
    rag: RAGConfig = RAGConfig()
    api: APIConfig = APIConfig()
    qdrant: QdrantConfig = QdrantConfig()
    log: LogConfig = LogConfig()
    legal: LegalConfig = LegalConfig()

    def get_api_key(self, provider: str) -> str:
        key_map = {
            "cerebras": self.cerebras_api_key,
            "gemini": self.gemini_api_key,
            "nvidia": self.nvidia_api_key,
            "openrouter": self.openrouter_api_key,
            "openrouter_meta": self.openrouter_api_key,
        }
        key = key_map.get(provider.lower())
        if not key:
            env_var = f"{provider.upper()}_API_KEY"
            raise ValueError(f"Set {env_var} in .env")
        return key


settings = Settings()

# Set FastEmbed cache to a persistent path (not /tmp/ which gets cleared on reboot)
import os

os.environ.setdefault("FASTEMBED_CACHE_PATH", settings.vector.store_path)
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
