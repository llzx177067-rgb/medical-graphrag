"""Centralised settings loaded from .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- DashScope ----
    base_url: str = "https://dashscope.aliyuncs.com"
    api_key: str = ""
    llm_model_name: str = "qwen-plus"
    llm_model_name_2: str = ""   # optional second model for round-robin (different rate-limit pool)
    embedding_model: str = "text-embedding-v3"
    rerank_model: str = "gte-rerank"

    # ---- Concurrency ----
    llm_concurrency: int = 8
    llm_rps: float = 4.0
    embed_concurrency: int = 8
    embed_batch_size: int = 10
    rerank_concurrency: int = 4

    # ---- Embedding ----
    embed_dim: int = 1024

    # ---- Storage ----
    postgres_dsn: str = "postgresql://medgraph:medgraph_pwd@127.0.0.1:5432/medgraph"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "medgraph_pwd"
    qdrant_url: str = "http://127.0.0.1:6333"

    qdrant_collection_q: str = "medgraph_q"
    qdrant_collection_a: str = "medgraph_a"
    qdrant_collection_topic: str = "medgraph_topic"

    # ---- Data ----
    data_file: str = "Professional.json"
    cache_dir: str = "data/cache"

    # ---- Retrieval ----
    recall_top_k_vector: int = 30
    recall_top_k_graph: int = 20
    rerank_top_k: int = 8

    @property
    def cache_path(self) -> Path:
        p = PROJECT_ROOT / self.cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
