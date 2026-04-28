"""DashScope embeddings client.

Auto-routes to the correct endpoint based on the configured model:
- OpenAI-compatible path  (`/compatible-mode/v1/embeddings`)
- Native text-embedding   (`/api/v1/services/embeddings/text-embedding/text-embedding`)
- Native multimodal       (`/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`)

Selection rules:
- Model name containing "vl" or "multimodal" → multimodal endpoint
- Otherwise → try OpenAI-compatible first, then native text on 404.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from ..config import get_settings
from ._base import DashScopeBaseClient


def _is_multimodal(model: str) -> bool:
    m = model.lower()
    return "vl" in m or "multimodal" in m


class EmbeddingClient(DashScopeBaseClient):
    log_tag = "model.embedding"

    def __init__(self, model: str | None = None, **kw):
        s = get_settings()
        super().__init__(
            api_key=s.api_key,
            base_url=s.base_url,
            concurrency=kw.pop("concurrency", s.embed_concurrency),
            rps=kw.pop("rps", s.llm_rps * 2),
            timeout=kw.pop("timeout", 60.0),
        )
        self.model = model or s.embedding_model
        self.batch_size = s.embed_batch_size
        self._dim = s.embed_dim

        self._compat_path = "/compatible-mode/v1/embeddings"
        self._native_text_path = "/api/v1/services/embeddings/text-embedding/text-embedding"
        self._native_mm_path = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

        # endpoint mode: 'compat' | 'native_text' | 'native_mm'
        if _is_multimodal(self.model):
            self._mode = "native_mm"
            logger.info("embedding model `{}` looks multimodal -> using multimodal endpoint", self.model)
        else:
            self._mode = "compat"

    # --- payload builders ----------------------------------------------------
    async def _embed_compat(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        data = await self.post_json(self._compat_path, payload)
        return [item["embedding"] for item in data["data"]]

    async def _embed_native_text(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"text_type": "document"},
        }
        data = await self.post_json(self._native_text_path, payload)
        try:
            embeddings = data["output"]["embeddings"]
            embeddings.sort(key=lambda x: x.get("text_index", 0))
            return [e["embedding"] for e in embeddings]
        except Exception:
            logger.error("native text-embedding unexpected shape: {}", data)
            raise

    async def _embed_native_mm_one(self, text: str) -> list[float]:
        """Multimodal endpoint: one call per text item.

        Format: input.contents = [{"text": "..."}], response.output.embeddings[0]
        """
        payload = {
            "model": self.model,
            "input": {"contents": [{"text": text}]},
        }
        data = await self.post_json(self._native_mm_path, payload)
        try:
            embeddings = data["output"]["embeddings"]
            embeddings.sort(key=lambda x: x.get("index", 0))
            return embeddings[0]["embedding"]
        except Exception:
            logger.error("native multimodal-embedding unexpected shape: {}", data)
            raise

    async def _embed_native_mm(self, texts: list[str]) -> list[list[float]]:
        # parallelise inside the client's semaphore (set by base concurrency)
        return await asyncio.gather(*[self._embed_native_mm_one(t) for t in texts])

    # --- public API ----------------------------------------------------------
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._mode == "native_mm":
            return await self._embed_native_mm(texts)
        if self._mode == "native_text":
            return await self._embed_native_text(texts)
        # compat with cascading fallback
        try:
            return await self._embed_compat(texts)
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "not_supported" in msg or "not support" in msg:
                logger.warning("compat embed failed -> trying native text endpoint: {}", e)
                self._mode = "native_text"
                try:
                    return await self._embed_native_text(texts)
                except Exception as e2:
                    msg2 = str(e2).lower()
                    if "url error" in msg2 or "400" in msg2 or "invalid" in msg2:
                        logger.warning("native text endpoint failed -> trying multimodal: {}", e2)
                        self._mode = "native_mm"
                        return await self._embed_native_mm(texts)
                    raise
            raise

    async def embed_batched(
        self,
        texts: list[str],
        *,
        progress: bool = False,
    ) -> list[list[float]]:
        """Embed `texts` honoring `batch_size` per call, in parallel."""
        if not texts:
            return []

        # Multimodal mode is single-text-per-call internally, so we don't need
        # an extra batching layer; just gather concurrently.
        if self._mode == "native_mm":
            return await self._embed_native_mm(texts)

        chunks: list[list[str]] = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]
        results: list[list[list[float]]] = [None] * len(chunks)  # type: ignore

        async def _run(i: int):
            results[i] = await self.embed(chunks[i])

        if progress:
            try:
                from tqdm.asyncio import tqdm_asyncio  # type: ignore

                await tqdm_asyncio.gather(*[_run(i) for i in range(len(chunks))])
            except Exception:
                await asyncio.gather(*[_run(i) for i in range(len(chunks))])
        else:
            await asyncio.gather(*[_run(i) for i in range(len(chunks))])

        flat: list[list[float]] = []
        for batch in results:
            flat.extend(batch)
        return flat

    @property
    def dim(self) -> int:
        return self._dim
