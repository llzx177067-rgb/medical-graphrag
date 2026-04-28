"""DashScope text rerank client."""
from __future__ import annotations

from typing import Any

from loguru import logger

from ..config import get_settings
from ._base import DashScopeBaseClient


class RerankClient(DashScopeBaseClient):
    """Cross-encoder rerank for (query, document) pairs.

    Uses the DashScope native rerank endpoint:
        POST /api/v1/services/rerank/text-rerank/text-rerank
    """

    log_tag = "model.rerank"

    def __init__(self, model: str | None = None, **kw):
        s = get_settings()
        super().__init__(
            api_key=s.api_key,
            base_url=s.base_url,
            concurrency=kw.pop("concurrency", s.rerank_concurrency),
            rps=kw.pop("rps", max(2.0, s.llm_rps)),
            timeout=kw.pop("timeout", 60.0),
        )
        self.model = model or s.rerank_model
        self._path = "/api/v1/services/rerank/text-rerank/text-rerank"

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {
                "top_n": top_n if top_n is not None else len(documents),
                "return_documents": return_documents,
            },
        }
        data = await self.post_json(self._path, payload)
        try:
            results = data["output"]["results"]
        except Exception:
            logger.error("rerank unexpected shape: {}", data)
            raise
        return [(r["index"], float(r["relevance_score"])) for r in results]
