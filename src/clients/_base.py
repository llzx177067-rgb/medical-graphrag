"""Shared async http base for DashScope (Aliyun BaiLian) endpoints.

The DashScope native endpoints follow the pattern:
    POST https://dashscope.aliyuncs.com/api/v1/services/<group>/<task>/<task>

We use httpx + asyncio.Semaphore for concurrency, aiolimiter for RPS,
and tenacity for retries with exponential back-off.

Every model call is logged with latency (ms), input size, http status and
(when available) usage tokens. Logs go through loguru — by default to stderr
plus a rotating file at ``data/cache/model_calls.log``.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging setup — runs once on first import.
# ---------------------------------------------------------------------------
_LOG_INITIALISED = False


def _init_logger() -> None:
    global _LOG_INITIALISED
    if _LOG_INITIALISED:
        return
    log_dir = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "model_calls.log"
    # Add a dedicated sink for model-call lines (filter by tag).
    logger.add(
        str(log_path),
        rotation="50 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <5} | {extra[tag]} | {message}",
        filter=lambda rec: rec["extra"].get("tag", "").startswith("model."),
    )
    _LOG_INITIALISED = True


_init_logger()


def _model_log(tag: str):
    """Returns a logger bound to a specific tag (e.g. 'model.llm')."""
    return logger.bind(tag=tag)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DashScopeError(RuntimeError):
    """Raised on non-recoverable DashScope errors."""


class _RateLimitError(RuntimeError):
    """Internal marker so tenacity retries on 429/5xx."""


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------


class DashScopeBaseClient:
    """Async http wrapper around DashScope endpoints."""

    log_tag: str = "model.base"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        concurrency: int = 8,
        rps: float = 4.0,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("DashScope api_key is empty; check .env")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(concurrency)
        self._limiter = AsyncLimiter(max_rate=max(1, int(rps)), time_period=1.0)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=concurrency * 2,
                max_keepalive_connections=concurrency,
            ),
            http2=False,
        )
        self._log = _model_log(self.log_tag)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    # ------------------------------------------------------------------
    # Hooks for subclasses to enrich logs
    # ------------------------------------------------------------------
    def _summarise_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a small log-friendly summary of the request."""
        out: dict[str, Any] = {}
        if "model" in payload:
            out["model"] = payload["model"]
        if "messages" in payload:
            out["msgs"] = len(payload["messages"])
            out["chars"] = sum(len(m.get("content", "") or "") for m in payload["messages"])
        if "input" in payload:
            inp = payload["input"]
            if isinstance(inp, list):
                out["n_inputs"] = len(inp)
                out["chars"] = sum(len(t) for t in inp if isinstance(t, str))
            elif isinstance(inp, dict):
                texts = inp.get("texts") or inp.get("documents") or []
                out["n_inputs"] = len(texts)
                out["chars"] = sum(len(t) for t in texts if isinstance(t, str))
        if "query" in payload:
            out["query_len"] = len(payload.get("query") or "")
        if "documents" in payload:
            out["n_docs"] = len(payload.get("documents") or [])
        return out

    def _summarise_response(self, data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        usage = data.get("usage") or {}
        if usage:
            out["usage"] = {
                k: usage.get(k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
                if k in usage
            }
        # native dashscope shape
        if "output" in data and isinstance(data["output"], dict):
            usg = data.get("usage") or data["output"].get("usage")
            if usg:
                out.setdefault("usage", usg)
        return out

    # ------------------------------------------------------------------
    async def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        url = self.base_url + path
        headers = self._headers(extra_headers)
        body = json.dumps(payload, ensure_ascii=False)

        req_id = uuid.uuid4().hex[:8]
        req_summary = self._summarise_payload(payload)
        self._log.info(
            "call_begin id={} path={} {}",
            req_id,
            path,
            " ".join(f"{k}={v}" for k, v in req_summary.items()),
        )

        attempt_no = 0
        t0 = time.perf_counter()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=20.0),
            retry=retry_if_exception_type(
                (_RateLimitError, httpx.TransportError, httpx.ReadTimeout)
            ),
            reraise=True,
        ):
            with attempt:
                attempt_no += 1
                attempt_t0 = time.perf_counter()
                async with self._sem:
                    async with self._limiter:
                        try:
                            resp = await self._client.post(url, headers=headers, content=body)
                        except (httpx.TransportError, httpx.ReadTimeout) as net_err:
                            dt = (time.perf_counter() - attempt_t0) * 1000
                            self._log.warning(
                                "call_net_err id={} attempt={} {:.1f}ms err={}",
                                req_id,
                                attempt_no,
                                dt,
                                type(net_err).__name__,
                            )
                            raise

                dt = (time.perf_counter() - attempt_t0) * 1000

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception as e:
                        self._log.error("call_bad_json id={} err={}", req_id, e)
                        raise DashScopeError(f"bad json: {e}: {resp.text[:200]}")
                    total = (time.perf_counter() - t0) * 1000
                    summary = self._summarise_response(data)
                    self._log.info(
                        "call_ok id={} attempts={} attempt_ms={:.1f} total_ms={:.1f} {}",
                        req_id,
                        attempt_no,
                        dt,
                        total,
                        " ".join(f"{k}={v}" for k, v in summary.items()) if summary else "",
                    )
                    return data

                if resp.status_code in (429, 500, 502, 503, 504):
                    self._log.warning(
                        "call_retry id={} attempt={} status={} {:.1f}ms body={}",
                        req_id,
                        attempt_no,
                        resp.status_code,
                        dt,
                        resp.text[:200],
                    )
                    await asyncio.sleep(0.2 + random.random() * 0.4)
                    raise _RateLimitError(f"http {resp.status_code}: {resp.text[:200]}")

                self._log.error(
                    "call_fail id={} attempt={} status={} {:.1f}ms body={}",
                    req_id,
                    attempt_no,
                    resp.status_code,
                    dt,
                    resp.text[:300],
                )
                raise DashScopeError(
                    f"DashScope {url} returned {resp.status_code}: {resp.text[:500]}"
                )

        raise DashScopeError("retry loop exhausted (unreachable)")
