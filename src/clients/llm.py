"""DashScope chat / generation client (OpenAI-compatible)."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger

from ..config import get_settings
from ._base import DashScopeBaseClient


class LLMClient(DashScopeBaseClient):
    """Wraps DashScope OpenAI-compatible /compatible-mode/v1/chat/completions.

    DashScope exposes both a *native* HTTP API and an OpenAI-compatible one.
    We use the OpenAI-compatible path because it covers Qwen, DeepSeek, GLM
    and other co-hosted models on BaiLian with a single payload shape.
    """

    log_tag = "model.llm"

    def __init__(self, model: str | None = None, **kw):
        s = get_settings()
        super().__init__(
            api_key=s.api_key,
            base_url=s.base_url,
            concurrency=kw.pop("concurrency", s.llm_concurrency),
            rps=kw.pop("rps", s.llm_rps),
            timeout=kw.pop("timeout", 120.0),
        )
        self.model = model or s.llm_model_name
        # OpenAI-compatible path
        self._chat_path = "/compatible-mode/v1/chat/completions"

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        response_format_json: bool = False,
        model: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        data = await self.post_json(self._chat_path, payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            logger.error("Unexpected LLM response shape: {}", data)
            raise

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> Any:
        """Force JSON output and parse. Falls back to substring extract if needed."""
        # First try with response_format_json=True
        try:
            content = await self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format_json=True,
                model=model,
            )
        except Exception:
            content = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, model=model)
        return _safe_json(content)

    async def chat_many(
        self,
        prompts: list[list[dict[str, str]]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> list[str]:
        tasks = [
            self.chat(p, temperature=temperature, max_tokens=max_tokens)
            for p in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)


def _safe_json(text: str) -> Any:
    """Best-effort JSON parsing for LLM output."""
    if not text:
        raise ValueError("empty LLM output")
    text = text.strip()

    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()

    # Find the outermost JSON braces / brackets
    starts = [i for i, c in enumerate(text) if c in "[{"]
    if starts:
        candidate = text[starts[0]:]
        for end in range(len(candidate), 0, -1):
            chunk = candidate[:end]
            if chunk[-1] in "]}":
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
    return json.loads(text)  # last resort, raises
