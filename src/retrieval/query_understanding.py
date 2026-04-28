"""LLM-based query understanding: rewrite + entity / specialty hint extraction."""
from __future__ import annotations

import json

from loguru import logger

from ..clients.llm import LLMClient
from ..ingest.prompts import ENTITY_TYPES


_SYSTEM = """你是一位中文医疗检索助手。给定用户问题，请输出 JSON：
{{
  "rewritten": "更适合检索的中文问题（保留关键医学术语，去除冗余）",
  "entities": [{{"name": "<术语>", "type": "<受控类型>"}}],
  "specialty_hint": "<可能涉及的科室名（如 内分泌及代谢紊乱 / 心血管疾病 / 儿科学），不确定写空字符串>"
}}
要求：
- 受控类型仅取自：{types}
- entities 只列医学概念实体，3-6 个为宜，少于 3 个也可
- specialty_hint 必须是真实的科室分类或留空
仅输出 JSON。"""


async def understand_query(llm: LLMClient, question: str) -> dict:
    sys_msg = _SYSTEM.format(types=", ".join(ENTITY_TYPES))
    try:
        data = await llm.chat_json(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": question.strip()},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        if not isinstance(data, dict):
            return {"rewritten": question, "entities": [], "specialty_hint": ""}
        data.setdefault("rewritten", question)
        data.setdefault("entities", [])
        data.setdefault("specialty_hint", "")
        # Filter entity types
        clean_ents = []
        for e in data.get("entities") or []:
            if isinstance(e, dict) and e.get("name") and e.get("type") in ENTITY_TYPES:
                clean_ents.append({"name": e["name"].strip(), "type": e["type"]})
        data["entities"] = clean_ents
        return data
    except Exception as exc:
        logger.warning("query_understanding fallback: {}", exc)
        return {"rewritten": question, "entities": [], "specialty_hint": ""}
