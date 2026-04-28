"""Answer generation with mandatory chunkId citation."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from ..clients.llm import LLMClient
from ..retrieval.pipeline import RetrievalBundle, RetrievedItem


_SYSTEM = """你是一位严谨的中文医学助手。基于给定的检索证据回答用户问题。

绝对要求：
1. 仅使用所提供证据中的内容；如证据不足，请直说"未在资料中找到充分证据"。
2. 在每段陈述末尾以 [chunkId] 形式给出引用，可以一段引用多个 chunkId。
3. 不要给出诊断结论或剂量建议；说明这是参考信息而非个体化医疗建议。
4. 输出结构：
   - 「核心答复」用 1-2 段话直接回答问题。
   - 「关键要点」3-6 条要点，每条末尾标 [chunkId]。
   - 「来源」列出本回答引用过的所有 chunkId（去重）。
"""


def _format_evidence(items: list[RetrievedItem], triples: list[dict]) -> str:
    blocks: list[str] = []
    for i, it in enumerate(items, 1):
        head = f"[E{i} | chunkId={it.chunk_id} | 科室={it.specialty} | 子主题={it.topic}]"
        blocks.append(f"{head}\nQ: {it.question}\nA: {it.answer}")
        for sib in it.siblings:
            blocks.append(
                f"   ↳ 同 chunk 兄弟问 [chunkId={sib['chunk_id']}]: {sib['question']}\n      {sib['answer']}"
            )
    if triples:
        tri_lines = [
            f"  - {t['head']} -[{t['rel']}]-> {t['tail']}  (来源 chunkId={t.get('chunk_id')})"
            for t in triples[:20]
        ]
        blocks.append("【知识图谱三元组】\n" + "\n".join(tri_lines))
    return "\n\n".join(blocks)


@dataclass
class AnsweredBundle:
    answer: str
    bundle: RetrievalBundle


async def answer_with_citations(
    llm: LLMClient,
    bundle: RetrievalBundle,
) -> AnsweredBundle:
    if not bundle.items:
        return AnsweredBundle(answer="未在资料中找到与该问题相关的证据。", bundle=bundle)

    evidence = _format_evidence(bundle.items, bundle.triples)
    user = (
        f"用户问题：{bundle.query}\n\n"
        f"（系统改写：{bundle.rewritten}）\n\n"
        f"【检索证据】\n{evidence}"
    )
    text = await llm.chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return AnsweredBundle(answer=text.strip(), bundle=bundle)
