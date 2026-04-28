"""All LLM prompts used during ingestion (entity / relation / summary)."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "Disease",      # 疾病/综合征/障碍
    "Drug",         # 药物/活性成分/制剂
    "Symptom",      # 症状/体征
    "Procedure",    # 手术/操作/检查方式
    "Lab",          # 实验室检查/影像/指标
    "Pathogen",     # 病原体（细菌、病毒、寄生虫、真菌）
    "Anatomy",      # 解剖部位/器官/组织
    "Gene",         # 基因/突变位点/染色体
    "Population",   # 特定人群（孕妇、新生儿、老年人等）
]

ENTITY_TYPE_DESC = {
    "Disease": "疾病、综合征、障碍。例：糖尿病、慢性阻塞性肺病、T315I 突变白血病",
    "Drug": "药物的活性成分或药品名称（不含剂量与制剂形式）。例：伊马替尼、阿莫西林",
    "Symptom": "症状或体征。例：咳嗽、发热、心悸",
    "Procedure": "诊疗操作、手术或检查方式。例：心电图、剖宫产、内镜下黏膜切除",
    "Lab": "实验室或影像学指标。例：肌酐、HbA1c、胸片",
    "Pathogen": "致病微生物或寄生虫。例：结核分枝杆菌、HIV、疟原虫",
    "Anatomy": "解剖结构、器官、组织。例：肝脏、视网膜、冠状动脉",
    "Gene": "基因、染色体或突变位点。例：BRCA1、T315I 突变",
    "Population": "特定人群定语，仅当作为独立医学概念时抽取。例：孕妇、早产儿、老年人",
}


ENTITY_SYSTEM_PROMPT = """你是一位中文医学命名实体抽取专家。请严格遵守要求：
1. 仅抽取下列受控类型的实体：{types}
2. 同一实体只输出一次。同义词/缩写/全称合并为同一条，name 用最规范的中文术语，aliases 列出文本中出现的所有写法。
3. 不要抽取剂量、频次、症状强度等修饰语；不要抽取"病人"、"医生"、"治疗"这类泛化概念。
4. 输出严格的 JSON 对象，结构为 {{"entities": [{{"name": str, "type": str, "aliases": [str], "description": str}}]}}。description 用 ≤30 字解释。
5. 如无可抽实体，输出 {{"entities": []}}。"""


def build_entity_prompt() -> str:
    type_lines = "\n".join(f"  - {t}: {ENTITY_TYPE_DESC[t]}" for t in ENTITY_TYPES)
    return ENTITY_SYSTEM_PROMPT.format(types="\n" + type_lines)


def build_entity_user(question: str, answer: str) -> str:
    return (
        "请从下面的中文医学问答中抽取实体，仅输出 JSON。\n\n"
        f"【问题】{question}\n\n【答案】{answer}\n\n"
        "JSON："
    )


# ---------------------------------------------------------------------------
# Relation extraction (run after entities of a chunk are known)
# ---------------------------------------------------------------------------

RELATION_TYPES = [
    "TREATS",            # Drug/Procedure -> Disease/Symptom
    "PREVENTS",          # Drug/Procedure -> Disease
    "CAUSES",            # Pathogen/Drug/Disease -> Disease/Symptom
    "PRESENTS_WITH",     # Disease -> Symptom
    "DIAGNOSED_BY",      # Disease -> Lab/Procedure
    "RISK_FACTOR_OF",    # any -> Disease
    "COMPLICATION_OF",   # Disease -> Disease
    "CONTRAINDICATED_FOR",  # Drug -> Disease/Population
    "SIDE_EFFECT_OF",    # Symptom/Disease -> Drug
    "DIFFERENTIAL_DX",   # Disease <-> Disease
    "INTERACTS_WITH",    # Drug -> Drug
    "AFFECTS",           # Drug/Disease -> Anatomy/Lab (兜底)
]

RELATION_SYSTEM_PROMPT = """你是一位中文医学关系抽取专家。给定一段问答以及其中已经抽出的实体列表，请抽取实体之间的医学关系。

允许的关系类型（严格使用英文常量）：
{types}

要求：
1. head 与 tail 必须严格使用提供的实体列表中的 name（区分大小写）。不要新建实体。
2. 一对实体如有多种关系，分别输出多条。
3. 关系应有文本依据；evidence 字段填该问答中支持该三元组的原句（≤80 字，不必完整）。
4. confidence ∈ [0,1]：明确陈述给 0.9，间接推断给 0.6。
5. 输出严格 JSON：{{"relations": [{{"head": str, "rel": str, "tail": str, "evidence": str, "confidence": float}}]}}
6. 若无可抽，输出 {{"relations": []}}。"""


def build_relation_prompt() -> str:
    type_lines = "\n".join(f"  - {t}" for t in RELATION_TYPES)
    return RELATION_SYSTEM_PROMPT.format(types=type_lines)


def build_relation_user(question: str, answer: str, entities: list[dict]) -> str:
    ent_lines = "\n".join(
        f"- name: {e['name']} | type: {e['type']}" for e in entities
    )
    return (
        f"【问题】{question}\n【答案】{answer}\n\n【已知实体】\n{ent_lines}\n\nJSON："
    )


# ---------------------------------------------------------------------------
# Topic / chunk summaries
# ---------------------------------------------------------------------------

TOPIC_SUMMARY_SYSTEM = """你是一位中文医学编辑。请基于下列同一子主题下的若干问答，写一段 100-180 字的子主题导读：
- 概括该子主题的核心内容（疾病、机制、诊疗要点）。
- 不要罗列问题，不要使用列表。
- 写成连续段落，让读者一眼掌握该子主题。
- 仅输出导读文本，不要前缀。"""


def build_topic_summary_user(specialty: str, topic: str, qa_pairs: list[dict]) -> str:
    body = "\n\n".join(
        f"Q: {p['question']}\nA: {p['answer']}" for p in qa_pairs[:8]
    )
    return f"科室：{specialty}\n子主题：{topic}\n\n{body}"
