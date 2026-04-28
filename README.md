# 医疗 GraphRAG · 基于 Professional.json + 阿里百炼

[![status](https://img.shields.io/badge/status-生产就绪-success)]() [![data](https://img.shields.io/badge/data-30,031%20QA%20%2F%2024%20科室-blue)]() [![eval](https://img.shields.io/badge/Recall@8-92.1%25-brightgreen)]()

端到端中文医疗 GraphRAG 系统：把 30,031 条临床医学问答（24 科室 / 2,533 子主题）转化为
**Specialty → Topic → Chunk → QAPair** 三层结构图 + **9 类医学实体 / 12 类临床关系** 的知识图谱，
配合 **Q-vec / A-vec / Topic-vec 三视角向量索引** 与 **Q+A+Graph 三路并行召回 + RRF 融合 + Cross-Encoder 重排**，
最终生成带 `[chunkId]` 引用回链的中文医学回答。

所有 LLM / Embedding / Rerank 调用均通过阿里百炼（DashScope）—— 双模型 round-robin、并发限速、重试、
完整调用日志全部内建。

---

## 1. 关键成果

| 维度 | 数值 |
|---|---|
| 输入数据 | 30,031 QA / 24 科室 / 2,533 子主题 / 6,092 chunks |
| 抽取规模 | 39,400 唯一实体（去重后）/ 186,557 mentions / 23,591 类型化关系 |
| 图谱规模 | Neo4j 78,080 节点 / 235,909 边（13 种实体类型 + 12 种关系类型）|
| 向量索引 | Qdrant 三 collection 共 62,595 向量（dim=2560）|
| **检索质量** | **QA Recall@8 = 92.1% / Chunk Recall@8 = 93.1% / Chunk MRR@8 = 0.913**（480 条 hold-out）|
| 端到端延迟 | 单条 retrieve 约 5–8 秒（含 LLM 查询理解 + 三路召回 + 重排）|
| 评估并发 | 32 路并行，480 条 12.5 分钟跑完 |

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          离线索引流水线（一次性）                          │
├──────────────────────────────────────────────────────────────────────────┤
│  Professional.json (30k QA)                                               │
│       │                                                                    │
│       ├──► [01] parse + 三层标签 ──►  Postgres (specialty/topic/chunk/qa) │
│       │                                                                    │
│       ├──► [02] DashScope Embedding ──►  Qdrant: medgraph_q / medgraph_a │
│       │                                                                    │
│       ├──► [03] Neo4j 骨架（Specialty→Topic→Chunk→QAPair, BELONGS_TO）   │
│       │                                                                    │
│       ├──► [04] DashScope LLM 实体抽取（双模型 RR）                        │
│       │       ├─► entities.jsonl（断点续跑）                              │
│       │       ├─► PG.entity / PG.qa_entity_mention                        │
│       │       └─► [09] dedup（aliases + 归一名）                          │
│       │                                                                    │
│       ├──► [05] DashScope LLM 关系抽取（schema-guided + evidence）        │
│       │       ├─► relations.jsonl                                          │
│       │       ├─► PG.relation                                              │
│       │       └─► Neo4j: Entity 节点 + 12 种类型化关系（APOC）            │
│       │                                                                    │
│       ├──► [06] LLM 子主题摘要 + Embedding ──► Qdrant medgraph_topic      │
│       │       └─► [06b] resync / [06c] 补做缺失                            │
│       │                                                                    │
│       ├──► [07] 分层抽样测试集（per-specialty）                            │
│       └──► [08] Recall@k / MRR 评估                                        │
└──────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          在线检索 + 生成                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  用户问题                                                                  │
│      │                                                                     │
│      ├──► [a] LLM 查询理解（改写 + 实体识别 + 科室推断）                   │
│      │                                                                     │
│      ├──► [b] 三路并行召回                                                 │
│      │     ├─► Q-vec    嵌入 → Qdrant medgraph_q   (top-30)               │
│      │     ├─► A-vec    嵌入 → Qdrant medgraph_a   (top-30)               │
│      │     └─► Graph    锚实体 → Neo4j 1-2 跳邻居  (top-20)              │
│      │                                                                     │
│      ├──► [c] RRF 融合（k=60, weights=[1.0, 1.0, 0.8]）                   │
│      │                                                                     │
│      ├──► [d] DashScope Cross-Encoder rerank → top-K（默认 8）            │
│      │                                                                     │
│      ├──► [e] 上下文组装（命中 QA + 兄弟 QA + 子图三元组）                 │
│      │                                                                     │
│      └──► [f] LLM 生成（强制 [chunkId] 引用回链）                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据画像

`Professional.json` 是一个临床医学问答数据集（默克手册风格），**已经是结构化的 Q&A**，每条带 `chunkId` 三层标签 `Specialty - Topic - part-N`。

| 维度 | 值 |
|---|---|
| 总 Q&A 条数 | **30,031** |
| 顶层科室 | **24 个**（儿科 4168 / 传染病 3763 / 妇产 2284 / 心血管 2098 …） |
| 唯一 chunkId | **6,092** |
| 子主题数 | **2,533** |
| 平均每 chunk Q 数 | 4.9（最多 25） |
| 问题平均长度 | 24 字 |
| 答案平均长度 | 267 字（最长 4,157） |

**关键观察 ——这数据自带 3 层人工标签**，不需要重新做 Leiden 社区聚类（标准 GraphRAG 的核心步骤）。
因此本方案保留 chunkId 作为天然社区，把工作重心放在**实体 / 关系图谱**与**多视角向量召回**上。

---

## 4. 技术栈

| 层 | 选型 | 用途 |
|---|---|---|
| **LLM** | DashScope `deepseek-v4-flash` + `qwen3.6-flash` | 双模型 round-robin（独立限流池）|
| **Embedding** | DashScope `qwen3-vl-embedding`（多模态）| 文本向量化，dim=2560 |
| **Rerank** | DashScope `qwen3-vl-rerank`| Cross-encoder 重排 |
| **关系库** | PostgreSQL 16 | QA / 实体 / 关系 / mention，结构数据 |
| **图库** | Neo4j 5.22 (Community + APOC) | 知识图谱、多跳查询 |
| **向量库** | Qdrant 1.11 | 三个 collection（dim=2560）|
| **API** | FastAPI + Uvicorn | REST + 静态前端 |
| **异步** | asyncio + httpx + aiolimiter | 客户端并发限速 |
| **部署** | Docker Compose | Neo4j / Qdrant / Postgres 一键启动 |

---

## 5. 数据 Schema

### Postgres
```sql
specialty         (name PK, n_topics, n_qa)
topic             (specialty FK, name, n_chunks, n_qa, summary)
chunk             (chunk_id PK, specialty, topic, part, n_qa, summary)
qa_pair           (qa_id PK, chunk_id FK, specialty, topic, part, seq, question, answer, qa_text generated)
entity            (entity_id PK, name, type, aliases TEXT[], description)
qa_entity_mention (qa_id, entity_id, surface) PK
relation          (rel_id, head_id, rel_type, tail_id, evidence_qa, evidence_chunk, confidence)
```

### Neo4j（标签 + 关系）
```
节点标签：
  :Specialty (24)            :Disease (11,109)
  :Topic (2,533)             :Drug (3,911)
  :Chunk (6,092)             :Symptom (7,368)
  :QAPair (30,031)           :Procedure (7,151)
  :Entity (39,400)           :Lab (3,203)
                             :Pathogen (1,170)
                             :Anatomy (3,315)
                             :Gene (1,165)
                             :Population (1,008)

关系类型：
  结构性： BELONGS_TO (38,656)、MENTIONS (173,662)
  治疗类： TREATS (4,743)、PREVENTS (1,078)、CONTRAINDICATED_FOR (329)、SIDE_EFFECT_OF (635)
  病理类： CAUSES (4,395)、RISK_FACTOR_OF (1,304)、COMPLICATION_OF (764)
  诊断类： PRESENTS_WITH (4,007)、DIAGNOSED_BY (3,186)
  鉴别 ：DIFFERENTIAL_DX (410)
  药物 ：INTERACTS_WITH (571)
  兜底 ：AFFECTS (2,169)
```

### Qdrant
```
medgraph_q       30,031 点（Question 嵌入向量，dim=2560）
medgraph_a       30,031 点（Q+A 拼接嵌入向量）
medgraph_topic   2,533  点（LLM 生成的子主题摘要嵌入）
```

---

## 6. 项目目录

```
graph_rag/
├── Professional.json              # 输入数据（27 MB）
├── .env                           # DashScope key + 并发参数
├── .env.example                   # 完整可调参数说明
├── docker-compose.yml             # Neo4j + Qdrant + Postgres
├── requirements.txt
├── README.md                      # 本文件
│
├── src/
│   ├── config.py                  # pydantic-settings 集中配置
│   ├── clients/                   # 阿里百炼客户端（async）
│   │   ├── _base.py               # 共用：rate limit / retry / 调用日志
│   │   ├── llm.py                 # OpenAI-compat chat
│   │   ├── embedding.py           # 多模态/纯文本自动路由
│   │   └── rerank.py              # native rerank
│   │
│   ├── ingest/                    # 离线流水线核心
│   │   ├── parse.py               # chunkId 三层解析（带 part 后缀剥离）
│   │   ├── storage.py             # PG schema + 批量入库
│   │   ├── prompts.py             # 实体/关系/摘要 prompt 模板
│   │   ├── entity_extract.py      # P2 实体抽取 + 归一（断点续跑）
│   │   ├── entity_dedup.py        # P2.5 别名归并 + Union-Find 去重
│   │   ├── relation_extract.py    # P3 关系抽取 + 推图（APOC）
│   │   └── summary.py             # P4 子主题摘要
│   │
│   ├── graph/                     # Neo4j 层
│   │   ├── client.py              # driver 单例
│   │   ├── schema.py              # 约束 + 索引
│   │   └── builder.py             # 骨架批量构建
│   │
│   ├── index/
│   │   └── vector_store.py        # Qdrant 抽象（query_points API）
│   │
│   ├── retrieval/                 # 在线检索
│   │   ├── query_understanding.py # LLM 查询改写 + 实体识别
│   │   ├── graph_recall.py        # 锚实体 → 1/2 跳子图召回
│   │   ├── fusion.py              # RRF
│   │   └── pipeline.py            # 三路并行 + RRF + rerank + 上下文组装
│   │
│   ├── generation/
│   │   └── answer.py              # 强制 chunkId 引用回链
│   │
│   ├── eval/
│   │   ├── build_test_set.py      # 分层抽样
│   │   └── evaluate.py            # Recall@k / MRR / 并行评估
│   │
│   └── api/
│       ├── server.py              # FastAPI 应用
│       └── static/
│           └── index.html         # 单页前端调试台
│
├── scripts/                       # 编号化运行脚本
│   ├── 00_smoke_test.py           # 模型 + 服务连通性
│   ├── 01_load_data.py            # JSON → Postgres
│   ├── 02_build_vectors.py        # Q-vec + A-vec → Qdrant（断点续跑）
│   ├── 03_build_graph_skeleton.py # 4 类节点 → Neo4j
│   ├── 04_extract_entities.py     # 实体抽取（断点续跑）
│   ├── 05_extract_relations.py    # 关系抽取 + 推图
│   ├── 06_build_topic_summaries.py
│   ├── 06b_resync_topic_vectors.py # Qdrant 出错时的补救
│   ├── 06c_fill_missing_summaries.py
│   ├── 07_build_testset.py
│   ├── 08_evaluate.py
│   └── 09_dedup_entities.py
│
└── data/
    ├── neo4j/  qdrant/  postgres/ # Docker 数据卷
    └── cache/                     # 抽取 jsonl + 调用日志 + 测试集
        ├── model_calls.log         # 所有模型调用统一日志（自动滚动）
        ├── entities.jsonl          # 实体抽取断点
        ├── relations.jsonl         # 关系抽取断点
        ├── testset.jsonl           # 480 条 hold-out
        └── eval_results.jsonl      # 每条评估明细
```

---

## 7. 流水线脚本（按顺序运行）

| 步骤 | 命令 | 说明 | 实测耗时 |
|---|---|---|---|
| 0 | `python scripts/00_smoke_test.py` | 验证 LLM/Embed/Rerank/PG/Neo4j/Qdrant | 5 s |
| 1 | `python scripts/01_load_data.py` | 解析 chunkId → Postgres | 11 s |
| 2 | `python scripts/02_build_vectors.py --view both` | Q+A 双视角 embedding（断点续跑）| 28 min（multimodal 单条调用，60k 次）|
| 3 | `python scripts/03_build_graph_skeleton.py` | 4 类节点 + BELONGS_TO 边 | 15 s |
| 4 | `python scripts/04_extract_entities.py` | 30,031 LLM 抽实体（双模型）| ~30 min（concurrency=128 双模型）|
| 9 | `python scripts/09_dedup_entities.py [--dry-run]` | 别名归并去重 | 8 s |
| 5 | `python scripts/05_extract_relations.py --push-graph` | 关系抽取 + Neo4j 推图 | ~33 min（25,907 LLM 调用）|
| 6 | `python scripts/06_build_topic_summaries.py --concurrency 8` | 2,533 子主题摘要 | ~10 min |
| 6c| `python scripts/06c_fill_missing_summaries.py` | 补做失败的摘要 | 8 min |
| 7 | `python scripts/07_build_testset.py --per-specialty 20` | 分层抽样 480 条 | <1 s |
| 8 | `python scripts/08_evaluate.py --top-k 8 --parallelism 32` | Recall@k / MRR | 12.5 min |

每个 LLM 步骤都基于 `data/cache/*.jsonl` **断点续跑**——异常中止再次运行会跳过已完成项。

---

## 8. 检索设计要点

### 三路召回（并行）
1. **Q-vec**：query embedding 与 30k 个 Question 向量做 cos 相似度（top-30）
2. **A-vec**：query embedding 与 30k 个 Q+A 拼接向量（top-30）
3. **Graph**：从 query 中识别的医学实体出发，在 Neo4j 中沿 1–2 跳邻居路径返回相关 QA（top-20）
   - 直接命中 (`q-MENTIONS->e`) 权重 1.0
   - 1 跳关系 (`a-rel->e`) 权重 0.7
   - 2 跳路径权重 0.4

### RRF 融合
$$ \text{score}(d) = \sum_{i \in \{q, a, g\}} \frac{w_i}{60 + \text{rank}_i(d)} $$
权重 `[1.0, 1.0, 0.8]`（图路稍降权，因其往往返回较多噪声候选）。

### Cross-Encoder rerank
DashScope `qwen3-vl-rerank` 对 query × top-32 候选做 pair-wise 打分，输出 top-8。

### 上下文组装
- 每条命中：QA + 同 chunk 兄弟 QA（最多 3 条，提供上下文）
- 全局：从锚实体出发的子图三元组（最多 30 条）

### 生成
- LLM 强制要求每段陈述末尾标 `[chunkId]` 引用
- 输出结构：核心答复 → 关键要点 → 来源列表

---

## 9. API 接口

服务地址：**http://127.0.0.1:8000**

| 端点 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 静态前端（单页应用）|
| `/health` | GET | 存活检查 |
| `/stats` | GET | PG / Neo4j / Qdrant 实时计数 |
| `/specialties` | GET | 24 科室 + QA 数 + 摘要覆盖率 |
| `/random_qa?specialty=&n=` | GET | 随机示例 QA（用于灵感）|
| `/entity?name=&limit=` | GET | 实体邻居（aliases / mention QAs / 关系）|
| `/retrieve` | POST | 检索（含三路子分数 + 耗时分解） |
| `/ask` | POST | 检索 + LLM 生成（带 chunkId 引用）|

### `/retrieve` 响应字段
```json
{
  "query": "...",
  "rewritten": "...",
  "entities": [{"name":"妊娠期糖尿病","type":"Disease"}, ...],
  "specialty_hint": "妇产科学",
  "items": [
    {
      "qa_id": "...",
      "score": 0.894,            // 重排得分
      "rrf_score": 0.040,
      "chunk_id": "...",
      "specialty": "...", "topic": "...",
      "sources": ["q","a","graph"],          // 哪些路命中
      "channel_scores": {"q":0.79,"a":0.73,"graph":1.0},
      "channel_ranks":  {"q":3,"a":5,"graph":1},
      "question": "...", "answer": "...",
      "siblings": [...]
    }
  ],
  "triples": [{"head":"...","rel":"TREATS","tail":"...", "confidence":0.9, "chunk_id":"..."}, ...],
  "timings_ms": {"query_understanding":5110, "query_embed":452, "recall":1121, "rerank":455, "triples":104, "total":7326},
  "counts":     {"vec_q":30, "vec_a":30, "graph":20, "fused":32, "rerank_kept":8, "triples":30}
}
```

---

## 10. 前端调试台

打开 http://127.0.0.1:8000/ 看到的内容：

| 区域 | 功能 |
|---|---|
| **顶部 stats 栏** | 实时显示 PG/Neo4j/Qdrant 各表/集合的计数；`↻ 刷新` 按钮 |
| **搜索栏** | 文本框 + 模式切换（Ask/Retrieve）+ 6 个示例问题 + 🎲 随机 + 📚 按科室浏览 |
| **查询理解面板** | 原问题 / LLM 改写后 / 识别到的实体（**按 9 类配色，可点击** 看邻居）/ 推断科室 |
| **耗时与计数** | 5 个流水线阶段 ms 耗时 + 6 个召回计数 |
| **核心回答** | （Ask 模式）LLM 生成回答；`[chunkId]` 引用是**可点击徽章**，点了滚动并高亮对应证据卡片 |
| **召回列表** | 每条卡片：排名、重排分、Q/A/Graph 来源徽章（hover 看子分数）、科室/子主题、原 Q&A、同 chunk 兄弟问、各路得分明细行 |
| **过滤** | 双重过滤：`来源`（仅 Q-vec / 仅 A-vec / 仅 Graph / 多路命中）+ `按科室` |
| **三元组面板** | 右栏：圆形 SVG 图（实体节点 + 关系箭头 + 标签）+ 三元组列表 |
| **JSON 调试** | 可折叠抽屉，展示原始 API 响应 |
| **实体弹窗** | 点击任意实体 chip 弹出：匹配实体（带 type / 描述 / aliases）+ 关联关系 + 提及该实体的 QA 列表 |

---

## 11. 模型调用日志

所有 LLM / Embedding / Rerank 调用统一写入 **`data/cache/model_calls.log`**（50 MB 滚动 + 5 份保留）。

格式：
```
2026-04-27 16:01:23.812 | INFO  | model.llm        | call_begin id=8a3f2b1c path=/compatible-mode/v1/chat/completions model=deepseek-v4-flash msgs=2 chars=312
2026-04-27 16:01:24.945 | INFO  | model.llm        | call_ok    id=8a3f2b1c attempts=1 attempt_ms=1130.4 total_ms=1132.1 usage={'prompt_tokens':156,'completion_tokens':87,'total_tokens':243}
2026-04-27 16:01:25.102 | WARN  | model.embedding  | call_retry id=2f9c... attempt=1 status=429 412.5ms body=...
```

实时跟踪：
```powershell
# Windows PowerShell
Get-Content data\cache\model_calls.log -Wait -Tail 100
Get-Content data\cache\model_calls.log -Wait | Select-String "model.llm"
Get-Content data\cache\model_calls.log -Wait | Select-String "call_fail|call_retry"

# Git Bash / WSL
tail -f data/cache/model_calls.log
tail -f data/cache/model_calls.log | grep model.embedding
tail -f data/cache/model_calls.log | grep -E "call_fail|call_retry"
```

字段含义：
- `id` — 8 位调用 ID（重试期间同一 id 多次出现）
- `attempt_ms` — 当次 HTTP 调用耗时
- `total_ms` — 含全部重试与队列等待
- `usage` — token 使用量（仅 LLM 才有）
- tag：`model.llm` / `model.embedding` / `model.rerank`

---

## 12. 关键参数（.env）

```ini
# === 阿里百炼 ===
BASE_URL=https://dashscope.aliyuncs.com
API_KEY=sk-...
LLM_MODEL_NAME=deepseek-v4-flash
LLM_MODEL_NAME_2=qwen3.6-flash       # 双模型 round-robin（独立限流池）
EMBEDDING_MODEL=qwen3-vl-embedding   # 多模态，会自动路由到 multimodal-embedding 端点
RERANK_MODEL=qwen3-vl-rerank

# === 并发与限速 ===
LLM_CONCURRENCY=128                  # 经验：64 安全 / 128 甜点 / 256 收益递减
LLM_RPS=100                          # aiolimiter 软速率上限
EMBED_CONCURRENCY=64
EMBED_BATCH_SIZE=10                  # 文本模型批大小（多模态此值无效）
RERANK_CONCURRENCY=8

# === 数据库 ===
POSTGRES_DSN=postgresql://medgraph:medgraph_pwd@127.0.0.1:5432/medgraph
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=medgraph_pwd
QDRANT_URL=http://127.0.0.1:6333

# === 检索参数 ===
RECALL_TOP_K_VECTOR=30
RECALL_TOP_K_GRAPH=20
RERANK_TOP_K=8
```

---

## 13. 实测性能 & 调优结论

### 实体抽取吞吐演进
| 配置 | 实测吞吐 | retry/min | 备注 |
|---|---|---|---|
| 单模型 conc=16 | ~2 ops/s | ~0 | 起点 |
| 单模型 conc=64 | ~4 ops/s | ~0 | |
| **双模型 conc=64** | **~8 ops/s** | ~0 | 加第二模型独立限流池 |
| **双模型 conc=128** | **~16 ops/s** | ~30 | **甜点** |
| 双模型 conc=256 | ~19 ops/s | ~580 | 收益递减，retry 暴涨 |

### Embedding 限制
- 多模态端点 (`qwen3-vl-embedding`) **不支持 batch**，每次只接受一个 input
- 实测稳态 ~36 ops/s（被服务端 throttle）
- 60,000 次单调用约 28 分钟

### 评估 480 条
- 串行：每条 ~7s × 480 = 56 min
- **并行 32**：1.07 s/it 稳态 = **12.5 min**
- 瓶颈：rerank concurrency=8

### Recall 实测
| 指标 | 值 |
|---|---|
| QA Recall@8 | **0.9208** |
| Chunk Recall@8 | **0.9313** |
| Chunk MRR@8 | **0.9130** |

---

## 14. 完整运行（首次部署）

```bash
# 1) 启动基础设施（Docker Desktop 需运行）
docker compose up -d

# 2) Python 环境
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt

# 3) 配置 .env，写入你的 DashScope API_KEY

# 4) 验证连通
python scripts/00_smoke_test.py

# 5) 离线流水线（按顺序，每步 ~分钟级）
python scripts/01_load_data.py
python scripts/02_build_vectors.py --view both        # 28 min
python scripts/03_build_graph_skeleton.py
python scripts/04_extract_entities.py                  # 30 min（重活，断点续跑）
python scripts/09_dedup_entities.py                    # 可选：实体去重
python scripts/05_extract_relations.py --push-graph    # 33 min
python scripts/06_build_topic_summaries.py --concurrency 8  # 10 min
python scripts/06c_fill_missing_summaries.py            # 补做缺失
python scripts/07_build_testset.py --per-specialty 20

# 6) 启动服务
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000/

# 7) 评估
python scripts/08_evaluate.py --top-k 8 --parallelism 32
```

---

## 15. 异常处理 & 已修复 Bug 一览

| 现象 | 根因 | 修复 |
|---|---|---|
| Embedding 调用 404 | `qwen3-vl-embedding` 是多模态模型，要走 `multimodal-embedding` 端点 + `input.contents=[{text}]` 格式 | `clients/embedding.py` 按模型名自动路由，单文本一次调用 |
| Qdrant upsert 偶发 WinError 10054 | 高并发 + 单批次过大 + 客户端默认 timeout | `vector_store.py` 改 batch=128 + 内置 3 次重试 + `timeout=120` |
| 02_vectors 断点重跑会重新 embed 全部 | 旧脚本无 resume | 加 `existing_point_ids()` 跳过已存在 |
| 06 摘要 940 条丢失 | 早期 quota 暴击导致 5 次重试也失败，第一个失败把 gather 拉崩 | 单任务 try/except 兜空字符串；新增 `06c_fill_missing_summaries.py` 单独补 |
| `query_understanding.py` `.format()` KeyError | prompt 里 JSON 示例的 `{}` 没转义 | 改 `{{}}` 双花括号 |
| Qdrant `client.search` 缺失 | qdrant-client v1.17 移除了 search() | 改用 `query_points().points` |
| LLM `chat_json` 偶发 JSON 解析失败 | LLM 输出含 markdown 代码围栏或裸文本 | `_safe_json` 优先剥围栏 + 找最外层 `{}/[]`，仍失败再 `json.loads` |
| Postgres exec 中含 `&&` 在 cmd 上失败 | Windows shell 不识别 | 全部用 sh / Git Bash |

---

## 16. 安全 / 合规注意

- 生成模板已声明"参考信息而非个体化医疗建议"，**不要去掉**
- 每个回答段落必须挂 `chunkId`，否则前端不显示——便于审计
- DashScope API key 仅写在 `.env`，**勿提交版本库**

---

## 17. 后续可扩展方向

- [ ] **冷启动加速**：把 query understanding 的 prompt 蒸馏成更短的格式（5s → <1s）
- [ ] **多阶段 retrieval**：第一阶段先用 topic-vec 定位 1-2 个相关子主题，再做 Q/A 召回
- [ ] **实体清洗 v2**：在 dedup 基础上再用 LLM 做语义合并（如"2 型糖尿病"vs"非胰岛素依赖型糖尿病"）
- [ ] **流式生成**：`/ask` 端点加 SSE，前端边收边渲染
- [ ] **跨科室桥接评估**：人工标注 200 条复合问题（如"妊娠合并糖尿病视网膜病变"），单独验证图召回的增益
- [ ] **答案溯源置信度**：基于 channel_scores 加权给每段答案打置信度
- [ ] **知识图谱可视化**：把右栏 SVG 升级成可拖拽的 force-layout（d3 / cytoscape）

---

## 18. 致谢

- **数据**：默克手册（Merck Manuals）风格的临床医学问答
- **模型**：阿里云百炼 DashScope（DeepSeek / Qwen 系列）
- **图存储**：Neo4j Community + APOC 插件
- **向量存储**：Qdrant
- **方案灵感**：Microsoft GraphRAG（社区聚类 + 摘要）；本项目采用人工标签替代 Leiden，更适合本数据集
