# 医疗 GraphRAG · 项目深度剖析文档

> 一份面向工程师的技术剖析文档。目标：让没读过代码的人也能理解这个系统怎么工作、每个文件干什么、有哪些设计取舍、以及哪些地方有坑。
>
> 项目根：`D:\sd\rag\graph_rag`，共 53 个受版本控制的文件（含 26MB 输入数据 `Professional.json`）。

---

## 一、这个项目是什么

一句话：**基于医疗 QA 语料（默克手册风格）的向量 + 知识图谱混合检索系统**，产出带 `[chunkId]` 引用回链的可溯源回答，附带一个前端"检索调试台"。

- **领域**：中文临床医学，30,031 条 QA / 24 科室 / 2,533 子主题。
- **技术底座**：阿里百炼（DashScope）的 LLM / Embedding / Rerank 三类模型 + 本地三套存储（Postgres + Neo4j + **Qdrant**）。
- **参考**：方案思路借鉴 Microsoft GraphRAG，但用**人工标签**（数据自带的 `chunkId` 三级分类）替代了 GraphRAG 的 Leiden 社区发现。
- **交付形态**：离线索引流水线（00~09 脚本）+ 在线 FastAPI 服务 + 原生 JS 单页调试台。

> ⚠️ 一个命名澄清：向量库是 **Qdrant**（`qdrant-client`），不是 Milvus。有些内部描述里误写成 Milvus，以代码为准。

---

## 二、整体架构

```
                离线索引流水线                         在线检索
┌───────────────────────────────────┐   ┌──────────────────────────────────────────┐
│ 00 探活(模型+三后端)               │   │  用户问题                                  │
│ 01 解析 Professional.json → PG     │   │    │                                      │
│ 02 建 Q/A 向量 → Qdrant            │   │    ▼                                      │
│ 03 建 Specialty/Topic/Chunk 骨架→Neo4j│  │  [1] 查询理解(LLM: 改写+实体+科室hint)    │
│ 04 LLM 抽实体 → PG                 │   │    │                                      │
│ 05 LLM 抽关系 → PG → Neo4j(+APOC)  │   │    ▼                                      │
│ 06 LLM 生成子主题摘要 → PG+Qdrant   │   │  [2] 三路并发召回                          │
│ 06b/06c 补丁(补向量/补摘要)         │   │      · Q-vec 向量检索 (specialty 过滤)     │
│ 07 分层抽样建测试集                 │   │      · A-vec 向量检索 (specialty 过滤)     │
│ 08 评测 Recall@k / MRR             │   │      · Neo4j 图召回 (锚点实体 0/1/2-hop)   │
│ 09 实体去重 (Union-Find, 幂等)      │   │    │                                      │
└───────────────────────────────────┘   │    ▼                                      │
                                        │  [3] RRF 融合 (k=60, 权重 [1.0,1.0,0.8])    │
                                        │    │                                      │
                                        │    ▼                                      │
                                        │  [4] Cross-Encoder 重排 (保留 top_k=8)      │
                                        │    │                                      │
                                        │    ▼                                      │
                                        │  [5] 取子图三元组 + 组装上下文               │
                                        │    │                                      │
                                        │    ▼                                      │
                                        │  [6] (Ask 模式) LLM 生成带 [chunkId] 引用的答案│
                                        └──────────────────────────────────────────┘
```

**三套存储各司其职：**

| 后端 | 角色 | 关键内容 |
|---|---|---|
| **Postgres 16** | 关系型真源 | QA 全文、分类层级、实体/mention/关系表；8 张表 |
| **Neo4j 5.22 (+APOC)** | 知识图谱 | 骨架层 Specialty/Topic/Chunk/QAPair + Entity 节点 + MENTIONS/类型化关系边 |
| **Qdrant 1.11** | 向量索引 | 三个 collection：`medgraph_q`（问题向量）、`medgraph_a`（问答合并向量）、`medgraph_topic`（子主题摘要向量） |

**关键数值（README 报告）：** Neo4j 78,080 节点 / 235,909 边；Qdrant 62,595 向量；QA Recall@8=92.1%、Chunk Recall@8=93.1%、MRR@8=0.913（480 条测试集）。

---

## 三、数据格式：Professional.json

唯一输入数据，26MB JSON 数组，**30,031 个对象，每个恰好 3 个字段**（全量校验，无缺失/多余/类型漂移）：

```json
{ "question": "str", "answer": "str", "chunkId": "str" }
```

### `chunkId` 的三层人工标签结构

格式：`<科室> - <子主题>[-<子主题细分>] - part-<N>`

- **科室**（24 个）：儿科学 4168、传染病 3763、妇产科学 2284、心血管疾病 2098……临床药理学仅 155。
- **子主题**（2,533 个 (科室,主题) 组合）。
- **part-N**：同一 (科室,主题) 下的分片序号，最大 part=32。

切分锚点在 `src/ingest/parse.py`：`" - "`（带空格）split 一次分科室/主题，再用正则 `-part-(\d+)\s*$`（**只在结尾匹配**）切 part——因为主题名本身可能含连字符（如 `临床药理学 - 药物-受体相互作用-part-1`，主题是"药物-受体相互作用"）。

### 值得注意的数据怪癖

1. **part 编号跳号**：611 个主题的 part 序号不连续（如 `传染病 - COVID-19` 的 part 是 `[1,2,3,4,6,7,11,14,17,18,19]`，缺 5/8/9/10/12/13/15/16）。解析**不能假设 1..N 连续**。
2. **主题名自带连字符/括号**：`药物-受体相互作用`、`呼吸道合胞病毒（RSV）和人偏肺病毒`。
3. **重复问题**：825 组相同 question 文本跨不同科室出现（distinct question=29,206）。
4. **科室名含分号**：`损伤; 中毒`、`免疫学; 过敏性疾病`。
5. **QA/chunkId**：均值 4.93，max 25（与 README "最多 25" 吻合）；问题均值 24 字、答案均值 267 字（最长 4,157），55.6% 答案含 Markdown。

---

## 四、代码结构总览（53 个文件）

```
graph_rag/
├── Professional.json          # 26MB 输入数据 (30,031 条 QA)
├── README.md                  # 530 行，唯一对外文档
├── .env.example               # 38 行，全部可配置参数模板
├── requirements.txt           # 依赖清单 (>= 最低版本，无上限)
├── docker-compose.yml         # 一键拉起 Neo4j/Qdrant/Postgres
├── LICENSE                    # MIT, 2026 llzx177067-rgb
├── .gitignore                 # 排除 .env、data 卷、cache 中间产物
│
├── scripts/                   # 离线流水线入口 (00~09)
│   ├── 00_smoke_test.py       # 连接性自检
│   ├── 01_load_data.py        # 解析入 PG
│   ├── 02_build_vectors.py    # 建 Q/A 向量
│   ├── 03_build_graph_skeleton.py
│   ├── 04_extract_entities.py # LLM 抽实体 (最重)
│   ├── 05_extract_relations.py# LLM 抽关系 + 推 Neo4j
│   ├── 06_build_topic_summaries.py
│   ├── 06b_resync_topic_vectors.py  # 补丁：补向量
│   ├── 06c_fill_missing_summaries.py# 补丁：补摘要
│   ├── 07_build_testset.py    # 转发 eval
│   ├── 08_evaluate.py         # 转发 eval
│   └── 09_dedup_entities.py   # Union-Find 去重
│
└── src/
    ├── config.py              # pydantic-settings 单例
    ├── clients/               # DashScope 三类模型客户端
    │   ├── _base.py           # 异步 HTTP 基座: 限流+重试+日志
    │   ├── llm.py             # chat / chat_json / chat_many
    │   ├── embedding.py       # 三端点路由+级联 fallback
    │   └── rerank.py          # cross-encoder 重排
    ├── ingest/                # 数据摄入
    │   ├── parse.py           # chunkId 三层解析 → QARecord
    │   ├── prompts.py         # 全部 LLM prompt + 类型枚举
    │   ├── entity_extract.py  # 实体抽取 (LLM→PG)
    │   ├── relation_extract.py# 关系抽取 (LLM→PG→Neo4j)
    │   ├── entity_dedup.py    # Union-Find 去重
    │   ├── summary.py         # 子主题摘要 (LLM→PG+Qdrant)
    │   └── storage.py         # PG DDL + 读写
    ├── graph/                 # Neo4j
    │   ├── schema.py          # 约束/索引 bootstrap
    │   ├── builder.py         # 骨架 upsert
    │   └── client.py          # driver 单例
    ├── index/vector_store.py  # Qdrant 三视图封装
    ├── retrieval/             # 检索
    │   ├── query_understanding.py
    │   ├── fusion.py          # RRF
    │   ├── graph_recall.py    # 图通道 0/1/2-hop
    │   └── pipeline.py        # 端到端混合检索
    ├── generation/answer.py   # 带引用答案生成
    ├── eval/                  # 测试集 + 评测
    └── api/
        ├── server.py          # FastAPI, 8 个端点
        └── static/index.html  # 1179 行原生 JS 调试台
```

---

## 五、逐模块剖析

### 5.1 配置（`src/config.py`）

- `pydantic-settings` 的 `Settings` 单例，`get_settings()` 用 `@lru_cache` 保证进程内唯一。
- 从项目根 `.env` 读全部参数，`case_sensitive=False, extra="ignore"`。
- **默认值里硬编码了三个后端的 DSN/密码**（内网工具，符合项目边界，但仍是明文默认）。
- 检索默认：`recall_top_k_vector=30`、`recall_top_k_graph=20`、`rerank_top_k=8`。
- `llm_model_name_2` 留空则单模型；设了则双模型 round-robin 分摊速率池。

### 5.2 DashScope 客户端（`src/clients/`）

**`_base.py`** — 所有模型调用的共享异步基座：
- 并发控制 = `asyncio.Semaphore(concurrency)` + `aiolimiter.AsyncLimiter(rps)`，顺序"先信号量后限流"。
- 重试 = `tenacity.AsyncRetrying`，指数退避（`min=1, max=20`），只重试 429/5xx/网络错误；400/401/404 **不重试**直接抛。
- 结构化日志：import 时注册 loguru sink 到 `data/cache/model_calls.log`（rotation 50MB），记录每次调用的 `req_id`、attempt、耗时、usage tokens。
- ⚠️ 非幂等的 chat 重试可能产生重复调用；`rps` 被 `int()` 截断（4.0→4）。

**`llm.py`** — OpenAI 兼容 `/compatible-mode/v1/chat/completions`：
- `chat` / `chat_json`（先带 `response_format` 试一次，失败去掉重试）/ `chat_many`。
- `_safe_json`：先抓 ```json``` 代码块，再从首 `[`/`{` 向后枚举 end 尝试解析，最后兜底。
- ⚠️ `chat_many` 的 `gather` 无 per-item 容错（当前 ingest 未用它，各 stage 自己 gather）。

**`embedding.py`** — 按模型名自动路由三种端点：
- `compat`（`/compatible-mode/v1/embeddings`）/ `native_text` / `native_mm`（名字含 `vl`/`multimodal`）。
- **级联 fallback**：compat 404 → 切 native_text → 再失败切 native_mm，且 mode **进程内粘滞**不回落。
- `embed_batched` 按 `embed_batch_size` 切片并发、保序。
- ⚠️ 固定 `text_type="document"`，未区分 query/document 两侧（潜在精度点）。

**`rerank.py`** — 原生 cross-encoder，`/api/v1/services/rerank/...`，返回 `[(index, score)]`，`top_n` 默认不截断（交给 pipeline）。

### 5.3 数据摄入（`src/ingest/`）

**`parse.py`** — 见上文 chunkId 切分；`qa_id = blake2b(chunkId||q||seq, 8字节)`，确定性可复现。

**`prompts.py`** — 全部 prompt + 受控词表：
- **9 类实体**：Disease, Drug, Symptom, Procedure, Lab, Pathogen, Anatomy, Gene, Population。
- **12 类关系**：TREATS, PREVENTS, CAUSES, PRESENTS_WITH, DIAGNOSED_BY, RISK_FACTOR_OF, COMPLICATION_OF, CONTRAINDICATED_FOR, SIDE_EFFECT_OF, DIFFERENTIAL_DX, INTERACTS_WITH, AFFECTS（兜底）。
- 实体 prompt 强约束：只受控类型、同义词合并、不抽剂量/修饰词、description ≤30 字。
- ⚠️ 类型枚举是**硬编码领域词表**，扩展要改这里。

**`entity_extract.py`** — 一条 QA 一次 LLM 调用（答案短）：
- 磁盘断点续跑（JSONL append + `done_ids`）；双模型 round-robin。
- `normalise_and_persist`：按 `(norm_name, type)` 归并，写 `entity` + `qa_entity_mention`。
- ⚠️ 坑：canonical 的 `name` 取**首次出现**的写法，若首次是别名则 `entity_id` 与规范名可能不一致（真正去重靠下游 `entity_dedup`）；description 截断 **80 字**（prompt 要求 30，不一致）。

**`relation_extract.py`** — 只抽"实体数 ≥2"的 QA：
- 约束 head/tail 必须来自已知实体、rel 必须受控、`evidence` 截 160 字。
- `push_graph_layer`：UNWIND 批推 Neo4j，用 **APOC** `apoc.create.addLabels` 加类型 label、`apoc.merge.relationship` 建类型化关系边（属性带 confidence/evidence）。
- ⚠️ `_name_to_entity_map` 类型冲突处理**与 docstring 不符**（注释说按频次，实现是"后写覆盖"）。**强依赖 APOC 插件**。

**`entity_dedup.py`** — 确定性 Union-Find（无 LLM）：
- 只在**同 type 内**合并（保守，避免 Gene 误并到 Disease）。
- 连边：归一化 name 完全相同，或 name↔alias 交叉。
- canonical 选 `(-mention_count, len(name), id)`；应用时"先 INSERT 后 DELETE"保 surface 粒度。幂等，支持 `--dry-run`。

**`summary.py`** — 每个 (specialty, topic) 生成 100–180 字导读：
- 采样每 topic **前 8 条** QA（按 part,seq 顺序，非随机）→ LLM（temp=0.3）→ PG `topic.summary` + Qdrant `medgraph_topic`。
- ⚠️ LLM 生成与向量 upsert 耦合在一次运行里，失败会留下 PG/Qdrant 不一致——这正是 06b/06c 存在的原因。

**`storage.py`** — Postgres 全部 DDL + 读写（psycopg3 **同步** API，CLI 场景够用）。8 张表，`qa_pair.qa_text` 是生成列 `question || '\n' || answer`。

### 5.4 图构建（`src/graph/`）

- **`schema.py`**：5 个唯一约束（按属性，如 `Topic.key`）+ 3 个索引，`IF NOT EXISTS`。
- **`builder.py`**：层级边 `QAPair-[:BELONGS_TO]->Chunk-[:BELONGS_TO]->Topic-[:BELONGS_TO]->Specialty`；QAPair 用生成器流式 upsert（不占内存）。**不**建 MENTIONS 边（那是 relation_extract 的活）。
- **`client.py`**：driver 进程级单例，无连接池/重试参数。

### 5.5 向量库（`src/index/vector_store.py`）

- 三 collection 封装，COSINE 距离，为 `specialty/topic/chunk_id` 建 KEYWORD payload 索引（供过滤检索）。
- **换嵌入模型自动重建**：`_ensure_collection` 比对 dim，不符则 delete & recreate。
- **确定性 point_id**：QA 用 `uuid.UUID(qa_id 左补 32 位)`（qa_id 是 blake2b 8 字节=16 hex）；topic 用 `uuid5(NAMESPACE_DNS, "specialty::topic")`。
- 非对称双视图：Q-vec 只嵌 `question`，A-vec 嵌 `question\nanswer`（截 1500 字）。
- ⚠️ `_qa_point_id` 依赖 qa_id ≤32 hex，定义变了会崩；upsert 每批 3 次重试无退避。

### 5.6 检索（`src/retrieval/`）

**`query_understanding.py`** — 一次 LLM 调用同时产出 `rewritten` / `entities`（3-6 个受控类型）/ `specialty_hint`。异常兜底为原问题。

**`graph_recall.py`** — 从锚点实体（含别名匹配）遍历 Neo4j，4 段 Cypher 打分：
- DIRECT 1.0（被 MENTIONS 直接提及）/ ONE_HOP 0.7 / BACKWARD 0.7 / TWO_HOP 0.4，取 evidence_qa 作候选。
- 同一 qa_id 取最高分；任一 Cypher 异常只降级不中断。

**`fusion.py`** — 标准 RRF：`score(d) = Σ w_i / (k + rank_i + 1)`，k=60。

**`pipeline.py`** — 端到端（时序）：
1. 查询理解 → 2. 三路并发召回（向量两路走 `asyncio.to_thread`，图召回直接同步跑）→ 3. 科室过滤**空则去掉 filter 重试** → 4. RRF 融合（权重 `[1.0, 1.0, 0.8]`，图通道低一点；取前 `max(rerank_top_k*4, 20)` 进重排）→ 5. cross-encoder 重排（失败 fallback 到 RRF 序）→ 6. 组装 `RetrievedItem`（含各通道原始分/rank、同 chunk 前 3 兄弟问）+ 子图三元组。

⚠️ RRF 权重/k、重排候选数、图召回权重 1.0/0.7/0.4 全部**硬编码在 pipeline**（非 config）；`graph_recall` 同步调用未包线程（潜在阻塞点，但 Neo4j 快）。

### 5.7 答案生成（`src/generation/answer.py`）

- 空证据直接返回"未找到相关证据"（不调 LLM）。
- `_format_evidence`：每条 item 带 `[E{i} | chunkId | 科室 | 子主题]` + Q/A + 兄弟问；末尾拼前 20 条三元组。
- system prompt 硬约束：只用证据、每段末尾 `[chunkId]` 引用、不给诊断/剂量。单次 `chat(temp=0.2, max_tokens=1200)`。
- ⚠️ **引用是 LLM 自己写的 `[chunkId]`，无后处理校验**该 chunkId 是否真在证据里（潜在幻觉引用点）。

### 5.8 评测（`src/eval/`）

- `build_test_set`：按 specialty 分层、每组 shuffle（seed=42）取前 20 条；**gold 是 chunk_id**（按 chunk 命中算，不是 qa_id）。
- `evaluate`：并发跑完整 pipeline，算 QA Recall@k / Chunk Recall@k / Chunk MRR@k。
- ⚠️ **测试集未做训练/测试隔离**：抽样的 QA 仍在向量库/图里，属"记忆式"评估，非真正泛化测试。

---

## 六、脚本流水线（`scripts/`）

执行顺序即编号顺序，每个脚本头部 `sys.path.insert` 把 `src` 纳入路径，统一 `loguru` + `get_settings()`。

| 脚本 | 职责 | CLI | 备注 |
|---|---|---|---|
| **00_smoke_test** | 探活 3 模型 + 3 后端 | — | 打印配置（key 打码）、计时 |
| **01_load_data** | 解析入 PG | — | 批量插 QA（batch 2000） |
| **02_build_vectors** | 建 Q/A 向量 | `--view {q,a,both}` `--limit` | 断点续跑（Qdrant point_id） |
| **03_build_graph_skeleton** | 建 Neo4j 骨架 | — | batch 1500 |
| **04_extract_entities** | LLM 抽实体（最重，30k QA 约 30-60min） | `--limit` `--no-resume` `--persist-only` | JSONL 断点续跑 |
| **05_extract_relations** | LLM 抽关系 + 推图 | `--limit` `--no-resume` `--persist-only` `--push-graph` | 只抽实体≥2 的 QA |
| **06_build_topic_summaries** | 子主题摘要 | `--limit` `--concurrency` | LLM+向量耦合 |
| **06b_resync_topic_vectors** | **补丁**：PG→Qdrant 重同步 topic 向量 | — | 不调 LLM，补 06 向量阶段失败 |
| **06c_fill_missing_summaries** | **补丁**：补 06 里没生成摘要的 topic | `--concurrency`(默认4) `--limit` | 并发调低避免抢配额 |
| **07_build_testset** | 建测试集 | 透传 | 纯转发 `eval.build_test_set.main` |
| **08_evaluate** | 评测 | 透传 | 纯转发 `eval.evaluate.main` |
| **09_dedup_entities** | 实体去重 | `--dry-run` | 幂等，可重复 |

**06b/06c 为什么存在**：06 把"LLM 生成摘要"和"向量 upsert"耦合在一次运行里。向量阶段瞬时断连失败 → **06b** 只补向量（不重跑昂贵 LLM）；个别 topic LLM 失败/超时 → **06c** 只补摘要。两者互补覆盖 06 的失败残差。

---

## 七、API 层（`src/api/server.py`）

FastAPI 应用，`_lifespan` 启动时**预热并单例化** LLM/Embedding/Rerank/Retriever 四个 client 存模块级 `_state`，关闭时 `aclose()`。

| 方法·路径 | 参数 | 返回 |
|---|---|---|
| `GET /health` | — | `{ok: true}` |
| `GET /specialties` | — | 各科室 + n_topics/n_qa/有摘要的 topic 数 |
| `GET /entity` | `name`, `limit` | 匹配实体 + 关联关系 + 提及的 QA |
| `GET /random_qa` | `specialty?`, `n` | 随机抽问题（`ORDER BY random()`） |
| `GET /stats` | — | PG/Neo4j/Qdrant 计数（Neo4j 用 APOC，不可用则回退） |
| `POST /retrieve` | `{question, top_k?}` | 检索 bundle（含三路通道分/rank） |
| `POST /ask` | `{question, top_k?, with_siblings}` | 检索 + LLM 生成带引用答案 |
| `GET /` `/ui/*` | — | 静态调试台 |

`_bundle_payload` 是**前后端唯一契约**：每个 item 含 `qa_id, score, rrf_score, sources[], channel_scores{}, channel_ranks{}`，顶层含 `query, rewritten, entities, specialty_hint, items[], triples[], timings_ms{}, counts{}`——正是调试台能可视化"每路召回的原始分/融合分/各阶段耗时"的根源。

### 前端调试台（`src/api/static/index.html`, 1179 行）

原生 JS + fetch，无框架。两栏布局（左：查询理解→耗时→答案→召回列表；右：三元组 SVG 图 + 原始 JSON）。

- **实体类型专属配色**（.et-Disease 等 9 类）、**来源标签配色**（q 蓝 / a 绿 / graph 紫）。
- 引用 `[chunkId]` 可点击 → `flashItem` 高亮并滚动到对应召回项。
- 实体 chip 可点击 → 弹窗展示该实体的匹配/关系/提及 QA。
- 召回列表受"来源筛选（multi=≥2 路命中）+ 科室筛选"双重过滤。
- 三元组环形布局 SVG（前 30 条）+ 文本列表（前 80 条）。
- 科室 chip 点击 → 随机取该科室问题自动搜索。

形成"检索结果 ↔ 知识图谱 ↔ 实体"的可调试闭环。

---

## 八、部署与配置

**`docker-compose.yml`** — 一键三服务，统一密码 `medgraph_pwd`：
- `neo4j:5.22-community`（7474 Browser + 7687 Bolt，**启用 APOC**：`NEO4J_PLUGINS=["apoc"]` + 放开 `apoc.*`，堆 1G/2G）
- `qdrant/qdrant:v1.11.0`（6333 REST + 6334 gRPC）
- `postgres:16-alpine`（5432）
- 无自定义网络、无 `depends_on`，靠各自 healthcheck。

**`.env.example`** 关键项：`BASE_URL`（百炼 endpoint）、三个模型名、`API_KEY`、`LLM_CONCURRENCY=8`、`LLM_RPS=4`、`EMBED_CONCURRENCY=8`、`EMBED_BATCH_SIZE=10`、`RERANK_CONCURRENCY=4`、`EMBED_DIM=1024`、三后端 DSN、三 collection 名、`DATA_FILE`、`CACHE_DIR`、三个检索 top-k。

**`.gitignore`** — 排除 `.env`（密钥永不提交）、`data/postgres|neo4j|qdrant`（运行时卷）、`data/cache/*.jsonl|*.log`（可再生中间产物）；但 `Professional.json` **未 ignore，随仓库提交**（26MB）。个人备份脚本 `scripts/99_backup_*.sh` 被排除（公司工具不进开源仓库）。

---

## 九、设计亮点

1. **可复现性贯穿**：`qa_id`/`entity_id`/`point_id` 全部确定性哈希（blake2b/uuid5）；02/04/05 断点续跑；09 幂等 + `--dry-run`。
2. **非对称双视图 + RRF**：Q-vec/A-vec 分开检索再 RRF 融合，对跨通道分数尺度不一致鲁棒（RRF 只用 rank）。
3. **三路召回互补**：向量（语义）+ 图（结构化关系，0/1/2-hop），图通道权重略低（0.8）。
4. **可调试性**：`_bundle_payload` 暴露每路原始分/rank/各阶段耗时，前端全可视化。
5. **补丁脚本体系**：06b/06c 把昂贵 LLM 和廉价向量/摘要分离，失败可低成本修复。
6. **鲁棒降级**：科室过滤空→去掉重试；rerank 失败→RRF 序；图召回异常→只降级不中断；查询理解失败→原问题。

## 十、已知坑 / 待改进

1. **测试集未做训练/测试隔离**——抽样的 QA 仍在索引里，是记忆式评估，Recall@8=92% 偏高，非真实泛化成绩。
2. **`EMBED_DIM` 文档矛盾**：README 全文写 dim=2560，`.env.example` 写 1024；实际以模型输出维度为准（代码用 `embed([first])` 探测真实 dim，config 只是初始猜测）。
3. **模型命名过期**：README 的 `deepseek-v4-flash`/`qwen3.6-flash`/`qwen3-vl-embedding` 与百炼真实模型 ID（`qwen-plus`/`text-embedding-v3`/`gte-rerank`）对不上，以 `.env` 为准。README 还提到 `LLM_MODEL_NAME_2` 双模型项但 `.env.example` 缺失。
4. **引用无后校验**：答案里 LLM 自写的 `[chunkId]` 不验证是否真在证据中（幻觉引用风险）。
5. **`_name_to_entity_map` 实现与注释不符**（类型冲突处理）。
6. **实体归一化 canonical 取首现写法**（`entity_extract`），真正去重依赖下游 09。
7. **部分参数硬编码**在 pipeline/extract（RRF 权重、重排候选数、图召回权重、description 截 80 字 vs prompt 说 30、topic 采样"前 8 条"）。
8. **同步调用未包线程**：`graph_recall`（Neo4j）直接在 async 事件循环里同步跑。
9. **DB 密码明文默认值**写在 config.py 默认参数里（内网工具可接受，但仍是明文）。
10. **`Professional.json` 入版本库**（26MB），git 仓库体积会持续膨胀。
11. **README "生产就绪" 是营销口径**——实际是单机 Docker Compose + 本地调试台。
