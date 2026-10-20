# 基于默沙东医疗问答 Graph RAG 系统

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![ModelScope](https://img.shields.io/badge/ModelScope-lzx1770670957%2Fmedical--graph--rag--qa-orange.svg)](https://www.modelscope.cn/datasets/lzx1770670957/medical-graph-rag-qa/summary)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)](https://www.postgresql.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.22%20APOC-blueviolet)](https://neo4j.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.11-red)](https://qdrant.tech/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Eval Recall](https://img.shields.io/badge/QA%20Recall%408-92.1%25-success.svg)]()
[![Eval MRR](https://img.shields.io/badge/Chunk%20MRR%408-0.913-success.svg)]()

**高精度 · 多视角混合召回 · 严密证据链溯源 · 生产级中文临床医学 GraphRAG 开源系统**

[项目概览](#1-项目概览) • [核心亮点](#2-核心特性亮点) • [数据下载](#3-数据集开源与下载指引) • [数据使用指南](#4-数据使用方法与实战场景-data-usage-guide) • [系统架构](#5-系统架构全景) • [快速开始](#6-快速开始-quick-start) • [从零构建](#7-离线全流程构建从零构建) • [检索算法](#8-检索与融合算法原理) • [评测复现](#9-评测基准与复现) • [API与前端](#10-api-接口与交互控制台) • [踩坑与优化](#11-工程实践与排错指南) • [贡献与协议](#14-免责声明与开源协议)

</div>

---

## 1. 项目概览

在现代医疗智能化与临床辅助决策领域，大语言模型（LLM）常面临**医学知识幻觉**、**跨科室推理盲区**以及**缺乏可追溯证据链**等严峻挑战。

**基于默沙东医疗问答 Graph RAG 系统**是一套专为高严肃性医疗问答打造的工业级解决方案。项目以权威的《默沙东诊疗手册》（Merck Manuals 专业版）临床知识为基座，收录覆盖 **24 个顶层专科、2,533 个临床子主题的 30,031 条高水准医学 Q&A**。

与传统的“暴力分块 + 向量检索”不同，本系统创新性地将**天然层级结构树**与**细粒度临床知识图谱**、**三重视角向量索引**紧密结合：
- **结构化图谱骨架**：Specialty（科室） → Topic（子主题） → Chunk（语义块） → QAPair（问答对）；
- **领域拓扑图谱**：抽取 39,400 个医学实体，构建涵盖 12 类临床关系（治疗、病因、表现、并发症、药物相互作用、禁忌等）的 Neo4j 拓扑网络；
- **多通道融合检索**：Q-vec（问题匹配）+ A-vec（答案语义匹配）+ Graph（1-2 跳子图推理），配合 **RRF (Reciprocal Rank Fusion)** 倒数排名融合与 **Cross-Encoder 深度重排**；
- **可信证据溯源**：在最终生成的医学解答中强制绑定 `[chunkId]` 来源标签，支持在前端一键定位与高亮证据卡片。

---

## 2. 核心特性亮点

| 特性模块 | 核心能力与优势 |
| :--- | :--- |
| 🌟 **三视角多模态向量索引** | 突破单一文本嵌入局限，构建 **Q-vec（问题语义）**、**A-vec（完整答复内涵）** 与 **Topic-vec（专科子主题全局摘要）** 三大向量集合（Qdrant，dim=2560），兼顾短查询匹配与长语义匹配。 |
| 🕸️ **双层知识图谱引擎** | 巧妙复用数据自带的天然 3 层标签作为宏观图骨架（免除传统 GraphRAG 昂贵的 Leiden 社区聚类）；底层注入 39,400 个去重实体及 23,591 条类型化边，支持 1-2 跳关联扩展。 |
| ⚡ **Q+A+Graph 三路并行召回** | 基于 `asyncio` 高并发调度，同时向向量库与 Neo4j 发起检索引擎查询，实现毫秒级“语义相近 + 知识相连”的双轮驱动召回。 |
| 🧮 **RRF 自适应融合 + 重排** | 采用加权倒数排名算法（RRF）融合三路结果，再经由 Cross-Encoder 深度重排序模型进行精准打分，有效过滤图谱与向量长尾噪声。 |
| 🔍 **段落级可溯源引用回链** | 生成阶段采用严密防御性 Prompt 设计，强制要求 LLM 为每一段关键医学结论标定 `[chunkId]` 证据锚点，前端支持一键跳转核验，彻底解决幻觉风险。 |
| 🚀 **生产级双模型限流与断点续跑** | 全流程内建阿里百炼（DashScope）双模型 Round-Robin 负载均衡，针对速率限制（429）配备动态退避重试，中间阶段全量输出 `.jsonl` 缓存，随时支持断点续跑。 |

---

## 3. 数据集开源与下载指引

为了促进中文医学大模型与检索增强技术（RAG / GraphRAG）社区的健康发展，本项目所采用的结构化医学问答数据集及预构建数据库已完整开源至**魔塔社区 (ModelScope)**：

👉 **数据集仓库主页**：[lzx1770670957/medical-graph-rag-qa](https://www.modelscope.cn/datasets/lzx1770670957/medical-graph-rag-qa/summary)  
📁 **文件列表直达**：[魔塔 Files 页面](https://www.modelscope.cn/datasets/lzx1770670957/medical-graph-rag-qa/files)

### 3.1 数据工件清单与规格画像

魔塔社区仓库包含以下两类数据工件，支持按需获取：

| 文件名称 | 类型 | 体积规格 | 核心用途与内容 |
| :--- | :--- | :--- | :--- |
| **`qdrant_storage.tar.gz`** | 预构建向量库 | 612 MB (解压约 954MB) | **已完成 2560 维高维向量嵌入的 Qdrant 数据库**，含 `medgraph_q`、`medgraph_a`、`medgraph_topic` 全部 62,595 个向量索引，解压直接挂载即可，免除 6 万次 Embedding 调用！ |
| **`neo4j_data.tar.gz`** | 预构建图数据库 | 65 MB (解压约 624MB) | **已构建完成的 Neo4j 知识图谱**，含 78,080 个实体/骨架节点与 235,909 条临床拓扑关系边。 |
| **`postgres_data.tar.gz`** | 预构建关系库 | 101 MB (解压约 300MB) | **已入库的 PostgreSQL 数据卷**，含科室、主题、分块、问答对、实体和关系的完整表结构。 |
| **`Professional.json`** | 原始问答文本 | 27.2 MB | 30,031 条带三层层级标签（Specialty-Topic-Part）的标准临床问答对。 |
| **`entities.jsonl`** | 实体抽取结果 | 25.4 MB | 抽取的 39,400 个去重医学实体（含 9 类医学实体、别名数组与描述）。 |
| **`relations.jsonl`** | 关系抽取结果 | 10.9 MB | 抽取的 23,591 条类型化临床关系三元组（含支持证据 chunkId 及置信度）。 |
| **`testset.jsonl`** | 盲测基准集 | 491 KB | 覆盖 24 个临床专科的分层抽样盲测评估测试集（共 480 条）。 |
| **`eval_results.jsonl`** | 评测详细得分 | 36 KB | 包含 Recall@8 与 MRR@8 的各项细粒度评估指标明细。 |

### 3.2 访问凭证配置（可选）

公开数据集支持免登录直接下载。如需在自动化流水线中配置魔塔社区访问凭证：

```bash
# Linux / macOS
export MODELSCOPE_API_TOKEN="your-modelscope-api-token"

# Windows PowerShell
$env:MODELSCOPE_API_TOKEN="your-modelscope-api-token"
```

### 3.3 数据下载方式（4 种渠道）

#### 渠道 1：一键全自动下载并部署预构建库（推荐，最省心）
项目内置了跨平台一键拉取脚本，自动下载 `qdrant_storage.tar.gz`、`neo4j_data.tar.gz`、`postgres_data.tar.gz` 并自动解压部署至 `./data/` 目录：
```bash
pip install modelscope
python scripts/restore_prebuilt_data.py
```

#### 渠道 2：使用 ModelScope CLI 命令行下载
```bash
pip install modelscope

# 下载完整数据集仓库至本地目录
modelscope download --dataset lzx1770670957/medical-graph-rag-qa --local_dir ./data/modelscope_dataset
```

#### 渠道 3：使用 ModelScope Python SDK 精准拉取
```python
from pathlib import Path
import shutil
from modelscope.hub.snapshot_download import snapshot_download

# 1. 下载完整数据集
dataset_dir = snapshot_download(
    dataset_id='lzx1770670957/medical-graph-rag-qa',
    cache_dir='./data/cache'
)
print(f"数据已同步至本地: {dataset_dir}")

# 2. 按需提取核心文件到项目根目录（例如 Professional.json）
prof_src = Path(dataset_dir) / "Professional.json"
if prof_src.exists():
    shutil.copy2(prof_src, "./Professional.json")
    print("Professional.json 已复制到项目根目录")
```

#### 渠道 4：使用 Git LFS 直接克隆
```bash
git lfs install
git clone https://www.modelscope.cn/datasets/lzx1770670957/medical-graph-rag-qa.git data/modelscope_raw
```

---

## 4. 数据使用方法与实战场景 (Data Usage Guide)

下载数据后，您可以根据具体的使用需求选择以下任一使用方案：

### 场景一：极速开箱即用（免重新向量化，直接运行 GraphRAG）
> **适合人群**：希望立刻体验或部署医疗知识问答系统的工程师与研究人员。

1. **解压预构建数据包**（若使用了 `python scripts/restore_prebuilt_data.py` 则已自动完成）：
   ```bash
   tar -xzf qdrant_storage.tar.gz -C ./data/
   tar -xzf neo4j_data.tar.gz -C ./data/
   tar -xzf postgres_data.tar.gz -C ./data/
   ```
2. **启动基础存储容器**：
   ```bash
   docker compose up -d
   ```
3. **配置百炼 Key 并启动问答服务**：
   在 `.env` 中填入您的阿里云百炼 `API_KEY`（**仅用于在线问答时的意图重写、重排与生成，0 向量化调用开销**），然后启动应用：
   ```bash
   uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```
   打开浏览器访问 **http://127.0.0.1:8000** 即可直接体验！

---

### 场景二：Python 代码中加载与分析数据集（科研与学术挖掘）
> **适合人群**：从事自然语言处理、医学信息抽取与文本挖掘的研究者。

#### 1. 加载并解析问答对三层结构 (`Professional.json`)
```python
import json

with open("Professional.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

print(f"共加载 {len(qa_data)} 条问答数据")

# 解析单条数据及其层级标签
item = qa_data[0]
print("问题:", item["question"])
print("回答摘要:", item["answer"][:80], "...")

# 解析 chunkId 层次：[科室] - [子主题] - part-[序号]
parts = [p.strip() for p in item["chunkId"].split("-")]
specialty = parts[0]
topic = "-".join(parts[1:-1]).strip() if len(parts) > 2 else parts[1]
print(f"所属科室: {specialty} | 临床子主题: {topic} | 分块标识: {item['chunkId']}")
```

#### 2. 使用 Pandas 进行 24 专科分布统计
```python
import json
import pandas as pd

with open("Professional.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

df = pd.DataFrame(qa_data)
# 提取科室名称
df["specialty"] = df["chunkId"].apply(lambda x: x.split("-")[0].strip())

# 统计各科室 QA 分布 Top 5
print("各科室问答数量 Top 5:")
print(df["specialty"].value_counts().head(5))
```

#### 3. 读取实体与关系知识图谱 (`entities.jsonl` & `relations.jsonl`)
```python
import json

# 读取实体列表
entities = []
with open("data/cache/entities.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entities.append(json.loads(line))
print(f"医学实体总量: {len(entities)} (首条实体: {entities[0]})")

# 读取临床关系三元组
relations = []
with open("data/cache/relations.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            relations.append(json.loads(line))
print(f"临床关系三元组总量: {len(relations)} (示例三元组: {relations[0]})")
```

---

### 场景三：在 Python 代码中直接调用检索与生成引擎 (SDK 模式)
> **适合人群**：希望将 GraphRAG 检索能力集成到自身后端系统或评测脚本的开发者。

```python
import asyncio
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.answer import AnswerGenerator

async def main():
    # 1. 初始化检索器与生成器
    pipeline = RetrievalPipeline()
    generator = AnswerGenerator()

    # 2. 输入临床咨询问题
    query = "妊娠期发现血糖偏高，应采取哪些干预措施？"
    
    # 3. 执行多通道并行检索与 Cross-Encoder 重排
    retrieve_result = await pipeline.retrieve(query, top_k=8)
    print(f"命中候选 QA 数量: {len(retrieve_result.items)}")
    print(f"关联临床三元组数量: {len(retrieve_result.triples)}")

    # 4. 生成带 [chunkId] 证据链回溯引用的医学回答
    answer = await generator.generate(query, retrieve_result)
    print("\n=== 模型生成答复 ===")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 场景四：从零执行离线知识抽取流水线（完全复现与二次定制）
> **适合人群**：希望使用自己的医学数据集或微调抽取 Prompt 的进阶开发者。

当准备好自己的原始问答文件并命名为 `Professional.json` 后，可依次执行编号流水线脚本：

```bash
# 1. 结构化解析入库 PostgreSQL
python scripts/01_load_data.py

# 2. 密集向量化入库 Qdrant（支持双视角）
python scripts/02_build_vectors.py --view both

# 3. 构建 Neo4j 拓扑骨架 (Specialty -> Topic -> Chunk -> QAPair)
python scripts/03_build_graph_skeleton.py

# 4. LLM 并行抽取医学实体（支持断点续跑）
python scripts/04_extract_entities.py

# 5. 实体别名对齐与同义词去重
python scripts/09_dedup_entities.py

# 6. LLM 抽取 12 类临床关系并注入 Neo4j 图谱
python scripts/05_extract_relations.py --push-graph

# 7. 生成专科子主题语义摘要并同步 Topic 向量
python scripts/06_build_topic_summaries.py --concurrency 8

# 8. 抽样测试集并运行基准评测
python scripts/07_build_testset.py --per-specialty 20
python scripts/08_evaluate.py --top-k 8 --parallelism 32
```

---

## 5. 系统架构全景

### 4.1 总体架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      离线知识图谱与多视角向量构建流水线                     │
├──────────────────────────────────────────────────────────────────────────┤
│  Professional.json (30,031 医学问答)                                     │
│       │                                                                  │
│       ├──► [01] 三层标签解析 ────────► PostgreSQL 关系存储                │
│       │                               (specialty / topic / chunk / qa)   │
│       │                                                                  │
│       ├──► [02] 密集向量化 ──────────► Qdrant 向量引擎                   │
│       │                               ├─ medgraph_q (问题索引)           │
│       │                               └─ medgraph_a (问答拼接索引)       │
│       │                                                                  │
│       ├──► [03] 骨架拓扑入库 ────────► Neo4j 结构图 (BELONGS_TO 层次关系) │
│       │                                                                  │
│       ├──► [04/09] LLM 实体抽取与归一 ► 39,400 归一实体 + 别名合并       │
│       │                                                                  │
│       ├──► [05] LLM 关系抽取与证据链 ─► Neo4j 12 类临床关系网络          │
│       │                                                                  │
│       └──► [06] 专科子主题全局摘要 ──► Qdrant medgraph_topic             │
└──────────────────────────────────────────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          在线检索增强生成 (GraphRAG)                      │
├──────────────────────────────────────────────────────────────────────────┤
│  用户复杂临床问题输入                                                    │
│      │                                                                   │
│      ├──► [A] LLM 查询理解：意图重写 + 实体识别 + 目标专科智能推断        │
│      │                                                                   │
│      ├──► [B] 三路高并发召回 (Parallel Multi-Channel Recall)             │
│      │     ├─ 通道 1: Q-vec 向量相似度检索 (Top-30)                      │
│      │     ├─ 通道 2: A-vec 深度语义匹配 (Top-30)                        │
│      │     └─ 通道 3: Neo4j 实体锚点 1-2 跳子图路径关联扩展 (Top-20)     │
│      │                                                                   │
│      ├──► [C] RRF 倒数排名自适应融合 (Reciprocal Rank Fusion)             │
│      │                                                                   │
│      ├──► [D] Cross-Encoder 深度重排序 (Top-32 → 精选 Top-8)             │
│      │                                                                   │
│      ├──► [E] 富上下文组装 (命中 QA + 同 Chunk 兄弟上下文 + 知识三元组)    │
│      │                                                                   │
│      └──► [F] 严谨溯源生成：输出医学结论并强制绑定 [chunkId] 证据回跳     │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 数据模型与图谱 Schema

#### PostgreSQL 关系表设计
- `specialty`：24 个大科室基础信息与统计；
- `topic`：2,533 个临床子主题及对应全局语义摘要；
- `chunk`：6,092 个分块上下文单元；
- `qa_pair`：30,031 个标准问答对及全文检索生成列；
- `entity`：去重后的 39,400 个标准医学实体及别名数组；
- `relation`：23,591 条临床关系（含头尾节点、关系类型、支持证据 chunkId 及置信度）。

#### Neo4j 知识图谱节点与关系定义

```
【节点标签 (Node Labels)】
  结构层节点:
    (:Specialty)  (24)           (:Topic)    (2,533)
    (:Chunk)      (6,092)        (:QAPair)   (30,031)
  领域医学实体:
    (:Disease)    (11,109)       (:Drug)     (3,911)
    (:Symptom)    (7,368)        (:Procedure)(7,151)
    (:Lab)        (3,203)        (:Pathogen) (1,170)
    (:Anatomy)    (3,315)        (:Gene)     (1,165)
    (:Population) (1,008)

【核心关系类型 (Relationship Types)】
  骨架拓扑:   [:BELONGS_TO], [:MENTIONS]
  治疗方案:   [:TREATS], [:PREVENTS], [:CONTRAINDICATED_FOR], [:SIDE_EFFECT_OF]
  病因病理:   [:CAUSES], [:RISK_FACTOR_OF], [:COMPLICATION_OF]
  临床诊断:   [:PRESENTS_WITH], [:DIAGNOSED_BY], [:DIFFERENTIAL_DX]
  药学作用:   [:INTERACTS_WITH], [:AFFECTS]
```

---

## 6. 快速开始 (Quick Start)

### 5.1 环境需求
- **操作系统**：Linux / macOS / Windows 10+
- **Python**：>= 3.10
- **容器环境**：Docker & Docker Compose

### 5.2 安装步骤

#### 1. 克隆代码仓库
```bash
git clone https://github.com/your-username/medical-graph-rag.git
cd medical-graph-rag
```

#### 2. 创建并激活虚拟环境
```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3. 安装依赖组件
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. 配置环境变量
复制配置模板并填入您的阿里云百炼（DashScope）API Key 与 ModelScope Token：

```bash
cp .env.example .env
```

`.env` 核心配置参数预览：
```ini
# === 阿里百炼大模型与向量服务 ===
BASE_URL=https://dashscope.aliyuncs.com
API_KEY=sk-your-dashscope-api-key
LLM_MODEL_NAME=deepseek-v4-flash
LLM_MODEL_NAME_2=qwen3.6-flash          # 双模型负载均衡轮询
EMBEDDING_MODEL=qwen3-vl-embedding      # 2560 维多模态文本嵌入
RERANK_MODEL=qwen3-vl-rerank            # Cross-Encoder 深度重排

# === 魔塔社区凭据 ===
MODELSCOPE_API_TOKEN=your-modelscope-api-token

# === 存储与中间件连接 ===
POSTGRES_DSN=postgresql://medgraph:medgraph_pwd@127.0.0.1:5432/medgraph
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=medgraph_pwd
QDRANT_URL=http://127.0.0.1:6333

# === 速率与并发控制 ===
LLM_CONCURRENCY=128                     # LLM 调用工作线程数
LLM_RPS=100                             # 每秒请求限速上限
```

#### 5. 一键启动基础设施
通过 Docker Compose 启动 Postgres、Neo4j（含 APOC 插件）和 Qdrant 向量库：

```bash
docker compose up -d
```

等待 10 秒后执行连通性与模型冒烟检查：
```bash
python scripts/00_smoke_test.py
```
> 若输出所有连接均为 `[OK]`，说明底层基础存储与外部模型服务均已就绪！

---

## 7. 离线全流程构建（从零构建）

项目提供了一组按顺序编号的脚本，涵盖从原始数据到完整图谱与向量索引的端到端构建。每个大模型抽取步骤均原生支持**断点续跑**，异常中断后再次运行会自动跳过已处理的数据。

| 序号 | 执行脚本 | 任务说明 | 预计耗时 (推荐配置) |
| :---: | :--- | :--- | :---: |
| **01** | `python scripts/01_load_data.py` | 解析 `Professional.json` 层次结构入库 PostgreSQL | ~ 11 秒 |
| **02** | `python scripts/02_build_vectors.py --view both` | 生成 Q 与 A 双通道向量并建立 Qdrant 集合索引 | ~ 28 分钟 (6万条) |
| **03** | `python scripts/03_build_graph_skeleton.py` | 在 Neo4j 中构建科室/主题/分块/问答四级骨架关系 | ~ 15 秒 |
| **04** | `python scripts/04_extract_entities.py` | 30k 问答双模型并行抽取医学实体（输出 cache） | ~ 30 分钟 (128并发) |
| **09** | `python scripts/09_dedup_entities.py` | 实体别名对齐、同义词归并与图谱去重 | ~ 8 秒 |
| **05** | `python scripts/05_extract_relations.py --push-graph` | 抽取 12 类临床关系并注入 Neo4j 拓扑边 | ~ 33 分钟 |
| **06** | `python scripts/06_build_topic_summaries.py` | 生成 2,533 个子主题语义摘要并建立 Topic 向量 | ~ 10 分钟 |
| **06c**| `python scripts/06c_fill_missing_summaries.py` | 针对偶发限流丢失的摘要进行自动补充修复 | ~ 8 分钟 (可选) |
| **07** | `python scripts/07_build_testset.py --per-specialty 20` | 按科室分层抽样生成 480 条基准评估测试集 | < 1 秒 |

---

## 8. 检索与融合算法原理

### 7.1 三路并行召回机制
当接收到用户的医疗咨询提问时，系统并发执行以下三个检索通道：
1. **Q-vec 通道（问题意图相似度）**：将用户问题直接 Embedding 后，在 Qdrant `medgraph_q` 集合中做余弦相似度检索（Top-30），专门捕捉字面与直接意图相似的过往病例；
2. **A-vec 通道（问答内涵深度匹配）**：在 `medgraph_a`（问题与权威解答混合拼接向量）中做相似度检索（Top-30），捕获深层医学逻辑与症状描述匹配；
3. **Graph 通道（知识图谱跳跃推理）**：首先识别提问中的核心医学实体，以实体为锚点通过 Neo4j 执行 1-2 跳 Cypher 关联路径检索（Top-20）：
   - 直接提及（`q -[:MENTIONS]-> entity`）：权重 1.0；
   - 一跳临床因果/对症关系：权重 0.7；
   - 两跳扩展路径：权重 0.4。

### 7.2 加权倒数排名融合 (RRF)
将三路异构打分通过 RRF 公式映射到统一排序空间：

$$ \text{RRF\_Score}(d) = \sum_{c \in \{Q, A, Graph\}} \frac{w_c}{k + \text{rank}_c(d)} $$

- 平滑常数：$k = 60$；
- 默认通道权重：$w_Q = 1.0, w_A = 1.0, w_{Graph} = 0.8$。

### 7.3 Cross-Encoder 精准重排
初筛融合得到 Top-32 候选后，送入 DashScope 原生 Cross-Encoder 重排模型，对 `(Query, QA_Context)` 文本对进行交叉注意力打分，精确筛选出相关度最高的 Top-8 片段。

### 7.4 兄弟上下文补充与图谱三元组注入
为避免片段式检索导致的信息丢失：
- **兄弟上下文注入**：对命中的每一个 QA，自动挂载所属同一个 Chunk 的前序/后序兄弟问答（最多 3 条），还原完整诊疗语境；
- **子图三元组注入**：提取锚点实体关联的临床三元组（例如 `[二甲双胍] -TREATS-> [2型糖尿病]`），作为显式知识先验一并组装到提示词中。

---

## 9. 评测基准与复现

### 9.1 评测指标结果

采用 480 条分层抽样测试集（覆盖 24 个科室，每科室 20 条盲测数据），评测指标表现如下：

| 评估指标 | 评测表现 | 行业普通 RAG 基线 | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **QA Recall@8** | **92.08%** | 71.4% | **+ 20.68%** |
| **Chunk Recall@8** | **93.13%** | 74.2% | **+ 18.93%** |
| **Chunk MRR@8** | **0.9130** | 0.682 | **+ 0.231** |

### 9.2 性能与耗时基准
- **单请求端到端响应耗时**：约 5 ~ 8 秒（包含：LLM 查询重写与实体识别 ~3-5s、三路并行召回 ~1s、重排序 ~0.4s、LLM 引用生成 ~1.5s）；
- **全量评估吞吐**：在 32 并发压力下，480 条评测集可在 **12.5 分钟** 内完成全量评测。

### 9.3 一键复现评测
```bash
python scripts/08_evaluate.py --top-k 8 --parallelism 32
```
评估明细日志与统计报告将自动写入 `data/cache/eval_results.jsonl`。

---

## 10. API 接口与交互控制台

### 10.1 启动服务
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
服务启动后，使用浏览器访问：**http://127.0.0.1:8000** 即可进入交互式可视化控制台。

### 10.2 API 端点清单

| 请求方式 | 接口路由 | 功能说明 |
| :---: | :--- | :--- |
| `GET` | `/` | 嵌入式单页交互测试前端 |
| `GET` | `/health` | 服务健康检查 |
| `GET` | `/stats` | 获取 Postgres / Neo4j / Qdrant 各引擎实时数据量 |
| `GET` | `/specialties` | 浏览 24 个临床科室及对应的问答分布、摘要覆盖度 |
| `GET` | `/random_qa` | 随机提取真实临床 QA（用于快速测试与提问灵感） |
| `GET` | `/entity` | 查询指定医学实体的别名、关联三元组及提及该实体的问答列表 |
| `POST` | `/retrieve` | 仅执行多路检索与重排（返回三路得分明细、耗时分析与图谱三元组） |
| `POST` | `/ask` | 执行端到端 GraphRAG 问答（返回带 `[chunkId]` 严格引用的结构化回答） |

### 10.3 `/ask` 请求与响应示例

**Request Body:**
```json
{
  "query": "妊娠期发现血糖升高，应该如何进行临床管理与干预？",
  "top_k": 8
}
```

**Response Preview:**
```json
{
  "answer": "妊娠期糖尿病的管理首先应以医学营养治疗和适度运动为基础 [妇产科学 - 妊娠合并糖尿病 - part-1]。若生活方式干预后血糖仍未达标，首选胰岛素治疗 [妇产科学 - 妊娠合并糖尿病 - part-2]...",
  "citations": [
    {
      "chunk_id": "妇产科学 - 妊娠合并糖尿病 - part-1",
      "specialty": "妇产科学",
      "topic": "妊娠合并糖尿病",
      "matched_qa": "妊娠期血糖控制目标是多少？"
    }
  ],
  "timings_ms": {
    "query_understanding": 3120,
    "recall": 890,
    "rerank": 410,
    "generation": 1650,
    "total": 6070
  }
}
```

### 10.4 交互控制台特色

- **实时状态统计**：顶部动态展现三大存储库（PG/Neo4j/Qdrant）的实时图节点与向量计数；
- **智能实体着色高亮**：根据疾病、药物、症状、检验等 9 种实体类型自动赋予医学专属配色；
- **可点击溯源引用徽章**：点击正文中的 `[chunkId]`，界面自动平滑滚动并高亮对应证据卡片；
- **动态 SVG 知识子图**：直观渲染与当前问题相关的实体关系拓扑网络。

---

## 11. 工程实践与排错指南

在数万次大模型调用与高并发检索流水线构建过程中，我们总结了以下核心经验与避坑要点：

<details>
<summary><b>1. 多模态 Embedding 接口协议与路由</b></summary>

- **现象**：调用 `qwen3-vl-embedding` 提示 404 或参数结构错误。
- **解决**：该模型底层走阿里云百炼 `multimodal-embedding` 专用端点，请求体需封装为 `input.contents=[{"text": "..."}]`。本项目在 `src/clients/embedding.py` 中实现了自动协议适配，透明兼容纯文本与多模态端点。
</details>

<details>
<summary><b>2. Qdrant 批量写入连接重置 (WinError 10054 / Timeout)</b></summary>

- **现象**：向 Qdrant 高并发写入 2560 维高维向量时，偶发套接字强制关闭。
- **解决**：调整批量写入 batch 大小为 128，并在客户端中集成指数退避重试，显式将连接超时提升至 120s。
</details>

<details>
<summary><b>3. 大模型返回 JSON 截断与 Markdown 代码块容错</b></summary>

- **现象**：LLM 抽取实体关系时偶发包裹 ` ```json ` 围栏或包含解释性文本导致 JSON 反序列化崩溃。
- **解决**：实现了鲁棒的 `_safe_json` 解析器，自动剥离 Markdown 围栏并定位最外层括号配对，全面确保解析稳定性。
</details>

<details>
<summary><b>4. 双模型并发池性能甜点</b></summary>

- **实测表现**：采用 `deepseek-v4-flash` + `qwen3.6-flash` 双模型 Round-Robin：
  - 单模型 64 并发：吞吐 ~4 ops/s；
  - 双模型 128 并发：吞吐提升至 **~16 ops/s**（达到最佳能效平衡点）；
  - 并发升至 256：服务端频繁触发 429 限流重试，整体收益递减。
</details>

---

## 12. 项目目录结构

```
graph_rag/
├── Professional.json              # 默沙东专业版 30k 问答数据集
├── .env.example                   # 环境变量参考模板
├── docker-compose.yml             # Postgres + Neo4j + Qdrant 编排
├── requirements.txt               # 运行环境依赖库
├── README.md                      # 开源说明文档
│
├── src/
│   ├── config.py                  # Pydantic 集中配置管理
│   ├── clients/                   # 阿里云百炼异步客户端（限速/重试/日志）
│   │   ├── _base.py               # 基础 HTTP 连接池与限流器
│   │   ├── llm.py                 # 双模型轮询 Chat Client
│   │   ├── embedding.py           # 2560 维向量多模态路由客户端
│   │   └── rerank.py              # Cross-Encoder 原生重排客户端
│   ├── ingest/                    # 知识抽取流水线
│   │   ├── parse.py               # chunkId 层级解析
│   │   ├── storage.py             # PostgreSQL 表结构与批量入库
│   │   ├── prompts.py             # 实体/关系/摘要提示词模版
│   │   ├── entity_extract.py      # 9 类医学实体抽取与断点保存
│   │   ├── entity_dedup.py        # 实体同义别名归并算法
│   │   ├── relation_extract.py    # 12 类临床关系抽取与 APOC 推图
│   │   └── summary.py             # 子主题全局语义摘要
│   ├── graph/                     # Neo4j 拓扑图谱管理
│   ├── index/                     # Qdrant 向量库封装
│   ├── retrieval/                 # 在线多路召回与融合引擎
│   │   ├── query_understanding.py # 意图理解、改写与实体识别
│   │   ├── graph_recall.py        # 1-2 跳子图拓扑关联召回
│   │   ├── fusion.py              # 加权 RRF 算法实现
│   │   └── pipeline.py            # 端到端并行流水线编排
│   ├── generation/
│   │   └── answer.py              # 证据链回溯生成控制器
│   ├── eval/                      # 评测基准套件
│   └── api/                       # FastAPI 服务与可视化前端
│
├── scripts/                       # 编号离线执行脚本 (00 ~ 09)
│   ├── restore_prebuilt_data.py   # 一键从魔塔下载预构建向量与图数据库开箱即用
│   ├── upload_to_modelscope.py    # 同步问答及图谱抽取数据至魔塔社区
│   └── package_and_upload_prebuilt.py # 打包并上传全量预构建数据库包至魔塔
└── data/                          # 挂载卷与断点续跑日志 (cache)
```

---

## 13. 路线图与社区贡献

欢迎广大学者、工程师与临床医学专家共同参与本项目建设！

### 路线图 (Roadmap)
- [ ] **查询理解轻量化蒸馏**：将在线 LLM 改写蒸馏至端侧轻量小模型，将首阶段耗时从 3s 压减至 300ms 以内；
- [ ] **多轮对话与问诊会话流**：支持临床追问、症状引导澄清与历史会话状态维护；
- [ ] **前端图谱可视化增强**：将右侧 SVG 拓扑升级为基于 Cytoscape / D3 的动态可拖拽 3D 力导向关系网络；
- [ ] **流式输出 (SSE)**：API 与前端支持 Server-Sent Events 流式打字生成与引用同步高亮。

### 参与贡献
1. Fork 本项目仓库并新建分支（如 `feature/amazing-feature`）；
2. 提交您的修改并撰写清晰的 Commit Message；
3. 推送分支至您的远程仓库并提交 Pull Request。

---

## 14. 免责声明与开源协议

### ⚠️ 医学与法律免责声明 (Medical Disclaimer)
> 1. 本项目所包含的所有代码、数据及模型输出内容，**仅供计算机科学、自然语言处理及信息检索领域的学术研究与技术交流使用**。
> 2. 系统输出的内容**绝不构成任何临床诊断结论、个体化治疗方案或专业医疗处方建议**。
> 3. 任何个人或机构面对具体疾病状况，应及时前往正规医疗机构咨询具备执业资质的专业医师。本项目开发者不对任何依据本项目输出采取的诊疗决策及后果承担任何法律责任。

### 📄 开源协议 (License)
本项目遵循 [MIT License](LICENSE) 开源协议。数据来源于《默沙东诊疗手册》（专业版），请在遵循原版权方声明的前提下合规使用。
