# 实施计划：开箱即用（免重新向量化）全量数据开源方案

## 1. 现状评估
1. **此前已上传魔塔社区的数据**：
   - `Professional.json` (30,031 条原始问答文本)
   - `entities.jsonl` (39,400 个抽取的医学实体)
   - `relations.jsonl` (23,591 条临床关系三元组)
   - `testset.jsonl` (480 条评测盲测集)
   - `eval_results.jsonl` (评测结果)
   - **结论**：当前魔塔仓库仅包含**文本和抽取产物**，**尚未包含已 Embedding 的向量数据（Qdrant 索引库文件）以及图数据库数据（Neo4j/Postgres 实例文件）**。

2. **如果只下载当前数据**：
   使用者需要重新运行：
   - `01_load_data.py` (数据入库 PG)
   - `02_build_vectors.py` (向阿里百炼发起 60,062 次 Embedding 请求，生成 Qdrant 向量，耗时约 28 分钟并消耗大量 API 配额)
   - `03_build_graph_skeleton.py` (Neo4j 骨架)
   - `05_extract_relations.py` (Neo4j 关系推图)
   - `06_build_topic_summaries.py` (生成子主题摘要与 Topic 向量)

## 2. “开箱即用·免重新向量化”的解决方案
为了让使用者“下载数据，配置好 DashScope API Key 后直接就能用，0 消耗 API 配额重新做向量化与图抽取”，需要将底层引擎的预构建数据持久化包上传至魔塔社区：

### 打包规格
- **`qdrant_storage.tar.gz`**（由 `data/qdrant` 压缩而来，包含 `medgraph_q`、`medgraph_a`、`medgraph_topic` 3 个集合，原始大小约 954MB，压缩后预计约 400-500MB）；
- **`neo4j_data.tar.gz`**（由 `data/neo4j` 压缩而来，包含 7.8 万节点与 23.5 万条边，原始大小约 624MB，压缩后预计约 100-150MB）；
- **`postgres_data.tar.gz`**（由 `data/postgres` 压缩而来，包含全量表结构与数据，原始大小约 300MB，压缩后预计约 40MB）。

### 使用者体验
下载后执行一键解压脚本：
```bash
tar -xzf qdrant_storage.tar.gz -C ./data/
tar -xzf neo4j_data.tar.gz -C ./data/
tar -xzf postgres_data.tar.gz -C ./data/
docker compose up -d
```
在 `.env` 中填入自己的百炼 `API_KEY`（仅用于在线提问时的意图理解、重排和生成），直接启动 `uvicorn src.api.server:app` 即可秒级开启医疗 GraphRAG 问答服务，完全无需重新向量化！
