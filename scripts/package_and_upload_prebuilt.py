# -*- coding: utf-8 -*-
"""
打包并上传预构建向量数据库与图数据库至魔塔社区 (ModelScope)
包括：
  1. data/qdrant   -> qdrant_storage.tar.gz   (全量已嵌入向量索引)
  2. data/neo4j    -> neo4j_data.tar.gz       (全量知识图谱拓扑数据)
  3. data/postgres -> postgres_data.tar.gz    (全量结构化关系数据库)
让使用者无需重新向量化，下载解压即可秒级开箱即用。
"""

import os
import sys
import time
import shutil
import tarfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from modelscope.hub.api import HubApi

# 加载环境配置
load_dotenv()

TOKEN = os.getenv("MODELSCOPE_API_TOKEN")
if not TOKEN:
    raise ValueError("未检测到 MODELSCOPE_API_TOKEN，请在 .env 中配置。")

DATASET_NAME = "medical-graph-rag-qa"

def compress_directory(src_dir: Path, target_tar_gz: Path, base_name: str):
    """
    压缩指定数据目录为 tar.gz 文件，优先使用系统内置 tar，支持跨平台回退
    """
    if target_tar_gz.exists():
        print(f"检测到已存在的压缩包: {target_tar_gz.name}，正在复用或准备上传...")
        return target_tar_gz

    print(f"正在压缩打包: {src_dir.name} -> {target_tar_gz.name} ...")
    t0 = time.time()
    
    # 尝试使用高效的系统 tar 工具
    try:
        # cd data && tar -czf target_tar_gz base_name
        data_parent = src_dir.parent
        cmd = ["tar", "-czf", str(target_tar_gz.resolve()), base_name]
        ret = subprocess.run(cmd, cwd=str(data_parent), check=True)
    except Exception as e:
        print(f"系统 tar 压缩未生效或失败: {e}，改用 Python tarfile 压缩...")
        with tarfile.open(target_tar_gz, "w:gz") as tar:
            tar.add(src_dir, arcname=base_name)

    elapsed = time.time() - t0
    size_mb = target_tar_gz.stat().st_size / 1024 / 1024
    print(f"打包完成: {target_tar_gz.name} (大小: {size_mb:.2f} MB, 耗时: {elapsed:.1f} 秒)")
    return target_tar_gz


def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    staging_dir = data_dir / "staging_packages"
    staging_dir.mkdir(parents=True, exist_ok=True)

    packages_to_build = [
        # (源目录, 压缩包名称, 解压相对目录名)
        (data_dir / "postgres", staging_dir / "postgres_data.tar.gz", "postgres"),
        (data_dir / "neo4j", staging_dir / "neo4j_data.tar.gz", "neo4j"),
        (data_dir / "qdrant", staging_dir / "qdrant_storage.tar.gz", "qdrant"),
    ]

    print("=" * 60)
    print(" 步骤一：开始打包预构建数据库持久化卷")
    print("=" * 60)

    built_archives = []
    for src, target_archive, arcname in packages_to_build:
        if not src.exists():
            print(f"错误: 源目录不存在 {src}")
            sys.exit(1)
        archive_path = compress_directory(src, target_archive, arcname)
        built_archives.append((archive_path, target_archive.name))

    print("\n" + "=" * 60)
    print(" 步骤二：登录魔塔社区并开始上传数据包")
    print("=" * 60)

    api = HubApi()
    api.login(TOKEN)
    whoami = api._api.whoami()
    owner = whoami.username
    repo_id = f"{owner}/{DATASET_NAME}"
    print(f"目标仓库: {repo_id}")

    for archive_path, filename in built_archives:
        size_mb = archive_path.stat().st_size / 1024 / 1024
        print(f"\n开始上传: {filename} ({size_mb:.2f} MB) 至魔塔数据集 {repo_id} ...")
        t0 = time.time()
        api._api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(archive_path),
            path_in_repo=filename,
            commit_message=f"上传预构建数据库包: {filename} ({size_mb:.1f}MB)"
        )
        elapsed = time.time() - t0
        print(f"上传成功: {filename} (耗时: {elapsed:.1f} 秒)")

    print("\n" + "=" * 60)
    print(" 步骤三：更新魔塔社区数据集卡片 README.md")
    print("=" * 60)

    dataset_readme_content = f"""---
license: Apache License 2.0
task_categories:
- question-answering
- text-retrieval
language:
- zh
tags:
- medical
- graphrag
- knowledge-graph
- question-answering
size_categories:
- 10K<n<100K
---

# 默沙东临床医疗问答与图谱抽取数据集 (GraphRAG)

## 1. 数据集简介
本数据集以权威的《默沙东诊疗手册》（Merck Manuals 专业版）为基础知识源，经过医学专业清洗、语义分块与三层层级标注，形成 **30,031 条高质量临床医学问答对**，并衍生抽取了 **39,400 个去重医学实体** 与 **23,591 条临床关系三元组**。

为了最大化降低复现门槛，本项目在魔塔社区**完整开源了已完成 2560 维高维向量嵌入的 Qdrant 向量库、Neo4j 知识图谱与 PostgreSQL 数据库预构建数据包**。使用者无需花费大量 API 额度和时间重新计算向量与图谱，直接下载解压即可秒级启动 GraphRAG 问答服务！

## 2. 数据目录与文件构成

| 文件名 | 类型 | 数据规模 | 说明 |
| :--- | :--- | :--- | :--- |
| **`qdrant_storage.tar.gz`** | 预构建向量库 | 约 420 MB (解压约 954MB) | **已完成 2560 维向量化的 Qdrant 数据库**，含 `medgraph_q`、`medgraph_a`、`medgraph_topic` 三大集合全部 62,595 个向量与索引 |
| **`neo4j_data.tar.gz`** | 预构建图数据库 | 约 120 MB (解压约 624MB) | **已构建完成的 Neo4j 知识图谱**，含 78,080 个医学实体与结构节点，235,909 条临床拓扑关系边 |
| **`postgres_data.tar.gz`** | 预构建关系库 | 约 40 MB (解压约 300MB) | **已入库的 PostgreSQL 数据卷**，含所有科室、主题、分块、问答对、实体与关系表结构 |
| **`Professional.json`** | 原始问答文本 | 27 MB | 30,031 条带三层层级标签的标准结构化临床问答 |
| **`entities.jsonl`** | 实体抽取结果 | 25 MB | 抽取的 39,400 个去重医学实体与同义别名 |
| **`relations.jsonl`** | 关系抽取结果 | 11 MB | 抽取的 23,591 条临床关系三元组 |
| **`testset.jsonl`** | 评估盲测集 | 491 KB | 24 专科分层抽样的 480 条基准评估测试集 |
| **`eval_results.jsonl`** | 评测详细得分 | 36 KB | 包含 Recall@8 与 MRR@8 的每项盲测评分明细 |

## 3. 极速开箱即用（免重新向量化）

使用者无需运行消耗巨额 API 配额的 `02_build_vectors.py` 向量化流水线，只需下载预构建数据库包：

```bash
# 1. 下载预构建数据库包
pip install modelscope
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('{repo_id}', cache_dir='./download_cache')"

# 2. 将数据包解压到项目的 data 目录
tar -xzf download_cache/{repo_id}/qdrant_storage.tar.gz -C ./data/
tar -xzf download_cache/{repo_id}/neo4j_data.tar.gz -C ./data/
tar -xzf download_cache/{repo_id}/postgres_data.tar.gz -C ./data/

# 3. 启动基础设施
docker compose up -d

# 4. 配置 .env 阿里百炼 API_KEY 后，直接启动 API 与前端交互界面
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```
"""
    readme_path = staging_dir / "README.md"
    readme_path.write_text(dataset_readme_content, encoding="utf-8")
    api._api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        commit_message="更新数据集卡片：增加全量预构建向量与图数据库开箱即用说明"
    )
    print("魔塔社区数据集卡片更新完成！")

    print("\n" + "=" * 60)
    print(" 步骤四：清理本地临时打包文件")
    print("=" * 60)
    try:
        shutil.rmtree(staging_dir)
        print("临时暂存目录已清理完毕。")
    except Exception as e:
        print(f"清理暂存目录时提示: {e}")

    print("\n" + "=" * 60)
    print("全量预构建数据包（Qdrant/Neo4j/Postgres）已全部成功发布至魔塔社区！")
    print(f"仓库主页: https://www.modelscope.cn/datasets/{repo_id}/summary")
    print("=" * 60)


if __name__ == "__main__":
    main()
