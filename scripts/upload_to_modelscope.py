# -*- coding: utf-8 -*-
"""
魔塔社区（ModelScope）数据集上传与同步脚本
将默沙东医疗问答数据及图谱抽取衍生数据发布至魔塔社区。
凭证安全：从环境变量或 .env 读取 MODELSCOPE_API_TOKEN，严禁硬编码。
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from modelscope.hub.api import HubApi
from modelscope_hub.constants import Visibility, RepoType

# 加载环境变量
load_dotenv()

TOKEN = os.getenv("MODELSCOPE_API_TOKEN")
if not TOKEN:
    raise ValueError("未检测到 MODELSCOPE_API_TOKEN，请在 .env 文件或系统环境变量中配置。")

DATASET_NAME = "medical-graph-rag-qa"
CHINESE_NAME = "默沙东临床医疗问答与图谱抽取数据集 (GraphRAG)"
DESCRIPTION = "基于默沙东诊疗手册（专业版）提取的 30,031 条结构化临床医学问答，以及 3.9 万去重医学实体与 2.3 万临床拓扑关系，专为中文医疗 GraphRAG 系统构建。"

def build_dataset_readme(target_dir: Path, repo_id: str):
    """
    生成魔塔社区专用的 Dataset Card (README.md)
    """
    readme_content = f"""---
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

# {CHINESE_NAME}

## 1. 数据集简介
本数据集以权威的《默沙东诊疗手册》（Merck Manuals 专业版）为基础知识源，经过医学专业清洗、语义分块与三层层级标注，形成 **30,031 条高质量临床医学问答对**，并衍生抽取了 **39,400 个去重医学实体** 与 **23,591 条临床关系三元组**。

该数据集是构建临床辅助决策、医疗知识图谱、医疗问答系统以及 **GraphRAG（图检索增强生成）** 的高质量基准语料。

## 2. 数据目录与文件构成
本数据集仓库包含以下核心数据工件：

| 文件名 | 数据规模 | 格式 | 说明 |
| :--- | :--- | :--- | :--- |
| **`Professional.json`** | 30,031 条 (27MB) | JSON Array | 核心临床问答库，每条包含问题、专业回答与三层结构标签 (`chunkId`) |
| **`entities.jsonl`** | 39,400 实体 (25MB) | JSONL | 从问答中提取的 9 类去重医学实体（疾病、药物、症状、检验等），含别名与描述 |
| **`relations.jsonl`** | 23,591 关系 (11MB) | JSONL | 抽取的 12 类临床关系三元组（TREATS, CAUSES, DIAGNOSED_BY, CONTRAINDICATED_FOR 等） |
| **`testset.jsonl`** | 480 条 (491KB) | JSONL | 覆盖 24 个临床专科的分层抽样盲测基准测试集 |
| **`eval_results.jsonl`** | 480 条评测记录 (36KB) | JSONL | GraphRAG 系统在测试集上的检索重排详细得分与命中情况 |

## 3. 数据格式样例

### 临床问答样例 (`Professional.json`)
```json
[
  {{
    "question": "什么是受体在细胞中的主要功能？",
    "answer": "受体在细胞中的主要功能是参与和介导化学信号的传递。这些受体位于细胞表面膜上或细胞质内，它们通过与特定的配体结合，直接或间接地调控细胞的生化过程...",
    "chunkId": "临床药理学 - 药物-受体相互作用 - part-1"
  }}
]
```

### 实体抽取样例 (`entities.jsonl`)
```json
{{
  "id": "e_01f92a",
  "name": "妊娠期糖尿病",
  "type": "Disease",
  "aliases": ["GDM", "妊娠糖尿病"],
  "description": "妊娠期间首次发生或发现的不同程度的糖耐量异常。"
}}
```

### 关系三元组样例 (`relations.jsonl`)
```json
{{
  "rel_id": "r_921a",
  "head": "二甲双胍",
  "rel_type": "TREATS",
  "tail": "2型糖尿病",
  "evidence_chunk": "内分泌与代谢疾病 - 糖尿病治疗 - part-2",
  "confidence": 0.95
}}
```

## 4. 快速下载与使用

### Python SDK 下载
```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载完整数据集到本地
dataset_dir = snapshot_download(
    dataset_id='{repo_id}',
    cache_dir='./data/modelscope_dataset'
)
print("数据集已下载至:", dataset_dir)
```

### 命令行快速拉取
```bash
modelscope download --dataset {repo_id} --local_dir ./data/modelscope_dataset
```

## 5. 开源协议与使用建议
- 原始数据来源于公开的临床参考手册知识，仅供自然语言处理、信息检索与大模型学术研究使用。
- 绝不构成任何临床个体化诊疗建议或处方依据。
"""
    readme_path = target_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")


def prepare_staging_directory(staging_dir: Path, repo_id: str):
    """
    整理并准备待上传的文件
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(__file__).resolve().parent.parent

    # 1. 复制 Professional.json
    prof_json = root_dir / "Professional.json"
    if prof_json.exists():
        print(f"正在准备: {prof_json.name} ({prof_json.stat().st_size / 1024 / 1024:.2f} MB)")
        shutil.copy2(prof_json, staging_dir / "Professional.json")
    else:
        print("警告: 未在根目录找到 Professional.json")

    # 2. 复制 data/cache 下的结构化文件
    cache_files = ["entities.jsonl", "relations.jsonl", "testset.jsonl", "eval_results.jsonl"]
    cache_dir = root_dir / "data" / "cache"
    for fname in cache_files:
        fpath = cache_dir / fname
        if fpath.exists():
            print(f"正在准备: {fname} ({fpath.stat().st_size / 1024 / 1024:.2f} MB)")
            shutil.copy2(fpath, staging_dir / fname)
        else:
            print(f"提示: 未在 data/cache 找到 {fname}，已跳过")

    # 3. 写入魔塔社区专用 README
    build_dataset_readme(staging_dir, repo_id)
    print("魔塔社区数据集卡片 README.md 构建完成")


def main():
    api = HubApi()
    print("正在使用 Token 登录魔塔社区 (ModelScope)...")
    api.login(TOKEN)

    # 获取当前用户信息
    whoami = api._api.whoami()
    owner = whoami.username
    repo_id = f"{owner}/{DATASET_NAME}"
    print(f"当前认证用户: {owner} ({whoami.email})")
    print(f"目标数据集仓库: {repo_id}")

    # 检查仓库是否存在，若不存在则创建公开数据集
    existing_repos = api._api.list_repos(owner=owner, repo_type="dataset")
    repo_names = [r.name for r in existing_repos.items]
    if DATASET_NAME not in repo_names:
        print(f"数据集仓库 {repo_id} 不存在，正在创建公开数据集...")
        try:
            api._api.create_repo(
                repo_id=repo_id,
                repo_type=RepoType.DATASET,
                visibility=Visibility.PUBLIC,
                license="Apache License 2.0",
                chinese_name=CHINESE_NAME,
            )
            print("数据集仓库创建成功！")
        except Exception as e:
            print(f"创建数据集时提示: {e}，尝试继续上传...")
    else:
        print(f"数据集仓库 {repo_id} 已存在，准备上传更新文件...")

    # 准备待上传的目录
    root_dir = Path(__file__).resolve().parent.parent
    staging_dir = root_dir / "data" / "staging_modelscope"
    prepare_staging_directory(staging_dir, repo_id)

    # 执行文件夹上传
    print(f"开始上传文件到魔塔社区数据集仓库: {repo_id} ...")
    api._api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging_dir),
        repo_type="dataset",
        commit_message="上传默沙东医疗问答与图谱抽取核心数据集 (GraphRAG)"
    )

    dataset_url = f"https://www.modelscope.cn/datasets/{repo_id}/summary"
    print("\n" + "=" * 60)
    print(" 上传成功！")
    print(f"数据集主页地址: {dataset_url}")
    print("=" * 60)

    # 清理临时暂存目录
    try:
        shutil.rmtree(staging_dir)
        print("暂存目录已清理")
    except Exception as e:
        print(f"清理暂存目录异常: {e}")

    return repo_id, dataset_url


if __name__ == "__main__":
    main()
