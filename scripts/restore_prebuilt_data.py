# -*- coding: utf-8 -*-
"""
一键从魔塔社区 (ModelScope) 下载并恢复预构建数据库数据
使用者无需重新调用大模型进行向量化与知识抽取，开箱即用：
  - 自动下载 qdrant_storage.tar.gz (已嵌入向量索引) 并解压到 data/qdrant
  - 自动下载 neo4j_data.tar.gz (已构建知识图谱) 并解压到 data/neo4j
  - 自动下载 postgres_data.tar.gz (已入库关系数据) 并解压到 data/postgres
"""

import sys
import tarfile
import subprocess
from pathlib import Path

DATASET_ID = "lzx1770670957/medical-graph-rag-qa"

def check_dependencies():
    """检查依赖库是否就绪"""
    try:
        import modelscope
    except ImportError:
        print("未检测到 modelscope SDK，请先执行安装：")
        print("  pip install modelscope")
        sys.exit(1)


def download_prebuilt_packages(cache_dir: Path):
    """从魔塔社区下载预构建数据包"""
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"正在从魔塔社区拉取预构建数据包: {DATASET_ID} ...")
    print("包含全量已 Embedding 向量索引库与知识图谱拓扑数据，无需重新计算。")
    download_dir = snapshot_download(
        dataset_id=DATASET_ID,
        cache_dir=str(cache_dir)
    )
    return Path(download_dir)


def extract_archive(tar_path: Path, target_dir: Path):
    """跨平台解压缩数据包到目标目录"""
    print(f"正在解压: {tar_path.name} -> {target_dir} ...")
    try:
        # 优先使用系统内置高效 tar 命令
        cmd = ["tar", "-xzf", str(tar_path.resolve()), "-C", str(target_dir.resolve())]
        subprocess.run(cmd, check=True)
    except Exception:
        # 兼容回退到 Python 原生 tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(target_dir)
    print(f"解压完成: {tar_path.name}")


def main():
    check_dependencies()

    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "download_cache"

    print("=" * 60)
    print(" 步骤一：从魔塔社区下载预构建数据包")
    print("=" * 60)

    dataset_path = download_prebuilt_packages(cache_dir)

    archives = [
        "postgres_data.tar.gz",
        "neo4j_data.tar.gz",
        "qdrant_storage.tar.gz",
    ]

    print("\n" + "=" * 60)
    print(" 步骤二：解压并部署预构建数据库")
    print("=" * 60)

    for arch_name in archives:
        arch_file = dataset_path / arch_name
        if arch_file.exists():
            extract_archive(arch_file, data_dir)
        else:
            print(f"提示: 未在下载目录中找到 {arch_name}，跳过此项。")

    print("\n" + "=" * 60)
    print(" 恭喜！预构建向量与图数据库恢复完毕！")
    print("=" * 60)
    print("接下来仅需两步即可开始使用：")
    print("  1. 启动基础设施: docker compose up -d")
    print("  2. 启动服务: uvicorn src.api.server:app --port 8000")
    print("浏览器访问: http://127.0.0.1:8000 开始医疗问答体验！")


if __name__ == "__main__":
    main()
