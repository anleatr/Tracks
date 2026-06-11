#!/usr/bin/env python3
"""
学习脚本：演示 GraphRAG 如何加载文档（与 index workflow 中 load_input_documents 一致）。

用法（在项目根目录下执行）:
    python learn/load_documents_demo.py
    或
    cd learn && python load_documents_demo.py  # 会自动定位项目根目录

会从 settings.yaml 读取 input 配置，从 input 目录加载文档，并打印每条文档的结构。
"""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

# 确保项目根在 path 中（便于用 graphrag 的 load_config）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphrag.config.load_config import load_config
from graphrag_storage import create_storage
from graphrag_input import create_input_reader


async def load_documents_from_input():
    """从 input 存储加载文档，与 load_input_documents workflow 逻辑一致。"""
    # 1. 加载项目配置（使用项目根目录下的 settings.yaml）
    root_dir = PROJECT_ROOT
    config = load_config(root_dir=root_dir)

    # 2. 确保 input 目录基于项目根解析（从任意 cwd 运行脚本都能找到 input）
    base = getattr(config.input_storage, "base_dir", None)
    if base and not Path(base).is_absolute():
        config.input_storage.base_dir = str((PROJECT_ROOT / base).resolve())

    # 3. 创建输入存储（对应 settings 里 input.storage：type: file, base_dir: "input"）
    input_storage = create_storage(config.input_storage)

    # 4. 创建输入阅读器（根据 config.input 的 type：text/csv/json 等）
    input_reader = create_input_reader(config.input, input_storage)

    # 5. 逐条迭代文档，转成与 pipeline 中一致的「行」格式
    rows = []
    async for doc in input_reader:
        row = asdict(doc)
        row["human_readable_id"] = len(rows)
        if "raw_data" not in row:
            row["raw_data"] = None
        rows.append(row)

    return rows, config


def main():
    print("=" * 60)
    print("GraphRAG 文档加载示例（learn/load_documents_demo.py）")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print()

    rows, config = asyncio.run(load_documents_from_input())
    print(f"输入配置: type={config.input.type}, storage base_dir={config.input_storage.base_dir}")
    print()

    if not rows:
        print("未加载到任何文档。请确认 input 目录下有待读取文件（如 .txt），且 settings 中 input 配置正确。")
        return

    print(f"共加载 {len(rows)} 条文档。")
    print()
    print("每条文档在 pipeline 中写入 'documents' 表时的列：")
    print(list(rows[0].keys()))
    print()
    print("-" * 60)
    for i, row in enumerate(rows):
        print(f"文档 {i} (human_readable_id={row['human_readable_id']})")
        print(f"  id: {row['id'][:64]}..." if len(row.get("id", "")) > 64 else f"  id: {row.get('id')}")
        print(f"  title: {row.get('title')}")
        print(f"  creation_date: {row.get('creation_date')}")
        text_preview = (row.get("text") or "")[:200]
        print(f"  text: {text_preview}...")
        print(f"  raw_data: {row.get('raw_data')}")
        print("-" * 60)

    # 可选：将前几条保存到 learn 目录便于查看
    out_file = PROJECT_ROOT / "learn" / "loaded_documents_sample.json"
    sample = [rows[0]] if rows else []
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"已将第一条文档的完整结构写入: {out_file}")


if __name__ == "__main__":
    main()
