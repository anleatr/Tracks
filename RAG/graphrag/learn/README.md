# Learn — GraphRAG 学习示例

本目录用于单独运行、理解 GraphRAG 各环节的逻辑，不依赖完整 pipeline。

## load_documents_demo.py

**作用**：演示「加载文档」步骤，与 index 流程中的 `load_input_documents` workflow 行为一致。

**逻辑简述**：

1. 从项目根目录的 `settings.yaml` 读取 `input`、`input_storage` 配置
2. 用 `create_storage(config.input_storage)` 创建输入存储（默认即 `input/` 目录）
3. 用 `create_input_reader(config.input, input_storage)` 创建阅读器（按 `input.type`：text / csv / json 等）
4. `async for doc in input_reader` 逐条得到 `TextDocument`，转成带 `human_readable_id` 的行（与写入 `documents` 表的格式一致）
5. 在终端打印每条文档的字段，并把第一条的完整结构写入 `learn/loaded_documents_sample.json`

**运行方式**（在项目根目录下）：

```bash
# 使用项目 venv
.venv/bin/python learn/load_documents_demo.py

# 或先激活环境
source .venv/bin/activate  # 或 conda activate graphrag
python learn/load_documents_demo.py
```

**前置条件**：

- `settings.yaml` 中 `input.storage.base_dir` 指向的目录存在（默认 `input/`）
- 该目录下有可被当前类型匹配的文件（如 `input.type: text` 时需有 `.txt` 文件）

脚本会基于项目根目录解析 `input` 路径，从任意当前工作目录执行均可。
