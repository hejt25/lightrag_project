# LightRAG — Copilot 指导说明

此文件为 AI 编码代理（Copilot / agents）在本仓库中快速上手的操作与约定说明。内容基于可发现的代码与测试；请在需要时向维护者确认运行时服务（例如本地 LLM 服务）是否可用。

要点速览
- 架构分层：`ingestion` → `embedding` → `storage` → `retrieval` → `generation` → `monitoring`。
- 关键入口：`src/run.py`（CLI）、`src/pipeline.py`（主流程类 `LightRAGPipeline`）、`src/config.py`（配置模型）。
- 向量存储默认使用 FAISS（`src/storage`），embedding 默认使用 `BAAI/bge-small-zh`（`src/embedding`）。
- LLM 默认配置在 `LightRAGConfig` 中（`src/config.py`）—`api_base` 指向本地/远端模型服务（默认 `http://localhost:8000/v1`）。

工作流与常用命令
- 安装依赖：
```bash
pip install -r requirements.txt
```
- 运行单元测试（在仓库根目录）：
```bash
pytest -q
```
- 以 CLI 使用（构建索引 / 查询 / 导出监控数据）：
```bash
python -m src.run --index --data ./data
python -m src.run --query "你的问题"
python -m src.run --export ./monitoring_out
```
- 编程方式快速使用：
```py
from src import EasyLightRAG
rag = EasyLightRAG(data_dir="./data", llm_api_base="http://localhost:8000/v1")
rag.build_index()
print(rag.query("什么是人工智能？"))
```

项目约定（可被 AI 直接利用）
- 配置集中：所有运行时参数在 `LightRAGConfig`（`src/config.py`），AI 代理应通过 `LightRAGConfig.from_json(path)` 或构造参数明确设置运行时行为。
- 数据目录：默认 `./data/raw`（`LightRAGConfig.data_path`），索引文件路径在 `vector_store.index_path`（默认 `./data/vector_store`）。修改配置而非硬编码路径。
- 组件初始化按固定顺序（见 `LightRAGPipeline._init_components`），任何补丁需保留该顺序以保证 tracing/metrics 正常记录。
- 监控：`Monitor` 与 `PerformanceMonitor` 会在 pipeline 中被使用，追踪 span 名称（如 `ingestion`, `embedding`, `storage`, `retrieval`, `generation`）是稳定的约定。

要点示例（从代码中抽取）
- 索引过程：在 `LightRAGPipeline.index()` 中，步骤为 `ingestion` → `embedding` → `storage`，并调用 `vector_store.add()` 与 `vector_store_manager.save_index()`。
- 查询过程：`LightRAGPipeline.query()` 先走 `RetrievalPipeline.retrieve()`，再走 `GenerationPipeline.generate()`；返回结构包含 `answer`, `metrics`, `contexts`。

集成点与外部依赖
- 本地 embedding/LLM：代码假定可访问模型（embedding via `sentence-transformers`；LLM via HTTP `api_base`）。确认本地模型或远端 API 可用。
- 向量存储：默认 FAISS（`faiss-cpu`），若替换成 Milvus/Qdrant，应实现相同的 `VectorStore` 接口（参见 `src/storage`）。

给 AI 代理的具体任务指南
- 当修改 pipeline 行为：优先在 `src/pipeline.py` 中查找对应阶段，保留 `monitor.span(...)` 包裹以保证监控记录。
- 若需要新增配置项：在 `src/config.py` 的 `LightRAGConfig` 中添加字段并更新 `to_dict()`/`save()` 方法。
- 写测试：参考 `tests/test_pipeline.py` 的风格 —— 测试应从根目录运行，测试会将 `src` 加入 sys.path。

若发现不一致或需要运行时权限/凭证：在 PR 描述里说明需要外部服务（例如 LLM endpoint），并给出最小复现命令。

快速检查点（PR 模板用）
- 功能变更是否修改 `LightRAGPipeline._init_components` 或 `Monitor` 使用？若是，列出影响的 span 名称与数据导出位置。
- 是否需要新增/更新 `requirements.txt`？在 PR 中同时更新并说明原因。

反馈
- 我已基于仓库可见代码与测试生成这些说明；如需我把文中某部分改为英语、补充运行示例或合并到现有文档，请指示。
