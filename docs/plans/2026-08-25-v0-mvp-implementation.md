# V0 MVP 实施计划

对应设计：[`docs/specs/2026-08-25-v0-mvp-design.md`](../specs/2026-08-25-v0-mvp-design.md)

## 交付顺序

1. 安装并验证 Windows standalone Ollama，通过本地配置指定模型目录，拉取 `qwen3:4b-instruct-2507-q4_K_M`。
2. 建立 Python 3.11、uv、Typer、Pydantic v2、SQLAlchemy 2、Alembic、bm25s、中文分词、pytest、Ruff 与 Pyright 工程。
3. 实现配置、数据库模型、事务、领域类型、稳定 ID 和 CLI 基础命令。
4. 实现顶层 Markdown 扫描、块级解析、链接规范化、保守尾部清理、位置映射和增量分类。
5. 实现 Ollama JSON Schema 分块、覆盖校验、有限重试、审核导出、清洗裁决与批准工作流。
6. 实现原文 BM25、不可变索引构建、原子 active 切换、统一候选与原文引用。
7. 实现 JSON 评测集及 P@K、Recall@K、MRR、nDCG、延迟记录。
8. 使用自造夹具完成单元及集成测试，验证隐私默认值与失败回滚。
9. 更新 README 和技术文档，运行 Ruff、Pyright、pytest 和 CLI smoke test。
10. 从真实语料中人工选择少量样本完成本机 Ollama 验收，记录结果，不批处理全部文章。

## 完成门槛

- 所有设计中的 V0 CLI 命令可执行并提供明确错误；
- 未批准版本无法进入索引；
- 新增、修改、未变化、缺失和失败回滚均有自动测试；
- 默认执行不产生外部正文请求，日志不包含完整正文；
- BM25 查询返回可追溯到规范正文段落的引用；
- 自动化质量检查通过，真实样本验收结果有记录。
