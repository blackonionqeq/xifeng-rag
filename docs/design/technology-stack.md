# 技术栈设计

状态：技术栈基线 v0.2。

## 1. 总体选择

RAG 核心、索引、模型调用、评测与 API 使用 Python。前端延后开发，届时使用 Svelte，不在第一阶段引入前端工程。

分阶段交付：

```text
阶段 1：Python 核心 + CLI + 自动化测试 + 评测
阶段 2：FastAPI
阶段 3：Svelte 前端
```

## 2. 框架策略

采用“自有核心 + 底层库”而非 LlamaIndex-first 或 LangChain-first：

- 自己维护领域数据模型、Index Profile、增量更新、RRF、引用和实验记录；
- 使用成熟库完成向量存储、Embedding、Reranker 和模型 API 调用；
- LlamaIndex 可用于参考实现、辅助评测或后续对照实验，但不作为主流水线的领域模型。

这样可以保持数据与引用规则可控，也能在项目展示中说明每个 RAG 阶段的真实实现。

## 3. Python 栈

| 职责 | 选择 |
|---|---|
| Python | Python 3.11 |
| 环境与依赖 | `uv` + `pyproject.toml` |
| 配置与边界模型 | Pydantic v2 + pydantic-settings |
| 结构化状态 | SQLite + SQLAlchemy 2 + Alembic |
| 向量存储 | Qdrant Client 本地持久化模式 |
| 关键词检索 | `bm25s` + 中文分词 |
| 离线语义分块 | Ollama standalone + `qwen3:4b-instruct-2507-q4_K_M` |
| Embedding | 默认 `Qwen/Qwen3-Embedding-0.6B`；基线 `BAAI/bge-base-zh-v1.5` |
| Reranker | 默认 `BAAI/bge-reranker-v2-m3`；后续对照 `Qwen/Qwen3-Reranker-0.6B` |
| CLI | Typer |
| API | FastAPI，阶段 2 加入 |
| 测试与质量 | pytest + Ruff + Pyright |
| 实验分析 | 自定义指标，按需使用 Pandas/Jupyter |

Qdrant 第一阶段使用进程内本地持久化模式，不要求启动独立服务。后续需要并发访问或独立部署时，可以迁移到 Qdrant Server，同时保留相同的客户端边界。

BM25 与 Dense 检索保持逻辑独立，通过自有 RRF 实现融合，便于保存各路原始排名和完成消融实验。

## 4. Generator

第一阶段只实现 DeepSeek API 适配器：

```text
provider: deepseek
base_url: https://api.deepseek.com
model: deepseek-v4-flash
```

DeepSeek API 使用 OpenAI 兼容的 Chat Completions 形式。API 密钥只从环境变量或未提交的本地密钥配置读取。

Generator 接口仍与具体服务商解耦。外部调用只用于交互式答案生成，不用于离线语义分块或批量 Contextual Retrieval。

第一阶段使用 `qwen3:4b-instruct-2507-q4_K_M` 完成离线语义分块和概括，实际上下文先设为 32K，并通过 Ollama JSON Schema 输出与 Pydantic 校验保证结构。它属于索引预处理组件，不作为本地 Generator 对外提供问答服务。

Windows 使用官方 Ollama standalone CLI，不安装托盘应用、不注册开机启动。程序放在 `D:\ai-tools\ollama`，模型通过 `OLLAMA_MODELS` 指向 `D:\ai-models\ollama`。Ollama 服务只在离线预处理时按需启动。模型或运行后端只有在本机长文验收不达标时才替换。

## 5. 前端

前端不属于第一阶段范围。核心与 API 稳定后使用 Svelte；是否需要 SvelteKit、路由框架或 UI 组件库，到阶段 3 根据实际页面需求再决定。

FastAPI 的响应模型应保持与前端框架无关，支持返回答案、引用、检索阶段分数和隐私提示。流式生成可以在阶段 2 设计，不提前影响阶段 1 的核心接口。

## 6. 暂不引入

第一阶段不引入：

- LangChain 或 LlamaIndex 作为全局编排框架；
- 通用插件注册系统；
- 独立 Qdrant 服务、PostgreSQL、Redis 或消息队列；
- 面向交互式问答的 Ollama、LM Studio 或其他本地 Generator 服务；离线分块使用的按需 Ollama 进程不属于此项；
- 前端构建链；
- GraphRAG 基础设施。

只有在已实现功能出现明确需求时，才增加对应依赖。
