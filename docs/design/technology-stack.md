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
| 配置与边界模型 | Pydantic v2 + pydantic-settings + python-dotenv |
| 结构化状态 | SQLite + SQLAlchemy 2 + Alembic |
| 向量存储 | Qdrant Client 本地持久化模式 |
| 关键词检索 | `bm25s` + `jieba` search mode（版本化封装） |
| 离线语义分块 | Ollama standalone + `qwen3:4b-instruct-2507-q4_K_M` |
| Embedding | 默认 `Qwen/Qwen3-Embedding-0.6B`；基线 `BAAI/bge-base-zh-v1.5` |
| Reranker | 默认 `BAAI/bge-reranker-v2-m3`；后续对照 `Qwen/Qwen3-Reranker-0.6B` |
| 上下文组装 | 自有确定性 Python 组件 + 规范 JSON |
| CLI | Typer |
| 本地管理、分块审核与评测策展页 | Python 标准库 HTTP 服务 + 原生 HTML/CSS/JavaScript |
| API | FastAPI，阶段 2 加入 |
| 测试与质量 | pytest + Ruff + Pyright |
| 实验分析 | 自定义指标，按需使用 Pandas/Jupyter |

Qdrant 第一阶段使用进程内本地持久化模式，不要求启动独立服务。后续需要并发访问或独立部署时，可以迁移到 Qdrant Server，同时保留相同的客户端边界。

BM25 与 Dense 检索保持逻辑独立，通过自有 RRF 实现融合，便于保存各路原始排名和完成消融实验。

Windows 本机推理通过 uv 的平台化 source 使用 PyTorch 官方 CUDA 13.0 wheel；其他平台继续
从默认 Python 索引解析 Torch。当前开发机 NVIDIA 驱动报告兼容 CUDA 13.1，因此选择 cu130。
PyTorch wheel 自带运行所需 CUDA 库；项目不编译 CUDA 扩展，不要求单独安装 CUDA Toolkit。

V1 基线使用 `sentence-transformers` 加载 `BAAI/bge-base-zh-v1.5`，文档与查询向量归一化后
写入每个 Index Build 独立的 Qdrant local storage。查询支持 BM25、Dense 与 Hybrid 三种
显式模式；当前 34 块 pilot 语料的 Hybrid 默认使用两路 Top-5 和 `rrf_k=60`，不直接混合
两路原始分数。候选深度来自 dev 集参数实验，语料和评测集扩充后必须重新验证。

V0 已采用 `jieba.Tokenizer` 的独立实例完成本地 search-mode 分词，封装版本为
`jieba-search-v1`。该选择是可替换的基线，后续分词器对照必须创建新的 Index Profile，
不得覆盖已有实验索引。

上下文组装不引入额外框架或 tokenizer 依赖。首版以完整 TopicChunk 数量和最终 JSON 字符数
控制范围，默认最多 5 块、50,000 字符；这是一致、供应商无关的异常保护。未来 Generator
适配器再负责供应商请求级 token 限制，不能用某个不匹配的本地 tokenizer 冒充精确计数。

## 4. Generator

第一阶段只实现 DeepSeek API 适配器。模型 ID 必须通过配置指定，并可使用
`xifeng generator models` 查询当前账号实际可用的模型；下面的名称是项目目标配置，不作为
业务代码中的永久常量：

```text
provider: deepseek
base_url: https://api.deepseek.com
model: deepseek-v4-flash
```

DeepSeek API 使用 OpenAI 兼容的 Chat Completions 形式。API 密钥从进程环境变量或 Git 忽略的项目根目录 `.env` 读取；进程环境优先，密钥不进入 YAML、数据库或普通日志。

回答使用 JSON Output 加本地 Pydantic 与引用 ID 校验。一次问答最多执行首次生成和一次格式
修复，不启用隐藏的 SDK 自动重试或 provider 回退。用户可见 Markdown 与原文来源由本地程序
确定性渲染；没有证据时不调用 API。

Generator 接口仍与具体服务商解耦。外部调用默认只用于交互式答案生成；语义分块仅允许通过单篇 `chunks retry`、逐次二次确认和本地审核校验使用外部 OpenAI-compatible profile，仍禁止批量外部分块和外部 Contextual Retrieval。

个人用户可选 `codex-cli` 作为单篇分块 provider，复用现有 ChatGPT 登录额度。该子进程只在
空临时目录中以 read-only、ephemeral 和 JSON Schema 约束模式运行，不加载用户配置、规则或
MCP；它在隐私分类上仍属于外部推理，并依赖用户自己的 ChatGPT 数据控制设置。

评测策展也复用同一个通用隔离 Codex CLI 客户端，默认模型为 `gpt-5.6-terra`。每次只允许
处理人工明确选择的一篇已索引文章；页面必须先显示问题、版本、字符数、单元数、provider、
模型和最大请求次数，再由用户在弹窗中确认。结构化输出只生成待审候选，评测冻结与运行阶段
不自动调用 Codex。

第一阶段默认使用 `qwen3:4b-instruct-2507-q4_K_M` 完成离线语义分块和概括，实际上下文先设为 32K，并通过 Ollama JSON Schema 输出与 Pydantic 校验保证结构。少数单篇重试可使用显式配置的 OpenAI-compatible 外部模型；外部 JSON mode 不能替代本地 Pydantic 与覆盖校验。这些模型都属于索引预处理组件，不作为本地 Generator 对外提供问答服务。

Windows 使用官方 Ollama standalone CLI，不安装托盘应用、不注册开机启动。程序放在 `D:\ai-tools\ollama`，模型通过 `OLLAMA_MODELS` 指向 `D:\ai-models\ollama`。Ollama 服务只在离线预处理时按需启动。模型或运行后端只有在本机长文验收不达标时才替换。

## 5. 前端

正式前端不属于第一阶段范围。第一阶段只有 CLI 按需启动、固定绑定回环地址的语料管理页与
分块审核页；它们不新增服务框架或前端构建链，也不承担通用 API 职责。核心与 API 稳定后使用 Svelte；
是否需要 SvelteKit、路由框架或 UI 组件库，到阶段 3 根据实际页面需求再决定。

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
