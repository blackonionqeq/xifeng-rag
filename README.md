# 西风文章 RAG 研究台

面向个人购买的长篇文章，构建一个本地优先、证据驱动的主题研究工具，同时作为可复现的 RAG 学习与求职展示项目。

当前已实现 V0 MVP 与 V1 单轮问答主链：确定性语料清洗、本地 AI 话题分块与人工审核、BM25 + Dense + RRF 混合检索、本地重排、确定性上下文组装、DeepSeek 结构化回答、稳定原文引用和基础检索评测。正式前端与多轮对话属于后续版本。

当前语料和 pilot 评测足以验证工作流，但不足以支撑参数或模型的泛化结论。近期重点是扩大已审核语料、建立正式 benchmark，并补齐不可回答、跨文章、引用与最终回答质量评测。具体进度见[版本路线与实验方向](docs/design/roadmap.md)。

## 当前能力

- **语料管理**：只读扫描外部 Markdown，确定性清洗、来源映射、内容哈希和增量变化分类；
- **分块审核**：本地 Ollama 生成话题块与文章概括，通过 JSON 或本地审核页人工确认；
- **文章反查**：使用已批准概括和话题信息构建独立的文章级 BM25 + Dense + RRF 搜索；
- **证据检索**：在已批准原文块上执行 BM25、Dense、RRF、本地 Cross-Encoder 重排，以及可选的分层检索和查询改写实验；
- **引用回答**：按预算组装带稳定来源的证据，由用户显式触发 DeepSeek 生成结构化单轮回答；
- **评测实验**：人工策展并冻结题集，在固定语料快照上评估检索质量、证据覆盖、延迟和失败案例。

## 安装

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --python 3.11
uv run xifeng init
```

本地密钥可放在 Git 忽略的项目根目录 `.env`：

```powershell
Copy-Item .env.example .env
```

```dotenv
DEEPSEEK_API_KEY=your-key-here
```

真实进程环境变量优先于 `.env`。本地模型和缓存路径通过 Git 忽略的 `config.local.yaml` 覆盖；技术选择和默认模型见[技术栈](docs/design/technology-stack.md)。

## 最短流程

```powershell
uv run xifeng doctor
uv run xifeng corpus manage
uv run xifeng corpus scan
uv run xifeng chunks propose <ARTICLE_ID>
uv run xifeng chunks review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
uv run xifeng index build
uv run xifeng search "要检索的问题" --mode hybrid --rerank --top-k 5
uv run xifeng ask "要研究的问题" --mode hybrid --rerank
```

首次导入时，`corpus scan` 会显示解析后的绝对语料路径和文章数。默认扫描只导入新增文章，
既有文章的正文变化只报告不落库；确认需要重新分块时使用 `corpus scan --include-modified`。
清洗有歧义的文章必须先人工裁决，只有已批准版本能够进入索引。`ask` 会在调用外部 Generator
前显示外发范围；没有证据时在本地直接拒答。

完整的导入、审核、实验检索、文章反查和评测命令见 [CLI 工作流指南](docs/guides/cli-workflows.md)。DeepSeek 配置见 [Generator 快速使用](docs/guides/deepseek-generator-setup.md)，评测题编写见[评测问题与 AI 草案审核指南](docs/guides/evaluation-question-writing.md)。

## 设计文档

- [总体架构](docs/design/architecture.md)
- [切块、上下文化与增量索引](docs/design/chunking-and-indexing.md)
- [版本路线与实验方向](docs/design/roadmap.md)
- [技术栈](docs/design/technology-stack.md)
- [Agent 工作约束](AGENTS.md)

具体功能的历史设计记录位于 `docs/specs/`，实施计划位于 `docs/plans/`。

## 语料与隐私

付费文章不存放在本仓库内。应用通过本地配置项 `corpus.source_dir` 只读加载外部目录顶层的 Markdown；不得将真实正文、审核产物或包含正文的索引提交到版本库。运行数据库、审核文件和索引位于 Git 忽略的 `data/`。

完整的语料处理、外发例外和隐私约束见 [AGENTS.md](AGENTS.md) 与[切块和索引设计](docs/design/chunking-and-indexing.md)。
