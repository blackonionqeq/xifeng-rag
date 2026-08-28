# 西风文章 RAG 研究台

面向个人购买的长篇文章，构建一个本地优先、证据驱动的主题研究工具，同时作为可复现的 RAG 学习与求职展示项目。

当前已实现 V0 MVP 与 V1 单轮问答主链：确定性语料清洗、本地 AI 话题分块与人工审核、
BM25、BGE Dense、RRF 混合检索、可显式启用的本地 Cross-Encoder 重排、确定性上下文组装、
DeepSeek 结构化回答、稳定原文引用和基础检索评测。正式前端与多轮对话属于后续版本。

## 当前能力

- 只读扫描外部目录顶层 Markdown；
- 确定性清洗、物理行/句末原子单元、链接锚文本保留、精确源位置映射和内容哈希；
- 新增、修改、未变化与缺失文章分类；
- 本地语料管理页，展示来源时间、分块与索引进度，并显式扫描更新；
- 使用本地 Ollama 和 JSON Schema 生成单层话题块提案；
- JSON/Markdown 人工审核产物、逐段索引决定和不可变批准清单；
- 单篇分块重试，可切换本地或显式确认的外部模型并使用一次性自定义 Prompt；
- 已批准话题块上的中文 BM25、BGE Dense、RRF 混合检索和原文引用；
- 使用 `BAAI/bge-reranker-v2-m3` 对 Hybrid 宽候选进行显式本地重排；
- 将排序候选按数量与字符预算组装为带稳定来源的本地证据上下文；
- 由用户显式触发 DeepSeek 单轮回答，支持结构化引用、部分回答、证据不足和模型补充分析；
- 本地评测集工作台、Codex 辅助候选答案与人工证据范围审核；
- 冻结语料快照上的 P@K、Recall@K、MRR、nDCG、证据覆盖与查询延迟评测。

## 安装

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --python 3.11
uv run xifeng init
```

本地密钥可放在 Git 忽略的项目根目录 `.env` 中。先复制示例，再填写实际值：

```powershell
Copy-Item .env.example .env
```

```dotenv
DEEPSEEK_API_KEY=your-key-here
```

真实进程环境变量优先于 `.env`；程序不会在普通输出中显示密钥。

离线语义分块默认使用 Windows standalone Ollama：

```text
程序：D:\ai-tools\ollama\ollama.exe
模型：D:\ai-models\ollama
模型名：qwen3:4b-instruct-2507-q4_K_M
```

若路径不同，在 Git 忽略的 `config.local.yaml` 中覆盖。Ollama 只用于本地预处理，不作为
交互式 Generator。单轮问答使用显式触发的 DeepSeek API，只发送最终组装的少量证据。

## 最短流程

```powershell
uv run xifeng doctor
uv run xifeng corpus manage
uv run xifeng corpus scan
uv run xifeng chunks list
uv run xifeng chunks propose <ARTICLE_ID>
uv run xifeng chunks retry <ARTICLE_ID> --provider <PROFILE> --model <MODEL_ID> --prompt-file <PATH>
uv run xifeng chunks retry <ARTICLE_ID> --provider codex-cli --model <MODEL_ID>
uv run xifeng chunks export-review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
uv run xifeng chunks review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
# 阅读并按需修改 data/reviews/*.review.json
uv run xifeng chunks approve data/reviews/<ARTICLE_VERSION>.review.json
uv run xifeng index build
uv run xifeng search "要检索的问题" --top-k 5
uv run xifeng search "要检索的问题" --mode bm25 --top-k 5
uv run xifeng search "要检索的问题" --mode dense --top-k 5
uv run xifeng search "要检索的问题" --mode hybrid --rerank --top-k 5
uv run xifeng search "要检索的问题" --mode hybrid --rerank --assemble
uv run xifeng generator models
uv run xifeng ask "要研究的问题" --mode hybrid --rerank
uv run xifeng ask "要研究的问题" --mode hybrid --rerank --supplement on
uv run xifeng eval curate
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --k 5
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --rerank --rerank-candidate-k 20 --k 5
```

首次导入前，`corpus scan` 会显示解析后的绝对语料路径和发现的文章数。清洗存在
歧义的文章会停在 `needs_cleanup_review`，必须通过
`corpus resolve-cleanup` 人工裁决。只有已批准版本能够进入索引。

`corpus manage` 在随机端口启动只绑定 `127.0.0.1` 的本地语料管理页。页面按文章聚合展示
抓取时间、文件修改时间、最新来源版本、当前索引版本、待审提案和索引块数量；“扫描语料
更新”按钮复用相同的增量扫描流程。存在待审提案时可从表格打开现有审核页。管理页不会
自动生成分块提案，也不会绕过单篇外发确认。

已确认不完整或不应导入的单篇文章可在本地配置的
`corpus.excluded_article_ids` 中按稳定文章 ID 排除。排除不会修改外部语料；扫描会单独
报告排除数量。话题分块以自然语义边界为准，短而连贯的完整文章允许只有一个块。
模型默认只提出语义边界，所有分段先设为进入索引。用户可在审核 JSON 中调整原子单元
边界，并将不需要检索的分段设为 `include_in_index: false`；这些内容仍保留在审核清单中，
但不会生成 TopicChunk。

`chunks retry` 始终只处理明确指定的单篇文章。默认仍使用本地 Ollama；选择外部 profile
时，程序会显示实际文章版本、字符数、服务商、模型和最大请求次数，并在发送完整清洗后
正文前逐次确认。`codex-cli` 可复用当前 ChatGPT 登录，在空临时目录以只读、临时会话和
Schema 约束模式运行；它仍属于外部推理，训练使用取决于该 ChatGPT 账户的数据控制设置。
自定义 Prompt 仅影响当前 retry。

`chunks review` 在随机端口启动只绑定 `127.0.0.1` 的临时审核页面。页面可调整原子单元
边界、修改检索辅助字段并逐块设置是否进入索引；保存结果写入独立的 `.draft.json`，不会
覆盖模型原始提案。草案成功保存后，可在页面二次确认“批准并更新索引”；若索引失败，
旧活动索引保持可用，页面可直接重试。模型重试仍由 CLI 单篇触发。

`eval curate` 启动独立的本地评测集工作台。第一版用于人工录入单篇检索问题、选择已索引
文章、编辑参考答案和按原子单元标注稳定证据范围。点击“AI 辅助判断”后，页面会先显示问题、
文章版本、正文哈希、字符数、原子单元数、Codex provider、模型和最大请求次数；只有用户在
弹窗中再次确认，才会把该篇完整清洗正文发送给隔离的 Codex CLI。结构化候选必须经人工
接受、修改并标记审核完成，不能自动成为金标准或冻结题集。

首批 6–8 题应建立为 `pilot`，用于验证工作流；pilot 结果不应原地升级为正式 holdout。
冻结会保存完整语料快照与题集内容哈希，`eval run` 只接受冻结数据集，并拒绝在不匹配的
Index Build 上运行。旧的 JSON 文件评测入口暂时兼容保留。

`index build` 默认构建原文 BM25 与 `BAAI/bge-base-zh-v1.5` Dense 索引；可用
`--profile bm25` 显式构建旧关键词基线。模型缓存目录通过本地配置设置，推荐放在非系统盘。
查询和冻结评测支持 `--mode bm25|dense|hybrid`；未指定时，包含 Dense 产物的新 Build 默认
使用 Hybrid，旧 BM25 Build 自动使用 BM25。当前 34 块 pilot 语料的 Hybrid 默认让两路各召回 5 块，再以
`rrf_k=60` 融合并返回请求的 Top-K。

本地重排必须通过 `--rerank` 显式开启，第一版只支持 Hybrid 模式。启用后 BM25 与 Dense
都会至少召回 `reranker.candidate_k` 个候选，RRF 去重融合后由本地 Cross-Encoder 按
`query + 文章标题 + original_text` 精排。默认 pair 总预算为 1024 tokens，超长候选的截断
会写入候选调试信息和冻结评测详情；模型或推理失败时命令直接失败，不静默回退到 RRF。
默认候选深度为 5，这是当前 6 题 pilot 上不损失证据覆盖且能改善 MRR/nDCG 的保守选择。
可用 `--rerank-candidate-k` 在单次搜索或评测中覆盖候选深度，便于在正式 dev benchmark
扩充后重新运行 5/10/20/30 对照。

`search --assemble` 在本地把排序候选转换为带 `E1`、`E2` 等查询内引用 ID 的规范 JSON
证据上下文，不调用 Generator，也不产生外部请求。默认最多选择 5 个完整 TopicChunk、总渲染
上限 50,000 字符；放不下的后续块会记录排除原因，排名第一的块若单独超限则明确失败，不静默
截断或改写原文。AI 话题标题与概括保留在内部结构中，但默认不进入 Generator 上下文；可通过
`context_assembly.include_ai_metadata` 开启用于后续单变量对照，它们始终不能作为作者原话引用。

`ask` 是显式的外部回答入口。调用前会显示 DeepSeek 服务地址、模型、入选证据数量、原文
字符数和最多两次外部请求；没有证据时在本地直接拒答，不读取 API 密钥也不产生网络请求。
回答将资料明确陈述与基于资料的推断分开，并逐段绑定有效的 `E1`、`E2` 引用。输出非法时
只允许一次结构修复，不会静默切换模型。`--supplement auto|on|off` 控制独立标注的模型补充
分析；该部分只能提供评价框架、风险和建议，不能凭模型记忆补充新的具体事实。

DeepSeek 模型 ID 可能随账号与服务版本变化。先运行 `generator models` 查询实际可用 ID，
命令会直接说明当前配置是否可用以及下一步怎么做。简短配置说明见
[DeepSeek Generator 快速使用](docs/guides/deepseek-generator-setup.md)。API 密钥仍只从进程
环境或 Git 忽略的 `.env` 读取。

出题范围、Top-K 约束、AI 参考答案、答案要点、证据范围和不确定性的人工审核标准，见
[评测问题编写与 AI 草案审核指南](docs/guides/evaluation-question-writing.md)。

## 设计文档

- [总体架构](docs/design/architecture.md)
- [切块、上下文化与增量索引](docs/design/chunking-and-indexing.md)
- [版本路线与实验方向](docs/design/roadmap.md)
- [技术栈](docs/design/technology-stack.md)
- [V0 MVP 设计](docs/specs/2026-08-25-v0-mvp-design.md)
- [V0 实施计划](docs/plans/2026-08-25-v0-mvp-implementation.md)
- [Codex CLI 单篇分块 Provider 设计](docs/specs/2026-08-26-codex-cli-chunk-provider-design.md)
- [本地语料管理页设计](docs/specs/2026-08-26-corpus-management-dashboard-design.md)
- [评测集策展与 Codex 辅助设计](docs/specs/2026-08-27-evaluation-dataset-curation-design.md)
- [确定性上下文组装设计](docs/specs/2026-08-28-deterministic-context-assembly-design.md)
- [DeepSeek Generator 设计](docs/specs/2026-08-28-deepseek-generator-design.md)
- [评测问题编写与 AI 草案审核指南](docs/guides/evaluation-question-writing.md)
- [Agent 工作约束](AGENTS.md)

## 语料来源

付费文章不存放在本项目内，也不提交到本项目的版本库。当前开发环境从以下外部目录只读加载文章：

```text
D:\codes\projects\test-reverse-wechat-official-accounts-paid-article\saved_articles
```

从项目根目录计算，对应的默认相对路径为：

```text
../test-reverse-wechat-official-accounts-paid-article/saved_articles
```

实现时通过本地配置项 `corpus.source_dir` 覆盖该位置。默认只索引目录顶层的 Markdown；完整的读取、隐私和排除规则见 [AGENTS.md](AGENTS.md) 与[切块和索引设计](docs/design/chunking-and-indexing.md)。

运行数据库、审核文件和索引位于 `data/`，该目录已被 Git 忽略。不得提交真实付费
正文、审核产物或索引。
