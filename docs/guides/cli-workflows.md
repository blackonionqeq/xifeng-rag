# CLI 工作流指南

本文汇总日常导入、审核、检索、问答和评测命令。项目概览和最短流程见仓库根目录 [README](../../README.md)；设计理由和数据约束见 [`docs/design/`](../design/)。

## 页面入口与一次性任务

```powershell
uv run xifeng workbench
```

| 类型 | 入口 | 说明 |
|---|---|---|
| 一级页面 | `xifeng demo` | 面试演示用的文章反查与带引用问答 |
| 一级页面 | `xifeng corpus manage` | 语料状态、扫描及审核入口 |
| 一级页面 | `xifeng eval curate` | 题集策展、检索评测运行、历史结果与基线对比 |
| 从属页面 | `xifeng chunks review ...` | 也可从语料管理页选择待审提案后打开 |
| 从属页面 | 细话题审核 | 只从语料管理页按文章打开 |
| 一次性任务 | `doctor`、索引构建、`eval answer-run` | 在终端运行；控制台提供可复制命令 |
| 页面或 CLI | `eval run` | 常规检索评测可在页面运行；实验开关保留 CLI |

项目控制台按需启动三个一级页面。再次点击运行中的服务会打开现有地址；关闭控制台会停止由
它启动的页面。原有独立命令保持可用。

## 环境检查与语料管理

```powershell
uv run xifeng doctor
uv run xifeng corpus manage
uv run xifeng corpus cleanup-dry-run
uv run xifeng corpus scan
# 确认需要为既有文章建立新版本时：
uv run xifeng corpus scan --include-modified
```

`corpus scan` 在首次导入前显示解析后的绝对语料路径和文章数。默认只导入新增文章；既有文章
的规范正文发生变化时只报告“已跳过正文更新”，不创建或切换版本，也不会触发重新分块。
明确接受这些成本时才使用 `--include-modified`。管理页对应选项“同时导入已修改文章”默认
不勾选。纯元数据更新和来源缺失标记不受此开关影响。清洗有歧义的文章会停在
`needs_cleanup_review`，需通过 `corpus resolve-cleanup` 人工裁决。

`corpus cleanup-dry-run` 只读比较活动版本与当前清洗规则，报告截断位置、字符数和链接计数，不显示正文或 URL，也不修改外部语料和业务数据。

`corpus manage` 启动仅绑定 `127.0.0.1` 随机端口的本地管理页，用于查看来源时间、版本、分块与索引进度，以及显式扫描更新。操作栏可在本地只读查看当前生效版本的规范正文和已批准分块，不解析 Markdown，也不会外发正文。它不会自动生成分块提案或绕过外发确认。已索引文章可逐篇点击“补充细话题”；页面会先披露版本、字符数、Codex 模型和一次请求上限，确认后才生成待审草案。

不完整或不应导入的文章可以稳定文章 ID 写入本地配置 `corpus.excluded_article_ids`；扫描会单独报告排除数量，不修改外部语料。

## 分块、审核与批准

```powershell
uv run xifeng chunks list
uv run xifeng chunks propose <ARTICLE_ID>
uv run xifeng chunks export-review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
uv run xifeng chunks review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
uv run xifeng chunks approve data/reviews/<ARTICLE_VERSION>.review.json
```

默认提案根据完整规范正文同时生成单层话题块、文章整体概括和可选细话题。模型只提出语义边界和检索辅助字段；审核时可调整原子单元边界、编辑概括和细话题，并用 `include_in_index` 决定分段是否进入索引。

`chunks review` 启动仅绑定回环地址的临时审核页。保存先写独立 `.draft.json`，不会覆盖原始提案；批准和索引更新失败时，旧活动索引保持可用。

### 单篇重试

```powershell
uv run xifeng chunks retry <ARTICLE_ID> --provider <PROFILE> --model <MODEL_ID> --prompt-file <PATH>
uv run xifeng chunks retry <ARTICLE_ID> --provider codex-cli --model <MODEL_ID>
```

也可以在 `uv run xifeng corpus manage` 的语料管理页中，对待分块或待审核文章点击“AI 分块”。
页面固定执行 `codex-cli + gpt-5.6-terra`，展示单篇完整正文的外发范围并逐次确认，成功后直接
打开新生成提案的分块审核页。需要更换 provider、模型或使用自定义 Prompt 时仍使用 CLI。

重试始终只处理明确指定的一篇文章。选择外部 provider 时，程序会在发送清洗后正文前显示版本、字符数、服务商、模型和最大请求数，并逐次要求确认。新结果是并列待审提案，不会自动批准；一次性 Prompt 仅影响本次重试。

## 既有文章概括与文章反查

### 反查评测命令选择

文章反查与正查证据评测使用不同题集和指标；不要用 `eval run` 评估“记得主题、忘了文章”的场景。根据评测目的选择命令：

| 目的 | 命令 | 覆盖范围 |
| --- | --- | --- |
| 复现旧累计文章文档基线 | `uv run xifeng eval article-discovery` | 单篇累计文档上的 BM25、Dense、Hybrid |
| 比较分层文章反查通道 | `uv run xifeng eval layered-article-discovery --candidate-k 20` | 概括、关键词、细话题、原文层的 BM25、Dense 与单元聚合 |
| 重放分层降级策略 | `uv run xifeng eval simulate-article-fallback <LAYERED_RESULT_JSON>` | 已保存分层结果上的候选降级与融合策略 |

`eval article-discovery` 是兼容旧行为的累计文档基线；分层文章反查的主评测使用 `eval layered-article-discovery`。后者要求 `--candidate-k` 与已构建的分层索引 Profile 一致。

新文章的整体概括随分块提案共同生成。以下命令只用于给既有完整批准清单补概括：

```powershell
uv run xifeng overviews propose
uv run xifeng overviews list
uv run xifeng overviews revise-pending --provider deepseek
uv run xifeng overviews approve <OVERVIEW_ID> --summary-file <PATH>
```

`overviews propose` 只读取标题、已批准话题标题、块级概括、关键词和索引决定，不读取 `original_text`。`revise-pending` 由用户显式触发，只外发上述派生元数据，不修改已批准概括，并保留来源供审计。

批准整体概括后可以构建独立文章级索引：

```powershell
uv run xifeng articles build-index
uv run xifeng articles build-layered-index --candidate-k 20
uv run xifeng articles search "记得的模糊主题" --top-k 5
uv run xifeng articles search "记得的模糊主题" --strategy cumulative_hybrid --mode hybrid
```

文章级搜索用于“记得主题但忘了文章”的反查。默认使用最新分层套件执行
`disagreement_cascade`：先查概括层，BM25/Dense 冲突时优先追加细话题层，否则按清晰度逐步
追加关键词、细话题或原文层，并融合所有已触发通道。输出会显示实际触发层。没有细话题的文章
仍可由其他层找回；需要复现旧行为时显式选择 `--strategy cumulative_hybrid`。分层套件缺失或
失效时命令会提示重建，不会静默切换策略。结果不是原文证据，也不参与现有 `search` 或 `ask`
的最终权重。

细话题用于补充容易被单独记住的案例、局部概念和机制。它独立于关键词和不可变分块清单，
允许整篇或任一父话题为空；只有批准后的细话题才进入文章级索引。历史文章的补充审核页必须
先保存草案，再确认批准并重建文章级索引，不会重建证据块索引。

## 证据索引与检索

```powershell
uv run xifeng index build
uv run xifeng search "要检索的问题" --top-k 5
uv run xifeng search "要检索的问题" --mode bm25 --top-k 5
uv run xifeng search "要检索的问题" --mode dense --top-k 5
uv run xifeng search "要检索的问题" --mode hybrid --rerank --top-k 5
```

默认索引构建和当前检索基线见[技术栈](../design/technology-stack.md)。本地重排通过 `--rerank` 显式开启；模型或推理失败时直接失败，不静默回退到 RRF。可用 `--rerank-candidate-k` 在单次搜索或评测中覆盖候选深度。

## 本地演示入口

```powershell
uv run xifeng demo
```

该命令启动只监听 `127.0.0.1` 的本地页面，集中展示“找文章”和“问文章库”。找文章复用默认
`disagreement_cascade`；问文章库固定使用当前 BM25 Top-5 回答基线。页面先在本地检索并展示
证据范围，只有点击“确认范围并生成回答”后才创建 DeepSeek 客户端。准备结果十分钟过期且只能
使用一次。使用 `--no-open` 可只打印地址，不自动打开浏览器。

“问文章库”会把最近 50 个问题及最近使用时间保存在 `data/demo-question-history.json`，便于重复
试题和挑选现场演示问题；相同问题自动去重，页面可以单条删除或全部清空。该文件不包含回答、
证据、引用或完整 Prompt。

本地页面可能显示付费原文引用。公开录屏、截图或部署时应使用自造短文本语料。

### 查询改写

```powershell
uv run xifeng search "要检索的问题" --mode hybrid --rerank --rewrite --top-k 5
```

`--rewrite` 只在 Hybrid + Reranker 路径显式启用。改写器只看当前问题，不读取语料或首轮候选；本地模型不可用或输出无效时会提示并退回原问题，不自动外发。

### 分层检索

```powershell
uv run xifeng index build --profile layered
uv run xifeng search "要检索的问题" --mode hybrid --rerank --layered --build <BUILD_ID>
```

分层 profile 额外构建父 TopicChunk 和确定性子块索引，不替换当前活动索引。查询必须同时指定 `--layered`、`--build` 和 `--rerank`。真实命中与邻居扩展分别记录。

### 上下文组装与诊断

```powershell
uv run xifeng search "要检索的问题" --mode hybrid --rerank --assemble
```

`--assemble` 在本地生成带 `E1`、`E2` 等查询内引用 ID 的规范证据 JSON，不调用 Generator。数量和字符预算、AI 元数据开关见配置及[总体架构](../design/architecture.md)。

证据观测层默认关闭，可在本地配置中启用：

```yaml
evidence_assessment:
  enabled: true
  observation_k: 5
```

它记录排序间隔和多路召回一致性，只用于诊断，不判断证据是否充分，也不自动切换检索策略。

## Generator 与问答

```powershell
uv run xifeng generator models
uv run xifeng ask "要研究的问题" --mode hybrid --rerank
uv run xifeng ask "要研究的问题" --mode hybrid --rerank --rewrite
uv run xifeng ask "要研究的问题" --mode hybrid --rerank --supplement on
```

`generator models` 用于查询账号实际可用的模型。`ask` 会在外部调用前显示服务、模型、证据数量、原文字符数和最大请求数；没有证据时在本地拒答。模型输出按本地 Schema 和引用 ID 校验，格式修复最多一次。

`--supplement auto|on|off` 控制独立标注的模型补充分析。该部分只能提供评价框架、风险和建议，不能凭模型记忆补充新的具体事实。配置方法见 [DeepSeek Generator 快速使用](deepseek-generator-setup.md)。

## 评测集与冻结实验

```powershell
uv run xifeng eval curate
uv run xifeng eval run <FROZEN_DATASET_ID> --split dev --mode hybrid --k 5
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --rerank --rerank-candidate-k 20 --k 5
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --rerank --rewrite --k 5
uv run xifeng eval run <FROZEN_DATASET_ID> --build <FROZEN_BUILD_ID> --mode hybrid --rerank --subtopics --k 5
uv run xifeng eval article-discovery
uv run xifeng eval compare-article-discovery <BASELINE_JSON> <CURRENT_JSON>
uv run xifeng articles build-layered-index --candidate-k 20
uv run xifeng eval layered-article-discovery --candidate-k 20
uv run xifeng eval compare-layered-article-discovery <BASELINE_JSON> <CURRENT_JSON>
uv run xifeng eval simulate-article-fallback <LAYERED_RESULT_JSON>
```

`eval curate` 启动本地评测工作台。“题集策展”用于录入问题、编辑参考答案和按原子单元标注
证据范围；AI 辅助候选需要在显示外发范围后由用户二次确认，且必须人工接受或修改才能成为
金标准。“运行结果”可对冻结题集运行 BM25、Dense 或 Hybrid 检索评测，选择本地 Reranker，
并查看汇总、难度/来源分组、逐题候选、证据覆盖和历史基线差值。页面只列出与题集语料快照
兼容的 Index Build；运行 holdout 前会再次确认其不能用于反复调参。

`eval run` 只接受冻结数据集，并拒绝在不匹配的 Index Build 上运行。pilot、正式 dev/holdout 的范围和实验纪律见[版本路线](../design/roadmap.md)，具体审核标准见[评测问题编写与 AI 草案审核指南](evaluation-question-writing.md)。
页面与 CLI 复用同一计分和 `evaluation_runs` 持久化逻辑。分层检索、查询改写、细话题召回及
候选深度覆盖等实验参数仍通过 CLI 显式运行，不放进首版页面的常用表单。

正式题集冻结后，可单独运行端到端回答与引用评测：

```powershell
uv run xifeng eval answer-run <FROZEN_DATASET_ID> --split dev --mode bm25 --k 5
```

该命令必须显式选择 `dev` 或 `holdout`。它先在本地完成整批检索和上下文组装，再显示每题证据
块数、原文字符数、服务、模型与最大请求数；确认后才顺序调用 Generator。自动结果评估拒答
状态、引用有效性、金标准证据与答案要点的引用覆盖、阶段失败和成本，不把证据覆盖称为答案
正确率。拒绝确认时不会创建外部客户端或发送证据。
正式 benchmark 在开发阶段必须使用 `--split dev`；只有最终评估时才运行 `--split holdout`。

`--subtopics` 在常规 Hybrid+Rerank 之上叠加细话题召回通道：问题命中已批准细话题后提名其父
TopicChunk，与 BM25/Dense 做四路加权 RRF 再重排。需要先 `articles build-layered-index` 生成分层
套件；必须与 `--build`、`--mode hybrid`、`--rerank` 同用，且与 `--layered`、`--rewrite` 互斥。
候选宇宙仍是冻结 Build 的话题块，宇宙外提名只计数不生效；每题与 run 级结果记录提名与丢弃审计。
pilot 6 题实验结果为整体退化，不默认启用；设计与实验结论见
[正查细话题召回通道设计](../specs/2026-09-03-subtopic-recall-channel-design.md)。

`eval article-discovery` 默认读取 Git 忽略的 `data/article_discovery_queries.json`，只运行其中
人工确认的问题，并在同一个文章级 Index Build 上依次比较 BM25、Dense 和 Hybrid。命令报告
Hit@1/3/5、MRR、延迟与不在 Build 中的目标文章数，并把不含正文和文章概括的逐题排名保存到
`data/evaluation_runs/`。可传入其他 JSON 路径、用 `--build` 固定 Build，或用 `--json` 输出
完整结构化结果。该命令与证据块的 `eval run` 使用不同金标准，二者不计算跨任务总分。

`eval compare-article-discovery` 比较两次已经保存的文章反查评测。命令会先校验题集内容哈希、
确认题数、评测深度、BM25/Dense 候选深度和 RRF 参数完全一致，再输出三个检索模式的聚合指标
变化及逐题正确文章排名变化；不可比时直接拒绝。使用 `--json` 可输出结构化对比结果。该命令
只读取不含正文的评测结果，不加载索引或模型。

`articles build-layered-index` 为概括、关键词、已批准细话题和批准范围内的原文分别创建独立、
默认不激活的 BM25 + Dense Build，并原子发布一份套件清单；候选深度属于 Profile，评测时必须
保持一致。重复构建时，未变化层直接复用既有 Build；变化层全量重建 BM25，并按稳定单元 ID
复用相同 Profile 的历史 Dense 向量。命令输出每层的向量复用数和新编码数。`eval
layered-article-discovery` 默认读取最新套件，在概括/关键词层运行 `best_unit`，
在细话题/原文层同时运行 `best_unit` 与最多三个单元的 `top_n_rank_fusion`，报告各层
Hit@1/3/5、MRR、延迟、目标不可用数、双路一致性和按单元数分组的结果。持久化记录只包含
稳定 ID、标题、父话题和范围，不包含正文、完整概括或细话题集合。

`eval simulate-article-fallback` 只读取上述安全结果，不加载索引或模型。它可用 `--policy`、
`--margin-ratio`、`--overlap` 和 `--rrf-k` 重放候选降级策略，同时报告各层触发比例、准确性、
平均/最坏延迟估算以及逐题改善和退化。阈值只能在扩充后的 dev 上选择，当前小题集结果不能
直接成为线上控制器配置。

`eval compare-layered-article-discovery` 只比较题集哈希、语料快照、四层 Profile、候选/结果深度、
聚合方法与 RRF 参数一致的运行；任一实验条件不一致都会拒绝输出单变量 A/B 结论。
