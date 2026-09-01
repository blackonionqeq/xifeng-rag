# CLI 工作流指南

本文汇总日常导入、审核、检索、问答和评测命令。项目概览和最短流程见仓库根目录 [README](../../README.md)；设计理由和数据约束见 [`docs/design/`](../design/)。

## 环境检查与语料管理

```powershell
uv run xifeng doctor
uv run xifeng corpus manage
uv run xifeng corpus cleanup-dry-run
uv run xifeng corpus scan
```

`corpus scan` 在首次导入前显示解析后的绝对语料路径和文章数。清洗有歧义的文章会停在 `needs_cleanup_review`，需通过 `corpus resolve-cleanup` 人工裁决。

`corpus cleanup-dry-run` 只读比较活动版本与当前清洗规则，报告截断位置、字符数和链接计数，不显示正文或 URL，也不修改外部语料和业务数据。

`corpus manage` 启动仅绑定 `127.0.0.1` 随机端口的本地管理页，用于查看来源时间、版本、分块与索引进度，以及显式扫描更新。它不会自动生成分块提案或绕过外发确认。

不完整或不应导入的文章可以稳定文章 ID 写入本地配置 `corpus.excluded_article_ids`；扫描会单独报告排除数量，不修改外部语料。

## 分块、审核与批准

```powershell
uv run xifeng chunks list
uv run xifeng chunks propose <ARTICLE_ID>
uv run xifeng chunks export-review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
uv run xifeng chunks review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
uv run xifeng chunks approve data/reviews/<ARTICLE_VERSION>.review.json
```

默认提案由本地 Ollama 根据完整规范正文同时生成单层话题块和文章整体概括。模型只提出语义边界和检索辅助字段；审核时可调整原子单元边界、编辑概括，并用 `include_in_index` 决定分段是否进入索引。

`chunks review` 启动仅绑定回环地址的临时审核页。保存先写独立 `.draft.json`，不会覆盖原始提案；批准和索引更新失败时，旧活动索引保持可用。

### 单篇重试

```powershell
uv run xifeng chunks retry <ARTICLE_ID> --provider <PROFILE> --model <MODEL_ID> --prompt-file <PATH>
uv run xifeng chunks retry <ARTICLE_ID> --provider codex-cli --model <MODEL_ID>
```

重试始终只处理明确指定的一篇文章。选择外部 provider 时，程序会在发送清洗后正文前显示版本、字符数、服务商、模型和最大请求数，并逐次要求确认。新结果是并列待审提案，不会自动批准；一次性 Prompt 仅影响本次重试。

## 既有文章概括与文章反查

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
uv run xifeng articles search "记得的模糊主题" --top-k 5
```

文章级搜索用于“记得主题但忘了文章”的反查。结果不是原文证据，也不参与现有 `search` 或 `ask` 的最终权重。

## 证据索引与检索

```powershell
uv run xifeng index build
uv run xifeng search "要检索的问题" --top-k 5
uv run xifeng search "要检索的问题" --mode bm25 --top-k 5
uv run xifeng search "要检索的问题" --mode dense --top-k 5
uv run xifeng search "要检索的问题" --mode hybrid --rerank --top-k 5
```

默认索引构建和当前检索基线见[技术栈](../design/technology-stack.md)。本地重排通过 `--rerank` 显式开启；模型或推理失败时直接失败，不静默回退到 RRF。可用 `--rerank-candidate-k` 在单次搜索或评测中覆盖候选深度。

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
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --k 5
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --rerank --rerank-candidate-k 20 --k 5
uv run xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --rerank --rewrite --k 5
```

`eval curate` 启动本地评测集工作台，用于录入问题、编辑参考答案和按原子单元标注证据范围。AI 辅助候选需要在显示外发范围后由用户二次确认，且必须人工接受或修改才能成为金标准。

`eval run` 只接受冻结数据集，并拒绝在不匹配的 Index Build 上运行。pilot、正式 dev/holdout 的范围和实验纪律见[版本路线](../design/roadmap.md)，具体审核标准见[评测问题编写与 AI 草案审核指南](evaluation-question-writing.md)。
