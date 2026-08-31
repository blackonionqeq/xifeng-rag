# V1 后续待办与新会话交接

> **历史快照：已被取代。** 本文记录 2026-08-28 时的状态，其中 Reranker、上下文组装和
> DeepSeek Generator 等 P0 项现已完成，不应再作为当前待办执行。当前能力以
> [README](../../README.md) 为准，后续优先级以[版本路线](../design/roadmap.md)为准；最新的
> 检索实验结论见[证据充分度观测层设计](../specs/2026-08-31-evidence-assessment-observation-design.md)。

更新时间：2026-08-28（Asia/Shanghai）

## 一句话状态

V0 本地研究台及 V1 的 BGE Dense + BM25 + RRF 混合召回基线已经完成并提交。当前最应该
继续的是补齐 `Reranker -> 上下文组装 -> Generator -> 原文引用/拒答` 的问答主链，而不是
立刻扩展 GraphRAG、查询改写或正式前端。

## 已完成基线

- 外部付费 Markdown 只读扫描、确定性清洗、稳定原子单元及源位置映射；
- 本地 AI 话题分块、人工审核、不可变批准清单和增量版本状态；
- 语料管理页、分块审核页与评测集策展页；
- Codex 辅助生成评测答案草案，完整文章外发前逐题确认，草案必须人工审核；
- 冻结语料快照和题集，在稳定原子单元证据范围上运行检索评测；
- 中文 BM25、`BAAI/bge-base-zh-v1.5` Dense、RRF Hybrid 三种检索模式；
- P@K、Recall@K、MRR、nDCG、evidence coverage 和查询延迟记录；
- 当前代码质量基线：85 个测试通过，Ruff 与 Pyright 通过；
- 当前工作已提交至 `f4f1d08`（`Add BGE hybrid retrieval`）。

当前 pilot 只有 6 道单篇 dev 题，没有 holdout，也没有应拒答题。这批题只适合开发和验证
工作流，不能原地充当正式 benchmark 的 holdout。

### 当前可复现实验

```text
corpus snapshot: 040cda83...
index build:     index_0d76f5fc66efbcb8ab931ba594e93d5f
index profile:   profile_648a714ffc47ac8494ccffcf438b44d8b369a77d994be779

BM25 run:        eval_98c42de43e30ca577fd205cffe5da456
Dense run:       eval_ebff15dafc613900f5e49ec7d943092d
Hybrid run:      eval_fc647529ebeab8ad47eaf97dea28b0ee
```

当前 34 个 TopicChunk 上，Top-5 结果为：

| 模式 | P@5 | Recall@5 | MRR | nDCG | Evidence coverage |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1333 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| Dense | 0.2000 | 0.6944 | 0.8333 | 0.7298 | 0.7377 |
| Hybrid | 0.2333 | 0.8611 | 0.8333 | 0.8108 | 0.9044 |

Hybrid 当前默认参数是 BM25 candidate depth 5、Dense candidate depth 5、`rrf_k=60`、最终
Top-K 5。该参数只经过小型 dev pilot 验证；扩大语料和题集后必须重新验证，不能当成通用
最优值。

## 待办优先级

### P0：先确定并实现本地 Reranker

目标是让混合召回先取较宽候选集，再由本地模型精排，输出统一候选结构并保留各阶段分数。

建议范围：

1. 先写 Reranker 设计和实施计划，明确候选深度、最终 Top-K、截断策略和失败回退；
2. 基线模型使用设计文档已选的 `BAAI/bge-reranker-v2-m3`；
3. 模型、revision、设备、最大长度和查询期参数进入评测运行元数据；
4. Reranker 关闭时保持当前 Hybrid 行为，加载或推理失败时不能静默改变实验配置；
5. 在同一冻结 pilot 上比较 Hybrid 与 Hybrid + Reranker，并检查逐题退化案例；
6. 全程本地运行，不向外部服务发送候选正文。

验收至少包括：稳定排序、分数记录、长块截断、空候选、模型失败、CLI 检索和冻结评测回归。

### P0：上下文组装、引用与拒答协议

Reranker 后紧接着实现独立的上下文组装层，不要把拼接规则藏进 Generator Prompt。

需要明确：

- 按 token/字符预算选择最终证据，去重相邻或高度重叠块；
- 每段证据分配稳定引用 ID，并关联文章标题、文章版本、原子单元范围和原文；
- contextual text、话题标题、摘要与关键词只能辅助检索，不能作为作者原话引用；
- Generator 输出结构化的 `answered` 或 `insufficient_evidence`；
- 每个可验证答案要点必须引用一个或多个实际提供的证据 ID；
- 程序校验引用 ID 存在、引用片段来自 `original_text`，缺失或非法引用时不展示为可靠答案；
- 检索永远返回 Top-K 不等于证据充分，拒答判断不能只靠“是否取到了块”。

提示词里写“找不到就说不知道”只能作为一层约束，不能替代结构化输出、程序校验和拒答评测。

### P0：DeepSeek Generator 与最小问答入口

实现现有设计里的 `Generator` 窄接口和 DeepSeek 适配器，随后提供最小可用的 CLI 问答入口。

必须遵守：

- 只有用户显式发起问答后才能调用外部 Generator；
- 调用前明确展示将外发最终入选的少量证据片段；
- API 密钥只读进程环境或 Git 忽略的 `.env`，环境优先；
- 普通日志、数据库和错误信息不保存完整提示上下文、付费原文或密钥；
- 保存 provider、model、Prompt 版本、生成参数、延迟和结构化结果状态；
- 若 `deepseek-v4-flash` 的实际可用模型名或 API 行为与旧设计不一致，先用当前官方文档核实并
  同步更新设计，不要凭旧名称直接实现；
- 先做单篇问答闭环，跨文章综合后置。

最小闭环应输出：答案或证据不足、逐条引用、所用文章，以及可选的各阶段检索与重排分数。

### P1：把 pilot 扩展成正式 benchmark

主链可运行后建立新的正式题集，而不是修改现有 pilot 的身份：

- 约 12 道 dev + 8 道 holdout；
- 加入少量明确不可回答的问题；
- 逐题人工确认参考答案、答案要点和稳定证据范围；
- holdout 不用于选择 RRF、Reranker、Prompt 或阈值；
- 后续再加入跨文章综合、观点演变和潜在矛盾题型；
- 对拒答至少区分正确拒答、错误作答和过度拒答；
- 增加引用有效性、要点覆盖、忠实度、完整性、延迟与失败类型记录。

现有 6 题可继续用作快速 dev 回归，但小样本指标只能用于发现问题，不能支撑泛化结论。

### P1：本地 Q&A API 与页面

核心问答链稳定后再增加 FastAPI 和正式页面。页面至少展示：

- 问题、答案状态及外发确认；
- 答案中的可点击引用；
- 原文证据、文章标题、版本和原子单元范围；
- BM25、Dense、RRF、Reranker 各阶段调试信息；
- 证据不足或生成失败的清晰状态。

先实现单篇/限定范围问答，再做全库跨文章研究体验。Svelte 相关选型留到页面需求实际确定时。

### P2：检索与索引优化实验

在正式 dev benchmark 上按单变量原则依次考虑：

1. 原文 Dense 与 AI 上下文化 Dense；
2. `bge-base-zh-v1.5` 与其他本地 Embedding；
3. `bge-reranker-v2-m3` 与其他本地 Reranker；
4. 话题长块、固定长度小块与父子块；
5. 更完整的 Contextual Retrieval；
6. 不同候选深度、最终 K、RRF 参数及上下文预算。

每次变化都创建或记录独立、可复现的配置身份，报告提升、退化、成本和失败案例。不要只留下
最优结果。

### P2：真正的增量向量索引

当前语料扫描和版本状态支持增量，但 Hybrid Build 会重新编码全部已批准 TopicChunk。语料量
明显增大后，再实现只补建新增/修改文章派生产物以及原子切换活动索引；必须覆盖新增、修改、
未变化、缺失和失败回滚场景。

### P3：后续研究项

- 跨文章主题综合与时间变化；
- 查询改写、HyDE、上下文压缩、Self-RAG 等独立实验；
- 本地交互式 Generator 对照；
- GraphRAG 及 vector-only / graph-only / hybrid 对照。

这些都不应抢在带引用、可拒答的基础问答闭环之前实现。

## 推荐的新会话起点

新会话建议直接以“设计并实现本地 Reranker”为第一项任务。开始时先按 `AGENTS.md` 阅读：

1. `README.md`；
2. `docs/design/architecture.md`；
3. `docs/design/chunking-and-indexing.md`；
4. `docs/design/roadmap.md`；
5. `docs/design/technology-stack.md`；
6. 本交接文档；
7. `docs/specs/2026-08-27-dense-bm25-rrf-retrieval-design.md`；
8. `docs/plans/2026-08-27-dense-bm25-rrf-retrieval-implementation.md`。

可以给新会话的首条任务：

> 读取 `docs/handoffs/2026-08-28-next-steps-handoff.md` 和项目要求的设计文档。先讨论并设计
> P0 本地 Reranker，基线考虑 `BAAI/bge-reranker-v2-m3`；确认设计后直接实现、测试，并在
> 当前冻结 pilot 上与 Hybrid 基线做同快照对照。不要提前实现 Generator 或前端。

## 已知坑与约束

- RRF 候选并非越深越好：在当前 34 块语料中，20/20 会让两路中等排名的噪声共同加分，
  反而把单路首位的正确块挤出 Top-5；5/5 明显更好。
- 检索参数必须进入 Index Profile 或 Evaluation Run 身份。此前遗漏候选深度和 `rrf_k` 曾导致
  旧 Build 被错误复用。
- 金标准证据必须绑定文章版本和原子单元范围，不能绑定可能随分块策略变化的 TopicChunk ID。
- `active` 不能同时代表最新来源版本和当前服务版本；新版本审核/建库期间旧版本继续服务是
  正常状态。
- BGE 当前最大输入长度为 512 tokens，长话题块会截断；实现 Reranker 和上下文组装时要显式
  记录并测试各模型的截断策略。
- Qdrant local storage 会被进程锁定；同一索引路径的评测不要并行运行。
- Windows 下模型下载、代理、TLS 和 Hugging Face 缓存曾出现不稳定；当前 BGE 权重已在本地
  缓存，不要无故重新下载或删除用户手工下载文件。
- 真实付费语料、审核草案、数据库和索引都在 Git 忽略的 `data/` 或外部目录中，禁止提交。
- 工作区可能包含用户改动；提交时精确列文件，不使用宽泛清理、reset 或 checkout。

## 暂不处理

- 不为未来可能性建设动态插件平台、工作流引擎或复杂依赖注入；
- 不把本地离线分块模型注册成第一阶段交互式 Generator；
- 不把 contextual text、摘要或自动推断伪装成作者原话；
- 不使用现有 pilot holdout 调参，因为当前根本没有 holdout；
- 不在没有正式 benchmark 的情况下因为单次分数高就宣布某模型或参数普遍更好。
