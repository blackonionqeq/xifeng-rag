# 总体架构设计

状态：设计基线 v0.2；本文只记录已经确认的设计决策。

## 1. 项目目标

系统首先是一个可日常使用的“作者观点研究台”，支持：

- 针对单篇文章提问并定位原文；
- 汇总作者关于某个主题的跨文章观点；
- 展示主张、论据、相关文章与时间信息；
- 保存可复现的评测结果，用于比较不同 RAG 方案；
- 后续升级为图增强 RAG，研究观点演变、关联和潜在矛盾。

第一阶段不建设通用插件平台，也不实现完整知识图谱。

## 2. 隐私边界

- Markdown 原文的自动或批量处理、Embedding、向量索引和 BM25 索引必须位于本机；只有单篇分块重试和下述评测集候选生成允许在逐次二次确认后外发清洗后的完整文章。
- 默认使用本地 Reranker，避免将大量候选片段发送给外部服务。
- 第一阶段 Generator 使用显式配置的 DeepSeek 模型，仅接收最终选中的少量片段，并由用户
  通过单轮问答命令显式触发；实际模型 ID 通过服务商模型列表核对，不在业务代码中永久绑定。
- 本地 Generator 保留接口但暂不实现。
- 语义分块和概括默认使用本地模型。少数不理想提案允许用户明确选择单篇文章、外部 provider、模型和一次性 Prompt，在显示外发范围并二次确认后生成并列提案；该路径不得批量调用或自动批准。
- 评测集候选可以由更强的外部 API 辅助生成。该例外只允许发送用户显式选择并二次确认的少量清洗后文章，不得成为自动批处理路径。
- 日志不得记录 API 密钥，也不得默认保存发送给外部模型的完整上下文。

## 3. 第一阶段主流程

语料位于本项目之外。当前开发环境使用：

```text
../test-reverse-wechat-official-accounts-paid-article/saved_articles/*.md
```

该位置由 `corpus.source_dir` 配置，不在代码中绑定绝对路径。源目录只读，项目只将派生数据写入自身的数据目录。

```text
saved_articles/*.md
  -> 确定性清洗、规范化与版本识别
  -> 默认本地 AI 生成单层话题分块与概括（可选单篇外部重试）
  -> 用户通过本地审核页或 JSON 审核并批准分块清单
  -> 构造待索引文本（原文 / AI 上下文化）
  -> 本地 Embedding 与 BM25 建库
  -> Dense + BM25 召回
  -> RRF 融合
  -> 本地 Reranker
  -> 话题块上下文组装
  -> DeepSeek V4 Flash Generator
  -> 带原文引用的回答
```

V0 先以已批准的 AI 话题块完成 BM25、引用和基础评测。V1 再加入 Dense、RRF、本地 Reranker 与 Generator；不提前抽象 HyDE、RAPTOR、压缩与路由等所有可能策略。已确认的保守查询改写实验使用普通、显式的 `QueryRewriter` 接口，详见[查询改写设计](../specs/2026-08-29-conservative-query-rewrite-design.md)。

## 4. 必要扩展边界

只保留五个边界清晰的小接口：

```text
ChunkStrategy
EmbeddingTextBuilder
EmbeddingProvider
Reranker
Generator
```

- `ChunkStrategy` 决定如何从带精确源位置的已编号语义原子单元生成和验证单层话题块；第一阶段默认使用本地 AI，并通过普通 Python 结构化生成适配器支持 OpenAI-compatible API 与隔离的 Codex CLI 单篇重试，不建设模型插件平台。
- `EmbeddingTextBuilder` 决定原文、确定性元数据或模型生成上下文如何组成索引文本。
- `EmbeddingProvider` 只负责本地文本向量化。
- `Reranker` 对融合后的候选话题块排序，可关闭或替换模型。
- `Generator` 根据问题与最终证据生成结构化回答；第一阶段实现 DeepSeek 适配器，本地适配器后续增加。

第一版 Generator 输出 `answered`、`partially_answered` 或 `insufficient_evidence`，并把资料明确
陈述与基于资料的推断分开。每个语料结论段必须绑定本次上下文中的证据 ID；本地程序校验 ID
并确定性渲染引用。可选的模型补充分析独立标注，不能引用语料或引入新的具体事实。详细协议见
[`docs/specs/2026-08-28-deepseek-generator-design.md`](../specs/2026-08-28-deepseek-generator-design.md)。

不建设动态插件发现、注册表或复杂依赖注入系统。实现可使用普通 Python 接口和显式配置。

第一阶段构建原文和 AI 上下文化两种索引 profile。语义分块默认只在本地运行；任何本地或外部提案都必须经过相同的本地结构校验和人工批准，话题标题与概括才可用于建库。

人工审核页是 CLI 启动的单用户本地工具，只绑定回环地址。它可以调整边界、检索辅助字段
和 `include_in_index`，先保存独立草案，再由用户二次确认批准不可变清单并原子更新活动
索引。索引更新失败时旧索引保持可用且可重试。页面不包含外部模型调用，因此不改变阶段
2 的通用 API 与阶段 3 的正式前端边界。

## 5. 统一检索结果

所有检索与未来图检索都输出统一候选结构：

```text
chunk_id
article_id
article_version
original_text
paragraph_range
retrieval_scores
source
```

其中 `source` 标识 dense、bm25、graph 等候选来源。未来 GraphRAG 只需产生相同结构的候选，再进入现有融合、重排和生成流程。

## 6. 确定性上下文组装

Reranker 之后由普通 Python `ContextAssembler` 将排序候选转换为结构化证据上下文。默认路径
保持完整 TopicChunk；手动启用的分层实验路径接收已完成同父块相邻扩展的子块候选。组装器只
执行完全重复去重、数量/字符预算、查询内引用编号、来源追踪、
排除原因和规范 JSON 渲染。默认最多 5 块、总渲染上限 50,000 字符；不做语义判断、块内
截断、文章配额、重新检索或网络调用。

AI 话题标题和概括保留为检索辅助元信息，默认不进入 Generator 上下文；即使显式开启，也
必须带 AI 生成标签且不能作为作者原话引用。完整渲染上下文只存在于内存，不在普通日志或
数据库复制正文。未来 RetrievalController 或抽取式 Compressor 插在 Reranker 与组装器之间，
组装器本身继续保持确定性。详细协议见
[`docs/specs/2026-08-28-deterministic-context-assembly-design.md`](../specs/2026-08-28-deterministic-context-assembly-design.md)。

## 7. 数据分层

```text
Article
  -> ArticleVersion
    -> ChunkProposal
    -> ApprovedChunkManifest
      -> TopicChunk
        -> IndexArtifact
```

- `Article` 保存稳定文章 ID、标题、源文件、抓取时间、文件修改时间和来源统计。管理页直接
  聚合现有结构化表，不维护第二份文章状态表。
- `ArticleVersion` 分别保存原始文件哈希、规范正文哈希、清洗规则版本和版本状态。
- `ChunkProposal` 保存本地模型提出的单元边界、内容角色、话题标题、概括及生成版本。
- `ApprovedChunkManifest` 保存用户确认的完整不可变分块方案，包括不进入检索的样板内容。
- `TopicChunk` 是人工确认的语义父块，也是默认检索单位；分层实验可在 Build 内从其原子单元
  范围确定性派生子块。子块不产生新的审核状态，也不改变批准清单。
- `IndexArtifact` 保存某个索引 profile 的索引文本、Embedding 模型和构建状态。
- `EvaluationDataset` 保存人工问题、审核后的参考答案与稳定证据范围；冻结时绑定完整语料
  快照与内容哈希，AI 候选只作为可审计草案存在。

结构化状态保存在本地 SQLite。向量库与全文索引均视为可以重建的派生产物。

评测集策展使用独立的本地工作台。问题由人工录入；Codex CLI 只在用户针对单篇文章查看
披露并逐次确认后生成候选答案、答案要点与证据范围。候选必须经人工修改和确认，评测运行
阶段不调用任何外部模型。金标准锚定文章版本和原子单元范围，不锚定易变的 TopicChunk ID。

本地语料管理页明确区分最新来源版本与当前索引版本。新正文完成审核和索引前，旧版本继续
服务是增量更新的正常状态，不应被简化为单个含义不清的“当前版本”。

## 8. Index Profile

索引构建方案由一个小型配置描述：

```yaml
name: approved_topic_context_qwen3
chunk_strategy: ai_topic_reviewed_v1
embedding_text: topic_context_v1
embedding_model: Qwen/Qwen3-Embedding-0.6B
```

不同 profile 使用独立索引，防止向量空间、切块方式或实验数据相互污染。Reranker 与 Generator 属于查询期配置，不要求重建索引。

父子分层 Profile 以非活动 `ready` 状态发布，必须通过 Build ID 显式查询，不替换默认活动
索引。详细协议见
[`docs/specs/2026-08-29-layered-child-retrieval-design.md`](../specs/2026-08-29-layered-child-retrieval-design.md)。

## 9. 可观测性与可复现性

每次评测至少记录：

- 语料快照与文章内容哈希；
- Index Profile 与模型版本；
- Reranker、Generator 和 Prompt 版本；
- 代码版本；
- 检索、引用、回答质量指标；
- 索引耗时、查询延迟和资源占用。

检索参数、具体 API 评测模型和 UI 形态留到实现或对应阶段实测决定，不再阻塞 V0 开工。
