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

- Markdown 原文的自动或批量处理、Embedding、向量索引和 BM25 索引必须位于本机；只有下述评测集候选生成允许显式选择少量完整文章外发。
- 默认使用本地 Reranker，避免将大量候选片段发送给外部服务。
- 第一阶段 Generator 使用外部 `deepseek-v4-flash`，仅接收最终选中的少量片段，并由用户显式触发。
- 本地 Generator 保留接口但暂不实现。
- 语义分块和概括会读取接近完整的文章，因此必须使用本地模型，并在用户确认后才进入索引。
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
  -> 本地 AI 生成单层话题分块与概括
  -> 用户审核并批准分块清单
  -> 构造待索引文本（原文 / AI 上下文化）
  -> 本地 Embedding 与 BM25 建库
  -> Dense + BM25 召回
  -> RRF 融合
  -> 本地 Reranker
  -> 话题块上下文组装
  -> DeepSeek V4 Flash Generator
  -> 带原文引用的回答
```

V0 先以已批准的 AI 话题块完成 BM25、引用和基础评测。V1 再加入 Dense、RRF、本地 Reranker 与 Generator；不提前抽象查询改写、HyDE、RAPTOR、压缩与路由等所有可能策略。

## 4. 必要扩展边界

只保留五个边界清晰的小接口：

```text
ChunkStrategy
EmbeddingTextBuilder
EmbeddingProvider
Reranker
Generator
```

- `ChunkStrategy` 决定如何从已编号段落生成和验证单层话题块；第一阶段实现本地 AI 语义分块。
- `EmbeddingTextBuilder` 决定原文、确定性元数据或模型生成上下文如何组成索引文本。
- `EmbeddingProvider` 只负责本地文本向量化。
- `Reranker` 对融合后的候选话题块排序，可关闭或替换模型。
- `Generator` 根据问题与最终证据生成结构化回答；第一阶段实现 DeepSeek 适配器，本地适配器后续增加。

不建设动态插件发现、注册表或复杂依赖注入系统。实现可使用普通 Python 接口和显式配置。

第一阶段构建原文和 AI 上下文化两种索引 profile。语义分块模型只在本地运行；其提案经用户批准后，话题标题和概括才可用于建库。

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

## 6. 数据分层

```text
Article
  -> ArticleVersion
    -> ChunkProposal
    -> ApprovedChunkManifest
      -> TopicChunk
        -> IndexArtifact
```

- `Article` 保存稳定文章 ID、标题、抓取时间和源文件。
- `ArticleVersion` 分别保存原始文件哈希、规范正文哈希、清洗规则版本和版本状态。
- `ChunkProposal` 保存本地模型提出的段落边界、话题标题、概括及生成版本。
- `ApprovedChunkManifest` 保存用户确认的不可变分块方案。
- `TopicChunk` 是唯一检索单位，保存段落范围、原文和检索辅助字段。
- `IndexArtifact` 保存某个索引 profile 的索引文本、Embedding 模型和构建状态。

结构化状态保存在本地 SQLite。向量库与全文索引均视为可以重建的派生产物。

## 7. Index Profile

索引构建方案由一个小型配置描述：

```yaml
name: approved_topic_context_qwen3
chunk_strategy: ai_topic_reviewed_v1
embedding_text: topic_context_v1
embedding_model: Qwen/Qwen3-Embedding-0.6B
```

不同 profile 使用独立索引，防止向量空间、切块方式或实验数据相互污染。Reranker 与 Generator 属于查询期配置，不要求重建索引。

## 8. 可观测性与可复现性

每次评测至少记录：

- 语料快照与文章内容哈希；
- Index Profile 与模型版本；
- Reranker、Generator 和 Prompt 版本；
- 代码版本；
- 检索、引用、回答质量指标；
- 索引耗时、查询延迟和资源占用。

检索参数、具体 API 评测模型和 UI 形态留到实现或对应阶段实测决定，不再阻塞 V0 开工。
