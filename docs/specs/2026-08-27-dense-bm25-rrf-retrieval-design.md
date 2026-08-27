# BGE Dense、BM25 与 RRF 混合检索设计

日期：2026-08-27
状态：已确认

## 目标

在现有已批准话题块与原文 BM25 基线上增加本地 Dense 检索，并通过自有 RRF 将两路排名
融合。第一轮只改变召回方式，不加入 contextual text 或 Reranker，以冻结的 pilot 数据集
分别比较 BM25、Dense 和 Hybrid，形成可复现的 V1 检索基线。

首个 Dense 模型使用 `BAAI/bge-base-zh-v1.5`。模型只在本地完成文档与查询向量化；向量、
模型缓存和索引均为本地派生产物，不进入 Git。

## 非目标

本次不实现：

- AI contextual text 的向量索引；
- Reranker；
- Generator 或回答组装；
- Qdrant Server、Qdrant Cloud 或远程 Embedding API；
- 动态模型插件发现；
- 使用正式 benchmark 的 holdout 调参。

## Index Profile 与实验边界

新增不可变混合 Index Profile，其有效配置至少包含：

```text
chunk_strategy_version
embedding_text_builder = original-text-v1
embedding_provider = sentence-transformers
embedding_model_id = BAAI/bge-base-zh-v1.5
embedding_model_revision
embedding_dimension
query_instruction
normalization = true
dense_distance = cosine
bm25_text_builder = original-text-v1
bm25_tokenizer = jieba-search-v1
fusion_algorithm = rrf-v1
bm25_candidate_k = 20
dense_candidate_k = 20
rrf_k = 60
```

Profile 只描述可复用的检索方案；批准清单 ID、文章版本和规范正文哈希属于具体 Index Build
的语料快照，不参与 Profile 身份。

不同模型、revision、维度、查询指令、索引文本构造方式或 BM25 分词方案必须创建新的
Profile 和 Index Build，不得覆盖旧实验。旧的纯 BM25 Build 保留并可显式用于评测。

活动 Hybrid Build 只在全部派生产物成功发布后切换。评测命令仍可指定非活动但与冻结语料
快照兼容的 Build，以复现 BM25 baseline。

## 配置

新增简单、显式的 Embedding 和检索配置，不建设注册表：

```yaml
embedding:
  provider: sentence-transformers
  model_id: BAAI/bge-base-zh-v1.5
  model_revision: null
  model_cache_dir: null
  device: auto
  batch_size: 16
  normalize_embeddings: true
  query_instruction: 为这个句子生成表示以用于检索相关文章：

retrieval:
  bm25_candidate_k: 20
  dense_candidate_k: 20
  rrf_k: 60
```

仓库示例不得绑定本机绝对路径。本机 `config.local.yaml` 将 `model_cache_dir` 设置为
`D:/ai-models/huggingface`。`device=auto` 允许运行库选择可用设备；最终解析出的设备必须写入
Build 元数据。批大小是运行参数，可以在不改变向量语义时调整，但应写入构建诊断。

首次模型下载是用户明确执行的安装或建库准备步骤。下载完成后，建库、检索和评测必须可在
不发送正文的本地路径上运行。模型下载失败时不得留下可见的半成品 Build。
`model_revision` 未显式指定时，Provider 必须解析并记录实际加载的 revision；Profile 身份和
Build 清单使用解析后的值，不能只记录易漂移的模型别名。

## EmbeddingProvider

保留架构中约定的窄接口：

```text
encode_documents(texts) -> normalized vectors
encode_queries(texts) -> normalized vectors
dimension -> int
model_metadata -> immutable metadata
close() -> None
```

第一版实现 `SentenceTransformersEmbeddingProvider`：

- 文档使用 `encode_document()`；
- 查询使用 `encode_query()`；
- BGE 查询侧采用版本化的官方中文检索 instruction，文档侧不添加该 instruction；
- 文档和查询都输出归一化向量；
- 相似度使用 cosine；
- 向量维度从已加载模型读取并校验，不在代码中写死；
- 批量编码，禁止逐块重复加载模型；
- 关闭或释放运行资源后再发布索引目录。

测试使用确定性的 Fake Embedding Provider，不下载模型、不访问网络，也不依赖 GPU。

## Dense 存储

使用 Qdrant Client 的本地持久化模式。每个 Index Build 使用自己目录下的独立 Qdrant
storage 和 collection；不同向量维度永不共用 collection。

每个 point 使用由稳定 `chunk_id` 确定性派生的 UUID 作为 Qdrant point ID，vector 为归一化
Dense 向量。原始 `chunk_id` 保存在 payload，payload 只保存检索与一致性校验所需的稳定
标识和短元数据，例如：

```text
chunk_id
article_id
article_version
manifest_id
start_paragraph
end_paragraph
```

payload 不复制 `original_text`、contextual text 或完整提示词。命中后根据 `chunk_id` 从
SQLite 的规范记录装配原文候选与引用。

Qdrant 本地目录有独占文件锁。CLI 命令只在操作期间持有单个客户端；完成查询或建库后必须
显式关闭。若未来需要多个长期进程并发访问，再单独设计 Qdrant Server，不在本次预建。

## 原子建库

一次 Hybrid Build 按以下顺序执行：

1. 校验语料目录、活动文章版本和批准清单，输出解析后的语料路径与文章数量；
2. 固化 Build 的语料成员、Profile 配置和预期产物；
3. 在 Build 临时目录构建原文 BM25；
4. 单次加载 BGE，批量编码所有 `include_in_index=true` 的话题块原文；
5. 在临时 Qdrant 目录创建 cosine collection 并批量写入 point；
6. 核对 BM25 文档集合、Dense point 集合与数据库 TopicChunk 集合完全一致；
7. 关闭 BM25/Qdrant 资源，写入完成清单与校验和；
8. 原子发布 Build 目录，并在数据库事务中切换活动 Build。

任一步失败都把 Build 标记为失败并保存不含正文的诊断。旧活动 Build 继续可用，临时产物不
得被搜索路径发现；数据库不得出现“已经切换但某一路索引未完成”的状态。

## 检索与 RRF

一次 Hybrid 查询包含两条相互独立的召回路径：

```text
query -> jieba search mode -> BM25 Top-20 -----+
                                                +-> RRF -> final Top-K
query -> BGE query vector -> Dense Top-20 ------+
```

默认参数为：

```text
bm25_candidate_k = 20
dense_candidate_k = 20
rrf_k = 60
final_top_k = 5
```

RRF 不混合不可比的 BM25 与 cosine 原始分数。对每个检索器返回的 1-based rank，按
`1 / (rrf_k + rank)` 计分，同一 `chunk_id` 的两路贡献相加。最终按以下稳定规则排序：

1. RRF 分数降序；
2. 最佳单路 rank 升序；
3. `chunk_id` 升序。

候选结构扩展为允许各阶段分数缺省，并保存：

```text
bm25_score
bm25_rank
dense_score
dense_rank
rrf_score
sources
```

`sources` 为 `bm25`、`dense` 或两者。所有模式最终都归一到同一个 RetrievalCandidate，原文
引用继续只来自 SQLite 中的 `original_text`。

查询接口支持 `bm25`、`dense`、`hybrid` 三种显式模式。`hybrid` 为新 Profile 的默认模式；
评测必须记录实际模式、候选深度和最终 Top-K。请求的最终 Top-K 大于候选深度时应明确校验
或提升内部候选深度，不能静默少返回。

## 评测

使用现有冻结 pilot 数据集和完全相同的语料快照运行三组实验：

1. 旧 Build 的 BM25 only；
2. 新 Build 的 BGE Dense only；
3. 新 Build 的 BGE Dense + BM25 + RRF。

继续报告 Precision@5、Recall@5、MRR、nDCG@5、evidence coverage@5、平均延迟和 P95，并
保存逐题排名。重点检查现有 BM25 将相关块排到 15、21、22 及 7、14 名的两个失败案例，
但不把“必须提升”作为实现正确性的替代标准；若 Dense 或 Hybrid 退化，应如实记录。

pilot 只用于验证实现和形成假设。后续参数调整只能使用正式 benchmark 的 dev 部分，不能
查看或使用 holdout 调参。

## 可观测性与隐私

Build 元数据和评测运行记录：模型 ID/revision、维度、查询 instruction 版本、归一化设置、
解析设备、构建耗时、查询模式、两路候选深度、RRF 参数、代码版本和资源统计。

普通日志不得包含：

- 付费正文或 Qdrant payload 中的正文副本；
- 完整查询向量或文档向量；
- 完整提示上下文；
- API 密钥或环境变量值。

Embedding、Qdrant、BM25 与评测日志均在本地完成，不调用远程推理服务。模型权重、缓存、
Qdrant storage 和 Build 目录继续由 Git 忽略。

## 验证要求

自动测试至少覆盖：

- Fake Provider 的文档/查询编码路径与维度校验；
- Dense 排序、Top-K 和稳定并列顺序；
- RRF 公式、两路去重、单路缺失和稳定排序；
- BM25、Dense、Hybrid 三种模式映射到统一候选；
- Qdrant point 与 TopicChunk 集合一致性；
- 模型加载、编码、Qdrant 写入和发布失败时的回滚；
- 不同模型或维度创建不同 Profile/Build；
- 默认测试不联网且日志不包含正文；
- 冻结快照对三种模式的兼容校验。

本机验收包括：显式下载 BGE、构建新的 Hybrid Build、分别运行三种模式、执行冻结 pilot
评测并保存逐题对比。完成门槛是实现与回滚正确、实验可复现；检索指标变化作为实验结果，
不通过修改金标准或隐藏退化来满足门槛。
