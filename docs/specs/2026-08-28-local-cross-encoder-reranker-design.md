# 本地 Cross-Encoder 重排设计

日期：2026-08-28  
状态：已实现并完成本机 pilot 验收

## 目标

在现有 `BM25 + Dense + RRF` 混合召回基线上增加可显式启用的本地
Cross-Encoder Reranker。混合召回先保留较宽候选集，Reranker 再根据查询与规范原文的
相关性排序并返回最终 Top-K。

第一版使用 `BAAI/bge-reranker-v2-m3`，全程在本地推理，不向外部服务发送查询或付费正文。
实现必须保留各召回阶段和重排阶段的分数，支持在同一冻结评测集上公平比较 Hybrid 与
Hybrid + Reranker，并以本机实测决定候选深度、输入长度和模型资源是否可接受。

## 非目标

本次不实现：

- 时间加权、时间衰减或按发布时间默认提升排名；
- ColBERT 索引或 Token 级后期交互检索；
- LLM/API Reranker；
- Generator、上下文组装、拒答协议或正式前端；
- AI 摘要、关键词或 contextual text 参与重排输入；
- 滑动窗口、分段最大值等长文本聚合策略；
- 动态 Reranker 插件发现或完整检索工作流引擎。

## 方案比较

### 方案 A：现有 `search()` 接受可选 Reranker（采用）

给当前统一检索入口增加可选 `Reranker` 与重排候选深度。未传 Reranker 时沿用现有行为；
传入时先装配较宽的统一候选，再批量打分、稳定排序并截取最终 Top-K。

该方案改动集中，BM25、Dense、RRF 的已有职责保持不变，CLI 与冻结评测也能复用同一个
查询入口。

### 方案 B：新增 `reranked_search()` 外层包装

包装现有 `search()` 可以让重排表面上更独立，但会重复传递候选深度、索引 Build、Embedding
Provider 和评测元数据，并容易让两个入口的验证规则发生漂移，因此不采用。

### 方案 C：新建完整 Retrieval Pipeline 类

完整 Pipeline 便于以后加入查询改写、路由、压缩等步骤，但当前只有一个已确认变化点。
提前引入编排层会扩大抽象和测试范围，因此不采用。

## 架构与数据流

第一版只允许 Hybrid 模式启用重排：

```text
query
  -> BM25 Top-N --------+
                         +-> RRF -> Top candidate_k
  -> Dense Top-N -------+              |
                                        v
                         local Cross-Encoder
                                        |
                                        v
                              stable final Top-K
```

当启用重排时，BM25 与 Dense 的实际查询深度都至少等于 `candidate_k`，RRF 再保留
`candidate_k` 个去重候选。本机 pilot 验收后默认 `candidate_k=5`。没有启用重排时继续使用当前
`bm25_candidate_k=5`、`dense_candidate_k=5`、`rrf_k=60` 和最终 Top-K 行为，不改变已有
Hybrid 基线。

第一版不允许 BM25-only 或 Dense-only 模式启用重排。这样“Hybrid + Reranker”具有单一、
清晰的实验含义；单路召回加重排可在基线完成后另行设计。

## Reranker 接口与生产实现

使用普通 Python `Protocol` 定义窄接口。接口接收查询和已经装配完成的统一候选，返回每个
候选对应的重排分数与截断诊断。测试使用确定性的 Fake Reranker，不下载模型、不访问网络、
不依赖 GPU。

生产实现使用 `sentence-transformers` 5.x 的 `CrossEncoder`：

- 使用关键字参数设置模型、revision、缓存目录、设备和最大输入长度；
- 使用 `predict()` 批量计算单标签相关性分数；
- 排序只依赖模型原始输出的相对次序，不把分数强制转换成概率；
- 每条 CLI 命令最多加载一次模型并复用；
- 模型加载和推理都必须在本机完成；
- 实际解析出的模型 revision、设备和输入上限进入运行元数据。

模型加载或推理失败时命令直接失败，不静默退回 RRF。否则一次标记为“启用重排”的评测
可能实际没有经过重排，破坏实验身份和可复现性。

## 重排输入

每个候选形成一对 Cross-Encoder 输入：

```text
第一段：<query>
第二段：文章：《<article_title>》\n<original_text>
```

输入模板版本固定为 `article-title-original-text-v1`。第一版不使用 `topic_title`、
`context_summary`、关键词或 contextual text，避免模型根据 AI 生成信息提升排名，却不能提供
同等强度的作者原文证据。

`max_pair_tokens` 表示一对输入的 token 总预算，而不是字符数，也不是所有候选共享的预算：

```text
query tokens + title/original_text tokens + special tokens <= max_pair_tokens
```

默认 `max_pair_tokens=1024`。查询作为第一段，文章标题和正文作为第二段；超限时保留完整的
短查询和位于第二段开头的标题，截断正文尾部。程序在送入模型前检测每对输入是否超过预算，
并将截断事实写入候选诊断和评测详情。

1024 是项目的初始运行预算，不代表模型架构的硬上限。滑动窗口会同时改变质量、延迟和分数
聚合方式，因此不在第一版暗中加入；若实测发现关键证据经常位于截断部分，再作为独立实验
比较更大的预算或窗口策略。

## 排序与统一候选

`RetrievalScores` 增加：

```text
reranker_score
reranker_rank
reranker_truncated
```

Reranker 直接决定最终顺序，不与 BM25、Dense 或 RRF 原始分数做加权混合。稳定排序规则为：

1. `reranker_score` 降序；
2. 原 RRF rank 升序；
3. `chunk_id` 升序。

BM25、Dense、RRF 的分数和排名继续保留。这样既不混合量纲不同的分数，也能在 CLI、评测
详情和后续问答调试中解释候选经历的每个阶段。

空候选直接返回空列表，不调用模型。模型输出数量必须与输入候选数量一致，分数必须为有限
数值；否则视为推理失败。任何索引引用缺失、模型输出非法或候选映射不一致都必须明确报错。

## 配置与显式启用

新增独立的查询期配置：

```yaml
reranker:
  provider: sentence-transformers
  model_id: BAAI/bge-reranker-v2-m3
  model_revision: null
  model_cache_dir: null
  device: auto
  batch_size: 8
  max_pair_tokens: 1024
  candidate_k: 5
```

示例配置不绑定本机绝对路径。本机可在 Git 忽略的 `config.local.yaml` 中设置模型缓存目录。
Reranker 属于查询期组件，不写入 Index Profile，也不触发索引重建。

CLI 增加显式 `--rerank`：

```text
xifeng search "问题" --mode hybrid --rerank
xifeng eval run <FROZEN_DATASET_ID> --mode hybrid --rerank
```

两条命令均支持 `--rerank-candidate-k N` 覆盖单次运行的候选深度；覆盖值必须进入冻结评测
元数据，不能改变 Index Profile。

没有 `--rerank` 时不加载模型，保持现有输出和延迟特征。显式启用但模式不是 Hybrid 时立即
拒绝，并给出清晰错误。候选深度和输入预算可以通过本地配置调整；每次评测保存解析后的
实际值，不能只记录默认值。

## 运行与评测元数据

冻结评测运行至少保存：

```text
reranker_enabled
reranker_provider
reranker_model_id
reranker_model_revision
reranker_device
reranker_batch_size
reranker_max_pair_tokens
reranker_candidate_k
reranker_input_template_version
reranker_load_duration_ms
reranker_truncated_candidate_count
```

逐题详情继续保存候选 ID、来源和各阶段分数，并增加每个候选的 Reranker 分数、排名和截断
标志。普通日志、运行元数据和错误信息不得写入查询全文、候选正文、完整模型输入或向量。

首次本机验收另外记录平均/P95 查询延迟、PyTorch CUDA 峰值显存；没有 CUDA 时记录实际
设备和进程内存增量。资源统计只用于工程判断，不参与排序。

## 自动测试

自动测试不得下载或加载真实模型，至少覆盖：

- 未开启重排时结果与现有 Hybrid 行为完全一致；
- RRF 先取宽候选，Fake Reranker 能改变最终 Top-K；
- 分数并列时按 RRF rank 和 `chunk_id` 稳定排序；
- Reranker 分数、排名和截断标志进入统一候选；
- 空候选不调用模型；
- 超长 pair 能确定性标记截断；
- 模型加载、批量推理、非法分数和输出数量不匹配时明确失败；
- BM25-only、Dense-only 与未显式启用场景不会误用 Reranker；
- CLI 文本与 JSON 输出包含正确的最终顺序和调试分数；
- 冻结评测保存完整 Reranker 配置和逐题详情；
- 新数据库迁移可升级现有数据库，旧评测记录仍可读取；
- 普通日志和异常诊断不包含测试夹具正文。

继续执行全部 pytest、Ruff 和 Pyright，防止重排改动破坏已有语料、分块、索引和评测流程。

## 本机验收与实验

在同一个冻结 pilot 和兼容的 Index Build 上依次运行：

```text
A. Hybrid，无 Reranker
B. Hybrid + Reranker，candidate_k=5
C. Hybrid + Reranker，candidate_k=10
D. Hybrid + Reranker，candidate_k=20
E. Hybrid + Reranker，candidate_k=30
```

每组报告 P@5、Recall@5、MRR、nDCG@5、evidence coverage@5、平均/P95 查询延迟、模型加载
时间、截断候选数和资源占用，并检查逐题提升与退化。当前只有 6 道 pilot 题，结果只能验证
工作流和形成假设，不能宣称模型或参数具有普遍优势，也不能将 pilot 原地当作正式 holdout。

质量没有提升不等于实现失败，但必须如实保存结果。完成门槛是：本地推理路径正确、排序稳定、
失败不静默回退、实验身份完整、隐私边界未被破坏，并能通过同快照对照判断工程成本。

## 2026-08-28 本机验收结果

硬件为 RTX 4060 Laptop GPU 8 GB，PyTorch `2.13.0+cu130`，最终 Top-K 为 5。冻结 pilot
只有 6 题，以下结果用于选择开发默认值，不代表普遍结论：

| 配置 | P@5 | Recall@5 | MRR | nDCG@5 | Evidence coverage | 平均延迟 | P95 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid | 0.2333 | 0.8611 | 0.8333 | 0.8107 | 0.9044 | 518 ms | 1216 ms | — |
| Rerank k=5, 1024 | 0.2333 | 0.8611 | 0.8889 | 0.8453 | 0.9044 | 1356 ms | 1694 ms | 2.93 GB |
| Rerank k=10, 1024 | 0.2000 | 0.7778 | 0.8333 | 0.7942 | 0.7568 | 1663 ms | 2127 ms | 3.07 GB |
| Rerank k=20, 1024 | 0.2667 | 0.9167 | 0.8750 | 0.8773 | 0.8524 | 2694 ms | 3315 ms | 3.07 GB |
| Rerank k=30, 1024 | 0.2667 | 0.9167 | 0.8750 | 0.8773 | 0.8524 | 3912 ms | 4564 ms | 3.07 GB |
| Rerank k=5, 2048 | 0.2333 | 0.8611 | 0.8889 | 0.8453 | 0.9044 | 1873 ms | 2331 ms | 2.88 GB |
| Rerank k=20, 2048 | 0.2667 | 0.9167 | 1.0000 | 0.9301 | 0.8524 | 5686 ms | 6421 ms | 2.88 GB |

`candidate_k=10` 在一题上把全部相关证据移出 Top-5；`candidate_k=20/30` 虽提高块级 Recall
与 nDCG，但让另一题的稳定证据覆盖明显退化，且 30 不再提升质量。2048 tokens 改善排序指标，
却没有恢复证据覆盖，延迟显著增加。因此开发默认值采用 `candidate_k=5`、
`max_pair_tokens=1024`、`batch_size=8`：只重排现有 Top-5，保持召回与证据成员不变，同时改善
首个相关结果和整体顺序。扩大正式 dev benchmark 后再重新比较宽候选和长输入策略。
