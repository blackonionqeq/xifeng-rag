# 分层子块检索与相邻上下文设计

日期：2026-08-29  
状态：已确认，待实施

## 目标

解决“相关内容确实存在于语料，但被较大的 TopicChunk 稀释，召回或重排未能识别”的问题。
第一版在现有已审核话题块之下生成确定性子块，同时检索父块与子块，再统一以子块粒度重排；
最终命中可带上同一父块内的前后相邻子块，为 Generator 补足局部语境。

该方案不要求为每类问题维护关键词或人工概括，也不修改已经批准的分块清单。

## 非目标

本次不实现：

- RAPTOR 式递归摘要树或多层模型概括；
- 只沿父块命中向下搜索的严格“从大到小”路由；
- 查询改写、HyDE、外部 Reranker 或新的融合算法；
- 跨 TopicChunk 的邻居扩展；
- 用当前单个失败问题直接确定生产默认参数。

## 方案选择

采用简化的两层“分层合并”：

```text
已审核 TopicChunk（父块） -> 确定性 child window（子块）

query -> 父块 Hybrid Top-K -> 展开父块的全部子块 --+
                                                   +-> 去重 -> 本地 Reranker -> 子块 Top-K
query -> 子块 Hybrid Top-K ------------------------+
                                                                  |
                                                                  v
                                                    同父块前后各扩一块
```

不采用纯“从大到小”，因为父块一旦漏召回，其子块没有机会进入候选集；也不采用完整 RAPTOR，
因为当前语料规模不需要递归层级，而且模型摘要会引入额外成本、审核和可复现性问题。

父、子两路是同一次查询中的并行逻辑召回，不是先后依赖的两次搜索。两层原始分数不直接相加，
最终相关性由同一个本地 Reranker 在短子块文本上判断。

## 子块生成

子块只从 `include_in_index=true` 的已批准 TopicChunk 派生，并遵守以下规则：

- 只组合父块内部连续的原子单元，不切断普通原子单元，也不跨父块；
- 从父块首个单元开始贪心向后组合：未达到 400 字符时继续加入下一单元，但加入后超过
  600 字符则在下一单元前结束；达到 400 字符后立即结束当前子块；
- 单个原子单元自身超过 600 字符时单独成块；末尾子块不足 200 字符时，仅在合并后不超过
  600 字符的情况下并回前一子块；
- 子块之间默认不重叠，局部语境由查询后的前后串连提供；
- 算法必须确定性运行，相同批准清单和配置产生完全相同的边界与 ID。

字符数按现有规范原文渲染函数生成的字符串计数。稳定子块 ID 由父 `chunk_id`、原子单元
起止范围和 splitter 版本派生，不依赖绝对路径。
子块正文始终由规范原子单元范围还原；它是可重建派生数据，不产生新的人工审核状态，
也不改变不可变的 `ApprovedChunkManifest` 或 `TopicChunk`。

上述数值是首个实验 Profile 的起点，不是跨语料的通用结论。后续变更切分规则或参数必须创建
新的 Index Profile，不能覆盖旧实验。

## 两层召回与统一重排

首个实验配置使用：

- 父块 Hybrid Top-5；
- 子块 Hybrid Top-20；
- 父块命中展开为其全部子块；
- 按 `child_id` 去重后，使用现有本地 Cross-Encoder 统一重排；
- 最终返回子块 Top-5，再执行相邻扩展。

当前语料统计下，候选并集预计不超过约 80–90 个短子块。Profile 设置 128 个重排候选上限。
构建完成时，以“子块数最多的 `parent_candidate_k` 个父块所含子块总数，再加
`child_candidate_k`”计算不考虑去重的最坏候选数；超过上限则 Build 校验失败，不得在查询时
静默丢弃某个父块的后半部分。待正式 dev benchmark 建立后再比较召回深度和上限。

每个统一子块候选至少保留：

```text
child_id
parent_chunk_id
article_id
article_version
unit_range
original_text
layer_sources       # parent / child
parent_scores       # 父块 BM25、Dense、RRF 分数与排名，可缺省
child_scores        # 子块 BM25、Dense、RRF 分数与排名，可缺省
reranker_score
reranker_rank
```

父、子分数分别记录以便诊断，但不做跨粒度加权。父块命中只负责提名子块；Reranker 直接决定
最终顺序。分数并列时依次按子块原始候选顺序和 `child_id` 稳定排序。没有启用本地 Reranker
时，第一版分层模式应拒绝运行，避免引入未经验证的跨层排序规则。

## 前后串连与引用边界

Reranker 选出最终子块后，每个命中默认取同一 TopicChunk 内的前一子块和后一子块：

- 不跨父块，也不跨文章版本；
- 多个命中的扩展范围重叠或连续时合并，避免重复正文；
- 真正命中的范围记录为 `matched_ranges`；送给 Generator 的范围记录为 `expanded_range`；
- 合并多个命中时保留全部 `matched_ranges`，不能只留下合并后的大范围；
- 字符预算优先保证命中子块；预算不足时按确定性顺序移除最外侧邻居，不截断命中正文。

Generator 看到并引用的是扩展后的规范原文，引用仍包含文章、版本和原子单元或字符位置。
检索评测只把 `matched_ranges` 计作召回命中；邻居带来的支持范围另记为扩展覆盖，不能把偶然
串连到的答案算成检索器自身命中。

相邻扩展完全由本地确定性代码完成，不调用模型，不修改批准清单，也不把 AI 概括当作原文。

## Index Profile、存储与增量更新

分层方案创建独立、手动启用的实验 Index Profile。其身份除现有字段外至少包含：

```text
retrieval_granularity = parent-child-v1
child_splitter_version
child_target_chars
child_max_chars
child_tail_merge_chars
parent_candidate_k
child_candidate_k
layer_candidate_limit
parent_bm25 / parent_dense 配置
child_bm25 / child_dense 配置
```

Reranker 和相邻扩展仍是查询期配置，不因模型或邻居数量变化重建索引；但每次评测必须记录其
完整配置。父、子索引属于同一个 Build，不能分别切换活动版本。

SQLite 中的文章版本、批准清单、TopicChunk 和原子单元继续是结构化事实来源。Build 保存可重建
的子块清单、范围映射、BM25 与 Dense 产物；向量 payload 只保存稳定 ID 和短元数据，不复制正文。
查询命中后按稳定范围从 SQLite 还原规范原文。

增量更新遵守现有原子发布规则：

- 未变化文章复用对应子块与 Embedding；
- 新增或修改且已批准的文章版本生成新的父、子派生产物；
- 缺失文章先标记 inactive，不立即物理删除；
- 父 BM25、父 Dense、子 BM25、子 Dense、映射清单和一致性校验全部成功后才发布 Build；
- 任一步失败都保留旧活动 Build，不暴露半成品。

该 Profile 初期只能显式选择，不替换当前默认索引。只有冻结 dev benchmark 显示稳定收益且
工程成本可接受后，才另行决定是否设为默认。

## 实验与验收

在同一冻结语料快照、问题集、Embedding 和 Reranker 下比较：

```text
A. TopicChunk Hybrid + Reranker（现有基线）
B. child Hybrid + Reranker
C. parent + child 并行召回 + Reranker
D. C + 前后串连
```

A、B、C 用人工确认的原子单元支持范围统一计算候选召回、重排后 Recall@K、MRR、nDCG 和
evidence coverage。D 的检索指标与 C 使用相同 `matched_ranges`，另报扩展证据覆盖、最终上下文
字符数和预算淘汰情况。各组同时记录平均/P95 延迟、Reranker 候选数、截断数、GPU 峰值显存、
提升题、退化题和失败案例。

“作者的职业身份经历过怎样的变化？”可加入开发集，但必须先人工确认文章版本与支持范围；
不得只围绕这一题调切分或候选深度。当前 pilot 规模不足以决定默认切换，实验结果只能形成假设。

自动测试至少覆盖：

- 子块边界、尾块合并、超长单元和稳定 ID；
- 子块不跨父块，未批准内容不进入索引；
- 父、子两路提名、去重、分数来源保留和稳定重排；
- 候选上限明确校验，分层模式未启用 Reranker 时拒绝运行；
- 相邻扩展不跨父块，重叠范围正确合并，预算优先保留命中；
- `matched_ranges` 与 `expanded_range` 分离，引用只来自规范原文；
- 新增、修改、未变化、缺失和任一父子索引失败时的回滚；
- 普通日志、Build payload 和评测元数据不包含付费正文。

完成门槛是分层 Build 可复现、查询与引用边界正确、失败不静默降级、旧默认路径保持不变，
并能在冻结数据上如实比较 A/B/C/D；质量没有提升必须保留并报告，而不是调整金标准。
