# 正查细话题召回通道设计

状态：已确认，2026-09-03。

## 1. 目标

正向证据检索评测（`xifeng eval run`，问题 → 证据话题块）目前只使用话题块自身的
BM25/Dense 通道。文章反查实验已经证明"问题命中细话题 → 定位所属文章"的召回价值；
本设计把同一思路引入正向检索：问题命中已批准细话题后，提名其父 TopicChunk 进入
候选池，与常规通道融合，检验 recall 与 evidence coverage 能否提升。

细话题在正查中的定位沿用交接文档的约束：只是召回辅助元数据，不成为独立证据块，
不参与引用；引用仍只能来自话题块的 `original_text`。本通道仅挂在 `eval run --subtopics`
实验链路上，不改变生产 `search` 命令的默认行为。

## 2. 检索链路

```text
运行开始（一次性）:
  读取话题块 Build 的 BM25 corpus           -> 候选宇宙 chunk_id 集合
  从分层套件解析 subtopics 层 Build         -> 单元清单与索引
  构建细话题 -> 父 chunk 映射:
    approved SubtopicSetRow (每 version 取最新)
    -> (version_id, parent_start_unit, parent_end_unit)
    == TopicChunkRow (version_id, start_paragraph, end_paragraph)
    父块不在候选宇宙内的条目丢弃并计数

每题:
  主通道: 话题块 Build 上的 BM25 + Dense（现状不变）
  细话题通道: subtopics 层 BM25 + Dense
    -> 单元命中查映射 -> 父 chunk 提名（宇宙外命中丢弃计数）
  四路加权 RRF: bm25=1.0, dense=1.0, subtopic-bm25=w, subtopic-dense=w
  -> 候选装配 -> Reranker 重排 -> top_k
```

单变量保证：候选宇宙仍是冻结 Build 的全部话题块，细话题只改变进入重排的候选池
构成；重排输入深度与指标计算方式与基线完全一致。

## 3. 关键设计点

### 3.1 映射与对齐键

批准集合的 `(parent_start_unit, parent_end_unit)` 由 `validate_for_manifest` 保证与
已批准 `include_in_index` 话题的段落范围精确一致，因此 `(version_id, 起止单元)` 即可
唯一 join 到 `TopicChunkRow`，不需要文本对齐。单元与批准条目的对齐键为
`(version_id, parent_topic, title, start_unit, end_unit)`。

同一 version 存在多条 approved 记录时按 `created_at` 取最新；superseded 集合整体忽略。
套件有单元但映射为空时直接报错，提示检查套件与 `--build` 是否匹配或重建套件。

### 3.2 N 路加权 RRF

`weighted_reciprocal_rank_fusion` 泛化现有两路 `reciprocal_rank_fusion`（保留不动）：
`rrf = Σ weight_channel / (rrf_k + rank_channel)`，排序键 `(-rrf, 通道内最小 rank,
chunk_id)`，等权双通道时与旧实现排序一致（有回归测试锁定）。通道权重默认 1.0；
细话题两路合计权重与主通道两路合计相同，等权下天然平衡。

### 3.3 候选深度

`subtopic_candidate_k` 默认 10。分层套件通常覆盖比冻结 Build 更多的文章，单元命中会
有一部分落在候选宇宙外被丢弃，深度取 10 以保证宇宙内平均仍有足量有效提名，同时不
让细话题通道主导融合排序。融合后仍截 `max(top_k, rerank_candidate_k)` 送入重排。

### 3.4 溯源与审计

- `RetrievalScores` 新增 `subtopic_bm25(_rank)` / `subtopic_dense(_rank)`，默认
  `None`，向后兼容；
- 候选 `source` 在被细话题通道提名时追加 `subtopic-bm25` / `subtopic-dense`；
- 每题 detail 新增 `subtopic_recall`（提名 chunk、提名细话题标题、丢弃命中数）；
- run 级 outcome 新增 `subtopics`（build_id、语料快照、套件单元数、映射/丢弃计数、
  覆盖块数与文章数、candidate_k、weight、rrf_k）。

`EvaluationRunRow` 表结构不变，审计信息全部进 details/outcome，与查询改写先例一致。

### 3.5 互斥约束

第一版 `--subtopics` 要求显式 `--build`、`--mode hybrid`、`--rerank`，且与
`--layered`、`--rewrite` 互斥，避免多实验通道叠加破坏单变量归因。

## 4. 隐私与安全

全链路本地：细话题层索引、Embedding、重排均在本地运行，无外部调用。细话题标题只
出现在本地评测 detail 中，不进入普通日志，不冒充作者原文。

## 5. 测试

- 等权双通道 RRF 与旧实现排序一致；四通道加权数学手算；
- 映射按父范围 join、superseded 忽略、同 version 取最新、宇宙外丢弃计数、空交集报错；
- 检索融合后溯源字段与 per-question 审计正确；
- `subtopics=False` 时 question detail 键集合与基线完全一致（回归锁）；
- CLI 互斥校验错误文案。

## 6. 实验计划

1. 重建分层套件（此前 subtopics 层未覆盖新增的 5 篇已批准文章）；
2. 基线：`eval run <dataset> --build <冻结Build> --mode hybrid --rerank --k 5`；
3. 处理组：同命令加 `--subtopics`；
4. 逐题对比 recall@5 / evidence_coverage@5 / MRR / precision@5，重点检查此前漏召的
   题目是否补回、是否有题因细话题提名挤掉正确块而退化；
5. 报告延迟变化与丢弃命中总量；6 题样本只作机制验证，不下泛化结论。

## 7. 实验结果（2026-09-03，负结果）

在 pilot 6 题冻结集（Build `index_0d76f5fc...`，34 块；套件 subtopics 层 216 单元
/ 11 篇文章，120 单元映射进宇宙）上，四路加权 RRF 相对基线整体退化：

| 指标 | 基线 | w=1.0 | w=0.5 |
|---|---:|---:|---:|
| precision@5 | 0.233 | 0.200 | 0.200 |
| recall@5 | 0.861 | 0.694 | 0.694 |
| MRR | 0.889 | 0.750 | 0.722 |
| evidence_coverage@5 | 0.904 | 0.747 | 0.738 |

逐题 4 题不变、题 5 MRR 0.33→0.50、题 6 recall 1.0→0.0。失败机制与完整分析见
具体实验结论已归纳在本文与版本路线中；包含真实文章标题和本地运行标识的历史交接记录不进入
公开仓库。

结论：细话题命中提名父块的通道结构在正查上有系统性缺陷（粒度跳变、提名通道事实
压过主通道排位、宇宙外浪费），**不默认启用**；后续若探索应改为重排候选池补充或
细话题范围子块粒度，而不是调融合权重。

实现过程中修复过一个对齐 bug：套件 units.json 的 `title` 是文章标题而非细话题标题，
对齐键因此使用 `(version_id, start_unit, end_unit)` 范围三元组（同一父话题下细话题
范围不重叠），不使用标题。

