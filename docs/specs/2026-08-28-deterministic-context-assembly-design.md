# 确定性上下文组装设计

日期：2026-08-28

## 1. 目标

在本地 Hybrid + RRF + BGE Reranker 之后增加一个确定、可审计的上下文组装层，输出可供
后续 Generator 使用的结构化证据上下文。该层负责完整证据块选择、预算、引用编号、来源追踪、
排除原因和状态分类，不负责重新检索、相关性判断、语义压缩、生成或最终证据充分性判断。

首版保持已批准 TopicChunk 完整，不在块内裁切。未来抽取式压缩可在组装器之前把完整块转换为
更小的真实原文范围，而无需改变引用与组装协议。

## 2. 范围与边界

主流程变为：

```text
Dense + BM25 -> RRF -> 本地 Reranker
                         -> 未来 RetrievalController / Compressor
                         -> ContextAssembler
                         -> Generator
                         -> 引用校验与拒答
```

首版包含：

- 保持 Reranker 输入顺序；
- 完整 TopicChunk 选择；
- 完全相同证据的确定性去重；
- 最大证据数与总渲染字符预算；
- 查询内引用 ID 与永久来源身份；
- 可配置的 AI 检索辅助元信息渲染；
- 结构化排除原因、业务状态和工程错误；
- 本地 CLI 预览与 JSON 输出。

首版不包含：

- 块内截断、段落选择或上下文压缩；
- 语义去重、文章多样性配额或重新排序；
- CRAG 式反思、查询改写或补充检索；
- Generator、回答、引用有效性判断或 Q&A 持久化；
- 外部网络调用。

## 3. 组件接口

新增普通 Python 组件 `ContextAssembler`。组装策略不读取问题内容；输入候选已经是特定问题的
排序结果，因此接口保持为：

```text
assemble(ranked_candidates, config) -> AssemblyResult
```

组装器不访问数据库、索引、模型或网络，不修改输入候选。

默认配置：

```yaml
context_assembly:
  max_evidence_items: 5
  max_total_chars: 50000
  include_ai_metadata: false
```

当前 34 个 TopicChunk 的最大单块为 5,441 字符，最长五块合计为 25,386 字符。50,000 字符
是异常保护而非填充目标；最大证据数才是当前相关性、隐私和外发范围的主要控制。DeepSeek
V4 Flash 当前标称 1M token 上下文，但模型窗口上限不应直接成为默认装载目标。

## 4. 领域结构

`AssemblyResult` 包含：

```text
status                    ready | no_evidence
evidence_items            最终入选证据
excluded_candidates       未入选候选及原因
usage                     输入、入选、排除与字符统计
config_snapshot           实际组装配置
rendered_context          仅存在于内存的规范 JSON
rendered_context_sha256   规范 JSON 的 SHA-256
```

`EvidenceItem` 包含：

```text
citation_id                E1、E2……，仅在单次组装结果内稳定
source_chunk_id
article_id
article_version
article_title
original_paragraph_range
selected_paragraph_range
original_text
retrieval_scores
reranker_rank
topic_title
context_summary
compression_method         首版固定为 none
```

首版要求：

```text
selected_paragraph_range == original_paragraph_range
compression_method == none
```

查询或配置变化后，`E1` 可以指向不同证据；永久可追溯性由 chunk、文章版本和原子范围共同
保证。`topic_title` 与 `context_summary` 始终保留在内部结构中，但是否发送给 Generator 由
配置决定。

`ExcludedCandidate` 至少包含候选身份、原始排名、假设渲染字符数和排除原因：

```text
duplicate
item_limit
char_budget
```

## 5. 选择算法

处理顺序固定为：

1. 校验配置；
2. 校验全部候选结构；
3. 按输入顺序去除相同 `chunk_id`，以及相同文章版本与完全相同原子范围的候选；
4. 逐个尝试完整加入候选；
5. 达到 `max_evidence_items` 后，将其余非重复候选标记为 `item_limit`；
6. 候选无法完整放入剩余字符预算时标记为 `char_budget`，继续尝试后续候选；
7. 为最终入选证据依次分配 `E1`、`E2`……，同时保留原始 Reranker 排名；
8. 生成规范 JSON、校验不变量并计算 SHA-256。

不允许静默截断。如果第一个有效且非重复的候选单独就超过总预算，返回
`budget_too_small`，不得跳过最相关证据后伪装为成功。

首版不设置单篇文章占比上限。多文章问题的召回完整性由检索评测发现，组装器不暗中加入低
相关候选。

## 6. Generator 上下文格式

使用确定性 JSON，而不是可能与原文内容混淆的自由文本分隔符：

```json
{
  "evidence": [
    {
      "citation_id": "E1",
      "article_title": "文章标题",
      "article_version": "版本 ID",
      "paragraph_range": [10, 15],
      "original_text": "作者原文"
    }
  ]
}
```

启用 `include_ai_metadata` 后，每条证据额外包含：

```json
{
  "ai_generated_retrieval_metadata": {
    "topic_title": "AI 话题标题",
    "context_summary": "AI 概括"
  }
}
```

字段名明确声明其为 AI 生成的检索辅助信息。AI 元信息不能获得独立引用 ID，也不能作为作者
原话引用。BM25、Dense、RRF、Reranker 分数、chunk ID、排除原因和内部统计不发送给
Generator。

规范 JSON 使用固定字段顺序、UTF-8 语义和 `ensure_ascii=false`。`max_total_chars` 计算最终
JSON 的完整字符数，包括字段名、来源元数据、可选 AI 元信息和原文。用户预览可以使用更友好
的显示格式，但真正发送的内容必须与计算哈希的 JSON 完全一致。

## 7. 状态与错误分类

正常业务结果：

| 状态 | 含义 | 后续行为 |
|---|---|---|
| `ready` | 至少组装出一条合法证据 | 可以交给 Generator |
| `no_evidence` | 检索候选为空 | 不调用 Generator，进入证据不足流程 |

工程错误：

| 错误码 | 含义 |
|---|---|
| `invalid_config` | 数量或字符预算非法 |
| `invalid_candidate` | 原文为空、来源缺失或范围非法 |
| `budget_too_small` | 第一条有效候选无法放入预算 |
| `render_invariant_failed` | 引用、数量、预算、渲染或哈希内部不一致 |

错误信息只能包含 ID、计数、预算和错误码，不得包含原文。未来评测至少能区分组装成功、检索
无证据、预算排除、重复、候选损坏、预算配置过小和内部错误。

## 8. 隐私与持久化

首版不新增数据库表。完整 `AssemblyResult`、`rendered_context` 与人类可读预览只存在于当前
进程内，不写普通日志或数据库。

以后实现 Q&A Run 时只保存：

- 入选 chunk ID、文章版本与原子范围；
- 配置快照；
- 排除原因与字符统计；
- `rendered_context_sha256`；
- 状态或错误码。

不重复保存 `original_text`、拼接全文或完整 Prompt。需要复现时从不可变文章版本重建并用哈希
核对。

## 9. CLI

扩展现有命令：

```powershell
xifeng search "问题" --mode hybrid --rerank --assemble
xifeng search "问题" --mode hybrid --rerank --assemble --json
```

该路径只执行本地检索、重排和组装，不调用 Generator，不产生网络请求。人类可读模式显示引用
ID、来源、范围、字符统计和排除原因；显式 `--json` 输出完整结构化组装结果。未使用
`--assemble` 时保持当前搜索行为。

## 10. 测试与验收

测试只使用自造短文本，覆盖：

- 保持输入顺序、引用编号、来源与分数；
- AI 元信息开关只改变外发 JSON，不丢失内部元信息；
- 相同 chunk、相同版本与范围去重；
- 数量限制、预算跳过后继续、首块超预算；
- 中文、换行、引号的确定性 JSON 序列化；
- 空候选、非法配置、非法候选和内部不变量分类；
- 原文绝不截断，输入对象不被修改；
- 同一输入与配置产生相同 JSON、排除结果和 SHA-256；
- CLI 人类可读与 JSON 输出；
- 未启用组装的搜索回归；
- 默认路径不调用网络，错误消息不包含原文。

本机真实语料验收只报告 Top-5 入选数、排除原因、字符数、AI 元信息字符增量、耗时和状态，
不得输出或提交付费正文。组装器不改变检索顺序，因此本阶段不宣称检索质量提升；生成质量对比
留到 Generator 接入之后。

完成标准是确定性组装、引用可追溯、状态可分类、全文不持久化、默认无网络请求，并通过全部
pytest、Ruff、Pyright 与真实语料只读烟测。
