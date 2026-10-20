# 端到端回答与引用评测设计

状态：已实现 v0.1（2026-10-20）。

## 1. 目标

现有冻结评测只回答“正确证据是否进入检索结果”。本设计增加一条显式触发的端到端评测链，
使用与交互式 `ask` 相同的检索、上下文组装和 Generator 协议，自动记录最终回答状态、引用与
运行成本，并把失败定位到检索、上下文或生成阶段。

第一版不建设人工评分页面，不调用 LLM 自动评委，也不输出笼统的“答案正确率”。自动结果只
覆盖能够由冻结金标准和结构化回答确定计算的指标；语义是否曲解原文仍属于可选人工抽查。

## 2. 已确认范围

- 只接受已冻结的数据库评测集和与其语料快照一致的 Index Build；
- 第一版只运行单文章题，复用现有 `dev/holdout`；
- 必须显式指定 `--split dev` 或 `--split holdout`，不提供 `all`；
- 使用当前配置的生产 Generator，第一版为 DeepSeek；
- 固定 `supplement=off`，避免外部知识或补充分析混入评测；
- 不增加证据充分性阈值、拒答路由或新的回答 Prompt；
- 不实现自动语义裁判，不把证据重叠指标称为答案正确率；
- 题量很小，人工只按需查看自动失败项和少量成功样本。

## 3. 命令与运行流程

新增独立命令，避免改变现有纯检索评测的语义：

```powershell
uv run xifeng eval answer-run <DATASET_ID> --split dev --mode bm25 --k 5
uv run xifeng eval answer-run <DATASET_ID> --split holdout --mode bm25 --k 5
```

它复用 `eval run` 的 Build、检索模式、Top-K 和 Reranker 参数。首版不支持旧 JSON 评测文件、
分层检索、查询改写或细话题实验开关，避免把实验分支混入首个回答基线。

一次运行分成两个阶段：

1. **本地准备**：验证冻结快照，完成全部所选问题的检索和确定性上下文组装，不调用网络；
2. **外发披露**：显示数据集与内容哈希、split、题数、provider、模型、每题证据块数与原文
   字符数、合计原文字符数，以及最多外部请求次数；
3. **一次确认**：用户确认本次明确列出的评测批次，确认只对本次运行有效；
4. **顺序生成**：逐题调用现有 `EvidenceBoundGenerator`，每题最多一次生成和一次格式修复；
5. **确定性计分**：根据冻结证据范围、组装后的证据位置和结构化回答计算自动指标；
6. **持久化**：保存安全结果和汇总，输出运行 ID。拒绝确认时不调用 Generator，也不创建
   已完成运行。

本地检索不到任何候选时沿用现有行为：直接产生 `insufficient_evidence`，`request_count=0`。
检索到近似但无效证据时，由同一个 Generator 返回回答状态；端到端评测只观察结果，不在前面
增加启发式拒答判断器。

## 4. 复用真实问答链

端到端评测不能复制一份简化版 `ask`。实现把上下文组装、无证据本地拒答与生成编排提取为
普通 Python 服务，由 `ask` 和 `eval answer-run` 共同调用；两条 CLI 路径仍复用同一个检索
入口。CLI 继续负责加载模型、显示隐私披露、确认和渲染输出，核心服务不读取终端输入。

共享结果至少包含：

- `AssemblyResult` 及引用 ID 到文章版本、段落范围和源 Chunk 的映射；
- `AnswerGenerationResult`；
- 检索、组装、生成和总耗时；
- Provider、模型、请求次数和安全的 usage 元数据。

交互式 `ask` 的用户可见行为保持不变。端到端评测固定关闭 supplementary analysis。

## 5. 自动指标定义

### 5.1 回答状态与拒答

设 `predicted_abstain` 为最终状态等于 `insufficient_evidence`：

- `correct_refusal`：题目 `answerable=false` 且 `predicted_abstain=true`；
- `missed_refusal`：题目 `answerable=false` 但返回 `answered` 或 `partially_answered`；
- `over_refusal`：题目 `answerable=true` 但返回 `insufficient_evidence`；
- 分别汇总正确拒答率、漏拒答率和过度拒答率；分母为对应题型数量，没有分母时返回 `null`。

这些指标判断的是结构化状态是否符合题目标注，不自动证明拒答说明写得充分。

### 5.2 引用结构

- `citation_id_valid_rate`：引用 ID 属于当次上下文的比例；现有本地校验通过时应为 1.0，生成
  失败则单独计错，不伪造该指标；
- `cited_evidence_count`：所有 grounded section 去重后的引用项数量；
- `off_gold_citation_rate`：引用范围与任何冻结金标准范围都不相交的引用项比例，仅作为诊断，
  因为补充的相关证据不一定是错误引用。

### 5.3 金标准证据引用覆盖

只对可回答且产生 grounded section 的题计算：

- `gold_evidence_citation_coverage`：所有被引用证据范围与金标准范围相交的原子单元并集，除以
  金标准原子单元并集；
- `answer_point_citation_coverage`：至少被一个引用证据范围覆盖的 `claim_group` 数量，除以金
  标准答案要点数量。

这两个指标只说明回答引用了完成答案所需的证据，不说明生成文本准确表达了答案要点。报告中
必须使用“证据引用覆盖”名称，不能缩写成“答案准确率”或“答案要点准确率”。

### 5.4 阶段覆盖与成本

每题同时保存：

- 检索 Top-K 对金标准证据的覆盖；
- 最终组装上下文对金标准证据的覆盖；
- 最终引用对金标准证据的覆盖；
- 检索、组装、生成、总延迟；
- Generator 请求次数、输入输出 token usage（Provider 返回时）和生成失败状态。

每次运行只包含一个 split；汇总记录该 split，并按难度和来源分组。不可回答题不进入证据覆盖
平均值。

## 6. 自动失败归因

按可观测事实为每题添加零到多个失败标签：

- `retrieval_miss`：检索 Top-K 未完整覆盖金标准证据；
- `assembly_loss`：检索已覆盖，但最终上下文丢失部分金标准证据；
- `citation_omission`：上下文已覆盖，但最终引用未完整覆盖答案要点；
- `over_refusal`：可回答题被拒答；
- `missed_refusal`：不可回答题给出语料结论；
- `generation_error`：外部调用或两次结构校验后仍失败；
- `off_gold_citation`：存在未与金标准范围相交的引用，仅作复核提示。

不自动产生 `unsupported_claim` 或 `semantic_error`，因为这两类需要理解回答文本与原文之间的
语义关系。

## 7. 数据模型与持久化

新增独立表，不复用纯检索的 `evaluation_runs`：

### AnswerEvaluationRun

保存运行 ID、dataset/build/hash/snapshot、split、检索模式与 K、Generator provider/model、
代码版本、工作区状态、运行状态、汇总指标、开始和完成时间。实际外发范围由运行前提示展示，
不在数据库重复保存证据正文或可还原正文的披露明细。

运行状态使用 `running`、`completed`、`completed_with_errors`、`failed`。重新运行创建新记录，
不覆盖旧结果。

### AnswerEvaluationItem

每题保存 question ID、结构化 `GeneratedAnswer`、请求次数、安全 usage、阶段延迟、引用到
Chunk/文章版本/段落范围的映射、自动指标、失败标签和错误类型。

允许保存生成后的短回答和限制说明，因为它们是本地 Git 忽略的评测结果。不得保存：

- 完整外发 Prompt；
- `rendered_context` 或证据原文；
- Provider 原始响应；
- API 密钥；
- 完整文章或可重建完整文章的批量正文副本。

## 8. 失败与中断

- 快照、Build 或 split 不合法：在本地准备前拒绝；
- 用户拒绝外发：不调用网络，返回未执行；
- 单题结构化输出两次失败：记录 `generation_error`，继续下一题；
- 鉴权、余额、网络或 Provider 配置错误：停止后续外部调用，保留已完成题目，运行标记为
  `completed_with_errors` 或 `failed`；
- 再次运行总是创建新记录，不覆盖或自动续传旧结果；
- 无证据本地拒答：正常完成，不要求 API 密钥，不计生成错误。

## 9. 隐私与确认

端到端评测属于用户显式触发的 Generator 调用，不属于评测集候选生成，也不允许外发完整
文章。所有检索、重排和组装先在本地完成，只发送每题最终入选的少量原文证据。

确认提示必须在网络调用前给出本次运行的实际范围，而不是笼统提示：

- dataset 名称、版本、内容哈希和 split；
- 问题数及其中可回答/不可回答数量；
- Provider、模型和服务地址；
- 每题证据块数、原文字符数与总原文字符数；
- 最大请求次数为需要外部生成的题数乘以 2；
- 本地无证据题不会外发。

确认凭据只对本次进程、本次冻结哈希和已经组装的上下文有效；问题、Build 或上下文变化后必须
重新准备并确认。不得提供永久确认或后台定时批量运行。

## 10. 测试与验收

自动测试使用自造短文本和假的 Generator，至少覆盖：

- dev/holdout 必须显式选择，拒绝 `all`；
- 冻结哈希和 Build 快照不匹配时不运行；
- 拒绝确认时零外部请求；
- 披露中的题数、字符数和最大请求次数与实际准备结果一致；
- 无候选时本地正确拒答且 `request_count=0`；
- 近似证据存在时 Generator 可返回正确拒答；
- 可回答题错误拒答计入 `over_refusal`；
- 引用范围到 claim group 和金标准原子单元覆盖的确定性计算；
- 检索命中、组装丢失和引用遗漏的失败归因；
- 单题生成失败和 Provider 系统性失败的停止行为；
- 数据库不保存 Prompt、证据原文、Provider 原始响应或密钥；
- `ask` 与评测共用编排服务后，原有回答、引用和无证据行为保持一致。

真实验收先运行 dev，核对自动汇总和一条成功、一条失败或拒答结果。开发选择完成后才显式
运行 holdout。两道不可回答题至少查看最终状态和限制说明，但人工查看不形成新的必填评分表。

## 11. 非目标

- LLM-as-a-judge；
- 新的人工审核工作台；
- 自动证据充分性阈值或拒答路由；
- 自动修改 Prompt、检索参数或题目标注；
- 跨文章回答评分；
- 外部事实核查；
- 将自动证据重叠分数解释为语义正确率。
