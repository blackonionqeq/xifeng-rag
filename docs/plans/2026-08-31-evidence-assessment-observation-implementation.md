# 证据充分度观测层实施计划

对应设计：[`docs/specs/2026-08-31-evidence-assessment-observation-design.md`](../specs/2026-08-31-evidence-assessment-observation-design.md)

## 交付顺序

1. 在配置中增加默认关闭的 `evidence_assessment.enabled` 与 `observation_k`，同步示例配置。
2. 在领域模型中增加带计算版本、状态和可空指标的 `EvidenceAssessment`，新建纯函数评估模块，
   基于最终有序候选计算分数间隔、文章数、Dense/BM25 双路支持率和原问题/改写共同命中率。
3. 在 CLI 编排层统一调用评估器：`search` 开启时输出候选与结构化评估，`ask` 在 Generator
   外发确认前显示简短诊断；关闭时保持现有输出与控制流。
4. 在冻结评测逐题详情中保存同一评估结果，并汇总各可用指标的有效样本数、均值、中位数、
   最小值和最大值；复用现有 JSON 字段，不增加数据库迁移。
5. 使用自造候选覆盖完整、空、数量不足、缺少 Reranker、单/双检索来源和有/无查询改写；
   增加 search、ask、eval 的开关与失败隔离测试。
6. 更新 README 中的配置与行为说明，运行完整 pytest、Ruff、Pyright、差异检查和 CLI smoke。
7. 在现有 6 题冻结试验集开启观测，保存并报告逐题原始信号；不在小样本上设置阈值或自动路由。

## 实现边界

- `assess_evidence` 是本地确定性纯函数，不读取正文内容、不产生网络请求、不写数据库。
- 第一阶段不定义评估器插件接口或注册表；稳定领域结果模型允许以后增加指标。
- `status=failed` 只用于真实计算异常；正常的缺失阶段使用 `null`，空候选仍是成功评估。
- 观测开关不能改变候选、排序、上下文、Generator 输入或分层检索选择。
- 普通 search/ask 不保存历史，eval 只扩展现有运行 JSON。
- 测试不启动模型、不下载依赖、不读取外部付费语料。
- 无关未跟踪目录 `tempnpm-cache/` 不纳入提交。

## 完成门槛

- 默认配置下所有现有行为、输出和测试保持不变；
- 三个入口复用同一个评估函数，公式和空值语义有直接单元测试；
- 启用后能观察 TopicChunk、分层、改写和无改写候选，不要求特定检索配置；
- 评测逐题记录与汇总可复现，并带 `calculation_version`；
- 评估失败不阻断检索或回答，诊断不包含问题或正文；
- 完整测试、Ruff、Pyright、CLI smoke 与 `git diff --check` 通过。
