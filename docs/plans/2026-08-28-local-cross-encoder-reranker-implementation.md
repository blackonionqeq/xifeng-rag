# 本地 Cross-Encoder 重排实施计划

对应设计：[`docs/specs/2026-08-28-local-cross-encoder-reranker-design.md`](../specs/2026-08-28-local-cross-encoder-reranker-design.md)

## 交付顺序

1. 扩展配置与统一候选模型，加入独立 Reranker 配置、分数、排名和截断诊断，保持默认关闭时完全兼容。
2. 定义窄 `Reranker` Protocol、结果与元数据模型，实现纯排序函数和确定性 Fake 测试。
3. 使用 `sentence-transformers` 5.x `CrossEncoder` 实现本地 provider，显式处理 pair token 预算、批量预测、有限分数校验、revision、设备、加载耗时和资源诊断。
4. 调整 Hybrid 查询数据流：启用时两路至少召回 `candidate_k`、RRF 保留宽候选、一次装配候选、重排后截取最终 Top-K；关闭时保持现有路径。
5. 给 `search` 与 `eval run` 增加显式 `--rerank`，限制为 Hybrid 模式，并确保每条命令只加载和关闭一次模型。
6. 扩展冻结评测详情与运行记录，保存 Reranker 配置、模型元数据、截断计数、加载时间和逐候选分数；增加 Alembic 迁移并保持旧记录兼容。
7. 补充配置、排序、搜索、CLI、评测、迁移、错误和隐私测试；测试只使用 fake，不下载模型、不联网。
8. 更新 README 与职责匹配的设计文档，运行 pytest、Ruff、Pyright 和 CLI smoke test。
9. 检查本地模型缓存；如权重不存在，在用户明确许可后下载。随后在相同冻结 pilot 上运行无重排和 candidate depth 5/10/20/30 对照，记录质量、截断、延迟和资源结果。

## 实现边界

- Reranker 是查询期组件，不改变 Index Profile 或 Index Build 身份；
- 只对 Hybrid 模式显式启用，不为 BM25/Dense 单路增加隐式行为；
- Reranker 直接决定最终顺序，不混合不同量纲的原始分数；
- 模型加载、推理或输出校验失败时整次查询/评测失败，不静默回退；
- 重排输入只含 query、文章标题和 `original_text`；
- 真实正文、完整 pair、向量与密钥不进入普通日志、数据库元数据或 Git。

## 完成门槛

- 未传 `--rerank` 时现有搜索、评测和 85 个基线测试行为不变；
- 启用后 RRF 宽候选、Cross-Encoder 排序和最终 Top-K 数据流正确且稳定；
- 每个候选保留 BM25、Dense、RRF 与 Reranker 的可解释阶段信息；
- 截断可检测、可记录，错误不会伪装成成功的重排实验；
- 冻结评测能够复现模型、revision、设备、输入预算、候选深度和代码状态；
- 自动测试、Ruff 和 Pyright 全部通过；
- 本机真实模型验收如实报告质量、延迟、截断和资源成本，不以指标必须提升替代实现正确性。
