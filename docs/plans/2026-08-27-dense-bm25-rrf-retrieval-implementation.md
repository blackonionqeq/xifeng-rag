# BGE Dense、BM25 与 RRF 混合检索实施计划

对应设计：[`docs/specs/2026-08-27-dense-bm25-rrf-retrieval-design.md`](../specs/2026-08-27-dense-bm25-rrf-retrieval-design.md)

## 交付顺序

1. 增加 `sentence-transformers` 与 `qdrant-client` 依赖，并扩展示例、本地配置模型与校验。
2. 扩展统一候选分数，增加检索模式、Dense 排名和 RRF 分数，同时保持旧 BM25 调用兼容。
3. 定义窄 `EmbeddingProvider`，实现 Sentence Transformers BGE provider 和确定性测试 fake。
4. 将现有 BM25 Build 泛化为可配置 Profile；在同一临时 Build 中生成 BM25 与本地 Qdrant。
5. 校验向量维度、point 集合和 TopicChunk 集合，关闭本地 Qdrant 后原子发布并切换活动 Build。
6. 实现 BM25、Dense、Hybrid 三种检索模式，以及 candidate depth 20、RRF k=60 的稳定融合。
7. 扩展 CLI build/search/eval 参数和评测运行元数据，允许同一冻结快照做三种模式对照。
8. 使用 fake provider 完成单元与集成测试，并验证失败回滚、Profile 隔离、离线默认测试和日志隐私。
9. 更新 README 与职责匹配的设计文档，运行 pytest、Ruff、Pyright 和 CLI smoke test。
10. 显式下载 BGE 到本地模型目录，建立 Hybrid Build，运行冻结 pilot 三组实验并保存逐题结果。

## 完成门槛

- 旧 BM25 Build 可继续显式搜索和评测；
- 新 Build 同时包含一致的 BM25 与 Dense 派生产物；
- 三种模式返回统一候选，并保留单路排名、原始分数和融合分数；
- 任一路失败时旧活动 Build 不受影响；
- 自动测试不下载模型、不联网、不写入付费正文；
- 冻结 pilot 的三组实验均保存指标、延迟、模式和失败案例。
