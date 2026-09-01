# 文章反查评测 CLI 实施计划

对应设计：[`docs/specs/2026-09-01-article-discovery-evaluation-design.md`](../specs/2026-09-01-article-discovery-evaluation-design.md)

1. 新增文章反查评测输入、运行结果与指标模型，校验现有 Git 忽略 JSON 并计算内容哈希。
2. 实现可复用的三模式评测服务，保存逐题排名、阶段分数、失败分类、延迟和运行来源信息。
3. 实现结果 JSON 的原子本地持久化，不复制正文或文章概括。
4. 在 `eval` 命名空间增加薄 CLI 入口，支持默认数据集、指定 Build 和结构化输出。
5. 使用自造夹具覆盖服务、输入校验、持久化与 CLI 依赖装配。
6. 更新 README、CLI 指南和路线状态，运行 Ruff、Pyright 与完整 pytest。
