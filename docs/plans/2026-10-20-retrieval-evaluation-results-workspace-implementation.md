# 检索评测运行与结果工作台实施计划

对应设计：[`docs/specs/2026-10-20-retrieval-evaluation-results-workspace-design.md`](../specs/2026-10-20-retrieval-evaluation-results-workspace-design.md)

1. 增加评测运行列表、详情和兼容性比较的只读服务，不迁移现有运行表。
2. 为评测工作台增加冻结题集、兼容 Build、异步任务和 holdout 二次确认校验。
3. 从独立命令和项目控制台注入复用 `run_frozen_evaluation` 的同一个本地运行器，并可靠关闭
   模型资源。
4. 增加“运行结果”视图，展示汇总、分组、逐题候选、按需原文和历史基线差值。
5. 覆盖结果查询、对比、异步运行、隐私和静态页面约束，运行完整质量检查。
6. 同步 README、架构与路线文档。
