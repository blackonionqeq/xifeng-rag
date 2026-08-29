# 分层子块检索与相邻上下文实施计划

对应设计：[`docs/specs/2026-08-29-layered-child-retrieval-design.md`](../specs/2026-08-29-layered-child-retrieval-design.md)

## 交付顺序

1. 扩展配置与领域模型，增加分层 Profile 参数、子块范围、父子来源分数、`matched_ranges` 和
   `expanded_range`；旧 TopicChunk 候选保持兼容。
2. 新建边界清晰的分层检索模块，实现确定性子块贪心切分、尾块合并、稳定 ID、清单序列化与
   候选上限校验；使用自造原子单元完成单元测试。
3. 将 Hybrid 建库中的 BM25、Dense 写入和自检抽成可复用内部函数；新增独立 layered Build，
   在同一临时目录生成父、子两套索引和不含正文的子块范围清单，全部成功后原子发布。
4. 实现父块 Hybrid Top-5 与子块 Hybrid Top-20 并行召回：父命中展开全部子块，按子块 ID
   去重，分别保留两层分数，再交给现有本地 Reranker 统一排序。没有 Reranker 时明确拒绝。
5. 在重排后执行同父块前后各一块扩展，合并重叠或连续范围；字符预算优先保留命中范围，
   只移除最外侧邻居。让现有上下文组装和 Generator 引用扩展后的规范原文。
6. 给 `index build` 增加 `--profile layered`，给 `search`、`ask`、`eval run` 增加显式
   `--layered`；默认 Hybrid Build 和查询行为保持不变。
7. 扩展冻结评测详情，按人工原子单元证据范围评价 `matched_ranges`，单独记录扩展覆盖、候选数、
   上下文字符数和分层查询配置；按需增加向后兼容的数据库迁移。
8. 覆盖子块、建库回滚、父子召回、稳定重排、邻居边界、预算、CLI、评测和隐私测试；更新
   README、示例配置及职责匹配的设计文档。
9. 运行 pytest、Ruff、Pyright 和 CLI smoke test；随后构建手动启用的真实 layered Build，
   在同一冻结开发集上依次保存 A/B/C/D 对照，质量提升与退化均如实报告。

## 实现边界

- SQLite 中现有文章版本、原子单元、批准清单和 TopicChunk 仍是事实来源；子块只存在于可重建
  Build 清单和检索产物中，不新增审核表。
- Build 清单、Qdrant payload、普通日志和评测运行元数据不复制付费正文；命中后从 SQLite
  原子单元范围还原正文。
- layered Profile 不自动替换当前默认 Profile；只有显式建库和显式查询才使用。
- 第一版固定 Hybrid + 本地 Reranker，不加入跨层加权、查询改写、RAPTOR 或跨父块扩展。
- 先保证新旧 Build 完整兼容和原子回滚，再运行真实模型实验；测试不得下载模型或联网。

## 完成门槛

- 相同批准清单和 Profile 可重复生成完全一致的子块边界、ID 与 Build 身份；
- 父、子四套索引及映射清单通过集合一致性校验后才发布，任一步失败时旧活动 Build 可用；
- 分层查询保留可解释的两层来源和分数，最终 Reranker 顺序稳定；
- 相邻扩展不跨 TopicChunk，`matched_ranges` 与 `expanded_range` 始终可区分，引用只来自规范原文；
- 默认现有 build/search/ask/eval 路径和已有测试行为不变；
- 自动测试、Ruff、Pyright 全部通过，并能用冻结数据复现 A/B/C/D 的质量、延迟和资源成本。
