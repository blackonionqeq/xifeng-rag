# 文章反查评测 CLI 设计

状态：已确认，最小 MVP。

## 目标

为现有文章级检索增加可复现的本地评测入口。人工只维护问题与确认后的正确文章，运行器自动
比较 BM25、Dense 和 Hybrid，计算 Hit@1/3/5、MRR 与延迟，并保存逐题排名。核心能力必须是
普通 Python 服务，CLI 只做依赖装配，以便后续本地前端由用户手动触发同一服务。

## 边界

- 输入沿用 Git 忽略的 `data/article_discovery_queries.json`，只评估
  `review_status: confirmed` 的题目；未确认题计入跳过数量。
- 一次运行固定比较 BM25、Dense、Hybrid，不搜索最佳权重，不加入话题级 Reranker。
- 使用一个显式或当前活动的文章级 Index Build；正确文章不在该 Build 时记录为数据不兼容，
  不把它误算成普通检索失败。
- 不复用证据检索的 `EvaluationRunRow`，因为它外键绑定证据块 Index Build。MVP 将结构化结果
  原子写入 `data/evaluation_runs/`；后续 UI 可直接调用服务并读取返回值，持久化后端可再替换。
- 不生成答案、不使用自动评委、不调用外部服务，也不保存正文或完整文章概括。

## 接口与数据流

`load_article_discovery_dataset(path)` 负责验证 schema、题目 ID、问题、目标文章及重复项。
`run_article_discovery_evaluation(...)` 接收数据库会话、已验证数据集、Embedding provider、Build
和检索参数，返回 Pydantic 结果模型。`save_article_discovery_evaluation(...)` 负责原子落盘。

CLI 入口为：

```powershell
uv run xifeng eval article-discovery data/article_discovery_queries.json
```

省略路径时默认读取项目 `data/article_discovery_queries.json`。可用 `--build` 指定可用 Build，
`--json` 将完整结果输出到 stdout；普通输出只显示运行 ID、结果文件、各模式汇总和失败题。

每种模式保存候选文章 ID、标题、最终排名和已有阶段分数。每题保存目标排名、Hit@1/3/5、
倒数排名、延迟和状态；汇总保存题数、各命中率、MRR、平均延迟及 P95。运行级元数据保存数据集
内容哈希、Build ID、语料快照、Profile、检索参数、代码提交和工作区状态。

## 错误与隐私

数据集非法、没有确认题、Build 不可用、索引文件缺失或模型失败时整次运行失败，不产生半份
结果。Embedding provider 使用严格本地缓存模式。目标文章不在 Build 时该题在三种模式均记为 `expected_article_not_in_build`，并从检索
指标分母中排除，同时在汇总中显式报告。普通日志与持久化结果均不包含正文、文章概括、模型
请求或 API 密钥。

## 验证

- 自造文章夹具验证三种模式、排名、Hit@K、MRR 和结构化逐题结果；
- 验证未确认题跳过、重复 ID/空问题等输入失败；
- 验证目标文章不在 Build 时的分类与有效样本数；
- CLI 测试验证默认路径、薄服务调用、JSON/普通输出和 provider 关闭；
- Ruff、Pyright 与完整 pytest 回归。
