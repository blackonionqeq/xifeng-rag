# V0 语义分块交接文档

更新时间：2026-08-25（Asia/Shanghai）

用途：供下一次 Codex 会话从当前状态继续，不需要重新回放本次长对话。

## 1. 明天开始时先做什么

按顺序阅读：

1. `AGENTS.md`
2. 本文
3. `docs/specs/2026-08-25-semantic-units-and-content-roles-design.md`
4. `docs/design/chunking-and-indexing.md`
5. 当前工作区的 `git status --short`

不要立即运行以下命令：

```powershell
uv run xifeng chunks propose
uv run xifeng chunks approve ...
uv run xifeng index build
```

无参数 `chunks propose` 会处理大量文章。当前两个真实提案也都没有获准，不应批准或建库。

## 2. 用户已经确认的决定

- 当前顶层 Markdown 共 113 篇。
- 只有《示例文章丙：从一个量化交易案例谈风险控制》已确认是不完整
  的未付费预览，需要排除。
- 其稳定 ID 是 `sn:2a602cf6b0a62299d0f31f4245bf6d4a`。
- 其他较短文章是完整付费文章；不能再按字数阈值批量排除。
- 最终话题块应按语义边界形成，不追求固定块数或目标字符数。
- 源 Markdown 的物理行和句子只是给模型选择边界的原子单元，不是最终检索块。
- 用户认为短文前部包含低价值内容，希望不要把这些内容当作核心检索正文。
- 当前自动角色判断仍不稳定，用户已要求暂停继续调 Prompt；不要未经沟通继续跑新模型实验。

## 3. 当前实现范围

V0 主体代码目前位于尚未整体提交的 `src/` 与 `tests/` 中，包括：

- 只读扫描外部顶层 Markdown；
- 确定性清洗、尾部目录处理、链接锚文本与源位置映射；
- cleaner 版本化与新增/修改/未变化/缺失分类；
- 可配置的单篇文章排除；
- Ollama 本地结构化话题分块与安全重试；
- JSON/Markdown 人工审核；
- approved manifest、TopicChunk、BM25、引用和基础评测。

本轮新增：

- `deterministic-v2-semantic-units`：每个非空物理行先成为候选单元，行内再按安全的
  中英文句末符号切分；
- Markdown 链接整体受保护，裸 URL 查询参数、小数和普通英文句点不会被误切；
- 每个原子单元保存原 Markdown 的准确 `source_start/source_end`；
- 模型只输出 `end_unit`，程序推导连续起点；
- `content_role` 支持 `substantive` 与 `boilerplate`；
- 新批准清单把人工确认后的完整 topics JSON 存入
  `approved_manifests.topics_json`；
- boilerplate 保留在批准清单中，但不生成 TopicChunk；全文 boilerplate 禁止批准；
- cleaner 或 Prompt 升级会创建并列版本/提案，不覆盖历史数据；
- 批量提案只选每篇最新的 active 版本，避免 cleaner v1/v2 同时运行。

为了兼容旧表结构，`ParagraphRow`、`start_paragraph`、`end_paragraph` 等字段名暂未重命名；
cleaner v2 起，它们的业务含义是“原子单元及其范围”。

## 4. 关键代码位置

- `src/xifeng_rag/config.py`：`excluded_article_ids` 本地配置与校验。
- `src/xifeng_rag/corpus.py`：原子单元、清洗、稳定 ID、增量扫描和排除。
- `src/xifeng_rag/domain.py`：Topic、GeneratedTopic、ContentRole 与覆盖模型。
- `src/xifeng_rag/chunking.py`：Ollama Schema、Prompt、安全校验、审核和批准。
- `src/xifeng_rag/cli.py`：扫描、提案、审核、索引等命令。
- `src/xifeng_rag/storage.py`：SQLAlchemy 状态表。
- `src/xifeng_rag/migrations/versions/20260825_0002_manifest_topics.py`：manifest JSON 迁移。
- `tests/`：当前共 40 项自动化测试。

相关设计：

- `docs/specs/2026-08-25-v0-mvp-design.md`
- `docs/specs/2026-08-25-semantic-chunking-and-corpus-exclusions-design.md`
- `docs/specs/2026-08-25-semantic-units-and-content-roles-design.md`
- `docs/design/chunking-and-indexing.md`

## 5. 本地环境

```text
项目：D:\codes\projects\xifeng-rag
外部语料：D:\codes\projects\test-reverse-wechat-official-accounts-paid-article\saved_articles
Ollama：D:\ai-tools\ollama\ollama.exe
模型目录：D:\ai-models\ollama
模型：qwen3:4b-instruct-2507-q4_K_M
上下文：32768
数据库：data\xifeng.sqlite3
```

真实排除 ID 已写入 Git 忽略的 `config.local.yaml`。外部语料目录没有被修改。

PyPI 曾出现 TLS 握手失败，导致 `uv` 尝试重新解析 `hatchling` 时失败。已有虚拟环境可用，
当前验证命令使用 `--no-sync`：

```powershell
uv run --no-sync ruff format --check src tests
uv run --no-sync ruff check src tests
uv run --no-sync pytest
uv run --no-sync pyright
```

最后一次结果：40 tests passed，Ruff 通过，Pyright 0 errors，`git diff --check` 通过。

## 6. Git 与数据库状态

最近提交：

```text
009f2c6  Design semantic units and content roles
60db3f5  Design semantic chunking and corpus exclusions
56d6fe4  Move V0 design to generic specs directory
1fb4225  Add V0 MVP implementation design
406ef53  初始化西风 RAG 项目设计与配置
```

重要：完整 V0 实现仍是未提交工作区，不能假设仓库是 clean。当前可见状态包含修改过的
README/设计文档，以及尚未追踪的 `src/`、`tests/`、`pyproject.toml`、`uv.lock` 等。
不要 reset、checkout 或清理这些文件。

当前数据库：

```text
Alembic revision: 20260825_0002
active articles: 112
active article versions: 224
chunk proposals: 8
approved manifests: 0
topic chunks: 0
```

224 个 active version 是 cleaner v1/v2 并存造成的。它保留旧版本以便未来批准版本继续
服务；提案 CLI 已保证每篇只选择最新 active version。当前没有批准清单或已发布 TopicChunk。

迁移前数据库备份：

```text
tmp/db-backups/xifeng-before-semantic-units-v2.sqlite3
```

## 7. 真实扫描结果

cleaner v2 扫描：

```text
发现文章：113
排除 1
修改 112
未变化 0
缺失 0
待清洗审核 3
```

两篇验收文章的新版本：

| 文章 | article ID | cleaner v2 version | 单元数 | 最大单元 |
|---|---|---|---:|---:|
| 示例文章丑：阅读策略的边界 | `sn:2e41ecbf67f6a389ba23e129cb322ce2` | `version_fcaf34f27d75dd1b2fc51575` | 126 | 106 字符 |
| 示例文章子：一个转折点，你这辈子真正的考试，才刚刚开始 | `sn:c6b25e2c9def50f50d32ddf9fddbf85f` | `version_deba2f80c23a160d4fbccf9d` | 692 | 103 字符 |

旧解析中第二篇曾有一个 13,029 字符的不可分割段落，占正文约 69%。新原子解析已经消除
这个结构性问题。

## 8. Prompt 实验结果

### v3：`topic-segmentation-v3-semantic-units-roles`

短文提案：`proposal_006f7d1abc77f8185b3446b40d2b0243`

- 共 6 块；
- 前两块被标为 boilerplate，三个交易原则为 substantive；
- 最后一块“个人认知反思与总结”被错误标为 boilerplate；
- 角色判断过度排除。

长文提案：`proposal_7bd213aaaad34fc44e92fef550b60303`

- 共 6 块；
- 前五个明显包含观点的块全部被误标为 boilerplate；
- 只有最后一块 substantive；
- 结果不可接受。

### v4：`topic-segmentation-v4-conservative-roles`

增加的安全规则：

- 默认所有块 substantive；
- 只有纯付费提示、操作说明、导航或推广固定模板才可 boilerplate；
- boilerplate 最多 2 块，原文总字符不得超过全文 15%；超过就进入安全纠错重试；
- 引言、故事、观点、解释、案例、反思和总结必须 substantive。

短文提案：`proposal_1c9a95e1868f680674b216aea1d6f85e`

- 共 9 块，全部 substantive；
- 不再错误删除末尾反思；
- 但付费门槛、认知成本等文章元话语也全部保留；
- 三条原则被拆成多个子话题，整体偏碎；
- 用户认为短文结果仍不理想，因此已暂停继续调 Prompt。

长文提案：`proposal_25e2e07a3d8f0e5f26fb7aeab39fed61`

- 共 4 块，全部 substantive；
- 块大小约为 3657 / 2241 / 7431 / 4665 个原文字符；
- 结构明显优于旧解析，主题标题也连贯；
- 第一次输出未覆盖最后单元，安全重试后成功；
- 这篇结果可作为原子单元方案有效的正面样本，但仍未批准。

当前导出的最新审核文件是 v4：

```text
data/reviews/version_fcaf34f27d75dd1b2fc51575.review.json
data/reviews/version_fcaf34f27d75dd1b2fc51575.review.md
data/reviews/version_deba2f80c23a160d4fbccf9d.review.json
data/reviews/version_deba2f80c23a160d4fbccf9d.review.md
```

数据库中的 v3/v4 提案都保留；同一版本的导出文件名相同，所以磁盘审核文件当前是最新
的 v4 内容，不是 v3。

## 9. 当前结论

已经验证有效：

- 物理行 + 句子级原子单元解决了极端长段落问题；
- 只输出 `end_unit` 比让 4B 模型重复计算起点稳定；
- 结构校验、安全诊断和 Prompt 并列版本机制工作正常；
- 长文 v4 的顶层语义块质量比旧方案明显更好。

尚未解决：

- 二分类 `substantive/boilerplate` 无法表达短文中的“作者自述或文章元话语，但对目标主题
  检索价值较低”；
- v3 过度排除，v4 又过度保留；
- 短文 v4 有过度细分子话题的问题；
- 当前 4B 模型仅靠继续堆提示词，收益已经不确定。

## 10. 建议明天从这里继续

不要先改 Prompt。推荐先做一个很小的人工金标准：

1. 打开短文 v4 Markdown 审核文件；
2. 人工写出真正希望保留的 3–5 个顶层话题边界；
3. 明确每个前言/自述块应该是：保留、降权，还是完全不索引；
4. 再决定领域角色是否从二分类扩展为三分类，例如：
   `substantive / article_meta / boilerplate`；
5. 用短文人工金标准和当前长文 v4 作为最小回归集，再比较：
   - 继续使用 qwen3:4b 的两阶段流程（先分块、后分类）；
   - 更强的本地模型；
   - AI 只提语义边界、角色完全人工审核；
   - `article_meta` 保留但降低检索权重，而不是彻底排除。

最保守、最容易落地的下一步是：保留当前原子解析和语义边界实现，暂时把所有角色默认
视为 substantive，先由人工在审核 JSON 中修改短文边界和角色。没有至少一篇人工金标准
前，不建议继续凭感觉迭代 Prompt。

## 11. 隐私与安全提醒

- 不要提交 `data/`、审核文件、索引或任何真实付费正文；
- 不要修改、移动或删除外部 `saved_articles` 文件；
- 不要在普通日志打印完整 Prompt 或付费正文；
- 真实 Ollama 分块只在本机运行；
- 不要调用外部 Generator 处理批量文章；
- 当前两篇提案均未批准，明天继续时保持这一状态，除非用户明确审核通过。

## 12. 可复制给新会话的起始请求

```text
请先阅读 AGENTS.md 和
docs/handoffs/2026-08-25-v0-semantic-chunking-handoff.md，检查当前 git/database 状态，
不要批量运行 chunks propose，也不要批准现有提案。我们从短文人工金标准和
substantive/article_meta/boilerplate 是否需要三分类开始讨论。
```
