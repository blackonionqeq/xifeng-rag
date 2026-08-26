# V0 语料管理与评测集交接文档

更新时间：2026-08-26（Asia/Shanghai）

用途：供下一次 Codex 会话直接延续当前状态，不需要重新回放本次长对话。

## 1. 新会话开始顺序

按顺序阅读：

1. `AGENTS.md`
2. 本文
3. `docs/specs/2026-08-26-corpus-management-dashboard-design.md`
4. `docs/specs/2026-08-26-local-chunk-review-ui-design.md`
5. `docs/design/roadmap.md`
6. 当前 `git status --short` 和数据库只读状态

不要在开始时批量运行：

```powershell
xifeng chunks propose
xifeng chunks approve ...
xifeng index build
```

不得批准数据库中遗留的待审提案。只有用户在审核页面明确批准的单篇草案才能进入索引。

## 2. 用户当前目标与偏好

- 用户偏好简单、可见的人工流程，不希望为少数异常文章持续堆复杂 Prompt。
- 默认分块流程保留；单篇不满意时可以显式选择其他模型和完全修改一次性 Prompt。
- 外部全文分块当前优先复用个人 ChatGPT Plus 的 `codex-cli`，模型为
  `gpt-5.6-terra`；用户已关闭 ChatGPT 的训练数据选项。
- 每篇全文外发仍必须显示文章、版本、清洗后字符数、原子单元数、provider、模型和最大请求
  次数，并取得该篇的明确确认；不得批量发送。
- 用户审核时可完全改边界、标题、概括、关键词和 `include_in_index`。
- 不索引块仍保留在不可变批准清单中。
- 用户不希望文档或目录使用 `superpowers` 名称；设计文档放在 `docs/specs`。
- 用户希望从命令行人工串联逐步转向本地可视化管理，但不希望第一版把外部模型长任务搬进
  管理页。

## 3. 当前已完成主流程

已经实现并验证：

- 外部顶层 Markdown 只读扫描、确定性清洗和原子单元编号；
- 新增、修改、未变化、缺失和清洗歧义状态；
- Ollama、DeepSeek 与隔离的 Codex CLI 单篇分块 retry；
- Codex 使用空临时目录、read-only sandbox、ephemeral、stdin 和 JSON Schema，不加载用户
  规则、MCP 或个人配置；
- JSON/Markdown 审核导出与浏览器人工审核；
- 边界增删/合并、检索字段完全编辑、逐块 `include_in_index` 和“定位此块”；
- 保存草案后在页面批准不可变清单并原子更新活动 BM25 索引；
- 索引失败时旧索引继续服务，同一清单可重试；
- 成功后切换到完成页，显示进入/排除块数、清单 ID、活动索引 ID，并可关闭审核服务；
- 本地语料管理页、显式扫描、来源元数据、筛选排序和已有有效待审提案入口。

## 4. 已索引真实文章

当前共 6 篇、34 个 TopicChunk：

| 文章 | 版本 | 索引块数 |
|---|---|---:|
| 一代版本一代神，比阶层更重要的，是分析社会阶段 | `version_c3ffee80ff2431d38fe155da` | 7 |
| 示例文章壬：收入路径的比较 | `version_bb6ab5d5bbb4f87310845259` | 6 |
| 示例文章丑：阅读策略的边界 | `version_fcaf34f27d75dd1b2fc51575` | 4 |
| 做事的，为什么斗不过那些做人的？ | `version_5eb2b5cc65f7960a79b60811` | 6 |
| 为什么一眼看破事物本质的人，照样过不好这一生 | `version_1ee93573900097e32cf746e8` | 6 |
| 示例文章子：一个转折点，你这辈子真正的考试，才刚刚开始 | `version_deba2f80c23a160d4fbccf9d` | 5 |

每篇文末付费文章目录均由用户在审核中排除索引；短文《示例文章丑：阅读策略的边界》的前言也被用户
排除。不要重新批准这些版本。

当前活动索引：

```text
index_119a21eaf38b9309f918b00946e5b518
profile: approved-topic-original-bm25-jieba-v1
topic chunks: 34
```

## 5. 本地语料管理页

启动：

```powershell
.venv\Scripts\xifeng.exe corpus manage
# 或
uv run xifeng corpus manage
```

功能：

- 显式“扫描语料更新”，不会打开页面就自动写数据库；
- 按标题或文章 ID 搜索；
- 按处理状态、已索引、未索引、有更新、来源缺失筛选；
- 按抓取时间、文件修改时间、首次导入时间、标题和状态排序；
- 显示最新来源版本、当前索引版本、提案数和索引块数；
- 只有“来源存在 + 最新版本为 `pending_review` + proposal 为 `pending_review`”时才显示
  审核入口；
- 已索引版本残留的历史待审提案不会显示可操作按钮；
- 审核入口启动独立随机端口的现有审核子服务；关闭审核页不关闭管理页；
- 页面和 API 固定绑定 `127.0.0.1`，使用随机令牌，无远程资源。

相关代码：

- `src/xifeng_rag/corpus_ui.py`
- `src/xifeng_rag/corpus_static/index.html`
- `src/xifeng_rag/corpus_static/app.js`
- `src/xifeng_rag/corpus_static/style.css`
- `src/xifeng_rag/migrations/versions/20260826_0004_article_source_metadata.py`
- `tests/test_corpus_ui.py`

来源元数据新增到 `articles`：

- `captured_at`：`index.json` 的抓取/导出时间；
- `source_modified_at`：规范 Markdown 文件修改时间；
- `source_size_bytes`；
- `source_character_count`：`index.json/chinese_char_count`；
- `metadata_synced_at`。

当前来源没有可靠发布时间或作者修订时间，因此没有伪造 `published_at` 或
`article_updated_at`。仅元数据变化不会创建新正文版本或触发分块/索引。

## 6. 真实数据库状态

最后一次真实扫描：

```text
发现 113
配置排除 1
新增 0
正文修改 0
元数据更新 112
未变化 0
缺失 0
新出现待清洗审核 0
```

数据库汇总：

```text
articles: 113
active articles: 112
articles with source metadata: 112
approved manifests: 6
topic chunks: 34
active index: index_119a21eaf38b9309f918b00946e5b518
```

数据库仍有多个 active `ArticleVersion`：6 个 indexed、6 个 needs_cleanup_review、206 个
pending_chunking。106 篇文章各有两个 active version，主要来自 cleaner 版本升级；增量切换
本来也允许旧索引版本和新待处理版本暂时共存。管理页不能使用单一 `active` 版本概念，而是
分别聚合“最新来源版本”和“当前索引版本”。

管理页聚合后的当前状态更适合用户理解：

```text
活动文章: 112
已索引: 6
待处理: 106
最新版本待清洗审核: 3
索引覆盖率: 5.4%
```

## 7. 评测集讨论结论

用户希望下一阶段开始评测集，而不是继续盲目导入文章。当前 6 篇、34 块足够跑通评测流程，
但不足以证明系统在大语料上普遍有效。

### 7.1 金标准锚点

证据必须保存为：

```text
article_id
article_version
start_paragraph
end_paragraph
claim/group
```

不要只保存当前 `topic_chunk_id`。未来切块变化时，应按稳定原子单元范围计算证据覆盖。

### 7.2 时间与语料范围

不要使用无范围的“作者对 AI 替代怎么看”。问题分为：

- `article`：明确限定某篇文章；
- `corpus_snapshot`：明确限定冻结语料快照或截至日期；
- `temporal`：明确比较早期和后期观点演变。

冻结评测集不得随着新文章静默修改。保留两条轨道：

- 回归评测：旧评测集 + 旧语料快照；
- 当前能力：新评测集版本 + 最新语料快照。

### 7.3 用户最常用的问题

用户最常问：

1. “这篇文章主要说了啥？”
2. “你认同哪些，不认同哪些？”

第一类应成为核心评测。它是整篇理解任务，不应只做全库 Top-K；路由后应读取目标文章所有
已索引话题块，再评主要观点覆盖、忠实度、引用和冗余。

第二类归入批判性阅读，不把具体立场作为唯一金标准。评测应检查：是否准确复述作者、区分
作者观点和模型分析、每个认同或质疑是否指向具体论点、理由/前提/反例是否清楚、是否杜撰
原文事实。第一版只做逻辑和前提分析；使用文章之外现实事实属于后续事实核查能力。

建议初始约 20 题：

- 6 题：每篇文章主要说了什么；
- 4 题：整篇批判性阅读；
- 4 题：具体观点追问；
- 3 题：跨文章总结或观点演变；
- 3 题：证据不足、应该拒答。

约 12 题开发集、8 题保留集；用户先提供 5–8 个真实问题，候选补充时不能先看当前检索结果。

### 7.4 广度查询

用户未来会问类似：

> 西风过去有哪些职业发展相关，且主要是“道”而不是“术”的文章？

现在登记为 `corpus_discovery` 探索题，不进入初始正式总分。等约 50% 或更多文章进入索引后，
再按冻结快照建立文章级“纳入/边界/排除”金标准，评 Precision、Recall 和排序。该任务需要
全库召回、按文章聚合、主题判断、道/术判断和证据说明，不是普通 Top-K 片段问答。

## 8. 建议下一步

推荐从评测集的最小数据结构和人工审核流程开始，不继续批量分块：

1. 设计本地 SQLite 的 `evaluation_datasets`、`evaluation_questions`、
   `evaluation_evidence`；
2. 支持 `draft/frozen`、`dev/holdout`、`article/corpus_snapshot/temporal` 和
   `answerable`；
3. 先录入 6 道“每篇文章主要说了什么”；
4. 在审核界面中用文章版本与原子单元范围标证据；
5. 冻结当前语料快照后运行第一轮 BM25 检索基线；
6. 批判性阅读先使用人工 rubric，不强行自动打唯一答案分数；
7. `corpus_discovery` 只保留非计分探索题。

这是新功能设计，按当前环境要求应先使用 brainstorming skill；用户通常倾向设计确认后直接
实现，但不要借此扩大到外部评测候选生成。若要把少量完整文章发给外部模型生成评测候选，
仍需逐篇披露并取得明确二次确认。

## 9. 测试与浏览器验收

最后一次验证：

```text
pytest: 68 passed
Ruff: passed
Pyright: 0 errors
node --check: corpus/review app.js passed
git diff --check: passed（只有既有 LF/CRLF warning）
```

使用 WSL Ubuntu 的 `agent-browser 0.27.1` 验收管理页：

- 初始页面正常加载，无浏览器错误；
- 搜索“高考”得到 1 条；
- “已索引”筛选得到 6 条；
- “待清洗审核”筛选得到 3 条；
- 点击真实“扫描语料更新”成功，正文修改为 0，活动索引未变化；
- QA 发现并修复“已索引版本的历史待审提案仍显示审核按钮”和“来源缺失仍可审核”的问题。

本机不得使用原生 Windows `agent-browser`，它会出现 CDP channel closed；必须按 `AGENTS.md`
使用 WSL Ubuntu，并先加载 nvm。

## 10. Git 状态

最近相关提交：

```text
11d1d3f  📝 Design local corpus management dashboard
7dbae63  📝 Design review completion state
88f7aed  📝 Design review approval and indexing action
404e471  📝 Add chunk location interaction
26a5e17  📝 Design local chunk review UI
f16d119  📝 Design isolated Codex CLI chunk provider
c06d2ed  📝 Design reusable dotenv secrets
2d3429a  📝 Refine semantic chunking with article outlines
```

重要：工作区仍不是 clean。完整 V0 实现的 `src/`、`tests/`、`pyproject.toml`、`uv.lock` 等从
早期开始就是未追踪文件；README、AGENTS 和多个设计文档也有未提交修改。不要 reset、
checkout、clean 或宽泛提交整个工作区。现有内容属于用户和此前会话，应逐文件保护。

本次设计文档已单独提交；管理页实现和测试位于上述未追踪实现树中，没有为了本功能强行创建
一个会夹带全部 V0 的大提交。

## 11. 隐私与安全提醒

- 外部语料目录只读；不得修改、移动或删除其中任何文件；
- 只使用顶层 Markdown 作为规范正文，不索引 `linked/` 或 HTML 副本；
- 不提交 `data/`、审核文件、索引、付费正文或包含正文的运行日志；
- 不批量调用外部模型处理完整文章；
- 不批准数据库中遗留提案；
- 评测候选全文外发也必须逐篇选择、披露和二次确认；
- Contextual Retrieval、Embedding、Reranker 和默认预处理保持本地；
- DeepSeek Generator 只能由用户显式触发，并只发送最终必要片段。

## 12. 可复制给新会话的起始请求

```text
请先阅读 AGENTS.md 和
docs/handoffs/2026-08-26-corpus-dashboard-evaluation-handoff.md，检查当前 git/database 状态，
不要批量运行 chunks propose，也不要批准现有提案。语料管理页已完成；我们从评测集的数据
结构、人工证据标注和“整篇总结/批判性阅读”评测流程继续。
```
