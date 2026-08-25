# V0 MVP 实现设计

状态：已批准，待实现。

## 1. 目标与范围

V0 建立一条可运行、可审核、可评测的本地 BM25 基线：

```text
语料扫描 -> 确定性清洗 -> 本地 AI 话题分块 -> 人工批准
-> BM25 原子建库 -> 检索与原文引用 -> 基础检索评测
```

V0 处理外部只读语料目录顶层的 Markdown 文件。完整正文、清洗、分块、索引和评测都留在本机。V0 不实现 Dense、RRF、Reranker、Generator、FastAPI、前端或图检索。

完成标准：

- CLI 可以分阶段执行完整流程，并在每一步显示可操作的状态或错误；
- 未经批准的文章版本不能进入索引；
- 搜索结果包含稳定原文引用和统一候选字段；
- 自动化测试覆盖正常流程、增量分类、隐私边界和失败回滚；
- 少量真实文章可通过本机 Ollama 验收，不自动批处理全部语料；
- 自造评测夹具可以计算 P@K、Recall@K、MRR、nDCG 和延迟。

## 2. 实现方式

采用“分层核心 + 显式 CLI 状态机”。各阶段通过领域模型和 SQLite 状态衔接，模型输出、审核文件和 BM25 索引都是可检查的明确产物。

没有采用以下方案：

- 文件产物作为主状态源：不利于事务、一致性和增量更新；
- 单命令流水线：不利于人工审核、断点恢复和失败诊断；
- 通用工作流或插件平台：超出 V0 的实际变化点。

## 3. 模块划分

```text
src/xifeng_rag/
├── config/          # YAML、环境变量和路径解析
├── corpus/          # 扫描、Markdown 解析、清洗和段落编号
├── domain/          # Pydantic 领域模型、稳定 ID 和状态枚举
├── storage/         # SQLAlchemy 映射、事务和仓储
├── chunking/        # Ollama、JSON Schema、覆盖校验和审核产物
├── indexing/        # 中文分词、bm25s 建库及原子发布
├── retrieval/       # 查询、统一候选和原文引用
├── evaluation/      # 检索指标和运行记录
└── cli.py
```

每个模块只依赖更底层的领域模型或窄接口。CLI 负责编排和呈现，不包含清洗、校验或检索业务逻辑。

## 4. CLI 边界

V0 提供以下命令：

```text
xifeng init
xifeng doctor
xifeng corpus scan
xifeng corpus resolve-cleanup ARTICLE_VERSION
xifeng chunks propose [ARTICLE_ID...]
xifeng chunks list
xifeng chunks export-review ARTICLE_VERSION
xifeng chunks approve MANIFEST.json
xifeng index build
xifeng search "问题" --top-k 5
xifeng eval run EVALSET.json
```

`init` 创建项目本地运行目录、配置模板和数据库结构。`doctor` 检查 Python、配置、语料目录、Ollama 服务和模型，但不修改语料或自动发送正文。

`corpus scan` 在启动模型或建库前输出解析后的语料绝对路径和顶层 Markdown 数量。源目录不存在或没有文章时以非零退出码结束，不创建空索引。

`corpus resolve-cleanup` 允许用户对歧义清洗选择保留全部，或明确指定从某一段落开始截断。决定写入审计记录。

`chunks approve` 只接受导出的、与当前文章版本及提案匹配的 manifest。用户可在批准前修改边界、标题、概括和关键词，所有修改仍须通过覆盖校验。

## 5. 语料扫描与确定性清洗

配置使用 `corpus.source_dir`，默认值为：

```text
../test-reverse-wechat-official-accounts-paid-article/saved_articles
```

路径相对于项目根目录解析。只读取目录顶层 `*.md`；忽略 HTML 和 `linked/`。任何流程都不得修改、移动或删除源文件。

清洗顺序：

1. 解析 Markdown 块级结构并保留源字符范围；
2. 将正文链接替换为锚文本，同时将 URL 记录为关系元数据；
3. 使用组合信号识别尾部目录或推广区；
4. 输出规范段落、删除范围、字符数、警告和两个内容哈希；
5. 为规范段落分配从 1 开始的稳定顺序编号。

只有同时满足接近文章尾部、连续大量文章链接、明确目录或备用号提示等高置信度组合规则时才自动截除。单个链接、普通参考资料或正文内引用不得触发截断。可疑但不确定的文章进入 `needs_cleanup_review`，不能生成分块提案。

清洗器规则以 `cleaner_version` 标识。阈值作为具名、可测试的版本化常量保存，首版参数属于待真实验收的基线，不视为最终效果结论。

## 6. 领域状态与结构化存储

SQLite 是结构化状态的唯一真源。核心表为：

- `articles`：稳定文章 ID、标题、规范来源标识和活动状态；
- `article_versions`：源哈希、规范正文哈希、清洗版本、正文和清洗审计；
- `paragraphs`：段落号、规范文本和原 Markdown 字符范围；
- `chunk_proposals`：模型元数据、Prompt、Schema、生成参数、状态和诊断；
- `approved_manifests`：不可变批准版本和批准时间；
- `topic_chunks`：段落范围、原文及 AI 生成的检索辅助字段；
- `index_builds`：语料快照、profile、路径、状态、耗时和失败信息；
- `evaluation_runs`：评测集标识、代码版本、指标、延迟和失败信息；
- `cleanup_decisions`：歧义清洗的人工决定和审计信息。

文章版本状态流为：

```text
discovered
  -> needs_cleanup_review | pending_chunking
  -> pending_review
  -> approved
  -> indexed
```

提案、批准清单、索引构建分别拥有独立状态，避免一个宽泛状态掩盖部分失败。

数据库使用 SQLAlchemy 2 typed declarative mapping。跨表状态切换在显式事务中执行，正常退出时提交，异常时回滚。SQLite 连接启用外键约束。

## 7. 稳定 ID 与增量更新

- 优先从可用的外部 `index.json` 映射 `sn:<hex>`；
- 当前语料目录没有 `index.json`，因此 V0 使用规范化来源标识的摘要生成 `article_id`；
- `article_version` 由规范正文哈希确定；
- `chunk_id` 包含文章版本、分块方案版本、批准清单版本和段落范围；
- Index Profile 包含分块方案、BM25 文本字段、分词器和实现版本。

扫描按源哈希和规范正文哈希分类：

- 新增：创建新版本并等待分块；
- 修改：构建新版本，旧批准版本继续服务；
- 未变化：复用既有状态和派生产物；
- 仅已排除尾部变化：规范正文哈希不变，复用批准结果；
- 缺失：标记 inactive，不物理删除。

新版本只有在批准并成功发布完整索引后才替换旧 active 版本。失败不得产生“数据库已切换、索引未完成”的可见状态。

## 8. 本地 AI 分块与审核

默认配置：

```text
runtime: Ollama standalone
model: qwen3:4b-instruct-2507-q4_K_M
context_window: 32768
temperature: 0
output: JSON Schema constrained
validation: Pydantic + coverage validators
ollama_home: D:\ai-tools\ollama
model_store: D:\ai-models\ollama
```

安装使用官方 Windows standalone 包，不安装托盘应用或开机服务。模型当前约 2.5 GB。程序按需检测或启动 `ollama serve`，通过仅绑定本机的 HTTP API 发送非流式请求，并设置 JSON Schema、`num_ctx=32768` 和最大输出限制。

模型输入为标题、可用发布时间和带编号的完整规范正文。输出只包含每个话题的起止段落、标题、概括和关键词，不复制正文。程序按段落范围回填 `original_text`。

校验要求：

- 所有规范段落恰好覆盖一次；
- 块连续、无重叠且不越界；
- 标题、概括非空；
- 关键词为有限的非空字符串列表；
- 异常短、异常长或异常多块只产生审核提醒，不擅自改变边界。

Schema 或覆盖校验失败时进行有限次数重试。重试 Prompt 只携带错误摘要和必要上下文，不把完整正文写入日志。最终失败保留诊断并停在人工处理状态。

审核导出同时包含：

- JSON：机器可读的提案和版本信息；
- Markdown：每块字数、概括、关键词及开头/结尾短预览。

批准后 manifest 不可变。文章正文变化后必须为新版本重新生成并批准。

## 9. BM25 索引与检索

V0 的 BM25 文本只使用 `original_text`，不混入 AI 标题、概括或关键词。中文分词边界封装为显式组件并记录版本；首版选用轻量本地中文分词，不访问网络。

bm25s 的 tokenized corpus 与返回 corpus 保持相同顺序和长度。返回 corpus 只保存必要标识，不复制完整付费正文；检索命中后由 SQLite 加载原文和引用元数据。

索引发布过程：

1. 选择所有 active 且 approved 的文章版本；
2. 计算语料快照和 Index Profile ID；
3. 在项目数据目录下的唯一临时构建目录建立索引；
4. 重新加载并执行最小自检；
5. 原子重命名为不可变构建目录；
6. 在数据库事务中将该构建设为 active；
7. 旧索引保留，可用于失败回滚和实验复现。

临时目录不对查询可见。构建失败时旧索引继续服务，未发布目录可安全清理。

检索统一返回：

```text
chunk_id
article_id
article_version
original_text
paragraph_range
retrieval_scores: {bm25: float}
source: [bm25]
```

CLI 另外展示文章标题、原文短片段和可定位的段落范围。AI 标题、概括和关键词可以作为辅助信息显示，但必须标为模型生成内容，不能作为作者引文。

## 10. 评测

评测集使用 JSON，至少包含：

```text
dataset_id
split
questions[]:
  question_id
  query
  relevant_chunk_ids
  should_abstain
```

`eval run` 固定查询参数并记录语料快照、Index Profile、代码版本、运行时间和逐题排名。计算：

- Precision@K；
- Recall@K；
- MRR；
- nDCG；
- 查询延迟分布。

本次实现用自造短文本夹具验证指标。真实约 20 题的冻结评测集需要后续人工建立，不能复制付费原文到仓库。

## 11. 错误、隐私与日志

- 源目录不存在或没有顶层 Markdown：报错并拒绝空建库；
- 清洗有歧义：停在审核状态；
- Ollama 未启动、模型缺失或超时：给出恢复命令，有限重试后停止；
- JSON 或覆盖校验失败：拒绝提案；
- manifest 与文章版本不匹配：拒绝批准；
- BM25 构建或自检失败：不切换 active build；
- 无 active 索引：查询错误中给出下一步命令。

普通日志只记录 ID、哈希、计数、模型名、参数版本、耗时和错误类别，不保存 API 密钥、完整 Prompt、完整正文或模型请求。审核文件和本地 SQLite 位于 Git 忽略的运行目录。

V0 不包含任何外部 Generator，也不应产生面向互联网的正文请求。Ollama 只监听本机地址。

## 12. 测试与真实验收

### 12.1 单元测试

使用自造短 Markdown 覆盖：

- 顶层文件筛选和路径错误；
- Markdown 链接锚文本保留、URL 分离；
- 高置信尾部清理和歧义判断；
- 段落编号及源字符映射；
- 内容哈希和稳定 ID；
- 分块完整覆盖、重叠、缺口和越界；
- 中文分词与 BM25 结果映射；
- P@K、Recall@K、MRR 和 nDCG；
- 引用只能来自 `original_text`。

### 12.2 集成测试

使用临时 SQLite、临时数据目录和伪 Ollama 响应覆盖：

- 新增、修改、未变化和缺失；
- 仅排除尾部变化时复用批准结果；
- 提案、导出、人工修改和批准；
- 未批准版本不能建库；
- 索引原子发布；
- 模型失败、校验失败和索引失败回滚；
- 默认路径没有外部网络请求；
- 日志不包含夹具正文。

### 12.3 本机真实验收

安装 standalone Ollama 并拉取指定模型后，从 113 篇真实文章中人工选择少量长短不同样本，记录：

- 输入 token 数；
- 推理耗时和资源峰值；
- JSON 一次合法率与重试次数；
- 话题块数量和长度提醒；
- 人工调整边界、标题或概括的数量；
- 清洗误删或漏删案例。

真实验收不会自动处理全部文章，也不会把正文、完整审核产物或索引提交到 Git。验收结果用于调整版本化规则与 Prompt，然后再决定是否扩大处理范围。

## 13. 文档同步

实现完成时更新：

- `README.md`：当前可用能力、安装和最短使用流程；
- `docs/design/technology-stack.md`：实际采用的中文分词依赖和运行方式；
- 其他设计文档：只在实现发现已批准设计需要变化时同步修改。

运行数据、真实语料衍生产物、模型文件和本地配置不得提交。
