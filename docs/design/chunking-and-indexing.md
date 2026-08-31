# 语义分块、上下文化与增量索引设计

状态：实现基线 v0.3；默认单层话题分块、人工审核和基础索引已实现，分层子块作为显式实验
Profile 存在，不改变批准边界。少数单篇文章允许显式确认后使用外部模型重试。

## 1. 规范语料

- 当前语料根目录为 `../test-reverse-wechat-official-accounts-paid-article/saved_articles`，由新项目根目录解析。
- 实际路径通过 `corpus.source_dir` 配置；启动时记录解析后的绝对路径和文章数量。
- 源目录只读，任何清洗、分块或索引过程都不得修改原始 Markdown。
- 只索引 `saved_articles/*.md`。
- `saved_articles/linked/` 是派生副本，不进入索引。
- 正文论述中实际引用的 Markdown 链接保留锚文本；URL 存入关系元数据但不参与 Embedding。
- 文末重复的付费文章目录、备用号推广区及其中链接从规范正文完全移除，不参与分块、概括、BM25、Embedding 或未来图关系抽取。
- HTML 用于必要时核对或展示，不作为重复正文写入索引。
- `index.json` 的 `captured_at`、来源字符数及规范 Markdown 的文件修改时间作为结构化来源
  元数据保存。仅这些元数据变化时不创建新正文版本，也不触发重新分块或索引。
- 规范化过程按非空物理行及中英文句末符号生成细粒度原子单元，保留单元顺序、有效的正文内链接和回溯原始 Markdown 的精确位置映射。Markdown 标题、独立链接行保持完整；英文句点、URL 和小数不作为通用切分边界。

### 1.1 确定性清洗

清洗发生在单元编号和 AI 分块之前，不修改原始 Markdown：

```text
原始 Markdown
  -> 解析块级结构
  -> 识别文末目录或推广区
  -> 保留正文内引用的锚文本并剥离 URL
  -> 生成规范正文与源位置映射
  -> 按物理行和安全句末边界生成原子单元
  -> 分配稳定单元编号与源位置
```

清洗规则采用保守、可审计的确定性实现，不让 AI 自由删除内容。每次清洗至少记录：

```text
source_content_hash
cleaner_version
removed_source_ranges
removed_character_count
cleanup_warnings
canonical_content_hash
```

第一阶段提供 `corpus manage` 本地语料管理页。它按文章聚合最新来源版本、当前索引版本、
待审提案和索引块数量，并通过显式按钮触发增量扫描；不自动扫描或调用分块模型。

文末目录通过“接近文章尾部、连续大量微信文章链接、备用号或付费文章目录提示”等组合信号识别，不能仅因出现一个链接就截断。高置信度规则命中时自动移除；存在歧义时标记 `needs_cleanup_review`，不生成分块提案，避免静默删除正文。

`source_content_hash` 用于发现外部文件发生过变化；`canonical_content_hash` 用于判定可检索正文是否变化。若只更新了已排除的文末目录且规范正文未变，复用原批准分块与索引。清洗规则升级时，即使源文件未变，也必须重新生成规范正文并比较哈希。

## 2. 第一阶段单层话题分块

付费文章篇幅差异较大，早期完整付费文章也可能只有约三千字。默认批准单元仍由本地 AI 按
语义边界提出单层话题块；文章长度不能作为正文是否完整的判断依据。分层实验只在批准后的
TopicChunk 内确定性生成子块，不要求用户审核第二套语义边界，也不替换默认 Profile。

处理步骤：

1. 清洗并规范化 Markdown，为物理行和安全句末边界形成的原子单元分配稳定编号，同时保存精确原始位置。
2. 将文章标题、发布时间和带编号的完整原子单元交给默认本地模型，或交给用户逐次确认的单篇外部重试模型。
3. 模型只返回每块的结束单元、话题标题、内容概括和关键词，不复制或改写正文，也不决定是否进入索引。
4. 程序推导连续起点，并根据单元范围从规范正文提取 `original_text`。
5. 程序验证结束单元严格递增、完整覆盖且没有越界。
6. 生成待审核文件，用户确认后才允许建立索引。

连续分块的起点由前一块终点唯一决定，因此模型不再输出冗余起点。结束边界必须严格
递增、没有越界且最后覆盖全文；程序不得自动猜测或修改模型给出的结束边界。

模型使用能完整表达文章结构的最少必要话题块。只在主题、论证对象、核心观点或论述阶段明显
转换时建立新块，同一条连续论证不因篇幅机械拆分；短而连贯的文章允许只有一个话题块。默认
批准流程不设置常见块数、目标字符数或字符数警告，也不增加 overlap。约 400 字符的确定性
子块只属于独立分层检索实验，不能反向改变批准清单。

默认 Prompt 采用 outline-first 顺序：先识别全文主旨以及标题、序号、作者声明的话题数量、
前言和总结等显式结构，再选择顶层边界。显式篇章结构优先于局部语义变化；案例、类比、
数据、原因、推导和过渡归入所属顶层章节，不能仅因可以单独命名就提升为顶层块。没有明确
结构信号时才完全依靠语义转换推断。V0 仍只输出单层顶层边界，不提前输出父子结构。

模型输出采用固定结构：

```json
{
  "topics": [
    {
      "end_unit": 18,
      "title": "房地产周期与政策目标",
      "summary": "作者分析……并认为……",
      "keywords": ["房地产", "政策", "周期"]
    }
  ]
}
```

模型提案规范化后，每块默认 `include_in_index=true`。是否进入索引完全由用户审核决定，
不再要求模型判断 `substantive/boilerplate`，也不设置样板块数量或字符占比阈值。话题标题、
概括和关键词属于模型生成的检索辅助信息，不是作者原话。

## 3. 审核工作流

AI 分块结果不能直接进入可查询索引：

```text
发现新增或修改文章
  -> 默认本地 AI 生成分块提案
  -> 可选：单篇换本地或外部模型生成并列提案
  -> 结构与覆盖校验
  -> pending_review
  -> 用户确认或修改
  -> approved manifest
  -> Embedding 与 BM25 建库
```

第一阶段生成两种审核产物：

- JSON：机器可读，保存单元边界、`include_in_index`、标题、概括、关键词和生成信息；
- Markdown：方便阅读，按话题展示索引决定、字数、概括、开头与结尾原文。

用户可以批准提案，也可以修改边界、索引决定或概括后批准。批准后将人工确认的完整话题 JSON 保存为不可变的 `approved chunk manifest`。`include_in_index=false` 的内容保留在清单中但不生成 `TopicChunk`；全部设为不索引的文章禁止批准。只有已批准的文章版本参与索引；原文或清洗器版本变化后，新版本必须重新生成并确认。审核期间，已批准的旧版本继续提供查询服务。

第一阶段提供 `chunks review <VERSION> --proposal <ID>` 本地审核页作为 JSON 手工编辑的辅助。
它固定绑定 `127.0.0.1`，使用一次性令牌，可显示编号原子单元、合并或拆分边界、编辑标题、
概括和关键词，并切换 `include_in_index`。保存结果写入独立 `.draft.json`，不覆盖原始导出、
不修改数据库提案。保存成功后，用户可以二次确认批准不可变清单并更新活动索引；批准和
索引使用各自的事务边界，索引失败不回滚有效清单，也不替换旧活动索引，并允许直接重试。
模型切换与自定义 Prompt 仍通过 `chunks retry` 执行，以保留完整文章外发披露和二次确认。

自动校验至少包括：

- 所有规范单元恰好被覆盖一次；
- 块之间连续且没有交叠；
- 起止编号存在且顺序正确；
- 标题和概括不为空；
- 标题、概括和关键词符合结构约束；语义边界质量由用户审核确认；
- 至少一个话题块设置为 `include_in_index=true`。

## 4. 索引字段与文本

每个已批准话题块分别保存：

```text
article_title
published_at
paragraph_range  # 兼容字段名，V0 v2 起表示原子单元范围
topic_title
context_summary
keywords
original_text
```

文章还分别保存微信原始 `published_at` 与本地 `captured_at`。前者从配套 HTML 的
`ori_create_time` 确定性读取，HTML 正文不进入索引；后者用于同步审计，不能冒充发布时间。

文章整体概括使用独立的 `ArticleOverview` 审核记录，不修改不可变批准清单。已有文章的草稿
输入为批准清单中的全部话题元数据，包括 `include_in_index=false` 的话题，不读取
`TopicChunk.original_text`。概括用于文章发现和回顾，不能作为作者原话或回答引用。

其中精确发布时间主要作为结构化元数据，用于过滤、排序和时间分析。只有时间对语义理解有实际帮助时，才将自然语言形式的时间背景加入 Embedding 文本，避免每个块重复精确日期造成噪声。

第一阶段至少保留两个可对照的 `EmbeddingTextBuilder`：

### 4.1 原文方案

```text
<original_text>
```

### 4.2 AI 上下文化方案

```text
文章：《article_title》
话题：<topic_title>
内容概括：<context_summary>
关键词：<keywords>
原文：<original_text>
```

AI 信息用于增强召回，不能替代原文。BM25 是否加入 AI 生成字段应作为独立实验变量；最初可使用原文 BM25，避免 Dense 与关键词检索同时变化而无法归因。

## 5. 分块模型与提示词

语义分块和概括会读取接近完整的付费文章，因此默认必须使用本地模型。少数单篇文章在默认提案不理想时，用户可以显式选择外部 profile、模型和一次性完整 Prompt；程序显示实际文章版本、清洗后字符数、服务地址、模型和最大请求次数，经逐次二次确认后才发送该篇清洗后正文。该路径禁止批量调用和自动批准。

提示词、输出 Schema 和校验规则是项目核心资产，必须版本化。可另外提供 Agent Skill 作为人工触发外壳，但 Skill 不得成为唯一实现路径。每份提案至少记录：

```text
source_content_hash
canonical_content_hash
cleaner_version
segmentation_model_id
segmentation_provider_id
model_runtime_and_quantization
prompt_version
custom_prompt_hash
schema_version
generation_parameters
generated_at
```

模型能力目标是稳定识别话题边界并生成忠实、简短的概括；第一阶段不要求模型推断知识图谱关系。

默认 `chunks propose` 对相同版本和配置保持幂等。`chunks retry` 强制指定单篇文章，每次生成
新的并列提案；自定义 Prompt 只对本次调用有效。外部 JSON mode 不能替代本地 Pydantic、
严格递增和完整覆盖校验。数据库与普通日志不得保存渲染后含完整正文的请求、外部原始响应
或 API 密钥。个人模式还可使用 `codex-cli` provider 复用当前 ChatGPT 登录；它必须在空临时
目录以只读沙箱、ephemeral 会话和显式 JSON Schema 运行，正文仅通过 stdin 传入。该路径仍
视为外部推理，程序不能替用户证明 ChatGPT 账户的训练开关状态。

V0 默认运行配置为：

```text
runtime: Ollama standalone CLI
model: qwen3:4b-instruct-2507-q4_K_M
context_window: 32768
output: JSON Schema constrained
validation: Pydantic + coverage validators
ollama_home: D:\ai-tools\ollama
model_store: D:\ai-models\ollama
```

Ollama 只作为按需启动的本地离线预处理器，不注册为项目的交互式 Generator。项目通过一个窄的本地结构化生成边界调用它，不建设通用模型服务抽象。模型输出失败、Schema 不合法或覆盖校验失败时允许有限次数重试；仍失败则保留诊断信息并进入人工处理状态，不自动建立索引。

本机验收应持续覆盖少量长短不同的真实文章，并记录实际 token 数、显存峰值、耗时、JSON
合法率和人工修改量。只有扩大样本后的实测质量不足时，再比较 `qwen3.5:4b` 或
`llama-cpp-python`，不因单篇失败同时实现多个运行后端。

## 6. 公平对照

第一阶段优先比较：

```text
A. 已批准话题块 + 原文 Dense + 原文 BM25
B. 已批准话题块 + AI 上下文化 Dense + 原文 BM25
C. B + 本地 Reranker
```

父子分块已作为手动实验 Profile 加入：保留已批准 TopicChunk 为父块，在父块内按原子单元
确定性生成约 400 字符子块；父、子 Hybrid 并行召回后统一以子块重排，再在同父块内串连相邻
子块。该实验不修改批准边界，真实命中与扩展范围分开评测。详细规则见
[`docs/specs/2026-08-29-layered-child-retrieval-design.md`](../specs/2026-08-29-layered-child-retrieval-design.md)。

后续再与固定长度小块或 contextual BM25 对比。所有实验使用同一语料快照和评测问题；比较索引文本或 Embedding 时保持已批准边界不变，比较分块策略时保持 Embedding、索引文本构造、召回数量等其他变量不变。一次只改变一个主要变量。

## 7. 稳定 ID

- `article_id` 优先使用现有 `index.json` 中的 `sn:<hex>`。
- 缺少 `sn` 时，使用规范化来源标识生成稳定 ID。
- `article_version` 由规范正文哈希确定；原始文件哈希另行保存用于变化检测。
- `chunk_id` 由文章版本、分块方案版本、批准清单版本及原子单元范围共同确定。
- `artifact_id` 还包含 EmbeddingTextBuilder、Prompt 和 Embedding 模型版本。

稳定 ID 用于引用、缓存、增量更新和旧实验复现。

## 8. 增量更新

```text
扫描源文件
  -> 计算原始文件哈希并生成规范正文
  -> 计算规范正文哈希
  -> 分类为新增 / 修改 / 未变化 / 缺失
  -> 为新增或修改版本生成待审核提案
  -> 批准后构建新的完整 Index Build
```

- 新增：生成 `pending_review` 提案，批准前不进入索引。
- 修改：旧版本保持 active；新版本批准并完整建库成功后，再停用旧版本。
- 同一文章有多个 active 版本时，分块命令先选择创建时间最新的 active 版本，再判断其是否
  处于 `pending_chunking` 或 `pending_review`；不能回退并为旧待处理版本生成提案。
- 仅被排除的文末目录变化：规范正文哈希不变，复用原批准结果和索引。
- 未变化：复用已批准清单和话题块；当前完整 Hybrid Build 仍会重新编码纳入该 Build 的全部
  已批准 TopicChunk。
- 缺失：标记 inactive，不立即物理删除，保留实验可追溯性。
- 分块 Prompt 或模型改变：即使最新文章版本已在 `pending_review`，仍产生并列的新提案，
  不覆盖旧提案或批准结果；当前 Prompt/Schema/模型提案已存在时直接复用，不重复调用模型。
- Embedding 模型或维度改变：建立新索引，不覆盖旧索引。
- Reranker 或 Generator 改变：不触发索引重建。

当前增量能力覆盖语料扫描、正文版本、提案与批准状态，但尚未实现向量和 BM25 派生产物的
逐文章增量写入。真正的增量索引需要在语料扩大后单独实现，并验证新增、修改、未变化、缺失
以及失败回滚；在完成前不得把当前完整 Build 描述为增量向量建库。

缓存键至少包含：

```text
canonical_content_hash
+ cleaner_version
+ segmentation_model_and_prompt_version
+ approved_manifest_version
+ embedding_text_builder_version
+ embedding_model_version
```

## 9. 引用规则

生成答案只能引用 `original_text`。引用至少展示文章标题、原子单元范围和原文片段，并支持打开本地文章定位对应源位置。AI 生成的话题标题、概括、关键词及未来图谱推断都必须与作者原话明确分离。
