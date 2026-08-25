# 语义分块、上下文化与增量索引设计

状态：V0 方案已确认；先确定性清洗，再采用本地 AI 单层话题分块，并由用户确认后入库。

## 1. 规范语料

- 当前语料根目录为 `../test-reverse-wechat-official-accounts-paid-article/saved_articles`，由新项目根目录解析。
- 实际路径通过 `corpus.source_dir` 配置；启动时记录解析后的绝对路径和文章数量。
- 源目录只读，任何清洗、分块或索引过程都不得修改原始 Markdown。
- 只索引 `saved_articles/*.md`。
- `saved_articles/linked/` 是派生副本，不进入索引。
- 正文论述中实际引用的 Markdown 链接保留锚文本；URL 存入关系元数据但不参与 Embedding。
- 文末重复的付费文章目录、备用号推广区及其中链接从规范正文完全移除，不参与分块、概括、BM25、Embedding 或未来图关系抽取。
- HTML 用于必要时核对或展示，不作为重复正文写入索引。
- 规范化过程保留段落顺序、有效的正文内链接和从规范段落回溯原始 Markdown 的位置映射。

### 1.1 确定性清洗

清洗发生在段落编号和 AI 分块之前，不修改原始 Markdown：

```text
原始 Markdown
  -> 解析块级结构
  -> 识别文末目录或推广区
  -> 保留正文内引用的锚文本并剥离 URL
  -> 生成规范正文与源位置映射
  -> 分配稳定段落编号
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

文末目录通过“接近文章尾部、连续大量微信文章链接、备用号或付费文章目录提示”等组合信号识别，不能仅因出现一个链接就截断。高置信度规则命中时自动移除；存在歧义时标记 `needs_cleanup_review`，不生成分块提案，避免静默删除正文。

`source_content_hash` 用于发现外部文件发生过变化；`canonical_content_hash` 用于判定可检索正文是否变化。若只更新了已排除的文末目录且规范正文未变，复用原批准分块与索引。清洗规则升级时，即使源文件未变，也必须重新生成规范正文并比较哈希。

## 2. 第一阶段单层话题分块

付费文章通常约 1.6 万字，并围绕四五个相对完整的话题展开。第一阶段不实现父子分块，而是由本地 AI 按语义边界提出单层话题块。

处理步骤：

1. 清洗并规范化 Markdown，为规范正文段落分配稳定的顺序编号，同时保存原始位置映射。
2. 将文章标题、发布时间和带编号的完整正文交给本地模型。
3. 模型只返回话题块的起止段落、话题标题、内容概括和关键词，不复制或改写正文。
4. 程序根据段落范围从规范正文提取 `original_text`。
5. 程序验证段落是否连续、完整覆盖且没有重叠。
6. 生成待审核文件，用户确认后才允许建立索引。

每篇文章通常产生 4–6 个话题块。块大小服从自然论述边界，软目标约 2,500–5,000 个中文字符，约 6,000 字符作为异常提醒阈值，而不是强制截断点。第一阶段不增加 overlap，也不恢复所谓父块。

模型输出采用固定结构：

```json
{
  "topics": [
    {
      "start_paragraph": 1,
      "end_paragraph": 18,
      "title": "房地产周期与政策目标",
      "summary": "作者分析……并认为……",
      "keywords": ["房地产", "政策", "周期"]
    }
  ]
}
```

话题标题、概括和关键词属于模型生成的检索辅助信息，不是作者原话。

## 3. 审核工作流

AI 分块结果不能直接进入可查询索引：

```text
发现新增或修改文章
  -> 本地 AI 生成分块提案
  -> 结构与覆盖校验
  -> pending_review
  -> 用户确认或修改
  -> approved manifest
  -> Embedding 与 BM25 建库
```

第一阶段生成两种审核产物：

- JSON：机器可读，保存段落边界、标题、概括、关键词和生成信息；
- Markdown：方便阅读，按话题展示字数、概括、开头与结尾原文。

用户可以批准提案，也可以修改边界或概括后批准。批准后保存不可变的 `approved chunk manifest`。只有已批准的文章版本参与索引；原文变化后，旧批准结果不再适用于新版本，新版本必须重新生成并确认。审核期间，已批准的旧版本继续提供查询服务。

自动校验至少包括：

- 所有规范段落恰好被覆盖一次；
- 块之间连续且没有交叠；
- 起止编号存在且顺序正确；
- 标题和概括不为空；
- 异常短块、异常长块和异常多块给出审核提醒。

## 4. 索引字段与文本

每个已批准话题块分别保存：

```text
article_title
published_at
paragraph_range
topic_title
context_summary
keywords
original_text
```

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

## 5. 本地模型与提示词

语义分块和概括会读取接近完整的付费文章，必须使用本地模型。该模型属于离线预处理组件，不替代交互式回答所使用的 DeepSeek Generator。

提示词、输出 Schema 和校验规则是项目核心资产，必须版本化。可另外提供 Agent Skill 作为人工触发外壳，但 Skill 不得成为唯一实现路径。每份提案至少记录：

```text
source_content_hash
canonical_content_hash
cleaner_version
segmentation_model_id
model_runtime_and_quantization
prompt_version
schema_version
generation_parameters
generated_at
```

模型能力目标是稳定识别话题边界并生成忠实、简短的概括；第一阶段不要求模型推断知识图谱关系。

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

首次实现应选取少量长短不同的真实文章做本机验收，记录实际 token 数、显存峰值、耗时、JSON 合法率和人工修改量。只有实测质量不足时，再比较 `qwen3.5:4b` 或 `llama-cpp-python`，不在 V0 同时实现多个运行后端。

## 6. 公平对照

第一阶段优先比较：

```text
A. 已批准话题块 + 原文 Dense + 原文 BM25
B. 已批准话题块 + AI 上下文化 Dense + 原文 BM25
C. B + 本地 Reranker
```

后续再与固定长度小块、父子分块或 contextual BM25 对比。所有实验使用同一语料快照和评测问题；比较索引文本或 Embedding 时保持已批准边界不变，比较分块策略时保持 Embedding、索引文本构造、召回数量等其他变量不变。一次只改变一个主要变量。

## 7. 稳定 ID

- `article_id` 优先使用现有 `index.json` 中的 `sn:<hex>`。
- 缺少 `sn` 时，使用规范化来源标识生成稳定 ID。
- `article_version` 由规范正文哈希确定；原始文件哈希另行保存用于变化检测。
- `chunk_id` 由文章版本、分块方案版本、批准清单版本及段落范围共同确定。
- `artifact_id` 还包含 EmbeddingTextBuilder、Prompt 和 Embedding 模型版本。

稳定 ID 用于引用、缓存、增量更新和旧实验复现。

## 8. 增量更新

```text
扫描源文件
  -> 计算原始文件哈希并生成规范正文
  -> 计算规范正文哈希
  -> 分类为新增 / 修改 / 未变化 / 缺失
  -> 为新增或修改版本生成待审核提案
  -> 批准后只构建受影响文章的派生产物
```

- 新增：生成 `pending_review` 提案，批准前不进入索引。
- 修改：旧版本保持 active；新版本批准并完整建库成功后，再停用旧版本。
- 仅被排除的文末目录变化：规范正文哈希不变，复用原批准结果和索引。
- 未变化：复用已批准清单、话题块和向量。
- 缺失：标记 inactive，不立即物理删除，保留实验可追溯性。
- 分块 Prompt 或模型改变：产生新的提案版本，不覆盖原批准结果。
- Embedding 模型或维度改变：建立新索引，不覆盖旧索引。
- Reranker 或 Generator 改变：不触发索引重建。

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

生成答案只能引用 `original_text`。引用至少展示文章标题、段落范围和原文片段，并支持打开本地文章定位对应段落。AI 生成的话题标题、概括、关键词及未来图谱推断都必须与作者原话明确分离。
