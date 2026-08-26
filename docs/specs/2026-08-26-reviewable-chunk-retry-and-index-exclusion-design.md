# 可审核分块重试与索引排除设计

状态：已确认；审核重试主流程已实现，outline-first 默认 Prompt 调整待实现。

## 1. 背景

V0 已能使用本地 Ollama 为完整文章生成单层话题分块提案，并在人工审核后生成
TopicChunk。真实短文验收表明，继续让 4B 模型同时判断语义边界和
`substantive/boilerplate` 容易在“过度排除”和“全部保留”之间摆动；模型还可能把一条
完整原则拆成多个子话题。

这类不理想提案预计只占少数。系统不再通过增加三分类和堆叠默认提示词解决所有情况，
而是保留低成本默认路径，并为单篇疑难文章提供人工审核、换模型和一次性自定义 Prompt
能力。

## 2. 目标

1. 默认流程继续使用当前本地 Ollama 模型，不增加日常审核成本。
2. 用户可以对指定单篇文章选择其他本地或外部模型，生成不覆盖历史的并列提案。
3. 用户可以为一次重试完整改写分块指令，且不影响下一篇文章或后续重试。
4. 用户可以在审核文件中调整原子单元边界，并明确决定每个分段是否进入索引。
5. 不进入索引的分段仍保留在批准清单中，以维持全文覆盖和审计能力。
6. 外部模型调用必须是单篇、逐次确认、可审计且不会在普通日志泄露正文。

## 3. 非目标

- 不引入 `substantive/article_meta/boilerplate` 三分类。
- 不让模型自动决定内容的重要程度。
- 不开发动态插件发现、通用模型注册平台或工作流引擎。
- 当前阶段不开发 Svelte 审核界面；JSON/Markdown 与 CLI 是唯一操作界面。
- 不提供批量外部分块。
- 不改变原子单元解析、不可变批准清单或 TopicChunk 引用规则。

## 4. 主流程

```text
单篇文章
  -> 默认本地模型生成提案
  -> 导出指定提案的审核文件
  -> 用户选择：
       - 直接修改边界或索引决定
       - 换本地模型重新生成并列提案
       - 二次确认后换外部模型重新生成并列提案
  -> 本地 Schema、边界和完整覆盖校验
  -> 用户批准选中的提案
  -> 只为 include_in_index=true 的分段生成 TopicChunk
```

默认 `chunks propose` 行为保持本地优先。专门的单篇 `chunks retry` 命令负责模型切换和
自定义 Prompt，使批量默认路径与可能外发正文的路径在命令层明确分离。

## 5. 审核数据模型

新审核话题使用行为明确的布尔字段：

```json
{
  "start_paragraph": 1,
  "end_paragraph": 10,
  "title": "付费门槛与阅读提醒",
  "summary": "……",
  "keywords": ["……"],
  "include_in_index": false
}
```

`start_paragraph/end_paragraph` 继续沿用兼容字段名，但业务含义仍是原子单元范围。用户只能
在已有原子单元之间调整边界，不能在半句话中间任意切割。

规则如下：

- 新模型提案规范化后，所有分段默认 `include_in_index=true`。
- 模型不再输出内容角色或索引决定。
- 审核者可以修改边界、标题、概括、关键词和 `include_in_index`。
- 所有分段必须连续、无交叠、无遗漏地覆盖全文。
- 批准清单保存全部分段，包括 `include_in_index=false` 的分段。
- 只有 `include_in_index=true` 的分段生成 TopicChunk。
- 全部分段均为 `false` 时禁止批准。

旧提案继续可读：缺少新字段时，`substantive` 映射为 `true`，`boilerplate` 映射为
`false`；若两个字段同时存在，以显式的 `include_in_index` 为准。新批准清单不再写入
`content_role`。现有 boilerplate 块数和正文占比安全阈值随新模型 Schema 一同移除，
因为排除决定已经转移给人工审核。

## 6. 模型与 Prompt

现有 `StructuredGenerator` 窄边界继续复用，只增加两个显式适配器：

- `OllamaClient`：本地默认路径，可按 JSON Schema 请求结构化输出。
- `OpenAICompatibleClient`：仅供本地配置的外部 profile 使用；JSON mode 或结构化能力由
  具体服务商决定。

这两个适配器不是公开插件系统。无论服务端提供何种结构化能力，Pydantic、结束单元严格
递增、全文覆盖和越界校验始终在本地执行。服务端只保证 JSON 对象时，不能把它视为通过
本地 Schema 校验。

每次重试先载入版本化默认分块指令。用户可以用 `--prompt-file` 完整替换本次指令；下一次
调用重新载入默认指令。文章标题和编号原子单元作为本次请求的数据载荷加入，响应 Schema
作为传输约束加入，二者不写回用户的 Prompt 文件。

默认 Prompt 采用 outline-first 决策顺序：

1. 先概括全文主旨，并查找 Markdown 标题、序号、“第一点/第二点/第三点”、作者声明的
   话题数量及总结信号；
2. 根据这些信号建立最少且完整的顶层提纲；显式篇章结构优先于局部语义变化；
3. 前言、并列原则和总结属于文章根级下的同级顶层章节；
4. 案例、类比、数据、原因、推导和过渡默认归入所属顶层章节，不提升为新的顶层块；
5. 只有出现与现有提纲无关的独立主题时才增加顶层块；没有明确结构信号时才完全依靠
   语义推断；
6. V0 仍只输出单层顶层边界，不输出父子结构。未来父子方案以文章版本为根父级，以前言、
   原则和总结为同级章节，再把案例与解释作为原则的子块。

提案保存用户实际提供的指令文本及其哈希，但不保存渲染后含完整文章的数据载荷。默认
Prompt 仍记录版本号。provider、模型、Prompt 哈希、Schema 和本次 retry ID 共同参与提案
身份；因此：

- 默认 `chunks propose` 对相同配置继续保持幂等；
- `chunks retry` 每次都生成新的并列提案，即使模型和 Prompt 与上次相同。

## 7. CLI

默认路径保持兼容：

```powershell
uv run xifeng chunks propose <ARTICLE_ID>
```

单篇重试使用独立命令：

```powershell
uv run xifeng chunks retry <ARTICLE_ID> `
  --provider <PROFILE> `
  --model <MODEL_ID> `
  --prompt-file <PATH>
```

约束：

- `ARTICLE_ID` 必填且只能有一个。
- 命令解析并显示实际选中的最新 active article version。
- `--provider` 指向本地配置 profile；默认仍为 Ollama。
- `--model` 可覆盖 profile 的默认模型。
- `--prompt-file` 可省略；省略时使用版本化默认 Prompt。

并列提案要求审核导出能够明确选择 proposal：

```powershell
uv run xifeng chunks export-review <ARTICLE_VERSION> --proposal <PROPOSAL_ID>
```

当一个版本只有一个待审提案时可以省略 `--proposal`；存在多个待审提案时必须明确指定。
导出文件名包含 proposal ID，避免同一文章版本的审核文件相互覆盖。`chunks list` 应显示
每个提案的 provider、模型、Prompt 标识、状态和创建时间，但不显示正文或完整 Prompt。

## 8. 外部 profile 与授权

外部服务使用普通本地配置，不在仓库中保存密钥：

```yaml
chunking:
  external_profiles:
    deepseek:
      base_url: https://api.deepseek.com
      api_key_env: DEEPSEEK_API_KEY
      default_model: deepseek-chat  # 示例；以本地账号实际可用模型为准
```

外部 profile 的首次实现采用 OpenAI-compatible Chat Completions 形态。配置只描述明确需要的
端点、密钥环境变量和默认模型，不建设模型发现或插件注册机制。

每次外部 `chunks retry` 在任何网络请求前显示：

- 文章标题与选中的 article version；
- 清洗后字符数和原子单元数；
- provider profile、base URL 和模型；
- 将发送完整清洗后正文的明确提示；
- 本次操作最多可能产生的 API 请求次数。

用户确认一次，只授权当前 retry 操作及提示中声明的有限结构纠错请求。再次运行命令必须
重新确认，不提供“以后不再提醒”。拒绝时不读取密钥、不创建提案且不发起网络请求。

发送内容只包括用户本次指令、文章标题、清洗后的编号原子单元和结构化输出要求。不得发送
原始 Markdown、已清除内容、其他文章或数据库数据。

当前 `AGENTS.md` 的“语义分块必须本地执行”在本功能及保护措施实现前继续有效。实现该设计
时，必须在同一改动中把它更新为“默认本地；单篇显式选择并二次确认时允许外部分块”的新
边界，并同步更新架构、分块与技术栈文档。

## 9. 存储与隐私

`chunk_proposals` 需要能够记录：

- provider profile；
- 实际模型 ID；
- 默认 Prompt 版本或自定义 Prompt 文本与哈希；
- Schema 版本；
- retry/run ID；
- 不含正文的生成元数据和安全诊断。

数据库不得保存渲染后完整请求、外部服务原始响应或 API 密钥。成功时只保存通过校验的规范
topics；失败时只保存 attempt、错误类型和不含输入/响应正文的结构诊断。普通日志同样不得
输出完整 Prompt、文章正文、原始响应或密钥。

旧数据库提案的 provider 视为 `ollama`，自定义 Prompt 和 retry ID 可为空。迁移不能修改
现有八个提案的 topics JSON 或状态。

## 10. 失败与状态

- API 密钥或 profile 缺失：在确认后的发送准备阶段报错，不创建可审核提案。
- 用户拒绝外发：正常取消，数据库不变。
- 网络、超时、JSON、Schema 或覆盖失败：保留安全失败诊断，不改变文章版本和其他提案。
- 有限纠错耗尽：提案状态为 `failed`，不能导出为可批准审核文件。
- 一个并列提案失败或被放弃：不覆盖现有待审提案，也不影响已批准版本。
- 批准时正文哈希、文章 active 状态和 proposal 归属仍须与审核文件匹配。
- 外部模型结果与本地模型结果拥有相同审核和批准权限，不存在自动批准特例。

## 11. 短文人工金标准

文章《示例文章丑：阅读策略的边界》的 cleaner v2 版本
`version_fcaf34f27d75dd1b2fc51575` 使用以下人工边界：

| 单元范围 | 人工定义 | `include_in_index` |
|---|---|---:|
| 1–10 | 付费门槛与阅读提醒 | `false` |
| 11–16 | 交易是思想与艺术，而非机械技术 | `true` |
| 17–44 | 第一原则：低买高卖与安全周期 | `true` |
| 45–81 | 第二原则：用大量尝试平衡翻台率与风险 | `true` |
| 82–106 | 第三原则：在可对冲环境中交易 | `true` |
| 107–126 | 三项原则的总结、局限与个人反思 | `true` |

这份金标准不提交真实正文，只记录文章版本、边界、人工标题和索引决定。它用于判断模型边界
是否接近人工期望，以及批准后是否只生成五个 TopicChunk。长文回归样本改为《一代版本一代
神，比阶层更重要的，是分析社会阶段》，article ID 为
`sn:a4f20486731cf34a929953f7d136bed5`，cleaner v2 版本为
`version_c3ffee80ff2431d38fe155da`。该长文只生成并导出新提案，在用户按近期阅读记忆审核前
不批准，也不把模型输出视为冻结金标准。

## 12. 验证

自动化测试至少覆盖：

1. 默认 `chunks propose` 不产生外部网络请求。
2. `chunks retry` 强制单篇文章，不能无参数或传入多个 ID 触发外部处理。
3. 外部 retry 未确认时零请求、零提案。
4. 确认信息包含实际文章版本、字符数、provider、模型和最大请求次数。
5. 外部请求只包含选中文章的清洗后数据。
6. 自定义 Prompt 只影响当前 retry，下一次恢复默认值。
7. provider、模型、Prompt 和 run ID 被正确记录；并列提案不互相覆盖。
8. 多提案版本必须显式选择导出对象，审核文件名不会覆盖。
9. 旧 `content_role` 正确映射到 `include_in_index`。
10. `false` 分段保留在批准清单，但不生成 TopicChunk。
11. 全部为 `false`、缺失覆盖、交叠或越界时拒绝批准。
12. 外部返回非法 JSON、错误结构或不完整覆盖时不能批准。
13. 数据库、普通日志与失败诊断不包含文章正文、渲染后 Prompt、原始响应或 API 密钥。
14. 短文金标准批准后恰好生成五个 TopicChunk，范围分别为 11–16、17–44、45–81、
    82–106 和 107–126。
15. outline-first Prompt 在短文中优先识别前言、三个显式原则和总结，不把原则内的案例或
    子论点提升为顶层块；在指定长文上检查顶层结构完整性和人工修改量，避免通过强制少块
    合并独立主题。

真实验收只允许对用户明确选择的单篇文章运行。不得批量运行 `chunks propose`，不得批准当前
已有的真实提案，直到用户根据本规范完成实际审核。
