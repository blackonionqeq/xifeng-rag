# 语料管理页单篇 AI 分块设计

日期：2026-09-15

## 目标

在 `corpus manage` 的文章操作栏增加“AI 分块”按钮。它固定使用 `codex-cli` provider 和
`gpt-5.6-terra` 模型，为最新 active 且处于 `pending_chunking` 或 `pending_review` 的单篇文章
生成新的并列分块提案，效果与以下命令一致：

```powershell
uv run xifeng chunks retry <ARTICLE_ID> --provider codex-cli --model gpt-5.6-terra
```

页面不提供 provider、模型或 Prompt 编辑入口，也不支持批量分块。

## 交互与隐私确认

点击按钮后，管理服务只读取并展示文章标题、最新文章版本、清洗后字符数、原子单元数、
provider、模型、最大请求数、完整清洗正文将发送给 OpenAI Codex 的提示，以及 ChatGPT 训练
使用取决于账户数据控制设置的说明。此时不得启动 Codex CLI 或创建提案。

用户在模态框中逐次点击“确认并开始生成”后，页面打开独立进度标签页并调用生成接口。取消
或关闭弹窗必须保持数据库不变。生成期间禁用该行按钮；同一文章版本的并发生成由服务端拒绝。

生成成功后创建新的并列提案，进度标签页直接跳转到该提案的现有分块审核页，管理页刷新文章
状态。已有待审提案不阻止再次生成。

## 实现边界

CLI 与管理页共同使用一个单篇 Codex 分块业务模块。该模块负责选择和再次校验最新 active
可重试版本、构造披露信息、执行 Codex 预检，并调用现有 `propose_chunks`。页面不得通过子进程
嵌套执行 `uv run xifeng`，也不得复制一套分块规则。

管理服务新增两个仅接受带会话令牌请求的 POST 接口：

- `/api/chunk-retries/disclosure`：返回单篇外发范围，不调用 Codex；
- `/api/chunk-retries/generate`：要求 `confirmed: true` 和披露时的 version ID，重新校验后生成
  提案并启动现有审核服务。

固定模型是该页面操作的产品约束，不读取 `codex_cli.default_model` 替代。Codex CLI 的可执行
文件、超时和最大重试数仍读取现有本地配置。

## 状态与失败处理

按钮只在文章来源存在，且最新版本满足 `chunks retry` 状态约束时显示。生成前若文章已缺失、
出现更新版本或状态已改变，接口返回冲突并要求刷新。Codex 预检、生成或本地结构校验失败时，
不得改变已有批准清单或索引；失败提案沿用现有 `propose_chunks` 的安全诊断规则。

普通日志和 HTTP 错误不得包含正文、完整 Prompt 或 Codex 原始响应。管理服务关闭时继续关闭
其启动的审核子服务。

## 验证

自动测试覆盖按钮可见条件、固定 provider/model、披露不调用生成器、明确确认、成功生成并
打开正确提案、重复请求、过期版本、令牌校验、安全错误和静态资源无远程依赖。CLI 原有取消
和重试测试继续通过，以证明共享模块没有改变命令行为。
