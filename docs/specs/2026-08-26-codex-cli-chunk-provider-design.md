# Codex CLI 单篇分块 Provider 设计

日期：2026-08-26

## 目标

允许个人用户复用已登录的 ChatGPT Plus Codex CLI，为明确选择的单篇文章生成语义分块提案，
避免另行购买 API 额度。该能力只替换提案生成器，不改变清洗、审核、批准和索引流程。

## 隐私定位

`codex-cli` 虽然由本机子进程启动，正文仍会发送给 OpenAI，因此属于外部推理。使用 ChatGPT
登录时，数据处理遵循该 ChatGPT 工作区的设置；V0 由用户负责关闭“用于改进模型”。程序无法
验证该账户开关，不将“已关闭训练”作为可自动证明的属性。

每次调用继续沿用单篇二次确认，显示文章标题、版本、清洗后字符数、原子单元数、provider、
模型和最多完整发送次数。未确认时不得启动 Codex、创建提案或读取任何额外凭据。

## 调用方式

新增 `codex-cli` 特殊 provider：

```text
uv run xifeng chunks retry <ARTICLE_ID> --provider codex-cli --model <MODEL_ID>
```

它实现现有 `StructuredGenerator` 协议，通过 `codex exec` 调用：

```text
codex exec
  --ephemeral
  --sandbox read-only
  --ignore-user-config
  --ignore-rules
  --skip-git-repo-check
  --output-schema <TEMP_SCHEMA>
  --output-last-message <TEMP_OUTPUT>
  --model <MODEL_ID>
  --cd <EMPTY_TEMP_DIR>
  -
```

完整 Prompt 仅通过 stdin 传入，不进入命令行参数。子进程在新建的空临时目录运行，Prompt
明确禁止工具调用，只要求返回 Schema 对象。`--ignore-user-config` 避免加载个人 MCP、规则和
其他行为配置，但仍复用 Codex CLI 保存的认证。

## 配置

新增简单配置：

```yaml
chunking:
  codex_cli:
    executable: codex
    default_model: gpt-5.6-terra
    timeout_seconds: 900
    max_retries: 1
```

默认只允许一次首次请求和一次结构纠错。`--model` 只覆盖当前重试。提案记录
`provider_id=codex-cli`、显式模型 ID、Prompt/Schema 版本以及非敏感生成元数据。

## 输出与故障

- 使用 Codex 的 `--output-schema` 约束最终 JSON，之后仍执行现有 Pydantic 与连续覆盖校验。
- 不把 stdout/stderr、完整 Prompt、正文或 Codex 会话写入普通日志。
- 只保存退出码、CLI 版本、认证类型等不含正文的诊断元数据。
- CLI 不存在、未登录、超时或输出无效时，提案按现有逻辑进入 `failed`，不得批准或索引。
- 临时目录退出调用后自动删除；`--ephemeral` 禁止保存 Codex rollout 会话。

## 测试与验收

单元测试覆盖命令参数、stdin 正文传递、隔离临时目录、Schema 输出、无正文日志、取消确认零
子进程以及失败诊断。真实验收只运行用户明确确认的短文，导出边界和标题供人工比较，不批准
提案、不构建索引。

