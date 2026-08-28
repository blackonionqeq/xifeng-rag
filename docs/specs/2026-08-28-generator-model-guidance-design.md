# Generator 模型指引设计

状态：待实现。

## 目标

让用户运行 `xifeng generator models` 后立即知道是否需要配置、去哪里配置以及下一条命令，
无需阅读长篇 API 说明。

## 改动

- 新增简短指南 `docs/guides/deepseek-generator-setup.md`；
- README 在 Generator 使用处链接该指南；
- `generator models` 在原始模型列表后显示当前配置模型、可用性、默认推荐和下一条命令；
- 当前模型不可用时显示 `config.local.yaml` 的最小 YAML 示例；可用时不要求重复配置；
- 不自动修改本地配置，不详细比较价格、上下文窗口或实验模型能力。

## 模型建议

- 默认使用 `deepseek-v4-flash`；
- `deepseek-v4-pro` 只作为后续质量对照；
- `deepseek-v4-flash-vision-exp` 不用于当前纯文本 RAG。

## 验证

测试当前模型可用与不可用两种输出，并保证命令不发送问题或语料、不修改配置文件。
