# DeepSeek Generator 快速使用

## 默认用法

项目默认使用 `deepseek-v4-flash`。如果下面的命令显示它“可用”，无需填写模型配置：

```powershell
uv run xifeng generator models
uv run xifeng ask "你的问题" --mode hybrid --rerank
```

模型用途：

- `deepseek-v4-flash`：当前纯文本 RAG 的默认选择；
- `deepseek-v4-pro`：用于后续回答质量对照；
- `deepseek-v4-flash-vision-exp`：当前项目不处理图片，不使用。

## 切换为 Pro

在项目根目录 Git 忽略的 `config.local.yaml` 中加入：

```yaml
generator:
  model: deepseek-v4-pro
```

API 密钥仍填写在项目根目录的 `.env`：

```dotenv
DEEPSEEK_API_KEY=your-key-here
```
