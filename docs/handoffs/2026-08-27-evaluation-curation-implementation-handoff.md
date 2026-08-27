# V0 评测集策展实现交接

更新时间：2026-08-27（Asia/Shanghai）

## 本次完成

- 新增独立命令 `xifeng eval curate` 和本地评测集工作台；
- 新增 draft/frozen、pilot/benchmark、dev/holdout 数据结构；
- 金标准证据改为文章版本与原子单元范围，不再依赖 TopicChunk ID；
- 冻结时保存完整索引语料成员、正文哈希、批准清单和题集内容哈希；
- 默认使用 `gpt-5.6-terra` 的 Codex CLI 生成单篇候选答案、答案要点与证据范围；
- 每次调用前由页面显示完整外发披露，并使用一次性、十分钟有效的确认凭据；
- Codex 继续使用空临时目录、ephemeral、read-only、忽略用户配置和规则、stdin 与 JSON
  Schema；
- AI 候选必须人工接受或拒绝，接受后仍为可编辑 draft，不自动完成人工审核；
- 评测运行器支持冻结数据集 ID、显式 Index Build、快照匹配、稳定证据动态映射和
  evidence coverage@K；
- 旧 JSON 题集运行入口暂时兼容保留；
- 通用 Codex CLI 结构化执行已从分块模块提取，分块路径保持兼容。

## 主要文件

- `src/xifeng_rag/evaluation_curation.py`
- `src/xifeng_rag/evaluation_ui.py`
- `src/xifeng_rag/evaluation_static/`
- `src/xifeng_rag/structured_generation.py`
- `src/xifeng_rag/evaluation.py`
- `src/xifeng_rag/storage.py`
- `src/xifeng_rag/migrations/versions/20260827_0005_evaluation_curation.py`
- `tests/test_evaluation_curation.py`
- `docs/specs/2026-08-27-evaluation-dataset-curation-design.md`

## 验证状态

```text
pytest: 79 passed
Ruff: passed
Pyright: 0 errors
node --check: passed
git diff --check: passed
```

浏览器验收使用 WSL Ubuntu `agent-browser` 和自造临时数据库完成：

- 创建 pilot draft；
- 人工录入单篇问题；
- 新增答案要点并通过两次点击选择连续证据；
- 打开 AI 外发披露并取消，未启动 Codex；
- 标记人工审核完成；
- 冻结数据集并确认全部编辑、AI、删除证据和保存控件立即变为只读；
- 页面无浏览器错误。

验收过程中发现并修复：冻结成功后列表已更新但当前审核面板未立即重载，导致控件暂时仍可
编辑。现在冻结后会重载当前问题，并禁用整块审核控件和证据范围选择。

## 未执行的真实操作

- 没有调用真实 Codex；
- 没有发送任何真实文章；
- 没有迁移或修改 `data/xifeng.sqlite3`；
- 没有创建真实评测集或冻结真实快照；
- 没有运行真实 BM25 pilot。

第一次真实使用应由用户运行：

```powershell
uv run xifeng eval curate
```

建议先创建 6–8 道 `pilot` 单篇 dev 题。每次 AI 辅助都在页面核对披露后单独确认。全部人工
审核完成后冻结 pilot，再运行：

```powershell
uv run xifeng eval run <FROZEN_DATASET_ID> --k 5
```

pilot 结果只用于验证和调整流程，不得原地升级为正式 holdout。

## Git 注意事项

设计文档已作为独立提交 `8b23dc9` 提交。完整 V0 的 `src/`、`tests/`、依赖文件等仍继承此前
会话的未追踪状态，README 与设计文档也包含此前未提交修改，因此本次实现没有宽泛 add 或
创建会夹带历史工作的实现提交。不要 reset、checkout 或 clean。
