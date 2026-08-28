# 确定性上下文组装实施计划

日期：2026-08-28

对应设计：`docs/specs/2026-08-28-deterministic-context-assembly-design.md`

## 目标

实现一个本地、确定、可审计的 ContextAssembler，将排序后的 `RetrievalCandidate` 转换为带
引用、预算统计、排除原因和规范 JSON 的 `AssemblyResult`，并通过现有 `search` CLI 提供本地
预览。首版不接入 Generator，不新增数据库表，不产生网络请求。

## 任务 1：领域模型与错误协议

修改：

- `src/xifeng_rag/domain.py`
- `tests/test_domain.py`

步骤：

1. 增加组装状态、排除原因和压缩方式枚举；
2. 增加配置快照、证据、排除候选、用量和结果模型；
3. 增加带稳定错误码的上下文组装异常；
4. 验证原子范围、引用 ID、首版未压缩不变量与用量非负约束；
5. 使用自造数据覆盖模型序列化与非法结构。

## 任务 2：纯 ContextAssembler

新增：

- `src/xifeng_rag/context_assembly.py`
- `tests/test_context_assembly.py`

步骤：

1. 实现候选结构预校验，错误消息不得包含原文；
2. 实现按输入顺序的 chunk ID 与版本/范围完全去重；
3. 实现完整块选择、数量限制、预算跳过后继续和首块超预算错误；
4. 实现可选 AI 元信息的固定字段 JSON 渲染；
5. 统计原文、AI 元信息和完整渲染字符数；
6. 校验引用连续、范围未压缩、预算和渲染重建不变量；
7. 计算规范 JSON SHA-256；
8. 覆盖空候选、顺序、去重、预算、中文转义、确定性、输入不变和错误隐私测试。

## 任务 3：配置

修改：

- `src/xifeng_rag/config.py`
- `config.example.yaml`
- `tests/test_config.py`

步骤：

1. 增加 `ContextAssemblySettings`；
2. 默认 `max_evidence_items=5`、`max_total_chars=50000`、
   `include_ai_metadata=false`；
3. 使用 Pydantic 约束正整数预算；
4. 验证 YAML 与环境变量覆盖。

## 任务 4：搜索 CLI 集成

修改：

- `src/xifeng_rag/cli.py`
- `tests/test_cli.py`

步骤：

1. 为 `search` 增加 `--assemble`；
2. 未启用时保持当前人类可读与 JSON 候选输出；
3. 启用时使用项目配置组装候选；
4. 人类可读模式展示状态、引用、来源、范围、字符用量与排除原因；
5. `--assemble --json` 输出完整 `AssemblyResult`；
6. 将组装错误转换为不含正文的清晰 CLI 失败；
7. 使用 monkeypatch 的自造候选验证无需模型和网络的 CLI 路径。

## 任务 5：文档同步

修改：

- `README.md`
- `docs/design/architecture.md`
- `docs/design/technology-stack.md`

步骤：

1. 记录上下文组装能力、默认配置和命令；
2. 明确 AI 元信息默认不外发且不可引用；
3. 记录完整块、字符预算、状态分类和未来 Compressor / RetrievalController 插入点；
4. 不重复设计全文，链接对应设计文档。

## 任务 6：验证与本地验收

执行：

```powershell
uv run pytest
uv run ruff check .
uv run pyright
git diff --check
```

在当前冻结 Index Build 上运行只读烟测，且只报告：

- 状态；
- 输入、入选和排除数量；
- 完整渲染字符数；
- AI 元信息开关的字符增量；
- 排除原因；
- 组装耗时。

不得在提交、普通日志或验收汇报中复制付费正文。最后精确暂存本计划涉及的文件并提交实现。
