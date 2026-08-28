# Generator 状态归一化设计

状态：待实现。

## 问题

DeepSeek JSON Output 只保证合法 JSON，不执行项目的跨字段约束。模型可能同时返回
`status=insufficient_evidence` 和非空 `grounded_sections`，导致一次生成和一次修复均失败。

## 方案

解析 JSON 后、最终 Pydantic 校验前，只处理这一种矛盾：

- 状态是 `insufficient_evidence`；
- `grounded_sections` 非空且结构合法；
- `limitations` 非空；
- 所有引用 ID 均来自本次证据上下文。

满足全部条件时，将状态归一化为 `partially_answered`。结果元数据记录
`status_normalized=true` 和原状态，不修改结论、引用或限制文本，也不增加外部请求。

缺少限制、引用非法、结论结构非法或其他状态矛盾时不归一化，继续执行现有的一次修复和
最终失败规则。不能把 `insufficient_evidence` 无条件改成部分回答。

## 验证

测试合法归一化、缺少限制不归一化、非法引用不归一化、元数据记录以及现有状态回归。
