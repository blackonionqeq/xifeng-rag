# 西风文章 RAG 研究台

面向个人购买的长篇文章，构建一个本地优先、证据驱动的主题研究工具，同时作为可复现的 RAG 学习与求职展示项目。

当前处于设计阶段，尚未开始编码。

## 计划能力

- 基于原文证据的单篇问答与跨文章主题研究；
- 本地 Embedding、混合检索和默认本地重排；
- 第一阶段使用 DeepSeek V4 Flash 生成，后续可增加本地模型；
- 新增和修改文章的增量更新；
- 可复现的 RAG 优化对照实验；
- 后续扩展为图增强 RAG。

## 设计文档

- [总体架构](docs/design/architecture.md)
- [切块、上下文化与增量索引](docs/design/chunking-and-indexing.md)
- [版本路线与实验方向](docs/design/roadmap.md)
- [技术栈](docs/design/technology-stack.md)
- [Agent 工作约束](AGENTS.md)

## 语料来源

付费文章不存放在本项目内，也不提交到本项目的版本库。当前开发环境从以下外部目录只读加载文章：

```text
D:\codes\projects\test-reverse-wechat-official-accounts-paid-article\saved_articles
```

从项目根目录计算，对应的默认相对路径为：

```text
../test-reverse-wechat-official-accounts-paid-article/saved_articles
```

实现时通过本地配置项 `corpus.source_dir` 覆盖该位置。默认只索引目录顶层的 Markdown；完整的读取、隐私和排除规则见 [AGENTS.md](AGENTS.md) 与[切块和索引设计](docs/design/chunking-and-indexing.md)。
