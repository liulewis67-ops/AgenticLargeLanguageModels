# Agentic Large Language Models｜智能体大语言模型

[简体中文](README.md) · [English](README.en.md)

这是论文 [*Agentic Large Language Models*（arXiv:2610.13455）](https://arxiv.org/abs/2610.13455) 的**非官方双语阅读版**，原作者为 Max van Duijn、Michiel van der Meer、Aske Plaat 和 Niki van Stein。

本项目将原文整理为按章节导航的 Markdown，并提供简体中文译文，便于在线阅读和对照学习。**中文翻译仍在持续人工审校，不能替代原文**；公式、图表、术语和引用如有疑问，请以[原始论文](https://arxiv.org/abs/2610.13455)为准。

- **在线阅读**：<https://liulewis67-ops.github.io/AgenticLargeLanguageModels/>
- **原始论文**：<https://arxiv.org/abs/2610.13455>
- **中文目录**：[docs/index-zh.md](docs/index-zh.md)
- **English index**：[docs/index-en.md](docs/index-en.md)
- **翻译与审校规范**：[TRANSLATION_GUIDE.md](TRANSLATION_GUIDE.md)

## 本地运行

需要 Node.js **22.19 或更新版本**、pnpm **11.1.3**（以 `package.json` 中的版本为准）。

```sh
pnpm install --frozen-lockfile
pnpm docs:dev
```

构建网站并检查生成结果：

```sh
pnpm docs:build
pnpm docs:check-dist
pnpm docs:preview
```

请编辑 `docs/` 中的中英文原始 Markdown；`site/content/` 由脚本自动生成，已被 Git 忽略。章节顺序与原始 PDF 页码映射记录在 `config/chapters.json` 中。

## 译文状态、来源与反馈

英文版为从 PDF 提取、校正并转换格式的阅读文本，中文版为项目译文。**两种语言的内容均可能存在提取、跨页断句、数学公式、引用链接或排版错误，尤其是尚未逐段人工核对的部分。**

欢迎通过 [Issues](https://github.com/liulewis67-ops/AgenticLargeLanguageModels/issues) 报告问题，并附上**章节、语言、PDF 页码、原文及建议修改**。这有助于进行可追溯的逐章审校。

arXiv 页面目前指向 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)；原作者、原始资料与其他说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
