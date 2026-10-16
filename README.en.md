# Agentic Large Language Models 双语阅读版

这是 [Agentic Large Language Models（arXiv:2610.13455）](https://arxiv.org/abs/2610.13455) 的非官方双语阅读整理。作者为 Max van Duijn、Michiel van der Meer、Aske Plaat 和 Niki van Stein。

**中文译文待人工复核。** 英文页面从原文 PDF 提取并重新排版。内容请以[原始论文](https://arxiv.org/abs/2610.13455)为准。arXiv 页面目前链接到 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)；其他署名及说明见[第三方材料声明](THIRD_PARTY_NOTICES.md)。

GitHub 账号、仓库和 Pages 地址确认后，将在此加入网站链接。

## 本地运行

需要 Node.js 22.19 或更高版本及 pnpm 11.1.3。pnpm 版本固定在 `package.json` 中。

```sh
pnpm install --frozen-lockfile
pnpm docs:dev
```

构建并检查静态网站：

```sh
pnpm docs:build
pnpm docs:check-dist
pnpm docs:preview
```

正文请编辑 `docs/` 下的中英文 Markdown。`site/content/` 是自动生成目录，已加入 Git 忽略规则。章节顺序和页码来自 `config/chapters.json`。

## 来源和翻译状态

英文版是原文的非官方提取与网页排版。中文页面为项目翻译，**待人工复核**。内容提取、翻译、排版及双语锚点对应可能存在误差。
