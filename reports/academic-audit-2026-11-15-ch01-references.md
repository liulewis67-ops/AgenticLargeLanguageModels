# 第 1 章与文末引文的补充学术审校（2026-11-15）

> 对象：`liulewis67-ops/AgenticLargeLanguageModels` 英文与中文第 1 章；原书：*Agentic Large Language Models*, arXiv:2610.13455v1（2026-10-09）。**证据等级：仓库 Markdown 逐行审查 + 官方 arXiv v1 HTML 正文与 References 对照。不是 472 页 PDF 的逐页视觉核验，也不是第三方文献实验复现。**

## 已确认的提取与呈现问题

1. **参考文献边界丢失。** 在 `docs/01-introduction/01-introduction-{en,zh}.md` 原先第 180、184、188 行，多条引文被 Marker 文本提取拼成超长行，不适合阅读或逐条追溯。官方 [arXiv v1 HTML](https://arxiv.org/html/2610.13455v1) 的 Introduction → References（约第 1183–1207 行）列出 **24 条**。这次恢复为独立 Markdown 项，分布在原有分段页面 **7 / 13 / 4 条**；保留全部原有 `page-7-0`、`page-8-0`、`page-9-0` 锚点、页标记、条目顺序和引用页链接。
2. **转换残留。** 更正 `Franc¸ois-Lavet` → `François-Lavet`、`Thomas Back` → `Thomas Bäck`、`LLaMEA` 标题的断行 `algo- ¨ rithm`，删除 `survey. ¨` 的游离重音符；将 Jurafsky 与 Martin 条目的破碎 Markdown URL 恢复为一个原始链接。
3. **文献与正文分界。** 在 Xi Chen 等人的书目条目末尾，原提取文本把引用页号 `22` 混成独立纯文本；现整理为通往第 1 章致谢页面的 `[22](#page-7-0)`。此为可追溯的 Markdown 引文格式整理，不等于新增引用。
4. **数学表述结构。** §1.2.2 的 `Agentic LLM = LLM + Harness` 是文内概念等式，不能作为与小节并列的 H4 标题。两种语言改为正文强调，不改变术语或公式。

## 自动验证

- `scripts/audit-bilingual-structure.mjs` 默认检查范围从第 2–23 章扩至 **第 1–23 章**；未改动目录前言及分部序言的既有范围。
- `scripts/audit-source-fidelity.mjs` 增加第 1 章专项回归：双语各有 24 条独立引文、分页分布 `7/13/4`、LLaMEA 名称、合法书目 URL、无特定 PDF 错误编码、概念等式不误作标题。
- 已在构造变更时验证第 1 章中英文 24 条参考文献的数量与分页分布；待 CI 实际执行 Node/pnpm 构建测试，不能将静态检查等同于构建成功。

## 保留的疑点与审校限制

- **原书自身的书目元数据问题不能被无声改写：** 官方 arXiv v1 HTML 的 Chollet 2021 条目列出 `Chollet and Chollet` 两位同姓作者，仓库照此保留。应对照出版社正式书目信息明确是否重复录入，必要时用“编者勘误”说明，不应伪装成原文。
- 参考文献中仍有原书大小写风格不统一、出版数据需要逐条查证等问题；本轮只纠正可确认的提取残留。
- 本轮未获取原始 472 页 PDF 的逐页渲染图像，页号边界仍依据仓库已有分割标记；官方 HTML 对照与原始 PDF 视觉审阅必须分别记录。
- 没有为第 1 章涉及的引用研究重新运行实验，也不作实验再现性或统计结论已证实的声明。

参考：[官方 arXiv 主页](https://arxiv.org/abs/2610.13455)、[原书 HTML v1 的引言与书目](https://arxiv.org/html/2610.13455v1)、[本仓库审校规范](../TRANSLATION_GUIDE.md)。
