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

## 学术审校与质量审计

新增[第 21–23 章机制可解释性与认知证据审校](reports/academic-audit-2026-11-17-ch21-ch23-interpretability-cognition.md)及[激活修补／心智理论／高风险智能体验证协议](reports/mechanistic-cognition-ch21-ch23-validation-protocol.md)：恢复第 21–22 章 **46/111** 条书目并保留第 23 章独立书目 **34** 条；特别审校 LayerNorm DLA 不可加性、Softmax 概率与 logits 混称，以及行为／表征／机制／内省与部署证据边界。

新增[第 19–20 章书目与 SICA 学术审校](reports/academic-audit-2026-11-17-ch19-ch20-synthetic-sica.md)及[合成数据与 SICA 四臂复现实验矩阵](reports/synthetic-data-sica-reproducibility-matrix.md)：恢复 87/42 条中英文书目、修复被图 19.4 截断的 PersonaHub 引文，明确真实／合成数据递归训练与 SICA 评测中的污染、过拟合和预算匹配限制。

新增[第 17–18 章学术审校与因果解释报告](reports/academic-audit-2026-11-16-ch17-ch18-social-simulation.md)及[四组社会模拟复现实验矩阵](reports/social-simulation-ch17-ch18-reproduction-matrix.md)：恢复 47/3 条参考文献、修复跨行作者名单；重点纠正人物匹配混杂、检索权重被禁用时的零效应，以及检索×非对称编码的 2×2 消融设置。

新增[第 14–16 章学术审校与方法学核验](reports/academic-audit-2026-11-16-ch14-ch16-methodology.md)：对照原书核对 1/90/29 条文献，修复跨页拆分的 DOI、ISBN、标题和链接，明确共享状态与 MCP 协议的差别，并完善行为研究的观测单位和协调实验的等预算对照要求。

新增[第 11–13 章引文、BM25 与临床证据审校](reports/academic-audit-2026-11-15-ch11-ch13-tools-retrieval-clinical.md)：核对并统一 24/34/40 条双语参考文献，标注 LangChain/MCP 的接口与协议边界、BM25 公式勘误、检索指标零分母处理和诊断随机试验的证据范围。

新增[第 8–10 章学术审校与 BALROG 复现检查](reports/academic-audit-2026-11-15-ch08-ch10-balrog.md)，并提供[BALROG 评测运行记录模板](reports/balrog-reproducibility-run-template.md)：统一 11/43/3 条书目，修复作者名单跨行和评测参数错误，明确论文原配置与低成本冒烟测试的区别。

新增[第 5–7 章参考文献、nanochat 可复现性与证据边界审计](reports/academic-audit-2026-11-15-ch05-ch07-integrity.md)；配套提供[nanochat 运行记录模板](reports/nanochat-reproducibility-run-template.md)，用于固定实验来源、环境、数据和评测协议。

新增[第 4 章规模律公式和参考文献核查](reports/academic-audit-2026-11-15-ch04-scaling-bibliography.md)：按 arXiv v1 将参考文献恢复为 72 条独立条目，纠正跨页 DOI / OpenReview 链接，并对 Chinchilla 计算最优指数增加算术回归检查。

继续核对[第 2–3 章原始书目条目和评价指标专项报告](reports/academic-audit-2026-11-15-ch02-ch03-bibliography.md)：恢复第 2 章中文版遗漏的前 6 条参考文献，按 arXiv v1 重建第 2 章 59 条与第 3 章 52 条双语书目，并对 Pass@K 原文歧义加编者校勘。

新增[第 1 章引文分条与 arXiv v1 文献对照审校](reports/academic-audit-2026-11-15-ch01-references.md)，修复首章 24 条被合并的引文，并对该问题建立持续回归检查。

双语结构审计现已覆盖**第 1–23 章**，包含四批可追踪的学术校勘、实验设计和证据等级。阅读[第 2–7 章审计](reports/academic-audit-2026-11-14.md)、[第 8–13 章审计](reports/academic-audit-2026-11-14-ch08-ch13.md)、[第 14–18 章审计](reports/academic-audit-2026-11-14-ch14-ch18.md)、[第 19–23 章审计](reports/academic-audit-2026-11-14-ch19-ch23.md)，以及[多智能体实验审计协议](reports/multi-agent-experiment-audit-protocol.md)。新增[第 14/18 章上游源码、引文真实性与可复现性专项审计](reports/source-code-audit-2026-11-14-ch14-ch18.md)及[第 14/18/21 章官方 arXiv v1 原文对照、公式与论文版本核验](reports/arxiv-source-concordance-2026-11-14-ch14-ch18-ch21.md)、[第 19–20 章原始论文数据规模、SICA 试验与复现性核验](reports/primary-evidence-audit-2026-11-15-ch19-ch20.md)及[第 15–16 章社会智能/协作实验来源与数值单位核验](reports/primary-evidence-audit-2026-11-15-ch15-ch16.md)、[第 15–17 章 ToMBench 全量数据质量、协调论文勘误与 OASIS 资源审计](reports/primary-source-data-audit-2026-11-15-ch15-ch17.md)及[第 17 章 CASA 六文件数据清单与命名游戏原始实验审计](reports/casa-and-norms-evidence-audit-2026-11-15-ch17.md)，其中明确区分“静态源码核验”“数值边界自检”“原 PDF 视觉核验”“真实实验复现”的完成程度。

```sh
pnpm docs:audit-structure      # CI 检查第 1–23 章
pnpm docs:audit-structure-all  # 手动扫描全部编号章节
pnpm docs:audit-formula       # 第 18/21 章编者数学边界自检；非原模型复现
pnpm docs:audit-source        # 官方 arXiv 原文核实过的固定事实回归检查
pnpm docs:audit-tombench --self-test  # 离线 JSONL 数据预检的人工夹具单元测试
pnpm docs:audit-tombench -- ../ToMBench/data  # 克隆上游数据后实际扫描全部 2860 条，不执行模型
pnpm docs:audit-casa --self-test       # CASA 人工夹具回归测试（不下载外部数据）
pnpm docs:audit-casa -- ../CASA --expect-audited-release  # 克隆作者固定版本后核对六份公开 JSON
```

**注意：**结构检查通过不表示已与原始 PDF 完成逐页对照。对于尚未核查的学术结论、数学公式和实验数据，请继续以原始论文为准。

## 译文状态、来源与反馈

英文版为从 PDF 提取、校正并转换格式的阅读文本，中文版为项目译文。**两种语言的内容均可能存在提取、跨页断句、数学公式、引用链接或排版错误，尤其是尚未逐段人工核对的部分。**

欢迎通过 [Issues](https://github.com/liulewis67-ops/AgenticLargeLanguageModels/issues) 报告问题，并附上**章节、语言、PDF 页码、原文及建议修改**。这有助于进行可追溯的逐章审校。

arXiv 页面目前指向 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)；原作者、原始资料与其他说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
