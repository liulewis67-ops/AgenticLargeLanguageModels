# 原书 arXiv v1 文本对照、公式与引文版本审校（2026-11-14）

> **审校级别：原始 arXiv HTML/LaTeX 渲染文本核对 + 外部论文版本核查。不是原始 PDF 图像逐页审定。** 原书的 PDF 为约 40 MB（arXiv 页面报告 41,760 KB）；当前审校环境未成功获取完整 PDF，因此不宣称已逐页审阅 PDF 的图片、版式、字体和完整页边界。arXiv HTML 在语义和公式方面是比第三方转换文本更直接的对照，但仍需日后依据 PDF 图像最终复核。

## 1. 锁定来源和审校范围

- 原书官方记录（提交 2026-10-09；v1；472 页）：https://arxiv.org/abs/2610.13455
- **本轮实际可访问并对照的原始全文：** https://arxiv.org/html/2610.13455v1
- 当前阅读版仓库：第 14 章原 PDF 元数据为 277–282；第 18 章为 353–358；第 21 章为 403–422。**这些是仓库原有映射**，非本轮 PDF 逐页视觉确认。
- 补充上轮的[第 14/18 章源码专项审计](source-code-audit-2026-11-14-ch14-ch18.md)，重点核对此前容易混淆的**原书忠实度 vs 上游代码现实**。

### 证据等级说明

| 标识 | 已完成工作 | 不能宣称的工作 |
| --- | --- | --- |
| H | 对照 arXiv v1 官方 HTML 的明确文本/公式 | PDF 逐页视觉准确性、页眉/页脚与排版 |
| C | 固定提交上的上游代码静态审查（见前一轮报告） | 原程序完整执行成功、下游结果复现 |
| P | 外部论文的出版页/摘要/指定 PDF 页面核查 | 全文数据重算、独立复现 |
| T | 本仓库无模型算术测试或静态回归测试 | 论文实验统计结果 |

## 2. 明确纠正：第 14 章源文本的文件名

**H-14-01，高优先级，已修复中英正文及前次错误编者说明。**

官方 HTML 第 14.3.1、14.3.2、14.3.3 节分别明确使用：
- `0_train_models.py`
- `1_basic_mcp.py`
- `2_llm_mcp.py`

本仓库原先将它们写成无数字前缀的 `train_models.py`、`basic_mcp.py`、`llm_mcp.py`，造成用户直接运行时找不到文件。上一个 PR #12 的编者注虽然给出了正确文件名，却错误称无前缀名为“原书简称”。**官方 arXiv v1 原文本身就包含数字前缀**，本次直接修复了中英文正文并撤回错误说明。

**H-14-02，中优先级，已修复英文正文。** 第 14.3.3 节原书引用准确函数标识 `make_case_text`。此前英文提取稿将其拆成普通词语 *make case text*，本轮恢复函数标识和 Markdown 代码排版；中文稿已有正确标识。

**证据：** https://arxiv.org/html/2610.13455v1（`14.3 Implementation Steps`）；配套代码：https://github.com/Agentic-LLMs/Agentic-AI-tutorials/tree/main/case3 。

## 3. 第 18 章公式 (18.1)：作者文本与代码实现分别核对

**H-18-01，原式确认：** 官方 arXiv HTML `18.3.4 中，式 (18.1) 的四个分量为新近性 `R`、相关性 `L`、重要性 `I`、效价 `V`；写明各项权重为 1、负面效价倍率 1.5。仓库英语/中文公式编号和数学结构与官方 HTML 相符，本轮**没有改动原书式子**。

**C-18-02，代码解释沿用 #12 的已完成结论：** Go 实现按原始效价符号判负；先对绝对值进行 min–max 缩放，再施加 `NegativityBias`，并最终乘 `ValenceWeight`。公开初始配置 `valence_w=0`，仅设置倍率不生效；课程 Exercise 3 又同时启用非对称编码，因此无法单凭结果识别检索权重的单独效应。**这是源码核对，不是公式在原论文中已充分规定的证明**。

**证据：** https://arxiv.org/html/2610.13455v1（`18.3.4`）；固定实现：https://github.com/fvdveen/generative_agents/blob/6425988301d2ad75244ef1d4b827d26d66a4bba8/simulation_server/agent/retrieve.go 。

## 4. 第 21 章原文差异与因果推断边界

**H-21-01，高优先级，已修复中英文正文。** 官方 arXiv v1 `21.7 的受扰／反事实输入示例是 *The Colosseum is in Rome*（罗马斗兽场位于罗马）；先前阅读版误写为 *The Eiffel Tower is in Rome*（埃菲尔铁塔位于罗马），属于非原文的替换。现已恢复原例，并加入标明“非原文”的双语编者说明。

**H-21-02，公式核验：** 官方 HTML 中式 (21.7) 是干净、受扰和修补三种运行的目标输出概率所构成的归一化恢复比率。本仓库数学表达式保持同构。上一轮 `pnpm docs:audit-formula` 的近零分母数值自检仍只属于 `T` 级算术校验，**不是激活修补实验复现**。

**方法学提示（编者，非原文）：** 作者提供的 *Eiffel Tower in Paris* 与 *Colosseum in Rome* 同时改变了实体和地点，不是严格单变量控制的输入对。不能据一个修补比率接近 1 就推出唯一因果机制；应记录目标 logit、替换位置、匹配的反事实分布和负对照。在**忠实复原作者示例**和**指出实验设计局限**之间，需要分别标明证据归属。

**证据：** https://arxiv.org/html/2610.13455v1（`21.7，式 21.7）。

## 5. 第 21 章引用的 26/28 注意力头差异：是版本差异

**P-21-03，已核验两个版本、保留原书。**

- 本书对应的引用是 Wang et al. (2022)，*Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*，arXiv:2211.00593。**arXiv 页面摘要写 26 个注意力头、七类**：https://arxiv.org/abs/2211.00593 。
- **ICLR 2023 出版 PDF 首页摘要写 28 个注意力头、七类**，并在引言中再次写 28；首页已与 PDF 页面图像核对：https://openreview.net/pdf/1e69126a0944a99f44e32245d87e213a30cc2eb2.pdf 。
- 第 21 章正文的 **26** 与书目所指 2022 arXiv 预印本一致；本轮**不擅自将原书替换为 28**，而是在中英文新增版本差异提示。

这也说明外部引文真实性不止需要检查“题名和作者存在”，还需要明确**版本与相应数值**，以免误将出版修订导致的差异判作模型复现失败。

### 本轮额外核实的论文元数据

| 书中所引 | 出版或原始科研来源 | 已核范围 |
| --- | --- | --- |
| Belrose et al. 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens* | https://arxiv.org/abs/2303.08112 | 作者、题名、年份及 affine probes 的论文摘要 |
| Syed, Rager & Conmy 2024, *Attribution Patching Outperforms Automated Circuit Discovery* | https://aclanthology.org/2024.blackboxnlp-1.25/ | ACL Anthology 出版记录、DOI **10.18653/v1/2024.blackboxnlp-1.25**、线性近似方法的论文摘要 |
| Wang et al. 2022/2023, IOI circuits | https://arxiv.org/abs/2211.00593 与上方 ICLR PDF | **版本相关的 26 / 28**，且 ICLR PDF 首页进行了视觉核对 |

上表**不等于完整核实第 21 章所有引文**，也不代表上述论文的实验已经重算。

## 6. CI 回归守卫

新增 `scripts/audit-source-fidelity.mjs` 与 `pnpm docs:audit-source`，仅检查已经人工核对过的**固定文本事实**：第 14 章三个真实文件名与 `make_case_text`、第 18/21 章公式编号、第 21 章反事实实例和版本数字备注在双语中保持正确。

它与 `pnpm docs:audit-structure`、`pnpm docs:audit-formula` 互补：结构脚本验证排版锚点，公式脚本验证边界算术，来源守卫防止本轮明确错误回归。**任何一个通过都不能证明已读取原始 PDF 页面或运行了原论文实验。**

## 7. 本轮完成与待办边界

| 项目 | 当前状态 |
| --- | --- |
| 官方 arXiv v1 HTML 核对第 14 章脚本名称和函数、18.1 公式、21.7 公式与原例 | **完成本报告列出的具体条目** |
| 修复中英文第 14 章可执行文件名、英文 `make_case_text`，以及第 21 章反事实示例 | **完成** |
| 第 21 章 Wang et al. 预印本 26 与 ICLR PDF 28 的版本对照 | **完成指定摘要核对** |
| 原书 472 页 PDF 可获取性、渲染与逐页图文对照 | **未完成** |
| 第 14 章三模型的独立数据集评测、真实 LLM 路由运行 | **未完成** |
| 第 18 章 NEVER 多种子模拟与 2×2 消融 | **未完成** |
| 第 21 章激活修补的真实模型实验和对照重测 | **未完成** |

优先后续动作：取得可渲染的原始书籍 PDF（最好固定 SHA256、v1），按物理页与 `config/chapters.json` 和各自 `page-*` 锚点比对；逐页记录公式、图注和引文的不同；然后独立运行上游代码、生成环境锁、模型快照、种子和评测原始数据。**在此前不应宣称 PDF 校对或研究复现已完成。**
