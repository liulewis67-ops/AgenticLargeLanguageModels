# 第15–16章：社会智能与多智能体协调的原始文献审校（2026-11-15）

> **分清证据等级。** 本轮对照原书的中英 Markdown 和作者论文：核对 ACL Anthology 的正式出版记录、LLM-Coordination 正式 PDF 首页图像、Hou 等人 2024 年 arXiv 摘要及后续 OpenReview 稿的文本、两个原始源码仓库的固定提交。**未取得或渲染 472 页原书 PDF，未执行任何模型调用、复算论文结果或跑多个独立实验种子。** 数值由原作者报告，非本项目复现实验所产。

## 1. 可直接验证的文本错误

| ID | 书稿位置 | 原有读本 | 原始来源 | 操作及证据范围 |
| --- | --- | --- | --- | --- |
| B16-01 | 第16章英文/中文参考文献 Agashe et al. 2025 | 页码 **8038–8057** | [ACL Anthology 正式出版记录](https://aclanthology.org/2025.findings-naacl.448/)及[论文 PDF 首页](https://aclanthology.org/2025.findings-naacl.448.pdf)，明确显示 **8053–8072** | **直接修复双语文末参考文献**，不改变原书引用作者、标题或讨论 |
| N15-01 | 第15章英文/中文 §15.4.2 EgoSocialArena | “11% behind humans / 11%” 容易被理解成相对百分比差 | [Hou et al. 2024，arXiv:2410.06195](https://arxiv.org/abs/2410.06195)摘要明确给出 **11.0 points** | **直接精确化为 11.0 percentage points / 11.0 个百分点**，并注明原作者文稿数值；不解释为人类行为能力的百分比降低 |
| V15-02 | 第15章同处 | 忽略引用的不同版本 | [后续 OpenReview 修订稿](https://openreview.net/pdf?id=ZCH7MwbXje)文本写 o1-preview 80.6、人类基准差 **7.7 points** | 在**非原文编者注**标记11.0/7.7分差为版本特定数字；**不合并或平均** |
| P16-02 | 第16章 §16.2.1 | 在同段讨论协调游戏表现和问答表现 | [Agashe et al. NAACL 2025](https://aclanthology.org/2025.findings-naacl.448/)摘要与论文正文 | 明确两项评测：**四个多轮纯协调游戏**与**198道单轮 CoordQA 选择题**，不能将问答正确率当作多轮成功率 |

**注意：**本报告中的“页码”均为**被引论文的正式出版页码**；不是原书第15/16章 PDF 的物理页位置。本轮 PDF 首页视觉核验仅针对 **Agashe et al. 2025 原始论文**，未增加原书 PDF 逐页审查完成数。

## 2. 第15章 EgoSocialArena：数字、版本、研究效度

- **引用的 2024 原论文摘要：**研究将社会智能分为认知、情境和行为三个维度，对八个模型开展实验，o1-preview 与人类总体成绩相差 **11.0 分**（不可理解为相对百分比）。
- **后续手稿版本：**可访问的 OpenReview 投稿 PDF 文本指出相差 **7.7 分**，以及 o1-preview 得分80.6。不能因为同名 EgoSocialArena 就静默替换2024书目所指的11.0；这些值分别对应不同的论文版本与可能不同的聚合配置。
- **固定代码快照：**[gyhou123/EgoSocialArena@6226bcb355e1be36f7636860080c7d695660f543](https://github.com/gyhou123/EgoSocialArena/tree/6226bcb355e1be36f7636860080c7d695660f543)，README公开对OpenAI、Together、Anthropic等API调用的环境依赖与示例数据、情景统计。README给出**2,195 data entries**；后续 OpenReview 稿又写**2,245 entries**。这进一步表明复现须冻结论文-数据集组合；该差异未经逐实例核查，**不能认定是哪50项造成**。
- **待检控制与统计：**固定题目 ID、人类评分者数量与汇总规则、LLM 供应商版本、调用成本、采样种子与失败重试规则；每个模型按游戏/对话/题目层次报告分层不确定性，避免把同一模型的若干输出当作独立受试者。作者结果不等于本项目自主复现。

## 3. 第16章 LLM-Coordination：出版记录与实验划分

**正式文献信息：**Saaket Agashe, Yue Fan, Anthony Reyna, Xin Eric Wang（2025），*LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models*，*Findings of ACL: NAACL 2025*，**8053–8072**，DOI [10.18653/v1/2025.findings-naacl.448](https://doi.org/10.18653/v1/2025.findings-naacl.448)。已对照正式 PDF 首页出版页码。

**实验组成（源自论文及 README）：**
- Agentic Coordination：**Overcooked、Hanabi、Collab Capture、Collab Escape**四个游戏；包含自对战和跨合作伙伴、不同代理架构的测试。
- CoordQA：来自这些游戏的**198道选择题**，区分**环境理解、心智理论（ToM）、联合规划**三个方面。单轮题目正确率和多轮游戏总奖励/胜率的测量单位不同。
- 论文报告的 GPT-4-turbo 条件含显式认知架构、提示策略等设置；上层架构与模型权重的贡献不能不经消融就混为一谈。基于本论文所选 RL 基线观察到的零样本稳健性，也不支持“所有RL方法均不具备跨伙伴能力”的总体断言。
- **上游代码快照：**[UCSB-AI/llm_coordination@586733a37d240b9592645bb0d7d10f73c1103f7f](https://github.com/UCSB-AI/llm_coordination/tree/586733a37d240b9592645bb0d7d10f73c1103f7f)。`src/agentic_evals/` 中包括 `hanabi_agent_benchmarking.py`、`overcooked_agent_benchmarking.py`、`collab_capture_benchmarking.py` 和 `collab_escape_benchmarking.py`；README写 Python 3.9、`pip install -r requirements.txt`，项目依赖外部 Overcooked/Hanabi 环境与 LLM API。
- **未执行：**未安装四套环境，未调用 GPT-4-turbo 或 RL 基线，未取得198道题的模型级原始输出，未复算统计不确定性。**静态定位四个评测入口不等于复现实验。**

## 4. 最小复验协议

1. **源码与环境：**固定两个源码提交、Python与完整依赖锁、全部游戏依赖及环境提交哈希；记录 API 模型快照、温度和系统提示。
2. **抽样单位：**EgoSocialArena按题/场景/人类标注来源划分，CoordQA按题目划分，游戏按场景和对手策略划分；禁止把多轮游戏中同一剧本的多个动作充当独立样本。
3. **对照消融：**至少比较零样本基础代理、带ToM、自验证及组合，在**同模型、同场景、同推理预算**下分析；跨伙伴测试冻结未见伙伴集合。
4. **指标与置信度：**游戏胜率/奖励与问答准确率分开汇总；报告分组bootstrap/多种子置信区间、任务失败、API开销与模型版本。对于11.0/7.7差异必须先确认两版数据、权重、参考人类评测的一致性。
5. **证据存档：**每轮保存每题原始响应、模型ID、评分器版本、对局轨迹、随机种子、资源账单、异常和重试数，再报告统计显著性和外部效度。

## 5. 本轮已实施的变更

- 第15章中英版纠正 **11% → 11.0 percentage points / 11.0 个百分点**，加入2024原论文与后续稿不同数值的解释。
- 第16章中英版直接纠正参考文献 **8038–8057 → 8053–8072**，加入多轮四游戏与单轮198题的评测单位区分，以及复现前置条件。
- `pnpm docs:audit-source` 扩展这两项具体文字事实的静态回归检查。
- README中英版加入本审校报告。
- **仍未完成：**原书原始 PDF 物理页逐页核查；第15–16章的真实模型实验与统计复算。

仅核对本文明确列出的论文和指标，不表示本章所有书目引用均已查实。
