# 第 8–10 章学术审校：元学习、智能体记忆、BALROG 代码与参考文献（2026-11-15）

## 审校依据与证据边界

- 原书：[arXiv:2610.13455v1 官方 HTML](https://arxiv.org/html/2610.13455v1)，第 8–10 章及各章 References。
- BALROG 论文：[ICLR 2025 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html)，作者 Paglieri 等；[开源项目](https://github.com/balrog-ai/BALROG)的 [evaluation.md](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)、[agents.md](https://github.com/balrog-ai/BALROG/blob/main/docs/agents.md)与 [Hydra config.yaml](https://github.com/balrog-ai/BALROG/blob/main/balrog/config/config.yaml)。
- 本轮逐项比较的是**原书 HTML、仓库中英文 Markdown 与 BALROG 上游文档/配置**；没有运行原论文模型，也没有完成原始 472 页 PDF 的逐页图像校验。验证结果属于**书目结构与可运行性缺陷审计**，不是完整实验复现。

## 一、参考文献回归修复

| 章节 | 原书 HTML 条目数 | 原始缺陷与修复 |
| --- | ---: | --- |
| 第 8 章，Metalearning | **11**（Brazdil 至 Wang） | 既有条目已基本逐条分开，但 `Jurgen Schmidhuber. ¨` 存在明显的 PDF 重音符提取污染；按正式姓名恢复 `Jürgen Schmidhuber`。 |
| 第 9 章，Interaction Styles | **43**（Brooks 至 Zhou） | 多条不同来源粘连为几千字符长段落；`Yujie Li ... and Tianrui` / `- Li. Improving...`、`Trung Quoc Luong ... and Hang` / `- Li. Reft...` 被误断成额外文献。恢复原作者 `Tianrui Li`、`Hang Li`，并清理已确认的 `LLaMEA` 标题与重音符污染。 |
| 第 10 章，BALROG case study | **3**（Paglieri、Shinn、Yao） | 英文版将三条压成单行，作者姓名/URL 严重失真；中文版此前已正确拆分且修复部分姓名。以中文稿中保留的外文书目作底本，统一双语 3 条记录，修复 `Łukasz Kuciński` 并保留 OpenReview 原论文链接。 |

所有章节保留物理分页标记及 `page-*` 引文锚点，中英文书目文字完全一致；这不自动证明每条书目出版元数据都经出版社逐项核实。

## 二、元学习及记忆架构的学术边界（第 8–9 章）

- **第 8 章：**原书把传统预训练、上下文学习及自我反思置于静态/动态元学习框架，便于直观比较，但不应把“存在两个逻辑环”当作严格的 MAML 式双层优化证明。此前稿件已有非原文编者说明：不同任务上的外层优化目标、支持/查询划分及元梯度不可省略；本轮保留这一区别，并守护书目作者身份。
- **第 9 章：**本书讨论“通过微调实现记忆”和 MCP 相关记忆管理。稿件已有明确的编者注区分**参数知识更新**与可寻址、可撤回的事件记忆，以及 MCP 的协议能力与宿主应用的具体存储策略。已有注释保留不回退。本轮将损坏的文献作者边界修复，避免来源不可辨识。

## 三、BALROG 代码审校（第 10 章）

**B-10-01：Hydra 配置键错误（可运行性）**

第 10 章自定义 Agent 的原始代码清单使用 `agent.max_history=16`，而 BALROG 官方 `balrog/config/config.yaml` 及 `docs/evaluation.md` 明确使用 `agent.max_text_history`。前者不是配置中声明的参数，可能导致 Hydra 拒绝命令。修复中英文**教学示例**为 `agent.max_text_history=16`，并在紧邻清单的编者注说明这是**与原文不同的显式修正**。该修正并未宣称适用于所有历史提交，应在固定 SHA 后确认。

**B-10-02：并行数量与教材叙述不一致（成本与资源）**

标准示例 `eval.num_workers=16`；原始自定义 Agent 示例写为 `eval.num_workers=64`，却称“fewer workers”。新教学示例选用 `eval.num_workers=4` 做低成本、低并发冒烟测试，注释明确不是原始论文或原书默认评测预算。真正比较结果时，需把 worker 并发、同样的游戏 episode、token/行动预算、模型快照等因素固定。

**B-10-03：结果统计和来源证据（此前校勘继续有效）**

原书建议“连续五次结果相同即可逐渐信任结果”，但不构成统计推断准则。中英文此前已有编者提示要求记录重复次数、随机种子和区间。本轮新增 [BALROG 复现实验运行记录模板](balrog-reproducibility-run-template.md)，约束模型和环境版本、数据与游戏回合、配对预算、行动轨迹、评分定义、成本、区间与失败分析。模板所有结果字段标明“待实测”，无虚构的成功率或复现结论。

## 四、自动回归与未完成项

`scripts/audit-source-fidelity.mjs` 新增 8–10 章双语引用数量 **11 / 43 / 3**、首末条来源、页面拼接长行、错误拆开的 `Li` 姓名、BALROG 作者及 OpenReview 链接和正确的自定义命令检查。任一中英书目不一致将报错。

仍待：
1. 使用原始 PDF 页 181–222 对图注、章节页码、代码图、公式字符逐页图像复核。
2. 固定 BALROG 的历史提交、依赖和环境镜像；实跑最小测试与完整基准，记录成本和统计区间后才能称“复现”。
3. 对正式 ICLR 论文、仓库历史和教材的命令版本作一次以 commit 为单位的运行条件追溯。最新仓库配置不能直接充当 2025 年论文的实验锁定环境。
