# 源码证据、引文真实性与实验可复现性审计：第 14 / 18 章（2026-11-14）

> **核验性质：** 这一轮取得了外部参考文献的出版记录与课程源码，完成**静态源码核对**。没有获取并逐页渲染原始 472 页 PDF，没有执行真实 LLM/医学模型/社会模拟，**不宣称实验复现成功、统计显著性或临床效度**。所有代码结论针对以下**固定提交**，并非对以后版本的承诺。

## 1. 原始书籍与 PDF 逐页审校状态

- 原书：*Agentic Large Language Models*，Max van Duijn、Michiel van der Meer、Aske Plaat、Niki van Stein，arXiv:2610.13455v1，2026-10-09，**472 个物理 PDF 页**：https://arxiv.org/abs/2610.13455
- 仓库 `config/chapters.json` 把第 14 章映射为物理页 **277–282**，第 18 章映射为 **353–358**。这些数字是当前仓库的*页码元数据*，并不是新核对过的 PDF 视觉事实。
- 当前审校环境无法完整获取和渲染原 PDF；网页的可提取版本可辅助对照标题及公式，但**不具有 PDF 页面边界、原图和字形的证据效力**。

| PDF 物理页 | 章节 | 需逐页检查的项目 | 本轮状态 |
| --- | --- | --- | --- |
| 277 | 14 | 标题、来源及 MCP 表述 | PDF 图像待核 |
| 278 | 14 | 教学仓库与训练代码文件名 | PDF 图像待核；**上游源码已核** |
| 279 | 14 | 规划器、执行器及样本生成 | PDF 图像待核；**上游源码已核** |
| 280 | 14 | LLM 规划器代码示例 | PDF 图像待核；**上游源码已核** |
| 281 | 14 | 评测与错误分解 | PDF 图像待核；**上游源码已核** |
| 282 | 14 | 讨论、引文和章节尾部 | PDF 图像待核 |
| 353 | 18 | 实验目标和 Generative Agents 引文 | PDF 图像待核；**书目信息已核** |
| 354 | 18 | 人格与心理学依据 | PDF 图像待核；**书目信息已核** |
| 355 | 18 | NEVER 与模型假设 | PDF 图像待核；**原论文依据已核** |
| 356 | 18 | 三角色初始条件、日程 | PDF 图像待核；**上游配置已核** |
| 357 | 18 | 公式 (18.1)、效价实现及安装链接 | PDF 图像待核；**上游 Go 实现已核** |
| 358 | 18 | 讨论与三篇参考书目 | PDF 图像待核；**书目元数据已核** |

**因此本轮原始 PDF 的“逐页视觉审定”完成数仍为 0/12；不得将上表的源码或书目核验计入 PDF 页面完成数。**

## 2. 固定的上游源码证据

| 对象 | 提交（main） | 直接检查的路径 |
| --- | --- | --- |
| 第 14 章教学代码 | `Agentic-LLMs/Agentic-AI-tutorials@b7b8f683fbcb0d51b6ba8d62b389c7be6c24de3a` | `case3/0_train_models.py`、`1_basic_mcp.py`、`2_llm_mcp.py`、`readme.md` |
| 第 18 章配套课程 | `DorotaMoravcikova/agentic-llm-course-excercises@9ebba180b49491acc387de1ba5e8d42b0e0b4dfb` | `course/COURSE_GUIDE.md`、`simulation_server/agent/retrieve.go`、`simulation/base_cafe_spiral/personas/*/bootstrap_memory/scratch.json` |
| 第 18 章推荐运行代码 | `fvdveen/generative_agents@6425988301d2ad75244ef1d4b827d26d66a4bba8` | `simulation_server/agent/retrieve.go`，与上表课程仓库所检文件 blob SHA 同为 `e228294e9b3c29911993eb0d63ea7fae5c9b45a2` |
| 原书中失效链接所指仓库 | `DorotaMoravcikova/generative_agents@d9417d9ce8f7983ebc9d30a8b66f1805bd62ef02` | 当前仅见 `main` 分支，没有 `course-exercises` |

仓库直接链接：[第 14 章 case3](https://github.com/Agentic-LLMs/Agentic-AI-tutorials/tree/b7b8f683fbcb0d51b6ba8d62b389c7be6c24de3a/case3)、[第 18 章课程指南](https://github.com/DorotaMoravcikova/agentic-llm-course-excercises/blob/9ebba180b49491acc387de1ba5e8d42b0e0b4dfb/course/COURSE_GUIDE.md)、[Go 检索源码](https://github.com/fvdveen/generative_agents/blob/6425988301d2ad75244ef1d4b827d26d66a4bba8/simulation_server/agent/retrieve.go)。

## 3. 第 14 章：可复现性阻断与证据

| ID | 严重度 | 已核实源码事实 | 意义 / 修复要求 |
| --- | --- | --- | --- |
| R14-01 | **高** | `0_train_models.py` 对 Pima、乳腺癌与心脏病数据集分别直接 `RandomForestClassifier().fit(X, y)`，**没有训练／测试拆分**；`1_basic_mcp.py` 和 `2_llm_mcp.py` 再从这些数据抽样生成案例。课程 README 也明确指出该缺陷。 | 当前端到端分类评价属于**训练集重采样**，不可解释为独立泛化准确率；需先划分患者／来源隔离的留出集合，再训练、封存模型与测试数据。 |
| R14-02 | **高** | `2_llm_mcp.py` 的 `call_llm_fn` 返回固定 JSON 演示字符串，`LLMPlannerAgent.step()` 只有任务注释与输入解包、未实现路由；底部的路由评估也标为 TODO。 | 原样脚本不是可运行的完整 LLM 对照实验；应要求实现、版本固定、JSON 校验、异常日志和可重跑评测。 |
| R14-03 | 中 | `1_basic_mcp.py` 的结果字段叫 `y_pred`，但图表条件检查 `"Prediction" in results_df.columns`；`2_llm_mcp.py` 同样存在此条件。 | 预测类别图不会进入预期绘制分支。检查图表代码时必须以实际字段名为准。 |
| R14-04 | 中 | `0_train_models.py` 中随机森林未提供 `random_state`，抽样 `sample(1)` 也不固定随机生成器；病例 ID 用 UUID 整数串前 4 位。 | 重跑无法严格对齐案例／模型，短 ID 还有碰撞风险；改用固定种子、真实唯一 ID 与全链路 provenance。 |
| R14-05 | 中 | 仓库中真实文件名包含数字前缀：`0_train_models.py`、`1_basic_mcp.py`、`2_llm_mcp.py`；原书用的是不带前缀的教学简称。 | 保留忠实原文，在双语编者注中提供可执行文件名与目录。 |

**静态判断，不是运行结果：** 第 14 章的“规划器性能”和“执行器分类性能”应独立核查，且对照组必须使用同一留出病例；现有脚本既不能支持公平的临床结论，也不能提供公平的 LLM vs 规则规划器因果比较。

## 4. 第 18 章：公式 (18.1) 的实现已定位

在上游 Go `simulation_server/agent/retrieve.go` 的 `retrieveForFocalPoints` 中，实际计算顺序为：

1. 对每条记忆提取带符号效价 `Valence`；
2. `absolute` 计算效价绝对值；
3. `normalizeMap(..., 0, 1)` 在候选记忆集合中按 min–max 缩放；
4. `negativityBias` 对**原始带符号效价小于零**的记忆，将归一化分数乘 `p.state.NegativityBias`；
5. 四个维度分别乘内部权重 1 和各自可配置 `RecencyWeight / RelevanceWeight / ImportanceWeight / ValenceWeight` 后相加；取评分最高的 30 条记忆。

因此，**1.5 是对归一化后负面效价的放大系数，而不是 `α_val` 本身**。此处上游实际还设有 `ValenceWeight` 的开关。其公开的 Dolores / Maeve 初始 `scratch.json` 均为 `valence_w=0`、`negativity_bias=1.0`、`asymetric_encoding=false`；课程指南第 3 练习要求手动启用效价检索和扩展编码。**仅把 negativity_bias 改成 1.5，而仍保留 valence_w=0，不会使效价项影响检索总分。**

- **边界样例：** 候选绝对效价 0、2、4、8 经缩放为 0、0.25、0.5、1；其中负面 −2 / −8 在倍率 1.5 时为 0.375 / 1.5，负面评分**可能超过 1**，因为倍率在归一化后施加；不能声称最终 V 总落在 [0,1]。
- **零方差样例：** 所有候选绝对效价相同时，`normalizeMap` 返回 0.5，而不是除以零；这是上游明确写出的例外处理。
- **可复现性注意：** `highestNValues` 返回 map，后续 map 遍历构建 `outNodes`；无额外排序时，相同候选集的最终呈现顺序并未由该函数保证稳定。记录最终召回集合与序列，并单独检查同分情形。
- **混杂项：** 两位咖啡师虽有相同的 `innate` 核心词，但各自 `learned`、`daily_plan_req` 与 `lifestyle` 不一致（例如休息时间、下班后活动不同）。在固定配置下直接比较 Dolores 与 Maeve，**不能把差异全部归因于记忆机制**。
- **第二个操纵因素：** 指南第 3 练习要求同时启用 valence-weighted retrieval 和 asymmetric encoding；该实验是两个架构改动的**联合处理**，如果研究目标是单独估计检索加权效应，需要再做“仅检索 / 仅编码 / 均开启 / 均关闭”的 2×2 消融。

**原书安装链接问题：** `https://github.com/DorotaMoravcikova/generative_agents/tree/course-exercises` 当前返回不存在的分支；课程官方指南实际推荐 `https://github.com/fvdveen/generative_agents`。保留原文链接作历史来源，另加编者勘误导航。

## 5. 文献真实性与证据层级

| 文献 | 外部记录 | 状态 |
| --- | --- | --- |
| Baumeister, Bratslavsky, Finkenauer & Vohs (2001), *Bad Is Stronger Than Good* | *Review of General Psychology* 5(4):323–370；DOI [10.1037/1089-2680.5.4.323](https://doi.org/10.1037/1089-2680.5.4.323) | **出版社书目已核实** |
| Park et al. (2023), *Generative Agents* | UIST 2023，DOI [10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763) | **ACM 书目已核实** |
| Park et al. (2022), *Social Simulacra* | UIST 2022；[Google Research 论文条目](https://research.google/pubs/social-simulacra-creating-populated-prototypes-for-social-computing-systems/) | **作者/年份/题名已核实** |
| Bowen, Kark & Kensinger (2018), *NEVER Forget* | `Psychonomic Bulletin & Review` 25(3):870–891，DOI [10.3758/s13423-017-1313-9](https://doi.org/10.3758/s13423-017-1313-9) | **PubMed 书目已核实**；这是**上游课程代码 README** 对 NEVER 的理论引用，不应偷偷冒充原书第 18 章现有参考书目 |
| Hou, Zhao, Wang & Wang (2026), MCP Landscape | `ACM TOSEM` 35(10)，DOI [10.1145/3796519](https://doi.org/10.1145/3796519) | **ACM 出版记录已核实** |

这些核验仅覆盖**明确列出的 5 项**，不等于原书全量引文校核。**元数据真实性不等于论文实验结论已独立复核。**

## 6. 下一阶段可执行复现门槛

- 第 14 章：固定上游提交、Python 与依赖版本；先进行患者/来源分组拆分；固定训练和采样种子；验证模型不读取测试标签；完成并记录真实 `LLMPlannerAgent.step`；同案配对比较、盲法评测和路由/分类双重混淆矩阵；记录 token、时延、失败与拒答。
- 第 18 章：固定 Go 提交与模型快照；对效价 0、同幅度正负、极小负值、全等、缺失值、同分、不同候选分布设置单元反例；明确变体配置；使用相同人物简介和日程的交叉/随机分配设计；将“检索加权”和“非对称编码”分开消融，再执行多种子的完整模拟。
- 第 21 章：公式 (21.7) 的归一化恢复量仍需保留近零分母规则及反事实干预控制；本轮没有执行模型激活修补，状态保持**方法审核、未复现**。

**当前最高证据等级：** 文献的出版社/数据库元数据已核实、外部源码在固定提交上已完成静态核查。**未达到：** 原 PDF 视觉逐页校对、上游程序完整运行、独立实验数据重算、论文级复现。

