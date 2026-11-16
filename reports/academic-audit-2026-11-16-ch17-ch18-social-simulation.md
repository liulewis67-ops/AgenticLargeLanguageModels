# 第 17–18 章学术审校：群体涌现、人物混杂、记忆检索与联合干预（2026-11-16）

## 1. 原始证据与审校范围

本轮对照：[*Agentic Large Language Models* arXiv:2610.13455v1 官方 HTML](https://arxiv.org/html/2610.13455v1) 第 17、18 章及各自 References；[fvdveen/generative_agents@6425988301d2ad75244ef1d4b827d26d66a4bba8](https://github.com/fvdveen/generative_agents/tree/6425988301d2ad75244ef1d4b827d26d66a4bba8) 的 Go 检索代码 `simulation_server/agent/retrieve.go`；[课程指南固定提交 9ebba180 中的 Exercise 3](https://github.com/DorotaMoravcikova/agentic-llm-course-excercises/blob/9ebba180b49491acc387de1ba5e8d42b0e0b4dfb/course/COURSE_GUIDE.md)；以及之前的 [Ch14/18 静态源码专项报告](source-code-audit-2026-11-14-ch14-ch18.md)。

本轮进行了**HTML→Markdown 书目结构核验、上游 Go 实现静态复核、实验因果识别审校、复现协议设计和回归检查**。没有实际启动 LLM/Go 社会模拟，也没有核对原始 PDF 物理页 335–358 的逐页截图。不得把已有算术自检当作真实世界实验结果。

## 2. 书目恢复：47 + 3，保持官方来源次序

| 章 | 原书 arXiv v1 HTML References | 既有错误 | 修复 |
| --- | ---: | --- | --- |
| 17 — Emergent Behavior | **47**：Ahn 到 Shuyan Zhou | 47 条原始书目被排版成无列表标记的普通段落；**Gao et al. (2023)** 的作者列表被拆为“Chen Gao, … Jinzhu Mao,”和“Jinghua Piao, …”；**Piao et al. (2025a)** 与 **Piao et al. (2025b)** 的作者姓名也各被截成两段。该错误使作者身份及文献条目数不可可靠检验 | 跨断行恢复完整姓名链并重建 **47** 条独立 Markdown 书目；对应 BALROG 的 `Kuciński`、`Fergus`、`Rocktäschel` 编码问题按前章已经核实的正确形式统一；保留所有页码锚点 |
| 18 — Case Study 4 | **3**：Baumeister、Park et al. (2023)、Park et al. (2022) | 三条文献为普通段落；Markdown 将两条 Park 文献顺序与官方 v1 HTML 不同 | 列表化为 **3** 条，并依 arXiv v1 原次序排列；不修改旧版 PDF 来源注释 |

中英外文书目直接统一为同一份条目数据。**书目数与次序的检查不等于逐条核对了所有出版社元数据或文中图形**。

## 3. 第 17 章：从观察到因果

本章早前的证据核查已包含：
- OASIS 百万模拟用户的计算限制与不能直接外推现实世界的说明（[原 NeurIPS 2024 论文](https://arxiv.org/abs/2411.11581)）。
- [Ashery 等人的命名游戏研究，*Science Advances* 2025](https://doi.org/10.1126/sciadv.adu9368)：不同模型/名称对所需的实验重复数不同；“未拒绝无偏假设”不等于证明等效。
- [CASA 正式 NAACL 2025 论文](https://aclanthology.org/2025.findings-naacl.222/)中的 **607+618+622=1847** 条公开数据与其来源版本；原有的静态 JSON 清单测试继续保留。

**本轮补充“可证伪实验方案”，而非重复声称这些旧核验是新发现。** 对群体规范形成/极化采用无互动、消息随机置换、脚本交流和自由互动等明确反事实处理，固定智能体参数、初始网络、交互调度和计算预算。每组跨完整随机种子重复，报告规范出现的比例、未收敛率、网络/记忆/人口规模消融与观测窗口。单轮模拟的数千条消息及单次模拟的上百万用户**不是独立模拟重复数**。

外部效度的边界：在人格、奖励、LLM 预训练偏见、网络与调度相同的前提下，能证实的是**在模拟系统内部的反事实差异**；没有人类与平台真实资料校准时，不能把结果直接写成真实政策效果。

## 4. 第 18 章：原书双条件、上游实现和因果混杂

### S18-01：源码证实「调倍数不等于改变排序」

[Go 检索源码固定提交](https://github.com/fvdveen/generative_agents/blob/6425988301d2ad75244ef1d4b827d26d66a4bba8/simulation_server/agent/retrieve.go) 的主要路径：

`extractValence → absolute → normalizeMap(0,1) → negativityBias → ValenceWeight → overall score`。

被归一化的是**候选记忆的绝对效价**；随后依据原始记忆的负号加倍。因此负值在乘以 1.5 后，其效价评分可以**大于 1**；不是负面直接以固定值 1.5 加到原始总分。起始配置中 `valence_w=0`，所以只修改 `negativity_bias=1.5`、而不启用 `valence_w>0`，**检索总分不会受该项影响**。此前 [公式算术回归](../scripts/audit-formula-regressions.mjs)已有正负样例、同幅度、零权重和近零分母自检；本轮保留并加固相应文字防回归。

### S18-02：源码与原书所称「仅一个变量不同」不一致

原书将 Dolores、Maeve 视为同样人格和环境、只改变负面记忆处理方式的匹配对照。已审计的上游 `scratch.json` 则含有不一致的 `learned`、生活方式和日程背景；二者的社交轨迹差异也可能源于**人物设定、上下文、时段与随机行为**。本轮给原文紧邻补充“理想化设计 ≠ 实际初始文件”的中英文编者勘误，建议使用相同人物基线模板、随机分配干预、交叉实验以及独立完整模拟重复。

### S18-03：课程实际上改变「检索」和「编码」两个机制

[课程指南 Exercise 3](https://github.com/DorotaMoravcikova/agentic-llm-course-excercises/blob/9ebba180b49491acc387de1ba5e8d42b0e0b4dfb/course/COURSE_GUIDE.md)明确要求同时启动 **valence-weighted retrieval** 与 **asymmetric sensory encoding**，但原书第 18.3.4 节把处理概括为加一项效价得分。之前编者建议的「人格提示 × 记忆检索」2×2 因子也无法单独识别此次课程联合干预。

因此本轮将编者方案更正为：先进行 **2×2：检索 R（关/开）× 编码 E（关/开）**，四组 `R0E0`、`R1E0`、`R0E1`、`R1E1`；分别估计检索、编码和交互作用。人格提示改变应独立对照，不要混入这一设计。实际字段拼写为 `asymetric_encoding`，需依据固定上游提交确认可设置性。

独立报告和可填写的完整实验配置位于 [Ch17–18 社会模拟复现矩阵](social-simulation-ch17-ch18-reproduction-matrix.md)。模板所有结果都标记「待实测」，没有虚构成功率、p 值或模型开销。

### S18-04：检索权重不等于心理学外推

`valence_w` 非零、实际检索名单变化以及下游交互变化是三种不同的可观测事件。必须分别记录记忆排序与选入、生成提示内容和完整社交行为（包括拒绝/失败）。即便获得统计差异，只能说明固定模拟条件下某项处理影响了系统输出，不能据此证明人类负面记忆具有相同的机制。

## 5. 回归、验证与未完成事项

`scripts/audit-source-fidelity.mjs` 增加 17、18 章的中英书目 **47/3** 条数、作者首末、碎片续行、双语一致性、图片数量和公式 (18.1) 保护，并要求“真实匹配对照”与“检索×编码 2×2 因子”编者警告存在。`scripts/audit-formula-regressions.mjs` 的既有负面效价算术例不冒充 Go/LLM 实际运行。

**尚未完成：**原始 PDF 物理页 335–358 的字形、图注与图形目视核对；从上游编译运行 Go、模拟实际触发四个处理条件；保存独立 seeds、完整代理轨迹和费用；执行外部社会行为数据标定与正式统计置信区间。第 19–20 章的后续工作应延续同样的“原始来源—实现—实验协议—外部效度”证据链。
