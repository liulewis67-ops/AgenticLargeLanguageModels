# 第19–20章原始论文、数据规模与复现性专项审计（2026-11-15）

> **证据边界：** 直接阅读了《Agentic Large Language Models》官方 arXiv v1 HTML 相应章节、所引 ACL 2022 和 ICLR 2025 Workshop 的**原论文 PDF 首页**，并核对 SICA 上游 GitHub 的固定提交。未取得/逐页渲染原书 472 页 PDF，没有执行编码智能体自修改或 Natural Instructions 模型训练。以下结果为**引文、表格与代码静态审计**，不是独立实验复现。

## 1. 核验清单和可信度

| ID | 原书位置 | 来源 | 已核验范围 | 结论 |
| --- | --- | --- | --- | --- |
| P19-01 | 第19章 §19.3（仓库映射物理 PDF 页363–390） | [Mishra et al., ACL 2022](https://aclanthology.org/2022.acl-long.244/) | [官方 PDF](https://aclanthology.org/2022.acl-long.244.pdf)首页摘要与方法，已查看首页图像 | **61 个任务、193k 条实例属实**；必须说明是任务级留出、不是随机混合样本 |
| P19-02 | 第19章 §19.3 | 同上 | 作者目标、BART 实验语境 | 原书说“LLMs benefit”应谨慎解释：原文的特定实验不证明今日各类聊天 LLM 的增益 |
| P20-01 | 第20章 §20.2.3（仓库映射物理 PDF 页391–402） | [Robeyns et al., ICLR 2025 Workshop](https://openreview.net/pdf?id=rShJCyLsOr) | 原论文 §4、表1、首页 PDF 图像 | **17% → 53% 属实**，但只针对**固定随机 50 题 SWE-bench Verified 子集**，非全部题目 |
| P20-02 | 第20章 §20.2.3 | 同上，Table 1 | 迭代0、14、15的列值 | SWE-bench 子集比例分别为0.17、0.53、0.51；**不构成每次迭代单调增长** |
| P20-03 | 第20章 §20.2.3 | 同上，Table 1 | 成本/时间/token列的原始表头与脚注 | 对比迭代0和14：平均每题总评测成本1.91→2.20美元，平均耗时130.2→117.1秒，平均token0.24M→0.25M。**这几列按作者定义是所有测试基准实例的汇总值，而非单独 SWE-bench 指标** |
| C20-01 | SICA 实验代码 | [MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent/tree/ed8275dca4d3c5dbf77229964351fe9b424797dc) | 上游 `master@ed8275dca4d3c5dbf77229964351fe9b424797dc` 的 README、benchmark registry、实现与依赖 | 默认 benchmark registry **仅启用 GSM8K**；直接按 README `python runner.py` 运行**不是**论文 §4 那组基准 |
| C20-02 | SICA 复现条件 | 同上 | `swebench_verified.py`、`requirements.txt`、`config.py`、`runner.py` | 代码能以 seed 和 subset_size 抽题，但**当前类默认 subset_size=100**，而论文使用50；依赖未整体锁版本，模型枚举也需与发表时快照对应 |

所有“已核验”结论都是当前读取到的**指定论文文本或上游仓库提交**；不能用旧实验准确率推断新模型或今日软件环境下的复现效果。

## 2. 第19章 Natural Instructions 的研究设计

原论文是 Mishra, Khashabi, Baral 与 Hajishirzi（2022），*Cross-Task Generalization via Natural Language Crowdsourcing Instructions*，ACL 2022，DOI [10.18653/v1/2022.acl-long.244](https://doi.org/10.18653/v1/2022.acl-long.244)。

- **任务数与样本数：**论文定义并汇总61项独立NLP任务、193k个输入输出实例；数字核对通过。
- **实验单位：**训练集和测试集按**任务**分开，测试时借助未见任务的自然语言说明。不要把同一任务内随机划分的普通监督学习精度误写为“跨任务泛化”。
- **实验结果外推：**论文在其训练及评测配置中验证输入指令有益，摘要明确讨论当时的生成式预训练语言模型及BART，**不等于新的LLM家族在所有合成任务上均有同等收益**。
- **版本标识：**该61任务版本与后续的SUPER-NATURALINSTRUCTIONS等扩展数据集不能混用统计。
- **未复现：**未获得原始划分ID与模型运行日志，未安装数据和重复训练，没有新增任何准确率/置信区间。

## 3. 第20章 Self-Improving Coding Agent 的原始结果与配置

原论文：Robeyns, Szummer 与 Aitchison（2025），*A Self-Improving Coding Agent*，ICLR 2025 Workshop SSI-FM（[原论文 PDF](https://openreview.net/pdf?id=rShJCyLsOr)）。

**原书保留**“17%→53%”的核心数值，新增明确标为非原文的补充：
- SWE-bench Verified 50 个固定随机题目的**特定子集**，不是完整基准；表1中迭代0→14为0.17→0.53、迭代15为0.51。
- 原论文 §4 还包含50个LiveCodeBench题目，以及自建文件编辑和符号定位任务。性能和效率记录**按基准项分别解释**；总体平均时间/成本/token不能等同“SWE-bench-only成本”。
- 发布源码的默认 registry 只启用 `GSM8KBenchmark`，`SWEBenchBenchmark` 被注释；源码 `SWEBenchBenchmark(seed=1, subset_size=100)` 的默认规模与论文的50不一致，且 `base_agent/requirements.txt` 大多不锁版本。
- 论文中的自改进不是底层LLM权重训练。要判定“自修改本身是唯一原因”，还需要在同一任务、模型、调用预算下对比**可编辑元层**与**固定元层/人工脚手架搜索**的因果对照。
- 论文可见最终表格不能单独恢复所有原始评测问题ID、运行种子、LLM精确快照与逐题日志；默认脚本启动成功也不足以宣称复现论文结果。

### 推荐最小重跑清单（为读者提供，不代表执行）

1. 固定 `MaximeRobeyns/self_improving_coding_agent@ed8275dca4d3c5dbf77229964351fe9b424797dc`，独立Docker环境。
2. 显式启用并记录全部四个论文基准；禁用默认与论文不一致的基准配置；抽取、冻结**确切50题实例ID**，不要只依赖不带数据集revision的随机种子。
3. 固定 SWE-bench 数据revision、swebench执行器版本、镜像、Python及包版本、LLM快照、推理温度、系统提示及成本计价规则。
4. 每次迭代保存候选代码、变更diff、完整运行轨迹、各题分数、失败原因、评价器输出、成本及汇总结果。
5. 对相同冻结题目做受控对照、多次独立运行、配对不确定性及对未见题目外推；避免用单轨迹最佳分数充当统计显著性结果。

## 4. 原书 PDF 的未完成项

本轮**尝试获取**官方 https://arxiv.org/pdf/2610.13455v1 ：PDF 浏览接口因文件大小限制无法解析；运行容器的网络下载也未成功。因此：**原书 PDF 物理页的视觉校对完成数未增加。**本报告中的“原书 §19 / §20”引用基于官方 v1 HTML，与原始 PDF 物理页的图文逐页验收仍应另行执行。

另见当前第19章英文／中文 Markdown：跨第6页图19.4的 `PersonaHub [Ge ... et al., 2025]` 书目引用被分页和图片插入打断。这是**待 PDF 核验的文档重流问题**，暂不擅自移动页面边界或原图。

## 5. 本轮修改与质量门禁

- 第19、20章共**四份中英文 Markdown**新增有明确“非原文”标识的一级文献校核注，**原作者正文、公式、图片、页锚与代码围栏保持不变**。
- 扩展 `pnpm docs:audit-source`，通过固定文本断言避免论文限定条件被编辑者无意删除；它不重新访问 ACL、OpenReview 或 GitHub，也不执行实验。
- 中英文 README 增加本报告导航，供读者从结果出发追溯原始文献。

**结论：**本轮将第19章数据规模与第20章突出数值的证据等级由“仅书稿解释”推进到“外部原论文 PDF 指定页面核查+源码静态核查”。原始书 PDF 的逐页保真审定、统计显著性重算和实验复现仍属未完成。
