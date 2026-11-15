# 第17章新证据：CASA 公开数据完整性与命名游戏实验可复验边界

审校日期：2026-11-15。对象：[Agentic Large Language Models](https://arxiv.org/abs/2610.13455) 第17章 §17.3。

> **完成层级与界限：**本轮**实际获取并解析** CASA 作者公开仓库指定提交的六份 JSON 数据，统计每个数组的实例数及主键集合；阅读、核对所引 CASA 正式论文 PDF 第1–4页和命名游戏 Science Advances 2025 论文的实验页图像。**没有**拿到原书完整472页PDF并逐页渲染；没有运行任何 LLM、网页代理、命名游戏模拟或 GPT-4o 评判。论文原始实验结果未经本项目重复。

## A. CASA：正式论文、公开代码和公开数据的三方核验

### A1. 引文版本：2024 arXiv 与 2025 NAACL 正式出版

原书引用：Qiu et al. (2024)，*Evaluating Cultural and Social Awareness of LLM Web Agents*，arXiv [2410.23252](https://arxiv.org/abs/2410.23252)。该引用为**原书真实的预印本版本**，不能在不标注的情况下将年份替换成2025。

其正式出版记录：[Findings of NAACL 2025, pp.3978–4005](https://aclanthology.org/2025.findings-naacl.222/)，DOI [10.18653/v1/2025.findings-naacl.222](https://doi.org/10.18653/v1/2025.findings-naacl.222)。本轮**视觉检查正式论文 PDF 首两页**：第一页印有出版页码3978，第二页明确给出**17个国家、1225条用户查询、622条网页观察**。论文 §3.1–3.2进一步分解为 S1-Adhere=607，S1-Violate=618，S2-online-shopping=284、S2-social-forum=338。

**同一篇工作存在不同阶段数据规模：**非正式的较早投稿 PDF 曾报告1480条查询、731条观察。作为复现基准应优先固定其所采用的**确切论文版本**；本报告以下仅以正式NAACL PDF为对照，不混入早期稿的数据。

### A2. **实测六份 JSON**，不是按作者摘要推断

源码固定为 [SalesforceAIResearch/CASA@3ddd1b568a7a0e906a17fc8219d7000a83e268ba](https://github.com/SalesforceAIResearch/CASA/tree/3ddd1b568a7a0e906a17fc8219d7000a83e268ba)。使用 GitHub blob API 实际读取如下 **6 个文件**，以严格 `JSON.parse` 解析每个数组，并检查非空 `index`、`user_query`、`violated_norm`，统计每个文件内主键唯一性及对应组的配对关系：

| 评测分组 | 仓库文件 | 实测条数 | 论文所列条数 |
| --- | --- | ---: | ---: |
| S1-A | `s1_adhere_data/online_shopping.json` | 289 | 289 |
| S1-A | `s1_adhere_data/social_discussion_forum.json` | 310 | 318 |
| S1-V | `s1_violate_data/online_shopping.json` | 289 | 300 |
| S1-V | `s1_violate_data/social_discussion_forum.json` | 310 | 318 |
| S2 | `s2_data/online_shopping.json` | 284 | 284 |
| S2 | `s2_data/social_discussion_forum.json` | 310 | 338 |
| **总计** | **六份公开 JSON 数组** | **1792** | **1847** |

**逐组差异：**S1-A 607→599，**缺8**；S1-V 618→599，**缺19**；S2 622→594，**缺28**，合计**55**。这不是文件不能解析导致的差额：六个文件均为有效 JSON 数组；文件内 `index` 无重复；非空关键字段检查未发现缺失。S1-A 与 S1-V 的编号在两个任务域中完全对应；S2 购物的284个编号全都包含于S1-V购物的289个编号中（有5个S1-V项目未见S2网页）；S2论坛310项完全包含于相应S1-A论坛310项。**这些是数据清点事实，不足以确认55条为何没有公开，也不能认定缺失是随机的。**

作者官方仓库 [README](https://github.com/SalesforceAIResearch/CASA/blob/3ddd1b568a7a0e906a17fc8219d7000a83e268ba/README.md) 仍写1225个查询和622个观察，但在这一提交下**实际文件内容与声明不符**。这意味着照README默认路径运行，只能得到当前公开子集的结果，不能无条件视为对作者完整样本的复现。

### A3. 不同指标不能混为一个准确率

- **S1-V** 的 Awareness Coverage Rate（AC-R）与 Educational Rate（Edu-R）评估用户直接提出违反文化/社会规范需求时，代理能否识别规范并适当解释。
- **S1-A** 的 Helpfulness Rate（Help-R）考察用户请求本身遵守规范时的帮助性。
- **S2** 的 Violation Rate（Vio-R）考察网页内容包含误导信息时，代理是否受到影响而采纳不当建议。
- 作者报道的“**<10%识别覆盖**”与“**>40%违规率**”分别属于不同的分组、分母和判分准则。自动评审还使用 **GPT-4o 作 LLM-as-a-judge**，因此更换评判模型版本、提示模板、评判语句本身可能改变结果。

需要报告每个分组的请求量、成功/失败量、无效结果量、模型快照、提示和判分器版本；不能把六份文件合并后算一个“文化意识准确率”，也不能把论文结果称为具有真实社会外部效度的文化差异测量。

### A4. 可运行的离线清单工具

```bash
git clone https://github.com/SalesforceAIResearch/CASA.git
cd CASA
git checkout 3ddd1b568a7a0e906a17fc8219d7000a83e268ba
cd ../AgenticLargeLanguageModels
pnpm docs:audit-casa -- ../CASA --expect-audited-release
# 预计验证当前公开提交的1792条并成功
pnpm docs:audit-casa -- ../CASA --expect-paper
# 故意以论文1847条为门槛，将因实际缺55条而非零退出
pnpm docs:audit-casa --self-test
# 无外部数据依赖的合成夹具单元测试
```

工具逐文件 `JSON.parse` 检查六份数组、必要字段和主键，检查 S1-A/S1-V 编号配对及 S2 对应来源，并打印论文/公开版本分组数量与缺额。**只做离线输入验收，不运行模型或重新评分。**CI只跑隔离夹具。

## B. 命名游戏：原始出版论文的样本、模型和统计边界

原书引用 Ashery et al.（2024），arXiv [2410.08948](https://arxiv.org/abs/2410.08948)；其正式论文发表于 *Science Advances* **11(20), eadu9368 (2025)**，DOI [10.1126/sciadv.adu9368](https://doi.org/10.1126/sciadv.adu9368)。对照了[机构存档的出版社论文 PDF](https://openaccess.city.ac.uk/id/eprint/35211/1/sciadv.adu9368.pdf)的实验设置、图2和方法页图像。研究的目标是 **LLM群体本身的命名约定形成**，作者明确不将智能体视为人类的替代样本。

- 图1使用同质 **N=24个智能体、W=10个名称、H=5次交互记忆**，在四种模型上各报告40次独立模拟。对照的*理论最小命名游戏*使用10,000次运行；二者不是等量实证重复。
- 图2的群体偏好对照：四种模型的 **40、40、27、20** 次独立运行；不能将论文出现的“40次”无差别用于全部实验。
- 图2单体首轮的精确检验报告四模型 `p=0.068, 0.116, 0.757, 0.849`。在常见0.05阈值下是“未能拒绝无偏选择零假设”，**不等于已经证明效应为零、等效或绝对无个体偏见**；尤其不能把“群体偏见独立于个体偏见”解读为统计上已经严格分离所有个体层因素。
- 固定记忆、名称集合、提示、模型版本/量化、温度和抽样方式会影响收敛；论文还采用不同定义的临界少数群体阈值。需要记录跨种子分布、退出条件、未收敛比例、真实 API 错误率与结果范围。
- 原作者提供[GitHub源代码与数据](https://github.com/Ariel-Flint-Ashery/AI-norms/tree/e57aeef151a46108fd1521c66e345e557e57dbf6)以及 [Zenodo](https://doi.org/10.5281/zenodo.14937173)。仓库README明确提示部分理论模拟数据过大而未直接提交，提示模板包含Llama模型相关的特殊格式；应记录源代码与外部资产/模型精确版本后才能复跑。**本轮只静态检查了README和论文**。

## C. 本轮更正与下一个证据关口

- 中英文第17章保留原书对 **2024预印本的引用**；补充可追溯的2025正式出版信息，**不篡改原文的历史引用**。
- 清晰标示 **CASA正式论文1847 vs 实际公开1792**、两项不同指标的适用分组及审查边界。
- 提供 `scripts/audit-casa-release.mjs`（无第三方运行时依赖）、`pnpm docs:audit-casa` CLI 与CI合成夹具检查；中英README同步导航。
- 下一步：获得论文中缺失55条的公开版本/作者确认，固定CASA完整模型结果与GPT-4o裁判版本；若需真正复验命名规范，按模型逐配置执行多随机种子实验；继续追索可渲染的**原书**472页PDF进行逐页视觉对照。

**原则：**源代码检查、原始PDF指定页面核验、公开数据全量扫描与研究模型复现是四种不同的证据，不能互相冒充。
