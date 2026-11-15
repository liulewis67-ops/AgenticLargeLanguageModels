# 第15–17章原始论文、公开数据与复现性审计（2026-11-15）

> **执行状态与证据等级：**本轮直接核对[原书官方 arXiv v1 HTML](https://arxiv.org/html/2610.13455v1)，查阅 ACL/NAACL/NeurIPS 原论文和官方出版元数据；**实际读取并逐条检查** ToMBench 固定源码提交下的20个 JSONL 文件（2,860 条）。同时静态阅读评测源码和 OASIS 资源表。**没有**取得原书 PDF 页面图像，没有跑模型推理、多种子模拟或独立重算学术显著性。上述四种工作不能互相替代。

## 1. 第15章：ToMBench 论文原文与模型测评之间的缺口

**原书出处：**第15章 §15.5.1，arXiv [2610.13455v1](https://arxiv.org/html/2610.13455v1) 原文明确声称 ToMBench 涵盖“all seven ATOMS mental state categories”，并声称新测试题得到不进入预训练语料的保证。我们**保留原书正文**，用明显标识的双语编者注纠正。

**原论文：** Zhuang Chen et al. (2024), *ToMBench: Benchmarking Theory of Mind in Large Language Models*, ACL 2024，DOI [10.18653/v1/2024.acl-long.847](https://doi.org/10.18653/v1/2024.acl-long.847)，[官方 PDF](https://aclanthology.org/2024.acl-long.847.pdf) 首页已经进行图像检查。

| 项目 | 原始来源可核实事实 | 解释界限 |
| --- | --- | --- |
| ATOMS 原始分类 | Beaudoin et al. 2020 包含**七个**维度、39项子能力 | 不能直接视为 ToMBench 具体测试覆盖 |
| ToMBench 分类 | 剔除需要视觉输入的 **Percept** 维度后，仅**六类**、**31**项子能力 | **原书“覆盖所有七类”错误**，引用来源不支持 |
| ToMBench 测试集 | **8**项 ToM 任务；全量 **2,860** 道选择题 | “8项任务”、“9个故事主题”、“6项能力类别”是不同分类层次 |
| 题目组成 | 论文表1任务视图 **2,470** 道；补充未覆盖能力至总数 **2,860** | 后续补充题不是新增的第九项任务 |
| 人类基线 | 作者招募 **20 名研究生**用于对照 | 不是具有充分人口代表性的“人类平均能力” |
| 双语构建 | **中文题目为原始撰写**，英文由 GPT-4 翻译再经人工检查 | “中英文分别从零编写”不准确；跨文化等值性需额外检验 |
| 预训练泄漏 | 论文发布时由作者新建样本并避免直接复用旧基准 | 无法保证2024年公开后仍未进入任意后来模型的训练语料 |

参考原论文 §2.2、§3、§4 和 Limitations；上游固定提交 [zhchen18/ToMBench@4f491f0784ed31ef93a8837615fbc48f885cf78d](https://github.com/zhchen18/ToMBench/tree/4f491f0784ed31ef93a8837615fbc48f885cf78d)。

## 2. **已实际执行的数据集逐条核查**：2,860 条、20 个文件

通过 GitHub API 读取上方提交的全部 `data/*.jsonl` 文件，不是从论文摘要推断数量。将每个**非空物理行**作为一条数据，严格用 JSON.parse 检查标准 JSON，再独立验证双语字段和答案标签。

| 文件 | 记录数 | 含非标准 `NaN` 的记录 |
| --- | ---: | ---: |
| Ambiguous Story Task | 200 | 0 |
| Completion of Failed Actions | 20 | 0 |
| Discrepant Desires | 20 | 0 |
| Discrepant Emotions | 40 | 0 |
| Discrepant Intentions | 40 | 0 |
| Emotion Regulation | 20 | 0 |
| False Belief Task | 600 | 0 |
| **Faux-pas Recognition Test** | **560** | **280** |
| Hidden Emotions | 80 | 0 |
| Hinting Task Test | 103 | 0 |
| Knowledge-Attention Links | 20 | 0 |
| Knowledge-Pretend Play Links | 30 | 0 |
| Moral Emotions | 40 | 0 |
| **Multiple Desires** | **20** | **20** |
| Percepts-Knowledge Links | 40 | 0 |
| Persuasion Story Task | 100 | 0 |
| Prediction of Actions | 20 | 0 |
| Scalar Implicature Test | 200 | 0 |
| **Strange Story Task** | **407** | **204** |
| Unexpected Outcome Test | 300 | 0 |
| **总计** | **2,860** | **504** |

**数据层的实际发现：**
- 2,860条记录数与论文一致，但**仅2,356条可直接用严格 JSON 解析**；504条存在未加引号的 `NaN` 记号，**不符合标准 JSON**。这不是说504条样本失效：Python 默认 `json.loads` 容忍 `NaN` 并将其表示为非标准浮点值，而严格解析器拒绝。
- Faux-pas 280条和 Strange Story 204条的中文 `选项C/选项D`（以及英文对应字段）是 `NaN`，合计 **484条**；Multiple Desires 20条的 `类型` 字段是 `NaN`。
- `Knowledge-Attention Links` 至少一条金标准答案为 `A. ` 而非标准单字母 `A`；如使用原样字符串比较，会把正确选择A判错。
- `序号\nINDEX` 在一个文件内大量重复，不是全局唯一的题目 ID（可能标记故事或子类）。对题目冻结划分应使用**源码提交、文件名、行号与输入内容哈希**，而不是仅凭 `INDEX`。
- 所有严格可解析行均具备非空中文和英文字段；对非标准行的语义完整性还应在本地预检脚本做显式空值归一化后重新统计。

**结论：**书中的“2,860个测试条目”计数可复核；但读取格式、二／四选项分流、标签规范化和独立样本单位仍是评测真实性的前置条件。不同解析库使用方式可能得到不同有效样本集或直接出错。

## 3. ToMBench 上游评测代码的关键边界

代码版本同上，以下是**代码阅读确认**而非模型运行结果：

| ID | 实现位置 | 源码事实 | 对复现的影响 |
| --- | --- | --- | --- |
| T15-C1 | `run_api.py` §`if d['选项C'] != None` | Python `json.loads` 把 `NaN` 读为浮点 NaN，`NaN != None` 为真；于是误进四选项 `format_prompt_4`，其对 C 调用 `.replace()` | 至少484个二选项样本存在**异常中断风险**；必须将 NaN 按缺失值规范化后再组装提示 |
| T15-C2 | `run_huggingface.py` §`enumerate(data[:10])` | **每个文件只处理前10题** | 默认调用不能声称评测全量2,860条，不能外推到完整任务/能力样本 |
| T15-C3 | `run_api.py` §`try: ... except: continue` | 文件读取失败时无异常归档而直接跳过；API 并发 `ThreadPoolExecutor(max_workers=32)` 没有逐项检查返回值 | 应显式统计实际请求数、成功数、失败数、重试数和遗漏题目，避免以部分集计算指标 |
| T15-C4 | `get_results.py` §`if answers[i] == most_common_element(preds[i])` | 答案未规范化，而公开数据出现 `A. ` | 即使预测了A，这条题目也可能被错误计分 |
| T15-C5 | `get_results.py` 的汇总循环 | 只有至少一次计分正确的任务才加入 `acc_per_task`；任务零分时不生成结果键；另假定每题始终有 `try_times` 次结果 | 若指标结果缺键/日志不完整，应视为失败而非忽略；建议对**所有定义的能力和任务**初始化计数与零分 |
| T15-C6 | `run_api.py` 使用 `random.seed(42)`，并多次打乱选项 | 选项顺序改变且同一题重试多次，线程中异步完成 | 固定题目、提示模板、随机状态、调用结果、顺序和模型版本后，再建立条件一致的结果与区间 |

### 本地可运行预检

新脚本 `scripts/audit-tombench-release.mjs` **不需要任何模型密钥或第三方依赖**：

```bash
git clone https://github.com/zhchen18/ToMBench.git
cd ToMBench
git checkout 4f491f0784ed31ef93a8837615fbc48f885cf78d
cd ../AgenticLargeLanguageModels
pnpm docs:audit-tombench -- ../ToMBench/data
# 对非标准 JSON 和非规范答案实行严格失败门禁：
pnpm docs:audit-tombench -- ../ToMBench/data --require-strict-json
```

工具实际读取本地已克隆的所有记录，输出各文件行数、可严格解析数、`NaN` 恢复统计、金标准标签异常、双语字段和重复局部序号。**默认将已知 NaN 标为诊断告警，不静默修改源数据**；严格模式失败便于评测前明确修复。CI 则只运行其**小型人工夹具自检**，不声称在 CI 下载了完整上游数据。

## 4. 第16章：LLM-Coordination 的书目和统计独立性

原始出版：Agashe, Fan, Reyna & Wang (2025), *LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models*, Findings of NAACL 2025，**第8053–8072页**，[DOI:10.18653/v1/2025.findings-naacl.448](https://doi.org/10.18653/v1/2025.findings-naacl.448)。

- **确定的原书书目错误：**arXiv v1 原书 §16.4 References 写 **8038–8057**，与[ACL出版记录](https://aclanthology.org/2025.findings-naacl.448/)及[PDF印刷首页](https://aclanthology.org/2025.findings-naacl.448.pdf)的 **8053–8072** 不一致。官方论文**PDF首页已视觉检查**。本项目直接纠正双语书目页码，正文加显式“编者修改”说明。
- **样本层级：**论文 §3.3 有 **66个场景**，其中 Overcooked 25、Hanabi 28、另外两种协作游戏共13；每个场景创建**3道问题**，总共198个 CoordQA 题。198不是198个统计独立游戏情境。做 bootstrap/置信区间时应优先以场景分组。
- **评价层级：**真实代理执行的 Agentic Coordination 与离线多项选择的 CoordQA 是不同测量，不可互相替代；四款游戏里尤其需要分开环境观测、协同计划以及伙伴信念推断。
- **静态源码版本：**[UCSB-AI/llm_coordination@586733a37d240b9592645bb0d7d10f73c1103f7f](https://github.com/UCSB-AI/llm_coordination/tree/586733a37d240b9592645bb0d7d10f73c1103f7f) 包含 agentic_evals、Hanabi、Overcooked 及权重/配置文件。**没有实际运行**四场游戏、198题或完成任何跨伙伴基准的重测。

## 5. 第17章：OASIS 的百万用户与计算成本

Yang et al. (NeurIPS 2024), [OASIS paper](https://openreview.net/pdf/0a4341351130959d0652b73fe47c62738586df11.pdf)，Appendix D.1/Table 2，含原始实验开销：

| 模拟规模 | A100 GPU | 小时/模拟时间步 | 新推文/步（千条） | 新评论/步（千条） |
| --- | ---: | ---: | ---: | ---: |
| 10,000 用户 | 2 | 0.2 | 0.6 | 0.9 |
| 100,000 用户 | 5 | 3.0 | 5.2 | 9.0 |
| **1,000,000 用户** | **27** | **18.0** | **48.5** | **97.1** |

这是作者报告的设备/实验组合，而不是任何电脑上的预测性能。**百万**是模拟用户/画像数量；系统按活跃度、推荐算法和时钟安排动作，不代表百万个同步或实时 LLM 调用。作者实验发现的信息传播/极化对网络生成、活动调度和推荐器依赖很大，不能当作真实人群的效应大小。

- 论文时期源码近邻：[camel-ai/oasis@fc04233443b46b74f8aefea7b42216aa1d1c3455](https://github.com/camel-ai/oasis/tree/fc04233443b46b74f8aefea7b42216aa1d1c3455)（2024-11-20）；
- 2026 年核查时代码：[camel-ai/oasis@fced8869353f858de4ebe7b8f60321415f238fad](https://github.com/camel-ai/oasis/tree/fced8869353f858de4ebe7b8f60321415f238fad)；已经发生框架/API/使用流程的后续变化。不能将2026年默认行为视为2024年论文实验环境；
- 目前**没有**27块A100的复现实验、原始日志、网络生成器版本/种子或独立多轨迹结果，故计算资源开销只可标为**文献核验**。

## 6. 本轮可信结论与下一道关口

| 工作 | 完成状态 |
| --- | --- |
| arXiv 官方 HTML 对照原书第15–17章相关文句 | 已完成本报告列出的具体条目 |
| ACL 2024 / NAACL 2025 原论文及第一页 PDF 图像核验 | 已完成指定来源 |
| ToMBench 20个数据文件/2860条物理记录扫描与异常定位 | **已实际执行** |
| 本地数据审核脚本 + 隔离夹具自检 | 已添加代码；CI结果另计 |
| 论文表格/DOI与原书书目冲突的编辑勘误 | 已更新中英文文稿 |
| 原书472页PDF页面图像逐页校对 | **未完成**：PDF下载/解析仍因体积及运行容器网络限制失败 |
| ToMBench所有模型得分、CoordQA与OASIS实跑与多种子统计 | **未完成**：需要固定外部环境和运行预算 |

**原则：**复现应当保存**数据哈希、样本ID、完整异常清单、模型精确快照、命令、推理轨迹、逐题输出、真实执行题数、成功/失败数和统计分析代码**。只通过数据计数、文献核验和 CI 夹具绝不能被标记为实验结果已复现。
