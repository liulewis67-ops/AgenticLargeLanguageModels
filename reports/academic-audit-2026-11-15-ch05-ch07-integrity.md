# 第 5–7 章学术审校：参考文献、实验可复现性与证据边界（2026-11-15）

## 证据来源与限制

本次对照[《Agentic Large Language Models》arXiv:2610.13455v1 官方 HTML](https://arxiv.org/html/2610.13455v1)第 5、6、7 章 References，以及[karpathy/nanochat README](https://github.com/karpathy/nanochat/blob/master/README.md)、[预训练实现](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py)、[SFT 实现](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py)。这些上游代码属于**持续变化的 master 分支**，不能充当固定历史提交的实验依据。

原始 PDF 约 31 MB，本轮在线 PDF 读取因文件体积限制失败，故**未完成 PDF 图像逐页核对**，也未重训原书所提模型或执行世界模型评测。以下结论属于官方 HTML 与仓库 Markdown 的对照及静态检查。

## 1. 书目完整性

| 章节 | 官方 HTML 条目数 | 来源条目首尾 | 发现与修复 |
| --- | ---: | --- | --- |
| 5 | **9** | Ben Allal → Vaswani | 两个超长段落、Diao 等人的论文标题跨原 PDF page-8/page-9；恢复独立条目边界。Bender 与 Diao 之间的裸编号 `119` 保留并标注为需回到 PDF 核查的疑点，不猜测其含义。 |
| 6 | **72** | Ahn → Zhou | 多个 1,500–3,000 字符书目长行；恢复条目，修复 `lan- ¨ guage`、`algo- ¨ rithm` 乱码及同一 URL 的重复拆分。 |
| 7 | **69** | Assran → Zuo | Assran 的 JEPA 论文标题跨页、其余引文集中成巨长段落；恢复条目，清理明显字符损坏。 |

保持页面 `<span id="page-*-0"></span>` 锚点、物理分页符以及文中引用地址，未把跨页正文续行误计为新文献。中英文书目按原始英文元数据保持一致。**条目完整不等于所有出版信息或引文真实性已得到独立第三方验证**。

## 2. 第 5 章 nanochat 的复现性

- 教学任务虽然指定了 `--depth`、训练数据、分词训练词表、评测任务及日志要求，但没有锁定 nanochat commit、`uv.lock`/依赖环境、CLI 参数、数据修订和哈希、划分与随机种子、GPU 和精度等，因此不能声明读者执行命令即可重现书稿的数值。
- 原书称 nanochat **2025 年 4 月**发布；上游 README 则将 **2025 年 10 月 13 日**称为最初的项目介绍。这里将其记录为**原文待核查的时间冲突**，而不是直接重写原文。
- nanochat README 的 `uv sync --extra gpu`、`uv.lock`、`runs/speedrun.sh`、`scripts/base_train.py`、`scripts/chat_sft.py` 随主分支改变。复现时必须首先固定完整 SHA，核对当前该提交下的 CLI。
- 新增[实验运行记录模板](nanochat-reproducibility-run-template.md)：记录环境、数据、每阶段 checkpoint SHA、输入 token、GPU 小时、bpb、ARC/GSM8K 数据切分与提示协议、基线来源、重复次数、置信区间。不填入任何虚构数据。

## 3. 第 6 章推理能力证据

关于 self-consistency 提升 CoT 准确率“10–20 个百分点”的概括，仅适用于相应模型、任务和测试配置，不应作为对所有新模型的普遍承诺。添加中英文**非原文编者注**，要求记录 CoT 基线、采样次数、温度、答案归一化、测试拆分、随机种子及不确定性。未复核原研究的原始统计数据，不能宣称统计显著性已独立确认。

## 4. 第 7 章世界模型证据

WorldGPT 的约 30× 速度、WebDreamer 的约 4–5× 样本效率，以及 WorldCoder 在不同任务上的效率描述来自不同研究及比较基线。三种指标的任务、分母、资源与协议不一致。添加中英文**非原文编者注**，要求分别核验原论文的环境、模型版本、预算、硬件、基线与重复运行。不能把书中这些数字直接组成一个统一排行榜，也不能把引用中的研究结果说成当前仓库已复现。

## 5. 回归与尚未完成的工作

扩展 `scripts/audit-source-fidelity.mjs` 检查 9/72/69 条书目、首末作者、双语一致性、异常合并长行、明显连写及 URL 拆分，并保护上述编者注。静态核查之外，仍需下载原 PDF 实际页面，核对缺失字符、图表及标号 `119`；选择固定 nanochat 提交后实跑 tokenizer、预训练冒烟测试与独立评测；对 WorldGPT/WebDreamer/WorldCoder 的原始实验指标逐项溯源。
