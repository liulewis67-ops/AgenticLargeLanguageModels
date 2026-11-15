# nanochat 学术实验复现记录模板 / Reproducibility Run Sheet

> **模板，不是实验结果。** 每次训练、分词和评测都填一份独立副本。未经真实运行，不得填入虚构的 loss、bpb、准确率或 GPU 小时数。适用于本书第 5 章的 Task 1–4；用于不同 nanochat 版本时，应重新核实 API 和脚本路径。

## 1. 代码与环境（Code provenance）

- 任务/实验 ID：`<experiment-id>`
- 运行日期（含时区）：`<ISO-8601>`
- nanochat URL：`https://github.com/karpathy/nanochat`
- 上游固定提交：`git rev-parse HEAD` 的完整 SHA：`<40-char-SHA>`
- 工作区状态（记录 `git status --porcelain`；若不干净须保留补丁）：`<clean/dirty + patch>`
- 安装命令及锁文件：`uv sync --extra gpu` / `uv sync --extra cpu`，`uv.lock` 对应提交及 SHA-256：`<...>`
- Python / PyTorch / CUDA / driver / GPU model / GPU count / VRAM：`<...>`
- 精度与并行策略（例如 BF16 / FP32、DDP、梯度累积）：`<...>`
- 完整训练/验证/推理命令（包括 CLI flags 与环境变量）：`<...>`
- 随机种子、确定性开关、重复次数：`<...>`

**注意：**上述安装命令来自 nanochat 官方 README，是否适用必须以**本次固定提交**为准。不要把主分支最新 CLI 视作 2026 年图书原始实验环境。

## 2. 数据与分词（Dataset & tokenizer）

| 项目 | 本次填写 |
| --- | --- |
| CLIMBMix 或替代语料版本、出处、许可 | 待记录 |
| 下载地址、修订版、数据文件哈希 | 待记录 |
| 清洗、去重、过滤和抽样规则 | 待记录 |
| 训练/验证/测试划分、污染排查 | 待记录 |
| BPE 输入字符数、词表规模、训练工具版本 | 待记录 |
| tokenizer 文件、哈希、特殊词元配置 | 待记录 |
| MMLU/GSM8K/SmolTalk 来源与各阶段样本量 | 待记录 |

## 3. 训练阶段与资源（Training & budget）

| 阶段 | 起始 checkpoint SHA | 数据/训练样本数 | 深度/参数量 | 步数与 token | GPU 小时 | 输出 checkpoint SHA |
| --- | --- | --- | --- | --- | --- | --- |
| Tokenizer | 不适用 | 待记录 | 词表待记录 | 待记录 | 待记录 | tokenizer 待记录 |
| Pretraining | 随机初始化 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| Mid-training | Pretraining SHA | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| SFT | Mid-training SHA | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |

- 优化器、调度器、batch size、sequence length、gradient accumulation：`<...>`
- 每阶段日志与绘图文件：`<path / SHA>`
- 训练曲线：training/validation bpb，验证窗口及评测步数：`<...>`
- 资源与价格：GPU 单价、租赁时长、总卡时、存储和传输成本；如估算碳排放，记录 PUE、功耗与电网因子：`<...>`

## 4. 评测协议（Evaluation）

| 模型阶段 | ARC-Easy | ARC-Challenge | GSM8K | 指标定义、置信区间、样本数 |
| --- | --- | --- | --- | --- |
| Pretrained | 待实测 | 待实测 | 待实测 | 待记录 |
| Mid-trained | 待实测 | 待实测 | 待实测 | 待记录 |
| SFT | 待实测 | 待实测 | 待实测 | 待记录 |

- benchmark 原始数据版本、split、去重/污染、评分脚本、答案提取规则：`<...>`
- prompt template、few-shot 示例/顺序、temperature、top-p、max tokens、采样次数：`<...>`
- 论文/模型卡基线：`<标题 / URL / 评测协议 / 能否横向比较>`
- 统计分析：`<seed-level results / interval / significance / paired-unpaired>`
- 已验证结果与仅引用的论文声明必须分栏呈现，不可把引用值当作本次实测结果。

## 5. 异常、偏差与复核状态（Audit trail）

- 预期与实际差异：`<...>`
- OOM、运行中断、恢复点、是否重试：`<...>`
- 不确定性：`<...>`
- 重现命令、日志、模型/分词器工件存放链接：`<...>`
- 核验状态：未运行 / 运行中 / 已运行但无法复现 / 本地重复成功 / 独立环境复现成功（附证据）。

参考：[本书第 5 章](../docs/05-case-study-1-building-llms-from-scratch/05-case-study-1-building-llms-from-scratch-zh.md)；[nanochat 官方 README](https://github.com/karpathy/nanochat/blob/master/README.md)。
