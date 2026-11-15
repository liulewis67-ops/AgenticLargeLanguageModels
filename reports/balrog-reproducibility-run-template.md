# BALROG 学术评测运行记录 / Reproducibility Run Sheet

> 此文件是**待填写的实验模板**，不是已经运行的结果，不得将“待实测”替换成猜测分数。对应本书第 10 章 BALROG 案例，并应与 [BALROG 上游文档](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)、[Hydra 配置](https://github.com/balrog-ai/BALROG/blob/main/balrog/config/config.yaml) 的**固定提交版本**一起使用。

## 一、环境和版本 / Environment and provenance

| 字段 | 填写值 |
| --- | --- |
| 实验 ID、创建时间与时区 | 待记录 |
| BALROG commit：`git rev-parse HEAD`，工作区状态与补丁 | 待记录 |
| Python、pip/conda 依赖锁定、操作系统、容器镜像 | 待记录 |
| 游戏环境/数据版本（各子环境逐项记录） | 待记录 |
| Agent 实现文件 SHA-256、调用 prompt/template | 待记录 |
| LLM/VLM 模型名称、**准确版本**、API 后端、区域 | 待记录 |
| 采样 temperature、top-p、max tokens、最大图像/文本历史长度 | 待记录 |
| 评测命令与 Hydra 完整解析配置（`--cfg job` 若版本支持） | 待记录 |
| 工件、日志、成本数据保存位置与校验和 | 待记录 |

**运行前静态检查：**查看固定提交 `balrog/config/config.yaml`，确认 `agent.max_text_history`（而非原书自定义示例中的 `agent.max_history`）与 `eval.num_workers` 等键能被 Hydra 解析。原始书稿自定义 Agent 清单使用 64 个 worker；修订的**教学冒烟测试**示例使用 4 个 worker。这不是原论文的默认实验条件。特别注意可用模型和 API 价格随时间变化。

## 二、实验设计 / Experimental protocol

| 项目 | 计划配置 | 已执行验证 |
| --- | --- | --- |
| Agent：naive / custom / CoT / reflection，策略与 prompt 差异 | 待记录 | 否 |
| 游戏：BabyAI、Crafter、TextWorld、BabaIsAI、MiniHack、NetHack | 按实验选择 | 否 |
| 每个游戏的 episode、任务、关卡与动作预算 | 待记录 | 否 |
| 随机种子：环境、模型、采样器分别固定 | 待记录 | 否 |
| 模型条件：文本/图像模式、历史长度、context window | 待记录 | 否 |
| 对照组：同环境、同 episode/seed、同模型和预算 | 待记录 | 否 |
| 每种条件的独立重复次数与预设停止规则 | 待记录 | 否 |
| 评测成本预算：LLM token、调用次数、美元与 wall time | 待记录 | 否 |

**必须执行配对比较**（若环境支持同一测试 episode/seed）：固定环境和任务后，仅改变 Agent 实现。不要把模型升级、更多 token、更多动作预算造成的变化归因于自我反思。

## 三、指标与统计 / Metrics and statistics

| 游戏 | 模型快照 | Agent | 运行次数 / episodes | Progression 均值±SD / CI | 成功率及区间 | Token、成本和时延 |
| --- | --- | --- | --- | --- | --- | --- |
| BabyAI | 待记录 | naive | 待实测 | 待实测 | 待实测 | 待实测 |
| BabyAI | 待记录 | custom | 待实测 | 待实测 | 待实测 | 待实测 |
| Crafter / TextWorld / BabaIsAI / MiniHack / NetHack | 待记录 | 按设计 | 待实测 | 待实测 | 待实测 | 待实测 |

- 解释 BALROG 的**progression** 指标定义以及每个环境不同的进度/奖励尺度。不要默认它等于“成功率”，也不要把跨游戏数值直接相减后作为统一效应。
- 统计单位：episode 还是跨任务/跨种子的模型运行；按单位选择独立或配对置信区间。记录组内重复、效应量、失败轨迹和样本量；必要时给出 bootstrap CI 及其采样层级。
- 预设反思停止条件、失败检测规则与 API 异常重试；不要以“连续 5 次相同结果”替代不确定性评估。
- 保留原始逐回合日志、环境 observation/action/reward、奖励转换方式、最终评分脚本版本与输出哈希。

## 四、执行证据和问题 / Evidence and deviations

- 精确启动命令、标准输出/错误、评测配置：`<...>`
- 代码及 agent 参数的变更和原因：`<...>`
- 成本、速率限制、模型升级/下线造成的偏差：`<...>`
- 是否观察到幻觉、策略循环、action parsing 错误、上下文溢出：`<...>`
- 与 [ICLR 2025 BALROG 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html)的可比性：`<协议匹配/不匹配，说明>`
- 状态：**未执行** / 冒烟检查通过 / 完整运行完成 / 原基线复现 / 独立复现。必须附证据才能更新状态。

**安全提醒：**使用付费 API 会产生调用费用，评测指令不应包含密钥。日志发布前应清理 API key、个人信息或可能受限制的内容。
