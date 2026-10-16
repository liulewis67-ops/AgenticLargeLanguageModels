<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 111-120; Marker split PDF pages 1-10. -->



{0}------------------------------------------------

## 第五章

# 案例研究 1：从零开始构建 LLM

#### 你将学到的内容

完成本案例后，你将能够：

- 说明 ChatGPT 风格 LLM 的端到端*训练流程*，并区分分词、预训练、中训练和监督微调各阶段的目的与机制。
- 配置并训练字节对编码（BPE）*分词器*，解释选择不同词汇表大小时的权衡。
- 从机制层面描述 *Transformer 架构*，包括注意力头、层数和嵌入维度，并分析这些超参数如何影响模型容量和计算预算。
- 使用 ARC-Easy、ARC-Challenge 和 GSM-8K 等标准*基准测试*评估训练好的语言模型，并批判性地分析结果及其失效模式。
- 批判性地思考 LLM 训练在伦理、环境和可及性方面的影响，包括成本如何构成参与门槛，以及数据整理如何体现价值取舍。
- 查找和使用开放访问 LLM 的相关资料，包括模型卡、训练报告、训练代码及其他 *Hub 条目*。
## 5.1 构建聊天机器人

我们将通过一个实际案例，了解现代 ChatGPT 风格语言模型从构建到部署的完整流程。案例采用 *nanochat*<sup>1</sup>：Andrej Karpathy 于 2025 年 4 月发布的轻量级全栈指令微调 LLM 项目。它以单一代码库实现，便于学习和实践。

<sup>1</sup><https://github.com/karpathy/nanochat>

{1}------------------------------------------------

<span id="page-1-0"></span>与 Karpathy 早期仅实现预训练的 *nanoGPT* 不同，nanochat 覆盖了完整训练流程：使用字节对编码（BPE）分词器，在网络文本上进行预训练，再在基准数据集上开展监督微调（SFT），并提供推理服务器和聊天网页界面。

nanochat 主要使用 Python 编写，依赖项很少，因此易于使用和扩展。代码库约有 8,000 行 Python 代码，另含少量 Rust 代码。<sup>2</sup>

nanochat 的训练成本也很有代表性：在配备 8 块 H100 GPU 的节点上运行约 2 小时、花费约 48 美元，即可训练出达到 GPT-2 水平的 LLM。相比之下，2019 年训练 GPT-2 的成本约为 43,000 美元 [\[Radford et al.,](#page-9-0) [2019\]](#page-9-0)。这样的算力对一般大学课程项目并不常见，但可以缩小实验规模，使其适合实践作业。这个案例会展示从原始字节到可运行聊天助手的完整过程，让每次梯度更新和设计决策都清晰可见。接下来将通过四项任务走完 nanochat 流程。

课程的 GPU 预算有限，因此你将训练一个*小型*模型（深度为 4；如果硬件不支持，可降至 2；见第 [5.3](#page-2-0) 节），再结合相关理论和已发布的 nanochat 基准结果，思考模型规模扩大后可能出现的情况。

## 5.2 背景：Karpathy 的 nanochat

本案例将让你动手实践现代 LLM 训练流程的各个阶段，从输入文本一直到部署聊天助手，并把第 2–4 章的理论概念落实到具体、可调试的代码中。

nanochat 默认通过一个复杂度参数来配置计算最优模型：--depth，用于指定 Transformer 层数。嵌入维度、注意力头数、学习率计划、训练时长和权重衰减等其他超参数，都会自动设置为近似最优值。这样既能控制配置的复杂度，也保留了理解各项参数作用的学习价值。

训练流程包含以下几个熟悉的阶段。

**预训练。** 模型使用约 60 亿个来自 CLIMBMix 的词元进行训练 [\[Diao et al.,](#page-8-0) [2026\]](#page-8-0)。CLIMBMix 是通过聚类和迭代自举构建的数据混合语料库。一个深度为 24、约有 14 亿参数的模型，在配备 4 块 H100 GPU 的计算节点上训练约 4 小时，验证损失达到 2.41（每字节 0.717 比特）。

**监督微调（SFT）。** 模型从预训练检查点出发，使用指令遵循与推理数据的混合进行微调：SmolTalk [\[Ben Allal et al.,](#page-8-0) [2024\]](#page-8-0)（46 万条通用对话）、MMLU（多项选择知识题，训练 3 个周期）、GSM-8K（小学数学题，训练 4 个周期）、身份

<sup>2</sup> 若觉得这个规模很大，可以注意到，同类专有系统的代码量可达数百万行。Nanochat 的目标是提供一个小型、独立的实现。

{2}------------------------------------------------

<span id="page-2-0"></span>对话，以及合成拼写任务，总计约 110 万个训练样本。与 InstructGPT 的训练方法相同，交叉熵损失*只针对助手词元计算*，不针对用户提示计算 [\[Ouyang et al.,](#page-9-0) [2022\]](#page-9-0)。

每个训练阶段开始前，系统都会使用 Rust 实现的训练器，在 20 亿个字符上训练 BPE 分词器；不到一分钟即可得到含 32,768 个词元的词汇表。

项目还附带网页推理服务器和轻量级 JavaScript 聊天界面，可立即与训练好的模型交互。

**补充资源。** 两门免费课程可以从不同角度补充本案例的内容。*Hugging Face LLM Course* [\[Hugging Face,](#page-9-0) [2022\]](#page-9-0) 介绍 Transformers、Datasets、Tokenizers 和 Accelerate 等库，以及用于分发和评估许多开放权重模型的 Hugging Face Hub。Mathieu Labonne 的 *LLM Course* [\[Labonne,](#page-9-0) [2023\]](#page-9-0) 面向实践者，涵盖预训练数据整理、参数高效微调（LoRA、QLoRA）、偏好对齐（DPO、PPO、GRPO）、量化（GGUF、GPTQ）和生产环境推理服务（vLLM、TGI）。这些资源有助于把 nanochat 放在更广泛的开放 LLM 生态中理解，并会在任务 3 和任务 4 中明确引用。你也可以查阅其他资料，并在报告中适当注明来源。

## 5.3 实验准备

**代码仓库。** 克隆课程仓库，并在编写代码前仔细阅读 README。

表 5.1：克隆 nanochat 仓库

    git clone git@github.com:karpathy/nanochat.git

**计算资源。** 不需要专用计算资源，可以使用自己的 GPU 笔记本电脑或更强的计算机。实验设计为每次运行最长约 8 小时。预训练使用 --depth 4 配置；只有硬件无法支持较大模型时，才使用 --depth 2。除非拥有充足的计算资源，否则不要尝试深度为 12 或更大的配置。

**日志记录。** nanochat 可按设定的时间间隔在终端记录训练损失、验证损失和每字节比特数，也提供报告和保存运行信息的功能。建议从一开始就启用结构化日志，并写入 .jsonl 文件；运行结束后再补日志会很麻烦。你可以尝试不同的日志格式。正式训练完整模型前，建议试用 wandb.ai 等服务。

{3}------------------------------------------------

## 5.4 任务 0：模型仓库热身

**任务说明。** 开始实验前，先浏览 Hugging Face Hub，了解不同模型常见的参数配置和训练设置。针对表 5.1 中的三个模型 ID，查找并记录：

1. 训练使用的词元数量或训练步数。
2. 模型架构，包括层数、隐藏维度、注意力头数和可训练参数量。
3. 训练数据来源（如能查到）。

| 模型 ID | 所属机构 |
|---|---|
| allenai/OLMo-1B | Allen Institute for AI |
| HuggingFaceTB/SmolLM2-360M | Hugging Face |
| Qwen/Qwen2.5-0.5B | Alibaba Cloud |

表 5.1：模型仓库热身练习所用的模型 ID

注意记录每项信息的出处：有些内容直接列在模型卡中，有些需要在 Files 标签下查看 config.json，还有些只能从配套论文中找到。请在报告中用简短表格整理结果，并为每项信息注明来源。

## 5.5 任务 1：分词

#### 5.5.1 应做的事项

使用 nanochat 的 Rust 分词器训练器，在 CLIMBMix 数据集中抽取 5 亿字符训练 BPE 分词器。分别使用 *8,192* 和 *32,768* 的词汇表大小完成两组实验。

#### 5.5.2 交付物

- 一份能够复现分词器训练过程的简短脚本。
- 一段不超过 1 页的报告，依据分词器输出回答以下问题：
  1. 什么是 BPE？它与词级、字符级分词有什么区别？请从数据中选取一个具体例子，或自行构造例子，画出合并树。不要照搬现成例子，并列出合并次数。

<sup>3</sup><https://huggingface.co/models>

{4}------------------------------------------------

<span id="page-4-0"></span>2. 词汇表大小如何影响固定文本样本的*压缩率*（每个字符对应的词元数）和*序列长度*？请展示两种分词器在英文文本上的测量结果。
3. 词汇表大小为什么会影响下游语言模型？请考虑嵌入矩阵的大小、softmax 分母，以及模型为稀有词元学习可靠表示所面临的困难。
4. 你观察到哪些异常或失败案例？分词器如何处理数字、源代码或非英语文本？

评估时请关注：BPE 解释是否准确、实证材料是否充分、对词汇表权衡的分析是否深入，以及对失败案例的讨论是否有洞察。

## 5.6 任务 2：预训练

#### 5.6.1 应做的事项

使用 nanochat 在 CLIMBMix 数据集上预训练 Transformer，深度设为 --depth 4；如果电脑性能不足，可以使用 --depth 2。定期记录训练损失、验证损失和每字节比特数（bpb）。

#### 5.6.2 交付物

- 训练集和验证集的 bpb 曲线图，并纳入报告。请使用 bpb，而非原始损失。
- 保存的模型检查点。
- 一段约 2 页的报告，内容包括：
  1. **深度参数：** 说明 --depth 控制什么，以及代码如何据此自动设置嵌入维度、注意力头数、学习率计划和训练时长等超参数。针对所选深度，报告层数、注意力头数、嵌入维度、可训练参数总量和训练词元总数。单一参数旋钮为何适用于该代码库？与独立调节各个超参数相比，它有哪些局限？
  2. **规模定律：** 使用 Chinchilla 框架 [\[Hoffmann et al.,](#page-9-0) [2022\]](#page-9-0)，根据模型参数量计算计算最优的训练词元数量，并与 nanochat 默认训练时长相比较。什么是*训练不足*和*训练过度*？
  3. **损失分析：** 绘制训练过程中 bpb 的变化曲线并解释其走势。曲线何时趋于平稳？训练曲线与验证曲线之间的差距说明什么？

{5}------------------------------------------------

- 4. **定性检查：** 不使用系统提示，从预训练模型采样五段续写（即原始的下一个词元预测）。模型在此阶段能做什么？你认为原因是什么？

评估时请关注：深度参数和自动配置说明是否正确、架构描述是否准确、规模定律计算是否恰当、损失分析是否充分，以及定性评估是否有洞察。

## 5.7 任务 3：中训练与监督微调

#### 5.7.1 应做的事项

nanochat 目前只包含两个训练阶段：预训练和 SFT。本书还讨论了增加中训练阶段如何帮助提升模型表现。在本任务中，你需要修改 scripts/chat_sft.py，新增中训练阶段，并通过实验证明其效果。目标是将*推理能力适配*（中训练）与*指令遵循*（SFT）分开。

**阶段 1——中训练。** 修改 scripts/chat_sft.py 中的 train_tasks 列表，只保留 MMLU 和 GSM-8K 两项知识与推理任务，删除其他对话和拼写数据。使用预训练基础检查点运行 chat_sft.py，并保存生成的检查点。

**阶段 2——SFT。** 再次修改 scripts/chat_sft.py，使 train_tasks 中只包含 SmolTalk。还需稍作修改，使脚本读取阶段 1 的检查点，而不是预训练基础模型。可参考 chat_sft.py 中的 load_model 调用。<sup>4</sup> 从阶段 1 检查点运行 chat_sft.py，在中训练之后继续进行 SFT。

每个阶段结束后，使用 scripts/chat_eval.py 在 ARC-Easy、ARC-Challenge 和 GSM-8K 上评估模型。

**自检。** 运行阶段 1 前，先查看数据：打印 MMLU 和 SmolTalk 数据集的前几行，了解其格式与内容。确认你能回答：每个数据集提供多少训练样本？单条训练样本是什么样的？

#### 5.7.2 交付物

- 两个检查点，分别对应中训练和 SFT 阶段。
- 一张表格，列出模型在每个阶段结束后于 ARC-Easy、ARC-Challenge 和 GSM-8K 上的评估分数。
- 一段约 3 页的报告，内容包括：

<sup>4</sup>提示：阅读 nanochat/checkpoint_manager.py，了解检查点的定位方式。

{6}------------------------------------------------

<span id="page-6-0"></span>1. **实现过程：** 说明为支持分阶段训练而对代码所做的修改，并列出实验中使用的超参数。
2. **损失掩码：** 为什么 SFT 损失只针对助手词元计算？请用一个具体的词元级示例，展示使用和不使用掩码时损失计算有何不同。
3. **数据集分析：** 描述各阶段使用的数据集及其大致规模，并说明选择这些数据的原因。数据集划分中可能体现了哪些价值取向或潜在偏差？这些因素可能如何塑造模型行为？
4. **基准分数：** 用表格报告每个阶段后的评估分数。阶段 1 与阶段 2 相比有哪些变化，原因是什么？还要与预训练结果进行比较，使分析涵盖三个阶段。指出任何令你意外的结果。
5. **参数高效微调：** nanochat 使用全参数微调，即 SFT 时更新所有参数。说明 LoRA（低秩适配）及其量化变体 QLoRA 与全参数微调有何不同，并比较它们在计算、内存和模型质量方面的权衡。对于本实验使用的模型深度，你会建议采用 PEFT 吗？

评估时请关注：实现说明是否清晰准确、损失掩码解释是否正确、数据集分析是否深入、基准分数及其解读是否准确完整，以及对 LoRA/QLoRA 的理解是否准确。

## 5.8 任务 4：推理、部署与批判性反思

#### 5.8.1 A 部分：推理与部署

启动 nanochat 的推理服务器和网页界面，并与训练好的模型交互。然后在报告中回答以下问题（约 1 页）：

1. **推理循环：** 说明自回归采样的工作原理，包括键值（KV）缓存如何避免解码各步间的重复计算。尝试不同的解码参数：什么是温度？温度取 0.1、0.7 和 1.5 时，输出质量和多样性有何变化？请展示并讨论模型生成的五个代表性示例。
2. **生产环境加固：** 列出 nanochat 推理服务器在大规模生产部署前需要进行的三项具体工程改进。

{7}------------------------------------------------

<span id="page-7-0"></span>3. **能力比较与规模扩展：** 查找至少两个公开可用模型在 ARC-Easy、ARC-Challenge 和 GSM-8K 上公布的基准分数：一个与 nanochat 模型规模相近，另一个显著更大。可以采用不同推理设置（如零样本或少样本），但要注明具体条件并引用所有来源。将自己的结果与已公布分数列入同一张表。你的模型与相近规模的模型相比表现如何？与同等规模、训练充分的模型之间的差距，说明数据质量、训练时长和对齐有何作用？

#### 5.8.2 B 部分：批判性反思

从以下三个提示中选择一个，写一段约 0.5–1 页的讨论。

*提示 1——可及性与普及化。* nanochat 的宣传语称，“100 美元能买到的最佳 ChatGPT”让个人也能开展 LLM 训练。请批判性地评估这一说法：哪些人仍无法使用这套训练流程？原因是什么？

*提示 2——数据与价值观。* nanochat 使用 CLIMBMix 进行预训练。CLIMBMix 是一种通过聚类迭代自举构建的数据混合语料库。构建该数据集时作出了哪些选择？这些选择体现了谁的价值观？换用不同的预训练语料，模型的特性可能会有何不同？

*提示 3——环境成本。* nanochat 的“低成本”训练使用一台配备 8 块 H100 GPU 的节点，运行约 4 小时。请采用成熟方法（例如 ML CO₂ Impact 计算器）估算自己训练运行的碳排放。<sup>5</sup> 将估算结果放在 LLM 环境影响的大背景下讨论。*成本低*是否就意味着*环境影响小*？

A 部分评估：推理循环描述的准确性、生产环境加固建议的可行性，以及基准比较的完整性和与当前先进水平的相关性。

B 部分评估：分析深度、证据使用情况，以及对既有假设的审视。

## 5.9 推荐阅读

- [Vaswani et al.](#page-9-0) [\[2017\]](#page-9-0) —— Transformer 原始论文；建议在任务 2 前阅读。

<sup>5</sup><https://mlco2.impact.compute.org>

{8}------------------------------------------------

<span id="page-8-0"></span>- [Hoffmann et al.](#page-9-0) [\[2022\]](#page-9-0) —— Chinchilla 规模定律；完成任务 2 的规模定律问题前必读。
- [Ouyang et al.](#page-9-0) [\[2022\]](#page-9-0) —— InstructGPT；是任务 3 中 SFT 和 RLHF 讨论的基础。
- [Diao et al.](#page-9-0) [\[2026\]](#page-9-0) —— CLIMBMix / Nemotron-CLIMB；完成任务 2 的数据集分析前，建议阅读其数据整理方法。
- [Bender et al.](#page-9-0) [\[2021\]](#page-9-0) —— 关于大型语言模型的批判性研究；与任务 4 的反思题相关。
- [Hugging Face](#page-9-0) [\[2022\]](#page-9-0) —— Hugging Face LLM Course；介绍 Transformers、Tokenizers、Datasets 等库及 Hugging Face Hub，可为任务 1、3、4 提供背景。
- [Labonne](#page-9-0) [\[2023\]](#page-9-0) —— Labonne 的 *LLM Course*；实践路线涵盖 LoRA/QLoRA、DPO、GGUF/GPTQ 量化和 vLLM/TGI 服务，建议在任务 3、4 前阅读。

## 5.10 结论

本章以 nanochat 为例，展示了完整的 LLM 训练流程。你训练了 BPE 分词器，观察 Transformer 如何从网络文本中学习，将模型调整为遵循指令，在标准基准上进行评估，并最终将其部署为聊天助手。

你的 depth-4 模型与前沿模型之间会有明显差距，因为你很可能无法获得数据中心级别的计算资源。不过，原则上这些实验可以扩展：增加模型深度和训练数据后重新运行，应该能得到能力更强的模型，这与规模定律的预测一致。完成作业 1 后，你就亲手训练了一个完整的 LLM，也会更有准备开展需要更大规模资源的后续实验。
## 参考文献

Loubna Ben Allal et al. SmolTalk, 2024. URL [https://huggingface.co/](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) [datasets/HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). [112](#page-1-0) Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, FAccT '21, page 610–623, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445922. URL <https://doi.org/10.1145/3442188.3445922>. 119 Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, et al. Nemotron-climb: Clustering-based

{9}------------------------------------------------

<span id="page-9-0"></span>iterative data mixture bootstrapping for language model pre-training. *Advances in Neural Information Processing Systems*, 38, 2026. [112,](#page-1-0) [119](#page-8-0) Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre. Training compute-optimal large language models. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. [115,](#page-4-0) [119](#page-8-0) Hugging Face. The hugging face LLM course, 2022. URL [https://](https://huggingface.co/learn/llm-course) [huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course). [113,](#page-2-0) [119](#page-8-0) Mathieu Labonne. LLM course, 2023. URL [https://github.com/mlabonne/](https://github.com/mlabonne/llm-course) [llm-course](https://github.com/mlabonne/llm-course). [113,](#page-2-0) [119](#page-8-0) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022. [113,](#page-2-0) [119](#page-8-0) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. [112](#page-1-0) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. [118](#page-7-0)