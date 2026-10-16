<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 63-82; Marker split PDF pages 1-20. -->



{0}------------------------------------------------

## 第三章

# 大语言模型训练流程

#### 你将学到的内容

本章将介绍语言模型的训练方法。学完后，你将能够：

- 描述语言模型训练流程的主要阶段，并解释各阶段的作用：*数据获取、预训练、微调、偏好调优和推理*。
- 区分*预训练与微调*，并说明为什么偏好调优在这两者之后进行。
- 解释*编码器/解码器*架构如何影响模型采用的训练目标。
- 确定用于评估语言模型的常用*基准测试*，并说明每个基准测试旨在衡量的能力。

## 3.1 训练模型

大语言模型通常按一系列阶段进行训练。首先，模型架构的参数随机初始化；随后通过*预训练*，在大规模文本语料上学习语言能力，训练目标通常是自回归预测。预训练得到的基础模型通常还不能直接满足具体应用需求，因此还需针对下游任务进行适配。这个阶段称为*后训练（post-training）*：使用特定类型的附加数据，引导模型表现出符合任务需求的行为，例如遵循指令或完成分类。

{1}------------------------------------------------

#### <span id="page-1-0"></span>3.1.1 编码器与解码器模型

如第二章所述，常见的模型架构有两类：编码器和解码器。编码器能够根据输入构建丰富的上下文化表示；不过，近期许多表现突出的模型采用了解码器架构。编码器适合分类、情感分析等理解任务，解码器则更擅长生成任务。解码器模型的参数规模和训练数据量不断增长，使其能够较好地泛化到未见过的上下文。

下面具体介绍仅解码器模型训练流程中的主要阶段。

#### 3.1.2 预训练/微调范式

现代训练流程成功的关键之一是迁移学习。常见做法是先在大型通用数据集上训练模型，再将其适配到新任务：模型先在大型通用语料上进行*预训练*，再用规模较小的专用数据集进行*微调*。由于任务数据集较小，而且微调从预训练权重继续开始，因此训练速度更快，也不必从头训练。

预训练与微调的区分是大语言模型训练流程的核心。模型可以在新场景中复用预训练阶段学到的语言知识，从而加快适配过程。大型商业模型的预训练可能需要数月。下游任务通常较为具体，例如特定领域的问答或情感分析，因此会使用专门的数据集进行微调。微调时会更新模型的部分权重，使模型学会新任务，同时尽量保留其通用语言能力。将模型适配到新任务通常需要几分钟到几小时。

#### 3.1.3 开放与闭源模型

流行的 LLM 包括 OpenAI 的 ChatGPT 系列 [\[Achiam et al.,](#page-13-0) [2023,](#page-13-0) [Ouyang et al.,](#page-16-0) [2022\]](#page-16-0)，Meta 的 LLaMa 家族 [\[Touvron et al.,](#page-18-0) [2023\]](#page-18-0)，Google 的 Gemma [\[Chowdhery](#page-14-0) [et al.,](#page-14-0) [2023\]](#page-14-0) 和 Gemini [\[Anil et al.,](#page-13-0) [2023\]](#page-13-0)，Anthropic 的 Claude 家族，Moonshot AI 的 Kimi，DeepSeek [\[Guo et al.,](#page-15-0) [2025\]](#page-15-0)，以及 Qwen [\[Bai et al.,](#page-13-0) [2025\]](#page-13-0) 等。

这些模型在公开程度上各不相同：开发者可能公开训练数据、披露模型架构，或发布训练好的权重。不过，商业模型的开发者很少公开模型及其训练过程的全部信息，部分原因是希望保持竞争优势。

作为回应，研究界开发了 BLOOM [\[Le Scao et al.,](#page-16-0) [2023\]](#page-16-0)、Pythia [\[Biderman et al.,](#page-14-0) [2023\]](#page-14-0) 和 OLMo [\[Groeneveld et al.,](#page-15-0) [2024\]](#page-15-0) 等完全开源模型，以推动对大语言模型的集体理解。这些模型公开了完整的训练流程，因此本章将重点讨论它们；我们也会介绍部分商业模型的特点。

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Diagram_2.jpeg) 
  
 图 3.1：LLM 训练流程概览 [\[Radford et al.,](#page-16-0) [2019,](#page-16-0) [Minaee et al.,](#page-16-0) [2024\]](#page-16-0)。
  
虽然这些模型都基于 Transformer 架构，但各个组件和整体设计仍有差异，部分差异源于硬件条件或开发者采用的特定优化。例如，Google 针对其 TPU 计算基础设施优化了 Gemini [\[Jouppi et al.,](#page-15-0) [2023\]](#page-15-0)。第四章将进一步讨论架构改进。

## 3.2 训练流程

下面进一步介绍标准 LLM 训练流程的各个阶段 [\[Radford et al.,](#page-16-0) [2019,](#page-16-0) [Minaee et al.,](#page-16-0) [2024\]](#page-16-0)。各阶段的组合顺序通常称为训练*配方*（recipe）。不同阶段可能使用不同的数据集和损失函数，并分别使模型获得特定能力。图 3.1 展示了一个示例。

#### 3.2.1 获取数据

训练流程的第一步是获取数据。预训练通常使用大规模通用文本语料库 [\[Brown et al.,](#page-14-0) [2020\]](#page-14-0)。这类语料包含无需人工标注的原始文本，通常汇集自经过筛选和未经筛选的在线来源，再通过清洗规则降低有害内容、隐私信息或受版权保护材料带来的风险。预训练语料规模庞大：现代 LLM 的训练数据往往达到数万亿个词元。常见数据集包括：

- Common Crawl：大型非营利性网页抓取数据集，包含数十亿个网页；Internet Archive（<https://archive.org>）也是一种数据来源。尽管规模庞大，Common Crawl 数据的质量仍有争议 [\[Baack,](#page-13-0) [2024\]](#page-13-0)。C4（Colossal Clean Crawled Corpus）等衍生数据集提供经过筛选的英文网页文本子集 [\[Raffel et al.,](#page-17-0) [2020\]](#page-17-0)，以提升数据质量。

{3}------------------------------------------------

- <span id="page-3-0"></span>• Dolma 3：一种面向多种任务表现而设计的数据混合方案，用于 OLMo 开源模型与开放数据项目 [\[Ettinger et al.,](#page-15-0) [2025,](#page-15-0) [Soldaini et al.,](#page-17-0) [2024\]](#page-17-0)。
- SmolLM3：采用三阶段训练策略，使用 11.2 万亿个词元，动态混合网页、数学和代码数据，以提升推理和知识获取能力 [\[Bakouch et al.,](#page-13-0) [2025\]](#page-13-0)。

近期研究指出，数据多样性对于跨领域知识和下游泛化能力的提升，可能与数据规模同样重要[\[Gao et al.,](#page-15-0) [2020\]](#page-15-0)。*The Pile* 是一个专门构建的异质语料库。在 GitHub 代码库、医学论文和哲学文本等不同领域中，使用 The Pile 训练的模型取得了更好的每字节比特（Bits Per Byte，BPB）<sup>1</sup>分数（见图[3.2\)](#page-4-0)）。因此，The Pile 的 BPB 常被用作基准，也可作为衡量模型跨领域知识和复杂推理能力的指标。

预训练数据的整理需要构建复杂的清洗流程，把多个来源的原始材料转化为干净的纯文本集合。这一过程通常兼顾质量与安全。整理工作首先会确定数据来源的优先级，并提高维基百科、书籍等高质量来源的权重。随后，研究者使用降噪规则去除模板文本、导航菜单和可能不适宜的内容。数据还会在 URL 和文档两个层面去重，以减少重复材料，避免模型记住重复内容并因此过拟合。

除准确性外，流程还必须确保所选数据符合安全与伦理要求，通常通过启发式规则进行筛选。毒性检测系统会尝试移除仇恨言论，但效果并不总是稳定。偏见缓解也是一项挑战：自动过滤器可能误标并过滤用特定方言（如非裔美国英语）写成的正当内容，从而使训练语料中的多元声音被意外排除。

#### 3.2.2 预训练

第二阶段是在整理后的语料上预训练模型，以获得通用语言模型。此阶段通常采用自监督的 Transformer 方法 [\[Vaswani et al.,](#page-18-0) [2017\]](#page-18-0)。自监督学习从文本本身构造训练目标，无需人工标注样本。编码器模型可以根据周围上下文预测被掩码的词元；仅解码器语言模型通常根据前文预测下一个词元。公式 3.1 展示的是后一类模型使用的自回归目标。这些训练目标有助于模型学习语言表示、语义关系和世界知识。

形式上，自回归语言建模目标是最大化序列 $\mathcal{U} = \{u_0, \ldots, u_n\}$ 中各词元的对数概率；每个词元的概率以其前面 $k$ 个词元为上下文条件（公式 3.1）。这里，$\mathcal{L}(\mathcal{U})$ 表示对数似然目标函数。

<sup>1</sup> Bits per byte（BPB，每字节比特数）是一种与分词器无关的信息论指标，用于衡量编码每个字节所需的平均比特数，反映编码效率。另见第 [3.3.2 节](#page-10-0)。

{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Diagram_2.jpeg)

图 3.2：The Pile 数据集中数据源的概览 [\[Gao et al.,](#page-15-0) [2020\]](#page-15-0)。

$\mathcal{L}(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1}; \Theta)$ (3.1)

语言模型通过可训练参数 $\Theta$ 估计这些概率，并使用基于梯度的更新方法进行优化。模型规模从数百万到超过一万亿个参数不等，因此训练通常需要大型数据集。预训练时会使用*教师强制*：模型作出预测后，下一步输入仍是真实词元，而不是模型自己的预测。推理时进行自回归生成，则会把模型的预测作为后续步骤的输入。正如[第二章 2.1.4 节](../02-foundations-of-language-modeling/02-foundations-of-language-modeling-zh.md#214-分词)所述，文本通过相应的*分词器*（tokenizer）转换为词元。

预训练的一项关键挑战，是让模型高效学习，并在数万亿次梯度更新中保持稳定。研究者会采用多种策略来控制训练过程，其中一种是课程学习。人们通常先学加法等简单概念，再学乘法、对数等较难的概念。语言模型训练也可采用类似思路：不同时混合所有类型的数据，而是按阶段选择数据来源。图 [3.3](#page-5-0) 展示了一个例子。课程会按照难度安排数据：先使用通用网页数据，再逐渐转向更难、更高质量的推理数据，例如数学和代码。这样做有望帮助模型更快学会复杂推理 [\[Bengio et al.,](#page-14-0) [2009\]](#page-14-0)。

{5}------------------------------------------------

<span id="page-5-0"></span>![](assets/_page_5_Diagram_2.jpeg) 
  
 图 3.3：训练 SmolLM3 的示例数据混合 [\[Bakouch et al.,](#page-13-0) [2025\]](#page-13-0)。
  
 ![](assets/_page_5_Figure_4.jpeg) 
  
 ![](assets/_page_5_Diagram_5.jpeg) 
  
 图 3.4：使用原始架构（a）和重排架构（b）训练 BERT 模型的损失 [\[Shoeybi et al.,](#page-17-0) [2020\]](#page-17-0)。
  
训练规模如此之大时，过程容易不稳定并出现损失突增，因此还需要其他稳定方法 [\[Takase et al.,](#page-17-0) [2025\]](#page-17-0)。权重衰减、层归一化和学习率调度等常用技术有助于稳定训练；专用优化器和梯度范数监控也有助于防止参数发散。图 3.4 展示了重新安排嵌入层如何改善训练稳定性。图中比较了参数量为 336M 和 752M 的 BERT 模型训练损失：原始架构在 336M 模型上表现良好，而 (b) 中的修改使 752M 模型也能稳定训练，并取得较低的损失 [\[Shoeybi et al.,](#page-17-0) [2020\]](#page-17-0)。

{6}------------------------------------------------

#### <span id="page-6-0"></span>3.2.3 微调

预训练之后，模型会进入后训练阶段，首先进行微调。实际应用中，人们通常用 LLM 回答问题、提供建议和检索信息，而不只是通过预测下一个词来续写文本。微调是在较小的任务专用数据集上继续训练预训练模型。在监督微调中，使用带标注的样例将通用模型适配到特定任务。一种参数高效的方法是低秩适配（Low-Rank Adaptation，LoRA）：冻结预训练模型权重，并添加可训练的低秩矩阵，从而减少需要训练的参数量 [\[Hu et al.,](#page-15-0) [2021,](#page-15-0) [Wei et al.,](#page-18-0) [2022,](#page-18-0) [Minaee et al.,](#page-16-0) [2024\]](#page-16-0)。

**监督微调（SFT）。** 微调可用于许多专门任务。通用模型会通过 SFT 学习遵循指令和执行推理步骤 [\[Ettinger et al.,](#page-15-0) [2025\]](#page-15-0)。SFT 对较小模型尤其有价值，能够以较高效率提升其通用对话、推理和函数调用能力。

SFT 也用于更专门的任务，例如减少有害输出。WildChat [\[Zhao et al.,](#page-18-0) [2024\]](#page-18-0) 收录了超过一百万条真实用户与 ChatGPT 的互动，展示了不同复杂度和有害程度的提示。

为了训练推理能力，OpenThoughts [\[Guha et al.,](#page-15-0) [2025\]](#page-15-0) 收录了从先进模型中蒸馏出的逐步推理过程，以提升模型解决数学、科学和编程问题的能力。人工收集的数据还可由合成样例补充：使用 GPT-5 等更强的教师模型生成指令和回答，填补特定数据缺口并扩大训练集 [\[Zhang et al.,](#page-18-0) [2026\]](#page-18-0)。

微调的用途不止于指令遵循和推理，也包括语言间翻译，以及医疗、客户服务等领域的问答。对于任务范围较窄、模型规模较小的工业应用，微调尤其有用。
  
[第 11 章](../11-how-agents-use-tools-to-act-in-the-world/11-how-agents-use-tools-to-act-in-the-world-zh.md)将讨论 LLM 如何使用工具，这是智能体系统的一项核心能力。模型也可以在监督微调阶段学习使用工具 [\[Wei et al.,](#page-18-0) [2022,](#page-18-0) [Minaee et al.,](#page-16-0) [2024\]](#page-16-0)。

微调还可塑造模型的角色风格并加深其专业知识。例如，组织可以训练模型采用特定语气，使客服机器人保持一致的品牌风格，既可以亲切正式，也可以机智随和。注入领域知识还能帮助模型掌握专业领域的术语和逻辑，例如特定行业的专利信息或专有遗留代码。

微调可以将通用模型适配为特定任务的专用模型，也能使更小、更高效的模型在某些任务上胜过更大的基础模型。

{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Diagram_2.jpeg) 
  
 图 3.5：RLHF 流程概览：训练奖励模型（左），再根据该模型给出的奖励优化语言模型（策略）[Dong et al.](#page-14-0) [\[2024\]](#page-14-0)。
 
#### 3.2.4 偏好调优

预训练模型可能复现甚至放大训练数据中不理想的模式，包括极端或有害内容。偏好调优是在后训练中加入的一个阶段，通过偏好反馈引导模型生成符合预期的回答。

这一阶段使用偏好数据，使微调后的模型更符合用户预期。其目标是鼓励模型给出有帮助且安全的回答，减少有害输出和幻觉，并提升指令遵循能力；这些方法并不能彻底消除相关问题。

常见方法包括基于人类反馈的强化学习（RLHF）[\[Ouyang et al.,](#page-16-0) [2022\]](#page-16-0)：先根据人类偏好训练奖励模型，再据此优化语言模型；以及直接偏好优化（DPO）[\[Rafailov et al.,](#page-16-0) [2024\]](#page-16-0)：直接利用偏好数据训练模型，无需单独训练奖励模型并进行强化学习。图 3.5 展示了 RLHF 流程。来自 AI 反馈的强化学习（RLAIF）[\[Bai](#page-13-0) [et al.,](#page-13-0) [2022\]](#page-13-0) 则使用 AI 模型生成偏好判断，减少对人工标注的依赖。

对齐并非易事。本章区分偏好对齐、价值对齐和安全对齐。偏好可能带有主观性，例如人们对写作风格的喜好各不相同。

**价值对齐。** 价值对齐关注公平、助人、诚实等高层原则，而非个人偏好。一种做法是用一套书面原则（即*宪法*）引导模型，明确开发者希望模型遵循的政策。

优化安全性和助人能力时，可能需要在事实回忆、创造力等其他能力上作出权衡 [\[Lin et al.,](#page-16-0) [2024\]](#page-16-0)。这种潜在代价称为*对齐税*。

{8}------------------------------------------------

<span id="page-8-0"></span>![](assets/_page_8_Diagram_2.jpeg) 
  
 图 3.6：RLHF（具体来说，PPO）与 RLVR（具体来说，GRPO）方法的比较 [\[Shao et al.,](#page-17-0) [2024\]](#page-17-0) 

**安全对齐。** 安全对齐为模型行为设定边界，以防止有害输出。偏好对齐可能侧重语气，价值对齐侧重指导原则；安全对齐还关注模型抵御对抗性提示和恰当拒绝不安全请求的能力。常见方法包括红队测试（通过越狱尝试查找模型漏洞）和护栏机制（拦截或引导高风险回答）。

**目标对齐。** 对于数学、编程等答案可验证的任务，基于可验证奖励的强化学习（RLVR）[\[Lambert et al.,](#page-16-0) [2024\]](#page-16-0) 会利用结果校验或其他真实信号奖励正确答案。例如，代码通过测试或数学答案正确，都可作为奖励依据。群相对策略优化（GRPO）[\[Guo et al.,](#page-15-0) [2025\]](#page-15-0) 是一种用于数学推理的 PPO 类方法。它会针对同一提示采样多个回答，并比较验证器给出的奖励；与 PPO 不同，GRPO 无需单独的价值模型，因此可减少内存占用。在一些设置中，确定性验证器会提供二元奖励，例如答对记为 +1、答错记为 −1 [\[Guo et al.,](#page-15-0) [2025,](#page-15-0) [Shao et al.,](#page-17-0) [2024\]](#page-17-0)。图 3.6 对比了基于 PPO 的 RLHF 与 GRPO。

#### 3.2.5 推理

模型完成训练后进入推理阶段，用户在此时与模型交互并获取输出。

用户通常通过自然语言提示（指令或问题）引导模型生成回答。提示的措辞和词序都可能影响模型的响应。

{9}------------------------------------------------

<span id="page-9-0"></span>因此，提示工程和上下文工程成为 LLM 用户需要掌握的重要技能 [\[Sahoo](#page-17-0) [et al.,](#page-17-0) [2024\]](#page-17-0)。

正如上一章所述，大型仅解码器模型能够进行*上下文学习*（ICL）[\[Brown et al.,](#page-14-0) [2020,](#page-14-0) [Wei et al.,](#page-18-0) [2022\]](#page-18-0)。ICL 提示会包含一条指令和少量示例，模型借助这些示例推断任务并生成回答。这是一种少样本学习形式，过程中不会更新模型参数 [\[Dong et al.,](#page-14-0) [2022\]](#page-14-0)。

本书后续章节会介绍上下文学习在智能体系统中的多种用法。推理时，提示和示例也是引导智能体 LLM 行为的一种方式。

#### 3.2.6 总结

本节介绍了一种常见的 LLM 训练流程示例。不同模型采用的训练配方各不相同；现代训练流程复杂且成本高昂，改进训练流程仍是当前研究的重点，例如 [Guo et al.](#page-15-0) [\[2025\]](#page-15-0) 和 [Du et al.](#page-14-0) [\[2025\]](#page-14-0) 的研究。

快速推理很重要，因此研究者开发了多种加速生成的方法。其中一项权衡是推理时投入的思考计算与回答质量之间的关系，这也是*测试时间缩放*研究的主题。增加推理计算会耗费更多时间和资源，但可能改善回答。[第四章](../04-scaling-laws-and-efficiency/04-scaling-laws-and-efficiency-zh.md)将讨论这类方法及其他效率优化技术。

### 3.3 基准测试与性能

训练完成后，可以用基准测试评估模型在不同任务和数据集上的表现。GLUE 和 SuperGLUE 等基准用于评估语言理解能力 [\[Wang et al.,](#page-18-0) [2018,](#page-18-0) [2019\]](#page-18-0)；SQuAD 等数据集用于阅读理解和问答 [\[Rajpurkar et al.,](#page-17-0) [2016\]](#page-17-0)；XSum 等数据集用于摘要生成 [\[Narayan et al.,](#page-16-0) [2018\]](#page-16-0)。其他评测任务还包括机器翻译 [\[Kocmi et al.,](#page-15-0) [2022,](#page-15-0) [Papineni et al.,](#page-16-0) [2002,](#page-16-0) [Sennrich et al.,](#page-17-0) [2015\]](#page-17-0) 和诗歌生成 [\[Zhang and Eger,](#page-18-0) [2024\]](#page-18-0)。这些任务展现了研究者评估的能力范围；与此同时，领域快速发展也让人们对 LLM 在不同领域和情境中的能力提出了许多问题。

接下来介绍常用的 LLM 评测基准和指标。

{10}------------------------------------------------

#### <span id="page-10-0"></span>3.3.1 常见任务

LLM 任务大体可归入前文介绍的两类语言处理任务：自然语言理解（NLU）和自然语言生成（NLG）。具体采用哪种指标，取决于任务类别、任务定义和数据集；下面分别讨论。

*性能*一词可能指任务表现（如准确率），也可能指计算性能（如速度）。评测通常关注任务表现，但推理成本和延迟也可能影响实际应用，这一点会在后续章节讨论。

#### 3.3.2 指标

用于监控或报告 LLM 任务表现的指标，取决于评测目标和模型所处的开发阶段。
 
**预训练指标。** 预训练阶段的一个核心问题是模型能否准确预测下一个词元。标准指标之一是*困惑度（perplexity）*，见公式 3.2。困惑度衡量模型进行下一个词元预测时的不确定性，并与交叉熵损失直接相关。困惑度越低，表示模型给实际观测到的词元分配的概率越高；因此困惑度越低越好。

$\operatorname{PPL}(X) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)$（3.2） 
  
 $\operatorname{BPB} = -\frac{1}{B}\sum_{i=1}^{N} \log_2 P(x_i \mid x_{<i})$（3.3） 
  
困惑度按词元计算。若要将其转换为词级指标，需要谨慎归一化 [\[Pimentel and Meister,](#page-16-0) [2024\]](#page-16-0)。即使经过调整，困惑度仍依赖分词器，因为不同的分词器会产生不同的子词序列和概率分布。

另一种指标是*每字节比特数*（BPB），见公式 3.3。它将损失归一化到字节级别，因此可用于比较采用不同分词器的模型 [\[Borgeaud et al.,](#page-14-0) [2022\]](#page-14-0)。该指标衡量编码一个字节平均需要多少比特。

**后训练指标。** 对于问题求解等答案可客观评分的任务，通常会依据标准答案计算成功率。若每道题只有一个正确答案（如选择题），常用指标是*准确率*，即答对题目的百分比。

{11}------------------------------------------------

<span id="page-11-0"></span>准确率越高越好。在每题只生成一个答案的标准设置中，模型每道题作答一次。若允许生成多个答案，*Pass@K* 衡量每道题在 $K$ 次尝试中至少有一次答对的比例；值越高越好。$K$ 的常见取值包括 1、5、10 和 100。


#### 3.3.3 常用基准

LLM 性能评估仍是活跃的研究领域，但许多数据集已成为衡量模型通用能力的常用基准。这些数据集仅用于*评估*模型质量，并非额外的训练数据，因此可视为半标准化基准。MLCommons 等项目正在推动建立领域标准 [\[Thiyagalingam et al.,](#page-18-0) [2022\]](#page-18-0)，这也体现出业界对客观、可靠评估方法日益增长的需求。

**数据污染。** 现代基准测试面临数据污染风险 [\[Balloccu et al.,](#page-13-0) [2024\]](#page-13-0)。随着预训练数据集不断扩大，有时甚至覆盖互联网上相当大比例的内容，域内数据与域外数据的界限也越来越难以界定。商业开发者往往不公开训练数据来源，这进一步加剧了问题。要衡量模型对未见数据的泛化能力，就需要确认模型此前没有接触过评测样本，也就是尽量避免*数据污染*。

基准测试通常会公开一套数据供开发者快速试验，同时保留另一套访问受限的数据来评估商业系统，以降低数据污染风险。为使基准长期保持实用，维护者也可能定期更新评测样本。

表 [3.1](#page-12-0) 列出了一些本书撰写时较为常用的通用基准，并非完整清单。

## 3.4 结论性意见

在 LLM 开发领域，Anthropic、OpenAI、Google、DeepSeek 和 Mistral 等商业开发者通常会发布技术报告，介绍模型在常见基准上的表现。这类报告往往以预印本形式发布，未经学术同行评审；出于商业原因，报告也可能省略实现科学复现所需的细节，例如数据构成和模型架构。即便如此，这些报告仍能帮助我们了解 LLM 的能力与局限。

现代 LLM 在不同基准上呈现出明显的性能层次。它们已能较好地完成广泛的知识类任务，但在复杂推理和自主智能体任务上仍面临困难。
{12}------------------------------------------------

<span id="page-12-0"></span>

| 基准 | 评估能力 | 任务形式 | 指标 |
|---|---|---|---|
| Humanity’s Last Exam（HLE） | 通用智能；多模态推理 | 简答；多项选择 | Pass@1 |
| Abstract Reasoning Challenge（ARC） | 通用智能 | 视觉网格模式补全 | Pass@1（私有测试集） |
| GPQA Diamond | 专家级 STEM 推理 | 多项选择 | Pass@1 |
| AIME 2025 | 高阶数学推理 | 整数作答（0–999） | Pass@1 |
| SWE-Bench | 智能体编程；真实世界软件工程 | 生成代码补丁 | Pass@1 |
| Fortress | AI 安全；国家安全风险评估 | 形式不一（专有） | 平均风险评分（ARS）与过度拒答评分（ORS） |
| MMLU | 广泛的通用知识 | 多项选择 | Pass@1 |
| GSM8K | 小学数学推理 | 多步算术 | Pass@1 |
| TruthfulQA | 事实正确性；幻觉检测 | 问答 | Pass@1 |

*表 3.1：常见 LLM 基准及其报告指标概览。*

在 MMLU、TruthfulQA 等静态知识和专业考试基准上，截至本书撰写时，GPT-5.2、Gemini 3 Pro 和 Claude 4.5 Opus 等前沿模型已接近饱和水平，得分持续高于 85–90%，超过人类专家基线。

新一代推理模型（如 OpenAI 的 o1、DeepSeek-R1 和 Qwen-3.5）近期进步显著，数学和科学任务的通过率已从较低水平提升至专家水平。尽管如此，SWE-Bench 仍是检验软件工程可靠性的严峻测试：顶尖代码智能体在验证子集上的得分已从不足 2% 提升至 40–50%，但要自主修复真实代码仓库中的问题，仍远比回答编程题困难。

**开放问题。** LLM 能力从何而来，仍有许多未解之处。随着训练和推理流程日趋复杂，我们越来越难得出确定结论。训练阶段仍有一些问题悬而未决：*哪些数据最重要？*

{13}------------------------------------------------

<span id="page-13-0"></span>*哪些训练步骤最关键？* 推理阶段也有问题待解：*如何诱发高质量的推理轨迹？* 例如，预训练和后训练哪个更重要，知识是否主要来自预训练、而后训练主要负责提供有效的任务接口，目前都没有定论 [\[Zhou et al.,](#page-18-0) [2023\]](#page-18-0)。
  
 本书的其余部分将讨论后训练和推理时的推理方法以及代理训练方法。但首先，我们将在下一章中研究规模律。

## 参考文献

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. [64](#page-1-0) Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023. [64](#page-1-0) Stefan Baack. A critical analysis of the largest source for generative ai training data: Common crawl. In *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency*, pages 2199–2208, 2024. [65](#page-2-0) Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [64](#page-1-0) Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022. [70](#page-7-0) Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Patino, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, ˜ Quentin Gallouedec, Kashif Rasul, Nathan Habib, Cl ´ ementine Fourrier, Hynek ´ Kydlicek, Guilherme Penedo, Hugo Larcher, Mathieu Morlon, Vaibhav Srivastav, Joshua Lochner, Xuan-Son Nguyen, Colin Raffel, Leandro von Werra, and Thomas Wolf. SmolLM3: smol, multilingual, long-context reasoner. [https:](https://huggingface.co/blog/smollm3) [//huggingface.co/blog/smollm3](https://huggingface.co/blog/smollm3), 2025. [66,](#page-3-0) [68](#page-5-0) Simone Balloccu, Patr´ıcia Schmidtova, Mateusz Lango, and Ondrej Dusek. Leak, ´ cheat, repeat: Data contamination and evaluation malpractices in closed-source LLMs. In Yvette Graham and Matthew Purver, editors, *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 67–93, St. Julian's, Malta, March 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.eacl-long.5. URL <https://aclanthology.org/2024.eacl-long.5/>. [74](#page-11-0)

{14}------------------------------------------------

<span id="page-14-0"></span>Yoshua Bengio, Jer´ ome Louradour, Ronan Collobert, and Jason Weston. Curriculum ˆ learning. In *Proceedings of the 26th annual international conference on machine learning*, pages 41–48, 2009. [68](#page-5-0) Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In *International Conference on Machine Learning*, pages 2397–2430. PMLR, 2023. [64](#page-1-0) Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 2206–2240. PMLR, 17–23 Jul 2022. URL [https://proceedings.mlr.press/v162/borgeaud22a.](https://proceedings.mlr.press/v162/borgeaud22a.html) [html](https://proceedings.mlr.press/v162/borgeaud22a.html). [73](#page-10-0) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. [65,](#page-2-0) [72](#page-9-0) Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240):1–113, 2023. [64](#page-1-0) Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. RLHF workflow: From reward modeling to online RLHF. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=a13aYUU9eU) [a13aYUU9eU](https://openreview.net/forum?id=a13aYUU9eU). [70](#page-7-0) Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey on in-context learning. *arXiv preprint arXiv:2301.00234*, 2022. [72](#page-9-0) Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1.5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*, 2025. [72](#page-9-0)

{15}------------------------------------------------

<span id="page-15-0"></span>Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, et al. Olmo 3, 2025. URL [https://arxiv.](https://arxiv.org/abs/2512.13961) [org/abs/2512.13961](https://arxiv.org/abs/2512.13961). [66,](#page-3-0) [69](#page-6-0) Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling, 2020. URL <https://arxiv.org/abs/2101.00027>. [66,](#page-3-0) [67](#page-4-0) Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. Olmo: Accelerating the science of language models, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.00838) [2402.00838](https://arxiv.org/abs/2402.00838). [64](#page-1-0) Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. *arXiv preprint arXiv:2506.04178*, 2025. [69](#page-6-0) Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025. [64,](#page-1-0) [71,](#page-8-0) [72](#page-9-0) Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. [69](#page-6-0) Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, et al. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. In *Proceedings of the 50th annual international symposium on computer architecture*, pages 1–14, 2023. [65](#page-2-0) Tom Kocmi, Rachel Bawden, Ondˇrej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Thamme Gowda, Yvette Graham, Roman Grundkiewicz, Barry Haddow, et al. Findings of the 2022 conference on machine translation (wmt22). In *Proceedings of the Seventh Conference on Machine Translation (WMT)*, pages 1– 45, 2022. [72](#page-9-0)

{16}------------------------------------------------

<span id="page-16-0"></span>Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. *arXiv preprint arXiv:2411.15124*, 2024. [71](#page-8-0) Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hess- ´ low, Roman Castagne, Alexandra Sasha Luccioni, Franc¸ois Yvon, Matthias Gall ´ e,´ et al. Bloom: A 176b-parameter open-access multilingual language model, 2023. [64](#page-1-0) Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, et al. Mitigating the alignment tax of rlhf. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 580–606, 2024. [70](#page-7-0) Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. *arXiv preprint arXiv:2402.06196*, 2024. [65,](#page-2-0) [69](#page-6-0) Shashi Narayan, Shay B Cohen, and Mirella Lapata. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. *arXiv preprint arXiv:1808.08745*, 2018. [72](#page-9-0) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022. [64,](#page-1-0) [70](#page-7-0) Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pages 311–318, 2002. [72](#page-9-0) Tiago Pimentel and Clara Meister. How to compute the probability of a word. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 18358–18375, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.1020. URL [https:](https://aclanthology.org/2024.emnlp-main.1020/) [//aclanthology.org/2024.emnlp-main.1020/](https://aclanthology.org/2024.emnlp-main.1020/). [73](#page-10-0) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. [65](#page-2-0) Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36, 2024. [70](#page-7-0)

{17}------------------------------------------------

<span id="page-17-0"></span>Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140):1–67, 2020. URL [http://jmlr.org/papers/v21/](http://jmlr.org/papers/v21/20-074.html) [20-074.html](http://jmlr.org/papers/v21/20-074.html). [65](#page-2-0) Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. *arXiv preprint arXiv:1606.05250*, 2016. [72](#page-9-0) Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? *arXiv preprint arXiv:2002.08910*, 2020. [72](#page-9-0) Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and Aman Chadha. A systematic survey of prompt engineering in large language models: Techniques and applications. *arXiv preprint arXiv:2402.07927*, 2024. [72](#page-9-0) Rico Sennrich, Barry Haddow, and Alexandra Birch. Improving neural machine translation models with monolingual data. *arXiv preprint arXiv:1511.06709*, 2015. [72](#page-9-0) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024. [71](#page-8-0) Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1909.08053) [1909.08053](https://arxiv.org/abs/1909.08053). [68](#page-5-0) Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Evan Walsh, Luke Zettlemoyer, Noah Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 15725–15788, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.840. URL [https://aclanthology.org/2024.](https://aclanthology.org/2024.acl-long.840/) [acl-long.840/](https://aclanthology.org/2024.acl-long.840/). [66](#page-3-0) Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. Spike no more: Stabilizing the pre-training of large language models, 2025. URL [https://arxiv.](https://arxiv.org/abs/2312.16903) [org/abs/2312.16903](https://arxiv.org/abs/2312.16903). [68](#page-5-0)

{18}------------------------------------------------

<span id="page-18-0"></span>Jeyan Thiyagalingam, Gregor von Laszewski, Junqi Yin, Murali Emani, Juri Papay, Gregg Barrett, Piotr Luszczek, Aristeidis Tsaris, Christine Kirkpatrick, Feiyi Wang, et al. Ai benchmarking for science: Efforts from the mlcommons science working group. In *International Conference on High Performance Computing*, pages 47–64. Springer, 2022. [74](#page-11-0) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023. [64](#page-1-0) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. [66](#page-3-0) Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP workshop BlackboxNLP: Analyzing and interpreting neural networks for NLP*, pages 353–355, 2018. [72](#page-9-0) Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. *Advances in neural information processing systems*, 32, 2019. [72](#page-9-0) Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*, 2022. [69,](#page-6-0) [72](#page-9-0) Ran Zhang and Steffen Eger. Llm-based multi-agent poetry generation in noncooperative environments. *arXiv preprint arXiv:2409.03659*, 2024. [72](#page-9-0) Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Guoyin Wang, and Fei Wu. Instruction tuning for large language models: A survey. *ACM Comput. Surv.*, 58(7), January 2026. ISSN 0360-0300. doi: 10.1145/3777411. URL [https://doi.org/10.1145/](https://doi.org/10.1145/3777411) [<sup>3777411</sup>](https://doi.org/10.1145/3777411). [69](#page-6-0) Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.](https://openreview.net/forum?id=Bl8u7ZRlbM) [net/forum?id=Bl8u7ZRlbM](https://openreview.net/forum?id=Bl8u7ZRlbM). [69](#page-6-0) Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36:55006–55021, 2023. [76](#page-13-0)

{19}------------------------------------------------

