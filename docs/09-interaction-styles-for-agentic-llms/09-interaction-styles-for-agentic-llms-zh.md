<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 193-210; Marker split PDF pages 1-18. -->

{0}------------------------------------------------

# 第 9 章

# 智能体大语言模型的交互风格

### 学习目标

本章介绍智能体交互风格与智能体记忆。学完后，你将能够：

- 识别人工智能中三种历史上的智能体类型，并说明它们与当代智能体 LLM 的联系。
- 描述智能体 LLM 的五种交互风格。
- 解释智能体为何需要记忆，并辨识几类智能体记忆架构。

# 9.1 人工智能中的智能体

智能体是人工智能领域的核心概念。本节先简要回顾人工智能研究中的几种智能体观点。

## 9.1.1 智能体在环境中行动并适应环境

在人工智能中，能够在现实或模拟世界中行动的实体称为*智能体*。研究者试图通过构建能够执行人类某些活动的系统，来理解智能的本质。

智能体是能够在环境中自主行动的软件系统。它有目标，会感知环境、作出决策并采取行动以实现目标，也可以从经验中学习。通过适应新信息，智能体能够改进自己的行为策略；用强化学习的术语来说，就是改进选择行动的策略。

传统计算机程序按预先设定的指令确定性地运行，因此适用于会计、控制等高度重视可预测性的任务。

{1}------------------------------------------------

<span id="page-1-0"></span>相比之下，LLM 智能体的行为较难确定和预测。它们能够改变行为、进行推理和规划，并使用工具为用户完成编程、客户服务和数据分析等工作流程。它们也可能产生幻觉或犯错，因此行为并非总是可靠。从这一点看，LLM 智能体有时更像人，而不像传统程序。

人工智能领域对智能体的研究由来已久。可以从三种角度来理解智能体：作为机器助手的智能体、具有信念—愿望—意图（BDI）的智能体，以及群体智能体。

#### 9.1.1.1 机器助手

工程学视角将智能体视为替人完成任务的工具。例如，工厂中的焊接机器人、飞机自动驾驶系统，以及需要在不可预测环境中寻找安全路线的自动驾驶汽车。

其他例子还包括 Alexa 和 Siri 等虚拟助手、帮助软件开发者提升效率的 Copilot 等编程助手，以及帮助用户处理投诉或退款等事务的客服机器人。机器助手也可能具备较强的推理能力 [Dorri 等，2018](#page-13-0)，而 LLM 正越来越多地应用于各类助手 [Plaat 等，2025](#page-14-0)。

#### 9.1.1.2 具有信念、愿望和意图的智能体

认知科学视角关注对人类认知过程的建模。BDI 框架中的智能体具有信念、愿望和意图。人工智能智能体可以实现认知模型，帮助研究者探讨推理、合作、创造、感知、身体行动和自我意识等问题。

BDI 智能体源于逻辑推理传统。它们被赋予目标，并通过行动追求这些目标。相关研究将行动视为实现智能体目标的手段 [Rao 等，1995](#page-14-0)；[Georgeff 等，1998](#page-13-0)。

#### 9.1.1.3 群体智能体

第三种视角关注大量简单智能体之间的交互。与由少量复杂智能体构成的多智能体系统不同，群体系统研究集体行为如何涌现，其规模可能超出任何单个智能体的推理能力。

{2}------------------------------------------------

<span id="page-2-0"></span>研究者会提出这样的问题：群体能否保护自己免受捕食者攻击？合作需要哪些条件？社会何时和平、何时陷入冲突？这些问题也引出了“超个体”的概念 [Johnson 和 Linksvayer，2010](#page-13-0)；[Wilson 和 Sober，1989](#page-15-0)。

群体智能体承袭了具身人工智能中的连接主义传统 [Brooks，1990](#page-13-0)。研究者关注智能体之间的通信如何产生合作等大规模行为模式。蚁群优化是著名的群体算法 [Dorigo 等，2007](#page-13-0)，NetLogo 则是用于研究这类行为的软件环境 [Wilensky 和 Rand，2015](#page-15-0)。<sup>1</sup>

多智能体系统和智能体 LLM 都可以用于研究上述三种方向：把智能体当作工具、把智能体视为认知实体，以及借助智能体研究社会行为。全书第四部分聚焦作为工具的智能体，第五部分聚焦认知和智能体社会。

## 9.1.2 将智能体 LLM 视为强化学习智能体

人工智能长期以来都在研究智能体如何行动，以及如何从环境中学习。强化学习研究智能体如何从反馈中学习 [Sutton 和 Barto，2018](#page-15-0)；多智能体强化学习则将这一视角扩展到相互作用的多个智能体 [Wong 等，2021](#page-15-0)。

下一节将介绍智能体 LLM 的五种交互风格。在此之前，我们先借助强化学习概念描述与世界交互的智能体。

图 [9.1](#page-3-0) 展示了由智能体和环境组成的强化学习系统。智能体采取行动，环境则返回新的状态，以及表示该状态优劣的奖励。

智能体利用奖励学习在不同状态下哪些行动更有利。它的目标是找到有效的行为策略，记为 \(\pi\)，将状态映射为行动；最优策略记为 \(\pi^*\)。如果策略由神经网络表示，其参数记为 \(\theta\)。

环境对智能体所处的世界建模：智能体的行动会改变环境状态，环境再返回奖励。这两个过程可以分别用状态转移函数 \(T\) 和奖励函数 \(R\) 表示。

<sup>1</sup> <https://www.netlogo.org>

{3}------------------------------------------------

<span id="page-3-0"></span>![](assets/_page_3_Diagram_2.jpeg)

*图 9.1：强化学习循环。智能体根据环境的转移函数 \(T\) 和奖励函数 \(R\) 提供的奖励反馈 \(r\)，学习参数化策略 \(\pi_\theta\)，以响应行动 \(a\)。* [Sutton 和 Barto，2018](#page-15-0)

![](assets/_page_3_Diagram_4.jpeg)

*图 9.2：由 LLM 参数化智能体策略 \(\pi_{\mathrm{LLM}_\pi}\) 的强化学习系统。该策略使用环境反馈 \(r\) 来选择行动 \(a\)。*

强化学习循环为理解智能体如何从环境反馈中学习提供了一个有用模型。从这个角度看，LLM 可以被视为将提示映射为答案的函数。它既可以在智能体一侧发挥作用，例如根据奖励反馈 *r* 为状态 *s* 选择行动 *a*，充当策略改进器；也可以在环境一侧发挥作用。上下文强化学习会用 LLM 改进策略，思维树则可以用 LLM 生成并评估候选行动。

![](assets/_page_3_Diagram_8.jpeg)

*图 9.3：智能体策略 π 以及环境的转移函数 T 和奖励函数 R 均由 LLM 实现的强化学习系统。每个 LLM 都通过提示生成行动、状态或奖励。*
{4}------------------------------------------------

<span id="page-4-0"></span>

| 风格 | 提示或交互方式 | LLM 的训练方式 | 示例 |
|---|---|---|---|
| **A. 查询** | 用户编写的零样本查询或指令 | 预训练 | GPT-2 [Radford 等，2019](https://cdn.openai.com/GPT_2_August_Report.pdf) |
| **B. 上下文学习** | 少样本示例 | 预训练 | GPT-3 [Brown 等，2020](#page-13-0) |
| **C. 思维链** | 推理步骤示例，或零样本思维链指令 | 预训练 | 思维链 [Wei 等，2022](#page-15-0)；零样本思维链 [Kojima 等，2022](#page-13-0) |
| **D. 自我反思** | 反馈执行框架、提示优化或搜索/强化学习 | 执行框架调用预训练模型 | 思维树 [Yao 等，2024](#page-16-0)；LLaMea [van Stein 和 Back，2024](#page-15-0)；ReAct [Yao 等，2022](#page-16-0) |
| **E. 强化微调** | 强化学习提示与优化执行框架 | 预训练并进行微调 | RLVR；Tulu 3 [Lambert 等，2024](#page-14-0) |

*表 9.1：五种智能体风格。*

LLM 也可以置于环境一侧，用来计算状态奖励 \(\mathrm{LLM}_R(s)\)，或预测下一状态 \(\mathrm{LLM}_T(s,a)\)。图 9.3 中，同一个 LLM 通过不同提示承担策略、状态转移和奖励函数三种角色。例如，思维树可以调用 LLM 评估某个状态。强化学习框架为比较不同智能体风格提供了统一的描述方式。

# 9.2 智能体 LLM 中的学习机制

智能体 LLM 是一个较新的研究领域，前面几章介绍了多种在智能体场景中使用 LLM 的方法。本节根据提示方式和训练方式梳理这些方法。

我们将其归纳为五种交互风格。表 9.1 概括了每种风格在推理时如何由执行框架提示 LLM、模型经过何种训练，以及有哪些代表性实例。不同风格都可以在适合的领域中构建实用助手或交互伙伴；更复杂的风格则旨在提升模型处理困难任务的能力。

表格第一列列出交互风格，从简单查询到强化微调；第二列说明提示方式，从静态指令到动态强化学习提示；第三列说明 LLM 是否经过预训练或微调；最后一列列出文献中的实例。

## 9.2.1 智能体交互风格

下面逐一介绍这五种风格。前三种使用静态提示，由用户提供，交互过程中不会改变。

{5}------------------------------------------------

![](assets/_page_5_Diagram_2.jpeg)

*图 9.4：风格 A——静态查询。*

![](assets/_page_5_Diagram_4.jpeg)

<span id="page-5-0"></span>*图 9.5：风格 B——在提示中提供少量示例的上下文监督学习。*

#### 9.2.1.1 A. 查询提示

第一种风格是给 LLM 一条简单指令，如图 9.4 所示。例如，用户输入“莉莉错过了火车。她感到非常……”并期待模型回答“生气”或“难过”。提示中只有查询，没有训练示例，因此这是零样本、指令式交互。

模型之所以能够回答许多这类问题，是因为预训练语料中可能包含相关信息，而指令微调则让模型学会遵循请求。GPT-2 是早期的此类模型之一。

#### 9.2.1.2 B. 上下文学习

上下文学习是利用提示中的示例进行学习的一种方式，也是监督学习和强化学习之外的重要学习形式。LLM 往往只需少量示例就能完成任务，这回应了机器学习领域的一项核心挑战。

与简单查询相比，少样本上下文学习会将若干示例与请求一同提供给模型。这些示例引导解码器利用参数中已有的相关知识。

图 9.5 展示了这种交互方式。GPT-3 是一个著名实例；Brown 等人介绍了语言模型的少样本评估方式 [Brown 等，2020](#page-13-0)。提示可以是：“补全这些对应关系：sea—la mer；oysters—huîtres；window—la fenêtre；cheese—……”预期答案是 *fromage*。

上下文学习最初是在足够大的模型中明显观察到的，例如拥有 1750 亿参数的 GPT-3。该行为出现所需的模型规模取决于模型和训练方法；随着方法改进，这一门槛也可能降低。

{6}------------------------------------------------

<span id="page-6-0"></span>![](assets/_page_6_Diagram_2.jpeg)

*图 9.6：风格 C。上：以推理步骤为示例的思维链上下文学习。下：零样本思维链，使用一种通常有效的指令提示。*

#### 9.2.1.3 C. 思维链

思维链提示要求模型生成中间推理步骤 [Wei 等，2022](#page-15-0)；图 9.6 上半部分展示了少样本形式。正如第 6 章所述，这种方法部分是为了解决 LLM 在多步数学应用题上的困难。早期思维链方法的一个局限是：每遇到新的推理任务，用户都需要编写相应的任务示例。

零样本思维链是后续发展之一，如图 9.6 下半部分所示。Kojima 等人发现，加入“让我们一步一步思考”这样的通用指令，可以提升模型在许多基准上的表现，但效果可能不如针对任务编写的提示 [Kojima 等，2022](#page-13-0)。

与上下文学习类似，思维链也利用提示中的示例。区别在于，这些示例展示的是*推理步骤*，而不仅是事实或输入—输出对。推理步骤可以引导模型逐步求解。第 129 页的图 6.1 给出了少样本和零样本思维链示例。

思维链通常需要能力足够强的模型才能奏效，不过更好的方法可能降低所需规模。也可以通过蒸馏，将大模型学到的推理方式迁移到参数量小得多的模型中 [Li 等，2023](#page-13-0)。

#### 9.2.1.4 D. 自我反思执行框架

前三种交互风格都使用单个静态提示，在许多基准上表现良好。对于规划或需要跟踪状态的任务，模型可能需要探索多种可能，并根据结果调整提示。

接下来介绍两种交互风格：由 LLM 外部的小型算法动态管理提示。这个算法称为*执行框架*。风格 D 通过上下文更新提示，但不修改 LLM 参数；风格 E 还会微调模型。

这两种风格都是动态方法。正如第 8 章所述，关键挑战之一是让由此形成的学习过程保持稳定。

{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Diagram_2.jpeg)

*图 9.7：风格 D——使用 LLM 进行自我反思。上：搜索过程提出并评估子问题；下：强化学习循环根据状态、行动和奖励历史更新策略提示。* [Yao 等，2024](#page-16-0)

图 9.7 展示了两种管理自我反思的方式：搜索候选子问题，以及根据反馈更新提示。在反馈循环中，策略提示会根据交互历史 \((s,a,r)\) 更新（另见第 20.2.1 节）。

{8}------------------------------------------------

<span id="page-8-0"></span>![](assets/_page_8_Diagram_2.jpeg)

*图 9.8：风格 E——强化微调 LLM 参数 \(\psi\)，同时通过强化学习更新策略参数 \(\theta\)。* [Luong 等，2024](#page-14-0)

#### 9.2.1.5 E. 强化微调执行框架

到目前为止，学习都发生在上下文中：模型根据提示里的示例作答，参数本身并未改变。上下文学习使用方便，但模型从经验中适应或保留知识的能力有限。如果希望模型从交互中学得更深入，就可能需要更新参数。我们将通过奖励信号训练并更新模型参数的方法称为*强化微调*。RLVR 是一个例子。

图 9.8 中，可训练的 LLM 参数记为 \(\psi\)。与图 9.7 下方的方式相比，现在会依据数据中的奖励信号更新模型本身。标准流程通常在推理之前进行微调；但在数学和编程任务中，推理阶段生成的多步轨迹也可能产生奖励信号，供之后微调使用。因此，强化微调会把推理与训练联系起来：模型在交互中生成训练目标，再用这些目标更新模型本身。这模糊了推理和微调之间的界限；传统监督微调则将训练与推理明确分开。

LLM 的开发通常包括三个阶段：预训练、微调和测试（推理）。风格 D 在推理阶段利用反馈进行学习；风格 E 则在微调阶段利用智能体交互结果更新模型。它并不表示普通测试阶段会即时更新模型参数。

风格 D 和 E 都需要谨慎设计，以确保收敛稳定。截至原书撰写时，报告这类结果的研究还不多。

{9}------------------------------------------------

# 9.3 智能体 LLM 的记忆架构

人类和其他生物都依赖记忆维持身份、记住关系、从经验中学习并调整行为。记忆丧失可能改变一个人的行为，严重时还会损害其身份认同。

智能体同样需要记忆。本节将说明智能体 LLM 系统为何需要记忆，并概述几类记忆架构。

### 9.3.1 为什么需要记忆

统计模型 \(f(x)\to\hat y\) 根据训练期间学到的关系 \(f\)，由输入 \(x\) 估计输出 \(\hat y\)。这种输入—输出函数本身不会保留训练后出现的新信息；新信息必须通过输入 \(x\) 提供。因此，早期聊天机器人会使用隐藏的*系统提示*加入对话历史，帮助模型维持连贯对话。

随着智能体 LLM 的应用增多，仅靠系统提示已不足以满足需求。以下场景说明了记忆的作用。

#### 9.3.1.1 连贯对话

许多 LLM 基准采用“提问—回答”形式，适合将输入映射到输出的统计模型。但在实际应用中，人们通常希望聊天机器人能够进行有意义的交流。为了保持对话连贯，机器人必须跟踪用户和助手之前说过的内容。

因此，智能体 LLM 的首要记忆需求，是能够访问相关对话历史。

#### 9.3.1.2 适应

根据第 1 章的定义，智能体 LLM 应具备学习能力。如果机器人正在切洋葱，而刀的位置发生了变化，它就应先找到刀，再继续切菜。智能体会与环境交互；环境发生变化时，智能体也应作出调整。

智能体可以依靠泛化能力很强的预训练策略，也可以通过灵活的架构识别何时需要适应、微调甚至重新训练。记忆能够支持这一过程。

{10}------------------------------------------------

<span id="page-10-0"></span>如第 6.2.4 节所述，自我反思智能体所使用的外部提示改进执行框架，需要借助缓冲区等记忆结构来显式优化提示。

#### 9.3.1.3 身份

经历会塑造个性和身份。若要让 LLM 智能体形成持续的身份，它必须将经历保存在记忆中供自我反思，或以其他方式处理经历并据此调整行为策略。

随着智能体执行框架的记忆架构改进，智能体或许能够更好地记住经历，并逐渐形成独特的个性。

#### 9.3.1.4 持续学习

传统机器学习通常针对特定任务和数据集训练模型，而人类会在一生中持续学习。持续学习研究如何让模型依次学习新任务，同时保留先前学到的能力 [Wang 等，2024](#page-15-0)；[Li 等，2025a](#page-14-0)。与元学习类似，持续学习也关注相关任务之间的泛化。其主要挑战包括灾难性遗忘，以及如何应对数据分布变化。

持续学习领域提出了多种支持此类训练的架构。

## 9.3.2 记忆架构

面向 LLM 智能体的记忆架构仍是一个较新的研究领域。以下小节介绍其中几类方法。

#### 9.3.2.1 微调

一种务实的做法是在新信息出现时将其纳入模型。模型参数本身可以充当记忆，因此可以用新示例微调模型，而不必开发独立的记忆管理算法。

在智能体场景中，微调必须足够快，以免用户明显感到延迟。LoRA 等参数高效微调方法是一种选择 [Yang 等，2024](#page-15-0)。这种方法无需额外记忆结构，但微调通常无法快到满足交互需求，因此上下文或执行框架方案可能更实用。

#### <span id="page-11-0"></span>9.3.2.2 长上下文窗口与压缩

另一种方法是管理提示。聊天机器人常将截断后的对话历史与用户最新消息一并放入提示中。实践中这种方法在许多情况下都有效，但上下文窗口容量有限，而且标准注意力机制的计算量会随序列长度呈平方级增长，因此处理长上下文代价较高。

已有两类应对思路。第一，长上下文架构通过不同方式处理长距离注意力 [Smith 等，2022](#page-14-0)；Liu 等人综述了近期方法 [2025](#page-14-0)。第二，提示压缩方法将提示压缩成更短的表示 [Li 等，2025b](#page-14-0)。提示还可以切分成多个块 [Sheng 等，2025](#page-14-0)，或由 LLM 生成代码并递归处理 [Zhang 等，2025a](#page-16-0)。

压缩也可以通过过滤冗余词元实现。例如，“提示可以通过过滤不必要的词元来压缩”可以缩写为“提示 压缩 过滤 不必要 词元”。另一种方法是软提示 [Lester 等，2021](#page-13-0)：它不使用离散文本词元，而是通过反向传播学习嵌入，并直接输入 Transformer。已有研究报告了约十倍 [Mu 等，2023](#page-14-0) 到数百倍 [Li 等，2025c](#page-14-0) 的压缩率。

#### 9.3.2.3 显式记忆管理

如果需要在多次 LLM 调用之间保存状态，一个直接的方法是由智能体执行框架提供存储，并在后续提示中加入相关内容。第 6.2.4 节介绍了用于自我反思的外部提示改进方法，包括 Reflexion 和 Buffer of Thoughts（图 6.8 和图 6.6）。

不同智能体的存储需求各异 [Zhang 等，2025b](#page-16-0)；[Chhikara 等，2025](#page-13-0)。记忆可以用文本或参数表示，常见操作包括读取、写入、压缩和遗忘；记忆系统还可以提供额外服务。例如，MemoryBank 会生成分层摘要，并使用提示持续构建用户交互画像 [Zhong 等，2024](#page-16-0)。

智能体也可以通过函数调用读写外部记忆。MemGPT 研究了这种“记忆即工具”的方式 [Packer 等，2023](#page-14-0)。A-mem 则用受邮箱启发的组织方式为记忆添加上下文描述 [Xu 等，2025](#page-15-0)。

显式记忆管理的一个局限是，LLM 可能没有经过有效管理记忆的训练。若需求可以预先确定，开发者也可以直接编写记忆管理代码。

{12}------------------------------------------------

<span id="page-12-0"></span>模型上下文协议（MCP）提供了一套上下文管理策略；第 11.4 节会进一步介绍该协议。

**学会使用记忆。** Mem1 采用另一种方法：每次交互后，智能体都会更新一份供后续 LLM 调用共享的紧凑状态。该状态大小固定，不会持续增长直至超出上下文窗口。系统通过在提示中加入推理风格标签来构造共享状态，并使用强化学习训练智能体整合状态 [Zhou 等，2025](#page-16-0)。Mem-\(\alpha\) 也使用强化学习训练智能体有效使用记忆 [Wang 等，2025](#page-15-0)。

AgeMem 将记忆管理操作整合进智能体策略，并进行端到端优化。它区分长期记忆和短期记忆，并使用三阶段强化学习方案训练智能体管理这两类记忆 [Yu 等，2026](#page-16-0)。

这仍是活跃的研究领域，预计还会出现新的研究和框架。Zhou 等人提供了一篇综述 [Zhou 等，2026](#page-16-0)。

**向量数据库。** 除了以可读文本保存提示，也可以直接保存提示的词元嵌入。文本适合精确或词面检索；嵌入则编码语义关系，有助于检索含义相近的内容。

向量数据库存储高维向量，并支持快速相似性搜索，因此适合在 LLM 应用中检索语义相关的内容 [Xie 等，2023](#page-15-0)；[Ma 等，2023](#page-14-0)。

## 9.4 结语

智能体是人工智能的核心概念。LLM 显著提升了智能体的沟通和推理能力。本章介绍了几种历史上的智能体观点，并将它们与当代系统联系起来，也概述了五种交互风格。

智能体需要行动，也需要记忆，才能与环境进行有意义的交互。智能体记忆仍是快速发展的研究方向。

下一章将从理论转向实践案例，并运用前文介绍的一些概念。

{13}------------------------------------------------

# <span id="page-13-0"></span>Bibliography
Rodney A Brooks. Elephants don't play chess. *Robotics and autonomous systems*, 6 (1-2):3–15, 1990. [195](#page-2-0) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. [197,](#page-4-0) [198](#page-5-0) Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. *arXiv preprint arXiv:2504.19413*, 2025. [204](#page-11-0) Marco Dorigo, Mauro Birattari, and Thomas Stutzle. Ant colony optimization. *IEEE computational intelligence magazine*, 1(4):28–39, 2007. [195](#page-2-0) Ali Dorri, Salil S Kanhere, and Raja Jurdak. Multi-agent systems: A survey. *Ieee Access*, 6:28573–28593, 2018. [194](#page-1-0) Michael Georgeff, Barney Pell, Martha Pollack, Milind Tambe, and Michael Wooldridge. The belief-desire-intention model of agency. In *International workshop on agent theories, architectures, and languages*, pages 1–10. Springer, 1998. [194](#page-1-0) Brian R Johnson and Timothy A Linksvayer. Deconstructing the superorganism: social physiology, groundplans, and sociogenomics. *The Quarterly Review of Biology*, 85 (1):57–79, 2010. [195](#page-2-0) Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213, 2022. [197,](#page-4-0) [199](#page-6-0) Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. *arXiv preprint arXiv:2411.15124*, 2024. [197](#page-4-0) Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameterefficient prompt tuning. In *Proceedings of the 2021 conference on empirical methods in natural language processing*, pages 3045–3059, 2021. [204](#page-11-0) Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. Symbolic chain-of-thought distillation: Small models can also "think" stepby-step. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2665–2679, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.150. URL <https://aclanthology.org/2023.acl-long.150/>. [199](#page-6-0)

{14}------------------------------------------------

- <span id="page-14-0"></span>Yujie Li, Xiangkun Wang, Xin Yang, Marcello Bonsangue, Junbo Zhang, and Tianrui
- Li. Improving open-world continual learning under the constraints of scarce labeled data. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2*, pages 1647–1658, 2025a. [203](#page-10-0) Zongqian Li, Yinhong Liu, Yixuan Su, and Nigel Collier. Prompt compression for large language models: A survey. In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 7182–7195, 2025b. [204](#page-11-0) Zongqian Li, Yixuan Su, and Nigel Collier. 500xcompressor: Generalized prompt compression for large language models. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 25081–25091, 2025c. [204](#page-11-0) Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context language modeling. *arXiv preprint arXiv:2503.17407*, 2025. [204](#page-11-0) Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang
- Li. Reft: Reasoning with reinforced fine-tuning. *arXiv preprint arXiv:2401.08967*, 2024. [201](#page-8-0) Le Ma, Ran Zhang, Yikun Han, Shirui Yu, Zaitian Wang, Zhiyuan Ning, Jinghan Zhang, Ping Xu, Pengjiang Li, Wei Ju, et al. A comprehensive survey on vector database: Storage and retrieval technique, challenge. *arXiv preprint arXiv:2310.11703*, 2023. [205](#page-12-0) Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens. *Advances in Neural Information Processing Systems*, 36:19327–19352, 2023. [204](#page-11-0) Charles Packer, Vivian Fang, Shishir G Patil, Kevin Lin, Sarah Wooders, and Joseph E Gonzalez. Memgpt: towards llms as operating systems., 2023. [204](#page-11-0) Aske Plaat, Max van Duijn, Niki van Stein, Mike Preuss, Peter van der Putten, and Kees Joost Batenburg. Agentic large language models, a survey. *Journal of Artificial Intelligence Research, arXiv preprint arXiv:2503.23037*, 84:29:1–29:74, 2025. [194](#page-1-0) Anand S Rao, Michael P Georgeff, et al. Bdi agents: From theory to practice. In *Icmas*, volume 95, pages 312–319, 1995. [194](#page-1-0) Boheng Sheng, Jiacheng Yao, Meicong Zhang, and Guoxiu He. Dynamic chunking and selection for reading comprehension of ultra-long context in large language models. *arXiv preprint arXiv:2506.00773*, 2025. [204](#page-11-0) Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for sequence modeling. *arXiv preprint arXiv:2208.04933*, 2022. [204](#page-11-0)

{15}------------------------------------------------

- <span id="page-15-0"></span>Richard S Sutton and Andrew G Barto. *Reinforcement Learning*. Adaptive Computation and Machine Learning series. Bradford Books, Cambridge, MA, 2 edition, November 2018. [195,](#page-2-0) [196](#page-3-0) Niki van Stein and Thomas Back. Llamea: A large language model evolutionary algo- ¨ rithm for automatically generating metaheuristics. *arXiv preprint arXiv:2405.20132*, 2024. [197](#page-4-0) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. [197,](#page-4-0) [198](#page-5-0) Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. *IEEE transactions on pattern analysis and machine intelligence*, 46(8):5362–5383, 2024. [203](#page-10-0) Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-{\alpha}: Learning memory construction via reinforcement learning. *arXiv preprint arXiv:2509.25911*, 2025. [205](#page-12-0) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824– 24837, 2022. [197,](#page-4-0) [199](#page-6-0)
- U. Wilensky and W. Rand. *An Introduction to Agent-Based Modeling: Modeling Natural, Social, and Engineered Complex Systems with NetLogo*. The MIT Press. MIT Press, 2015. ISBN 9780262731898. URL [https://books.google.nl/](https://books.google.nl/books?id=LQrhBwAAQBAJ) [books?id=LQrhBwAAQBAJ](https://books.google.nl/books?id=LQrhBwAAQBAJ). [195](#page-2-0) David Sloan Wilson and Elliott Sober. Reviving the superorganism. *Journal of theoretical Biology*, 136(3):337–356, 1989. [195](#page-2-0) Annie Wong, Thomas Back, Anna V Kononova, and Aske Plaat. Multiagent deep ¨ reinforcement learning: Challenges and directions towards human-like approaches. *arXiv preprint arXiv:2106.15691*, 2021. [195](#page-2-0) Xingrui Xie, Han Liu, Wenzhe Hou, and Hongbin Huang. A brief survey of vector databases. In *2023 9th international conference on big data and information analytics (BigDIA)*, pages 364–371. IEEE, 2023. [205](#page-12-0) Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. Amem: Agentic memory for llm agents. *arXiv preprint arXiv:2502.12110*, 2025. [204](#page-11-0) Menglin Yang, Jialin Chen, Jinkai Tao, Yifei Zhang, Jiahong Liu, Jiasheng Zhang, Qiyao Ma, Harshit Verma, Regina Zhang, Min Zhou, et al. Low-rank adaptation for foundation models: A comprehensive review. *arXiv preprint arXiv:2501.00365*, 2024. [203](#page-10-0)

{16}------------------------------------------------

<span id="page-16-0"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*, 2022. [197](#page-4-0) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 36, 2024. [197,](#page-4-0) [201](#page-8-0) Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, and Libing Wu. Agentic memory: Learning unified long-term and short-term memory management for large language model agents. *arXiv preprint arXiv:2601.01885*, 2026. [205](#page-12-0) Alex L Zhang, Tim Kraska, and Omar Khattab. Recursive language models. *arXiv preprint arXiv:2512.24601*, 2025a. [204](#page-11-0) Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model-based agents. *ACM Transactions on Information Systems*, 43(6):1–47, 2025b. [204](#page-11-0) Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In *Proceedings of the AAAI conference on artificial intelligence*, volume 38, pages 19724–19731, 2024. [204](#page-11-0) Wei Zhou, Xuanhe Zhou, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu Li, Feiyu Xiong, and Fan Wu. Are we ready for an agent-native memory system? *arXiv preprint arXiv:2606.24775*, 2026. [205](#page-12-0) Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. *arXiv preprint arXiv:2506.15841*, 2025. [205](#page-12-0)

{17}------------------------------------------------

