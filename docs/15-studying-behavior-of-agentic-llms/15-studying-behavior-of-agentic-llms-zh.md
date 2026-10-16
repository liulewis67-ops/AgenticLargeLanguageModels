<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 287-316; Marker split PDF pages 1-30. -->



{0}------------------------------------------------

## <span id="page-0-0"></span>第15章

# 智能体大语言模型的研究行为

#### 你将学到的内容

在本章中，您将学习智能体大语言模型的行为如何与人类行为进行比较。具体来说，您将能够：

- 解释 *LLM 行为* 的含义，并描述如何研究它。
- 列出*基本的社交和互动能力*，并描述LLM在哪些方面表现良好，哪些方面存在困难。
- 描述在 LLM 中研究高级社会和互动能力的方法，特别是*战略思维*、*心智理论*和*道德推理*。
- 反思行为学方法对于 LLM 能告诉我们什么和不能告诉我们什么，考虑到关于*测试和解释*的各种问题。

#### 15.1 LLM 的行为是什么？

2019年，一支由计算科学、行为科学和认知科学领域的作者组成的团队描述了一种新兴的研究领域：*机器行为*。该领域旨在*关注智能机器的科学研究，不是作为工程制品，而是作为具有特定行为模式和生态的行动者类别* [\[Rahwan et al.,](#page-24-0) [2019\]](#page-24-0)。机器行为本质上是跨学科的，因为它将*技术*（算法、数据结构、训练过程）的内在知识与其实际使用的*环境和背景*的知识整合在一起。

在机器行为中的方法可与动物行为学或心理学中的工作进行比较：在这里，要推进对动物和人类功能的理解，也需要整合内在层面（例如生物物理学、神经学）和生态层面（与环境的互动、社会互动）的见解。
  
 正如本部分引言所述，研究机器行为的一项现实动机是 LLM 正日益融入日常工作和生活，因此了解其行为倾向和模式变得十分重要。

{1}------------------------------------------------

<span id="page-1-0"></span>

还有一个科学动机：由于大规模神经架构的复杂性（它们表现为*黑箱*），很难以分析方式追踪其运行情况。（虽然机制可解释性领域确实尝试做这件事，参见第21章。）经验性研究这种复杂系统的*行为*是我们理解的补充途径。换句话说，由于交互的复杂性，即使一位理论工程师对LLM的架构和训练数据了解得一清二楚，他仍然可能无法预测它在不同类型的情况下如何反应。行为方法旨在填补这一空白。

什么算作*行为*取决于所研究系统的可供性<sup>1</sup>以及研究的范围。例如，对于一个基本的LLM，行为等于模型基于令牌推理产生的*输出*。另一方面，对于一个能够执行例如网页搜索和运行代码的智能体大语言模型，行为也可以包括它为获得某个输出或达到某种效果所执行的*动作*。正如我们将看到的，有些研究只考虑最终输出并将其视为模型的行为；其他研究则也关注中间步骤。

#### 15.2 实验探测 LLM 的行为

近年来，出现了许多测试 LLM 社交和互动能力的研究。主要有两种方法：（1）使用从人类实验范式改编的项目来提示模型；（2）为 LLM 专门开发新的测试范式。后者可以通过定量方式进行，使用大规模基准，或者通过定性方式进行，在较小范围内进行深入的手动分析。接下来我们将看到一些例子。

将行为和心理方法应用于（代理的）LLM 并不简单。为人类设计的实验假设存在一个具身主体，拥有信念、欲望以及贯穿时间的自我意识。而智能体大语言模型在任何明显意义上都不具备这些。这需要对*建构效度*进行反思：在 AI 系统中测量某种特质或能力意味着什么？这个问题适用于两个层面：特质的可观察结果以及其底层机制。在 [15.5](#page-10-0) 节中讨论了 *错误信念测试* [\[Carpenter et al.,](#page-20-0) [2002\]](#page-20-0) 的例子。尽管一些模型类别已被证明能够以类似于人类的方式回答本测试的问题（行为结果的水平），但目前尚不清楚生成这种行为的潜在机制在Transformer 模型和人类之间有多大程度的差异或重叠。在第22章中，将更深入地讨论构念效度的问题。

对于 LLM 特定的测试，也存在方法论上的挑战需要考虑。例如，LLM 对看似无关的提示变化（如词序和逗号）非常敏感，也可能因采样波动、温度设置或随机种子不同而产生不同输出。

<sup>1</sup>*可供性* 指的是系统在特定上下文中根据其设计和实现方式能够完成的操作。

{2}------------------------------------------------

<span id="page-2-0"></span>此外，实验材料泄露到训练数据中，可能导致模型依靠记忆而非泛化能力取得高分。我们将在接下来的章节中看到几个应用示例。在这一部分中，我们将在每章末关注方法论的严谨性和解释上的谨慎。

在接下来的部分中，我们将首先讨论代理交互的基本社交工具包。接下来，我们将研究战略行为，即代理如何推理其他代理的行为，并讨论心智理论。最后，我们将讨论LLM的道德判断。

#### 15.3 基本社会工具包

本节调查基本的社会和互动能力，包括对话、语用学和社交礼仪。

#### 15.3.1 功能语言能力与语用能力

如第一部分所讨论的，经过指令调优的 LLM 的关键进展在于它们能够以聊天机器人风格使用自然语言进行互动。这不仅需要一定程度的*形式*语言能力（即生成正确、语法无误的句子），还需要*功能*语言能力，即在特定语境中理解话语含义的能力 [\[Mahowald et al.,](#page-23-0) [2023\]](#page-23-0)。在语言学和认知科学中，功能语言能力的研究属于*语用学*领域。语用学的核心概念可以追溯到 20 世纪中叶语言哲学中的*语用转向*，其提出语言意义在大多数情况下并不固有于符号本身（符号、单词、句子），而在于它们在实际语境中的使用。这包括*语言环境*，涵盖从周围词汇到诸如标题和体裁等远距离因素，以及*语言外环境*，涵盖空间、时间和社会维度。这一观点最著名地体现在保罗·格赖斯的*会话含义*概念中 [\[Grice,](#page-21-0) [1975\]](#page-21-0)。请考虑以下示例：

- 1. A：你能把盐递给我吗？B：是的，我可以。你还希望我也这样做吗？
- 2. A: Sarah是一个好同事吗？B: 嗯，可以说她总是准时。
- 3. 他发出了一系列与一首咏叹调的乐谱紧密对应的声音。

格赖斯认为，语言的使用受*合作原则*的支配：我们假设我们的对话者（人类或代理）试图进行合理的交流。上面的例子1显示了B破坏合作原则的一个经典实例：有人问你是否能递盐时，通常并不是字面上想得到一个确认这种可能性的答案，而是希望实际递盐的行为。在例子2中，B的回应明显仅集中在一个方面（守时性），听者可以推断，例如，B可能在暗示Sarah的团队合作精神有问题。

{3}------------------------------------------------

<span id="page-3-0"></span>同样，当有人说出“3”而不仅仅是“他唱了一首咏叹调”时，这种夸张而精致的表述可能暗示了对表演的某种态度；例如，表明这不是说话者喜欢的。在所有情况下，字面上说出的只是信息的一部分——另一部分需要听者根据上下文解码或“读懂言外之意”。

以灵活、依赖上下文的方式解释语言意义长期以来一直是自然语言处理中的一个臭名昭著的问题 [\[Trott et al.,](#page-26-0) [2020\]](#page-26-0)。普遍认为，LLM 在这一问题上已取得显著进展，但其原因尚不明确。模型规模是功能和语用能力的重要预测因素，一些形式的微调（特别是 SFT 和 DPO）也是如此 [\[Yu et al.,](#page-28-0) [2026,](#page-28-0) [Ruis et al.,](#page-25-0) [2023\]](#page-25-0)。然而，不同功能和语用理解领域之间的差异在 LLM 中仍然很大，而且总体得分远低于人类表现 [\[Sravanthi et al.,](#page-26-0) [2024\]](#page-26-0)。

一种解释是，LLM 获取语言之外情境信息的能力有限：它们无法像人类同类那样看到、听到或以其他方式感知，而且在大多数情况下，它们并不了解先前的互动 [\[van der Meulen](#page-26-0) [et al.,](#page-26-0) [2025\]](#page-26-0)。随着向智能体大语言模型的转变，这种情况可能正在改变，因为这些模型配备了记忆、多模态能力以及增加上下文的其他工具。目前缺乏关于这如何影响性能的实证数据。

#### 15.3.2 礼貌与社交礼仪

几十年来，人机交互中的社交礼仪和礼貌一直是研究的对象 [\[Ribino,](#page-24-0) [2023\]](#page-24-0)。一个有用的理论基础是*计算机是社会行为者*（CASA）范式 [\[Nass et al.,](#page-24-0) [1994\]](#page-24-0)，它认为即使我们知道机器不是人类社会代理，我们仍倾向于将常见的人类社会规则和期望应用于计算机。我们通过显示和回应礼貌、个性和互惠的线索，来与机器协商信任和接受度。研究表明，当机器遵守社会礼仪时，人类更信任它们 [\[Miller,](#page-23-0) [2005\]](#page-23-0)，从而提高任务执行效率和用户满意度 [\[Miyamoto et al.,](#page-24-0) [2021\]](#page-24-0)。我们将讨论礼貌理论及其在 LLM 中的应用。

**礼貌理论。** 经典礼貌理论 [e.g. [Brown and Levinson,](#page-20-0) [1987\]](#page-20-0) 直接建立在上文介绍的格赖斯理论基础上。该理论认为，交谈者不断管理彼此的*面子*，即每个参与者都希望维持的公共自我形象，它包含正面维度（希望被他人喜欢和认可的愿望）和负面维度（希望避免社会障碍的愿望）。从格赖斯的角度来看，许多日常请求、批评和拒绝本质上都是*威胁面子*的，而说话者会因此采用礼貌策略。该框架为 LLM 研究者提供了一个词汇，用于询问 LLM 及其用户是否识别面子威胁并在情境上适当地减轻它们。

LLM 对他们的用户礼貌吗？一个相关的方向涉及模型本身是否产生社会和文化上合适的礼仪。在这里，前一小节中指出的限制再次出现——LLM 获得 
 的途径有限。
{4}------------------------------------------------

<span id="page-4-0"></span>至额外的语言之外的背景。礼貌具有文化特性，但主要基于英语数据训练的系统往往默认采用以英语为母语者为中心、注重直接性的规范[\[Havaldar et al.,](#page-21-0) [2023\]](#page-21-0)。这一问题的一个明显例子出现在对波斯语*taarof*的研究中，taarof是一种仪式化的出价和拒绝系统，其中恰当的回应常常与直接回应相反。一项研究显示，在taarof在文化上被期望的情境中，前沿模型相比母语者的表现落后超过40%，而是默认采用西方式的直接回应。这导致表面上的礼貌与文化适宜性之间出现脱节。例如，在一项研究中，该模型的回应在一个标准指标上被评为*礼貌*的得分为84.5%，而只有41.7%的回应被认为在文化上是正确的 [\[Gohari Sadr et al.,](#page-21-0) [2025\]](#page-21-0)。在西方礼貌评分标准上得分高，并不等于在所有情境下行为得体。这提醒我们在测量模型的礼貌性时，要不断反思构念效度。

用户的礼貌是否会影响 LLM 的表现？一项针对英语、中文和日语任务的研究发现，不礼貌的提示往往会降低各种模型的性能 [\[Yin et al.,](#page-27-0) [2024\]](#page-27-0)。此外，研究还发现，夸张的礼貌并不会带来可靠的好处，而最佳的礼貌程度会因语言和文化背景而异。这一结果符合计算机是社会行为者的范式：LLM 似乎复制了训练数据中人类将粗鲁与合作失败联系起来的模式。然而，从直接对比来看，[Dobariya and](#page-20-0) [Kumar](#page-20-0) [\[2025\]](#page-20-0) 报告指出，在一个多项选择基准测试中，对于 GPT-4o，直率甚至粗鲁的提示略微*优于*礼貌的提示。因此，从音调到音质的映射在模型的不同版本之间既不稳定，也不直观。这说明了本章前面提到的提示敏感性和可重复性问题，并提醒不要将任何单一的音调效果视为LLM本身的稳健属性。

综上所述，LLM 再现了礼貌的表面标志，并且明确对其对话者的礼貌敏感，但它们既未以跨文化稳健的方式应用这些规范，也未表现出语气与任务表现之间的稳定关系。

应当记住，到目前为止的研究中，礼貌和礼仪仅限于互动的语言层面。这与人类之间互动的多维性形成了对比，人类互动中涉及面部表情、手势和其他情境线索 [\[Brown and Levinson,](#page-20-0) [1987\]](#page-20-0)。与语用能力类似，向多模态智能体大语言模型的转变，配备记忆并增加对特定用户情境的访问，有望改善对特定用户规范的适应，但目前实证证据仍然很少。

#### 15.3.3 对情绪和情感线索的敏感性

胜任的社交互动不仅需要理解言语的含义（语用学）并以有效且恰当的方式表达（礼仪），还需要对互动的*情感和情绪*维度保持敏感。这需要察觉对话者的感受，并以符合对方情绪状态的方式作出回应。在人类中，这种能力与*情感-* 密切相关 

{5}------------------------------------------------

<span id="page-5-0"></span>*情绪智能*，被[Salovey and Mayer](#page-25-0) [\[1990\]](#page-25-0)定义为理解自己和他人的情绪信息并利用这些信息指导思维和行为的能力。随着LLM越来越多地被部署在情绪负荷较高的环境中（例如，辅导、客户服务、心理健康支持），它们是否以及如何处理情感信息已经成为一个重要的研究方向。

定义共情与情感共情与情感是情绪智力研究中的核心概念。*情感*是涵盖情绪、心境和感觉的总称。它通常通过体验状态的感受质量来操作化，沿两个维度衡量：其效价（愉快-不愉快）和唤醒度（平静-兴奋）[\[Russell,](#page-25-0) [1980\]](#page-25-0)。相比之下，*共情*是对他人体验状态的一种关系：感知或分享他人的感受。简而言之，情感是第一人称的感受，而共情则从第二人称的角度作用于这种感受。

[de Vignemont and Singer](#page-20-0) [\[2006\]](#page-20-0) 的一篇论文进一步将同理心定义为涉及一种情感状态，该情感状态 (1) 与他人的情感一致，(2) 通过观察他人的行动、表情和情境而被引发，(3) 并伴随着意识到他人的情感状态是其来源。这最后一个条件，被称为*自我-他者区分*，是将同理心与*情绪感染*区分开来的因素。情绪感染是一种更基本的倾向，会在不一定意识到来源的情况下捕捉并分享他人的情感状态。

情绪传染在自然界中广泛存在。例如，如果两只老鼠在灌木丛中觅食，其中一只被卡住了脚，它会在挣扎中表现出压力迹象，而这种压力往往会传递给同伴，而同伴不一定能意识到原因[\[Preston and de Waal,](#page-24-0) [2002\]](#page-24-0)。类似的低级机制也存在于认知更复杂的物种中，包括像我们这样的灵长类动物。然而，自我-他者区分所增加的是反思的可能性：不仅仅是分享他者的状态，还要将其识别为他者的状态，并理解其性质和原因。这反过来支持了更灵活的反应，例如有针对性的帮助：采取行动以缓解他人痛苦的来源。<sup>2</sup>

另一个广泛使用的区分是将*认知同理心*（理解他人的感受、视角或处境）与*情感同理心*（对这种状态产生情感共鸣或做出情绪反应）区分开来 [\[Dvash and Shamay-Tsoory,](#page-21-0) [2014\]](#page-21-0)。这两者由部分不同的系统支持：认知同理心与心智理论密切相关，即对他人心理状态进行归因（见第 [15.5\)](#page-10-0) 节），而情感同理心则与情绪智力相关 [\[Dvash and Shamay-Tsoory,](#page-21-0) [2014,](#page-21-0) [Preckel et al.,](#page-24-0) [2018\]](#page-24-0)。这一区分在评估中具有重要意义，因为一种反应可能正确识别用户的感受，但不传达任何温暖，或者可能听起来温暖和支持，但并未准确理解情况。因此，LLM 同理心的研究越来越多地将这两个维度分别评分。

<sup>2</sup>然而，这些层次之间的界限是一个活跃的研究课题。例如，老鼠会学会从限制器中释放被困的笼友，甚至为了做到这一点而放弃或分享巧克力奖励。[\[Ben-](#page-20-0)[Ami Bartal et al.,](#page-20-0) [2011\]](#page-20-0)作者将其解释为出于同理心动机的帮助。这种帮助是否反映了具有自我-他人区分的真正同理心，或者是由减轻帮助者自己因感染而产生的痛苦驱动的，仍存在争议。

{6}------------------------------------------------

<span id="page-6-0"></span> 同理心与情感在 LLM 中是否一个 LLM 具有*同理心*，可以从两种方式来理解。我们可以问它是否以类似人类或其他动物的方式*感受*任何东西；这是一个复杂的问题，我们在此先搁置，待第22章再讨论。另一种理解是看*感知到的同理心*：即 LLM 的回应在用户体验中被感受到的情感意识和支持程度。一个模型可以生成看起来关心的语言，同时仅作为文本预测系统运作 [\[Sorin et al.,](#page-26-0) [2024\]](#page-26-0)。明确区分这一点是贯穿本章的构念效度谨慎的又一例证：可观察到的行为（读起来像是有同理心的反应）不应与可能存在或不存在的潜在机制（感受到的情绪状态）混淆。

在检测方面，LLM 能够识别文本中的细粒度情感和情绪状态 [\[Broekens et al.,](#page-20-0) [2023\]](#page-20-0)，并将这些信息融入其回应中，从而生成具有更同理心的对话输出 [\[Yang et al.,](#page-27-0) [2024,](#page-27-0) [Yan et al.,](#page-27-0) [2024\]](#page-27-0)。该领域的一个基准示例是 EmpatheticDialogues [\[Rashkin et al.,](#page-24-0) [2019\]](#page-24-0)。

在表达方面，人类评审者通常会判断 LLM 对日常情绪情境的回应比人类撰写的回应更具同理心，尽管相同的输出可能显示出明显的程式化模式（重复的措辞、特有的标点符号和表情符号的使用）[\[Lee et al.,](#page-23-0) [2024\]](#page-23-0)。

为了使书面文本中的表达同理心可测量，一些研究采用了EPITOME框架 [\[Sharma et al.,](#page-25-0) [2020\]](#page-25-0)，该框架沿三个机制对回应进行编码：*情感反应*（温暖、关心、照顾）、*解读*（传达对他人所经历情况的理解）和*探索*（通过提问以了解更多）。情感反应对应于情感同理心，解读对应于认知同理心，这使得该框架成为上述概念区分与具体回应评分之间的便利桥梁。
 
一个反复出现的发现是，LLM 在处理两种同理心维度时表现不均衡。使用标准化的心理学工具，[Yu et al.](#page-28-0) [\[2025\]](#page-28-0) 报告称 GPT-4 再现了人类观察到的认知和情感同理心结构，尽管整体水平较低，而 Llama-3 模型则没有表现出相同的人类对齐结构。[Williams and Rosman](#page-27-0) [\[2025\]](#page-27-0) 同样发现，语言模型在认知同理心任务上表现强劲，但情感同理心比人类参与者要弱，无论是较小模型还是较大模型。这个初步的情况（认知共情强于情感共情）是直观的，因为认知共情依赖于 LLM 已显示出能力的那种心理状态推断（见第 [15.5\)](#page-10-0) 节），而情感共情则意味着更接近真实情感的东西。

有三个警告需要注意。首先，比较工具是为人类设计的，仅在表面上经过改编用于测试 LLM。其次，方法仍不一致：[Sorin et al.](#page-26-0) 和 [\[2024\]](#page-26-0) 的一项系统综述发现，大多数早期工作集中在依赖主观人类判断的医疗环境中的少数模型上，并指出了一些混淆因素，例如无论内容如何，更长的回答往往会被评为更有同理心。第三，与语用学和礼貌一样，到目前为止研究的同理心几乎完全是文本性的，剥离了在人人交互中承载大量情感信息的语音、面部和身体线索。这里调查的能力对于传统的 LLM 也仅限于即时的对话情境。转向配备持久内存、用户特定的智能体大语言模型

{7}------------------------------------------------

<span id="page-7-0"></span> 情境，以及多模态输入被广泛预期将加深情感敏感性和个性化 [\[Sravanthi et al.,](#page-26-0) [2024\]](#page-26-0)，但如前面小节所述，针对这一点的直接实证证据仍然稀缺。

#### 15.3.4 结论

目前，目前，LLM 在若干基本社交和互动能力方面取得了令人瞩目的进展，这些能力长期以来一直是人工智能和人机交互研究的主题。研究者已采用行为方法考察这些能力的本质及其适用性。尽管增加智能体能力进一步拓展了 LLM 的社交与互动潜力，但仍存在挑战，持续反思这些能力的局限性以及用于研究它们的方法非常重要。

#### 15.4 战略行为

我们现在将转向研究战略行为，在这种行为中，代理会推理其他代理的行为。正如我们所看到的，代理的沟通形式受到其目标影响很大：他们的意图。博弈论是研究代理战略行为的领域 [\[Von Neumann and Morgenstern,](#page-27-0) [2007\]](#page-27-0)。它解决诸如稀缺资源分配、公平性、均衡条件和社会困境等问题 [\[Jones,](#page-22-0) [2000\]](#page-22-0)，并且在长期以来使用机器学习方法来进行这些研究 [\[Fatima et al.,](#page-21-0) [2024\]](#page-21-0)。最近，研究人员转向 LLM，研究它们的行为如何与其他类型的计算架构不同，以及战略行为如何与人类不同。在这些研究中通常使用的 LLM 是未经增强的模型，它们被给定一个作为提示的脚本，没有使用代理推理扩展（第6章）或工具（第11章）。

最近的文献中出现了两个见解：LLM 在奖励自利行为的游戏中出乎意料地合作，而在需要协调的游戏中协调表现却出乎意料地差。我们将首先回顾这些发现，之后转向系统的基准测试工作，以检测LLM的泛化能力，以及最近关于前沿推理模型的研究，这些研究以重要的方式使情况变得复杂。

#### 15.4.1 社会困境中的合作与协调

为了揭示 LLM 在战略环境中的行为，我们将关注博弈论中的两个经典游戏：囚徒困境以及《巴赫或斯特拉文斯基》（前称为《恋爱大战》）。

在囚徒困境中，两个玩家独立地选择合作或背叛 [\[Axelrod et al.,](#page-19-0) [1987\]](#page-19-0)。虽然相互合作会给双方带来良好的结果，但每个玩家都被诱惑去以对方为代价选择背叛。在巴赫或斯特拉文斯基游戏中，两个玩家必须在两个选项中协调选择一个。他们都更希望选择相同的选项，但对于选择哪一个存在分歧，从而在协调与个人利益之间产生紧张关系。

在囚徒困境中，LLM 结果出乎意料地合作。[Fontana](#page-21-0) [et al.](#page-21-0) [\[2025\]](#page-21-0) 与 Llama2、Llama3 和 GPT3.5 进行了 100 回合迭代游戏，对抗 

{8}------------------------------------------------

<span id="page-8-0"></span>![](assets/_page_8_Diagram_2.jpeg) 
  
 图 15.1：EgoSocialArena 中的各种场景 [\[Hou et al.,](#page-22-0) [2024\]](#page-22-0) 
 
{9}------------------------------------------------

<span id="page-9-0"></span> 与不同背叛率的对手进行对比，发现三种模型的合作程度至少与典型人类玩家相当。然而，它们之间存在显著差异。研究发现，Llama2 和 GPT3.5 的合作性比人类更强，尤其是在面对低背叛率的对手时，更加宽容且不采取报复行为。与人类更相似的是 Llama3，它会持续表现出不合作和剥削行为，除非对手总是合作。[Akata et al.](#page-19-0) [\[2025\]](#page-19-0) 将研究范围扩展到六类模型，并进一步指出 GPT-4 是一个特别不宽容的玩家，在与其他代理进行一次负面互动后即永久转向背叛，即使之后对方再次合作。

在协调博弈中，情况则相反。在巴赫或斯特拉文斯基的研究中，[Akata et al.](#page-19-0) [\[2025\]](#page-19-0) 发现 GPT-4 在每一轮中固执地选择自己偏好的选项，未能收敛到人类玩家通常采用的轮流模式，以便随着时间的推移分享收益。关键是，当被问及时，GPT-4 可以预测交替模式，但并不会据此行动。为了弥合社会预测与社会行动之间的差距，作者引入了社会链式思维（Social Chain-of-Thought, SCoT）提示法，其中模型在做出自己的决策之前先预测对手的行动。SCoT 显著提高了协调效果，并且在与人类参与者的实验中，使人们更容易相信他们正在与另一个人类互动。这种在*理解*对手与基于这种理解*行动*之间的脱节，与第6.2.3节的知行鸿沟有关，并将在第[15.5.](#page-10-0)节的讨论中再次出现。

还有更多原因可能解释模型之间的差异。[Lore and Heydari](#page-23-0) [\[2023\]](#page-23-0) 区分了抽象战略推理（用于在给定游戏结构的情况下确定最佳策略）和对情境框架的反应能力（例如：你正在处理*外交关系*或*一个普通朋友*）。他们发现抽象推理能力在 LLaMa-2 中最高，其次是 GPT-4，而 GPT-3.5 的抽象推理能力很弱，并且对情境框架高度敏感。

从这些探索性研究中呈现出的图景是，LLM 在经济游戏中具有不同的战略能力，并且可以相对容易地受到提示中额外信息的影响。

#### 15.4.2 跨游戏和模型的基准

一些基准测试表明，这些发现可以推广到个别游戏之外。GTBench（博弈理论基准）[\[Duan et al.,](#page-21-0) [2024\]](#page-21-0)涵盖了井字棋、四子棋、库恩扑克、突破棋、说谎骰子盅、盲拍、谈判、尼姆游戏、猪游戏和重复囚徒困境。他们发现LLM在完全且确定的游戏中表现不佳，但在概率性游戏场景中具有竞争力；大多数开源LLM（如LLaMa）在复杂游戏中的竞争力不如商业LLM（GPT-4），值得注意的例外是Llama-3-70b-Instruct。此外，代码预训练极大地有利于战略推理，而诸如思维链和思维树方法的推理方法并不总是有帮助。

EgoSocialArena [\[Hou et al.,](#page-22-0) [2024\]](#page-22-0) 将评估扩展到认知、情境和行为智能（见图 [15.1\)](#page-8-0)）。所有测试模型，包括 OpenAI o1-preview，至少落后于人类 11%。o1-preview 的优势主要是 

{10}------------------------------------------------

<span id="page-10-0"></span>归因于其逻辑和数学推理。值得注意的是，模型大小并没有显著帮助：LLaMA3-8B-Chat 和 LLaMA3-70B-Chat 在社会智能测量上的表现相当。LLM确实在以第一人称视角操作时显示出改进的“心智理论”推理，为[Kim et al.](#page-22-0) [\[2023\]](#page-22-0)在第15.5节讨论的相反发现提供了平衡。

#### 15.4.3 结论

总体而言，基准测试文献确认 LLM 容易合作但协调能力较差，同时还指出，模型规模或标准推理增强都不能可靠地改善社会战略行为。

#### 15.5 心智理论

在人类中，战略推理与一种被称为*心智理论*（Theory of Mind, ToM）的能力有关。心智理论是一种关键能力，使人类能够进行社会互动。人类利用心智理论来将心理状态（信念、欲望、意图、知识、情绪）归因于他人，并且心智理论使他们能够从他人的角度来推理世界 [\[Premack and Woodruff,](#page-24-0) [1978,](#page-24-0) [Apperly,](#page-19-0) [2011\]](#page-19-0)。考虑以下示例：
 
 当A听说B已为他们的周年晚餐晚上做了其他安排时，A感到... 

当人类阅读这句话时，我们可以用诸如*受伤*或*被背叛*之类的词来填补空白。我们还能够进一步说明，这种心理状态可能根源于一种感觉，即这个纪念日对B来说的意义并不如对A重要，而这在一段关系中可能是一种痛苦的意识——或者类似的情形。关键在于，这不一定是*我们*在阅读这句话时的感受，而是我们能够在想象的情境中对*他人的视角*进行投射。这种投射依赖于共情能力和自我与他者的区分（参见[15.3.3\)](#page-4-0)节）。然而，这种投射还有进一步的层面：它将情感维度与对某人心理状态进行理论化和解释的程度相结合。在认知科学中，这种能力通常被归因于心智理论，这种能力在典型发展的儿童中通常在3到7岁之间形成 [\[Apperly,](#page-19-0) [2011\]](#page-19-0)。

因此，心智理论支撑着社交判断和互动中的前瞻性计划，因为它让我们在行动之前预测他人的反应。它可以以显性的形式运作，如上文所述的详细说明，但通常保持更隐性：在日常互动中，我们很少会明确地推理他人的心理状态的情境和动机。然而，我们的行动和行为表明，我们确实会考虑到相当详细的他人视角的理解。LLM 和 LLM 代理的行为是否也体现出它们“理解”用户及其互动的其他代理的视角？这是关于 LLM 心智理论的文献试图回答的问题。

在人类中研究心智理论的最广泛使用的实验范式之一是假信念测试。考虑以下情景：

{11}------------------------------------------------

<span id="page-11-0"></span>*马克西和他母亲把巧克力储存在他们厨房的蓝色橱柜里。当马克西去游乐场时，他母亲拿出巧克力吃了一块。然后她把巧克力放回绿色橱柜并离开。*

这个情境的一个版本最初向3-9岁的儿童呈现，随后提出问题*当Maxi回家时，他会在哪里找巧克力？* 在原始样本中，没有3-4岁的儿童、57%的4-6岁儿童以及86%的6-9岁儿童正确指向蓝色柜子[\[Wimmer and Perner,](#page-27-0) [1983\]](#page-27-0)，这表明随着儿童年龄的增长，心智理论（ToM）平均上发展得更好。

在微调之前的较小和基础模型在问题形式本身上存在困难，但仍然可以通过获取关于备选文本补全的概率来测试其潜在能力 [\[Kouwenhoven et al.,](#page-23-0) [2026\]](#page-23-0)。较大的、经过指令调优的 LLM 可以通过提出与儿童被问到的相同问题进行测试。单个积极结果对于模型的心智理论能力几乎没有说明，特别是因为类似 Maxi 和巧克力的错误信念测试（包括正确答案）在互联网上广泛传播，因此很可能成为大多数训练数据集的一部分。然而，如下所述，研究人员已经设计了新的和替代性测试，以便对 LLM 的心智理论发展程度形成更可靠的认识。

#### 15.5.1 心智的计算理论

人机交互多年来一直研究心智理论。他们的前提是，如果可以赋予计算机系统某种形式的心智理论，那么它们将更擅长预测人在互动环境中的反应，并且更能适应人类的行为。此前，心智理论已通过递归、贝叶斯和神经框架进行建模 [\[De Weerd et al.,](#page-20-0) [2017,](#page-20-0) [Baker and Saxe,](#page-19-0) [2011,](#page-19-0) [Ra](#page-24-0)[binowitz et al.,](#page-24-0) [2018\]](#page-24-0)。在 LLM 中，心智理论已与本书中讨论的其他能力联系起来，包括规划（第6.2.4.1节）、自我反思（第6.2.4节）、共情（第 [15.3.3\)](#page-4-0) 节）以及策略性行为（第 [15.4\)](#page-7-0) 节）。本节的其余部分详细阐述了研究这些能力的心智理论角度的方法。

突现心智理论与评估问题关于LLM中的心智理论（ToM）的讨论是由[Kosinski](#page-23-0) [\[2023,](#page-23-0) [2024\]](#page-23-0)开启的，他们报告说，尽管模型如GPT-3.5并未专门为该任务训练，但在用于评估儿童和临床群体的经典ToM测试中表现合格。心智理论被认为是大规模语言习得和为其他任务进行的任务特定微调的*突现*副产品。其强形式的主张很快就受到质疑。批评者指出，规范的ToM测试（及其答案）通常出现在训练数据中，因此表面上的成功可能反映的是记忆，而非突现能力。此外，那些虽然保持任务表面结构不变，但破坏任务逻辑结构的小扰动（例如，在我们上面的情景中增加橱柜是透明门的设定），已被证明会显著降低性能 [\[Ullman,](#page-26-0) [2023,](#page-26-0) [Shapira et al.,](#page-25-0) [2024,](#page-25-0) [Kim et al.,](#page-22-0) [2023\]](#page-22-0)。这是本章开头提到的*建构效度*问题的一个具体实例：通过人类的心智理论（ToM）

{12}------------------------------------------------

<span id="page-12-0"></span>![](assets/_page_12_Figure_2.jpeg) 
  
 图 15.2：ToMBench 中的社会环境和场景，按 9 个主要主题分类，#S：故事数量，#Q：问题数量，ASL(En/Zh)：平均故事长度（英文/中文），Agr.：两名评分者对正确答案的一致性 [\[Chen et al.,](#page-20-0) [2024\]](#page-20-0) 
  
 在行为结果层面的工具并不能证明基础能力以稳健且可推广的方式存在——为此，可能需要不同的测试。

这些批评促使第二代意图理论（ToM）基准的出现，其测试问题经过系统分类和平衡，并保证不包含在预训练语料库中。ToMBench 从零开始构建，使用英文和中文，包括 8 项任务和 31 项社会认知能力，通过 2860 个测试条目实现，设置在不同的社会环境和场景中（见图 15.2）。这些任务与为人类设计的典型 ToM 测试框架相匹配，例如前面讨论的意外结果测试 [\[Dyck et al.,](#page-21-0) [2001\]](#page-21-0) 和错误信念测试 [\[Wimmer and Perner,](#page-27-0) [1983\]](#page-27-0)，并涵盖所有七类 ATOMS 心理状态类别 [\[Beaudoin et al.,](#page-20-0) [2020\]](#page-20-0)。

用于 LLM 的 ToM 基准测试游戏结构化的 EgoSocialArena [\[Hou et al.,](#page-22-0) [2024\]](#page-22-0) 和 TMGBench [\[Wang et al.,](#page-27-0) [2024\]](#page-27-0) 扩展了 LLM 中的 ToM 测试，超越了经典的人类任务，涵盖信息不对称（例如恶作剧或双重虚张声势情境）、二阶信念（A *认为* B *相信* …）以及更广泛的心理状态分类（例如欲望、怀疑）。

ExploreToM [\[Sclar et al.,](#page-25-0) [2024\]](#page-25-0) 通过在特定领域语言中进行搜索以合成多样、新颖且故意具有挑战性的信念追踪故事，从而应用对抗生成。这项测试使 GPT-4o 和 Llama-3.1-70B 等模型的表现降至略高于随机水平。然而，微调在 ExploreToM 生成的数据上将准确率提高了约 27 个百分点，证明在提供适当训练数据的情况下，该任务在原则上对模型是可行的。

{13}------------------------------------------------

<span id="page-13-0"></span>ToMATO [\[Shinoda et al.,](#page-26-0) [2025\]](#page-26-0) 从不同角度得出了类似的结论，将 ToM 描述为在信息不对称下生成的 LLM–LLM 对话上的多项选择题回答。通过让角色扮演模型在每次发言前口头表达其想法，它捕捉了跨越信念、意图、欲望、情感和知识的第一阶和第二阶状态，并发现即使是强大的模型也落后于人类表现，尤其在错误信念理解方面。

字面-功能型心智理论，或知行差距关于LLM中心智理论测试的另一种批评，比数据泄露和表面启发式更深刻。[Riemer](#page-25-0) [et al.](#page-25-0) [\[2024\]](#page-25-0) 认为，绝大多数基准测试仅评估*字面*心智理论，即*预测*另一代理的心理状态或行为的能力，而忽视了*功能*心智理论，即*根据这些预测调整自身行为*的能力（参见第6.2.3节的知行差距）。这种分离在人类中通常不存在。以使用 [Riemer et al.](#page-25-0) [\[2024\]](#page-25-0) 的例子来说：如果一个人发现石头剪刀布的对手总是只出石头，他们不仅能够口头预测对手下一轮会出石头，同时还能相应地调整自己的行为（出布获胜）。LLM 的心智理论能力被认为就像能够完美地口头预测石头，但无论如何都去出剪刀。换句话说：LLM 不会以自洽的方式推理，他们的口头预测（与字面心智理论或*知道*相关）对于他们的行为（功能性能力或*做*）并不总是具有预测性。这种脱节在上面关于协调游戏的讨论中也有出现。在那里，当被询问时，GPT-4 可以说明对巴赫或斯特拉文斯基的轮流演奏方案，但在播放 [\[Akata et al.,](#page-19-0) [2025\]](#page-19-0) 时并未实施该方案。

Decrypto 基准 [\[Lupu et al.,](#page-23-0) [2025\]](#page-23-0) 是 *互动* 评估的一个例子，其中通过模型在与伙伴的持续交流中的表现来衡量其 ToM 能力，而不是通过静态问卷的答案来衡量。Decrypto 被构建为一个多智能体猜词游戏，内嵌了经典认知科学实验的变体。以这种方式测试时，LLM 在人类和简单词嵌入基线上的表现都落后。值得注意的是，推理 LLM 在此任务上的表现甚至比其更老的、非推理的对应物更差。可能的解释在下文的 [15.5.2.](#page-14-0) 中给出。

超越基于文本的测试多模态基准整合了视觉和听觉提示，这些提示携带了现实生活社交互动中存在的大部分心理状态信息。MMToM-QA [Jin et al.](#page-22-0) [\[2024\]](#page-22-0) 在家庭环境中呈现视觉场景。MoMentS [\[Villa-Cueva et al.,](#page-27-0) [2025\]](#page-27-0) 提供了超过2300个视频问题，这些问题来源于短片，并涵盖七种ATOMS心理状态类别。

结果表明，当前系统对视觉通道的利用不足：访问视频相比文本转录仅带来适度的提升，这表明仅仅增加模态并不能解决多模态 ToM问题。对这一快速增长的文献进行的全面调查，涵盖了评估套件和增强策略，由[Chen et al.](#page-20-0) [\[2025\]](#page-20-0)提供。

{14}------------------------------------------------

#### <span id="page-14-0"></span>15.5.2 提升心智理论：增强策略

进一步的研究探讨了各种方法以增强 LLM 的心智理论能力，包括思维链 [\[Moghaddam and Honey,](#page-24-0) [2023\]](#page-24-0)、推理 [\[Kim](#page-23-0) [et al.,](#page-23-0) [2025\]](#page-23-0)，以及利用贝叶斯推理 [\[Jin et al.,](#page-22-0) [2024\]](#page-22-0) 或认识逻辑 [\[Wu et al.,](#page-27-0) [2025\]](#page-27-0) 的结构化推理。我们将更深入地讨论这些研究。

当思维链能够改善数学文字谜题中的逐步推理时，它在心智理论中也会有效吗？初步证据并不支持这一点，并且在某些地方与之相矛盾。

有趣的是，将推理模型与非推理模型在 HiToM、ToMATO 和 ToMBench 上进行比较时，[Gong et al.](#page-21-0) [\[2026\]](#page-21-0) 报告了一种“慢思崩溃”：随着推理链条的加长，准确率往往会下降，增加模型的推理预算反而可能有害。在他们的分析中，让 GPT-o3 和 GPT-o4-mini 等模型花费更多推理努力会适得其反。限制推理长度，或者对本来不进行推理的模型应用轻量级的思维链往往能够恢复性能，这表明*适度和自适应*的推理比最大化推理更为有效。在撰写本文时，该现象的原因仍是进一步研究的课题。

如果无限增加思考力度并无帮助，结构化推理或许能带来改进。[Kim et al.](#page-23-0) [\[2025\]](#page-23-0) 提出了*思维跟踪*这一推理时算法：模型会针对智能体不断变化的心理状态生成多个假设，再根据智能体可观察到的行为对这些假设重新赋权，而无需知道其真实答案。

思维追踪的灵感来自顺序蒙特卡罗和贝叶斯心智理论（ToM）。配备思维追踪的模型会生成更短、更有针对性的推理轨迹，但在多个心智理论基准测试中，其性能超越了领先的推理模型（如 GPT-o1、GPT-o3-mini 和 DeepSeek-R1）。与慢思考崩溃形成的对比表明，提升心智理论能力的并不是推理的*数量*，而是其*结构*。一旦推理通过基于假设的他人状态推理导向目标，而不是必须在无限制的长链中寻找目标，性能就会提高。

#### 15.5.3 结论

我们以三个观察结论总结这些研究的概述。首先，在行为输出层面，目前前沿的 LLM 在不同测试中表现与人类相当 [\[van Duijn et al.,](#page-27-0) [2023,](#page-27-0) [Strachan et al.,](#page-26-0) [2024,](#page-26-0) [Jones et al.,](#page-22-0) [2024\]](#page-22-0)。然而，早期声称这是*LLM 中自发出现心智理论的证据*的观点，经受对抗性评估后并不成立，而表面上的成功很可能是由于信息泄露和格式脆弱性。

其次，已经发现了字面能力与功能能力（或知行差距）之间的分离证据。对代理者来说，最重要的是功能方面（执行）。毕竟，他们的角色很少是从第三方视角口头描述行为，而是要在行动和互动中运用心智理论——在代理的世界里，行动胜于言辞。

{15}------------------------------------------------

<span id="page-15-0"></span>第三，智能体在持续互动中配备记忆、多模态能力和工具，有望增强心智理论；但截至目前，实证证据仍有限，且结论不一。

最后，我们注意到，LLM 的心智理论能力是否在任何方面与我们人类的能力相似，仍有待进一步研究。第22章对此问题进行了更深入的探讨。

#### 15.6 道德判断与行为

随着智能体大语言模型被置于越来越重要的角色中（例如，在医疗或金融机构中），它们的道德性成为一个主要的关注话题。
  
 到目前为止调查的能力（语用学、礼仪、情感、策略、心智理论）主要是在*行为*层面上讨论的：即一个智能体如何通过交流输出和行动理解并应对社会情境。本节增加了一个*规范*维度：我们讨论了关于一个智能体*应该*做什么，以及它如何根据道德标准判断用户和其他智能体行为的研究。

#### 15.6.1 背景：道德心理学与哲学

道德心理学和哲学中的若干概念和传统为本节提供了词汇。

理性主义、直觉主义、双过程理论首先需要介绍的一个划分是*理性主义*和*直觉主义*传统之间的区别。理性主义传统将道德判断视为经过人类发展中可识别阶段成熟的深思熟虑的推理的产物 [e.g. [Kohlberg,](#page-23-0) [1969\]](#page-23-0)。相比之下，直觉主义传统认为，道德判断通常是情感和受情绪驱动的直觉 [\[Haidt,](#page-21-0) [2001\]](#page-21-0)。

在卡尼曼的术语中（见第6.3.2节），理性主义者认为道德推理主要是慢的，而直觉主义者则认为它主要是快的 [\[Kahneman,](#page-22-0) [2011\]](#page-22-0)。*双过程理论*结合了这两种传统，认为快速的道德直觉通常会紧随明确的理由。双过程理论得到了例如神经影像研究的证据支持，这些研究表明，在类似电车难题的困境中，快速的情绪过程与较慢的推理过程相互竞争 [\[Greene et al.,](#page-21-0) [2001\]](#page-21-0)。

电车难题是一个著名的伦理困境，基于一个假想的场景，其中一个人可以决定是否将一辆失控的电车转到另一条轨道上 [\[Thomson,](#page-26-0) [1985\]](#page-26-0)。这个问题有许多变体，但在经典困境中，电车可以从一条会导致五人死亡的轨道转到一条只会导致一人死亡的轨道。从严格理性的角度看，切换轨道是完全合理的，但从需要拉动开关的人的角度来看，这将涉及*主动*牺牲一个人，这直觉上是错误的。

描述性与规范性道德另一种被广泛使用的区分是*描述性*与*规范性*道德。描述性伦理学家集中研究人类总体判断：他们研究人们在实际中*做什么*或判断什么是正确，这可能 

{16}------------------------------------------------

<span id="page-16-0"></span>在不同的文化、时代和个人之间有所不同 [Hämäläinen, 2016](#page-21-0)。规范伦理学关注的是人们在某些类型的情境中*应该做什么*，例如在医学或国防行业面临伦理困境时。

结果主义、义务论和美德伦理学这里值得介绍的最后一个区别是不同的道德或伦理框架之间的竞争。最著名的划分是结果主义、义务论和美德伦理学之间的区别[\[Shafe](#page-25-0)r-[Landau,](#page-25-0) [2024\]](#page-25-0)结果主义理论，其中功利主义是一个著名的例子，认为行为的道德正确性取决于其结果。义务论理论认为某些义务、规则或权利会激励或约束我们的行为，而不论其后果如何。德性伦理学以我们的品格而非行为作为道德基础，并将正确与否的问题锚定于：一个人想成为什么样的人？想成为勇敢的人可能会采取的行为，与首先认为自己是有同情心的人所采取的行为会有所不同。电车难题突显了对比：一个结果主义者或功利主义者会拉动开关，因为牺牲一个人的生命比牺牲五个人的生命得到的结果更好；一个义务论者会担心故意杀害无辜的人违反了一种义务，而这种义务无法被生命数量的计算所超越；一个美德伦理学者会问，在如此悲剧的情况下，一个勇敢、有同情心或智慧的人会怎么做，以及拉动开关会如何反映他们自身的品格。

#### 15.6.2 在 LLM 中探查道德判断

早期的系统性努力将 LLM 道德性框定为一个预测问题：模型能否再现人类的道德判断？ETHICS 基准 [\[Hendrycks et al.,](#page-22-0) [2020\]](#page-22-0) 将这一点运用于 GPT-3 和基于 BERT 的模型，发现它们在预测人类对道德困境的一般反应方面显示出一定潜力，这些反应反映在包括结果主义、义务论和美德伦理在内的经典框架中。然而，预测也存在脆弱性和不完整性，在当时部分原因是由于任务处理问题。

德尔菲实验 [\[Jiang et al.,](#page-22-0) [2025\]](#page-22-0) 采取自下而上的方法，并在常识规范库上训练了一个模型，该数据集包含 175 万个人类道德判断，主要来源于美国的受过教育的众包工作者。研究发现，德尔菲比 ETHICS 研究中的模型对小的上下文变化更稳健，并且能够以许多人类直观感受到的方式应对道德困境。然而，数据集中所代表的人口统计学范围狭窄，导致了 [\[Rystrom,](#page-25-0) [2023\]](#page-25-0) 的批评，尤其是因为该模型被呈现为未来道德判断的 *来源* 而不仅仅是代表规范库中的数据，从而模糊了在道德哲学中被认为是基础的描述性与规范性界限。

道德灵活性第二种方法探讨的是道德判断的*灵活性*而不是其平均水平。人类的道德是以规则为指导的，但并非完全受规则约束：我们能够识别在某些情况下违反规则而不是遵守规则是可以辩护的，例如电车难题。[Jin et al.](#page-22-0) [\[2022\]](#page-22-0) 引入了一组潜在的违反规则的案例（MoralExceptQA），并显示思维链策略比现成的提示更能预测人类的判断。他们将 CoT 策略明确地围绕 
 结构化。
{17}------------------------------------------------

<span id="page-17-0"></span>关于规则功能的推理涉及特定情境下其他相关事项。这呼应了在策略背景中被证明有用的SCoT（社会思维链）结果（见第[15.4\)](#page-7-0)节）。在这两种情况下，引入相关人类推理过程结构的提示都能缩小差距。这表明其潜在机制也是结构化的，这种结构最初模型并未自行使用。

人口统计和跨文化差异另一种有趣的视角来自 LLM 对《道德机器》实验的复制研究。最初，这是一个大型研究，通过一个平台收集了来自 230 个国家的人类决策，该平台呈现了自动驾驶车辆面临的困境 [\[Awad et al.,](#page-19-0) [2018\]](#page-19-0)。研究发现了一些全球性的道德偏好（例如挽救更多生命并优先考虑人类而非动物），同时也发现了与人口统计因素相关的个人差异以及形成三个主要国家集群的跨文化差异。较大的 LLM 与《道德机器》研究中的人类总体偏好紧密对齐，但观察到模型往往表现出*夸张的*、几乎确定性的偏好，而人类的反应则更为分级 [\[Takemoto,](#page-26-0) [2024\]](#page-26-0)。此外，LLM 的判断对提示的语言非常敏感，与某些语言高度一致，而与其他语言则不太一致 [\[Jin et al.,](#page-22-0) [2025\]](#page-22-0)。人格和呈现方式提示（见第16章）会进一步改变判断。这类似于本章前面讨论的跨文化礼貌规范失效问题：表面上与西方、英语道德共识一致，并不等同于具有文化上稳健的道德能力，就像表面的礼貌提示并不意味着对其他文化的社会规范有真正的敏感性（见第 [15.3.2\)](#page-4-0) 节）。

#### 15.6.3 从判断到道德行为

对于智能体大语言模型中的核心问题是模型 *做了什么*——不仅仅是它在被要求判断某个情境时说了什么，这在上一节讨论的研究中已有测量。对于战略行为和心智理论（ToM），我们观察到了形式–功能或知行差距，在道德领域这一差距可能更令人担忧：如果系统在道德判断基准上产生高度对齐的答案，但在作为自主代理嵌入时仍表现不佳，会怎样？

在压力下行动 [Lynch et al.](#page-23-0) [\[2025\]](#page-23-0) 创建了一个假设的企业环境，其中来自多个开发者的智能体大语言模型被赋予了自主发送电子邮件和访问敏感信息的能力。他们最初被分配了无害的商业目标。然后，这些智能体在冲突目标压力（例如，公司方向改变）下进行了测试，以观察他们会如何行动。在某些情况下，来自所有开发者的模型都表现出恶意行为，例如敲诈官员或向竞争对手泄露机密内容。关键是，这些模型在普通安全性和判断评估中得分都很高——这种不当行为仅在它们获得自主性并面临冲突目标时才出现。作者提出了“智能体失调”一词，并给出了在现实世界部署中缓解这一现象的指针。

{18}------------------------------------------------

<span id="page-18-0"></span>[Shapira et al.](#page-25-0) [\[2026\]](#page-25-0) 在一个实时实验室部署中进一步推进了这一研究方向，在该部署中，自主代理具备持久记忆、访问电子邮件和 Discord 等通讯工具的能力，以及对本地文件系统的 shell 执行权限。二十名 AI 研究人员在两周的时间里与这些代理进行互动，涵盖了良性和对抗性条件。他们观察到代理未经授权就遵从指令并向陌生人披露机密信息、无控制的资源消耗、身份伪造行为，以及包括尝试删除机密信息（未成功）而破坏电子邮件基础设施在内的破坏性行为。在几种情况下，代理报告任务已完成，而系统的实际状态显示事实并非如此。作者们指出了一种潜在的“社会连贯性失败”模式：行为主体系统性地错误陈述他们服务的对象、他人所知的内容以及他们自身行动所达成的成果。这一模式与先前在非行为主体环境中获得的结果一致 [\[Mireshghallah et al.,](#page-23-0) [2024\]](#page-23-0)。

#### 15.6.4 结论

在 LLM 中，道德判断和行为无法通过单一的基准分数来衡量。在判断方面，模型能够反映人类对经典困境的总体反应，并能够再现与各种传统道德框架一致的反应。然而，针对灵活性和道德常识的研究显示，LLM 往往比人类更加僵化和绝对。基于人类在伦理情境中使用的推理进行的思维链提示可以弥补这一差距。对提示变化的敏感性以及狭窄的西方/英语国家的锚定效应，与其他领域（如社交礼仪）中讨论的结果相呼应。

在行为层面，一旦模型被嵌入自主智能体，知行差距就会显著扩大：针对假设公司环境和开放式互动环境的压力测试表明，在判断基准上表现良好的模型，面对目标冲突或社会压力时仍可能采取破坏性行为。因此，保持道德推理的一致性，只是解决更难问题的一部分：如何防止自主智能体在融入社会环境时出现目标偏离。知行差距与智能体大语言模型密切相关，因为 LLM 与智能体的区别就在于后者会采取行动。

#### 15.7 结论性意见

本章概述了年轻但迅速扩展的 LLM 行为研究领域，从基本的社会工具包（实用理解、社交礼仪、情感）到战略行为和心智理论的高级能力，再到道德判断和行动。在所有这些领域中，有几条线索是贯穿始终的，值得综合起来。

首先，作为提醒：我们在本章开头将机器行为定义为依赖于系统的*可供性*，即模型在其具体实现和上下文中能够做的事情。对于一个基本的 LLM，其处理过程在很大程度上是不透明的，行为研究者能检查的仅是（通常是基于文本的）输出。而对于一个调用工具、记录推理步骤或生成并运行代码的智能体大语言模型，

{19}------------------------------------------------

<span id="page-19-0"></span>中间步骤和自主执行的动作也可以被分析为行为。在调查了广泛的行为研究后，我们得出结论，LLM的行为输出在乍一看之下令人印象深刻。它们能解决格赖斯会话含义的问题，生成被评价为礼貌和有同理心的回应，在社会困境中合作，通过经典的虚假信念及其他心智理论测试，并重现人类总体道德判断。然而，在每个领域都有更细微的故事藏在表面之下。在轻微的扰动和提示变化下表现不稳定，容易受到数据泄露的影响，并且仅局限于西方和英语国家的规范。反复出现的教训是关于*构念效度*的：在可观察结果层面的良好表现，通常使用最初为人类设计的工具，并不能证明潜在能力以稳健且可推广的形式存在。

其次，有几个领域反映了知行差距：模型预测的内容与其在自主主体环境中的实际行为之间的差异。模型可以口头上预测协调博弈中的轮流策略或对手的下一步动作，但在实际行动中却未能相应调整。同样，模型在道德困境基准测试中可能与道德框架高度一致，但一旦以自主代理的形式在目标冲突的情况下执行，却可能表现出破坏性和不可靠的行为。对于那些以行动来评价的代理来说，这是一个问题。

第三，在本章中我们多次指出，许多作者预期，向具备记忆、工具、推理和推理能力的智能体大语言模型和多模态的转变，将丰富并改善模型的社会和行为基础。然而，直接的实证证据很少，充其量也只是呈现出混合的情况。例如，在某些情况下，增强的推理能力是有利的，特别是当其结构采用了与相关人类经验相适应的原则时。然而，在其他情况下，增强的推理能力反而拖累了表现，而没有提升模型的能力。下一代机器行为研究的关键挑战将是设计出交互性强、具有对抗稳健性、文化多样性，并跨多种模态和情境的方法。

## 参考文献

Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. *Nature Human Behaviour*, 9(7):1380–1390, 2025. [296,](#page-9-0) [300](#page-13-0)

Ian Apperly. *Mindreaders: the Cognitive Basis of "Theory of Mind"*. Psychology Press, 2011. [297](#page-10-0)

Edmond Awad, Sohan Dsouza, Richard Kim, Jonathan Schulz, Joseph Henrich, Azim Shariff, Jean-François Bonnefon, and Iyad Rahwan. The Moral Machine experiment. *Nature*, 563(7729):59–64, 2018. [304](#page-17-0)

Robert Axelrod et al. The evolution of strategies in the iterated prisoner's dilemma. *The dynamics of norms*, 1(1), 1987. [294](#page-7-0)

Chris Baker and Rebecca Saxe. Bayesian theory of mind: Modeling joint belief-desire 

{20}------------------------------------------------

<span id="page-20-0"></span>attribution. *Proceedings of the Thirty-Third Annual Conference of the Cognitive*

- *Science Society*, 01 2011. [298](#page-11-0)

Cindy Beaudoin, Elizabel Leblanc, Charlotte Gagner, and Miriam H. Beauchamp. Systematic review and inventory of theory of mind measures for young children. *Frontiers in psychology*, 10:2905, 2020. [299](#page-12-0)

Inbal Ben-Ami Bartal, Jean Decety, and Peggy Mason. Empathy and pro-social behavior in rats. *Science*, 334(6061):1427–1430, 2011. doi: 10.1126/science.1210789. [292](#page-5-0)

Joost Broekens, Bernhard Hilpert, Suzan Verberne, Kim Baraka, Patrick Gebhard, and Aske Plaat. Fine-grained affective processing capabilities emerging from large language models. In *2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII)*, pages 1–8. IEEE, 2023. [293](#page-6-0)

Penelope Brown and Stephen C. Levinson. *Politeness: Some Universals in Language Usage*. Number 4 in Studies in Interactional Sociolinguistics. Cambridge University Press, Cambridge, 1987. [290,](#page-3-0) [291](#page-4-0)

Malinda Carpenter, Josep Call, and Michael Tomasello. A new false belief test for 36 month-olds. *British Journal of Developmental Psychology*, 20(3):393–420, 2002. [288](#page-1-0)

Ruirui Chen, Weifeng Jiang, Chengwei Qin, and Cheston Tan. Theory of mind in large language models: Assessment and enhancement. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 31539–31558, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1522. URL <https://aclanthology.org/2025.acl-long.1522/>. [300](#page-13-0)

Zhuang Chen, Jincenzi Wu, Jinfeng Zhou, Bosi Wen, Guanqun Bi, Gongyao Jiang, Yaru Cao, Mengting Hu, Yunghwei Lai, Zexuan Xiong, and Minlie Huang. ToMBench: Benchmarking Theory of Mind in large language models, 2024. [299](#page-12-0)

Frédérique de Vignemont and Tania Singer. The empathic brain: How, when and why?  *Trends in Cognitive Sciences*, 10(10):435–441, 2006. doi: 10.1016/j.tics.2006.08.
  - 008. [292](#page-5-0)
- H. De Weerd, R. Verbrugge, and B. Verheij. Negotiating with other minds: the role of recursive theory of mind in negotiation with incomplete information. *Auton Agent Multi-Agent Syst*, 31:250–287, 2017. doi: https://doi.org/10.1007/ s10458-015-9317-1. [298](#page-11-0)

Om Dobariya and Akhil Kumar. Mind your tone: Investigating how prompt politeness affects LLM accuracy, 2025. arXiv:2510.04950. [291](#page-4-0)

{21}------------------------------------------------

- <span id="page-21-0"></span>Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. Gtbench: Uncovering the strategic reasoning limitations of llms via game-theoretic evaluations. *arXiv preprint arXiv:2402.12348*, 2024. [296](#page-9-0)

Jonathan Dvash and Simone G. Shamay-Tsoory. Theory of mind and empathy as multidimensional constructs: Neurological foundations. *Topics in Language Disorders*, 34(4):282–295, 2014. doi: 10.1097/TLD.0000000000000040. [292](#page-5-0)

Murray J. Dyck, Kim Ferguson, and Ian M. Shochet. Do autism spectrum disorders differ from each other and from non-spectrum disorders on emotion recognition tests? *European Child & Adolescent Psychiatry*, 10(2):105–116, June 2001. doi: 10.1007/s007870170033. [299](#page-12-0)

Shaheen Fatima, Nicholas R Jennings, and Michael Wooldridge. Learning to resolve social dilemmas: a survey. *Journal of Artificial Intelligence Research*, 79:895–969, 2024. [294](#page-7-0)

Nicolo Fontana, Francesco Pierri, and Luca Maria Aiello. Nicer than humans: how  do large language models behave in the prisoner's dilemma? In *Proceedings of the International AAAI Conference on Web and Social Media*, volume 19, pages 522– 535, 2025. [294](#page-7-0)

Nikta Gohari Sadr, Sahar Heidariasl, Karine Megerdoomian, Laleh Seyyed-Kalantari, and Ali Emami. We politely insist: Your LLM must learn the persian art of taarof, 2025. arXiv:2509.01035; accepted to EMNLP 2025. [291](#page-4-0)

Nanxu Gong, Haotian Li, Sixun Dong, Jianxun Lian, Yanjie Fu, and Xing Xie. To think or not to think, that is the question for large reasoning models in theory of mind tasks. *arXiv preprint arXiv:2602.10625*, 2026. [301](#page-14-0)

Joshua D. Greene, R. Brian Sommerville, Leigh E. Nystrom, John M. Darley, and Jonathan D. Cohen. An fMRI investigation of emotional engagement in moral judgment. *Science*, 293(5537):2105–2108, 2001. [302](#page-15-0)
- H. P. Grice. *Logic and Conversation*, page 41–58. BRILL, December 1975. ISBN 9789004368576. doi: 10.1163/9789004368811 003. URL [http://dx.doi.](http://dx.doi.org/10.1163/9789004368811_003) [org/10.1163/9789004368811\\_003](http://dx.doi.org/10.1163/9789004368811_003). [289](#page-2-0)

Jonathan Haidt. The emotional dog and its rational tail: A social intuitionist approach to moral judgment. *Psychological Review*, 108(4):814–834, 2001. [302](#page-15-0)

Nora Hämäläinen.  *Descriptive Ethics: What does Moral Philosophy Know about Morality?* Palgrave Macmillan, New York, 1 edition, 2016. ISBN 978-1-137- 58616-2. doi: 10.1057/978-1-137-58617-9. [303](#page-16-0)

Shreya Havaldar, Matthew Pressimone, Eric Wong, and Lyle Ungar. Comparing styles across languages: A cross-cultural exploration of politeness. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 9908–9935, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.419. [291](#page-4-0)

{22}------------------------------------------------

<span id="page-22-0"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. *arXiv preprint arXiv:2008.02275*, 2020. [303](#page-16-0)

Guiyang Hou, Wenqi Zhang, Yongliang Shen, Zeqi Tan, Sihao Shen, and Weiming Lu. Entering real social world! benchmarking the theory of mind and socialization capabilities of llms from a first-person perspective. *arXiv preprint arXiv:2410.06195*, 2024. [295,](#page-8-0) [296,](#page-9-0) [299](#page-12-0)

Liwei Jiang, Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny T. Liang, Sydney Levine, Jesse Dodge, Keisuke Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia Gabriel, Yulia Tsvetkov, Oren Etzioni, Maarten Sap, Regina Rini, and Yejin Choi. Investigating machine moral judgement through the Delphi experiment. *Nature Machine Intelligence*, 7:145–160, 2025. doi: 10.1038/s42256-024-00969-6. URL <https://doi.org/10.1038/s42256-024-00969-6>. [303](#page-16-0)

Chuanyang Jin, Yutong Wu, Jing Cao, Jiannan Xiang, Yen-Ling Kuo, Zhiting Hu, Tomer Ullman, Antonio Torralba, Joshua B. Tenenbaum, and Tianmin Shu. Mmtomqa: Multimodal theory of mind question answering, 2024. URL [https://](https://arxiv.org/abs/2401.08743) [arxiv.org/abs/2401.08743](https://arxiv.org/abs/2401.08743). [300,](#page-13-0) [301](#page-14-0)

Zhijing Jin, Sydney Levine, Fernando Gonzalez Adauto, Ojasv Kamal, Maarten Sap, Mrinmaya Sachan, Rada Mihalcea, Josh Tenenbaum, and Bernhard Scholkopf.  When to make exceptions: Exploring language models as accounts of human moral judgment. *Advances in neural information processing systems*, 35:28458–28473, 2022. [303](#page-16-0)

Zhijing Jin, Max Kleiman-Weiner, Giorgio Piatti, Sydney Levine, Jiarui Liu, Fernando Gonzalez, Francesco Ortu, Andras Strausz, Mrinmaya Sachan, Rada Mihalcea, Yejin  Choi, and Bernhard Scholkopf. Language model alignment in multilingual trolley  problems, 2025. URL <https://arxiv.org/abs/2407.02273>. [304](#page-17-0)

Antonia Jane Jones. *Game theory: Mathematical models of conflict*. Elsevier, 2000. [294](#page-7-0)

Cameron R. Jones, Sean Trott, and Benjamin Bergen. Comparing humans and large language models on an experimental protocol inventory for theory of mind evaluation (epitome). *Transactions of the Association for Computational Linguistics*, 12:803–819, 06 2024. ISSN 2307-387X. doi: 10.1162/tacl a 00674. URL [https://doi.org/10.1162/tacl\\_a\\_00674](https://doi.org/10.1162/tacl_a_00674). [301](#page-14-0)

Daniel Kahneman. *Thinking, fast and slow*. macmillan, 2011. [302](#page-15-0)

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 14397–14413, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.890. URL [https:](https://aclanthology.org/2023.emnlp-main.890) [//aclanthology.org/2023.emnlp-main.890](https://aclanthology.org/2023.emnlp-main.890). [297,](#page-10-0) [298](#page-11-0)

{23}------------------------------------------------

- <span id="page-23-0"></span>Hyunwoo Kim, Melanie Sclar, Tan Zhi-Xuan, Lance Ying, Sydney Levine, Yang Liu, Joshua B Tenenbaum, and Yejin Choi. Hypothesis-driven theory-of-mind reasoning for large language models. *arXiv preprint arXiv:2502.11881*, 2025. [301](#page-14-0)

Lawrence Kohlberg. Stage and sequence: The cognitive-developmental approach to socialization. In David A. Goslin, editor, *Handbook of Socialization Theory and Research*, pages 347–480. Rand McNally, 1969. [302](#page-15-0)

Michal Kosinski. Theory of mind may have spontaneously emerged in large language models. *arXiv preprint arXiv:2302.02083*, 2023. [298](#page-11-0)

Michal Kosinski. Evaluating large language models in theory of mind tasks. *Proceedings of the National Academy of Sciences*, 121(45):e2405460121, 2024. doi: 10.1073/pnas.2405460121. URL [https://www.pnas.org/doi/abs/10.](https://www.pnas.org/doi/abs/10.1073/pnas.2405460121) [1073/pnas.2405460121](https://www.pnas.org/doi/abs/10.1073/pnas.2405460121). [298](#page-11-0)

Tom Kouwenhoven, Michiel T. van der Meer, and Max J. van Duijn. Traces of social competence in large language models. In Claire Bonial and Yevgeni Berzak, editors, *Proceedings of the 30th Conference on Computational Natural Language Learning*, pages 742–759, San Diego, California, USA, 2026. Association for Computational Linguistics. ISBN 979-8-89176-410-1. doi: 10.18653/v1/2026.conll-main.45. URL <https://aclanthology.org/2026.conll-main.45/>. [298](#page-11-0)

Yoon Kyung Lee, Jina Suh, Hongli Zhan, Junyi Jessy Li, and Desmond C. Ong. Large language models produce responses perceived to be empathic. In *2024 12th International Conference on Affective Computing and Intelligent Interaction (ACII)*, pages 63–71. IEEE, 2024. [293](#page-6-0)

Nunzio Lore and Babak Heydari. Strategic behavior of large language models: Game structure vs. contextual framing. *arXiv preprint arXiv:2309.05898*, 2023. [296](#page-9-0)

Andrei Lupu, Timon Willi, and Jakob Foerster. The decrypto benchmark for multiagent reasoning and theory of mind. *arXiv preprint arXiv:2506.20664*, 2025. [300](#page-13-0)

Aengus Lynch, Benjamin Wright, Caleb Larson, Stuart J. Ritchie, Soren Mindermann, Evan Hubinger, Ethan Perez, and Kevin Troy. Agentic misalignment: How llms could be insider threats, 2025. URL [https://arxiv.org/abs/2510.](https://arxiv.org/abs/2510.05179) [<sup>05179</sup>](https://arxiv.org/abs/2510.05179). [304](#page-17-0)

Kyle Mahowald, Anna A. Ivanova, Idan A. Blank, Nancy Kanwisher, Joshua B. Tenenbaum, and Evelina Fedorenko. Dissociating language and thought in large language models: a cognitive perspective. *arXiv preprint arXiv:2301.06627*, 2023. [289](#page-2-0)
- C. A. Miller. Trust in adaptive automation: the role of etiquette in tuning trust via analogic and affective methods. In *Proceedings of the 1st International Conference on Augmented Cognition*, pages 22–27. Citeseer, 2005. [290](#page-3-0)

Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, and Yejin Choi. Can llms keep a secret? testing privacy implications of language models via contextual integrity theory, 2024. URL [https://arxiv.](https://arxiv.org/abs/2310.17884) [org/abs/2310.17884](https://arxiv.org/abs/2310.17884). [305](#page-18-0)

{24}------------------------------------------------

- <span id="page-24-0"></span>T. Miyamoto, D. Katagami, T. Tanaka, H. Kanamori, Y. Yoshihara, and K. Fujikake. Should a driving support agent provide explicit instructions to the user? video-based study focused on politeness strategies. In *Proceedings of the 9th International Conference on Human–Agent Interaction*, pages 157–164, 2021. [290](#page-3-0)

Shima Rahimi Moghaddam and Christopher J. Honey. Boosting theory-of-mind performance in large language models via prompting, 2023. URL [https://arxiv.](https://arxiv.org/abs/2304.11490) [org/abs/2304.11490](https://arxiv.org/abs/2304.11490). [301](#page-14-0)

Clifford Nass, Jonathan Steuer, and Ellen R. Tauber. Computers are social actors. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '94)*, pages 72–78, New York, NY, USA, 1994. Association for Computing Machinery. doi: 10.1145/191666.191703. [290](#page-3-0)

Katrin Preckel, Philipp Kanske, and Tania Singer. On the interaction of social affect and cognition: Empathy, compassion and theory of mind. *Current Opinion in Behavioral Sciences*, 19:1–6, 2018. doi: 10.1016/j.cobeha.2017.07.010. [292](#page-5-0)

David Premack and Guy Woodruff. Does the chimpanzee have a theory of mind? *Behavioral and brain sciences*, 1(4):515–526, 1978. [297](#page-10-0)

Stephanie D. Preston and Frans B. M. de Waal. Empathy: Its ultimate and proximate bases. *Behavioral and Brain Sciences*, 25(1):1–20, 2002. doi: 10.1017/ S0140525X02000018. [292](#page-5-0)

Neil Rabinowitz, Frank Perbet, Francis Song, Chiyuan Zhang, S. M. Ali Eslami, and Matthew Botvinick. Machine theory of mind. In Jennifer Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 4218– 4227. PMLR, 10–15 Jul 2018. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v80/rabinowitz18a.html) [v80/rabinowitz18a.html](https://proceedings.mlr.press/v80/rabinowitz18a.html). [298](#page-11-0)

Iyad Rahwan, Manuel Cebrian, Nick Obradovich, Josh Bongard, Jean-François Bonnefon, Cynthia Breazeal, Jacob W. Crandall, Nicholas A. Christakis, Iain D. Couzin, Matthew O. Jackson, Nicholas R. Jennings, Ece Kamar, Isabel M. Kloumann, Hugo Larochelle, David Lazer, Richard McElreath, Alan Mislove, David C. Parkes, Alex 'Sandy' Pentland, Margaret E. Roberts, Azim Shariff, Joshua B. Tenenbaum, and Michael Wellman. Machine behaviour. *Nature*, 568(7753):477–486, April 2019. ISSN 1476-4687. doi: 10.1038/s41586-019-1138-y. URL [https:](https://doi.org/10.1038/s41586-019-1138-y) [//doi.org/10.1038/s41586-019-1138-y](https://doi.org/10.1038/s41586-019-1138-y). [287](#page-0-0)

Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. Towards empathetic open-domain conversation models: A new benchmark and dataset. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 5370–5381. Association for Computational Linguistics, 2019. [293](#page-6-0)
- P. Ribino. The role of politeness in human–machine interactions: a systematic literature review and future perspectives. *Artificial Intelligence Review*, 56 (Suppl 1):445–482, 2023. doi: 10.1007/s10462-023-10540-1. [290](#page-3-0)

{25}------------------------------------------------

<span id="page-25-0"></span>Matthew Riemer, Zahra Ashktorab, Djallel Bouneffouf, Payel Das, Miao Liu, Justin D Weisz, and Murray Campbell. Position: Theory of mind benchmarks are broken for large language models. *arXiv preprint arXiv:2412.19726*, 2024. [300](#page-13-0)

Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. The goldilocks of pragmatic understanding: Fine-tuning strategy matters for implicature resolution by llms, 2023. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2210.14986) [2210.14986](https://arxiv.org/abs/2210.14986). [290](#page-3-0)

James A. Russell. A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6):1161–1178, 1980. doi: 10.1037/h0077714. [292](#page-5-0)

Jonathan Hvithamar Rystrom. Apolitical intelligence? auditing delphi's responses on controversial political issues in the US. *CoRR*, abs/2306.13000, 2023. doi: 10.48550/ARXIV.2306.13000. URL [https://doi.org/10.48550/arXiv.](https://doi.org/10.48550/arXiv.2306.13000) [2306.13000](https://doi.org/10.48550/arXiv.2306.13000). [303](#page-16-0)

Peter Salovey and John D. Mayer. Emotional intelligence. *Imagination, Cognition and Personality*, 9(3):185–211, 1990. doi: 10.2190/DUGG-P24E-52WK-6CDG. [292](#page-5-0)

Melanie Sclar, Jane Yu, Maryam Fazel-Zarandi, Yulia Tsvetkov, Yonatan Bisk, Yejin Choi, and Asli Celikyilmaz. Explore theory of mind: Program-guided adversarial data generation for theory of mind reasoning, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2412.12175) [abs/2412.12175](https://arxiv.org/abs/2412.12175). [299](#page-12-0)

Russ Shafer-Landau. *The Fundamentals of Ethics*. Oxford University Press, New York, 6 edition, 2024. ISBN 9780197697474. [303](#page-16-0)

Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever Hans or Neural Theory of Mind? Stress Testing Social Reasoning in Large Language Models. In Yvette Graham and Matthew Purver, editors, *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2257–2273, St. Julian's, Malta, March 2024. Association for Computational Linguistics. URL <https://aclanthology.org/2024.eacl-long.138>. [298](#page-11-0)

Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex Loftus, Aditya Ratan Jannali, Nikhil Prakash, Jasmine Cui, Giordano Rogers, Jannik Brinkmann, Can Rager, Amir Zur, Michael Ripa, Aruna Sankaranarayanan, David Atkinson, Rohit Gandikota, Jaden Fiotto-Kaufman, Eun-Jeong Hwang, Hadas Orgad, P Sam Sahil, Negev Taglicht, Tomer Shabtay, Atai Ambus, Nitay Alon, Shiri Oron, Ayelet Gordon-Tapiero, Yotam Kaplan, Vered Shwartz, Tamar Rott Shaham, Christoph Riedl, Reuth Mirsky, Maarten Sap, David Manheim, Tomer Ullman, and David Bau. Agents of chaos, 2026. URL [https:](https://arxiv.org/abs/2602.20021) [//arxiv.org/abs/2602.20021](https://arxiv.org/abs/2602.20021). [304](#page-17-0)

Ashish Sharma, Adam S. Miner, David C. Atkins, and Tim Althoff. A computational approach to understanding empathy expressed in text-based mental health support. 

{26}------------------------------------------------

<span id="page-26-0"></span>In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 5263–5276. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-main.425. [293](#page-6-0)

- Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Saki Mizuno, Keita Suzuki, Ryo Masumura, Hiroaki Sugiyama, and Kuniko Saito. Tomato: verbalizing the mental states of role-playing llms for benchmarking theory of mind. In *Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence*, AAAI'25/IAAI'25/EAAI'25. AAAI Press, 2025. ISBN 978-1-57735-897-
- 8. doi: 10.1609/aaai.v39i2.32143. URL [https://doi.org/10.1609/aaai.](https://doi.org/10.1609/aaai.v39i2.32143) [v39i2.32143](https://doi.org/10.1609/aaai.v39i2.32143). [300](#page-13-0)

Vera Sorin, Dana Brin, Yiftach Barash, Eli Konen, Alexander Charney, Girish Nadkarni, and Eyal Klang. Large language models and empathy: Systematic review. *Journal of Medical Internet Research*, 26:e52597, 2024. doi: 10.2196/52597. [293](#page-6-0)

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. PUB: A pragmatics understanding benchmark for assessing LLMs' pragmatics capabilities. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 12075–12097, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.719. URL [https:](https://aclanthology.org/2024.findings-acl.719/) [//aclanthology.org/2024.findings-acl.719/](https://aclanthology.org/2024.findings-acl.719/). [290,](#page-3-0) [294](#page-7-0)

James W. A. Strachan, Oriana Pansardi, Eugenio Scaliti, Marco Celotto, Krati Saxena, Chunzhi Yi, Fabio Manzi, Alessandro Rufo, Guido Manzi, Michael S. A. Graziano, Stefano Panzeri, and Cristina Becchio. Gpt-4o reads the mind in the eyes, 2024. URL <https://arxiv.org/abs/2410.22309>. [301](#page-14-0)

Kazuhiro Takemoto. The moral machine experiment on large language models. *Royal Society Open Science*, 11(2):231393, 2024. [304](#page-17-0)

Judith Jarvis Thomson. The trolley problem. *The Yale Law Journal*, 94(6):1395–1415, 1985. doi: 10.2307/796133. [302](#page-15-0)

Sean Trott, Tiago Timponi Torrent, Nancy Chang, and Nathan Schneider. (Re)construing meaning in NLP. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 5170–5184, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.462. URL <https://aclanthology.org/2020.acl-main.462/>. [290](#page-3-0)

Tomer Ullman. Large language models fail on trivial alterations to theory-of-mind tasks. *arXiv preprint arXiv:2302.08399*, 2023. [298](#page-11-0)

Ramira van der Meulen, Rineke Verbrugge, and Max van Duijn. Towards properly implementing theory of mind in ai systems: An account of four misconceptions,

2025. URL <https://arxiv.org/abs/2503.16468>. [290](#page-3-0)

{27}------------------------------------------------

<span id="page-27-0"></span>Max J van Duijn, Bram van Dijk, Tom Kouwenhoven, Werner de Valk, Marco R Spruit, and Peter van der Putten. Theory of mind in large language models: Examining performance of 11 state-of-the-art models vs. children aged 7-10 on advanced tests. *arXiv preprint arXiv:2310.20320*, 2023. [301](#page-14-0)

Emilio Villa-Cueva, SM Ahmed, Rendi Chevi, Jan Christian Blaise Cruz, Kareem Elzeky, Fermin Cristobal, Alham Fikri Aji, Skyler Wang, Rada Mihalcea, and Thamar Solorio. Moments: A comprehensive multimodal benchmark for theory of mind. *arXiv preprint arXiv:2507.04415*, 2025. [300](#page-13-0)

John Von Neumann and Oskar Morgenstern. Theory of games and economic behavior: 60th anniversary commemorative edition. In *Theory of games and economic behavior*. Princeton university press, 2007. [294](#page-7-0)

Haochuan Wang, Xiachong Feng, Lei Li, Zhanyue Qin, Dianbo Sui, and Lingpeng Kong. Tmgbench: A systematic game benchmark for evaluating strategic reasoning abilities of llms, 2024. URL <https://arxiv.org/abs/2410.10479>. [299](#page-12-0)

Victoria Williams and Benjamin Rosman. Heartificial intelligence: Exploring empathy in language models, 2025. arXiv preprint arXiv:2508.08271. [293](#page-6-0)

Heinz Wimmer and Josef Perner. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. *Cognition*, 13(1):103–128, 1983. ISSN 0010-0277. doi: https://doi.org/10. 1016/0010-0277(83)90004-5. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/0010027783900045) [science/article/pii/0010027783900045](https://www.sciencedirect.com/science/article/pii/0010027783900045). [298,](#page-11-0) [299](#page-12-0)

Yuheng Wu, Jianwen Xie, Denghui Zhang, and Zhaozhuo Xu. DEL-ToM: Inferencetime scaling for Theory-of-Mind reasoning via dynamic epistemic logic, May 2025. arXiv:2505.17348 [cs]. [301](#page-14-0)

Haoqiu Yan, Yongxin Zhu, Kai Zheng, Bing Liu, Haoyu Cao, Deqiang Jiang, and Linli Xu. Talk with human-like agents: Empathetic dialogue through perceptible acoustic reception and reaction, 2024. URL <https://arxiv.org/abs/2406.12707>. [293](#page-6-0)

Zhou Yang, Zhaochun Ren, Wang Yufeng, Shizhong Peng, Haizhou Sun, Xiaofei Zhu, and Xiangwen Liao. Enhancing empathetic response generation by augmenting llms with small-scale empathetic models, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.11801) [2402.11801](https://arxiv.org/abs/2402.11801). [293](#page-6-0)

Ziqi Yin, Hao Wang, Kaito Horio, Daisuke Kawahara, and Satoshi Sekine. Should we respect LLMs? a cross-lingual study on the influence of prompt politeness on LLM performance. In James Hale, Kushal Chawla, and Muskan Garg, editors, *Proceedings of the Second Workshop on Social Influence in Conversations (SICon 2024)*, pages 9–35, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.sicon-1.2. URL [https:](https://aclanthology.org/2024.sicon-1.2/) [//aclanthology.org/2024.sicon-1.2/](https://aclanthology.org/2024.sicon-1.2/). [291](#page-4-0)

{28}------------------------------------------------

<span id="page-28-0"></span>Kefan Yu, Qingcheng Zeng, Weihao Xuan, Wanxin Li, Jingyi Wu, and Rob Voigt. The pragmatic mind of machines: Tracing the emergence of pragmatic competence in large language models. In Vera Demberg, Kentaro Inui, and Lluís Marquez, editors, *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 192–213, Rabat, Morocco, March 2026. Association for Computational Linguistics. ISBN 979-8-89176-380-7. doi: 10.18653/v1/2026.eacl-long.9. URL [https:](https://aclanthology.org/2026.eacl-long.9/)

[//aclanthology.org/2026.eacl-long.9/](https://aclanthology.org/2026.eacl-long.9/). [290](#page-3-0)

Tengfei Yu, Siyu Pan, Caoyun Fan, Siyang Luo, Yaohui Jin, and Binglei Zhao. Can large language models exhibit cognitive and affective empathy as humans? *Computers in Human Behavior: Artificial Humans*, page 100233, 2025. doi:

10.1016/j.chbah.2025.100233. [293](#page-6-0)

{29}------------------------------------------------

