<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 15-24; Marker split PDF pages 1-10. -->



{0}------------------------------------------------

## <span id="page-0-0"></span>第一章

# 引言

大语言模型（LLM）迅速风靡全球。它们能用自然语言与我们交流，也能对现实世界进行推理，因此成为写作、答疑等多种任务中广受欢迎的工具。

LLM 的回答已经影响着我们的决策和行动。如果 LLM 能直接在现实世界中行动，会发生什么？如果 LLM 还能彼此互动，又会怎样？智能体大语言模型将 LLM 的能力与现实世界中的行动能力结合起来。本书将介绍智能体大语言模型如何运行、行动和互动，以及这对我们这些使用者意味着什么。

## 1.1 为什么要读这本书？

人工智能（AI）正在改变世界；能够*行动*的大语言模型则以新的方式推动这种变化。要了解变化的影响，AI 领域的学生和研究人员既要理解这项技术，也要了解正在出现的应用。这样才能帮助我们影响技术发展的方向、考虑加入防护措施，并进一步推动这一领域的发展。

这本书是为我们大学开设的关于智能体大语言模型的研究生课程而写的。在智能体大语言模型中，人工智能研究的两个领域汇聚在一起：代理的研究，以及大语言模型的研究。代理正在改变LLM的领域，反过来，LLM也在改变代理的领域。虽然已经有关于LLM及其工作原理的优秀书籍（例如[Jurafsky and Martin](#page-8-0) [\[2025\]](#page-8-0)），但行动能力使智能体大语言模型成为独特的存在。我们认为，需要一门课程和一本书来解释智能体大语言模型是如何工作的，以及它们如何实现以前不可能实现的事情。

## 1.2 概念与定义

LLM 的语言能力令人惊叹，它们说起话来就像人一样。如今，LLM 已开始具备智能体能力，其作用远超文本交流：

{1}------------------------------------------------

它们还能够*推理*、*使用工具*（例如编写并运行代码）、浏览网页获取信息，并与其他智能体或外部世界*互动*。

由此引出本书采用的两个定义。

*智能体*会采取行动、观察行动产生的效果，并利用从环境获得的反馈来调整自己的行动策略。

*智能体大语言模型*是融合了 LLM 能力的智能体，例如使用自然语言交流、编写代码或处理信息。

#### 1.2.1 提供协助的智能体与进行互动的智能体

行动能力提升了 LLM 在社会多个领域中的应用价值和适用范围。智能体大语言模型有哪些用途？本书将其主要用途分为两类：

(1) 智能体大语言模型用于在医疗、物流和金融等应用领域*协助*人类用户（第四部分）。

(2) 智能体大语言模型彼此*互动*，以完成任务，或在人类行为模拟中发挥作用（第五部分）。

近年来，这两个领域都发表了许多研究论文。我们将在整本书中提供关键研究和趋势的概述。

#### 1.2.2 智能体执行框架（harness）

从工程角度看，如何构建智能体大语言模型？它由两部分组成：（1）LLM；（2）一段称为*执行框架*（harness）的代码。

#### 智能体大语言模型 = LLM + 执行框架

执行框架是一段包裹模型并控制其调用方式的代码。这里的 harness 也有“马具”之意：马具能帮助人驾驭马匹、引导其力量。执行框架负责构造提示、处理回答、调用外部工具、管理状态、系统提示和技能，并循环执行任务所需的“推理—行动—观察”流程，无须人工逐步介入。因此，智能体执行框架也可以看作一个灵活的自动提示工程师（第二部分第 6 章）。更重要的是，它还能根据行动反馈，通过强化学习*调整*智能体的行为策略（第三部分第 9 章）。

#### 1.2.3 多模态模型

对于需要与用户及周围环境互动的智能体来说，理解自然语言至关重要。但许多任务还要求智能体处理其他模态的信息（如图像），或其他格式的文本（如电子表格和代码）。因此，智能体大语言模型的应用范围已经超出自然语言，

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Diagram_2.jpeg)

图 1.1：递归自我改进循环，连接三类智能体大语言模型：推理、行动和互动（红色，对应第二、三/四部分和第五部分）。影响某一类别的概念用绿色表示。

还包括多模态模型、代码模型和世界模型。因此，本书会根据上下文灵活使用 *LLM* 一词：有时专指仅处理语言的系统，有时则泛指以 Transformer 为基础、能够处理自然语言、代码、图像、工具指令或其他相关模态的大型模型（第二部分第 7 章）。如需区分这两种含义，我们会明确说明所采用的定义。

#### 1.2.4 训练数据

智能体大语言模型标志着 LLM 工程领域的又一次范式转变，也改变了机器学习的侧重点。长期以来，LLM 主要在模型训练阶段学习；如今，模型在推理阶段（即被调用使用时）学习的作用也日益突出。

这带来两项优势。第一，它为动态的智能体行为提供了灵活性和空间，正如第 8、9 章将要介绍的。第二，几乎整个互联网都已被用作模型的预训练数据 [\[Sutskever,](#page-8-0) [2024\]](#page-8-0)，因此生成更多相关的高质量数据很有价值。智能体可以通过与环境互动自行生成这类数据。图 1.1 展示了这一过程；互动产生的数据还能用于后续训练。

视觉—语言—动作模型（VLA）就是一个例子：它们从机器人动作序列中学习（第 7 章）。另一个例子见第 17 章：智能体组成的社会可以通过互动让模型变得更智能，就像人们在交流中学习新的行为和想法一样。

因此，智能体大语言模型不仅能支持有用的应用，还能生成更多训练数据，形成递归的自我改进循环。图 1.1 展示了互动数据生成的良性循环。红色部分表示主要类别：推理（第二部分）、行动（第三、四部分）和互动（第五部分）；绿色部分表示会直接影响某一类别或通过反馈循环产生影响的概念。

{3}------------------------------------------------

## 1.3 读者对象与写作目标

本书主要面向人工智能和计算机科学的研究生和研究人员。对于其他领域的学生和学者，例如认知科学、社会科学或人文学科的学生和学者，如果他们对机器学习和技术挑战有内在的兴趣，也可能会感兴趣。

本书的目的是介绍和回顾智能体大语言模型领域中技术和应用的进展。我们深入讨论了智能体大语言模型工程的主要概念。我们提供了科学文献的参考资料，以及源代码和数据集的链接，以便您可以尝试构建自己的智能体。请注意，该领域是新的，并且仍在不断发展。为了能够跟上最新发展，彻底理解基础知识非常重要。我们的目标是为您提供这两方面的内容。

我们使用这本书来教授莱顿大学计算机科学（人工智能方向）和创意智能与技术硕士项目中的课程智能体大语言模型。提供了一个包含支持材料（讲义幻灯片、作业、新闻）的网页。<sup>1</sup>

亲爱的读者，我们希望您在阅读本书时能够像我们写作时一样享受其中。智能体大语言模型研究是一个新领域，而这本书同样非常新颖。新的发展每天都在发生。我们非常欢迎您通过网站提供反馈和对后续版本的建议！

## 1.4 前提条件

本书面向具备计算机编程、人工智能、机器学习和自然语言处理本科基础知识的读者。

我们假设你会使用 Python 和 PyTorch，并了解机器学习的基本原理，例如分类与回归、训练集与测试集，以及过拟合和欠拟合。你应当熟悉决策树、集成方法和随机森林等经典机器学习方法，也应了解深度学习的基本概念，包括人工神经网络（全连接多层感知器、卷积神经网络和循环神经网络）、反向传播、损失函数（如交叉熵）和激活函数（如 ReLU）。此外，你还应熟悉强化学习，了解序列决策问题、智能体与环境的区别，并会使用一些策略学习算法（如 Q-learning、DQN、PPO 和 SAC）。最后，你应具备自然语言处理基础，了解分词、词嵌入、注意力机制和 Hugging Face。

即使你对其中部分或全部主题还不熟悉，本书仍能为你提供帮助。我们会简要介绍必要的基础知识，使你能够以适当的深度学习智能体大语言模型，并找到进一步学习的切入点。

<sup>1</sup><https://agenticllms.liacs.nl>

{4}------------------------------------------------

<span id="page-4-0"></span>对于其中一些主题，手边备有合适的教材会很有帮助。以下是一些推荐读物：

- 用于经典机器学习：[Alpaydin](#page-7-0) [\[2021\]](#page-7-0)，[Flach](#page-7-0) [\[2012\]](#page-7-0)，
- 用于深度监督机器学习：[Bishop and Bishop](#page-7-0) [\[2023\]](#page-7-0)，[Goodfellow](#page-7-0) [et al.](#page-7-0) [\[2016\]](#page-7-0)，[Chollet and Chollet](#page-7-0) [\[2021\]](#page-7-0)，[Stevens et al.](#page-8-0) [\[2020\]](#page-8-0)，
- 针对强化学习：[Sutton and Barto](#page-8-0) [\[2018\]](#page-8-0)，[Graesser and Keng](#page-8-0) [\[2019\]](#page-8-0)，[Plaat](#page-8-0) [\[2022\]](#page-8-0)，[Franc¸ois-Lavet et al.](#page-7-0) [\[2018\]](#page-7-0)，
- 自然语言处理和 Transformer 模型：[Jurafsky and Martin](#page-8-0) [\[2025\]](#page-8-0)。

## 1.5 课程安排

我们在本校的智能体大语言模型研究生课程中使用本书。全书力求简洁而全面，内容覆盖这一领域的核心知识，深度适合研究生课程。为了方便深入学习，我们列出了大量科学文献；带 * 标记的章节和小节则介绍较高级的主题。

不同大学的课程安排可能有所不同（见表 [1.1](#page-5-0)）。我们教授的课程共 14 讲，内容涵盖多个方向：第 1 讲为导论；第一部分（语言建模）3 讲；第二、三部分（推理与智能体）3 讲；第四部分（工具与助手）3 讲；第五部分（社会模拟）3 讲；最后 1 讲用于期末评估。课程包含 4 个案例研究，并在课堂上评分。课程不另设理论考试，最终成绩取 4 项作业的平均分。

你可以根据自己的情况灵活使用本书。本书主要介绍智能体大语言模型的两类应用场景：（1）助理；（2）社会模拟。

如果课程另设一门深入讲授 LLM 的课程，那么聚焦于智能体助理的核心课程可以只涵盖第二至第四部分。作业可选案例研究 2 和 3。

如果课程聚焦于社会模拟，则核心内容应包括第二、三和第五部分；作业可选案例研究 2 和 4。

根据大学课程的安排，这两种应用方向都可以按需与第一部分（语言建模）结合。

第六部分介绍高级主题。第 19 章讨论合成数据生成，适用于第四部分的助理应用；第 21 章讨论机制可解释性，可帮助读者深入理解第一、二部分中的语言模型和推理模型；第 20 章介绍高级架构，也适用于第二部分；第 22 章回顾智能体认知，适用于第五部分的社会模拟。

表 [1.1](#page-5-0) 汇总了针对三种不同课程目标使用本书内容的方案。

<sup>2</sup>参见 [Gunasekar et al.](#page-8-0) [\[2023\]](#page-8-0)。

{5}------------------------------------------------

<span id="page-5-0"></span>

| 课程目标 | 部分 | 案例研究 | 章节 | 可选章节 |
|------------------------|-------------------|------------|-----------------|----------|
| 综合课程 | I、II、III、IV、V | 1、2、3、4 | 1–18 | 19–22 |
| 助理核心课程 | II、III、IV | 2、3 | 6–14 | 2–5、19 |
| 社会模拟核心课程 | II、III、V | 2、4 | 6–10、11、15–18 | 2–5、22 |

表 1.1：不同课程目标对应的部分和章节

## 1.6 方法、范围与概述

本书注重实践。介绍每个新概念时，我们都会提供理论背景，并举例说明如何应用。书中讨论数据集、训练算法、基准测试和训练环境，提供重要智能体算法的伪代码，并通过实践案例帮助你学习。

本书分为七个部分，每部分由若干章节组成，围绕一个主题展开。

第一部分介绍大语言模型基础。第 2 章讨论分词、嵌入、循环神经网络、注意力机制、Transformer 架构和自回归训练等重要的自然语言处理概念；自回归训练是 LLM 训练的核心方法。本章还介绍监督学习和强化学习。第 3 章详细讲解 LLM 训练流程，包括数据集获取、预训练、微调、指令微调、价值对齐和推理，并介绍常用的 LLM 性能基准。第 4 章讨论规模定律、小型语言模型和能耗。要使用智能体大语言模型，你需要了解 Transformer 模型预训练和微调的基础知识。第一部分的最后一章通过实践案例指导你从零训练一个 Transformer 模型。

第二部分介绍推理、自我反思和世界模型；这些都是高级智能体大语言模型的重要组成部分。第 6 章从上下文学习讲起，说明思维链如何帮助 LLM 解决简单的数学应用题并进行多步推理。我们还将介绍智能体如何通过执行框架自我反思，改进上下文学习并增强推理能力。前面讨论的模型都以语言为主。要在现实世界中行动，智能体还必须理解图像等其他模态。第 7 章将讨论视觉—语言模型、从决策序列中学习的模型，以及能够执行机器人动作的模型。为了行动时不撞上墙壁，智能体需要一定的物理知识；本章还将介绍世界模型及其与基于模型的强化学习之间的关系。第 8 章将进一步讨论智能体如何学习，以及为什么这类学习也可视为元学习。

接下来，第三部分第 9 章将从人工智能发展史的角度回顾智能体，

{6}------------------------------------------------

并讨论智能体大语言模型的不同交互模式。要在现实世界中行动，智能体还需要记忆。本部分将介绍不同的智能体记忆架构，并以实践案例收尾：你可以基于 BALROG 基准套件中的高难度谜题和游戏，构建一个能够自我反思的智能体。

第四部分从理论转向实践，介绍社区开发的智能体大语言模型框架，以及如何使用工具在现实世界中行动。部分框架为专有软件，许多则是开放的，可从 GitHub 下载。本部分聚焦于能够行动并为用户完成实际任务的智能体。第 11 章介绍用于构建智能体大语言模型的工具链、框架和 API；第 12 章介绍检索增强方法，以及如何搜索 LLM 训练后出现的最新信息；第 13 章概述其实际应用，包括购物、医疗和交易助理，以及辅助科学研究的系统。第四部分的最后一章通过案例研究讲解如何构建医疗诊断智能体，也会分析它的局限，提醒读者不要盲目采纳智能体的医疗建议。

第五部分从行为和认知的角度研究智能体大语言模型，而非关注技术实现。在介绍机器行为研究之后，第 15 章将在受控实验中考察单个 LLM 和智能体大语言模型的社会行为。第 16 章转向多智能体系统，讨论如何通过构建或提示智能体大语言模型扮演特定角色、围绕目标互动。第 17 章进一步研究由多个智能体组成的开放式互动系统，以及合作、规范执行等*涌现行为*。相关案例研究将带你深入了解智能体的社会行为。

第六部分介绍更高级的内容。第 19 章讨论合成数据生成。与其他深度学习模型一样，LLM 在很大程度上仍是“黑箱”；本书将介绍一些揭开其内部机制的技术。第 20 章讨论高级架构，包括在决策轨迹上训练的 Transformer、循环式和递归式 LLM，以及上下文强化学习、自我修改智能体和元智能体等高级智能体架构。第 21 章介绍机制可解释性工具，用于分析 Transformer 架构中的神经元活动模式，从而更好地理解 LLM 的推理过程。第 22 章回顾智能体大语言模型行为与认知方面的近期研究，包括心智理论和意识等前沿问题。

最后，第七部分综合本书介绍的智能体大语言模型相关内容。第 23 章讨论这类模型目前能做什么、仍有哪些能力局限，并展望其未来发展。

请注意，部分章节和小节标题旁标有 * 号，表示内容较为进阶，可能更适合论文研究，而非课程教学。

{7}------------------------------------------------

## <span id="page-7-0"></span>1.7 致谢

本书源于我们对智能体大语言模型的兴趣。这份兴趣促成了我们与学生、同事之间的许多讨论，也推动了令人振奋的研究。后来，我们决定写这本书并开设一门研究生课程。我们相信，智能体大语言模型是一个重要课题，AI 领域的许多学生都希望了解它。

本书部分内容基于或受到我们此前工作的启发（例如，[van Duijn et al.](#page-9-0) [\[2023\]](#page-9-0)、[Van Der Meer et al.](#page-8-0) [\[2024\]](#page-8-0)、[Huisman et al.](#page-8-0) [\[2021\]](#page-8-0)、[Moerland et al.](#page-8-0) [\[2023\]](#page-8-0)、[Plaat](#page-8-0) [\[2022\]](#page-8-0)、[Plaat et al.](#page-8-0) [\[2025b,a\]](#page-8-0)、[van Stein and Bäck](#page-9-0) [\[2024\]](#page-9-0)、Chen et al. [2026]、[van Wetten et al.](#page-9-0) [\[2025\]](#page-9-0) 以及 [Xing et al.](#page-9-0) [\[2025\]](#page-9-0)）。感谢这些工作的所有合著者；没有你们的贡献，本书无法完成！

第 18 章关于生成式智能体社会的案例研究由 Dorota Moravčíková 和 Friso van der Veen 撰写，改编自他们此前的相关工作（[Park et al.](#page-8-0) [\[2023\]](#page-8-0)）。感谢他们为本书调整这项工作，并为前面章节提出宝贵意见。

我们感谢研究所的学生和同事。特别感谢 Thomas Bäck、Joost Batenburg、Joost Broekens、Xi Chen、Lennard Froma、Bas Haring、Anna Kononova、Tom van Kouwenhoven、Thomas Moerland、Rob van Nieuwpoort、Mike Preuss、Peter van der Putten、Zhaochun Ren、Derya Soydaner、Suzan Verberne、Fien van Wetten、Annie Wong 和 Tarik Zaciragić，感谢大家进行了许多富有成效的讨论。

## 参考文献

Ethem Alpaydin. *Machine learning*. MIT press, 2021. [19](#page-4-0)

C.M. Bishop and H. Bishop. *Deep Learning: Foundations and Concepts*. Computer Science. Springer International Publishing, 2023. ISBN 9783031454684. URL <https://books.google.nl/books?id=0uTgEAAAQBAJ>. [19](#page-4-0) Xi Chen, Aske Plaat, and Niki van Stein. How does chain of thought think? mechanistic interpretability of chain-of-thought reasoning with sparse autoencoding. In *AAAI2026, arXiv preprint arXiv:2507.22928*, 2026. 22 Francois Chollet and Franc¸ois Chollet. *Deep learning with Python*. simon and schuster, 2021. [19](#page-4-0) Peter Flach. *Machine learning: the art and science of algorithms that make sense of data*. Cambridge university press, 2012. [19](#page-4-0) Vincent Franc¸ois-Lavet, Peter Henderson, Riashat Islam, Marc G Bellemare, and Joelle Pineau. An introduction to deep reinforcement learning. *arXiv preprint arXiv:1811.12560*, 2018. [19](#page-4-0) Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep learning*. MIT press, 2016. [19](#page-4-0)

{8}------------------------------------------------

<span id="page-8-0"></span>Laura Graesser and Wah Loon Keng. *Foundations of deep reinforcement learning: theory and practice in Python*. Addison-Wesley Professional, 2019. [19](#page-4-0) Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar Teodoro Mendes, Allie Del ´ Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sebastien ´ Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need, 2023. URL <https://arxiv.org/abs/2306.11644>. [19](#page-4-0) Mike Huisman, Jan N Van Rijn, and Aske Plaat. A survey of deep meta-learning. *Artificial Intelligence Review*, 54(6):4483–4541, 2021. [22](#page-7-0) Daniel Jurafsky and James H. Martin. *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, with Language Models*. Stanford, 3rd edition, 2025. URL [https:](https://web.stanford.edu/~jurafsky/slp3/) [//web.stanford.edu/˜jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/). Online manuscript released August 24, 2025. [15,](#page-0-0) [19](#page-4-0) Thomas M Moerland, Joost Broekens, Aske Plaat, Catholijn M Jonker, et al. Modelbased reinforcement learning: A survey. *Foundations and Trends® in Machine Learning*, 16(1):1–118, 2023. [22](#page-7-0) Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th annual acm symposium on user interface software and technology*, pages 1–22, 2023. [22](#page-7-0) Aske Plaat. *Deep reinforcement learning*. Springer, 2022. [19,](#page-4-0) [22](#page-7-0) Aske Plaat, Max van Duijn, Niki van Stein, Mike Preuss, Peter van der Putten, and Kees Joost Batenburg. Agentic large language models, a survey. *Journal of Artificial Intelligence Research, arXiv preprint arXiv:2503.23037*, 84:29:1–29:74, 2025a. [22](#page-7-0) Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and Thomas Back. Multi-step reasoning with large language models, a survey. ¨ *ACM Computing Surveys, arXiv preprint arXiv:2407.11511*, 58(6):160:1–160:35, 2025b. [22](#page-7-0) Eli Stevens, Luca Antiga, and Thomas Viehmann. *Deep Learning with PyTorch: Build, train, and tune neural networks using Python tools*. Manning, 2020. [19](#page-4-0) Ilya Sutskever. Acceptance speech for test of time award. In *Advances in Neural Information Processing Systems*, Vancouver, British Columbia, Canada, 2024. [17](#page-2-0) Richard S Sutton and Andrew G Barto. *Reinforcement Learning*. Adaptive Computation and Machine Learning series. Bradford Books, Cambridge, MA, 2 edition, November 2018. [19](#page-4-0) Michiel Van Der Meer, Enrico Liscio, Catholijn Jonker, Aske Plaat, Piek Vossen, and Pradeep Murukannaiah. A hybrid intelligence method for argument mining. *Journal of Artificial Intelligence Research*, 80:1187–1222, 2024. [22](#page-7-0)

{9}------------------------------------------------

<span id="page-9-0"></span>Max J van Duijn, Bram van Dijk, Tom Kouwenhoven, Werner de Valk, Marco R Spruit, and Peter van der Putten. Theory of mind in large language models: Examining performance of 11 state-of-the-art models vs. children aged 7-10 on advanced tests. *arXiv preprint arXiv:2310.20320*, 2023. [22](#page-7-0) Niki van Stein and Thomas Back. Llamea: A large language model evolutionary algo- ¨ rithm for automatically generating metaheuristics. *arXiv preprint arXiv:2405.20132*, 2024. [22](#page-7-0) Fien van Wetten, Aske Plaat, and Max van Duijn. Baba is llm: Reasoning in a game with dynamic rules. *arXiv preprint arXiv:2506.19095*, 2025. [22](#page-7-0) Peiwen Xing, Aske Plaat, and Niki van Stein. Cocomposer: Llm multi-agent collaborative music composition. *arXiv preprint arXiv:2509.00132*, 2025. [22](#page-7-0)
