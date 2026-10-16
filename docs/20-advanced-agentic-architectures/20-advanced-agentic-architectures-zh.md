<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 391-402; Marker split PDF pages 1-12. -->



{0}------------------------------------------------

## 第20章

# \*高级 Transformer 与智能体架构

#### 你将学到的内容

在本章中，你将了解先进的 Transformer 与智能体架构，并能够：

- 理解*图灵完备性*，以及 Transformer 在理论上具有图灵完备性。
- 说明在*决策轨迹*而非自然语言上训练的 Transformer 如何执行规划。
- 说明*循环式语言模型*如何绕过分词层，直接连接嵌入。
- 说明智能体如何通过在上下文中保存状态—动作—奖励三元组的历史来执行强化学习。
- 讨论*自我修改型智能体*的优势与局限。
- 说明如何将元学习用于执行框架设计，构建能够改进基础执行框架的*元执行框架*。

## 20.1 Transformer 架构

大语言模型的成功推动了 Transformer 架构领域的新研究。LLM 擅长语言任务，但在推理和规划等任务上的表现仍有不足，因此研究者开始探索 Transformer 的计算能力边界。本章将讨论其中几个方向：Transformer 是否具有图灵完备性（第 [20.1.1](#page-1-0) 节）；直接在决策轨迹而非自然语言上训练 Transformer，能否提升规划与回溯能力（第 [20.1.2](#page-1-0) 节）；以及通过嵌入连接模型的循环式语言模型。

{1}------------------------------------------------

<span id="page-1-0"></span>直接在嵌入层面上。

#### 20.1.1 Transformer 具有图灵完备性

LLM 在推理时难以管理状态，这似乎限制了它们能够执行的计算类型。在理论计算机科学中，*图灵完备性*用于描述一种计算系统是否具有通用计算能力，包括正确维护状态的能力。阿兰·图灵在 1936 年的一篇论文中提出了这一概念 [\[Turing,](#page-10-0) [1936\]](#page-10-0)。图灵完备的系统能够模拟其他计算设备，并执行任何可由算法描述的计算；但具体计算可能需要大量时间和内存。大多数通用编程语言都具有图灵完备性。<sup>1</sup>

LLM 以及其他基于 Transformer 的模型在状态管理上的困难，似乎暗示 Transformer 架构可能不具备图灵完备性。然而，Perez et al. 在 2021 年研究了这一问题，并证明 Transformer 能够模拟图灵机，因此具有图灵完备性 [Perez et al.](#page-10-0) [\[2021\]](#page-10-0)。

这一结果令人鼓舞：它表明 Transformer 正确管理状态并不存在根本性的理论障碍。剩下的挑战可能在于找到合适的提示、预训练数据和微调数据。

不过，理论上可行并不意味着实际实现容易或高效。例如，对 Transformer 架构进行扩展，可能有助于更高效地管理状态。下一节将介绍一些帮助模型解决决策与规划问题的方法。

## 20.1.2 基于决策轨迹的预训练与回溯

Transformer 具有图灵完备性这一发现推动了规划与决策领域的研究。本节介绍直接在决策轨迹上预训练 Transformer 并取得良好结果的工作。这里讨论的模型并非在自然语言上训练的 LLM，而是在决策轨迹上训练的 Transformer。

首先，考虑直接在决策轨迹上预训练 Transformer：输入是状态—动作—奖励三元组序列，*而非*语言词元。多项研究探讨了 Transformer 能否通过反馈学习策略，并通过探索找到更好的动作。[Chen et al.](#page-8-0) [\[2021\]](#page-8-0)、[Lee et al.](#page-9-0) [\[2023\]](#page-9-0) 和 [Wang et al.](#page-11-0) [\[2024\]](#page-11-0) 表明，基础 Transformer 架构确实能够在典型规划任务的决策轨迹上进行预训练和微调。
 的决策轨迹上进行预训练和微调。
<sup>1</sup>For example, the Python programming language is Turing complete, but simple regular expression matching, without backreferencing, is not.

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Diagram_2.jpeg) 
  
 图 20.1：倒立摆问题：仅通过移动小车向左或向右来平衡摆 
 
未经语言训练、仅在决策轨迹上预训练的 Transformer，也能表现出上下文内强化学习（ICRL）。目前，这一能力已在 Boyan 链和倒立摆等简单任务中得到展示（见图 20.1）。在国际象棋和 Hex 中，研究者使用极大极小搜索 [\[Markeeva et al.,](#page-9-0) [2024,](#page-9-0) [Knuth and](#page-9-0) [Moore,](#page-9-0) [1975\]](#page-9-0) 或蒙特卡洛树搜索（MCTS）[\[Browne et al.,](#page-8-0) [2012\]](#page-8-0) 生成的轨迹来预训练决策 Transformer，也取得了令人鼓舞的结果 [\[Schultz et al.,](#page-10-0) [2024,](#page-10-0) [Kempinski et al.,](#page-9-0) [2025\]](#page-9-0)。模型学会了向前搜索若干步。其他研究也报告了 Transformer 在奥赛罗任务上的积极结果 [\[Li et al.,](#page-9-0) [2023,](#page-9-0) [Kempinski et al.,](#page-9-0) [2025,](#page-9-0) [Chen et al.,](#page-8-0) [2025\]](#page-8-0)。

另一项研究探索了如何通过*自我回溯*训练 Transformer 执行决策 [\[Yang et al.,](#page-11-0) [2026a\]](#page-11-0)。该方法仍以决策轨迹训练模型，但加入特殊词元 `<backtrack>`。训练中遇到错误时，模型会学会输出这一词元并寻找其他解法；推理时，模型据此执行“扩展—回溯—选择”流程，探索当前状态的备选方案。

在 24 点游戏和迷宫导航等典型规划任务上，自我回溯方法优于小型 DeepSeek-R1 和 Qwen3 等标准基线。通过加入专用词元和回溯功能，该方法使 Transformer 能够执行搜索，也模糊了训练与推理的边界。

这些结果令人鼓舞，但 Transformer 在状态管理方面的弱点究竟在多大程度上得到了解决，仍有待研究。在更大规模问题上开展推理时 ICRL 也是一个活跃的研究方向。

{3}------------------------------------------------

## <span id="page-3-0"></span>20.1.3 循环式 LLM 与递归多智能体系统

提升 Transformer 的表达能力，有望使其成为更通用的计算架构。本节还将介绍两个相关方向：循环式语言模型和递归系统。
  
让 LLM 智能体通过自然语言交流很容易，因为它们都经过这类文本训练。但这种方式需要反复进行分词与反分词，既耗时，也可能造成嵌入表示不一致。能否绕过分词器，直接连接嵌入，并在嵌入层面完成训练与推理？

[Giannou et al.](#page-9-0) [\[2023\]](#page-9-0) 展示了循环式 Transformer 如何模拟基础计算器、线性代数库，以及使用反向传播的上下文学习算法。循环式 Transformer 会将输入神经元直接连接到同一模型或其他模型的输出神经元。后续研究进一步探索了这一架构 [Yang et al.](#page-11-0) [\[2023\]](#page-11-0)、[Saunshi et al.](#page-10-0) [\[2025\]](#page-10-0)、[Zhu et al.](#page-11-0) [\[2025\]](#page-11-0)。

递归多智能体系统是直接连接不同模型嵌入的另一种发展方向 [\[Yang et al.,](#page-11-0) [2026b\]](#page-11-0)。它将异构智能体接入协作循环，从而支持分布内潜在思维生成，以及智能体之间的潜在状态传递。该系统采用内外循环学习算法（元学习，见第 8 章），在多轮递归过程中通过共享的梯度信用分配协同优化整个系统。使用四个智能体在数学、科学、医学、搜索和代码生成基准上的实验显示，训练过程稳定；与单智能体、多智能体及递归基线相比，准确率提高数个百分点，推理速度翻倍，词元用量也以相近比例下降。

RecursiveMAS 的一项创新是引入两级递归链接。由于梯度需要穿过多个模型，多智能体系统的训练容易变得不稳定。RecursiveMAS 冻结底层 LLM 的参数，只训练轻量级的 RecursiveLink 模块，以缓解这一问题；训练分为两个阶段。

## 20.2 智能体架构

除了 Transformer 架构方面的进展，智能体 LLM 研究也催生了新的智能体架构。本节先介绍上下文内强化学习：当提示包含状态—动作—奖励三元组的历史时，LLM 可以据此改进策略；随后讨论自我修改型智能体的优缺点，最后介绍元智能体。

## 20.2.1 上下文内强化学习

回顾第 2.2.3 节，强化学习旨在求得一种策略，使智能体在每个状态下都能选择最优动作 [\[Sutton and Barto,](#page-10-0) [2018\]](#page-10-0)。智能体逐渐收敛到该策略的过程中，会在每个时间步根据环境反馈的奖励学习如何选择动作。

{4}------------------------------------------------

图 20.2：作为参考，标准强化学习循环中，智能体从环境根据动作 $a$ 给出的反馈奖励 $r$ 学习一个由 $\theta$ 参数化的策略 $\pi_\theta$；动作从策略 $\pi_\theta$ 中选择，等等 [\[Sutton and Barto,](#page-10-0) [2018\]](#page-10-0)。
<span id="page-4-0"></span>![](assets/_page_4_Diagram_2.jpeg) 
  
  
这些奖励会增强智能体对其策略所选动作的信心（见图 20.2）。上一节讨论的自我反思方法也借鉴了强化学习的思想，用于智能体执行框架。接下来介绍如何通过上下文内强化学习优化提示。
 
研究发现，当外部提示优化器向 LLM 提供环境交互历史时，LLM 在某些自我反思循环中能够改进策略。这种现象称为上下文内强化学习（ICRL）[\[Song et al.,](#page-10-0) [2025\]](#page-10-0)，见图 [20.3](#page-5-0)。

[Krishnamurthy et al.](#page-9-0) [\[2024\]](#page-9-0) 探讨了经过语言训练的通用 LLM 能否在上下文中进行探索并从反馈中学习。此后的多项研究对此给出了肯定答案 [\[Demircan et al.,](#page-8-0) [2024,](#page-8-0) [Song et al.,](#page-10-0) [2025,](#page-10-0) [Moeini et al.,](#page-10-0) [2025,](#page-10-0) [Monea et al.,](#page-10-0) [2024,](#page-10-0) [Nie et al.,](#page-10-0) [2024,](#page-10-0) [Wang et al.,](#page-11-0) [2024\]](#page-11-0)。
 
在 ICRL 中，外部执行框架会反复调用 LLM，并将不断增长的状态—动作—奖励历史作为上下文提供给它（图 [20.3](#page-5-0)）。LLM 据此改进隐式策略，逐轮选出更好的动作；从这个意义上说，模型在内部执行了强化学习的核心策略改进步骤。

在 ICRL 中，智能体算法选择一个*动作*并执行一步，然后从环境收到一个*状态—奖励*对（见图 [20.3](#page-5-0)）。在马尔可夫决策过程中（MDP），前一状态和所采取的动作包含确定下一状态所需的全部信息。因此，策略 LLM 的提示会累积 $(s,a,r)$ 三元组历史，供模型改进动作选择。保留观测历史是表示状态的一种自然方式；不过，如何将 LLM 的提示优化准确建模为 MDP，仍是活跃的研究问题。

在 ICRL 中，LLM 充当智能体的策略函数。也可以使用另一个提示，让 LLM 扮演环境并生成下一状态与奖励；为避免智能体意外利用环境模型掌握的信息，通常应使用独立的模型。这仍是一个活跃的研究方向。
  
上下文内强化学习通过多轮提示执行：每轮中，环境 LLM 生成回应后，另一个 LLM 会为该回应计算数值奖励。

{5}------------------------------------------------

<span id="page-5-0"></span>![](assets/_page_5_Diagram_2.jpeg) 
  
 图 20.3：ICRL：使用强化学习 
 的提示优化
下一轮中，智能体 LLM 会收到相同任务，以及先前的回应和奖励。随着上下文逐渐增长，回应质量可能随之提高，使模型能够在推理时最大化数值奖励。Song et al. [\[2025\]](#page-10-0) 在思维树的应用中验证了 ICRL，包括 24 点游戏、创意写作和 ScienceWorld；结果显示，该方法优于 Self-Refine 和 Reflexion。

ICRL 在 LLM 内部如何运作？推理时，LLM 通过 Transformer 网络的前向计算生成词元。研究发现，这一前向计算可能会隐式地执行强化学习 [\[Wang et al.,](#page-11-0) [2024\]](#page-11-0)。权重内学习是模型在训练阶段通过参数学会规律；上下文学习是模型在推理阶段通过提示中的少量示例学习；上下文内强化学习则是模型利用交互过程中逐渐累积的状态—动作—奖励序列进行策略改进。将这些历史信息纳入上下文后，LLM 的前向计算便可隐式地优化策略。

LLM 可以充当智能体和策略优化器，也可以充当环境模型，计算下一状态和奖励。使用不同模型有助于避免智能体意外利用环境模型的信息。这仍是一个活跃的研究方向。

## 20.2.2 自我修改型超智能体

本书前面介绍过基于搜索算法和强化学习的自我反思提示优化方法。这些外部算法由研究人员编写，在智能体学习过程中保持不变，因此提示优化能力受限于固定算法本身。

LLM 擅长编写和评估计算机代码 [\[Zhuge et al.,](#page-11-0) [2024\]](#page-11-0)。原则上，我们可以让 LLM 为智能体提示或上下文内强化学习编写执行框架；还可以让它在具体问题上运行代码，并根据实验结果改进执行框架本身。如果 LLM 能够依据实验结果改进提示优化代码，这种方法或许会优于

{6}------------------------------------------------

<span id="page-6-0"></span>文献中已有的标准方法。

自我修改代码是指程序在运行过程中改变自身指令，这一概念在计算机科学中由来已久 [\[Cai et al.,](#page-8-0) [2007\]](#page-8-0)，也可作为条件分支的一种替代方式。从汇编语言到 Lisp [\[Abelson and Sussman,](#page-8-0) [2022\]](#page-8-0) 和 Python，许多语言都支持自我修改。它有时能提升性能，例如即时编译 [\[Ansel et al.,](#page-8-0) [2011\]](#page-8-0)。但自我修改也会降低程序的可预测性，使调试更困难，并带来安全隐患。因此，对行为可预测性要求较高时，通常会谨慎采用这种方法。

自我修改型智能体的一个优势是开放性：它们能够持续改进，而不受静态算法预设边界的限制 [\[Schmidhuber,](#page-10-0) [2007\]](#page-10-0)。但如何安全地实现这种改进仍是一项挑战 [\[Zhang et al.,](#page-11-0) [2026b\]](#page-11-0)。Darwin-Gödel Machine-Hyperagents 将（1）解决目标任务的智能体与（2）负责修改自身和任务智能体的元智能体整合进一个可编辑程序。由于元层的修改流程本身也可被编辑，系统原则上既能改进任务执行行为，也能改进产生后续改进的机制 [\[Zhang et al.,](#page-11-0) [2026c\]](#page-11-0)。不过，实验尚未实现完全开放：任务分布、父代选择方式和评估协议仍是固定的。

方法越开放，越难控制改进过程，确保系统持续进步而不发生退化 [\[Cemri et al.,](#page-8-0) [2025,](#page-8-0) [Ecoffet et al.,](#page-9-0) [2020\]](#page-9-0)。更一般地说，强大的元学习方法必须保证学习过程稳定；生成能力越强，评估与适应环境的能力也需要相应提升。

### 20.2.3 元智能体

许多自我反思方法采用元学习式的两层优化，这一原则也可以用于设计智能体执行框架。近年来出现了多种此类架构：内层学习循环由执行框架和 LLM 组成，外层循环则负责改进内层执行框架的配置。内层框架通常管理提示、记忆、工具、安全和控制，并可用自然语言或代码表示；外层循环分析内层的执行轨迹，据此调整框架配置或代码。下面简要介绍几种方法。

自我改进编码智能体 [\[Robeyns et al.,](#page-10-0) [2025\]](#page-10-0) 建立在自我修改型智能体的思路上。研究者展示了配备基本编程工具的智能体系统能够自主修改自身；该方法将软件工程基准任务上的表现从 17% 提升至 53%。

Meta-Harness [\[Lee et al.,](#page-9-0) [2026\]](#page-9-0) 借助外部记忆和文本优化取得了良好结果。图 [20.4](#page-7-0) 展示了其流程：（1）文件系统保存此前候选框架的源代码、执行轨迹和得分，智能体读取这些信息后生成新的执行框架；（2）新框架在
 上进行评估
{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Diagram_2.jpeg) 
  
 图 20.4：Meta 执行框架方法示意图 [\[Lee et al.,](#page-9-0) [2026\]](#page-9-0) 
  
 ![](assets/_page_7_Diagram_4.jpeg) 
  
 ![](assets/_page_7_Figure_5.jpeg) 
  
 图 20.5：Shepherd Meta 代理环境示意图 [\[Yu et al.,](#page-11-0) [2026\]](#page-11-0) 
  
评估任务上接受测试；（3）随后，系统将新代码、推理轨迹和评估得分存入文件系统的新目录，再开始下一轮迭代。

Self-Harness [\[Zhang et al.,](#page-11-0) [2026a\]](#page-11-0) 和 Agentic Harness Engineering [\[Lin et al.,](#page-9-0) [2026\]](#page-9-0) 进一步探索了执行框架设计，并报告了软件工程任务上的显著提升。Recursive Self-Evolving Agents [\[Nguyen et al.,](#page-10-0) [2026\]](#page-10-0) 则聚焦于自我改进智能体如何筛选候选方案。

Luo et al. 描述了另一种应用 [\[2026\]](#page-9-0)：利用元执行框架设计一个能将科学论文转化为会议海报的执行框架。这项工作有两方面贡献：提出构建元执行框架的方法，并提供一个具体的海报设计框架；作者也将后者用于自己的论文。
  
Shepherd 系统 [\[Yu et al.,](#page-11-0) [2026\]](#page-11-0) 提供了一个可运行的元智能体编程环境。<sup>2</sup> Shepherd 将 LLM 智能体定义为 Python 函数，因此可以监控、回滚、分叉、修改并恢复其执行轨迹。
  
 <sup>2</sup><https://shepherd-agents.ai> 

{8}------------------------------------------------

<span id="page-8-0"></span>早期实验在三个基准测试上都显示出明显的性能提升（见图 [20.5](#page-7-0)）。

Ren et al. 对自我改进型智能体设计进行了更全面的综述 [\[2026\]](#page-10-0)，其分类法明确纳入了这种两层架构。

## 20.3 结语

本章介绍了 Transformer 架构与智能体架构领域的一些进展。这两个领域目前都十分活跃。

## 参考文献

- Harold Abelson and Gerald Jay Sussman. *Structure and Interpretation of Computer Programs: JavaScript Edition*. MIT press, 2022. [397](#page-6-0)

- Jason Ansel, Petr Marchenko, Ulfar Erlingsson, Elijah Taylor, Brad Chen, Derek L Schuff, David Sehr, Cliff L Biffle, and Bennet Yee. Language-independent sandboxing of just-in-time compilation and self-modifying code. In *Proceedings of the 32nd ACM SIGPLAN conference on Programming language design and implementation*, pages 355–366, 2011. [397](#page-6-0)

- Cameron B Browne, Edward Powley, Daniel Whitehouse, Simon M Lucas, Peter I Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A survey of monte carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in games*, 4(1):1–43, 2012. [393](#page-2-0)

- Hongxu Cai, Zhong Shao, and Alexander Vaynberg. Certified self-modifying code. In *Proceedings of the 28th ACM SIGPLAN Conference on Programming Language Design and Implementation*, pages 66–77, 2007. [397](#page-6-0)

- Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, et al. Why do multi-agent llm systems fail? *arXiv preprint arXiv:2503.13657*, 2025. [397](#page-6-0)

- Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. *Advances in neural information processing systems*, 34:15084–15097, 2021. [392](#page-1-0)

- Xinyi Chen, Yifei Yuan, Jiaang Li, Serge Belongie, Maarten de Rijke, and Anders Søgaard. What if othello-playing language models could see? *arXiv preprint arXiv:2507.14520*, 2025. [393](#page-2-0)

- Can Demircan, Tankred Saanum, Akshay K Jagadish, Marcel Binz, and Eric Schulz. Sparse autoencoders reveal temporal difference learning in large language models. *arXiv preprint arXiv:2410.01280*, 2024. [395](#page-4-0)

{9}------------------------------------------------

<span id="page-9-0"></span>- Adrien Ecoffet, Jeff Clune, and Joel Lehman. Open questions in creating safe open-ended AI: Tensions between control and creativity. In *Artificial Life Conference Proceedings 32*, pages 27–35. MIT Press, 2020. [397](#page-6-0)

- Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In *International Conference on Machine Learning*, pages 11398–11442. PMLR, 2023. [394](#page-3-0)

- Benjamin Kempinski, Ian Gemp, Kate Larson, Marc Lanctot, Yoram Bachrach, and Tal Kachman. Game of thoughts: Iterative reasoning in game-theoretic domains with large language models. In *Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems*, pages 1088–1097, 2025. [393](#page-2-0)

- Donald E Knuth and Ronald W Moore. An analysis of alpha-beta pruning. *Artificial intelligence*, 6(4):293–326, 1975. [393](#page-2-0)

- Akshay Krishnamurthy, Keegan Harris, Dylan J Foster, Cyril Zhang, and Aleksandrs Slivkins. Can large language models explore in-context? *arXiv preprint arXiv:2403.15371*, 2024. [395](#page-4-0)

- Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, and Emma Brunskill. Supervised pretraining can learn in-context reinforcement learning. *Advances in Neural Information Processing Systems*, 36:43057–43083, 2023. [392](#page-1-0)

- Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-harness: End-to-end optimization of model harnesses. *arXiv preprint arXiv:2603.28052*, 2026. [397,](#page-6-0) [398](#page-7-0)

- Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=DeG07_TcZvT>. [393](#page-2-0)

- Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui, et al. Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses. *arXiv preprint arXiv:2604.25850*, 2026. [398](#page-7-0)

- Yaxin Luo, Haobin Jiang, Jialv Zou, Xu Huang, Wenhao Yan, Haodong Li, Zhengrong Yue, Jing Li, Xiaofu Chen, Xiaohan Zhao, et al. Autodesign: Meta-harness optimization for long-horizon agentic design. *arXiv preprint arXiv:2608.13560*, 2026. [398](#page-7-0)

- Larisa Markeeva, Sean McLeish, Borja Ibarz, Wilfried Bounsi, Olga Kozlova, Alex Vitvitskyi, Charles Blundell, Tom Goldstein, Avi Schwarzschild, and Petar

{10}------------------------------------------------

<span id="page-10-0"></span>Veličković. The CLRS-Text algorithmic reasoning language benchmark. *arXiv preprint arXiv:2406.04229*, 2024. [393](#page-2-0)

- Amir Moeini, Jiuqi Wang, Jacob Beck, Ethan Blaser, Shimon Whiteson, Rohan Chandra, and Shangtong Zhang. A survey of in-context reinforcement learning. *arXiv preprint arXiv:2502.07978*, 2025. [395](#page-4-0)

- Giovanni Monea, Antoine Bosselut, Kiante Brantley, and Yoav Artzi. LLMs are in-context reinforcement learners, 2024. [395](#page-4-0)

- Michael Nguyen, Quoc Nguyen, and Paul Vuong. Recursive self-evolving agents via held-out selection. *arXiv preprint arXiv:2606.28374*, 2026. [398](#page-7-0)

- Allen Nie, Yi Su, Bo Chang, Jonathan N Lee, Ed H Chi, Quoc V Le, and Minmin Chen. Evolve: Evaluating and optimizing llms for exploration. *arXiv preprint arXiv:2410.06238*, 2024. [395](#page-4-0)

- Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is Turing-complete. *Journal of Machine Learning Research*, 22(75):1–35, 2021. [392](#page-1-0)

- Zhe Ren, Yimeng Chen, Dandan Guo, Guowei Rong, Tonghui Li, RB Xiong, Qingfeng Lan, Wenyi Wang, Li Nanbo, Yibo Yang, et al. Self-improvements in modern agentic systems: A survey. *arXiv preprint arXiv:2607.13104*, 2026. [399](#page-8-0)

- Maxime Robeyns, Martin Szummer, and Laurence Aitchison. A self-improving coding agent. *arXiv preprint arXiv:2504.15228*, 2025. [397](#page-6-0)

- Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi. Reasoning with latent thoughts: On the power of looped transformers. *arXiv preprint arXiv:2502.17416*, 2025. [394](#page-3-0)

- Jürgen Schmidhuber. Gödel machines: Fully self-referential optimal universal self-improvers. In *Artificial general intelligence*, pages 199–226. Springer, 2007. [397](#page-6-0)

- John Schultz, Jakub Adamek, Matej Jusup, Marc Lanctot, Michael Kaisers, Sarah Perrin, Daniel Hennes, Jeremy Shar, Cannada Lewis, Anian Ruoss, et al. Mastering board games by external and internal planning with language models. *arXiv preprint arXiv:2412.12119*, 2024. [393](#page-2-0)

- Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Yanjun Qi, and Shangtong Zhang. Reward is enough: Llms are in-context reinforcement learners. *arXiv preprint arXiv:2506.06303*, 2025. [395,](#page-4-0) [396](#page-5-0)

- Richard S Sutton and Andrew G Barto. *Reinforcement Learning*. Adaptive Computation and Machine Learning series. Bradford Books, Cambridge, MA, 2 edition, November 2018. [394,](#page-3-0) [395](#page-4-0)

- Alan Turing. On computable numbers, with an application to the entscheidungsproblem. *J. of Math*, 58(345-363):5, 1936. [392](#page-1-0)

{11}------------------------------------------------

<span id="page-11-0"></span>- Jiuqi Wang, Ethan Blaser, Hadi Daneshmand, and Shangtong Zhang. Transformers learn temporal difference methods for in-context reinforcement learning. *arXiv preprint arXiv:2405.13861*, 2024. [392,](#page-1-0) [395,](#page-4-0) [396](#page-5-0)

- Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. *arXiv preprint arXiv:2311.12424*, 2023. [394](#page-3-0)

- Xiao-Wen Yang, Xuan-Yi Zhu, Ding-Chu Zhang, Wen-Da Wei, Jie-Jing Shao, Zhi Zhou, Lan-Zhe Guo, and Yu-Feng Li. Step back to leap forward: Self-backtracking for symbolic reasoning and planning in language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 40, pages 27657–27665, 2026a. [393](#page-2-0)

- Xiyuan Yang, Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, Jindong Jiang, Hanghang Tong, Tong Zhang, Markus J. Buehler, Jingrui He, and James Zou. Recursive multi-agent systems, 2026b. URL <https://arxiv.org/abs/2604.25917>. [394](#page-3-0)

- Simon Yu, Derek Chong, Ananjan Nandi, Dilara Soylu, Jiuding Sun, Christopher D Manning, and Weiyan Shi. Shepherd: Enabling programmable meta-agents via reversible agentic execution traces. *arXiv preprint arXiv:2605.10913*, 2026. [398](#page-7-0)

- Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, and Shuyue Hu. Self-harness: Harnesses that improve themselves. *arXiv preprint arXiv:2606.09498*, 2026a. [398](#page-7-0)

- Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin godel machine: Open-ended evolution of self-improving agents, 2026b. URL <https://arxiv.org/abs/2505.22954>. [397](#page-6-0)

- Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jeff Clune, Minqi Jiang, Sam Devlin, and Tatiana Shavrina. Hyperagents, 2026c. URL <https://arxiv.org/abs/2603.19461>. [397](#page-6-0)

- Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. Scaling latent reasoning via looped language models. *arXiv preprint arXiv:2510.25741*, 2025. [394](#page-3-0)

- Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. Agent-as-a-judge: Evaluate agents with agents. *arXiv preprint arXiv:2410.10934*, 2024. [396](#page-5-0)
