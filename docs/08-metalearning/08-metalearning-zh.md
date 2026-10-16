<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 181-188; Marker split PDF pages 1-8. -->

{0}------------------------------------------------

# 第 8 章 元学习

#### 学习目标

学完本章后，你将能够：

- 识别大语言模型（LLM）训练流程中的不同学习层级。
- 说明上下文学习、少样本学习与元学习之间的联系。
- 描述由内层学习循环和外层学习循环构成的元学习系统。
- 解释移动目标问题，以及动态元学习为何难以稳定。

# 8.1 为什么要研究元学习？

正如前文所述，自我反思式 LLM 方法很有潜力，但要让训练过程收敛并不容易。在智能体 LLM 系统中，学习可能发生在多个层级，有时还会同时进行。在机器学习中，能够跨多个层级进行学习的算法称为元学习。元学习常被概括为“学会如何学习”，例如，学会为某项任务选择更有效的提示。不同层级的学习也可能彼此干扰，导致学习无法收敛。

为了理解 LLM 多层级学习算法为何有时会失效，本章将进一步介绍元学习，并讨论如何提高动态、富含反馈的元学习方法的稳定性。

元学习是指跨多个层级进行学习的算法。它们学习如何适应任务，从而只需少量示例，就能解决模型此前未曾专门训练过的新任务。

{1}------------------------------------------------

### <span id="page-1-0"></span>8.1.1 深度学习速度较慢

元学习在机器学习领域有着悠久的历史（例如，参见 [Schmidhuber，1987](#page-7-0)、[Hospedales 等，2021](#page-6-0) 和 [Huisman 等，2021](#page-6-0)）。元学习借鉴了人类学习的特点：人们常被认为能够学会如何学习。提高学习速度也是元学习发展的动因之一。

传统深度学习的训练速度较慢。典型任务使用的神经网络可能包含数百万乃至数十亿个参数，训练还需要规模庞大的数据集，往往包含数百万个样本。从零开始训练网络可能需要数小时甚至数月。专用并行硬件（如 GPU）的出现，是深度学习得以广泛应用的重要原因之一 [Krizhevsky 等，2012](#page-7-0)。

因此，研究者开始探索少样本学习：只用少量示例训练，就能达到较高准确率的方法。

#### 8.1.2 实现少样本学习

不同的机器学习算法各有适用的问题类型。多数训练算法也包含可调设置，以改善性能，例如学习率、动量、参数初始化方式；对于多层算法，还可以选择底层训练算法。这些设置称为*超参数*，它们决定算法如何学习。我们能否从数据中学到合适的超参数？

不同问题可能需要不同的超参数设置。例如，对于较平滑的数据集，较高的学习率或局部搜索算法可能效果较好；对于起伏较大的数据集，较低的学习率或全局搜索算法可能更合适。最佳配置取决于要解决的任务，而配置本身也可以学习。跨不同任务学习如何配置算法，称为*元学习*。与此相对，学习模型的常规参数称为*基础学习*。

元学习的目标是让算法更快地学会新任务。与其从零开始套用通用算法，不如先利用一组相关任务，学习如何为将要处理的任务类型配置学习算法。

学到合适的配置后，模型或许只需少量示例就能完成训练，并在新任务的未见样本上取得较高准确率。元学习通过利用相关知识而非从零开始，来支持少样本学习。

元学习由一个外层（元训练）循环和一个内层（基础训练）循环组成。在经典机器学习中，外层可能选择学习率或学习算法本身，并在多个训练任务上评估不同算法 [Brazdil 等，2022](#page-6-0)。在深度学习中，神经网络权重的初始化常被用作元学习方式之一 [Finn 等，2017](#page-6-0)。智能体 LLM 中也存在不同层级的学习，我们将在下一章继续讨论。元学习领域的理论与研究成果有助于分析这些系统。

{2}------------------------------------------------

<span id="page-2-0"></span>

## 8.2 内循环与外循环

与其他监督学习方法类似，元学习包含*元训练*阶段和*元测试*阶段：前者在一组相关任务上训练，后者则在一个此前未见的新任务上测试。元学习的目标是学到一组在多种训练任务上都适用的训练算法超参数，从而提高模型的泛化能力。

模型参数通过优化循环学习。损失函数衡量模型预测与训练样本之间的差异；学习算法据此更新参数，以减小损失。循环会遍历数据集并调整模型参数，目标是尽量降低总损失。

学会学习包含两个循环。*内循环*针对单个任务中的示例优化模型参数；*外循环*则针对某个应用领域中的一组任务，优化学习算法的超参数。换言之，内循环学习如何完成某个任务，外循环学习如何跨任务配置学习过程。

设基础学习任务 \(\mathcal{T}_i\) 的数据集为 \(D_i\)，损失函数为 \(\mathcal{L}_i\)，即 \(\mathcal{T}_i=(D_i,\mathcal{L}_i)\)。数据集由输入与标签对组成，\(D_i=\{(x_j,y_j)\}\)，并划分为训练集和测试集：\(D_i=\{D_{i,\mathrm{train}},D_{i,\mathrm{test}}\}\)。对于每项任务，模型 \(\hat f_{\theta_i}(D_{i,\mathrm{train}})\) 通过最小化训练损失 \(\mathcal{L}_i(\theta_i,D_{i,\mathrm{train}})\) 得到估计。模型 \(\hat f\) 由一个包含超参数 \(\omega\) 的算法定义。根据算法不同，超参数可能包括学习率 \(\alpha\)、初始参数 \(\theta_0\) 及其他常数。

内循环优化基础参数 \(\theta_i\)，外循环优化超参数 \(\omega\)，使模型在一组基础任务上表现良好。简化表示如下：

\[
\omega^* = \arg\min_{\omega}\; \mathcal{L}\!\left(\arg\min_{\theta_i}\mathcal{L}_{\omega}(\theta_i,D_{i,\mathrm{train}})\right)
\]

内层优化利用各任务的训练数据求出参数 \(\theta_i\)；外层优化则在多项任务上评估由此得到的学习过程，并选择 \(\omega\)。最终得到优化后的超参数集合 \(\omega^*\)。

元学习完成后，经过调整的算法应能适用于某一应用领域中的一组任务。我们可以期待它在新任务上只看少量示例，就能对未见样本取得较高准确率。

{3}------------------------------------------------

<span id="page-3-0"></span>
### 迁移学习

元学习与迁移学习密切相关。迁移学习先在一个任务上预训练神经网络，再将其权重迁移到后续的微调阶段。因此，迁移学习可以看作基于网络初始化的元学习方法的一次迭代，也就是只有一项基础学习任务的元学习。

# 8.3 大语言模型与元学习

元学习出现在 LLM 开发的多个环节。一种常见方法是用先前学到的知识初始化模型，使它能更快地学习相关任务。这就是迁移学习，标准 LLM 流程中的预训练和微调阶段都会用到它。反复微调可以看作一种元学习过程；LLM 是通过训练流程支持少样本学习的典型例子。上下文学习也是机器学习领域的一项重要突破 [Brown 等，2020](#page-6-0)。

在智能体 LLM 系统中，少样本适应还有其他形式。本节将介绍静态和动态元学习。在静态元学习中，一个循环的权重不会因其他循环的结果而改变；在动态元学习中，不同循环会相互影响。

#### 8.3.1 静态 LLM 元学习

先来看几个 LLM 静态元学习的例子。

#### 8.3.1.1 预训练流程

标准 LLM 训练流程是一种复杂的多阶段迁移学习方案。它可以取得较强的零样本表现，例如，基于预训练和微调获得的知识，回答“莫扎特出生在哪里？”这类常见问题。

流程中的各阶段是内层优化循环。关于迁移哪些权重、如何在后续阶段微调的决策，则可以视为外层循环的超参数选择。

这属于静态元学习，因为某一阶段训练结束后，其权重不会受到后续阶段训练结果的影响。各阶段之间的权重保持固定，训练按顺序进行。

#### 8.3.1.2 上下文学习

从零样本学习进一步发展到上下文学习后，用户可以在提示中提供少量示例，让 LLM 据此生成更好的答案。上下文学习是一种少样本学习形式。预训练模型相当于内循环，提示中的示例则引导外循环。

{4}------------------------------------------------

这属于静态元学习，因为提示中的示例不会改变模型权重。

#### 8.3.1.3 思维链

普通上下文学习以*事实*作为少样本示例，侧重内容；思维链提示则以*步骤或过程*作为示例，侧重推理过程。

同样，预训练模型是内循环，提示示例引导外循环，而模型权重保持不变。

#### 8.3.1.4 静态元学习的稳定性

标准 LLM 流程由依次运行的迁移学习阶段（内循环）组成。内层训练循环一旦收敛，该阶段便不会再次运行。微调结果不会反馈给预训练器，预训练器也不会根据微调结果调整超参数后重新训练。由于各阶段之间没有反馈，它们不会通过相互作用引发不稳定。

动态元学习则不同，其中的迭代过程可能彼此影响。

### 8.3.2 动态 LLM 元学习

智能体 LLM 系统中存在多种动态元学习形式。不同迭代可能并行运行，并对彼此的收敛过程产生不稳定影响。

前文介绍过自我反思和 RLVR。在这些方法中，外循环可以改变内循环，一个内循环的结果也可能影响另一个内循环。这会形成不断变化的训练目标，进而导致训练发散。结果还可能很脆弱：随机种子、提示措辞或元学习过程持续时间等输入或超参数的微小变化，都可能引起显著差异。此外，同一模型若承担生成、评估等不同角色，其偏差和幻觉倾向也可能被放大。

下面来看智能体 LLM 中的动态元学习示例。

#### 8.3.2.1 自我反思

在自我反思（第 6.2.4.2 节）中，智能体执行框架负责管理提示，并以不同角色调用 LLM，例如评估当前状态或改进答案。

从元学习角度看，*模型*是内层学习循环，*执行框架*是外层循环。虽然模型权重保持不变，执行框架会根据模型的回答生成新的提示。这样的反馈会使外层迭代以非线性且难以预测的方式相互作用。

如果同一个模型被用于不同角色，其缺陷（包括偏差和幻觉）也可能被放大。

{5}------------------------------------------------

#### <span id="page-5-0"></span>8.3.2.2 RLVR

带可验证奖励的强化学习（RLVR；参见第 6.2.4.3 节）常用于代码和数学任务：模型在推理阶段生成的结果，会在训练阶段用于微调。

从元学习角度看，微调是内循环，推理阶段的推理过程是外循环。RLVR 将外循环的结果反馈给内循环的训练阶段，使两个学习过程相互耦合。需要谨慎处理被当作真实答案的推理结果，以免推理阶段的错误在反复微调与推理中被放大。

# 8.4 稳定动态元学习

当内外循环之间的相互影响缺乏足够阻尼时，训练可能变得不稳定。循环之间的振荡可能不断增大，而不是逐渐消退。

缓解方法之一是增加阻尼，或让不同循环以不同速度变化。例如，可以降低其中一个循环的学习率，或让某些内循环参数比其他参数更长时间保持固定。另一个办法是使用类型或规模不同的模型，例如混合专家模型 [Jiang 等，2024](#page-6-0)。学生—教师架构则有意安排能力较强和较弱的模型承担不同角色。使用不同模型或许能够减轻幻觉放大和循环间干扰。

更一般地说，当元学习表现不佳时，可以引入更多多样性以促进探索，减少导致局部最优或训练发散的贪心行为。找到有效的解决办法通常需要通过实验积累经验。调试智能体 LLM 系统可能颇具挑战。

# 8.5 结语

LLM 训练可以理解为一种元学习。标准 LLM 训练使用迁移学习，而智能体 LLM 系统中的推理时推理与训练则进一步运用了元学习思想。自我反思等动态方法很有潜力，但内循环与外循环之间的反馈可能使训练过程不稳定。即使是此前表现可靠的智能体，也可能意外出现这种问题。

如果智能体使用已训练好的模型与世界交互，并且还根据交互反馈继续训练，那么可以把模型视为基础学习器（内循环），把与世界交互的过程视为元学习器（外循环）。前文关于训练流程内部元学习的注意事项，也适用于训练流程之外的元学习。不过，在真实世界中交互时，错误可能造成更严重的后果。

在线机器学习（包括深度强化学习）同样面临训练不稳定和目标不断变化的问题。<span id="page-6-0"></span>深度强化学习提出了多种稳定化技术，例如重放缓冲区和目标网络 [Mnih 等，2015](#page-7-0)；Rainbow 也是一例 [Hessel 等，2018]。这些方法通常通过错开两个学习过程的时间节奏来减少耦合，例如放慢其中一个过程。分层推理模型（HRM）也采用了类似思路 [Wang 等，2025](#page-7-0)：它使用两层元学习，并通过让两个层级以不同速度运行来稳定学习。

设计智能体 LLM 系统时，了解这些稳定化方法很有帮助。下一部分将概述智能体 LLM 与世界交互的不同方式，从静态方式逐步过渡到动态方式。
# 参考文献

Pavel Brazdil, Jan N Van Rijn, Carlos Soares, and Joaquin Vanschoren. *Metalearning: Applications to automated machine learning and data mining*. Springer Nature, 2022. [182](#page-1-0) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. [184](#page-3-0) Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *International conference on machine learning*, pages 1126–1135. PMLR, 2017. [183](#page-2-0) Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018. 187 Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Metalearning in neural networks: A survey. *IEEE transactions on pattern analysis and machine intelligence*, 44(9):5149–5169, 2021. [182](#page-1-0) Mike Huisman, Jan N Van Rijn, and Aske Plaat. A survey of deep meta-learning. *Artificial Intelligence Review*, 54(6):4483–4541, 2021. [182](#page-1-0) Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep ´ Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, ´ Thibaut Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed. Mixtral of ´ experts, 2024. URL <https://arxiv.org/abs/2401.04088>. [186](#page-5-0)

{7}------------------------------------------------

<span id="page-7-0"></span>Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012. [182](#page-1-0) Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533, 2015. [187](#page-6-0) Jurgen Schmidhuber. ¨ *Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook*. PhD thesis, Technische Universitat¨ Munchen, 1987. ¨ [182](#page-1-0) Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model. *arXiv preprint arXiv:2506.21734*, 2025. [187](#page-6-0)