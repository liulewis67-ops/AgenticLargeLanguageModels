<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 363-390; Marker split PDF pages 1-28. -->



{0}------------------------------------------------

# <span id="page-0-0"></span>第 19 章

# \*用于训练智能体大语言模型的合成数据生成

#### 你将学到的内容

本章介绍合成数据在智能体大语言模型开发中的作用。学完本章后，你将能够：

- 描述*数据*在 LLM 训练流程中的重要性。
- 解释从更强的模型向较弱的学生*提炼知识*的一般循环。
- 描述用于替代人工生成内容的合成数据的*流水线组件*。
- 概述用于*策划和评估*合成数据的各种策略。
- 通过分析人工生成数据的*风险*，比较使用合成数据的利弊。

#### 19.1 数据的重要性

LLM 快速扩展，显著推高了对训练和评估数据的需求。有人把训练数据称为*数字石油*，以强调其对 AI 发展的重要性。随着模型架构等其他环节逐渐趋同，数据也日益成为竞争差异所在 [\[Kim et al.,](#page-18-0) [2025a\]](#page-18-0)。模型训练需要从互联网抓取数万亿词的人类文本，但这类数据终究有限，因此训练数据是否会成为规模扩展的主要瓶颈，值得关注（见图 [19.1](#page-1-0) 和 [Villalobos et al.（2024）](#page-23-0)）。随着数据需求增长，获取数据的成本和耗时也不断增加，许多容易取得的数据源正在枯竭。隐私、版权和偏差问题也促使研究者超越传统的网络文本筛选，转向合成数据生成 [\[Liang et al.,](#page-19-0) [2022\]](#page-19-0)。

{1}------------------------------------------------

<span id="page-1-0"></span>![](assets/_page_1_Figure_2.jpeg) 
  
 图 19.1：用于训练 LLM [\[Villalobos et al.,](#page-23-0) [2024\]](#page-23-0) 的可用人工生成公开文本和数据集规模的预测。库存和数据集规模预测线的交点表示预期库存将被充分利用的中位年份（2028 年）。
  
LLM 越来越多地生成训练下一代 LLM 本身所需的数据：模型被提示生成指令、对话、推理痕迹，甚至整个训练语料库，这些随后被用来微调、蒸馏和评估其他模型 [\[Long et al.,](#page-19-0) [2024\]](#page-19-0)。与获取和标注人工生成文本相比，这种自我强化的循环降低了数据收集的成本和工作量，同时也提供了对生成数据集的风格、难度和覆盖范围的精细控制。到目前为止，这种方法已经引起了显著的工业 [\[Kapania et al.,](#page-18-0) [2025\]](#page-18-0) 和学术兴趣 [\[Viswanathan et al.,](#page-23-0) [2025,](#page-23-0) [Alismail and Lanquillon,](#page-14-0) [2025\]](#page-14-0)，应用范围从通用指令调优到训练智能行为所需的更专业数据，例如工具使用、规划和多步推理（如第17页图1.1所示）。

对模型生成数据日益增长的依赖，也使缩放定律纳入了一个新变量：预训练语料中的合成数据占比。SYNTHLLM 等近期研究 [\[Qin et al.,](#page-21-0) [2025\]](#page-21-0) 展示了这一变量的重要性：该方法从源文档中自动提取高层概念，再通过图算法重新组合，生成多样的合成语料，使不同规模的模型呈现可预测、经修正的缩放规律。在这种缩放规律下，性能收益会更早趋缓（早于预训练中常用的其他词元预算），较大模型达到最佳性能所需的词元也比小模型少。这说明，合成数据不仅能调节数据量，也能影响由此产生的缩放曲线形态。

除规模扩展外，生成工具使用、规划和多步推理等智能体能力所需的数据，也越来越依赖多智能体协作，而非单个生成模型。

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Diagram_2.jpeg) 
  
 图 19.2：组织本章内容的合成数据循环。生成器从一个小的种子集生成原始合成数据，然后经过整理、过滤和评分。数据用于训练或蒸馏一个模型，该模型随后被评估，并可以作为下一次迭代的生成器（虚线）。

专业化智能体可以分别承担提出任务、执行任务和评议结果等职责，通常能生成质量更高、更多样且结构更贴近智能体训练需求的数据。Matrix [\[Wang et al.,](#page-24-0) [2025a\]](#page-24-0) 是这一方向的一个例子：该去中心化框架把控制流和数据流都表示为智能体之间通过分布式队列传递的消息，因此无需中央协调器也能横向扩展。下面进一步看一看合成数据训练流程。

#### 19.2 蒸馏回路

在智能体 LLM 训练流程中使用合成数据，整体上遵循图 19.2 所示的蒸馏循环。生成器（一个强模型，或由多个智能体组成的编排系统）从少量种子数据或真实数据出发，生成大量原始合成数据。这批数据随后会经过*筛选整理*（参见第 [19.4](#page-9-0) 节），再用于训练或*蒸馏*模型。关键在于，训练得到的模型本身还可以成为下一轮迭代的生成器。这种自我强化的闭环可能严重影响训练稳定性和下游性能（另见第 8 章关于元学习稳定性的讨论）。

蒸馏是合成数据生成的一种特殊情形：生成器是*能力强于学生、且其能力正是我们希望迁移的模型*。在知识蒸馏中，紧凑的*学生模型*学习模仿能力更强的*教师模型* [\[Hinton et al.,](#page-17-0) [2015,](#page-17-0) [Hu et al.,](#page-17-0) [2022\]](#page-17-0)。学生可以直接针对教师的输出预测进行优化，也可以只用教师生成的合成数据训练。蒸馏旨在以更低的推理和数据成本获得教师模型的高效版本，同时尽量保留其任务性能。蒸馏已广泛用于智能体场景，例如 [Li et al.（2025a）](#page-19-0) 和 [Du et al.（2025）](#page-16-0) 的工作。蒸馏也可用于为指令微调构造偏好数据。例如，Zephyr [\[Tunstall et al.,](#page-23-0) [2024\]](#page-23-0) 通过 AI 反馈训练：由更强的教师模型生成并排序输出，以较少的数据和训练成本使小模型与用户意图对齐。

{3}------------------------------------------------

<span id="page-3-0"></span>**蒸馏的四个维度。**具体方法可以从四个维度进行分类：（1）*信号*：学生模仿的是教师的 logits、完整输出序列，还是教师给出的标量奖励；（2）*数据来源*：被评分的序列由教师生成，还是从学生模型采样；（3）*时机*：数据是预先一次性生成（离策略），还是随着学生模型变化而持续生成（在策略）；（4）*教师身份*：教师是更强的独立模型，还是学生模型自身。

我们现在将讨论这四个蒸馏轴。

#### 19.2.1 信号：蒸馏目标

当我们以*白盒*方式获取教师模型的概率分布 $q$ 时，会训练学生模型 $p_\theta$，使其下一个词元的概率分布与教师模型相匹配。蒸馏时通常用共享温度 $\tau$ 对分布进行软化（对于 logits $z$，$q^\tau(\cdot) \propto \exp(z/\tau)$）。在序列语料上，通用的蒸馏损失可写为：

$$
\mathcal{L}(\theta)=\mathbb{E}_{x}\left[\sum_t D\left(q^\tau(\cdot\mid x_{<t})\,\|\,p^\tau_\theta(\cdot\mid x_{<t})\right)\right]
\tag{19.1}
$$

其中，求和遍历词元位置 $t$，$D$ 表示某种散度（最常用的是 KL 散度）。温度 $\tau>1$ 会使两个分布更平缓，让学生也能学习教师模型的*相对*置信度。若只能获取教师生成的*文本*（黑盒蒸馏，例如从专有 API 蒸馏时的常见情形），我们无法直接观察 $q$，只能看到从中采样得到的词元。此时，软目标 $q^\tau(\cdot\mid x_{<t})$ 会退化为采样词元上的 one-hot 向量，散度 $D$ 则变为词元级交叉熵，公式 [19.1](#page-3-0) 因而退化为针对教师生成内容的常规监督微调。第 [19.3](#page-5-0) 节将介绍 Self-Instruct [\[Wang et al.,](#page-24-0) [2023b\]](#page-24-0)；这种方法从数据侧实施蒸馏，因此属于黑盒方法。

#### 19.2.2 数据来源：教师—学生范式

在蒸馏中，直观的假设是更大、更强的教师总是更好。然而，也存在例外 [\[Xu et al.,](#page-25-0) [2025a,](#page-25-0) [Li et al.,](#page-19-0) [2025b\]](#page-19-0)。一个有能力的教师可能产生的示例过于冗长、过于复杂，或者风格与小型学生不匹配，导致学生无法吸收。这一现象被称为*小模型可学习性差距*。图 [19.3](#page-4-0) 展示了一种情况，其中小型学生从较弱教师的较短推理中学到的东西比从前沿模型的复杂思维链中学到的更多。实际意义是，教师的选择应当根据学生当前的能力进行，而不是单独追求最大化 [\[Zhang et al.,](#page-26-0) [2026a\]](#page-26-0)。

#### 19.2.3 时间安排：离策略与在策略蒸馏

第三个维度涉及数据生成的时机。最常见的做法是*离策略蒸馏*：先从教师模型采样一批固定的输出序列，再

{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Figure_2.jpeg) 
  
  图 19.3：小模型可学习性差距的示意图。小型学生模型未必能从较长的思维链推理中持续获益；使用较小教师模型给出的较短思维链进行微调，效果可能更好 [\[Li et al.,](#page-19-0) [2025b\]](#page-19-0)。

然后用这些序列微调学生模型（如前述黑盒蒸馏）。这种方法简单且易于并行，但会造成训练与推理不匹配：训练期间，学生只接触教师生成的序列；部署时，它却必须接续生成自己的、可能有缺陷且与训练数据不同的序列。这种*暴露偏差*会沿着长程智能体轨迹不断累积。

*在策略蒸馏*则通过从学生自身采样序列、再由教师为其评分来应对这一问题，整个过程都在循环中进行。广义知识蒸馏（GKD）利用教师对这些序列的反馈，在学生自己生成的输出上训练学生 [\[Agarwal et al.,](#page-14-0) [2024\]](#page-14-0)。因此，纠正直接作用于学生实际访问的状态分布。

这种在策略的视角明确展现了蒸馏与强化学习、尤其是模仿学习之间的联系。公式 [19.1](#page-3-0) 中逐词元计算的教师—学生散度，可作为*密集的词元级奖励*。如此一来，蒸馏就转化为一种强化学习：由教师提供奖励信号，而不是依赖稀疏的结果检查器 [\[Lu and Lab,](#page-20-0) [2025\]](#page-20-0)。混合方案更进一步，在强化学习阶段只在有帮助时选择性模仿教师 [\[Zhang et al.,](#page-26-0) [2026b\]](#page-26-0)。
 
#### 19.2.4 教师身份：自蒸馏与自我提升

教师模型和学生模型不必是不同的模型。如果两者相同，蒸馏就成为自我提升：模型生成候选解答，通过检查后保留这些解答，并据此继续训练。STaR [\[Zelikman et al.,](#page-26-0) [2022a\]](#page-26-0) 就以这种方式引导推理：反复生成推理过程，保留得出正确答案的过程，再用它们进行微调。

另一种方法是自博弈微调（Self-Play Fine-Tuning，SPIN） [\[Chen et al.,](#page-15-0) [2024\]](#page-15-0)。模型在多轮迭代中区分自身生成的内容与参考数据，从而无需外部教师也能改善对齐。或许出人意料的是，即使没有新的数据源，*重复自蒸馏*仍能带来可测量的收益，但收益并非无限 [\[Pareek et al.,](#page-20-0) [2024\]](#page-20-0)。

{5}------------------------------------------------

<span id="page-5-0"></span>
 
当候选输出能够以较低成本*验证*时，自我提升最有效。此时仍可有效利用外部信号，例如代码和数学问题的答案易于执行或检查。以 CURE 为例，这是一个强化学习框架，它依据模型编码与单元测试生成之间的互动结果设计奖励，使两种能力共同进化；训练并不依赖标准答案代码作为监督。由于代码易于运行和检查，编码模型与测试编写模型可以相互适应，从而在困难问题上进行长程优化。自我挑战和自我进化智能体也遵循类似思路：它们自行提出越来越难的任务，并随时间更新生成策略 [\[Zhou et al.,](#page-26-0) [2025,](#page-26-0) [Huang et al.,](#page-17-0) [2025,](#page-17-0) [Wang et al.,](#page-24-0) [2025b\]](#page-24-0)。

#### 19.3 模拟人类生成的内容

从自然产生的人类生成文本（无论是直接获取的还是从网络抓取的）转变开，或许可以最好地体现在从精心策划的人类生成指令数据向合成生成在其中扮演更核心角色的数据集的演变中。我们现在将讨论这些方法，这些数据集的核心仍然是基于人类生成的数据。

作为一个资源示例，人类仍然支撑了大部分数据，Natural Instructions [\[Mishra et al.,](#page-20-0) [2022\]](#page-20-0) 包含 61 个不同的任务、它们的人类编写的指令以及 193k 个任务实例（输入-输出对）。这些指令是从现有 NLP 数据集的指令文档中收集的，全部映射到一个统一的模式中。LLM 的实验表明，模型在观察指令及示例时会受益，尽管这些模型的表现仍远低于估计的性能上限。

后续的工作集中在增加独特任务和任务类型的数量 [\[Wang et al.,](#page-24-0) [2022\]](#page-24-0)、改变指令的长度 [\[Bach et al.,](#page-14-0) [2022\]](#page-14-0)，或包含特别困难的任务 [\[Srivastava et al.,](#page-23-0) [2023\]](#page-23-0)。重要的是，所有这些元数据集（数据集的集合）最终仍然是基于人工基准构建的：其中的每个样本都是由人编写的。因此，可以在网上自由获得的数据有一个实际的限制。

由于这些方法依赖于大量人类监督，以众包数据集的形式进行，最近的工作集中于生成能够填补相同角色的合成数据。在这里，我们看到了诸如 Unnatural Instructions [\[Honovich et al.,](#page-17-0) [2023\]](#page-17-0) 之类数据集的创建，这些数据集包含创造性和多样化的指令，其收集几乎不需要人工劳动，而是依赖于提示各种 LLM。实验表明，在稍微噪声较大的 Unnatural Instructions 数据集上训练的模型，其结果与在高质量、人类策划的数据上训练的模型相当，如图 [19.4.](#page-6-0) 所示。该结果表明，机器生成的数据为扩展和多样化训练数据集提供了一种成本效益高的替代人力的方式。

生成模型对类人文本、代码和推理的模拟为训练和评估机器学习系统开辟了新的方法。通过对强大的模型进行提示，研究人员可以生成高度多样化和复杂的数据集，这些数据集模拟人类内容的生成。例如，PersonaHub [\[Ge](#page-17-0) 

{6}------------------------------------------------

<span id="page-6-0"></span>![](assets/_page_6_Figure_2.jpeg) 
  
 图 19.4：比较非自然指令与超自然指令的扩展实验 [\[Honovich et al.,](#page-17-0) [2023\]](#page-17-0)，控制数据集规模下的模型性能，在两个基准测试上进行测试：超自然指令（左）和 LMentry（右）。
  
 [et al.,](#page-17-0) [2025\]](#page-17-0) 利用大量的角色集合来引导生成过程，使用它们来提示 LLM，以生成比无条件生成更多样的合成数据。

与自我提升类似，在模型输出较易验证的领域，合成数据训练已显示出很大潜力，例如数学 [\[Yang et al.,](#page-25-0) [2023,](#page-25-0) [Liu et al.,](#page-19-0) [2023\]](#page-19-0)、编程 [\[Haluptzok et al.,](#page-17-0) [2023\]](#page-17-0) 和游戏 [\[Wang et al.,](#page-24-0) [2023a\]](#page-24-0)。

#### 19.3.1 生成训练数据

接下来重点讨论训练数据的生成。合成数据在训练中的应用，已从小众的数据增强技巧发展为现代模型对齐和推理流程的核心组成部分。Self-Instruct 等技术 [\[Wang et al.,](#page-24-0) [2023b\]](#page-24-0) 表明，小型开放权重模型可以利用更强的专有模型生成的数据进行指令微调。如图 [19.5](#page-7-0) 所示，该方法先用少量人工编写的指令为 LLM 提供种子，再提示模型迭代生成新的任务描述、输入和输出。模拟人类生成内容后，模型无需大量人工标注者，也能获得复杂的零样本能力。

**利用现有数据。**另一种做法是将经过筛选的人类数据与合成数据结合，而不是让后者取代前者。[Gunasekar et al.](#page-17-0) [\[2023\]](#page-17-0) 发现，混合人类编写的教材和合成文本进行训练，可以得到能力很强的语言模型。这说明，经过筛选的人类数据与机器生成数据是互补关系，而非替代关系。合成数据的一个问题是，即使以人类数据为种子，也可能缺乏多样性，并继承错误和幻觉。后续改进尝试基于真实文档生成数据 [\[Lupidi et al.,](#page-20-0) [2025,](#page-20-0) [Yue et al.,](#page-26-0) [2024\]](#page-26-0)。

{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Diagram_2.jpeg) 
  
 图 19.5：自我指导管道 [\[Wang et al.,](#page-24-0) [2023b\]](#page-24-0)。

扩充有限种子集的另一种方法是*回译*。它最初用于机器翻译：将句子从源语言译为目标语言，再译回源语言，以合成平行语料 [\[Edunov et al.,](#page-16-0) [2018\]](#page-16-0)。这一思路也可用于推理任务：除了为已有问题生成解答，还可以根据已有解答构造新问题。MathGenie [\[Lu et al.,](#page-20-0) [2024\]](#page-20-0) 将该方法用于数学推理，从一组问题—解答对出发扩展解答，再把它们回译为新问题。由于每个新问题都建立在经过验证的解答之上，该方法既扩大了训练数据规模、提升了多样性，也减少了错误和幻觉。

**人工参与的流程。**全自动生成并非总是足够。模型可能难以生成特定情景或与特定领域相关的案例，通常是因为生成 LLM 存在自身的盲点，也因此难以产出真正有信息量的数据 [\[Lippmann et al.,](#page-19-0) [2024\]](#page-19-0)。
 
这些问题促使研究者采用*人工参与的流程*：由人来引导、筛选或修改机器生成的内容，而不是从头创作每个样本。WANLI [\[Liu et al.,](#page-19-0) [2022\]](#page-19-0) 让语言模型提出有挑战性的自然语言推理样例，再由人工标注者筛选和重新标注，兼顾自动生成的规模优势与人工质量控制。

EvalAssist [\[Santillán Cooper et al.,](#page-22-0) [2025\]](#page-22-0) 将类似思路用于评估工具，借助合成数据支持并加速人工判断。另一种做法是先让语言模型生成种子文本，再由人工评分者编辑，之后以编辑结果为起点构建更大规模的全合成数据集。这样，人力主要用于纠正模型，而无需逐条撰写样本。

{8}------------------------------------------------

#### <span id="page-8-0"></span>19.3.2 生成评估数据

除了训练之外，合成生成也可用于评估。若测试题泄露到现代 LLM 的预训练语料库中，静态基准就可能受到污染。Dynabench 等动态评测框架通过让人和模型共同参与持续构造有挑战性的样例，来应对静态基准的相关局限 [\[Kiela et al.,](#page-18-0) [2021\]](#page-18-0)。这种人机协同的数据构建过程不应与合成数据生成混为一谈。

此外，*LLM-as-a-judge* 范式 [\[Zheng et al.,](#page-26-0) [2024\]](#page-26-0) 通过提示能力较强的模型为其他模型的输出打分，模拟人工评估。此类评分与人类偏好的相关性较高，也能显著缩短评估周期。

也可以通过人工参与的方式，为 LLM 构建专门的高难度基准，例如 MATH<sup>2</sup>。在该基准的构建过程中，多个 LLM 通过多轮提示迭代生成并调整数学题及解答；随后，人工标注者在 LLM 辅助下扩展题目并核查答案 [\[Shah et al.,](#page-22-0) [2024\]](#page-22-0)。另一种高效生成合成评估数据的做法，是让标注者编辑已有样本，使其更难回答，而非从头编写新样本 [\[Yuan et al.,](#page-25-0) [2021\]](#page-25-0)。

**自动化红队测试。**机器生成评估的一种具体用途，是通过*自动化红队测试*开展面向安全的评估。一个 LLM 会生成对抗性提示，在模型部署前探测其漏洞 [\[Ganguli et al.,](#page-16-0) [2022\]](#page-16-0)。研究人员可以批量生成潜在恶意提示，对模型进行压力测试，并提升其抵御有害输出、越狱和行为失调的能力；这种规模是人工红队难以企及的 [\[Perez et al.,](#page-21-0) [2022\]](#page-21-0)。由于这些提示随后还可用于微调目标模型、弥补已发现的缺陷，红队测试处于合成评估数据与合成训练数据的交界处。

#### 19.3.3 生成智能体数据

在智能体场景中，合成数据必须覆盖完整轨迹，而不只是单轮指令—响应对。智能体与环境交互时，会产生一连串工具调用、观察结果和决策 [\[Kang et al.,](#page-18-0) [2025\]](#page-18-0)。这类数据超出了常规 LLM 在单个输入—输出对层面进行训练或评估的需求。即使两者通常由相同的生成模型产出，合成智能体数据也比用于训练底层 LLM 的数据更复杂。下面看一个例子。

**示例 2。**考虑为一个工具使用型智能体合成一条训练轨迹，流程大致遵循 APIGen-MT [\[Prabhakar et al.,](#page-21-0) [2025\]](#page-21-0) 的两阶段方案。

{9}------------------------------------------------

<span id="page-9-0"></span>首先，*任务提议模型*会构造一个合理的用户目标、可用工具集合以及标准答案：

*用户：*“莱顿这个周末天气如何？我需要带伞吗？” *工具：* `get_forecast(city, days)`。

随后，*执行模型*展开交互，发出中间工具调用，并在回答前读取（模拟的）工具返回结果：

`get_forecast("Leiden", 2)` → `{周六：有雨，周日：晴朗}` → *“周六会下雨，周日晴朗；周六记得带伞。”*

最后，*验证器*会检查工具调用格式是否正确、参数是否符合请求，以及最终答案是否与工具返回结果一致。只有通过验证的轨迹才会保留。

请注意，这里需要合成的不只是一个输入—输出对，还包括用户目标、工具模式、工具调用、观察结果，以及能够验证完整轨迹的检查器。

智能体数据生成通常需要编排多个智能体，以完成包含人工输入和工具使用的多轮对话 [\[Wu et al.,](#page-24-0) [2024\]](#page-24-0)。APIGen-MT [\[Prabhakar et al.,](#page-21-0) [2025\]](#page-21-0) 通过两阶段框架生成可验证且多样化的多轮智能体数据。TOUCAN [\[Xu et al.,](#page-25-0) [2025b\]](#page-25-0) 则通过模拟真实工具调用，以更大规模构建工具智能体数据集。除了为既定技能生成轨迹，一些研究还明确合成数据，将强模型的新技能或行为迁移给较弱模型，这一过程称为*生成式教学* [\[Crispino et al.,](#page-15-0) [2024\]](#page-15-0)。面向智能体的合成数据也不限于文本；类似流程还可生成图像、表格数据，以及其他交互或行为数据。

生成智能体合成数据时，一个反复出现的挑战是控制任务复杂度。例如，深度研究智能体需要通过长程在线工具交互解决复杂问答，但训练数据往往无法精细控制难度和质量，导致合成任务不够复杂。[Pandit et al.](#page-20-0) [\[2025\]](#page-20-0) 提出一种流程，逐步提高问答任务的复杂度，直到前沿基线网络智能体无法完成。这种课程学习方法依据智能体能力直接校准任务难度，而不是依赖固定启发式规则。

AReaL-SEA [\[Gao et al.,](#page-17-0) [2026\]](#page-17-0) 从另一个角度提升可靠性：多智能体引擎结合可执行的实例级检查器，生成以工具调用为依据的对话；随后，闭环自我进化流程会持续更新提示，以提高可靠性。

#### 19.4 数据整理

合成数据生成后，需要检查其质量并进行筛选整理。通常，这包括过滤、评分和挑选样本，只保留有用的内容。

{10}------------------------------------------------

<span id="page-10-0"></span>评估合成数据是否有用，可以采用两种互补的方法：*外部评估*，衡量数据对下游任务的影响；或*内部评估*，直接检查生成数据或生成过程的属性。

实践中，数据筛选整理通常由多阶段过滤流程完成，将下文讨论的标准转化为具体操作。常见步骤包括：*语义去重*（依据嵌入相似度移除近重复样本，以保持多样性）、*困惑度过滤*（使用参考模型剔除不连贯或偏离分布的文本）、规则或格式检查，以及*奖励模型或评审打分*（保留高质量样本）。开创性论文 *Textbooks Are All You Need* [\[Gunasekar et al.,](#page-17-0) [2023\]](#page-17-0) 提醒我们，筛选整理的重要性往往不亚于数据规模：严格的质量过滤可能比单纯增加数据量更有效。缺少这一步时，模型很容易学到教师模型生成内容中的伪影和统计异常。

#### 19.4.1 外部评估

外部评估回答一个简单问题：合成数据是否真的有助于下游任务？这也称为*效用评估*。通常的做法是使用合成数据（单独使用或与真实数据混合）训练或微调模型，再用留出的真实世界测试集评估模型。它是判断合成数据是否有用的最直接方法。例如，加入合成数据后，如果预训练模型的表现有所改善，就可以认为这些数据有用；若以困惑度衡量，改善意味着困惑度下降。但这种方法成本也最高，因为必须完成一次完整训练后才能得到评估结果。若调整生成流程或改用另一批数据，就需要重新训练和评估。因此，外部评估不适合在生成过程中逐条筛选样本。

#### 19.4.2 内部评估

内部评估直接考察生成内容的特征，无需承担下游训练的成本。研究者已提出若干用于评估合成文本质量的维度 [\[Ramesh et al.,](#page-21-0) [2025\]](#page-21-0)。也可以不只评估输出，而是直接测试生成流程所用的语言模型 [\[Kim et al.,](#page-18-0) [2025b\]](#page-18-0)。与检查生成数据相比，这种方法能更早、以更低成本提供反馈。

**数据正确性。**生成样本只有在事实和逻辑上都可靠时才有用。合成数据通常由其他 LLM 生成，因此会继承生成模型的失效模式，例如虚构事实、推理不一致，或标签与内容不符。这些错误随后可能传播到使用该数据训练的模型中。常见的正确性检查方法包括将生成结果与人工标注进行比较 [\[Long et al.,](#page-19-0) [2024,](#page-19-0) [Chim et al.,](#page-15-0) [2025\]](#page-15-0)，以及通过规则检查器或独立评审模型进行验证 [\[Chen et al.,](#page-15-0) [2025a,](#page-15-0)

{11}------------------------------------------------

<span id="page-11-0"></span>[Yang et al.（2025）](#page-25-0) 和 [Setlur et al.（2024）](#page-22-0)，或者对同一输入多次生成的结果进行一致性检查 [\[Zelikman et al.,](#page-26-0) [2022b,](#page-26-0) [Seegmiller et al.,](#page-22-0) [2025\]](#page-22-0)。

**数据多样性。**合成数据的一项关键挑战是保持多样性。Logit 抑制和温度采样可以增加多样性，但可能降低正确性 [\[Chung et al.,](#page-15-0) [2023\]](#page-15-0)。多样性也很难衡量：词汇层面的启发式指标（如 ROUGE）和基于嵌入相似度的神经指标常被用作代理，但每种指标都只能捕捉数据变化的一部分。

MetaSynth 通过生成合成数据来解决这个问题，这些数据通过元提示增强了多样性，其中一个语言模型协调多个专家 LLM 代理协作生成数据 [\[Riaz et al.,](#page-22-0) [2025\]](#page-22-0)。

**数据污染。**合成数据面临的严重风险之一是污染 [\[Deng et al.,](#page-16-0) [2024,](#page-16-0) [Balloccu et al.,](#page-14-0) [2024\]](#page-14-0)。生成数据的 LLM 曾在规模庞大、过滤不充分的网络数据集上预训练，因此可能记住基准内容，并在生成新训练样本时复现或近似改写这些内容。经过多轮生成后，合成数据可能逐渐趋近已有基准，从而人为抬高评估分数，例如代码生成基准中的分数 [\[Matton et al.,](#page-20-0) [2024\]](#page-20-0)。防止此类泄漏不能只依赖精确匹配过滤，还应对已知基准进行 n-gram 重叠搜索、成员推断式探测 [\[Puerto et al.,](#page-21-0) [2025,](#page-21-0) [Huang et al.,](#page-17-0) [2026\]](#page-17-0)，或使用动态数据集构建流程 [\[Chen et al.,](#page-15-0) [2025b\]](#page-15-0)。

**公平与偏差。**LLM 会从训练数据中继承并放大社会偏见。如果再用这些模型生成合成语料，就可能进一步延续刻板印象、人口群体代表性失衡，或不同群体之间生成质量不平等的问题 [\[Gallegos et al.,](#page-16-0) [2024\]](#page-16-0)。合成数据尤其值得关注，因为它会大规模生成，并常被直接用作新 LLM 的训练材料。

最后，内部指标只能近似反映数据效用。数据集可能多样性得分很高、也通过正确性检查，却没有改善下游表现，甚至造成损害。词汇和神经多样性指标衡量的是表层变化，可能漏掉重要的冗余（如反复出现的推理模式），也可能高估多样性。正确性检查依赖可信参考、规则验证器或评审模型；面对开放式或主观生成内容时，这些工具可能不存在，也可能不可靠。因此，内部评估适合作为补充检查，不能真正取代外部评估。

#### 19.5 风险

使用合成数据训练智能体 LLM 并非没有代价。本节再讨论几项相关风险，包括：
{12}------------------------------------------------

<span id="page-12-0"></span>模型崩溃、偏差放大、安全与隐私漏洞，以及法律和伦理问题。

#### 19.5.1 模型崩溃

图 [19.2](#page-2-0) 所示闭环的核心风险，是模型会不断以自身输出为训练材料。*模型崩溃*是指连续几代模型递归地使用 AI 生成数据训练，导致原始分布的多样性逐渐消失 [\[Shumailov et al.,](#page-23-0) [2024\]](#page-23-0)。罕见事件和低概率的语言细节被采样到的机会更少，于是每一代都会丢失更多长尾信息，最终收敛到重复且高概率的模式。一些实验表明，只用合成数据反复训练会带来递减、乃至转为负面的收益 [\[Singh et al.,](#page-23-0) [2024\]](#page-23-0)；与多来源语料相比，依赖少数合成来源也会增加模型的脆弱性 [\[Schaffelder and Gatt,](#page-22-0) [2026\]](#page-22-0)。这些发现也与合成数据训练的小型学生模型有关：如果教师生成的语料对长尾事实覆盖不足，学生模型仍可能存在知识缺口。研究发现，事实问答准确率与模型预训练数据中相关文档的数量有关 [\[Kandpal et al.,](#page-18-0) [2023\]](#page-18-0)。另有研究发现，模仿模型可能学到更强教师的*表达风格*，但在模仿数据覆盖不足的任务上，事实性和能力表现仍有差距 [\[Gudibande et al.,](#page-17-0) [2023\]](#page-17-0)。常见的缓解办法是用真实的人类数据持续锚定语料，而不是只用合成数据训练。

#### 19.5.2 偏见放大

生成模型可能弱化少数观点。在论点摘要任务中，通用 LLM 和专用模型都难以呈现只被少数人表达的观点，也难以处理来源多样的数据；多样化训练数据可能改善零样本泛化 [\[van der Meer et al.,](#page-23-0) [2024\]](#page-23-0)。如果合成数据重复这些主流模式，在其上训练的学生模型可能进一步放大教师模型的社会偏见。相关偏见可能表现为刻板关联、人口群体代表性失衡，或群体间输出质量不平等 [\[Gallegos et al.,](#page-16-0) [2024\]](#page-16-0)。合成数据会被大规模生成并直接用于训练，闭环迭代还可能叠加这些影响。

#### 19.5.3 安全风险

合成训练数据还会带来安全风险。依赖生成数据会增加训练数据遭恶意污染的攻击面。如果教师模型或生成流程的任何环节遭到入侵，攻击者就可能发动数据投毒。合成语料往往庞大到无法逐条审计，因此攻击者可能植入隐蔽后门或恶意的指令—响应样本，直到学生模型部署后才触发。研究表明，即使只污染网络级语料中的极小部分，也可能做到且成本低廉 [\[Carlini et al.,](#page-15-0) [2024\]](#page-15-0)。如果“教师”是训练数据和对齐方式均不透明、且不受控制的第三方模型，风险还会进一步上升。

{13}------------------------------------------------

#### <span id="page-13-0"></span>19.5.4 隐私保护

合成数据常被视为保护数据隐私的一种方案。其基本设想是：合成数据不包含个人可识别信息（PII）[\[Dwork and Roth,](#page-16-0) [2014\]](#page-16-0)，同时仍能反映私有数据（如医疗记录）的统计特征。

然而，这类数据仍有被模型记忆并泄露的显著风险。如果生成阶段缺乏严格的隐私保证，生成模型可能无意中重建并泄露训练数据中的敏感原文片段 [\[Carlini et al.,](#page-14-0) [2021\]](#page-14-0)。隐私评估也显示，现成的隐私保护方法可能失效 [\[Ramesh et al.,](#page-21-0) [2024\]](#page-21-0)。

常见方法之一是差分隐私 [\[Dwork et al.,](#page-16-0) [2006\]](#page-16-0)。它为生成过程提供保证，限制数据集中增加或删除某一个体记录时输出分布可能发生的变化。该界限由隐私预算控制：预算越小，保护越强，但效用损失也越大 [\[Yue et al.,](#page-25-0) [2023\]](#page-25-0)。

#### 19.5.5 版权与使用条款

合成数据的吸引力之一，是它看似可以避开抓取互联网规模人类文本所涉及的法律不确定性；此类抓取已引发版权诉讼。2025 年的 *Bartz v. Anthropic* 和解案就是一例。此前，地区法院认为，LLM 训练本身具有很强的转换性，可能构成合理使用；但该案还涉及底层语料的获取方式：Anthropic 使用了数十万本盗版书。最终和解金额约为每部作品 3,000 美元，总额约 15 亿美元。其他案件（如 *Kadrey v. Meta*）也提出了类似问题。这说明，训练数据的版权和使用条款正受到越来越严格的审查。

合成数据并不能完全避开这一问题。从专有模型蒸馏还会引出另一项风险：许多商业 LLM API 的服务条款明确禁止使用其输出训练竞争模型。因此，即使不考虑版权，黑盒教师生成的数据也可能涉及许可问题。2025 年，围绕竞争系统是否使用前沿 API 输出训练的指控，使这一矛盾受到公开关注。对实践者而言，生成器的许可条款与被替代的人类数据许可同样需要核查。

#### 19.6 结论性意见

合成数据已从小众的数据增强方法，转变为 LLM 训练流程（第 3 章）的核心组成部分；一个推动因素是人类文本可能成为规模扩展的瓶颈（第 4 章）。强模型和多智能体编排能够生成指令数据、评估基准，以及近期出现的整段智能体工具调用与决策*轨迹*（第 9 章和第 11 章）。
  
由于原始生成结果往往带有噪声，数据筛选整理、外部效用评估，以及对正确性、多样性和污染的内部检查，都与生成过程本身同样重要。

{14}------------------------------------------------

<span id="page-14-0"></span>不同设置都可视为知识蒸馏的形式，并可按信号、数据来源、时机（离策略或在策略）和教师身份这四个维度来分类。
  
复用同一模型生成的合成数据进行自我改进，有望推动可验证推理的发展（第 6 章）。但在所有这些流程中，都必须仔细审查模型使用的数据：若缺少真实数据锚定和严格审计，模型崩溃、偏差放大、安全、隐私及许可风险就会随着每轮迭代不断累积。

#### 参考文献

- Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In *International Conference on Learning Representations*, 2024. [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html); [arXiv:2306.13649](https://arxiv.org/abs/2306.13649). [367](#page-4-0)

- Ahmad Alismail and Carsten Lanquillon. A Survey of LLM-Based Methods for Synthetic Data Generation and the Rise of Agentic Workflows. In Helmut Degen and Stavroula Ntoa, editors, *Artificial Intelligence in HCI. HCII 2025*, Lecture Notes in Computer Science, vol. 15821, pp. 119–135. Springer, Cham, 2025. URL <https://doi.org/10.1007/978-3-031-93418-6_9>. [364](#page-1-0)

- Stephen H. Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V. Nayak, Abheesht Sharma, Taewoon Kim, M. Saiful Bari, Thibault Fevry, Zaid Alyafeai, Manan Dey, Andrea Santilli, Zhiqing Sun, Srulik Ben-David, Canwen Xu, Gunjan Chhablani, Han Wang, Jason Alan Fries, Maged S. Al-shaibani, Shanya Sharma, Urmish Thakker, Khalid Almubarak, Xiangru Tang, Dragomir Radev, Mike Tian-Jian Jiang, and Alexander M. Rush. PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts. In Valerio Basile, Zornitsa Kozareva, and Sanja Stajner, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, pages 93–104, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-demo.9. URL <https://aclanthology.org/2022.acl-demo.9/>. [368](#page-5-0)

- Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondřej Dušek. Leak, Cheat, Repeat: Data Contamination and Evaluation Malpractices in Closed-Source LLMs. In Yvette Graham and Matthew Purver, editors, *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 67–93, St. Julian's, Malta, March 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.eacl-long.5. URL <https://aclanthology.org/2024.eacl-long.5/>. [374](#page-11-0)

- Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. Extracting Training Data from Large Language Models. In *30th USENIX Security Symposium (USENIX Security 21)*, pages 2633–2650. USENIX Association, August 2021. ISBN 978-1-939133-24-3. URL <https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting>. [376](#page-13-0)

{15}------------------------------------------------

<span id="page-15-0"></span>- Nicholas Carlini, Matthew Jagielski, Christopher A. Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramèr. Poisoning Web-Scale Training Datasets Is Practical. In *2024 IEEE Symposium on Security and Privacy (SP)*, pages 407–425, 2024. doi: [10.1109/SP54263.2024.00179](https://doi.org/10.1109/SP54263.2024.00179). [375](#page-12-0)

- Meng Chen, Philip Arthur, Qianyu Feng, Cong Duy Vu Hoang, Yu-Heng Hong, Mahdi Kazemi Moghaddam, Omid Nezami, Duc Thien Nguyen, Gioacchino Tangari, Duy Vu, Thanh Vu, Mark Johnson, Krishnaram Kenthapadi, Don Dharmasiri, Long Duong, and Yuan-Fang Li. Mastering the Craft of Data Synthesis for CodeLLMs. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 12484–12500, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.620. URL <https://aclanthology.org/2025.naacl-long.620/>. [373](#page-10-0)

- Simin Chen, Yiming Chen, Zexin Li, Yifan Jiang, Zhongwei Wan, Yixin He, Dezhi Ran, Tianle Gu, Haizhou Li, Tao Xie, and Baishakhi Ray. Benchmarking large language models under data contamination: A survey from static to dynamic evaluation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 10080–10098, Suzhou, China, November 2025b. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.511. URL <https://aclanthology.org/2025.emnlp-main.511/>. [374](#page-11-0)

- Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. In *Proceedings of the 41st International Conference on Machine Learning*, vol. 235 of *Proceedings of Machine Learning Research*, pp. 6621–6642. PMLR, 2024. [PMLR](https://proceedings.mlr.press/v235/chen24j.html); [arXiv:2401.01335](https://arxiv.org/abs/2401.01335). [367](#page-4-0)

- Jenny Chim, Julia Ive, and Maria Liakata. Evaluating Synthetic Data Generation from User Generated Text. *Computational Linguistics*, 51(1):191–233, March 2025. doi: 10.1162/coli_a_00540. URL <https://aclanthology.org/2025.cl-1.6/>. [373](#page-10-0)

- John Chung, Ece Kamar, and Saleema Amershi. Increasing Diversity While Maintaining Accuracy: Text Data Generation with Large Language Models and Human Interventions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 575–593, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.34. URL <https://aclanthology.org/2023.acl-long.34/>. [374](#page-11-0)

- Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, and Chenguang Wang. Agent Instructs Large Language Models to Be General Zero-Shot Reasoners. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria

  {16}------------------------------------------------

  <span id="page-16-0"></span>Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, vol. 235 of *Proceedings of Machine Learning Research*, pp. 9458–9549. PMLR, 21–27 July 2024. URL <https://proceedings.mlr.press/v235/crispino24a.html>. [372](#page-9-0)

- Chunyuan Deng, Yilun Zhao, Yuzhao Heng, Yitong Li, Jiannan Cao, Xiangru Tang, and Arman Cohan. Unveiling the Spectrum of Data Contamination in Language Model: A Survey from Detection to Remediation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 16078–16092, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.951. URL <https://aclanthology.org/2024.findings-acl.951/>. [374](#page-11-0)

- Weihua Du, Pranjal Aggarwal, Sean Welleck, and Yiming Yang. Agentic-R1: Distilled Dual-Strategy Reasoning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 12029–12043, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.604. URL <https://aclanthology.org/2025.emnlp-main.604/>. [365](#page-2-0)

- Cynthia Dwork and Aaron Roth. The Algorithmic Foundations of Differential Privacy. *Foundations and Trends® in Theoretical Computer Science*, 9(3–4):211–487, 2014. URL <https://doi.org/10.1561/0400000042>. [376](#page-13-0)

- Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating Noise to Sensitivity in Private Data Analysis. In Shai Halevi and Tal Rabin, editors, *Theory of Cryptography (TCC 2006)*, vol. 3876 of *Lecture Notes in Computer Science*, pp. 265–284. Springer, Berlin and Heidelberg, 2006. URL <https://doi.org/10.1007/11681878_14>. [376](#page-13-0)

- Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. Understanding Back-Translation at Scale. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 489–500, Brussels, Belgium, October–November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1045. URL <https://aclanthology.org/D18-1045/>. [370](#page-7-0)

- Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. Bias and Fairness in Large Language Models: A Survey. *Computational Linguistics*, 50(3):1097–1179, 2024. doi: 10.1162/coli_a_00524. URL <https://aclanthology.org/2024.cl-3.8/>. [374](#page-11-0), [375](#page-12-0)

- Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, and Jack Clark. Red Teaming
  {17}------------------------------------------------
  <span id="page-17-0"></span>Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned, 2022. URL <https://arxiv.org/abs/2209.07858>. [371](#page-8-0)

- Jiaxuan Gao, Jiaao Chen, Chuyi He, Wei-Chen Wang, Shusheng Xu, Hanrui Wang, Di Jin, and Yi Wu. From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents, 2026. URL <https://arxiv.org/abs/2601.22607>. [372](#page-9-0)

- Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling Synthetic Data Creation with 1,000,000,000 Personas. *SynthData workshop paper at ICLR 2025*, 2025. URL <https://openreview.net/forum?id=uBCAtA6M73>. arXiv preprint arXiv:2406.20094, 2024. [368](#page-5-0)

- Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The False Promise of Imitating Proprietary LLMs, 2023. URL <https://arxiv.org/abs/2305.15717>. [375](#page-12-0)

- Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks Are All You Need, 2023. URL <https://arxiv.org/abs/2306.11644>. [369](#page-6-0), [373](#page-10-0)

- Patrick Haluptzok, Matthew Bowers, and Adam Tauman Kalai. Language Models Can Teach Themselves to Program Better. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=SaRj2ka1XZ3>. [369](#page-6-0)

- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015. <https://arxiv.org/abs/1503.02531>. [365](#page-2-0)

- Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 14409–14428, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.806. URL <https://aclanthology.org/2023.acl-long.806/>. [368](#page-5-0), [369](#page-6-0)

- Chengming Hu, Xuan Li, Dan Liu, Xi Chen, Ju Wang, and Xue Liu. Teacher-Student Architecture for Knowledge Learning: A Survey, 2022. URL <https://arxiv.org/abs/2210.17332>. [365](#page-2-0)

- Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-Evolving Reasoning LLM from Zero Data. arXiv:2508.05004, 2025. <https://arxiv.org/abs/2508.05004>. [368](#page-5-0)

- Yuanhong Huang, Huili Wang, Xueying Bai, Jinrui Wang, Jiajun Liu, Ziqin Wang, Wanchun Ni, Shangguang Wang, and Tao Qi. Robust Membership Inference for Large Language Models under Adversarial Generative Corruption. In Maria Liakata,
  {18}------------------------------------------------
  <span id="page-18-0"></span>Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 39531–39547, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.1835. URL <https://aclanthology.org/2026.acl-long.1835/>. [374](#page-11-0)

- Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to learn long-tail knowledge. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 15696–15707. PMLR, 23–29 Jul 2023. URL <https://proceedings.mlr.press/v202/kandpal23a.html>. [375](#page-12-0)

- Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, and Sung Ju Hwang. Distilling LLM Agent into Small Models with Retrieval and Code Tools. In *Advances in Neural Information Processing Systems 38*, 2025. doi: 10.52202/085713-3553. URL <https://proceedings.neurips.cc/paper_files/paper/2025/hash/99263f9bf46874e2b8b7f104b5063864-Abstract-Conference.html>. [371](#page-8-0)

- Shivani Kapania, Stephanie Ballard, Alex Kessler, and Jennifer Wortman Vaughan. Examining the Expanding Role of Synthetic Data Throughout the AI Development Pipeline. In *Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency*, FAccT '25, pages 45–60, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 979-8-4007-1482-5. doi: 10.1145/3715275.3732005. URL <https://doi.org/10.1145/3715275.3732005>. [364](#page-1-0)

- Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, Zhiyi Ma, Tristan Thrush, Sebastian Riedel, Zeerak Waseem, Pontus Stenetorp, Robin Jia, Mohit Bansal, Christopher Potts, and Adina Williams. Dynabench: Rethinking Benchmarking in NLP. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4110–4124, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.324. URL <https://aclanthology.org/2021.naacl-main.324/>. [371](#page-8-0)

- Seongho Kim, Jihyun Moon, Juntaek Oh, Insu Choi, and Joon-Sung Yang. Survey and Evaluation of Converging Architecture in LLMs Based on Footsteps of Operations. *IEEE Open Journal of the Computer Society*, 6:1214–1226, 2025a. doi: [10.1109/OJCS.2025.3587005](https://doi.org/10.1109/OJCS.2025.3587005). [363](#page-0-0)

- Seungone Kim, Juyoung Suk, Xiang Yue, Vijay Viswanathan, Seongyun Lee, Yizhong Wang, Kiril Gashteovski, Carolin Lawrence, Sean Welleck, and Graham Neubig. Evaluating Language Models as Synthetic Data Generators. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings

  {19}------------------------------------------------

  <span id="page-19-0"></span>of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6385–6403, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.320. URL <https://aclanthology.org/2025.acl-long.320/>. [373](#page-10-0)

- Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tiannan Wang, Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang, and Wangchunshu Zhou. Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL, 2025a. URL <https://arxiv.org/abs/2508.13167>. [365](#page-2-0)

- Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, and Radha Poovendran. Small Models Struggle to Learn from Strong Reasoners. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, *Findings of the Association for Computational Linguistics: ACL 2025*, pages 25366–25394, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1301. URL <https://aclanthology.org/2025.findings-acl.1301/>. [366](#page-3-0), [367](#page-4-0)

- Weixin Liang, Girmaw Abebe Tadesse, Daniel Ho, L. Fei-Fei, Matei Zaharia, Ce Zhang, and James Zou. Advances, Challenges and Opportunities in Creating Data for Trustworthy AI. *Nature Machine Intelligence*, 4(8):669–677, 2022. doi: 10.1038/s42256-022-00516-1. URL <https://doi.org/10.1038/s42256-022-00516-1>. [364](#page-1-0)

- Philip Lippmann, Matthijs T. J. Spaan, and Jie Yang. Illuminating Blind Spots of Language Models with Targeted Agent-in-the-Loop Synthetic Data, 2024. URL <https://arxiv.org/abs/2403.17860>. [370](#page-7-0)

- Alisa Liu, Swabha Swayamdipta, Noah A. Smith, and Yejin Choi. WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 6826–6847, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.508. URL <https://aclanthology.org/2022.findings-emnlp.508/>. [370](#page-7-0)

- Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. TinyGSM: achieving >80% on GSM8k with small language models, 2023. URL <https://arxiv.org/abs/2312.09241>. [369](#page-6-0)

- Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. On LLMs-driven Synthetic Data Generation, Curation, and Evaluation: A Survey. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 11065–
  {20}------------------------------------------------
  <span id="page-20-0"></span>11082, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.658. URL <https://aclanthology.org/2024.findings-acl.658/>. [364](#page-1-0), [373](#page-10-0)

- Kevin Lu and Thinking Machines Lab. On-Policy Distillation. *Thinking Machines Lab: Connectionism*, 2025. doi: [10.64434/tml.20251026](https://doi.org/10.64434/tml.20251026). URL <https://thinkingmachines.ai/blog/on-policy-distillation/>. [367](#page-4-0)

- Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, and Hongsheng Li. MathGenie: Generating synthetic data with question back-translation for enhancing mathematical reasoning of LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2732–2747, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.151. URL <https://aclanthology.org/2024.acl-long.151/>. [370](#page-7-0)

- Alisia Maria Lupidi, Carlos Gemmell, Nicola Cancedda, Jane Yu, Jason E Weston, Jakob Nicolaus Foerster, Roberta Raileanu, and Maria Lomeli. Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources. In *Will Synthetic Data Finally Solve the Data Access Problem? Workshop at ICLR 2025*, 2025. URL <https://openreview.net/forum?id=z9rR3btRFI>. arXiv preprint arXiv:2409.08239 (2024). [369](#page-6-0)

- Alexandre Matton, Tom Sherborne, Dennis Aumiller, Elena Tommasone, Milad Alizadeh, Jingyi He, Raymond Ma, Maxime Voisin, Ellen Gilsenan-McMahon, and Matthias Gallé. On leakage of code generation evaluation datasets. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 13215–13223, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.772. URL <https://aclanthology.org/2024.findings-emnlp.772/>. [374](#page-11-0)

- Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-Task Generalization via Natural Language Crowdsourcing Instructions. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3470–3487, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.244. URL <https://aclanthology.org/2022.acl-long.244/>. [368](#page-5-0)

- Shrey Pandit, Xuan-Phi Nguyen, Yifei Ming, Austin Xu, Jiayu Wang, Caiming Xiong, and Shafiq Joty. Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms. *arXiv preprint arXiv:2510.13913*, 2025. URL <https://arxiv.org/abs/2510.13913>. [372](#page-9-0)

- Divyansh Pareek, Simon S. Du, and Sewoong Oh. Understanding the Gains from Repeated Self-Distillation. *arXiv preprint arXiv:2407.04600*, 2024. URL <https://arxiv.org/abs/2407.04600>. [368](#page-5-0)

<span id="page-21-0"></span>

{21}------------------------------------------------

- Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. Red Teaming Language Models with Language Models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 3419–3448, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.225. URL <https://aclanthology.org/2022.emnlp-main.225/>. [371](#page-8-0)

- Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Manoj Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Weiran Yao, Huan Wang, Silvio Savarese, and Caiming Xiong. APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, *Advances in Neural Information Processing Systems*, volume 38. Curran Associates, Inc., 2025. doi: 10.52202/085713-2184. URL <https://proceedings.neurips.cc/paper_files/paper/2025/hash/5e3661f7fe4c8ac5652d62eb3d3c96ea-Abstract-Datasets_and_Benchmarks_Track.html>. [371](#page-8-0), [372](#page-9-0)

- Haritz Puerto, Martin Gubri, Sangdoo Yun, and Seong Joon Oh. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, *Findings of the Association for Computational Linguistics: NAACL 2025*, pages 4165–4182, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl.234. URL <https://aclanthology.org/2025.findings-naacl.234/>. [374](#page-11-0)

- Zeyu Qin, Qingxiu Dong, Xingxing Zhang, Li Dong, Xiaolong Huang, Ziyi Yang, Mahmoud Khademi, Dongdong Zhang, Hany Hassan Awadalla, Yi R. Fung, Weizhu Chen, Minhao Cheng, and Furu Wei. Scaling Laws of Synthetic Data for Language Models. In *Proceedings of the Second Conference on Language Modeling (COLM 2025)*, 2025. URL <https://openreview.net/forum?id=UmUXPXHtdl>. [364](#page-1-0)

- Krithika Ramesh, Nupoor Gandhi, Pulkit Madaan, Lisa Bauer, Charith Peris, and Anjalie Field. Evaluating Differentially Private Synthetic Data Generation in High-Stakes Domains. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 15254–15269, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.894. URL <https://aclanthology.org/2024.findings-emnlp.894/>. [376](#page-13-0)

- Krithika Ramesh, Daniel Smolyak, Zihao Zhao, Nupoor Gandhi, Ritu Agarwal, Margrét V. Bjarnadóttir, and Anjalie Field. SynthTextEval: Synthetic Text Data Generation and Evaluation for High-Stakes Domains. In Ivan Habernal, Peter Schulam, and Jörg Tiedemann, editors, *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages

{22}------------------------------------------------

<span id="page-22-0"></span>487–499, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-334-0. doi: 10.18653/v1/2025.emnlp-demos.35. URL <https://aclanthology.org/2025.emnlp-demos.35/>. [373](#page-10-0)

- Haris Riaz, Sourav Sanjukta Bhabesh, Vinayak Arannil, Miguel Ballesteros, and Graham Horwood. MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, *Findings of the Association for Computational Linguistics: ACL 2025*, pages 18770–18803, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.962. URL <https://aclanthology.org/2025.findings-acl.962/>. [374](#page-11-0)

- Martín Santillán Cooper, Zahra Ashktorab, Hyo Jin Do, Erik Miehling, Werner Geyer, Jasmina Gajcin, Elizabeth M. Daly, Qian Pan, and Michael Desmond. Synthetic Data for Evaluation: Supporting LLM-as-a-Judge Workflows with EvalAssist. In Ivan Habernal, Peter Schulam, and Jörg Tiedemann, editors, *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 1–11, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-334-0. doi: 10.18653/v1/2025.emnlp-demos.1. URL <https://aclanthology.org/2025.emnlp-demos.1/>. [370](#page-7-0)

- Max Schaffelder and Albert Gatt. Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, *Findings of the Association for Computational Linguistics: ACL 2026*, pages 7265–7293, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.360. URL <https://aclanthology.org/2026.findings-acl.360/>. [375](#page-12-0)

- Parker Seegmiller, Kartik Mehta, Soumya Saha, Chenyang Tao, Shereen Oraby, Arpit Gupta, Tagyoung Chung, Mohit Bansal, and Nanyun Peng. FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, *Findings of the Association for Computational Linguistics: EMNLP 2025*, pages 24746–24766, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1346. URL <https://aclanthology.org/2025.findings-emnlp.1346/>. [374](#page-11-0)

- Amrith Setlur, Saurabh Garg, Xinyang (Young) Geng, Naman Garg, Virginia Smith, and Aviral Kumar. RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems*, volume 37, pages 43000–43031. Curran Associates, Inc., 2024. doi: 10.52202/079017-1361. URL <https://proceedings.neurips.cc/paper_files/paper/2024/hash/4b77d5b896c321a29277524a98a50215-Abstract-Conference.html>. [374](#page-11-0)

- Vedant Shah, Dingli Yu, Kaifeng Lyu, Simon Park, Jiatong Yu, Yinghui He, Nan Rosemary Ke, Michael Mozer, Yoshua Bengio, Sanjeev Arora, and Anirudh Goyal. AI-Assisted Generation of Difficult Math Questions. Accepted at *The 4th Workshop on Mathematical Reasoning and AI (MATH-AI)* at NeurIPS 2024. URL <https://arxiv.org/abs/2407.21009>. [371](#page-8-0)

{23}------------------------------------------------

- <span id="page-23-0"></span>Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Anderson. The Curse of Recursion: Training on Generated Data Makes Models Forget, 2024. URL <https://arxiv.org/abs/2305.17493>. [375](#page-12-0)

Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL <https://openreview.net/forum?id=lNAyUngGFK>. Expert Certification. [375](#page-12-0)

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models. *Transactions on Machine Learning Research*, 2023. URL <https://openreview.net/forum?id=uyTL5Bvosj>. arXiv:2206.04615. URL <https://arxiv.org/abs/2206.04615>. [368](#page-5-0)

- Lewis Tunstall, Edward Emanuel Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct Distillation of LM Alignment. In *First Conference on Language Modeling*, 2024. URL <https://openreview.net/forum?id=aKkAwZB6JV>. [365](#page-2-0)

- Michiel van der Meer, Piek Vossen, Catholijn M. Jonker, and Pradeep K. Murukannaiah. An Empirical Analysis of Diversity in Argument Summarization. In Yvette Graham and Matthew Purver, editors, *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2028–2045, St. Julian's, Malta, March 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.eacl-long.123. URL <https://aclanthology.org/2024.eacl-long.123/>. [375](#page-12-0)

- Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235, pages 49523–49544, 2024. URL <https://openreview.net/forum?id=ViZcgDQjyG>. [363](#page-0-0), [364](#page-1-0)

- Vijay Viswanathan, Xiang Yue, Alisa Liu, Yizhong Wang, and Graham Neubig. Synthetic Data in the Era of Large Language Models. In Yuki Arase, David Jurgens, and Fei Xia, editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 5: Tutorial Abstracts)*, pages 11–12, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-255-8.
  {24}------------------------------------------------
  <span id="page-24-0"></span>doi: 10.18653/v1/2025.acl-tutorials.7. URL <https://aclanthology.org/2025.acl-tutorials.7/>. [364](#page-1-0)

- Dong Wang, Yang Li, Ansong Ni, Ching-Feng Yeh, Youssef Emad, Xinjie Lei, Liam Robbins, Karthik Padthe, Hu Xu, Xian Li, Asli Celikyilmaz, Ramya Raghavendra, Lifei Huang, Carole-Jean Wu, and Shang-Wen Li. Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework, 2025a. URL <https://arxiv.org/abs/2511.21686>. [365](#page-2-0)

- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv preprint arXiv:2305.16291*, 2023a. [369](#page-6-0)

- Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, and Mengdi Wang. Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, *Advances in Neural Information Processing Systems*, volume 38, pages 143630–143664. Curran Associates, Inc., 2025b. doi: 10.52202/085713-4809. URL <https://proceedings.neurips.cc/paper_files/paper/2025/hash/d38653cdaa8e992549e1e9e1621610d7-Abstract-Conference.html>. [368](#page-5-0)

- Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 5085–5109, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.340. URL <https://aclanthology.org/2022.emnlp-main.340/>. [368](#page-5-0)

- Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-Instruct: Aligning Language Models with Self-Generated Instructions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 13484–13508, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.754. URL <https://aclanthology.org/2023.acl-long.754/>. [366](#page-3-0), [369](#page-6-0), [370](#page-7-0)

- Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W
  {25}------------------------------------------------
  <span id="page-25-0"></span>White, Doug Burger, and Chi Wang. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. In *ICLR 2024 Workshop on Large Language Model (LLM) Agents*, 2024. URL <https://openreview.net/forum?id=uAjxFFing2>. [372](#page-9-0)

- Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. Stronger Models are Not Always Stronger Teachers for Instruction Tuning. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 4392–4405, Albuquerque, New Mexico, April 2025a. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.224. URL <https://aclanthology.org/2025.naacl-long.224/>. [366](#page-3-0)

- Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, and Rameswar Panda. TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments, 2025b. URL <https://arxiv.org/abs/2510.01179>. [372](#page-9-0)

- Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Shengjie Ma, Aofan Liu, Hui Xiong, and Jian Guo. LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, *Findings of the Association for Computational Linguistics: ACL 2025*, pages 3236–3256, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.169. URL <https://aclanthology.org/2025.findings-acl.169/>. [374](#page-11-0)

- Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J Prenger, and Animashree Anandkumar. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, *Advances in Neural Information Processing Systems*, volume 36, pages 21573–21612. Curran Associates, Inc., 2023. URL <https://proceedings.neurips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Datasets_and_Benchmarks.html>. [369](#page-6-0)

- Ann Yuan, Daphne Ippolito, Vitaly Nikolaev, Chris Callison-Burch, Andy Coenen, and Sebastian Gehrmann. SynthBio: A Case Study in Faster Curation of Text Datasets. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. URL <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/9fc3d7152ba9336a670e36d0ed79bc43-Abstract-round2.html>. [371](#page-8-0)

- Xiang Yue, Huseyin Inan, Xuechen Li, Girish Kumar, Julia McAnallen, Hoda Shajari, Huan Sun, David Levitan, and Robert Sim. Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the
  {26}------------------------------------------------
  <span id="page-26-0"></span>Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1321–1342, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.74. URL <https://aclanthology.org/2023.acl-long.74/>. [376](#page-13-0)

- Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. MAmmoTH2: Scaling Instructions from the Web. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems*, volume 37, pages 90629–90660. Curran Associates, Inc., 2024. doi: 10.52202/079017-2877. URL <https://proceedings.neurips.cc/paper_files/paper/2024/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html>. [369](#page-6-0)

- Eric Zelikman, Jesse Mu, Noah D. Goodman, and Yuhuai Tony Wu. STaR: Self-Taught Reasoner Bootstrapping Reasoning with Reasoning. *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [Google Research](https://research.google/pubs/star-self-taught-reasoner-bootstrapping-reasoning-with-reasoning/). [367](#page-4-0)

- Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping Reasoning with Reasoning. In *Advances in Neural Information Processing Systems*, volume 35, pages 15476–15488. Curran Associates, Inc., 2022. doi: [10.52202/068431-1126](https://doi.org/10.52202/068431-1126). [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf). [374](#page-11-0)

- Hengyuan Zhang, Shiping Yang, Xiao Liang, Chenming Shang, Yuxuan Jiang, Chaofan Tao, Jing Xiong, Hayden Kwok-Hay So, Ruobing Xie, Angel X. Chang, and Ngai Wong. Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 14619–14637, San Diego, California, United States, July 2026a. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.666. URL <https://aclanthology.org/2026.acl-long.666/>. [366](#page-3-0)

- Zhaoyang Zhang, Shuli Jiang, Yantao Shen, Yuting Zhang, Dhananjay Ram, Shuo Yang, Zhuowen Tu, Wei Xia, and Stefano Soatto. Reinforcement-aware Knowledge Distillation for LLM Reasoning. In *Proceedings of the 43rd International Conference on Machine Learning*, volume 306 of *Proceedings of Machine Learning Research*, pages 156056–156069. PMLR, 2026b. URL <https://proceedings.mlr.press/v306/zhang26bw.html>. [367](#page-4-0)

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems*, volume 36, pages 46595–46623. Curran Associates, Inc., 2024. doi: 10.52202/075280-2020. URL <https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html>. [371](#page-8-0)

- Yifei Zhou, Sergey Levine, Jason E Weston, Xian Li, and Sainbayar Sukhbaatar. Self-Challenging Language Model Agents. In D. Belgrave, C. Zhang,
  {27}------------------------------------------------
  <span id="page-27-0"></span>H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, *Advances in Neural Information Processing Systems*, volume 38, pages 113959–113991. Curran Associates, Inc., 2025. doi: 10.52202/085713-3801. URL <https://proceedings.neurips.cc/paper_files/paper/2025/hash/a5a305fac88fb4ae40969cfec5eef48d-Abstract-Conference.html>. [368](#page-5-0)


