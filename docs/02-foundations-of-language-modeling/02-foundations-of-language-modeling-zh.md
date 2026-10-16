<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 29-62; Marker split PDF pages 1-34. -->



{0}------------------------------------------------

# 第二章

# 语言建模基础与 LLM

#### 你将学到的内容

在本章中，你将学习如何在机器学习模型中处理自然语言。具体来说，你将能够：

- 描述*语言建模*的核心概念，包括分词以及早期的词表示方法，并解释每种方法旨在解决什么问题。
- 区分语言模型训练中的几种学习范式——*监督学习、自回归学习和强化学习*——并说明它们各自对最终模型的作用。
- 解释 *嵌入* 如何将词义表示为向量，并预测语义相关的词在彼此之间的位置关系。
- 解释*注意力机制*如何在词元之间建立灵活且依赖上下文的联系，并追踪给定输入中的查询、键和值如何相互作用。
- 描述 *Transformer* 架构如何以注意力机制为基础构建，并指出其主要组成部分。
- 对比*编码器*（分类）和*解码器*（生成）模型，并为给定的NLP任务选择合适的模型。

# 2.1 自然语言处理

语言是人类交流的主要媒介，形式多种多样，包括印刷书籍、数字媒体和口语交流。语言研究长期以来一直是多学科研究的重点。在人工智能和计算机科学领域，语言尤为重要：我们借助自然语言表达复杂思想，也用编程语言编写算法。

{1}------------------------------------------------

图 2.1：与 ELIZA 的典型对话

<span id="page-1-0"></span>语言也是软件及其用户之间交互的重要媒介。

几十年来，让机器理解人类语言一直是人工智能的重要目标；该领域的主要教材也对此进行了论述 [\[Jurafsky and Martin,](#page-29-0) [2025\]](#page-29-0)。自然语言处理（NLP）及其相关领域计算语言学汇集了这方面的研究。<sup>1</sup>本章将概述通过计算方法处理*自然语言*（人类语言）时面临的复杂性，并回顾多年来为此发展的基础技术。

机器翻译是 NLP 早期的重要目标之一。自 20 世纪 50 年代机器翻译研究兴起以来 [\[Hutchins,](#page-29-0) [1999\]](#page-29-0)，语言处理技术取得了长足进步，并对各行各业产生了深远影响。技术的广泛应用又推动了面向实际需求的研发。在 NLP 领域，基础研究与现实应用相辅相成，应用范围也因此扩展到机器翻译之外。

近年来，语言技术对社会产生了深远影响。我们每天都会与 ChatGPT、Claude 等聊天机器人互动，使用多种语言交流，也会使用集成了增强搜索功能的产品。这些进展都建立在早期研究成果之上。一个著名的早期案例是 ELIZA：约瑟夫·韦岑鲍姆于 1966 年开发的心理治疗聊天机器人 [\[Weizenbaum,](#page-32-0) [1966\]](#page-32-0)。ELIZA 只根据简单规则替换词语，是一种初级聊天机器人（见图 2.1），但人们很早就赋予它类似人类的特质。早期文本处理系统依靠空格等分隔符识别词语；语言技术如今已远远超越这些简单方法。

<sup>1</sup>起初相距较远的两个领域，近年来越来越接近，一个从语言学方面起步，另一个从计算方面起步。

{2}------------------------------------------------

<span id="page-2-0"></span>
图 2.2：标准 NLP 流程，改绘自 [Manning 等人](#page-30-0) [\[2014\]](#page-30-0)

![](assets/_page_2_Diagram_2.jpeg)

#### 2.1.1 语言理解与语言生成

在讨论计算机如何处理语言之前，我们先看看人类如何理解语言。

交流随时间展开，语言也按序列进行处理。人们逐个说出词语，并依次接收它们。计算机处理语言时同样使用序列：书面文本中的字符、语音中的音节，或早期语言模型中的完整词语。

接下来，我们会进一步介绍为处理自然语言序列而发展出的技术。这些技术分为自然语言*理解*和自然语言*生成*两类，下面分别简要说明。

**自然语言理解** 自然语言理解（NLU）旨在开发能够捕捉意义和意图的方法。NLU 的方法会把非结构化的人类语言映射为结构化形式，使计算机能够解释内容并据此采取行动。这些方法通常包含多个处理层次，其设计借鉴了语言学分析（见图 2.2），例如：

- **分词**：确定有意义的语言单元（如词语）的边界，以便进一步处理。
- **句法分析**：识别语法结构，例如主语、动词和宾语。
- **命名实体识别（NER）**：识别并分类姓名、日期、地点等重要实体。
- **情感分析**：判断文本所表达的情绪倾向。
- **共指消解**：判断代词或其他指代表达指向哪个实体。例如，在“这台电脑是新的，但它已经很慢了”中，判断“它”是否指“这台电脑”。

{3}------------------------------------------------

<span id="page-3-0"></span>- **话语分析**：理解一句话的含义如何受到前文影响。

自然语言理解面临的主要挑战之一是歧义 [\[Roth,](#page-31-0) [1998,](#page-31-0) [Liu](#page-29-0) [等，](#page-29-0) [2023\]](#page-29-0)。人类语言中充满歧义，例如一词多义（如 *bank*），以及需要借助世界知识才能理解的语境细微差别。自然语言还有许多其他挑战；它像人类一样丰富而复杂。

**自然语言生成** 自然语言生成（NLG）旨在开发能够产出可读、自然语言表达的方法。相关任务既包括自由形式的文本生成，也包括将结构化数据或模型表示转换为自然语言表达。

语言建模（LM）是 NLG 的一项重要任务：根据已有上下文预测序列中的下一个元素。从数学上说，这需要为词语序列分配概率分布。传统方法使用简单的计数模型估计这一分布，通常进一步假设只根据固定窗口内的前几个词来预测下一个词。

现代方法利用神经网络，根据较长的上下文生成流畅的句子。这些句子在人看来也可能有意义。<sup>2</sup>有效的语言生成不只要求语法正确，还要求内容连贯（逻辑流畅）且衔接自然（语言表达彼此关联）。当前的 NLG 系统会使用采样、集成等方法，在创造性和可预测性之间取得平衡，使生成文本读起来自然而非机械 [\[Li and Jurafsky,](#page-29-0) [2017,](#page-29-0) [Dong 等，](#page-28-0) [2022,](#page-28-0) [Gatt and Krahmer,](#page-28-0) [2018\]](#page-28-0)。NLG 方法通常分为两类：

- **抽取式生成**：选取并重新排列已有文本片段，早期摘要工具常采用这种方法。
- **抽象式生成**：从头生成全新的句子，现代模型（如 ChatGPT）通常采用这种方法。

大多数 LLM 使用抽象式生成。

接下来，我们将介绍早期的基础语言建模方法：先讨论自然语言的复杂性，再介绍分布式表示原理、分词方法和早期的语言表示方法。

### 2.1.2 自然语言的复杂性

自然语言难以由机器理解，原因有很多。语言中的细微差别可能承载或改变含义，因此理解自然语言不能只分析其结构。

<sup>2</sup>正如 ELIZA 的例子所示，让句子在人看来似乎有意义，有时并不困难。

{4}------------------------------------------------

<span id="page-4-0"></span>话语发生的语境也会影响交流。理解语言不能停留在字面层面，还要把握其中蕴含的意图、暗示和细微差别。[\[Wittgenstein,](#page-32-0) [1953,](#page-32-0) [Grice,](#page-28-0) [1975,](#page-28-0) [Krause and Vossen,](#page-29-0) [2024\]](#page-29-0)。下面，我们介绍语言学中的若干研究主题，每个主题都从不同角度探讨语言能够表达什么。

**句法** 句法研究词语如何组合成结构完整的句子。在语言学中，它指特定语言中决定句子结构的一套规则、原则和过程。英语常见的语序是主语—动词—宾语。例如，在 *The cat bites the dog*（猫咬狗）中，动词 *bites* 位于主语 *cat* 之后，因此实施咬的动作的是猫。

**形态学** 形态学研究词语如何根据语法功能改变形式。词形变化与句法有关，而形态规则及其例外因语言和语系而异。在前面的例句中，动词 *bite* 变为 *bites*；词尾 *-s* 表示一般现在时的第三人称单数。

**语义学** 在语言学中，词汇语义学研究词语的意义。在自然语言里，词义来自词语之间的关系。例如，意义相近的词构成*同义词*，如 *car* 和 *automobile*、*vanish* 和 *disappear*；意义相反的词构成*反义词*，如 *hot* 和 *cold*、*buy* 和 *sell*。词语之间还存在许多其他关系。

在自然语言处理（NLP）中，计算语义学研究如何在计算机中高效表示意义。一种简单的句子表示方法，是用一串索引指向语言词表中的词语向量。然而，这种编码只记录词语是否出现，并未记录词语之间的关系；因此，这种朴素表示丢失了词语的语义信息。

词语的意义源于它们之间的关系，因此我们需要有效编码这些关系的方法。正如本章后文将介绍的，多年来人们提出了多种解决方案，包括 Transformer 架构和注意力机制。

**语用学** 除了词语的字面意义，人们也会考虑说话者的意图和话语语境。语言在特定社会语境中的使用方式会影响其表达效果（*意义即使用*）[\[Baker and Hacker,](#page-27-0) [2008\]](#page-27-0)。研究人们为有效交流而自然遵循的隐含规则，这一领域称为*语用学*。

一组基础原则是格赖斯会话准则[\[Grice,](#page-28-0) [1975\]](#page-28-0)，即人们用来推断含义的隐含合作原则。例如，数量准则要求说话者提供恰好满足对话需要的信息。

{5}------------------------------------------------

这些准则塑造了我们的交流，有助于让表达易于理解并体现合作意图。对机器来说，理解讽刺、潜台词或占位词（例如 “thingy”）需要掌握这些规则，而不能只依赖静态词典。

**话语** 语言很少以孤立句子的形式出现，而是由段落、对话或完整文档等较大单元构成。话语研究关注句子如何连接成连贯的整体。机器若要理解文本，就必须能够跨句追踪实体；这项任务称为共指消解。例如，机器需要判断代词 *it* 指的是前文提到的猫还是狗。话语理解对于文档摘要、机器翻译和长篇对话等复杂 NLP 任务至关重要。

除了这些主题之外，还存在其他研究领域，例如语音学（语言的声音在人与人交流中携带额外意义的研究）、社会语言学（研究语言如何根据社会因素如阶级或地区而变化）、词典学（编纂词典的科学）等。

正如上述概述所示，自然语言生成与理解都是庞大而具有挑战性的领域。语言并非一成不变：它会随时间、地域、文化和传播媒介而变化。这种变化使语言技术的开发始终充满挑战。原文在此处以“:-)”表示轻松语气。

在我们介绍最近提供至少部分解决这些挑战的技术之前，我们先讨论启发现代自然语言处理转向的基本假设。

由于自然语言丰富而复杂，很难在计算机中表示。本节接下来将介绍分布式表示原理：词语的语义由其周围的上下文塑造。随后，我们会介绍分词方法，即如何将句子拆分为词元；最后讨论如何利用词元向量构建能够编码意义的表示。

#### 2.1.3 分布式表示

我们如何才能教会机器理解复杂的语义关系和语法结构，而无需手动编码所有内容？答案在于分布假说，这是现代自然语言处理中的一个基础概念。其核心思想在20世纪50年代末被提出：

*“你可以通过一个词所处的语境来了解它。”（Firth，1957）*

这表明，可以根据一个词通常出现时周围的词来推断它的意义。出现在相似语境中的词很可能具有相似的意义。例如，如果单词 *coffee* 和 *tea* 经常出现在类似 *drink*、*sip* 或 *brew* 这样的动词附近，那么机器可能会推断出它们之间存在语义关系，而无需明确的定义或其他人提供的信息。

{6}------------------------------------------------

<span id="page-6-0"></span>![](assets/_page_6_Figure_2.jpeg)

图 2.3：使用 word2vec 嵌入进行向量运算的示意图[\[Singerep,](#page-32-0) [2022\]](#page-32-0)。

为了将分布假说用于计算，我们借助线性代数，采用*向量语义学*框架。每个词都表示为高维连续空间中的一个点。含义相近的词在空间中彼此靠近，可用余弦相似度等几何度量衡量其语义相似性（见图 2.3 和第[2.3.1 节](#page-14-0)）。向量运算也可以表达词语之间的语义关系。早期方法通过稀疏共现矩阵构建语义向量，统计词语在彼此上下文中共同出现的次数，类似第[2.1.5 节](#page-8-0)介绍的方法。现代方法则更常使用稠密、低维的向量，即*词嵌入*。这类表示通常由神经网络学习：模型可根据上下文预测词语，也可根据词语预测上下文。*word2vec* 架构推动了这类方法的普及[\[Mikolov et al.,](#page-30-0) [2013\]](#page-30-0)。

基于向量的表示有助于捕捉词义，但如果为每个词都单独分配一个向量，词表规模可能会大到难以实际使用。

词级方法处理罕见词和未见词时也会遇到困难，因为词表中没有这些词对应的向量。因此，模型在为语言单位分配向量之前，需要先确定一组规模可控的基本单位。现代方法会把文本拆分成更小的单元，以缓解这些问题并处理词表外（OOV）输入。这个过程称为分词，下面将具体讨论。

{7}------------------------------------------------

#### 2.1.4 分词

计算机要处理语言，首先需要将文本切分为有意义的单元，也就是词元。无论原始文本来自网络还是语音转写，任务都是把连续的字符序列拆成可供模型处理的片段。这个过程称为*分词*。稳健的分词策略不能只按空格切分，还要适应不同语言和方言、处理复杂标点及表情符号等新形式，并将所得词元映射到固定大小的词表中。

文本切分是许多 NLP 流程中的关键步骤。切分看似简单，但标点用法可能造成歧义，例如缩写既可能写作 N.L.P.，也可能写作 NLP。简单的规则分词器常常难以应对自然语言的不规则性。示例 1 展示了几种构造词元的方法。

示例 1（分词策略）。考虑下面这个句子：“Leiden is beautiful in summer.”

#### 基于空格和标点的词切分

这种方法通常按照空格和标点切分文本：`["Leiden", "is", "beautiful", "in", "summer", "."]`

#### 字符分词

这种方法将文本分解为单个字符。

`["L", "e", "i", "d", "e", "n", " ", "i", "s", " ", "b", "e", "a", "u", "t", "i", "f", "u", "l", " ", "i", "n", " ", "s", "u", "m", "m", "e", "r", "."]`

#### 子词分词（例如，WordPiece）

这种方法将常见单词保留为完整的词元，但将复杂或罕见的单词分解为基础子词。

`["Lei", "##den", "is", "beautiful", "in", "summer", "."]`

**词级词元化** 将每个以空格分隔的字符串视为一个独立词元。这样可以保留句法结构，但词表会变得很大，并带来词表外（OOV）问题：模型无法处理训练时未见过的词。

词表过大也会使*独热编码*效率低下。若词表 $V$ 中的每个词 $w$ 都表示为向量 **x** $\in \{0,1\}^{|V|}$，且只有与 $w$ 对应的维度为 1，那么向量维度 $|V|$ 会随词表规模线性增长。

{8}------------------------------------------------

<span id="page-8-0"></span>词汇表。由于这些独热向量随后将构成我们神经网络的输入，对于不包含某些词汇表中所有单词的文本，我们最终会创建稀疏的、计算效率低下的表示，并且无法捕捉单词之间的语义关系。

**字符级分词** 似乎能直接解决词表不断扩大的问题，多年来也一直有人研究[\[Mielke et al.,](#page-30-0) [2021\]](#page-30-0)。这种方法将文本拆成单个字符，使词表大小受字母表规模限制，几乎消除了 OOV 问题。但它会显著拉长序列，增加计算负担。此外，单个字符本身缺少语义，模型必须先学会词的形态，再掌握更广泛的语言模式，因此训练时间更长。

**子词分词** 目前的 LLM 通常采用折中方案：常见词保留为完整词元，罕见词则拆分为更小且有意义的组成部分。常见实现包括：

- 字节对编码（BPE）：一种迭代算法，从单个字符开始，将最频繁的相邻字符对合并为单个词元，直到达到目标词汇量为止[\[Sennrich et al.,](#page-31-0) [2016\]](#page-31-0)（例如，*highest* → [*high*, *est*]）。BPE 最初是一种数据压缩算法[\[Gage,](#page-28-0) [1994\]](#page-28-0)。
- WordPiece：一种迭代算法，类似于 BPE，根据似然标准而非原始频率合并词元对，旨在最大化训练数据的概率[\[Wu et al.,](#page-32-0) [2016\]](#page-32-0)。

#### 2.1.5 语言表示的早期方法

完成分词后，下一个问题是如何将这些语言单位（词元）组合成连贯且有意义的句子。人类语言中的词元排列受到严格的句法和语义规则约束。如果计算方法能够准确生成合理的文本序列，我们或许可以说，这些方法掌握了相应的句法和语义规律，至少能够复现这些规律。

如何构建能够实现这一目标的数学表示？其中一项核心任务是*语言建模*。从数学上说，语言模型用于计算一串词在特定语境中共同出现的联合概率。这个概率有助于区分逻辑通顺、流畅的句子与随机的词元序列。例如，在英语中，*the cat sat* 的概率显著高于 *sat cat the*，因为前者符合既有的句法规则。

由于书面语言具有序列结构，估计句子的联合概率本质上是一个序列估计问题。

{9}------------------------------------------------

<span id="page-9-0"></span>语言建模的总体目标，是根据构成句子的词序列 $w_1,\ldots,w_k$，量化句子 $S$ 在特定语言中出现的统计可能性：

$P(S)=P(w_1,\ldots,w_k)$ (2.1)

若词表大小为 $|V|$、句长为 $k$，可能的词序列共有 $|V|^k$ 种。因此，随着句长增加，直接计算精确的联合概率会很快变得不可行。概率链式法则可以将联合概率分解；在适当的条件独立假设下，这种分解使实际应用中的计算变得可行[Jurafsky 和 Martin，2025](#page-29-0)：

$P(w_1,\ldots,w_k)=P(w_1)\prod_{i=2}^{k}P(w_i\mid w_1,\ldots,w_{i-1})$ (2.2)

概率语言模型通过最大化联合概率，学习偏好语法正确、逻辑通顺的词语序列。模型学到的概率分布实际上反映了人类语言中未明说的规律[Manning 和 Schütze，1999](#page-29-0)。早期方法通过最大似然估计（MLE），根据切分后的文本语料估计概率。最简单的做法是根据词语在语料中的相对频率计算其边缘概率：

$P(w)=\frac{\operatorname{count}(w\text{ in corpus})}{\text{total number of words}}$ (2.3)

为了让条件概率的计算更可行，研究者使用 *n-gram 模型*。$n$-gram 是文本样本中连续出现的 $n$ 个项。与根据整个前文计算一个词的概率（这很快会变得难以处理）不同，n-gram 模型采用马尔可夫假设：一个词的概率只取决于它前面的 $n-1$ 个词，因而截断了上下文。例如，二元模型（$n=2$）只根据紧邻的前一个词近似历史信息。它根据两个词共同出现的频率，再除以第一个词的频率，来估计条件概率：

$P(w_i\mid w_1,\ldots,w_{i-1})\approx P(w_i\mid w_{i-1})=\frac{\operatorname{count}(w_{i-1},w_i)}{\operatorname{count}(w_{i-1})}$ (2.4)

类似地，三元模型（$n=3$）根据前两个词预测当前词。这种简化降低了计算成本，使得直接从语料估计概率成为可能。

选择 $n$ 需要权衡。较大的 $n$ 能捕捉更丰富的上下文和更长距离的句法依赖，生成更自然的语言；但它也会加剧数据稀疏问题，因为具体词序列出现得越来越少，模型因而更常遇到概率为零的情况。较小的 $n$ 能提供更稳健的频次统计，但模型对更广泛语义和长距离依赖的理解也会受限。

{10}------------------------------------------------

在标准语言模型中，未见过的 $n$-gram 概率为零；由于句子联合概率是各项概率的乘积，这会使整个句子的概率也变为零。为避免这一问题，通常对计数进行平滑处理。常见方法是*拉普拉斯平滑*（又称加一平滑）：给词表中的每个词项都加上一个虚拟计数 1，并在分母中加上词表大小 $|V|$，从而保持概率分布有效：

$P_{\mathrm{Laplace}}(w) = \frac{\operatorname{count}(w\text{ in corpus})+1}{\text{total number of words}+|V|}$ (2.5)

这些概率和计数方法最终会产生语言的数值表示。良好的表示能够编码我们直观理解的词语属性，包括词义、语法功能以及词语之间的关系。早期 NLP 将相对频率和共现计数组织成向量，以词语在文本中的分布来表示词语。

基于计数的表示虽然易于处理，但会产生高维、稀疏的向量，导致计算效率低下，也难以泛化到未见数据。因此，现代 NLP 转而使用神经网络学习稠密、低维的向量表示，即*嵌入*。模型不再依赖显式计数，而是通过根据上下文预测词语来学习连续表示，从而更高效地捕捉深层语义关系。我们将在第[2.3 节](#page-13-0)介绍这些现代方法。先简要回顾三种重要的机器学习方法。

# 2.2 自然语言处理的机器学习方法

介绍了语言表示的基础后，接下来讨论用于训练现代人工智能大模型的学习算法。LLM 借助更广泛的机器学习领域中的方法进行训练。要理解 LLM，需要熟悉三种核心范式及其在语言建模中的应用：*监督学习*、*自回归学习*和*强化学习*。本节简要介绍这些方法，并提供相关参考资料以供深入阅读。

### 2.2.1 监督学习

监督学习是机器学习的基础。它依赖预先提供的标签来*监督*训练过程。训练数据集由输入示例及其标签配对组成，记为 $D=\{(x_i,y_i)\}$；我们用它来训练预测模型 $f_\theta$ 的参数 $\theta$。模型对每个输入 $x$ 计算估计值 $\hat{y}$，即 $f_\theta(x)\rightarrow\hat{y}$。使用模型计算估计值称为*推理*，也称*前向传播*。

{11}------------------------------------------------

<span id="page-11-0"></span>
前向传播后，我们用损失函数 $L$ 衡量估计值 $\hat{y}$ 与真实标签 $y$ 之间的差异。常见损失函数包括用于回归的平方误差和 $L=\sum_i(\hat{y}_i-y_i)^2$，以及用于分类的交叉熵损失 $L=-\sum_c y_c\log(\hat{y}_c)$。

监督训练算法通过调整模型参数，尽量减小训练样本的总损失。在*反向传播*阶段，模型将损失的梯度向后传递，并据此更新参数，使下一次前向传播的损失降低。在训练样本上不断减小损失的过程称为*训练*。模型对全部训练数据完成一轮前向和反向计算，称为一个*训练轮次*（epoch）；为提高效率，每轮可拆分为多个*批次*（batch）。

训练结束后，我们用训练集之外的新样本测试模型 $f_\theta$，评估它对未见样本的泛化能力。

如果模型在训练集上表现良好，说明它已较好地拟合这些样本；如果它在测试集上也表现良好，则说明泛化能力较强。训练集表现好、测试集表现差，通常表示模型*过拟合*：模型可能训练过久或参数过多，以至于连训练数据中的噪声也一并拟合。若两个数据集上的表现都较差，则属于*欠拟合*，可能是训练不足，或模型过于简单，无法表示数据的复杂性。

通常，变量较少的低维问题（如简单回归）用较小的模型和较少的训练数据就能取得良好效果；变量较多的高维问题（如图像分类）则需要参数量更大的模型（通常是深度神经网络）和更多训练数据。图像分类领域的突破性表现[\[Krizhevsky et al.,](#page-29-0) [2012\]](#page-29-0)是在构建大型 ImageNet 数据集后才实现的。该数据集包含数百万张人工标注图像[\[Deng et al.,](#page-28-0) [2009\]](#page-28-0)。

监督学习也是 LLM 训练的核心技术之一，例如下一章将介绍的监督微调。关于监督学习的基础知识，可参考相关教材[\[Bishop and Bishop,](#page-27-0) [2023,](#page-27-0) [Goodfellow et al.,](#page-28-0) [2016\]](#page-28-0)。

#### 2.2.2 自回归学习

许多生成式 LLM，尤其是仅解码器模型，采用自回归训练。与监督学习不同，这种方法不需要人工标注的输入—输出配对数据集，而是利用大量文本。模型根据前文预测下一个词元，并以原文本中实际出现的下一个词元作为学习目标。因此，自回归训练属于自监督学习。

在自回归预训练中，模型根据位置 $t$ 之前的词元 $x_1,\ldots,x_{t-1}$ 作为上下文，预测当前位置的目标词元 $x_t$。这种下一个词元预测目标是许多现代 NLP 训练流程的核心[\[Radford et al.,](#page-31-0) [2018\]](#page-31-0)。

{12}------------------------------------------------

<span id="page-12-0"></span>![](assets/_page_12_Diagram_1.jpeg)

图 2.4：强化学习示意图，改绘自 [Sutton 和 Barto](#page-32-0) [\[2018\]](#page-32-0)。

自然语言处理教材也介绍了自回归学习[Jurafsky 和 Martin，2025](#page-29-0)。有关这些方法的更多细节，请参见第[2.3 节](#page-13-0)和第三章。

#### 2.2.3 强化学习

监督学习和自回归学习通常使用预先收集的数据，因此本节将其作为*离线*方法介绍。在线强化学习（RL）则可以在没有固定标注数据集、但能够评估状态质量时进行学习。智能体执行动作 $a$，使环境从状态 $s$ 转移到 $s'$，随后获得奖励 $r(s')$。由于训练数据由智能体在交互过程中生成，RL 称为在线方法。

智能体利用奖励信号调整参数化策略 $\pi_\theta(s)\rightarrow a$，该策略为每个状态选择动作。在序列决策问题中，奖励可能取决于此前的一系列动作，这就形成了*信用分配问题*：应把奖励归因于哪些动作？智能体通过反复观察状态、采取动作并接收奖励来探索状态空间，从而学习在各状态下选择动作的最优策略 $\pi^*(s)$。

在线强化学习会形成反馈循环。训练初期，智能体经验有限，策略只得到少数状态下的反馈。如果它只利用已有信息，始终选择当前看来最好的动作，就可能只访问环境的一小部分，并停留在局部最优策略。为了寻找更好的策略，智能体还需要探索，例如尝试随机动作，并尽可能覆盖整个状态空间。

强化学习通过环境反馈训练智能体。通常，用奖励函数判断输出好坏，比重新创建完整的标注数据集更容易。因此，许多强化学习算法都关注如何有效探索。强化学习问题通常建模为马尔可夫决策过程（MDP），包括状态表示 $S$、动作函数 $A(s)$、状态转移函数 $T(s,a)\rightarrow s'$ 和奖励函数 $R(s)$。环境提供数值奖励以表示状态质量，奖励可以为正也可以为负。智能体的目标是找到策略 $\pi$，为每个状态 $s$ 选择最优动作 $a^*$。图 2.4 展示了智能体如何通过与环境在线交互来学习策略；环境可以计算

{13}------------------------------------------------

<span id="page-13-0"></span>
函数 $T$ 和 $R$（环境会告知智能体：其动作导致了什么新状态，以及该状态的质量如何）。

如果状态空间足够小，能够装入内存，就可以使用 Q-learning 等精确的表格型方法[\[Watkins and Dayan,](#page-32-0) [1992\]](#page-32-0)。对于更大的状态空间，可用深度神经网络等函数逼近方法；代表算法包括深度 Q-learning（DQN）[\[Mnih et al.,](#page-30-0) [2015\]](#page-30-0)和近端策略优化（PPO）[\[Schulman et al.,](#page-31-0) [2017\]](#page-31-0)。在深度强化学习中，策略由参数化神经网络表示，并像监督式深度学习那样通过反向传播训练。

强化学习是现代推理型语言模型的重要基础之一。相关教材包括 Sutton 和 Barto 的开创性著作[\[2018\]](#page-32-0)，以及其他入门资料[\[Graesser and Keng,](#page-28-0) [2019,](#page-28-0) [Plaat,](#page-31-0) [2020,](#page-31-0) [2022\]](#page-31-0)。在 LLM 训练中，强化学习用于强化微调和多步推理（第六章将详细讨论）。强化学习还以智能体与环境的交互为核心，这与智能体式 LLM 的设定相契合。智能体式 LLM 可将强化学习用于上下文策略学习、提示词优化、世界模型，以及模拟多个智能体与环境之间的交互；本书后文将进一步讨论这些应用。

如果缺少预先收集的数据集，但智能体能够通过与环境交互来生成经验，强化学习就可能适用。考虑采用强化学习时，应先回答两个问题：（1）当前问题是否属于序列决策问题？如果是，（2）状态空间有多大？能否使用精确的表格方法，还是必须采用函数逼近？

掌握这些语言与机器学习原理后，我们回到语言建模。

# 2.3 语言建模

掌握语言表示与机器学习基础后，我们来看 NLP 的三种主要范式：符号方法、统计方法和神经方法[\[Kautz,](#page-29-0) [2020,](#page-29-0) [Deng and Liu,](#page-28-0) [2018\]](#page-28-0)。它们代表了不同的语言表示思路，也体现了处理人类语言复杂性的不同方式。随着计算能力提升，研究者能够处理更多数据、训练更大的模型，并将更多机器学习方法用于语言处理。

**符号方法。** 早期的规则系统（包括类似 ELIZA 的方法）依赖人工明确编写的指令，并把语言视为形式系统，试图通过规则推导含义。它们设计透明、易于解释，但

{14}------------------------------------------------

<span id="page-14-0"></span>也较为脆弱：严格规则（如 *if-then-else*）难以处理现实语言中的自然变化、习惯表达和歧义。若要将这类系统扩展到更复杂的语言任务，人工必须逐一枚举大量边界情况；除极小的问题领域外，这种做法都不切实际。

**统计方法。** 20 世纪 80 年代，自然语言处理逐渐转向经验式的概率建模。隐马尔可夫模型和 $n$-gram 语言模型等方法不再依赖显式句法规则，而是利用大型文本语料估计词序列的概率。这种数据驱动的方法更易扩展，也比符号系统更能处理语言歧义和噪声。不过，依赖马尔可夫假设的模型受限于局部词语模式，难以捕捉长距离依赖和更广泛的上下文。

**神经方法。** 2010 年前后，神经网络和深度学习逐渐成为主流。神经方法不再用显式规则或离散计数表示语言，而是学习连续空间中的稠密低维向量。尤其是 Transformer 等现代架构，借助注意力机制建模整个序列的上下文。这类方法具有较强的泛化能力，能够流畅地生成文本，也适用于跨语言任务；但神经模型往往难以解释，并且需要大量数据和计算资源。

当代 NLP 流程仍从这三种范式中汲取思路。在介绍下一章的训练流程示例前，我们先了解构建智能体式 LLM 所需的核心组件：词嵌入、上下文化嵌入、序列建模、注意力、Transformer、编码器（如 BERT）和解码器（如 GPT）。

### 2.3.1 嵌入

一种常见的词义表示方法，是统计目标词附近各词出现的次数。可以在目标词左右各取 $n$ 个词作为上下文窗口。例如，在句子 *There are two cookies left on the table* 中，若 $n=2$，*cookies* 的上下文词是 *are*、*two*、*left* 和 *on*。在大型语料上滑动窗口并统计词语共现次数，就能构建共现矩阵：每行代表一个目标词，每列代表一个上下文词，每个单元格记录该上下文词出现在目标词附近的次数。例如，*cherry* 向量在 *pie* 列中的数值可能较高（如 442），而在

{15}------------------------------------------------

<span id="page-15-0"></span>*computer* 列中的数值较低（如 2）。*strawberry* 的向量也会呈现类似模式，反映出这两个词在语义上的相似性。

将词语表示为向量后，它们就位于一个高维向量空间中，每个维度对应一个上下文词。含义相近且上下文相似的词会指向相近的方向，因此可以用数值方式衡量语义相似度。常用方法是*余弦相似度*：计算向量 **a** 和 **b** 夹角 $\theta$ 的余弦值，如公式 2.6 所示。

余弦相似度 $=\cos(\theta)=\frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}=\frac{\sum_{i=1}^{n}a_i b_i}{\sqrt{\sum_{i=1}^{n}a_i^2}\sqrt{\sum_{i=1}^{n}b_i^2}}$ (2.6)

前文提到，基于计数的向量会形成高维空间，其维度随词表规模线性增长。由于维度数量决定下游模型的计算复杂度，我们希望控制向量维度，转而创建稠密嵌入。经典的 word2vec 算法可以实现这一点[\[Mikolov et al.,](#page-30-0) [2013\]](#page-30-0)。该方法会随机初始化嵌入矩阵 $W$，再通过迭代更新，使其符合“相似词出现在相似上下文中”这一假设。word2vec 框架包含两种方法：跳字模型和连续词袋模型，见图[2.5](#page-16-0)。

*跳字模型*（skip-gram）根据给定的目标词预测周围可能出现的上下文词。模型会提高真实目标词—上下文词配对（正样本）之间的余弦相似度，同时降低随机配对（负样本）之间的相似度。由于每个目标词—上下文词配对都作为独立训练样本，即使是低频词的错误预测也会受到惩罚。因此，skip-gram 擅长捕捉细粒度的语义差异，也能为罕见词生成较准确的表示。

*连续词袋模型*（CBOW）则反过来，根据一组上下文词预测一个目标词。预测前先对上下文向量求平均，相当于弱化各个上下文词的细节。这种平滑处理形成了 CBOW 的特点：它比 skip-gram 训练快得多，也善于捕捉高频词的一般句法关系，但对低频词的敏感度较低。

#### 2.3.2 转向上下文化嵌入

Word2vec 推动了高质量词嵌入的发展，但它生成的是*静态嵌入*：无论词语出现在哪种语境中，它始终映射到同一个向量。对于一词多义，这会造成根本问题。以 *bank* 在下面两个句子中的用法为例：

- 我需要去 *bank*（银行）取钱。
- 游泳者坐在河岸（*bank*）上。

{16}------------------------------------------------

<span id="page-16-0"></span>![](assets/_page_16_Diagram_2.jpeg)

图 2.5：跳字模型（左）和连续词袋模型（右）构建词嵌入的示意图[Zhang 等，2023](#page-32-0)。

在静态模型中，*bank* 只有一个固定表示，需要在不同词义之间折中，最终得到的向量既不能准确表示金融含义，也不能准确表示地理含义，甚至可能是两者的混合。BERT、GPT 等现代架构（本章后文会介绍）通过生成*上下文化嵌入*来解决这个问题。模型不再查找静态词表，而是处理整个序列，并根据周围的词（如 *money*、*withdraw* 与 *river*、*swimmer*）动态计算 *bank* 的向量，从而表示该词在当前语境中的具体含义。

## 2.3.3 循环神经网络

嵌入能够表达单个词的复杂语义，但我们仍需要高效处理词语序列的方法。例如，训练分类模型时，通常要作出独立性假设，以简化计算依赖。我们可以用滑动窗口把相关词语的嵌入拼接成一个向量来表示句子，但这会迅速增大分类器的输入规模，也难以扩展到长文本。

在人类理解语言时，我们对下一个词的预期取决于此前看到或听到的内容。此前的模型通过独立性假设近似这种规律，如公式[2.2](#page-9-0)所示。形式上，设输入序列为 $x_1,x_2,\ldots,x_t$，对应时间步 $1$ 到 $t$。此时，下一个词的预测可以写作 $P(x_{t+1}\mid x_1,\ldots,x_t)$；也可以把 $x_{t+1}$ 称为时间步 $t$ 的预测目标 $y_t$。

为了利用序列中的历史信息并生成相应预测，模型需要在当前输入之外保留先前的隐藏状态。*循环神经网络*（RNN）通过循环连接实现这一点：当前时刻的状态取决于当前输入和前一时间步的网络状态。RNN 除了生成预测值（例如词语或词元）$y_t$，还会跟踪隐藏状态 $h_t$。

{17}------------------------------------------------

<span id="page-17-0"></span>![](assets/_page_17_Diagram_2.jpeg)

图 2.6：展开后的 RNN（图片摘自 Chris Olah 的博客）[Olah 等，2015](#page-30-0)。

![](assets/_page_17_Diagram_4.jpeg)

图 2.7：LSTM 的重复模块（图片摘自 Chris Olah 的博客）[Olah 等，2015](#page-30-0)。

RNN 可用以下方程表示；图 2.6 展示了 RNN 架构及其随时间展开的形式，图 2.7 则展示了 LSTM 模块：

$h_t=f(Uh_{t-1}+Wx_t+b_h)$ (2.7)

$y_t=g(Vh_t+b_y)$ (2.8)

这里引入了新的可训练参数矩阵 **U**、**V** 和 **W**，以及偏置项 **b**，让模型能够学习如何更新隐藏状态并生成输出 logits。理论上，RNN 可以处理任意长度的序列，因此在表示上下文时不受 n-gram 式马尔可夫假设的限制。

RNN 使用*时间反向传播*（BPTT）进行训练：将时间依赖关系展开，把梯度从时间步 $t$ 逐步传回更早的时间步 $t-1,t-2,\ldots$。由于每个时间步的参数更新都递归依赖前一时间步，这一过程较为敏感。链式法则要求重复进行矩阵乘法，可能导致*梯度消失或梯度爆炸*[\[Bengio et al.,](#page-27-0) [1994,](#page-27-0) [Pascanu et al.,](#page-31-0) [2013\]](#page-31-0)。因此，在 RNN 中控制信息流并不容易。与较近时间步的信息相比，早期时间步的信息对梯度的影响往往较弱，实践中 RNN 也因此难以学到重要信号。

{18}------------------------------------------------

<span id="page-18-0"></span>**LSTM**（长短期记忆网络）最初用于缓解梯度问题[Hochreiter，1997](#page-29-0)。LSTM 通过门控机制控制信息流，决定保留、更新或输出哪些信息（见图[2.7](#page-17-0)）。在每个时间步 $t$，LSTM 维护两个状态：细胞状态 $c_t$ 和隐藏状态 $h_t$：

$f_t=\sigma(U_fh_{t-1}+W_fx_t)$

$k_t=c_{t-1}\odot f_t$

$g_t=\tanh(U_gh_{t-1}+W_gx_t)$

$i_t=\sigma(U_ih_{t-1}+W_ix_t)$

$j_t=g_t\odot i_t$

$c_t=j_t+k_t$

$o_t=\sigma(U_oh_{t-1}+W_ox_t)$

$h_t=o_t\odot\tanh(c_t)$

其中，$\odot$ 表示两个向量的逐元素乘法，也称 Hadamard 积。除 LSTM 外，还有门控循环单元（GRU）等变体，在计算复杂度和可训练参数数量之间取得平衡[Cho 等，2014](#page-28-0)。

循环神经网络在语言建模的发展中发挥了重要作用。有关 RNN、LSTM 和 GRU 的更多说明，可参见 [Olah 等，2015](#page-30-0)、[Bishop 和 Bishop，2023](#page-27-0)以及[Goodfellow 等，2014](#page-28-0)。

**编码器—解码器** 目前为止，我们将任务表述为语言建模：根据前文预测下一个词。对于机器翻译等任务，还可以采用另一种任务形式。

机器翻译看起来像是根据输入 $x_1,\ldots,x_t$ 预测词元 $x_{t+1}$，但它实际属于*序列到序列*（seq2seq）问题：输入是某种语言的句子 $x_1,\ldots,x_n$，输出是另一种语言的译文 $y_1,\ldots,y_m$。输入和输出长度 $n$、$m$ 可以不同，译文的语序也可能发生变化。例如：

*We will celebrate the relief of Leiden*（7 个词）

*Wij gaan Leids Ontzet vieren*（5 个词）

编码器—解码器架构适用于这类输入、输出长度可能不同的 seq2seq 任务。编码器和解码器是两个不同的 RNN，分别处理可变长度的输入和输出。编码器逐步处理输入序列，并将其中的信息压缩成定长向量，通常称为上下文向量。<sup>3</sup>

{19}------------------------------------------------

<span id="page-19-0"></span>![](assets/_page_19_Picture_2.jpeg)

![](assets/_page_19_Diagram_3.jpeg)

图 2.8：编码器—解码器架构示意图。

这种做法会形成一个*信息瓶颈*：输入序列 $x_1,\ldots,x_t$ 的信息被压缩到单个隐藏向量中，因而难以避免信息损失。架构见图 2.8。

上下文向量连接编码器和解码器。它是编码器的最终隐藏状态，并作为解码器的初始隐藏状态。该向量表示整个源句子的语义，帮助解码器生成目标序列。解码器是另一个 RNN，以这个上下文向量为条件，逐个生成输出词元，直到生成特殊的序列结束词元。

原则上，这种方法能够奏效。但将较长的输入句子压缩成一个固定长度的上下文向量会造成信息损失，即*瓶颈问题*。此外，RNN 读到长序列末尾时，往往难以保留开头的信息。这两个问题促使研究者寻找在编码器和解码器之间传递信息的动态方法，下一节将介绍其中的注意力机制。

#### 2.3.4 注意力

为解决瓶颈问题，研究者尝试将整个输入序列中的信息有效传递给解码器。生成过程中，解码器可以据此判断输入序列的哪些部分与下一个词的生成有关。

这一研究催生了多种注意力机制[\[Bahdanau 等，](#page-27-0) [2015；Kim 等，](#page-29-0) [2017\]](#page-29-0)。图[2.9](#page-20-0)展示了注意力机制对翻译效果的影响：以 BLEU 分数衡量翻译质量[\[Papineni 等，](#page-31-0) [2002\]](#page-31-0)，分数越高越好。采用注意力机制的 RNNsearch 架构在长句翻译中仍能保持较高质量；标准 RNN 编码器—解码器模型 RNNenc 的表现则会随句子变长而明显下降。

注意力的核心思想是：上下文向量不再只依赖编码器最后一个隐藏状态，而是对所有编码器隐藏状态进行加权组合。这样，模型就能回看输入序列中与当前预测有关的部分。

<sup>3</sup>这里的上下文向量不要与 LSTM 中的 context vector 混淆。NLP 研究领域有时会在较短时间内以不同方式使用同一术语，因此论文中的用语也会随时间变化。

{20}------------------------------------------------

<span id="page-20-0"></span>![](assets/_page_20_Figure_2.jpeg)

图 2.9：生成译文的双语评估替代（BLEU）分数与句子长度的关系[\[Bahdanau 等，](#page-27-0) [2015\]](#page-27-0)。

生成当前词时，模型会为每个输入词元 $x_i$ 分配注意力权重 $\alpha_{ji}$，以控制其对第 $j$ 个输出的影响。相应的上下文向量是所有编码器隐藏状态的加权和：

$\mathbf{c}_j=\sum_{i=1}^{n}\alpha_{ji}\mathbf{h}_i$ (2.9)

权重 $\alpha_{ji}$（通常称为注意力权重）不是固定的，而是动态计算的。隐藏状态 $h_i$ 表示词元或词语的语义。模型可以通过比较当前解码器状态与各编码器隐藏状态的相似度（例如点积）来估计相关性，再对分数应用 softmax，使权重之和为 1。由此得到输入词元上的概率分布，表示模型当前关注的位置。

注意力提高了翻译准确性，但 RNN 仍然需要按顺序逐个处理词元，无法充分利用 GPU 的并行计算能力。若要扩展到大型数据集，就需要去掉这种循环结构，改用可以并行处理全部词元的固定上下文宽度结构。*自注意力*正是实现这一目标的方法。公式 2.10 将上下文化表示写作：

$\mathbf{a}_3=\sum_{j=1}^{5}\alpha_{3j}\mathbf{x}_j=\sum_{j=1}^{5}\operatorname{softmax}(\mathbf{x}_3\cdot\mathbf{x}_j)\mathbf{x}_j$ (2.10)

{21}------------------------------------------------

$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V$ (2.11)

<span id="page-21-0"></span>其中，$Q$ 表示当前关注内容的查询矩阵，$K$ 是供查询比较的键矩阵，$V$ 则包含要提取的实际内容；$d_k$ 是键向量的维度，用于缩放分数。随着 $d_k$ 增大，点积 $QK^{\mathsf T}$ 可能变得很大，使 softmax 落入梯度较小的区域。除以 $\sqrt{d_k}$ 可以缓解这一问题，使训练更稳定。

查询、键和值经过各自的线性投影后共同构成注意力计算。单个注意力头可以学习词元之间的一类关系；但语言中存在多种关系，例如句法关系（“这个名词对应哪个动词？”）、语义关系（“这个词是否属于某个隐喻？”）和指代关系（“他指的是谁？”）。多头注意力让模型能够并行关注不同关系，再将各头的结果拼接为更丰富的表示。多头注意力可写为公式 2.12：

$\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O$ (2.12)，其中 $\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)$。

#### 2.3.5 Transformer

如何用注意力机制构建语言模型架构？下面介绍一种由 Vaswani 等人在开创性工作中提出的架构——Transformer[\[2017\]](#page-32-0)。

注意力机制是 Transformer 的核心。原始 Transformer 架构由编码器和解码器组成，见图[2.10](#page-22-0)：图左侧是编码器堆栈，右侧是解码器堆栈。大多数 LLM 采用仅解码器架构。

图中展示的 Transformer 架构主要由三部分组成：嵌入层、Transformer 块和输出概率。

1. **嵌入层**：先对输入文本进行分词，再将词元映射为低维嵌入向量，并加入位置编码以保留词序。

{22}------------------------------------------------

<span id="page-22-0"></span>![](assets/_page_22_Diagram_2.jpeg)

图 2.10：Transformer 架构示意图[\[Godoy,](#page-28-0) [2020；Vaswani 等，](#page-32-0) [2017\]](#page-32-0)。

2. **Transformer 块**：这是架构的核心组件。大多数模型会将多个 Transformer 块堆叠成若干层。每个块包含以下部分：
   - **自注意力机制**：在词元之间建立长距离联系，捕捉上下文及词元间的关系。多个注意力头从不同角度学习这些关系，例如局部句法关系和更广泛的语义上下文。编码器使用自注意力；解码器使用掩码自注意力，避免模型看到未来词元。
   - **多层感知器（MLP）**：全连接前馈网络，逐个处理词元。注意力层建立词元之间的联系，MLP 则进一步变换每个词元的表示。
   - **归一化与残差连接**：Transformer 块通常使用归一化、dropout 正则化和残差连接，以缓解梯度消失或爆炸并提升训练稳定性。

{23}------------------------------------------------

<span id="page-23-0"></span>3. **输出概率**：线性层和 softmax 层将处理后的嵌入映射为概率，用于预测下一个词元。

与 RNN、LSTM 等早期方法相比，Transformer 有两项主要优势：一是能够大规模并行处理；二是能够建立细粒度的长距离语义联系，因此兼具速度和语义准确性。

Transformer 自提出以来一直主导语言建模。随着大规模、高质量训练语料的出现，基于 Transformer 的 LLM 在问答、数据到文本生成等应用任务中取得了出色表现。研究者持续扩大模型规模，希望获得更强能力[\[Brown et al.,](#page-27-0) [2020\]](#page-27-0)。[第四章](../04-scaling-laws-and-efficiency/04-scaling-laws-and-efficiency-zh.md)将讨论如何进一步提升性能。

Transformer 架构较为复杂，网上有交互式教程可供深入学习。<sup>4</sup>相关案例见[第五章](../05-case-study-1-building-llms-from-scratch/05-case-study-1-building-llms-from-scratch-zh.md)。

### 2.3.6 仅编码器模型（BERT）

原始 Transformer 将编码器和解码器组合在同一架构中。现代模型则通常只采用其中一种，因为两类架构在不同任务上各有优势。

编码器通过掩码语言建模（MLM）等方法训练，把输入压缩为有意义的表示（嵌入），适合文本分类、情感分析等自然语言理解（NLU）任务。解码器则更适合文本生成、改写和摘要等自然语言生成（NLG）任务。

仅编码器模型旨在为输入文本生成高质量的向量表示。与只能关注过去词语的因果解码器不同，编码器具有双向注意力：序列中的每个词元都可以同时关注其他词元，从而利用全局上下文构建复杂的语义表示。训练编码器时，常采用掩码语言建模。以原始 BERT 为例，约 15% 的词元位置会被选作预测目标：其中 80% 替换为 `[MASK]`，10% 替换为随机词元，另有 10% 保持不变[\[Devlin et al.,](#page-28-0) [2018\]](#page-28-0)。模型根据上下文预测这些位置原有的词元，通常使用交叉熵损失进行训练。由于预测时可以同时利用左侧和右侧上下文，MLM 训练会促使模型学习语法、语义以及一定的常识。

除 `[MASK]` 外，编码器训练通常还会使用其他特殊词元：

- `[CLS]`（分类词元）：放在序列开头。经过所有网络层处理后，其最终状态可作为整个序列的汇总表示。

<sup>4</sup>例如，参见 <https://poloclub.github.io/transformer-explainer/>

{24}------------------------------------------------

<span id="page-24-0"></span>- `[SEP]`（分隔词元）：标记句子或片段的边界，对问答等任务尤为重要。
- `[PAD]`（填充词元）：将同一训练批次中的序列补齐到相同长度。

最著名的仅编码器模型是 BERT，即“来自 Transformer 的双向编码器表示”[\[Devlin et al.,](#page-28-0) [2018\]](#page-28-0)。BERT 及其后续模型 RoBERTa[\[Liu et al.,](#page-30-0) [2019\]](#page-30-0)和 ALBERT[\[Lan et al.,](#page-29-0) [2019\]](#page-29-0)都很受欢迎。

仅编码器模型适用于迁移学习：先将输入编码为嵌入，再针对分类等下游任务进行微调。构建这类模型通常分为两个阶段：

1. **预训练**：先在包含数十亿词的大型语料库上训练模型，使其学习语言规律，例如 *King* 与 *Man*、*Queen* 与 *Woman* 之间的关系。此时模型还没有针对特定任务训练。预训练成本高，可能需要数周甚至数月。
2. **微调**：在预训练模型顶部添加一个小型分类层（通常称为*分类头*），然后使用垃圾邮件检测或情感分析等任务专用数据继续训练整个模型。由于模型已经学到语言规律，通常只需几千个示例就能适应新任务。微调成本较低，可能只需数小时或数分钟；若从头训练，则需要更多数据和时间。

介绍完仅编码器模型后，下面转向仅解码器模型。

#### 2.3.7 仅解码器模型（GPT）

文本生成通常使用仅解码器模型。解码器通过自回归训练重建数据或生成序列，通常一次生成一个词元。下一个词元只能依赖已经生成的内容，这种注意力形式称为*因果注意力*。模型不能利用未来词元来预测当前词元；因此，解码器必须使用前瞻掩码（通常为三角矩阵），将未来词元的注意力分数设为 $-\infty$。例如，预测 *The boy ate the...* 中的 *apple* 时，模型在训练中不能提前看到 *apple*，只能依赖此前的上下文。编码器使用掩码语言建模，解码器则使用常规的*因果语言建模*，见[第三章的公式 3.1](../03-llm-training-pipeline/03-llm-training-pipeline-zh.md#page-4-0)。

因此，解码器的预训练与微调承担着不同作用。

{25}------------------------------------------------

<span id="page-25-0"></span>预训练时，解码器学习预测下一个词元；微调时，则使用另一类数据和相同的损失函数继续训练。这也带来了*上下文学习*（ICL，又称提示学习）：解码器经过训练能够续写序列，因此只需给它一个提示（上下文），它就能根据提示中的示例生成答案，而无需更新模型权重。模型仿佛是从提示示例中学习[\[Brown et al.,](#page-27-0) [2020\]](#page-27-0)。大型解码器易于通过 ICL 使用，不必为每项任务进行昂贵的专门训练或微调，因此 ICL 受到广泛关注。

仅解码器模型包括 GPT（生成式预训练 Transformer）系列[\[Radford et al.,](#page-31-0) [2018, 2019; Brown et al.,](#page-27-0) [2020; Ouyang et al.,](#page-30-0) [2022; Achiam et al.,](#page-27-0) [2023\]](#page-27-0)。这类模型已取得显著成功，并广泛应用于现实场景。开源和闭源模型还在不断涌现。本章讨论的训练方法构成了现代 LLM 流程的基础，[第三章](../03-llm-training-pipeline/03-llm-training-pipeline-zh.md)将详细介绍。

# 2.4 结语

本章梳理了自然语言处理从语言学基础发展到当代大语言模型神经网络架构的历程。人类语言结构复杂，既有句法和形态，也包含细微的语义与语用特征。NLP 经历了符号、统计和神经网络等范式转变；这些发展带来了稠密嵌入和注意力等现代语言模型的核心组件，也推动了数据规模、模型规模和算力的增长。编码器通常适合自然语言理解，解码器通常适合自然语言生成。两类模型都依赖预训练和微调；解码器模型还支持基于提示的上下文学习。

#### 2.4.1 尚待解决的问题

尽管现代 LLM 表现出色，也有许多实用场景，但仍面临重要挑战。理解这些挑战有助于发现研究空白，并构建更稳健的系统；这些问题也与智能体式 LLM 密切相关。

**规模扩展。** 当前语言建模通常遵循*规模定律*：随着参数量、数据量和计算预算增加，模型的泛化能力可能提升。但扩大规模也带来显著成本。达到最先进水平的系统可能需要数十亿参数，训练算力成本可达数百万美元，并造成明显的环境影响。如何通过高效架构提升模型对训练中未见任务的泛化能力，仍是开放问题。当前研究正在探索模型剪枝和量化（见[第四章](../04-scaling-laws-and-efficiency/04-scaling-laws-and-efficiency-zh.md)）；合成数据生成（见[第十九章](../19-synthetic-data-generation/19-synthetic-data-generation-zh.md)）则可能在不增加额外训练数据的情况下提升泛化能力。

{26}------------------------------------------------

<span id="page-26-0"></span>但模型规模与泛化能力之间的最佳平衡点仍未确定。

**数据稀缺** NLP 训练数据集和主要基准大多以英语为主，例如 GLUE（General Language Understanding Evaluation，通用语言理解评估）[\[Wang et al.,](#page-32-0) [2018\]](#page-32-0)和 SQuAD（问答）[\[Rajpurkar et al.,](#page-31-0) [2016\]](#page-31-0)。这些数据通常来自大规模网络抓取。中文、印地语等资源较丰富的语言也能受益于此；而荷兰语、弗拉芒语等低资源语言缺少足够的数字内容和文化资源，难以从头训练现代 Transformer。模型会继承训练数据中的文化特点，因此非英语使用者可能遇到性能下降，或遭遇偏向西方视角与语言规范的算法偏见。

**多语言性。** 由于可用语言资源差异很大，NLP 很少能为每种语言提供通用方案。把英语模型的成功迁移到其他语言会遇到许多语言学难题。土耳其语、芬兰语和阿拉伯语等语言的形态更丰富，前缀和后缀能够向动词或名词编码大量信息；一个土耳其语词可能对应完整的英语句子。如果模型没有用这些语言进行训练，就难以体现其丰富的语言结构。

不同语言还可能采用不同的书写系统。处理非拉丁文字（例如汉字，以及希伯来语和阿拉伯语使用的辅音音素文字，即 abjad）也会带来困难，需要设计合适的分词和嵌入策略，才能得到有效的子词单元。

**语言演变** 语言是不断变化的，变化速度往往快于模型重新训练的速度，因此会产生词表外（OOV）问题。即使采用子词分词，例如字节对编码（BPE）[\[Gage,](#page-28-0) [1994\]](#page-28-0)或 WordPiece[\[Song et al.,](#page-32-0) [2021\]](#page-32-0)，模型仍可能难以处理训练截止时间之后出现的新词、俚语、技术术语和人名。2022 年训练的模型也可能不了解 2026 年的全球事件或文化变化（例如“Z 世代凝视”）。如何持续更新模型知识，同时避免遗忘旧信息（灾难性遗忘），仍是开放问题[\[Wu et al.,](#page-32-0) [2024\]](#page-32-0)。

**超越文本的词元化。** 随着人们尝试理解传统文本以外的信息，文本与其他模态之间的界限正在模糊。多模态表示的一项挑战，是将图像、音频和视频编码到与文本相同的潜在空间中。文本由离散的词或字符构成，而音频和视频是需要采样、量化的连续信号。要把视频帧或音频片段映射成可与文本共同处理的词元，可能需要视觉 Transformer（ViT）[\[Han et al.,](#page-29-0) [2022\]](#page-29-0)和对比式语言—图像预训练（CLIP）[\[Radford et al.,](#page-31-0) [2021\]](#page-31-0)等专门方法。[第七章](../07-vision-action-and-world-models/07-vision-action-and-world-models-zh.md)将讨论多模态模型。

{27}------------------------------------------------

<span id="page-27-0"></span>**可解释性。** 一个重要的开放问题是模型内部如何运作。我们可以测量模型的表现，却往往难以解释它为何给出特定答案。模型也可能出错或产生幻觉，以令人信服的方式给出错误回答。在军事、法律和医疗等高风险应用中，缺乏透明度可能成为语言模型落地的重要障碍。[第二十一章](../21-mechanistic-interpretability-for-llm-agents/21-mechanistic-interpretability-for-llm-agents-zh.md)将介绍机制可解释性等研究方法。

除上述问题外，语言建模还面临许多社会技术挑战，包括伦理对齐和长篇逻辑推理等。NLP 持续发展，一个问题的解决往往会为下一代学生、研究者和工程师带来更多深层问题。

### 2.4.2 后续内容

本章介绍了从早期方法到现代解码器模型的语言建模基础。接下来，[第三章](../03-llm-training-pipeline/03-llm-training-pipeline-zh.md)将进一步讨论智能体式 LLM 的训练流程。

{28}------------------------------------------------

<span id="page-28-0"></span>Language models are few-shot learners. *Advances in neural information processing*

- *systems*, 33:1877–1901, 2020. [52,](#page-23-0) [54](#page-25-0) KyungHyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On the properties of neural machine translation: Encoder-decoder approaches. *CoRR*, abs/1409.1259, 2014. URL <http://arxiv.org/abs/1409.1259>. [47](#page-18-0) Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009. [40](#page-11-0) Li Deng and Yang Liu. *Deep learning in natural language processing*. Springer, 2018. [42](#page-13-0) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018. [53](#page-24-0) Chenhe Dong, Yinghui Li, Haifan Gong, Miaoxin Chen, Junxin Li, Ying Shen, and Min Yang. A survey of natural language generation. *ACM Comput. Surv.*, 55(8), December 2022. ISSN 0360-0300. doi: 10.1145/3554727. URL [https://doi.](https://doi.org/10.1145/3554727) [org/10.1145/3554727](https://doi.org/10.1145/3554727). [32](#page-3-0) Philip Gage. A new algorithm for data compression. *The C Users Journal*, 12(2): 23–38, 1994. [37,](#page-8-0) [55](#page-26-0) Albert Gatt and Emiel Krahmer. Survey of the state of the art in natural language generation: Core tasks, applications and evaluation. *Journal of Artificial Intelligence Research*, 61:65–170, 2018. [32](#page-3-0) Daniel Voigt Godoy. dl-visuals. [https://github.com/dvgodoy/](https://github.com/dvgodoy/dl-visuals/?tab=readme-ov-file) [dl-visuals/?tab=readme-ov-file](https://github.com/dvgodoy/dl-visuals/?tab=readme-ov-file), 2020. URL [https://github.](https://github.com/dvgodoy/dl-visuals/?tab=readme-ov-file) [com/dvgodoy/dl-visuals/?tab=readme-ov-file](https://github.com/dvgodoy/dl-visuals/?tab=readme-ov-file). [51](#page-22-0) Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In *Advances in Neural Information Processing Systems*, pages 2672–2680, 2014. [47](#page-18-0) Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep learning*. MIT press, 2016. [40](#page-11-0) Laura Graesser and Wah Loon Keng. *Foundations of deep reinforcement learning: theory and practice in Python*. Addison-Wesley Professional, 2019. [42](#page-13-0)
- H. P. Grice. *Logic and Conversation*, page 41–58. BRILL, December 1975. ISBN 9789004368576. doi: 10.1163/9789004368811 003. URL [http://dx.doi.](http://dx.doi.org/10.1163/9789004368811_003) [org/10.1163/9789004368811\\_003](http://dx.doi.org/10.1163/9789004368811_003). [33](#page-4-0)

{29}------------------------------------------------

<span id="page-29-0"></span>Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on vision transformer. *IEEE transactions on pattern analysis and machine intelligence*, 45(1):87– 110, 2022. [55](#page-26-0) S Hochreiter. Long short-term memory. *Neural Computation MIT-Press*, 1997. [47](#page-18-0) John Hutchins. Retrospect and prospect in computer-based translation. In *Proceedings of Machine Translation Summit VII*, pages 30–36, Singapore, Singapore, September 13-17 1999. URL <https://aclanthology.org/1999.mtsummit-1.5/>. [30](#page-1-0) Daniel Jurafsky and James H. Martin. *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, with Language Models*. Stanford, 3rd edition, 2025. URL [https:](https://web.stanford.edu/~jurafsky/slp3/) [//web.stanford.edu/˜jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/). Online manuscript released August 24, 2025. [30,](#page-1-0) [38,](#page-9-0) [41](#page-12-0) Henry Kautz. The third ai summer, henry kautz,. AAAI 2020 Robert S Engelmore Memorial Award Lecture, 02 2020. [42](#page-13-0) Yoon Kim, Carl Denton, Luong Hoang, and Alexander M Rush. Structured attention networks. *arXiv preprint arXiv:1702.00887*, 2017. [48](#page-19-0) Lea Krause and Piek T.J.M. Vossen. The Gricean maxims in NLP—a survey. In Saad Mahamood, Nguyen Le Minh, and Daphne Ippolito, editors, *Proceedings of the 17th International Natural Language Generation Conference*, pages 470–485, Tokyo, Japan, September 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.inlg-main.39. URL [https://aclanthology.org/2024.](https://aclanthology.org/2024.inlg-main.39/) [inlg-main.39/](https://aclanthology.org/2024.inlg-main.39/). [33](#page-4-0) Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012. [40](#page-11-0) Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. *CoRR*, abs/1909.11942, 2019. URL [http://arxiv.org/abs/](http://arxiv.org/abs/1909.11942) [1909.11942](http://arxiv.org/abs/1909.11942). [53](#page-24-0) Jiwei Li and Dan Jurafsky. Neural net models of open-domain discourse coherence. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel, editors, *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 198–209, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1019. URL [https://aclanthology.](https://aclanthology.org/D17-1019/) [org/D17-1019/](https://aclanthology.org/D17-1019/). [32](#page-3-0) Alisa Liu, Zhaofeng Wu, Julian Michael, Alane Suhr, Peter West, Alexander Koller, Swabha Swayamdipta, Noah Smith, and Yejin Choi. We're afraid language models aren't modeling ambiguity. In Houda Bouamor, Juan Pino, and Kalika Bali,

{30}------------------------------------------------

- <span id="page-30-0"></span>editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 790–807, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.51. URL [https:](https://aclanthology.org/2023.emnlp-main.51/) [//aclanthology.org/2023.emnlp-main.51/](https://aclanthology.org/2023.emnlp-main.51/). [32](#page-3-0) Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. *CoRR*, abs/1907.11692, 2019. URL [http:](http://arxiv.org/abs/1907.11692) [//arxiv.org/abs/1907.11692](http://arxiv.org/abs/1907.11692). [53](#page-24-0)
- C. Manning and H. Schutze. *Foundations of Statistical Natural Language Processing*. Foundations of Statistical Natural Language Processing. MIT Press, 1999. ISBN 9780262133609. URL [https://books.google.nl/books?](https://books.google.nl/books?id=ZL34DwAAQBAJ) [id=ZL34DwAAQBAJ](https://books.google.nl/books?id=ZL34DwAAQBAJ). [38](#page-9-0) Christopher Manning, Mihai Surdeanu, John Bauer, Jenny Finkel, Steven Bethard, and David McClosky. The Stanford CoreNLP natural language processing toolkit. In Kalina Bontcheva and Jingbo Zhu, editors, *Proceedings of 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, pages 55–60, Baltimore, Maryland, June 2014. Association for Computational Linguistics. doi: 10.3115/v1/P14-5010. URL [https://aclanthology.org/](https://aclanthology.org/P14-5010/) [P14-5010/](https://aclanthology.org/P14-5010/). [31](#page-2-0) Sabrina J. Mielke, Zaid Alyafeai, Elizabeth Salesky, Colin Raffel, Manan Dey, Matthias Galle, Arun Raja, Chenglei Si, Wilson Y. Lee, Beno ´ ˆıt Sagot, and Samson Tan. Between words and characters: A brief history of open-vocabulary modeling and tokenization in nlp, 2021. URL [https://arxiv.org/abs/2112.](https://arxiv.org/abs/2112.10508) [<sup>10508</sup>](https://arxiv.org/abs/2112.10508). [37](#page-8-0) Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of ´ word representations in vector space. In Yoshua Bengio and Yann LeCun, editors, *1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings*, 2013. URL [http:](http://arxiv.org/abs/1301.3781) [//arxiv.org/abs/1301.3781](http://arxiv.org/abs/1301.3781). [35,](#page-6-0) [44](#page-15-0) Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533, 2015. [42](#page-13-0) Christopher Olah et al. Understanding lstm networks, 2015. URL [http://colah.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) [github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). [46,](#page-17-0) [47](#page-18-0) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022. [54](#page-25-0)

{31}------------------------------------------------

<span id="page-31-0"></span>Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pages 311–318, 2002. [48](#page-19-0) Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In Sanjoy Dasgupta and David McAllester, editors, *Proceedings of the 30th International Conference on Machine Learning*, volume 28 of *Proceedings of Machine Learning Research*, pages 1310–1318, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v28/pascanu13.html) [v28/pascanu13.html](https://proceedings.mlr.press/v28/pascanu13.html). [46](#page-17-0) Aske Plaat. *Learning to play: reinforcement learning and games*. Springer, 2020. [42](#page-13-0) Aske Plaat. *Deep reinforcement learning*. Springer, 2022. [42](#page-13-0) Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. *OpenAI Blog*, 2018. [40,](#page-11-0) [54](#page-25-0) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. [54](#page-25-0) Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [55](#page-26-0) Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. *arXiv preprint arXiv:1606.05250*, 2016. [55](#page-26-0) Dan Roth. Learning to resolve natural language ambiguities: A unified approach. In Jack Mostow and Chuck Rich, editors, *Proceedings of the Fifteenth National Conference on Artificial Intelligence and Tenth Innovative Applications of Artificial Intelligence Conference, AAAI 98, IAAI 98, July 26-30, 1998, Madison, Wisconsin, USA*, pages 806–813. AAAI Press / The MIT Press, 1998. URL [http://www.](http://www.aaai.org/Library/AAAI/1998/aaai98-114.php) [aaai.org/Library/AAAI/1998/aaai98-114.php](http://www.aaai.org/Library/AAAI/1998/aaai98-114.php). [32](#page-3-0) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017. [42](#page-13-0) Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Katrin Erk and Noah A. Smith, editors, *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1715–1725, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1162. URL <https://aclanthology.org/P16-1162/>. [37](#page-8-0)

{32}------------------------------------------------

<span id="page-32-0"></span>Singerep. File:word vector illustration.jpg. Wikimedia Commons, 2022. URL [https://commons.wikimedia.org/wiki/File:Word\\_vector\\_](https://commons.wikimedia.org/wiki/File:Word_vector_illustration.jpg) [illustration.jpg](https://commons.wikimedia.org/wiki/File:Word_vector_illustration.jpg). Accessed: 2026-01-08. Licensed under CC BY-SA 4.0. [35](#page-6-0) Xinying Song, Alex Salcianu, Yang Song, Dave Dopson, and Denny Zhou. Fast wordpiece tokenization. In *Proceedings of the 2021 conference on empirical methods in natural language processing*, pages 2089–2103, 2021. [55](#page-26-0) Richard S Sutton and Andrew G Barto. *Reinforcement Learning*. Adaptive Computation and Machine Learning series. Bradford Books, Cambridge, MA, 2 edition, November 2018. [41,](#page-12-0) [42](#page-13-0) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. [50,](#page-21-0) [51](#page-22-0) Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP workshop BlackboxNLP: Analyzing and interpreting neural networks for NLP*, pages 353–355, 2018. [55](#page-26-0) Christopher JCH Watkins and Peter Dayan. Q-learning. *Machine learning*, 8(3):279– 292, 1992. [42](#page-13-0) Joseph Weizenbaum. Eliza—a computer program for the study of natural language communication between man and machine. *Commun. ACM*, 9(1):36–45, January 1966. ISSN 0001-0782. doi: 10.1145/365153.365168. URL [https://doi.](https://doi.org/10.1145/365153.365168) [org/10.1145/365153.365168](https://doi.org/10.1145/365153.365168). [30](#page-1-0) Ludwig Wittgenstein. *Philosophical Investigations*. Basil Blackwell, Oxford, 1953. [33](#page-4-0) Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, and Gholamreza Haffari. Continual learning for large language models: A survey. *arXiv preprint arXiv:2402.01364*, 2024. [55](#page-26-0) Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google's neural machine translation system: Bridging the gap between human and machine translation, 2016. URL <https://arxiv.org/abs/1609.08144>. [37](#page-8-0) Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola. File:Continuous Bag of Words model (CBOW).svg. Wikimedia Commons, 2023. URL [https://en.wikipedia.org/wiki/File:Continuous\\_Bag\\_of\\_](https://en.wikipedia.org/wiki/File:Continuous_Bag_of_Words_model_(CBOW).svg) [Words\\_model\\_\(CBOW\).svg](https://en.wikipedia.org/wiki/File:Continuous_Bag_of_Words_model_(CBOW).svg). Accessed: 2026-01-09. Original source: d2l-ai/d2l-en. Licensed under CC BY-SA 4.0. [45](#page-16-0)

{33}------------------------------------------------

