<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 241-262; Marker split PDF pages 1-22. -->



{0}------------------------------------------------

# <span id="page-0-0"></span>第12章

# 智能体大语言模型中的搜索与检索

#### 你将学到的内容

在本章中，你将了解为什么检索对于需要超出训练数据知识的自主系统来说是一项基本能力。具体来说，你将能够：
- 描述完整的 *检索增强生成* 流程，从索引和分块到查询制定和答案生成。
- 实现自适应检索方法，如*Self-RAG*和*Self-Ask*，并在Python中构建密集检索系统。
- 将检索扩展到图形、代码、表格数据和多模态来源，而不仅限于纯文本。
- 区分自主型 RAG 与简单的一次性检索，包括实时和流式变体。
- 使用标准指标评估检索系统，例如 *Recall@K*、*平均倒数排名* 和 *可信度* 分数。

# 12.1 一般检索

在考察智能体大语言模型如何咨询外部知识之前，有必要将检索置于其自身更为悠久的学科历史中。信息检索（IR）是一个成熟的计算机与信息科学领域，其历史早于神经网络数十年。许多推动现代检索增强生成（RAG）流程的理念，都是经典信息检索研究的直接延续。该领域的标准参考著作，如[Salton and McGill](#page-19-0)、[\[1983\]](#page-19-0)，以及后来[Manning](#page-18-0)、[et al.](#page-18-0)、[\[2008\]](#page-18-0)，仍提供了本章所依赖的概念词汇，例如：索引、词项加权、排序和相关性。

{1}------------------------------------------------

<span id="page-1-0"></span>


#### 12.1.1 词袋模型

早期信息检索系统把文档和查询表示为“词袋”，并试图回答一个简单问题：哪些词项能让文档与查询良好匹配？Spärck Jones（1972）[参考文献](#page-19-0)通过逆文档频率（IDF）给出了统计学上的答案：稀有词项比几乎无处不在的词项更能区分文档。根据词项在文档中的出现频率（词频，TF）以及它在整个文档集合中的稀有程度（IDF）对词项加权，就得到 TF-IDF 系列方法；这类方法至今仍广泛用于稀疏检索。

词频统计词项 \(t\) 在文档 \(d\) 中出现的次数：

\[\operatorname{TF}(t,d)=\frac{f_{t,d}}{\sum_{t'\in d} f_{t',d}}\tag{12.1}\]

其中，\(f_{t,d}\) 是词项 \(t\) 在文档 \(d\) 中出现的原始次数。逆文档频率会降低那些出现在大量文档中的词项的权重：

\[\operatorname{IDF}(t,D)=\log\frac{|D|}{|\{d\in D:t\in d\}|}\tag{12.2}\]

其中，\(D\) 表示文档集合。将二者结合，就得到经典的 TF-IDF 权重：

\[\operatorname{TF-IDF}(t,d,D)=\operatorname{TF}(t,d)\times\operatorname{IDF}(t,D)\tag{12.3}\]

#### 12.1.2 向量空间模型

Salton 等人（1975）[参考文献](#page-19-0)将这一思路形式化为*向量空间模型*：把每篇文档和每个查询表示为高维词项空间中的向量，再按文档向量与查询向量的余弦相似度对文档排序。现代稠密检索沿用了这一模式：将文档和查询表示为同一几何空间中的点，并用距离或相似度衡量相关性。

更具体地说，文档 \(d\) 和查询 \(q\) 分别表示为词项权重向量 \(\vec d=(w_{1,d},w_{2,d},\ldots,w_{n,d})\)，通常使用 TF-IDF 权重（公式 12.3）。相关性可用查询向量和文档向量之间的余弦相似度近似表示：

\[\operatorname{sim}(\vec q,\vec d)=\cos(\theta)=\frac{\vec q\cdot\vec d}{\|\vec q\|\,\|\vec d\|}\tag{12.4}\]

文档按余弦相似度从高到低排序。稀疏检索和稠密检索执行的相似度运算相同，区别在于向量的来源：稀疏检索使用原始词项共现信息，稠密检索使用学习得到的嵌入向量，但其几何原理不变。
{2}------------------------------------------------

#### <span id="page-2-0"></span>12.1.3 潜在与稀疏表示

另外两个发展将经典检索和现代检索联系起来。首先，[Robertson and](#page-19-0) [Zaragoza](#page-19-0) [\[2009\]](#page-19-0) 提出了*概率相关框架*，该框架将检索视为一个估计文档在给定其词项统计情况下对查询的相关性概率的问题。该框架产生了 BM25 [Robertson et al.](#page-19-0) [\[1994\]](#page-19-0)，一种词项加权和排序函数，其对查询 $q$ 对文档 $d$ 的排序为：
$\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{IDF}(t,d)\cdot\frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}$ (12.5)

其中 $|d|$ 是文档 $d$ 的长度，avgdl 是集合中文档的平均长度，$k_1$ 和 $b$ 是控制词频饱和度和长度归一化的自由参数（典型值：$k_1\in[1.2,2.0]$，$b=0.75$）。BM25 仍然是稀疏检索基线，在第 [12.3.](#page-3-0) 节的混合检索讨论中被引用。

其次，[Deerwester et al.](#page-17-0) [\[1990\]](#page-17-0) 引入了*潜在语义分析*（LSA），它将奇异值分解应用于词-文档矩阵，以发现潜在主题，并根据语义而非精确词汇重叠来匹配文档。LSA 是第一个广泛使用的技术来解决词汇不匹配问题，这也是今天密集和语义检索的动机所在。
LSA 在词-文档矩阵 $X\in\mathbb{R}^{m\times n}$ 上操作，其中 $m$ 是词汇表大小，$n$ 是文档数量。其条目通常是 TF-IDF 权重（公式 [12.3\)](#page-1-0)）。应用截断奇异值分解，$X\approx U_k\Sigma_kV_k^\mathsf{T}$ (12.6)

只保留 $k$ 个最大的奇异值到 $\Sigma_k$，从而生成 $k$ 秩的 $X$ 近似。然后，在这个降维到 $k$ 的潜在空间中表示文档和词，即使没有精确的词汇重叠，语义相关的词和文档也会彼此接近。
从经典信息检索到本章中使用的神经检索方法的最后一座桥梁，是词的学习型密集向量表示的出现。[Mikolov et al.](#page-19-0) [\[2013\]](#page-19-0) 表明，通过训练简单的神经网络结构以从上下文预测词，可以产生捕捉丰富句法和语义规律的连续词嵌入。这项工作，以及随后出现的编码器架构，为密集检索、DPR 风格的双编码器，乃至最终的检索增强生成提供了表示构建模块。
换句话说，信息检索是一个有几十年历史的学科，关注一个恒久的问题：给定一个查询，哪些大型集合中的部分内容实际上与之相关？随着LLM的到来，变化的不是问题本身，而是利害关系：模型现在检索证据不仅仅是为了为人类读者对文档进行排序，而是为了影响它们自身的生成。我们现在将讨论这一转变为什么重要。
{3}------------------------------------------------

# <span id="page-3-0"></span>12.2 检索为何对智能体大语言模型很重要

大语言模型反映的是训练时可获得的知识快照。模型参数无法包含训练语料定稿之后产生的信息；对于时效性强、专有或长尾知识，模型也可能并不掌握。检索增强通过在推理时查询外部来源，并依据检索到的证据生成答案，来缓解这些限制。

检索增强生成（RAG）将外部知识源与预训练语言模型参数中编码的知识结合起来。多数 RAG 系统在推理时检索，也有一些会将检索机制纳入微调甚至预训练。常见做法是为非结构化文本建立索引，以便高效查找；有些系统还会把内容组织成知识图谱。查询处理阶段可以扩展查询，或将复杂问题拆分为多个子查询 [Cong et al., 2024](#page-17-0)。

REALM [Guu et al., 2020](#page-18-0)、DPR [Karpukhin et al., 2020](#page-18-0) 和 RAG [Lewis et al., 2020](#page-18-0) 等早期系统推动了这一方向的发展。现代 RAG 还涉及索引、查询规划、重排序、引用与来源归属，以及持续更新数据。近期综述梳理了该领域及其评估方法 [Gao et al., 2023b](#page-18-0)；[Huang et al., 2024](#page-18-0)；[Yu et al., 2024](#page-20-0)；[Sharma et al., 2025](#page-19-0)；[Peng et al., 2024](#page-19-0)。

**何时检索。** 当任务依赖大量事实知识（如定义、实体、数字或文献），信息变化很快（如新闻、价格或文档），答案需要附带引用，或系统必须遵循指定的事实来源（如合规门户、内部知识库）时，检索尤其有用。在智能体系统中，检索还可以与规划和其他工具调用交替进行（见第 11 章）。
# 12.3 检索增强生成管道

一个实用的RAG系统通常实现以下七个阶段：
- 1. 索引：索引阶段是第一步。此步骤包括：将所有原始源材料，如 HTML 页面、PDF、表格或代码，转换为干净、一致的文本。将重要结构保留为元数据（标题、锚点、时间戳、作者信息）。将内容分块，形成有意义的边界（如句子或章节）并保留小的重叠，以免丢失上下文。创建两种类型的可搜索表示：*稀疏* 表示，如 BM25（经典的词项匹配方法）或 SPLADE [\[Formal et al.,](#page-18-0) [2021\]](#page-18-0)（一种扩展词项的现代变体），适用于精确关键词匹配；以及 *密集* 表示，即捕捉语义匹配的嵌入向量。存储两者可以让系统在同一次搜索中找到精确匹配（ID、数字）和语义匹配（释义、同义词）。

{4}------------------------------------------------

- <span id="page-4-0"></span>2。查询构建：将用户的问题转化为一个或多个精确的搜索查询。这可能涉及为清晰起见重写问题，将复杂问题拆分为子问题（多跳查询）、添加相关术语（查询扩展），甚至生成一个假设答案作为搜索目标（HyDE）[\[Gao et al.,](#page-18-0) [2023a\]](#page-18-0)。对于结构化数据源，生成特定查询，例如 SQL 或图查询。目标是让搜索引擎准确理解要查找的内容。
- 3. 候选检索：使用构建的查询从索引中获取可能相关的块。密集检索器（例如 DPR [\[Karpukhin et al.,](#page-18-0) [2020\]](#page-18-0)，Contriever [\[Izacard et al.,](#page-18-0) [2022\]](#page-18-0)）通过语义匹配；稀疏方法（BM25, SPLADE）通过精确词匹配；混合方法结合两者。为了在大规模下快速搜索，可以使用近似最近邻（ANN）索引，如 HNSW [\[Malkov](#page-18-0) [and Yashunin,](#page-18-0) [2020\]](#page-18-0) 或 IVF-PQ [\[Jégou et al.](#page-18-0), [2011\]](#page-18-0)。对于代码或图像，使用专用编码器。这一步是为了广撒网而不失焦点。
- 4. 重新排序：对检索到的候选项池进行重新排序，使用更准确但较慢的模型，如交叉编码器（BERT、MiniLM）。晚期交互模型如 ColBERTv2 [\[Santhanam et al.,](#page-19-0) [2022\]](#page-19-0) 在质量和速度之间取得平衡。这确保了顶部结果是最有可能回答问题的项。
- 5. 上下文组装：去除重复内容，确保覆盖查询的不同方面（多样性），优先选择最新或与领域相关的来源。将选定的段落及其元数据（来源、URL、日期）打包成上下文包，以提供给 LLM。这为模型提供了经过整理、可靠的信息集。
- 6. 基于事实的生成：要求 LLM 使用提供的上下文生成答案，并为每个论点提供引用。如果证据不足，模型应选择弃权或请求澄清。一些系统，如 Self-RAG [\[Asai](#page-17-0) [et al.,](#page-17-0) [2023\]](#page-17-0)，会动态决定是否检索更多信息。重点是生成既正确又有充分支持的答案。
- 7. 归属与日志记录：在答案旁显示来源，以便读者验证声明。记录整个过程，包括查询、检索的文档和版本，用于调试、评估和合规。这使系统透明且可审计。

一个简单的 RAG 系统遵循上面概述的七步流程。列出的 [12.1](#page-5-0) 显示了这一过程的 Python 风格伪代码。该函数接受用户问题，将其重写为一个或多个搜索查询，检索并重新排序候选段落，组装上下文窗口，最后生成答案，同时跟踪使用了哪些段落。我们将在下面详细说明。
{5}------------------------------------------------

<span id="page-5-0"></span>

```python
# Input: question q, search index, and number of passages k
def rag_answer(q, index, k=8):
    # 1. Construct search queries (possibly decompose multi-hop questions).
    queries = rewrite_question(q)
    # 2. Retrieve candidate passages from the index.
    candidates = retrieve(index, queries, k=200)
    # 3. Re-rank with a cross-encoder or late-interaction model.
    ranked = rerank(candidates)
    # 4. Assemble a context window from the top-k passages.
    context = assemble(ranked[:k])
    # 5. Generate an answer conditioned on the context and provide citations.
    answer, citations = generate_answer(q, context)
    return answer, citations
```

*清单 12.1：基本 RAG 设置的高级流程。*
# 12.4 自适应检索方法

传统的RAG系统会检索固定数量的段落，而不管是否实际需要检索。这种“始终开启”的策略简化了工程实现，但可能损害效率和准确性。最近的研究探索了“自适应”或“门控”检索策略，这些策略教模型决定何时进行搜索以及查阅多少证据。这些方法与第6.2.4节介绍的自我反思理念相一致：语言模型不是盲目地吸收检索到的文本，而是会对自身的信心进行内省，只有在识别到知识空白时才触发检索。
具体来说，[Asai et al.](#page-17-0) [\[2023\]](#page-17-0) 提出了 Self-RAG，它训练一个单一模型来生成答案、请求检索，并使用特殊的*反思标记*对自己的输出进行批评。当模型的草稿答案不确定时，它会发出 <RETRIEVE> 标记以调用搜索工具，并在下一步中将检索到的段落纳入。实验表明，这种学习到的门控机制相比静态 RAG 提高了事实准确性，同时避免了不必要的调用。
一种更轻量的方法是 Self-Ask [\[Press et al.,](#page-19-0) [2022\]](#page-19-0)，其中模型将复杂问题分解为更简单的子问题，向搜索工具发出这些问题，并使用答案来组成最终响应。Self-Ask 可以与规划代理结合，使模型能够交错进行推理、工具调用和信息检索。Graph Toolformer [\[Zhang,](#page-20-0) [2023\]](#page-20-0) 扩展了这一思路，通过训练策略将搜索查询插入程序图。在推理时，LLM 根据当前的推理状态动态决定是从文本索引、知识图谱还是数据库中检索。

除了这些具体的例子之外，自适应检索可以通过多种方式实现：
- 基于置信度的门控：模型评估其不确定性（例如，通过熵或校准指标），并在置信度低于阈值时触发检索。
- **迭代检索：** 系统不会一次性获取所有证据，而是先检索少量段落并进行推理，再决定是否需要新一轮检索。

{6}------------------------------------------------

需要进行<span id="page-6-0"></span>这种方法适用于多跳问题。
- 查询规划：代理将用户查询分解为子任务，分派给不同的工具（搜索、数据库、计算器），并整合结果。只有当计划需要外部知识时，才会调用检索器。
- 学习检索：强化学习或有监督的微调可以教模型在适当的时候发出 <RETRIEVE> 令牌。Self-RAG 是一种实现方式；其他方法则探索与生成器端到端训练检索策略。

自适应检索通过避免不必要的搜索来减少延迟和成本，并且通常通过将注意力集中在缺失的知识上来提高答案质量。然而，设计有效的门控策略需要仔细评估（参见第 [12.7\)](#page-14-0) 节）并可能在训练期间需要额外的监督。
#### 12.4.1 Self-RAG 与 Self-Ask 的自适应检索

[Asai et al.](#page-17-0) [\[2023\]](#page-17-0) 的 Self-RAG 框架通过在生成过程中发出一个特殊的 <RETRIEVE> 令牌来教模型按需请求检索。清单 12.2 中的伪代码勾画了此类循环的运作方式。模型维护一个不断增长的上下文，并在生成输出令牌与在需要时引入新证据之间交替进行。
```python
# Input: question q, search index, and trained Self-RAG model
def self_rag_answer(q, index, model):
    context = []
    output_tokens = []
    # Generate tokens sequentially.
    for token in model.generate_stream(q, context):
        if token == "<RETRIEVE>":
            # The model requests more information.
            new_docs = retrieve(index, rewrite_question(q), k=5)
            context.extend(new_docs)
        else:
            output_tokens.append(token)
            if token == "<END>":
                break
    return detokenize(output_tokens)
```

*清单 12.2：Self-RAG 高层伪代码示例。*
需要注意的是，为了让 Self-RAG 能够工作，基础模型应该被训练或微调以生成 <RETRIEVE> 标记。另一方面，Self-Ask [\[Press et al.,](#page-19-0) [2022\]](#page-19-0) 并不依赖于语言模型。Self-Ask 会将一个复杂的问题分解成更简单的子问题，为每个子问题检索证据，并综合生成最终答案。以下高层伪代码展示了实现这一策略的一种方法：
{7}------------------------------------------------

```python
def self_ask(q, index):
    sub_questions = decompose(q)
    # For example: "When was Einstein born?" and "What is the Nobel Prize?"
    intermediate_answers = []
    for sq in sub_questions:
        docs = retrieve(index, rewrite_question(sq), k=5)
        answer = summarize(docs)
        intermediate_answers.append(answer)
    final_answer = compose(intermediate_answers)
    return final_answer
```

*清单 12.3：Self-Ask 高层伪代码示例。*
#### 12.4.2 Python中的一个简单密集检索示例

sentence-transformers 等稠密检索工具包使索引小型语料库并执行语义搜索变得容易。例 12.4 展示了如何对语料库中的句子进行嵌入、计算与查询的相似度并显示顶级结果。
```python
from sentence_transformers import SentenceTransformer, util

# Prepare a small corpus.
texts = [
    "The Eiffel Tower is in Paris.",
    "The capital of France is Paris.",
    "Berlin is the capital of Germany.",
    "The Great Wall is in China.",
]
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(texts, convert_to_tensor=True)

def semantic_search(question, k=2):
    query_emb = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    top_results = scores.topk(k)
    for score, idx in zip(top_results.values, top_results.indices):
        print(texts[idx], "(score:", score.item(), ")")

# Example usage.
semantic_search("What is the capital of France?")
```

*清单 12.4：Python 稠密检索示例。*
#### 12.4.3 查询重写与扩展

有效检索通常先把原始问题转换成索引容易匹配的形式。大语言模型可以改写问题、扩展缩略词、补充同义词，或生成假设文档作为搜索目标。
{8}------------------------------------------------

<span id="page-8-0"></span>在 HyDE 方法中，系统会将生成的文本嵌入，并将其作为伪查询使用。<sup>1</sup> 清单 12.5 展示了如何调用 LLM 生成多个查询变体并合并为搜索查询集合。
```python
def rewrite_question(q, n_variants=3):
    """Use an LLM to generate alternative phrasings and hypothetical answers.

    :param q: The user question.
    :param n_variants: Number of rewrites to generate.
    :return: A list of expanded queries.
    """
    variants = [q]
    # Ask the model for paraphrases and clarifications.
    paraphrases = llm.generate(["Rewrite: " + q] * n_variants)
    variants.extend(paraphrases)
    # Ask the model to guess an answer (HyDE) and use it as a pseudo-query.
    guess = llm.generate(["Answer briefly: " + q])[0]
    variants.append(guess)
    # Optionally add entity aliases from a knowledge base.
    aliases = lookup_aliases(q)
    variants.extend(aliases)
    return variants
```

*清单 12.5：查询重写示例。*
# 12.5 超越文本：图、代码、表格数据与多模态检索

虽然大多数RAG系统在无结构文本上运行，但许多智能应用需要对结构化数据、源代码或图像进行推理。将检索扩展到纯文本之外带来了新的机遇，但也引入了建模挑战。
**Graph-RAG。** 知识图谱将实体编码为节点，将关系编码为边。在语料上构建轻量级图可以让检索器沿着链接（例如 *works for*，*located in*）查找，而不仅仅依赖于词袋相似度。GraphRAG 调查将工作流程形式化为三步：基于图的索引、图引导的检索和图增强生成 [\[Peng et al.,](#page-19-0) [2024\]](#page-19-0)。索引从文本中提取实体和关系，对其进行规范化并构建邻域结构。然后，检索会沿图扩展查询，以找到相关节点及其关联段落。

<sup>1</sup>HyDE 代表假设文档嵌入，并生成一个合理的答案，然后将其嵌入并用于检索支持段落 [\[Gao et al.,](#page-18-0) [2023a\]](#page-18-0)。
{9}------------------------------------------------

<span id="page-9-0"></span>图结构有助于全局推理和消除歧义。生成模型还可以借助图结构将事实组织成连贯叙述，并避免相互矛盾。
#### 12.5.1 基于知识图谱的检索示例

GraphRAG 系统将知识表示为实体和关系的图，并使用图遍历来检索相关节点 [\[Peng et al.,](#page-19-0) [2024\]](#page-19-0)。可以使用 networkx 库构建一个轻量级实现。在示例 12.6 中，我们构建了一个小型知识图，并实现了两个简单的检索原语：（1）获取实体的直接邻居；（2）寻找两个实体之间的最短路径。这些操作可以支持多跳推理。
```python
import networkx as nx

def build_knowledge_graph():
    G = nx.DiGraph()
    # Add nodes representing entities.
    G.add_node("Albert Einstein", type="person")
    G.add_node("Nobel Prize", type="award")
    G.add_node("Physics", type="field")
    G.add_node("1921", type="year")
    # Add edges representing relations.
    G.add_edge("Albert Einstein", "Nobel Prize", relation="received")
    G.add_edge("Nobel Prize", "Physics", relation="category")
    G.add_edge("Nobel Prize", "1921", relation="year")
    return G

def neighbours(G, entity):
    """Return all outgoing neighbors of an entity."""
    return list(G.successors(entity))

def shortest_path(G, source, target):
    """Return the entity sequence along the shortest path."""
    try:
        return nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return []

# Example usage.
G = build_knowledge_graph()
print(neighbours(G, "Albert Einstein"))  # ['Nobel Prize']
print(shortest_path(G, "Albert Einstein", "Physics"))
# ['Albert Einstein', 'Nobel Prize', 'Physics']
```

*清单 12.6：Python 图检索示例。*

在一个完整的 GraphRAG 系统中，图节点会存储指向原始文本段落的指针，检索过程将返回节点序列以及与之关联的段落给 LLM。图引导的检索可以通过将推理基于明确的边来减少幻觉，并通过促进相关事实之间的一致性来实现。
{10}------------------------------------------------

<span id="page-10-0"></span>**Code-RAG。** 源代码具有丰富的语法结构。源代码的语义不同于自然语言。为了实现有效的代码检索，应该在函数或模块层面进行索引，包括签名、文档字符串、抽象语法树（AST）和依赖图。诸如 CodeBERT [\[Feng et al.,](#page-17-0) [2020\]](#page-17-0) 之类的模型可以学习代码与自然语言的联合嵌入；它们支持诸如自然语言代码搜索、代码摘要和文档生成等任务。在 Code-RAG 中，检索器可以获取相关的函数、API 和测试用例，而重新排序器可以使用代码感知的交叉编码器或图神经网络来优先匹配精确结果。然后将检索到的代码片段、注释和依赖元数据提供给 LLM，它可以回答问题、提出修复建议或生成新代码，同时引用源文件。评估应考虑代码的正确性、类型安全性以及对依赖关系的遵守情况。

### 12.5.2 使用 AST 和嵌入的代码检索示例

获取源代码需要理解其语法和语义。一种简单的方法是使用预训练模型（如 CodeBERT [\[Feng et al.,](#page-17-0) [2020\]](#page-17-0)）将代码片段和自然语言查询进行嵌入。另一种补充方法是将代码解析为抽象语法树（AST）并搜索其结构。清单 [12.7](#page-11-0) 中的示例展示了两者：我们使用 CodeBERT 对函数进行语义搜索嵌入，并使用 Python 的 ast 模块提取函数定义和文档字符串以进行索引。
此示例对导入模块中的函数建立索引，并在查询与编码的源代码之间执行余弦相似度搜索。在实际操作中，人们可能会将代码嵌入与词法特征（标识符、注释）结合，并采用基于 AST 的匹配来处理结构化查询（例如，查找调用特定 API 的所有函数）。对于需要在工具使用期间读取、执行或修改现有代码的智能系统来说，具备代码感知的检索能力至关重要。
### 12.5.3 多模态和异构来源

表格及其他异构文档现有的RAG方法在应用于包含文本和表格组件的异构文档时面临关键限制 [\[Yu et al.,](#page-20-0) [2025\]](#page-20-0)。一个主要挑战是*结构信息丢失*。当表格数据被线性化或分块用于基于文本的RAG时，其固有的关系结构，即行与列的相互依赖关系，会受到破坏。这种结构丢失阻碍了LLM执行准确的多跳异构推理的能力。这表明，仅仅将非文本数据转换为文本是不够的。为了解决这些挑战，已经提出了如*TableRAG* [\[Yu et al.,](#page-20-0) [2025\]](#page-20-0)这样的混合框架，将文本理解和对表格数据的复杂操作统一起来。

TableRAG 使用 SQL 编程和执行将表相关查询视为不可分割的推理单元，从而提高了计算效率和推理准确性。同样，*MixRAG* [\[Zhang et al.,](#page-20-0) [2025\]](#page-20-0) 已被证明能够显著提升异构文档 RAG 的检索性能，这种 RAG 需要对文本数据和分层表格数据进行联合检索和推理。
{11}------------------------------------------------

<span id="page-11-0"></span>

```python
from transformers import AutoTokenizer, AutoModel
import torch
import ast
import inspect

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def encode_code(code_str):
    inputs = tokenizer(code_str, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def extract_functions(module):
    """Return a list of (name, source, docstring) tuples."""
    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        source = inspect.getsource(obj)
        doc = inspect.getdoc(obj) or ""
        functions.append((name, source, doc))
    return functions

# Index a module's functions.
import example_module  # Your codebase.
functions = extract_functions(example_module)
embeddings = [encode_code(src) for (_, src, _) in functions]

def search_code(query, k=2):
    query_emb = encode_code(query)
    scores = [
        torch.nn.functional.cosine_similarity(query_emb, emb)[0]
        for emb in embeddings
    ]
    # Return the top-k functions with highest similarity.
    topk = sorted(zip(scores, functions), key=lambda x: x[0], reverse=True)[:k]
    for score, (name, src, doc) in topk:
        print(name, "(score:", score.item(), ")\n", doc)

# Example usage: search for a sorting function.
search_code("sort a list of numbers")
```

*清单 12.7：Python 中的代码检索示例。*
这些方法的成功表明，纯神经（基于嵌入）的RAG方法对于结构化数据来说是不够的。将符号推理（例如SQL）与神经检索相结合，是异构RAG的关键设计模式，这反映了智能体大语言模型在增强数据理解和精确性方面朝向神经-符号集成的更广泛趋势。
**多模态 RAG。** 许多知识库包含表格、图形、图表、音频和视频。一种简单的方法是使用 OCR 提取任何文本内容，并将其余部分作为纯文本处理。一种更丰富的方法是存储图像或其他模态的嵌入（例如，CLIP 或 ViT 表示）并将它们与标题和上下文配对。查询可以跨模态联合编码，从而允许系统
{12}------------------------------------------------

检索与文本相关的图表或方程。例如，研究代理可能会检索一张图及其标题以回答数据分析问题，或者教育代理可能会从教科书中获取示意图。多模态 RAG 仍然是一个活跃的研究领域。挑战包括跨模态对齐和高维嵌入的高效索引。此外，非文本证据应在生成的答案中得到适当的体现。
#### 12.5.4 多模态检索示例

有时，能动系统除了文本之外，还需要检索图像、图表或音频片段。多模态模型，例如 CLIP 或对比语言-音频预训练（CLAP），将自然语言和其他模态嵌入到共享空间中。清单 12.8 中的高级伪代码概述了如何使用 CLIP 查找与文本查询相关的图像：
```python
from PIL import Image
import clip
import torch

model, preprocess = clip.load("ViT-B/32")

def embed_image(path):
    image = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        return model.encode_image(image)

def embed_text(text):
    text_tokens = clip.tokenize([text])
    with torch.no_grad():
        return model.encode_text(text_tokens)

# Build an index of image embeddings.
image_paths = ["cat.jpg", "dog.png", "city.png"]
image_embs = [embed_image(p) for p in image_paths]

def search_images(query, k=2):
    query_emb = embed_text(query)
    scores = [
        torch.nn.functional.cosine_similarity(query_emb, emb)[0]
        for emb in image_embs
    ]
    topk = sorted(zip(scores, image_paths), key=lambda x: x[0], reverse=True)[:k]
    return topk

# Example usage: search for a picture of a city.
print(search_images("a skyline of a large European city"))
```

*清单 12.8：在 Python 中使用 CLIP 检索相关图像。*

多模态检索会返回最相似图像的文件名。在 RAG 环境中，检索到的图像可以通过视觉编码器提取标题或物体，然后将这些内容整合到语言模型的上下文中。因此，多模态代理可以将其输出基于文本和视觉证据。
{13}------------------------------------------------

# <span id="page-13-0"></span>12.6 代理式 RAG：超越简单检索

虽然传统的RAG系统通过外部知识增强LLM，但它们通常是被动的，依赖于静态的、预定义的流程。一个重要且近期的进展是*Agentic RAG*的概念，它将自主AI代理直接整合到检索增强生成流程中[\[Singh et al.,](#page-19-0) [2025\]](#page-19-0)。这一范式转变使RAG从单纯的工具变为一个主动的、智能的、由代理驱动的过程。Agentic RAG系统的特点是能够动态管理检索策略、迭代地优化上下文理解，并调整工作流程以满足复杂任务的需求。

Agentic RAG 代表了一种更复杂的检索循环形式，其中 LLM 代理本身负责协调检索过程。这与更简单的 RAG 方法不同，因为它采用了连续的反馈循环，这是高级代理系统的基础。代理不仅仅是一次性查询转换，而是可以根据中间结果和持续的交互主动优化其查询。这种方法解决了传统 RAG 系统的局限性，传统系统通常受限于静态工作流程，缺乏多步骤推理和复杂任务管理所需的适应性 [\[Singh et al.,](#page-19-0) [2025\]](#page-19-0)。

#### 12.6.1 代理式 RAG 的关键组成部分

自主型RAG框架通常包含几个关键组件，使其具备动态行为：
- 动态查询生成：代理根据其对任务的当前理解和检索到的上下文来制定查询，从而实现更精确和有效的搜索。这可以包括查询扩展、分解或在初始检索失败时的重写。
- 迭代优化：代理会评估获取信息的质量，并在必要时使用精炼后的查询启动新的检索周期。这个迭代过程确保最终的响应基于最相关和最准确的信息。
- 自主工具使用：该代理可以选择并使用各种工具，包括不同的检索模型（例如，稠密与稀疏）、排序算法和外部 API，以优化其检索策略。有关代理如何使用工具的更多信息，请参见第 11 章。
- 自我纠正与反思：代理可以评估自身输出的准确性和完整性。如果它检测到幻觉或无支持的主张，它可以触发新的检索和推理过程来纠正错误。

Agentic RAG 的兴起表明 RAG 在 LLM 系统中的角色发生了变化。检索从被动的数据查找转变为主动的、智能的能力，这一能力是智能体认知功能的核心部分，使其能够与外部知识源进行更复杂、适应性更强的交互。这一转变对于开发能够处理复杂现实任务的健壮且可靠的 LLM 智能体至关重要。
{14}------------------------------------------------

#### <span id="page-14-0"></span>12.6.2 实时与流式 RAG

虽然传统的 RAG 系统对于静态数据是有效的，但它们通常并不适用于需要持续实时信息的应用。本节探讨了基于持续更新、高吞吐量数据流运行的 RAG 系统的架构考虑和挑战。这类数据流的示例可以在新闻推送、社交媒体或金融市场中找到。对于流数据，仅仅查询更新后的静态索引已经不再足够。检索现在涉及动态索引管理和持续的知识同化。
在为静态数据设计的传统RAG系统中，预处理速度慢，不适合实时分析 [\[Zhu,](#page-20-0) [2025,](#page-20-0) [Sankaradas et al.,](#page-19-0) [2025\]](#page-19-0)。我们应该如何在内存预算和高数据摄取率下保持数据的新鲜度和语义覆盖？诸如 *StreamingRAG* [\[Sankaradas et al.,](#page-19-0) [2025\]](#page-19-0) 的方法尝试通过实时动态发展的知识图来克服这些问题。
LLM 代理用于医学、物流和金融等动态领域，必须能够访问实时、持续更新的信息。在这种快速变化的环境中，静态 RAG 会产生过时或不准确的信息。在许多代理应用中，实时 RAG 对获取先进、及时且可靠的信息是必要的。
# 12.7 评估

评估检索增强系统比评估独立语言模型更复杂，因为检索器和生成器都会影响最终答案。评估时应分别考察各组件，并分析它们协同工作时的表现。

**检索指标。** 对于给定查询，将检索到的段落与一组标准文档进行比较。常见指标包括：

1. **Recall@k**（公式 [12.7](#page-15-0)）和 **Precision@k**（公式 [12.8](#page-15-0)），用于衡量前 \(k\) 个结果的召回率和精确率。
2. **平均倒数排名（MRR）**（公式 [12.9](#page-15-0)）和平均准确率，用于衡量排序质量。
3. **归一化折扣累积增益（nDCG）**（公式 [12.10](#page-15-0)），对排名靠前的相关结果赋予更高权重。

BEIR 包含 18 个异质数据集，涵盖问答、事实核查和实体检索等任务。其结果显示，后期交互模型和重排序模型能够提升零样本性能，但会增加成本 [Thakur et al., 2021](#page-19-0)。KILT 在统一的维基百科快照上评估知识密集型任务，同时衡量任务表现和来源追溯能力，鼓励系统引用所依据的段落 [Petroni et al., 2021](#page-19-0)。

{15}------------------------------------------------

<span id="page-15-0"></span>

\[\operatorname{Recall}@k=\frac{|\{\text{相关文档}\}\cap\{\text{前}k\text{个检索结果}\}|}{|\{\text{相关文档}\}|}\tag{12.7}\]

\[\operatorname{Precision}@k=\frac{|\{\text{相关文档}\}\cap\{\text{前}k\text{个检索结果}\}|}{k}\tag{12.8}\]

\[\operatorname{MRR}=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\operatorname{rank}_i}\tag{12.9}\]

其中，\(\operatorname{rank}_i\) 是查询 \(i\) 的第一个相关文档的排名。

\[\operatorname{nDCG}@k=\frac{\operatorname{DCG}@k}{\operatorname{IDCG}@k}\tag{12.10}\]

其中

\[\operatorname{DCG}@k=\sum_{i=1}^{k}\frac{2^{\operatorname{rel}_i}-1}{\log_2(i+1)}\]

\(\operatorname{rel}_i\) 表示排名第 \(i\) 的文档的分级相关性。

**生成指标。** 组装上下文后，还需评估生成答案的质量。BLEU、ROUGE 和 BERTScore 等自动指标往往作用有限，因为很多任务并没有唯一的标准答案。更合适的指标包括*忠实度*和*有用性*。忠实度可通过蕴涵模型衡量：对于生成结果 \(y\) 和检索上下文 \(x\)，检查每项陈述是否得到所引段落支持，例如使用自然语言推理模型或引用核查。

\[\operatorname{忠实度}=\frac{\text{得到上下文蕴涵的输出数}}{\text{输出总数}}\tag{12.11}\]

有用性（或效用）衡量答案是否满足用户的信息需求，通常需要人工评估或专门的 LLM 评估器。在交互场景中，延迟和成本也很重要。有用性可以依据人工或模型评分计算：

\[\operatorname{有用性}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{score}(y_i,\operatorname{query}_i)\tag{12.12}\]

评分可以采用李克特量表，也可以采用成对偏好判断。

**联合评估与新指标。** 检索和生成应分别评估。系统可能检索到无关文档却猜对答案，也可能检索到理想证据却未能使用。近期研究因此提出统一评估框架。Auepora 流程建议分别报告检索和生成得分，再合并成综合指标 [Yu et al., 2024](#page-20-0)。RAGAS 等工具包提供自动化

{16}------------------------------------------------

基于小模型的评估流程，用于计算上下文召回率、答案正确性和引用准确性。评估自适应方法时（见第 [12.4](#page-5-0) 节），还要衡量检索触发策略：模型在应该检索时实际检索的频率，以及不必要地跳过检索的频率。

**来源归属与时效性。** 在许多应用中，模型必须标明信息来源并避免使用过时信息。来源归属测试通过自然语言推理将陈述与证据对应起来，并检查网址与答案是否一致。时效性指标衡量答案是否反映最新有效来源，也可以衡量版本变化带来的偏差。对于金融、医疗等时效敏感领域，来源归属和时效性尤其重要。

仔细评估有助于改进系统设计。若检索召回率高但忠实度低，可能需要改进重排序；若忠实度高但有用性低，则可能是上下文覆盖不足。为便于错误分析和审计，应透明记录查询、检索段落和模型输出。
# 12.8 结语

对于需要提供最新信息的智能体大语言模型系统，检索至关重要。语言模型是知识的静态快照；RAG 可以让答案保持新颖、透明且有依据。有效的 RAG 需要妥善设计索引、查询规划、检索、重排序、上下文组装和有依据的生成流程，并采用自适应方法让模型判断何时以及如何检索。检索还可以扩展到文本以外的数据，如知识图谱、代码和多模态内容。要构建可信系统，严谨的评估不可或缺。

**运行维护。** 在生产环境中，检索器是持续运行的服务，而非一次性脚本。爬虫、Webhook 和文档管道应持续摄取新数据。要跟踪数据版本和弃用情况，并为实体维护时间线，以便优先采用较新的事实。检索时应用访问控制，建索引时对个人身份信息进行脱敏。为控制延迟和成本，可以缓存嵌入向量及前 \(k\) 个结果，在多轮对话中复用检索结果，用抽取式摘要压缩上下文，按领域调整 \(k\) 和分块大小，并只对排名靠前的候选结果重排序。

**实践建议。** 可以从混合检索系统开始：将性能良好的稠密检索器与 BM25 或 SPLADE 结合，用交叉编码器对前 100 个结果重排序，再把 6 到 12 个段落交给 LLM。合理切分语料，例如使用 300–500 词元的窗口、10–15% 的重叠，保留标题和表格，并附上时间戳等元数据。查询也要仔细设计：重写复杂问题，

{17}------------------------------------------------

<span id="page-17-0"></span>

拆分多跳问题，扩展实体别名，并应用领域过滤器。为保证透明度，应要求答案附带引用。如果证据不足，则提出后续问题或拒绝作答。

采用基于证据的系统设计，并衡量实际效果。持续跟踪检索召回率、依据充分性、时效性和成本。尝试重排序器，并记录系统运行轨迹以便调试。

#### 动手试试！

想体验检索增强生成，可以试试 [Google Colab 上的 RAG 教程](https://colab.research.google.com/github/Agentic-LLMs/Agentic-AI-tutorials/blob/main/rag_tutorial.ipynb)。

**融入智能体 AI。** 检索是智能体大语言模型的重要组成部分，可用于规划、自我反思以及现实世界中的行动（见第 13 章）。与 Self-RAG 和 Graph Toolformer 类似，智能体流程可以依据模型的置信度或计划动态调用检索器。它们可以查询文本、知识图谱、代码和图像等不同来源，发挥各自的优势。因此，*模型*与其*记忆*之间的界限逐渐模糊，RAG 成为智能体认知循环的一部分。

**延伸阅读。** 若要深入了解，可阅读关于 RAG、评估和图检索的综述 [Gao et al., 2023b](#page-18-0)；[Huang et al., 2024](#page-18-0)；[Yu et al., 2024](#page-20-0)；[Sharma et al., 2025](#page-19-0)；[Peng et al., 2024](#page-19-0)。基础工作包括 REALM、DPR 和早期 RAG 方法 [Guu et al., 2020](#page-18-0)；[Karpukhin et al., 2020](#page-18-0)；[Lewis et al., 2020](#page-18-0)。GraphRAG、评估和代理式 RAG 仍在快速发展，持续关注新论文与基准有助于了解最新进展。
# 参考文献

Akari Asai, Victor Zhong, Shi Chen, Hannaneh Hajishirzi, et al. Self-rag: Learning to retrieve, generate, and critique for language models. *arXiv preprint arXiv:2310.11511*, 2023. [245,](#page-4-0) [246,](#page-5-0) [247](#page-6-0)

Youan Cong, Cheng Wang, Pritom Saha Akash, and Kevin Chen-Chuan Chang. Query optimization for parametric knowledge refinement in retrieval-augmented large language models. *arXiv preprint arXiv:2411.07820*, 2024. [244](#page-3-0)

Scott Deerwester, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6):391–407, 1990. doi: 10.1002/(SICI) 1097-4571(199009)41:6⟨391::AID-ASI1⟩3.0.CO;2-9. [243](#page-2-0)

Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. CodeBERT: A pre-trained model for programming and natural languages, 2020. URL [https:](https://arxiv.org/abs/2002.08155) [//arxiv.org/abs/2002.08155](https://arxiv.org/abs/2002.08155). Accepted to Findings of EMNLP 2020. [251](#page-10-0) 

{18}------------------------------------------------

<span id="page-18-0"></span>Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE v2: Sparse lexical and expansion model for information retrieval. *arXiv preprint arXiv:2109.10086*, 2021. [244](#page-3-0)

Luyu Gao, Zhuyun Dai, and Jamie Callan. HyDE: Hypothetical document embeddings for zero-shot dense retrieval. In *NeurIPS*, 2023a. [245,](#page-4-0) [249](#page-8-0)

Yang Gao et al. Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*, 2023b. [244,](#page-3-0) [258](#page-17-0)

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In Hal Daumé III and Aarti Singh, editors, *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pages 3929–3938. PMLR, 13–18 Jul 2020. URL [https://proceedings.mlr.press/v119/](https://proceedings.mlr.press/v119/guu20a.html) [guu20a.html](https://proceedings.mlr.press/v119/guu20a.html). [244,](#page-3-0) [258](#page-17-0)

Yifu Huang et al. A survey on retrieval-augmented text generation for large language models. *arXiv preprint arXiv:2404.10981*, 2024. [244,](#page-3-0) [258](#page-17-0)

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. *Transactions on Machine Learning Research*, 2022. ISSN 2835-8856. URL <https://openreview.net/forum?id=jKN1pXi7b0>. [245](#page-4-0)

Hervé Jégou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest-neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1):117–128, 2011. doi: 10.1109/TPAMI.2010.57. [245](#page-4-0)

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In *Empirical Methods in Natural Language Processing*, 2020. [242,](#page-1-0) [244,](#page-3-0) [245,](#page-4-0) [258](#page-17-0)

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020. [244,](#page-3-0) [258](#page-17-0)

Yu A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Trans. Pattern Anal. Mach. Intell.*, 42(4):824–836, April 2020. ISSN 0162-8828. doi: 10.1109/TPAMI.2018.2889473. URL [https://doi.org/10.1109/TPAMI.](https://doi.org/10.1109/TPAMI.2018.2889473) [2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473). [245](#page-4-0)

Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. *Introduction to Information Retrieval*. Cambridge University Press, Cambridge, UK, 2008. [241](#page-0-0)

{19}------------------------------------------------

<span id="page-19-0"></span>Tomáš Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In Yoshua Bengio and Yann LeCun, editors, *1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings*, 2013. URL [http:](http://arxiv.org/abs/1301.3781) [//arxiv.org/abs/1301.3781](http://arxiv.org/abs/1301.3781). [243](#page-2-0)

Bo Peng et al. Graph retrieval-augmented generation: A survey. *arXiv preprint arXiv:2408.08921*, 2024. [244,](#page-3-0) [249,](#page-8-0) [250,](#page-9-0) [258](#page-17-0)

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, et al. KILT: a benchmark for knowledge intensive language tasks. In *NAACL*, 2021. [255](#page-14-0)

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. *arXiv preprint arXiv:2210.03350*, 2022. [246,](#page-5-0) [247](#page-6-0)

Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009. doi: 10.1561/1500000019. [243](#page-2-0)

Stephen Edward Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. Okapi at trec. *NIST special publication*, 1994. [243](#page-2-0)

Gerard Salton and Michael J. McGill. *Introduction to Modern Information Retrieval*. McGraw-Hill, New York, NY, 1983. [241](#page-0-0)

Gerard Salton, Anita Wong, and Chung-Shu Yang. A vector space model for automatic indexing. *Communications of the ACM*, 18(11):613–620, 1975. doi: 10.1145/361219.361220. [242](#page-1-0)

Murugan Sankaradas, Ravi K. Rajendran, and Srimat T. Chakradhar. StreamingRAG: Real-time contextual retrieval and generation framework, 2025. URL [https://](https://arxiv.org/abs/2501.14101) [arxiv.org/abs/2501.14101](https://arxiv.org/abs/2501.14101). [255](#page-14-0)

Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In *NAACL*, 2022. [245](#page-4-0)

Chirayu Sharma et al. Retrieval-augmented generation. *arXiv preprint arXiv:2506.00054*, 2025. [244,](#page-3-0) [258](#page-17-0)

Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented generation: A survey on agentic rag. *arXiv preprint arXiv:2501.09136*, 2025. [254](#page-13-0)

Karen Spärck Jones. A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1):11–21, 1972. doi: 10.1108/eb026526. [242](#page-1-0)

Nandan Thakur, Nils Reimers, Johannes Daxenberger, and Iryna Gurevych. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *NeurIPS Datasets and Benchmarks*, 2021. [255](#page-14-0)

{20}------------------------------------------------

<span id="page-20-0"></span>Hong Yu et al. Evaluation of retrieval-augmented generation: A survey. *arXiv preprint arXiv:2405.07437*, 2024. [244,](#page-3-0) [256,](#page-15-0) [258](#page-17-0)

Xiaohan Yu, Pu Jian, and Chong Chen. Tablerag: A retrieval augmented generation framework for heterogeneous document reasoning, 2025. URL [https://arxiv.](https://arxiv.org/abs/2506.10380) [org/abs/2506.10380](https://arxiv.org/abs/2506.10380). [251](#page-10-0)

Chi Zhang, Qiyang Chen, and Mengqi Zhang. Mixture-of-rag: Integrating text and tables with large language models, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2504.09554) [2504.09554](https://arxiv.org/abs/2504.09554). [251](#page-10-0)

Jiawei Zhang. Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt. *arXiv preprint arXiv:2304.11116*, 2023. [246](#page-5-0)

Yuzhou Zhu. From static to dynamic: A streaming rag approach to real-time knowledge base, 2025. URL <https://arxiv.org/abs/2508.05662>. [255](#page-14-0)

{21}------------------------------------------------

















