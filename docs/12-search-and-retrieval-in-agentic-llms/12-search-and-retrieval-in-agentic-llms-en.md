<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 241-262; Marker split PDF pages 1-22. -->



{0}------------------------------------------------

# <span id="page-0-0"></span>Chapter 12

# Search and Retrieval in Agentic LLMs

#### What You Will Learn

In this chapter you will learn why retrieval is a fundamental capability for agentic systems that need knowledge beyond their training data. Specifically, you will be able to:

- Describe the full *retrieval-augmented generation* pipeline, from indexing and chunking to query formulation and answer generation.
- Implement adaptive retrieval methods such as *Self-RAG* and *Self-Ask*, and build a dense retrieval system in Python.
- Extend retrieval beyond plain text to graphs, code, tabular data, and multimodal sources.
- Distinguish agentic RAG from simple one-shot retrieval, including real-time and streaming variants.
- Evaluate retrieval systems using standard metrics such as *Recall@K*, *Mean Reciprocal Rank*, and *faithfulness* scores.

# 12.1 Retrieval in General

Before examining how agentic LLMs consult external knowledge, it is worth situating retrieval within its own, much older, disciplinary history. Information retrieval (IR) is a mature field of computer and information science that predates neural networks by decades. Many of the ideas that power modern retrieval augmented generation (RAG) pipelines are direct descendants of classical information retrieval research. The standard reference works of the field, [Salton and McGill](#page-19-0) [\[1983\]](#page-19-0) and later [Manning](#page-18-0) [et al.](#page-18-0) [\[2008\]](#page-18-0), still provide the conceptual vocabulary such as: indexing, term weighting, ranking and relevance, that this chapter builds on.

{1}------------------------------------------------

<span id="page-1-0"></span>


#### 12.1.1 Bag of Words

Early information-retrieval systems represented documents and queries as *bags of words* and asked a simple question: which terms make a document a good match for a query? Spärck Jones [1972](#page-19-0) gave this question a statistical answer through inverse document frequency (IDF), observing that rare terms distinguish documents better than terms found almost everywhere. Weighting terms by how often they occur in a document (term frequency, TF) and how rare they are across the collection (IDF) yields the TF-IDF family of weighting schemes, still widely used in sparse retrieval.

Term frequency counts how often a term \(t\) occurs in a document \(d\):

\[\operatorname{TF}(t,d)=\frac{f_{t,d}}{\sum_{t'\in d} f_{t',d}}\tag{12.1}\]

where \(f_{t,d}\) is the raw count of term \(t\) in document \(d\). Inverse document frequency down-weights terms that occur in many documents:

\[\operatorname{IDF}(t,D)=\log\frac{|D|}{|\{d\in D:t\in d\}|}\tag{12.2}\]

where \(D\) is the document collection. Combining the two gives the classic TF-IDF weight:

\[\operatorname{TF-IDF}(t,d,D)=\operatorname{TF}(t,d)\times\operatorname{IDF}(t,D)\tag{12.3}\]

#### 12.1.2 Vector Space Models

Salton et al. [1975](#page-19-0) formalized this idea in the *vector space model*, representing every document and query as a vector in a high-dimensional term space and ranking documents by their cosine similarity to the query. This model established a pattern that persists in modern dense retrieval: represent documents and queries as points in a shared geometric space, then use distance or similarity as a measure of relevance.

More formally, a document \(d\) and query \(q\) are each represented as a vector of term weights, \(\vec d=(w_{1,d},w_{2,d},\ldots,w_{n,d})\), typically using TF-IDF weights (Equation 12.3). Relevance is approximated by the cosine similarity between the query and document vectors:

\[\operatorname{sim}(\vec q,\vec d)=\cos(\theta)=\frac{\vec q\cdot\vec d}{\|\vec q\|\,\|\vec d\|}\tag{12.4}\]

Documents are ranked by decreasing cosine similarity. The operation is the same in sparse and dense retrieval; only the vectors differ. Sparse retrieval uses raw term co-occurrence, while dense retrieval uses learned embeddings, but the geometric principle remains unchanged.
{2}------------------------------------------------

#### <span id="page-2-0"></span>12.1.3 Latent and Sparse Representations

Two further developments bridge classical and modern retrieval. First, [Robertson and](#page-19-0) [Zaragoza](#page-19-0) [\[2009\]](#page-19-0) developed the *probabilistic relevance framework*, which treats retrieval as a problem of estimating the probability that a document is relevant to a query given its term statistics. This framework produced BM25 [Robertson et al.](#page-19-0) [\[1994\]](#page-19-0), a term-weighting and ranking function which ranks document $d$ for query $q$ as:

$\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{IDF}(t,d)\cdot\frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}$ (12.5)

where $|d|$ is the length of document $d$, avgdl is the average document length in the collection, and $k_1$ and $b$ are free parameters that control term-frequency saturation and length normalization, respectively (typical values: $k_1\in[1.2,2.0]$, $b=0.75$). BM25 remains the sparse-retrieval baseline referenced throughout the hybrid retrieval discussions in Section [12.3.](#page-3-0)

Second, [Deerwester et al.](#page-17-0) [\[1990\]](#page-17-0) introduced *latent semantic analysis* (LSA), which applies singular value decomposition to the term-document matrix to uncover latent topics and match documents by meaning rather than by exact term overlap. LSA was the first widely used technique to address the vocabulary mismatch problem, the same problem that motivates dense and semantic retrieval today.

LSA operates on the term-document matrix $X\in\mathbb{R}^{m\times n}$, where $m$ is the vocabulary size and $n$ the number of documents. Its entries are typically TF-IDF weights (Eq. [12.3\)](#page-1-0). Applying truncated singular value decomposition,

$X\approx U_k\Sigma_kV_k^\mathsf{T}$ (12.6)

retains only the $k$ largest singular values in $\Sigma_k$, producing a rank-$k$ approximation of $X$. Documents and terms are then represented in this reduced $k$-dimensional latent space, where semantically related terms and documents lie close together even without exact term overlap.

The final bridge from classical IR to the neural retrieval methods used throughout this chapter is the emergence of learned, dense vector representations of words. [Mikolov et al.](#page-19-0) [\[2013\]](#page-19-0) showed that simple neural network architectures, trained to predict a word from its context, produce continuous word embeddings that capture rich syntactic and semantic regularities. This work, together with the encoder architectures that followed, supplied the representational building blocks on which dense retrieval, DPR-style bi-encoders, and ultimately retrieval-augmented generation are built.

Retrieval, in other words, is a decades-old discipline concerned with a single enduring question: given a query, which pieces of a large collection are actually relevant to it? What has changed with the arrival of LLMs is not the question but the stakes: models now consult retrieved evidence not merely to rank documents for a human reader, but to condition their own generation. We will now discuss why that shift matters.

{3}------------------------------------------------

# <span id="page-3-0"></span>12.2 Why Retrieval Matters for Agentic LLMs

Large language models are snapshots of knowledge available at training time. Their parameters cannot contain information created after the training corpus was assembled, and they may lack timely, proprietary, or long-tail information. Retrieval augmentation addresses these limits by consulting external sources at inference time and conditioning generation on the retrieved evidence.

Retrieval-augmented generation (RAG) combines external knowledge sources with the knowledge encoded in a pretrained language model. Most RAG systems retrieve at inference time, although retrieval can also be incorporated into fine-tuning or pretraining. They commonly index unstructured text for efficient access; some also organize the material into a knowledge graph. Query processing may expand queries or split complex questions into subqueries [Cong et al., 2024](#page-17-0).

Early systems such as REALM [Guu et al., 2020](#page-18-0), DPR [Karpukhin et al., 2020](#page-18-0), and RAG [Lewis et al., 2020](#page-18-0) helped establish this design space. Modern systems also address indexing, query planning, re-ranking, citation and attribution, and continuous data refresh. Recent surveys cover the field and its evaluation practices [Gao et al., 2023b](#page-18-0); [Huang et al., 2024](#page-18-0); [Yu et al., 2024](#page-20-0); [Sharma et al., 2025](#page-19-0); [Peng et al., 2024](#page-19-0).

**When to retrieve.** Retrieval is especially useful when a task is knowledge-intensive (for example, it asks for definitions, entities, numbers, or references); when information changes quickly (such as news, prices, or documentation); when answers need citations; or when a system must follow a designated source of truth, such as a compliance portal or internal wiki. In agentic systems, retrieval can be interleaved with planning and other tool use (see Chapter 11).
# 12.3 Retrieval-Augmented Generation Pipeline

A practical RAG system typically implements the following seven stages:

- 1. Indexing: The indexing phase is the first step. This step involves: Taking all raw source material, such as HTML pages, PDFs, tables, or code, and converting it into clean, consistent text. Keep important structure as metadata (headings, anchors, timestamps, authorship). Break content into chunks with meaningful boundaries (e.g., sentences or sections) and small overlaps so context isn't lost. Create two types of searchable representations: *sparse* ones, like BM25 (a classic term-matching method) or SPLADE [\[Formal et al.,](#page-18-0) [2021\]](#page-18-0) (a more modern variant that expands terms), which are good for exact keyword matches; and *dense* ones, which are embeddings that capture meaning for semantic matches. Storing both lets the system find exact matches (IDs, numbers) and semantic matches (paraphrases, synonyms) in the same search.

{4}------------------------------------------------

- <span id="page-4-0"></span>2. Query Construction: Turn the user's question into one or more precise search queries. This can involve rewriting the question for clarity, breaking complex questions into sub-questions (multi-hop), adding related terms (query expansion), or even generating a hypothetical answer to use as a search target (HyDE) [\[Gao et al.,](#page-18-0) [2023a\]](#page-18-0). For structured data sources, generate specific queries such as SQL or graph lookups. The aim is to make the search engine understand exactly what to look for.
- 3. Candidate Retrieval: Use the constructed queries to pull in potentially relevant chunks from the index. Dense retrievers (e.g., DPR [\[Karpukhin et al.,](#page-18-0) [2020\]](#page-18-0), Contriever [\[Izacard et al.,](#page-18-0) [2022\]](#page-18-0)) match by meaning; sparse methods (BM25, SPLADE) match by exact terms; hybrids combine both. To search quickly at scale, use approximate nearest neighbor (ANN) indexes like HNSW [\[Malkov](#page-18-0) [and Yashunin,](#page-18-0) [2020\]](#page-18-0) or IVF-PQ [\[Jégou et al., 2011](#page-18-0). For code or images, use specialized encoders. This step is about casting a wide net without losing focus.
- 4. Re-ranking: Take the pool of retrieved candidates and reorder them by relevance using more accurate but slower models, like cross-encoders (BERT, MiniLM). Late-interaction models such as ColBERTv2 [\[Santhanam et al.,](#page-19-0) [2022\]](#page-19-0) balance quality and speed. This ensures that top results are the ones most likely to answer the question.
- 5. Context Assembly: Remove duplicates, ensure coverage of different aspects of the query (diversity), and prefer recent or domain-relevant sources. Bundle the selected passages, along with their metadata (source, URL, date), into a context package to feed into the LLM. This gives the model a curated, trustworthy information set.
- 6. Grounded Generation: Ask the LLM to generate the answer using the provided context, with citations for each claim. If evidence is insufficient, the model should abstain or ask for clarification. Some systems, such as Self-RAG [\[Asai](#page-17-0) [et al.,](#page-17-0) [2023\]](#page-17-0), decide on the fly whether to retrieve more information. The focus is on answers that are both correct and well-supported.
- 7. Attribution and Logging: Show the sources alongside the answer so readers can verify claims. Log the entire process including queries, retrieved documents and versions, for debugging, evaluation, and compliance. This makes the system transparent and auditable.

A straightforward RAG system follows the seven-step pipeline outlined above. Listing [12.1](#page-5-0) shows Python-style pseudo-code for this process. The function takes a user question, rewrites it into one or more search queries, retrieves and re-ranks candidate passages, assembles a context window and finally generates an answer while tracking which passages were used. We will go into more details below.

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

*Listing 12.1: High-level flow of a basic RAG setup.*
# 12.4 Adaptive Retrieval Methods

Traditional RAG systems retrieve a fixed number of passages regardless of whether any retrieval is actually needed. This *always–on* strategy simplifies engineering but can hurt both efficiency and accuracy. Recent research explores *adaptive* or *gated* retrieval strategies that teach the model to decide when to search and how much evidence to consult. These methods align with the self-reflection ideas introduced in Section 6.2.4: rather than blindly ingest retrieved text, the language model introspects about its own confidence and triggers retrieval only when it recognizes a knowledge gap.

Concretely, [Asai et al.](#page-17-0) [\[2023\]](#page-17-0) propose Self-RAG, which trains a single model to generate answers, request retrieval, and critique its own outputs using special *reflection tokens*. When the model's draft answer is uncertain, it emits a <RETRIEVE> token to call the search tool and incorporates the retrieved passages in the next step. Experiments show that this learned gating improves factuality over static RAG while avoiding unnecessary calls.

A more lightweight approach is Self-Ask [\[Press et al.,](#page-19-0) [2022\]](#page-19-0), where the model decomposes a complex question into simpler sub-questions, issues them to a search tool, and uses the answers to compose a final response. Self-Ask can be combined with planning agents, enabling models to interleave reasoning, tool calls, and retrieval. Graph Toolformer [\[Zhang,](#page-20-0) [2023\]](#page-20-0) extends this idea by training a policy to insert search queries into a program graph. At inference time the LLM dynamically decides whether to retrieve from a text index, a knowledge graph, or a database, based on the current state of reasoning.

Beyond these specific examples, adaptive retrieval can be implemented in several ways:

- Confidence-based Gating: The model estimates its uncertainty (e.g., from entropy or calibration metrics) and triggers retrieval when confidence falls below a threshold.
- **Iterative retrieval.** Instead of fetching all evidence at once, the system retrieves a few passages, reasons over them, and decides whether another search round is needed.

{6}------------------------------------------------

<span id="page-6-0"></span>This approach is useful for multi-hop questions.

- Query Planning: An agent decomposes the user query into sub-tasks, dispatches them to different tools (search, database, calculator), and integrates the results. Retrievers are called only when the plan requires external knowledge.
- Learning to Retrieve: Reinforcement learning or supervised finetuning can teach the model to emit <RETRIEVE> tokens at appropriate times. Self-RAG is one instantiation; other methods explore training retrieval policies end-to-end with the generator.

Adaptive retrieval reduces latency and cost by avoiding unnecessary searches, and it often improves answer quality by focusing attention on missing knowledge. Designing effective gating policies, however, requires careful evaluation (see Section [12.7\)](#page-14-0) and may demand additional supervision during training.

#### 12.4.1 Adaptive Retrieval via Self-RAG and Self-Ask

The Self-RAG framework of [Asai et al.](#page-17-0) [\[2023\]](#page-17-0) teaches the model to request retrieval on demand by emitting a special <RETRIEVE> token during generation. The pseudocode in Listing 12.2 sketches how such a loop operates. The model maintains a growing context and alternates between generating output tokens and pulling in new evidence when needed.

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

*Listing 12.2: High-level pseudocode demonstrating Self-RAG.*

It should be noted that for Self-RAG to work, the foundation model should be trained or fine-tuned to produce the <RETRIEVE> token. Self-Ask [\[Press et al.,](#page-19-0) [2022\]](#page-19-0) on the other hand is not dependent on the language model. Self-Ask decomposes a complex question into simpler sub-questions, retrieves evidence for each and synthesizes the final answer. The following high-level pseudo-code shows one way to implement this strategy:

{7}------------------------------------------------

```python
def self_ask(q, index):
    sub_questions = decompose(q)
    # Example: "When was Einstein born?" and "What is the Nobel Prize?"
    intermediate_answers = []
    for sq in sub_questions:
        docs = retrieve(index, rewrite_question(sq), k=5)
        answer = summarize(docs)
        intermediate_answers.append(answer)
    final_answer = compose(intermediate_answers)
    return final_answer
```

*Listing 12.3: High-level pseudocode demonstrating Self-Ask.*

#### 12.4.2 A Simple Dense Retrieval Example in Python

Dense retrieval libraries such as the sentence transformers toolkit make it easy to index a small corpus and perform semantic search. Listing 12.4 illustrates how to embed a corpus of sentences, compute similarities to a query and display the top results.

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

*Listing 12.4: A simple dense retrieval example in Python.*

#### 12.4.3 Query Rewriting and Expansion

Effective retrieval begins by transforming a question into a form the index can match. An LLM can paraphrase it, expand acronyms, add synonyms, or generate a hypothetical document to use as a search target. 

{8}------------------------------------------------

<span id="page-8-0"></span>With HyDE, the generated text is embedded and used as a pseudo-query.<sup>1</sup> Listing 12.5 shows a function that asks an LLM to produce query variants and combines them into a set of search queries.

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

*Listing 12.5: A simple example of query rewriting.*

In practice, the rewrite question function might call an API such as OpenAI's gpt-5 to generate paraphrases and hypothetical answers, then use a thesaurus or entity-linking tool to add aliases. Query expansion improves recall but can also introduce noise.

# 12.5 Beyond Text: Graphs, Code, Tabular Data, and Multimodal Retrieval

While most RAG systems operate over unstructured text, many agentic applications require reasoning over structured data, source code, or images. Extending retrieval beyond plain text opens up new opportunities but also introduces modeling challenges.

**Graph-RAG.** A knowledge graph encodes entities as nodes and relations as edges. Building a lightweight graph over the corpus allows the retriever to follow links (e.g., *works for*, *located in*) rather than relying solely on bag-of-words similarity. The GraphRAG survey formalizes a three-step workflow: graph-based indexing, graph-guided retrieval, and graph-enhanced generation [\[Peng et al.,](#page-19-0) [2024\]](#page-19-0). Indexing extracts entities and relations from text, canonicalizes them and builds neighborhood structures. Retrieval then expands the query through the graph to find related nodes and associated passages.

<sup>1</sup>HyDE stands for Hypothetical Document Embeddings and generates a plausible answer which is then embedded and used to retrieve supporting passages [\[Gao et al.,](#page-18-0) [2023a\]](#page-18-0).

{9}------------------------------------------------

<span id="page-9-0"></span>The graph supports global reasoning and disambiguation. The generation model can use its structure to organize facts into a coherent narrative and avoid contradictions.

#### 12.5.1 Graph-based retrieval example

GraphRAG systems represent knowledge as a graph of entities and relations and use graph traversals to retrieve relevant nodes [\[Peng et al.,](#page-19-0) [2024\]](#page-19-0). A lightweight implementation can be built with the networkx library. In the example in Listing 12.6 we construct a small knowledge graph and implement two simple retrieval primitives: (1) get the immediate neighborhood of an entity, and (2) find the shortest path between two entities. These operations can support multi-hop reasoning.

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

*Listing 12.6: Graph-based retrieval primitives in Python.*

In a full GraphRAG system the graph nodes would store pointers to the original text passages, and the retrieval procedure would return both the sequence of nodes and the associated passages to the LLM. Graph-guided retrieval can reduce hallucinations by grounding reasoning in explicit edges and by encouraging consistency between related facts.

{10}------------------------------------------------

<span id="page-10-0"></span>**Code-RAG.** Source code has a rich syntax-structure. The semantics of source code differs from natural language. For code retrieval to be effective, it should index at the function or module level, including signatures, docstrings, abstract syntax trees (ASTs) and dependency graphs. Models such as CodeBERT [\[Feng et al.,](#page-17-0) [2020\]](#page-17-0) learn joint embeddings for code and natural language; they support tasks like natural-language code search, code summarization, and documentation generation. In Code-RAG, the retriever can fetch relevant functions, APIs and test cases, and the re-ranker can use code-aware cross-encoders or graph neural networks to prioritize exact matches. Retrieved code snippets, comments and dependency metadata are then provided to the LLM, which can answer questions, suggest fixes, or generate new code while citing the source files. Evaluation should consider correctness of the code, type safety and adherence to dependencies.

### 12.5.2 Code retrieval example using ASTs and embeddings

Retrieving source code requires understanding its syntax and semantics. An easy approach is to embed code snippets and natural language queries with a pretrained model such as CodeBERT [\[Feng et al.,](#page-17-0) [2020\]](#page-17-0). Another complementary approach is to parse the code into an abstract syntax tree (AST) and search its structure. The example in Listing [12.7](#page-11-0) demonstrates both: we embed functions with CodeBERT for semantic search and use Python's ast module to extract function definitions and docstrings for indexing.

This example indexes the functions in an imported module and performs a cosine similarity search between the query and the encoded source code. In practice one might combine code embeddings with lexical features (identifiers, comments) and incorporate AST-based matching to handle structural queries (e.g., find all functions that call a specific API). Code-aware retrieval is essential for agentic systems that need to read, execute or modify existing code during tool use.

### 12.5.3 Multi-modal and heterogeneous sources

Tabular and other heterogeneous documents Existing RAG approaches face critical limitations when applied to heterogeneous documents, comprising both textual and tabular components [\[Yu et al.,](#page-20-0) [2025\]](#page-20-0). A primary challenge is *Structural Information Loss*. When tabular data is linearized or chunked for text-based RAG, its inherent relational structure, the interdependence of rows and columns, is compromised. This loss of structure impedes the LLM's ability to perform accurate multi-hop heterogeneous reasoning. This indicates that merely converting non-textual data to text is insufficient. To address these challenges, hybrid frameworks such as *TableRAG* [\[Yu et al.,](#page-20-0) [2025\]](#page-20-0) have been proposed, that unify textual understanding and complex manipulations over tabular data.

TableRAG employs SQL programming and execution to treat table-related queries as indivisible reasoning units, thereby enhancing both computational efficiency and reasoning fidelity. Similarly, *MixRAG* [\[Zhang et al.,](#page-20-0) [2025\]](#page-20-0) has been shown to significantly improve retrieval performance for heterogeneous document RAG, which demands joint retrieval and reasoning across textual and hierarchical tabular data.

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

*Listing 12.7: Code-based retrieval in Python.*

The success of these approaches demonstrates that a purely neural (embeddingbased) RAG approach is insufficient for structured data. The integration of symbolic reasoning (e.g., SQL) alongside neural retrieval is a critical design pattern for heterogeneous RAG, reflecting a broader trend in Agentic LLMs towards neuro-symbolic integration for enhanced data understanding and fidelity.

**Multimodal RAG.** Many knowledge bases contain tables, figures, diagrams, audio and video. A straightforward approach is to extract any textual content using OCR and treat the remainder as plain text. A richer approach stores embeddings for images or other modalities (e.g., CLIP or ViT representations) and pairs them with captions and context. Queries can be encoded jointly across modalities, allowing the system to 

{12}------------------------------------------------

retrieve relevant figures or equations that accompany the text. For example, a research agent might retrieve a plot and its caption to answer a data-analysis question, or an educational agent might fetch diagrams from a textbook.

Multi-modal RAG remains an active research area. Challenges include cross-modal alignment and efficient indexing of high-dimensional embeddings. Furthermore, non-textual evidence should be properly grounded in the generated answer.

#### 12.5.4 Multi-modal retrieval example

Agentic systems sometimes need to retrieve images, diagrams or audio clips in addition to text. Multi-modal models such as CLIP or Contrastive Language-Audio Pretraining (CLAP) embed both natural language and other modalities into a shared space. The high-level pseudo-code in Listing 12.8 outlines how to use CLIP to find images relevant to a textual query:

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

*Listing 12.8: Using CLIP to find relevant images in Python.*

The multimodal retrieval returns the filenames of the most similar images. In a RAG setting the retrieved images could be fed through a vision encoder to extract captions or objects, which are then incorporated into the context for the language model. Multimodal agents can thus ground their outputs in both textual and visual evidence.

{13}------------------------------------------------

# <span id="page-13-0"></span>12.6 Agentic RAG: Beyond Simple Retrieval

While traditional RAG systems augment LLMs with external knowledge, they are often passive, relying on a static, pre-defined pipeline. A significant and recent advancement is the concept of *Agentic RAG*, which integrates autonomous AI agents directly into the retrieval-augmented generation pipeline [\[Singh et al.,](#page-19-0) [2025\]](#page-19-0). This paradigm shift moves RAG from a mere tool to an active, intelligent, and agent-driven process. Agentic RAG systems are characterized by their ability to dynamically manage retrieval strategies, iteratively refine contextual understanding, and adapt workflows to meet complex task requirements.

Agentic RAG represents a more sophisticated form of a retrieval loop, where the LLM agent itself is responsible for orchestrating the retrieval process. This differs from simpler RAG methods by embracing a continuous feedback loop that is fundamental to advanced agentic systems. Rather than a single-shot query transformation, the agent can actively refine its queries based on intermediate results and ongoing interaction. This approach addresses the limitations of conventional RAG systems, which are often constrained by static workflows and lack the adaptability required for multistep reasoning and complex task management [\[Singh et al.,](#page-19-0) [2025\]](#page-19-0).

#### 12.6.1 Key Components of Agentic RAG

Agentic RAG frameworks typically incorporate several key components that enable their dynamic behavior:

- Dynamic Query Generation: The agent formulates queries based on its current understanding of the task and the retrieved context, allowing for more precise and effective searches. This can include query expansion, decomposition, or rewriting in response to initial retrieval failures.
- Iterative Refinement: The agent evaluates the quality of the retrieved information and, if necessary, initiates a new retrieval cycle with a refined query. This iterative process ensures that the final response is grounded in the most relevant and accurate information.
- Autonomous Tool Use: The agent can select and use a variety of tools, including different retrieval models (e.g., dense vs. sparse), ranking algorithms, and external APIs, to optimize its retrieval strategy. See Chapter 11 for additional information on how an agent can use tools.
- Self-Correction and Reflection: The agent can assess its own output for accuracy and completeness. If it detects a hallucination or an unsupported claim, it can trigger a new retrieval and reasoning process to correct the error.

The rise of Agentic RAG indicates a shift in the role of RAG in LLM systems. Retrieval shifts from a passive data lookup to an active, intelligent capability that is integral to the agent's cognitive functions, allowing for more sophisticated, adaptive interactions with external knowledge sources. This shift is crucial for developing robust and reliable LLM agents capable of handling complex, real-world tasks.

{14}------------------------------------------------

#### <span id="page-14-0"></span>12.6.2 Real-Time and Streaming RAG

While traditional RAG systems are effective for static data, they are often not designed for applications that require continuous, real-time information. This section looks into the architectural considerations and challenges of RAG systems that operate on continuously updating, high-throughput data streams. Examples of such streams are found in news feeds, social media, or financial markets. For streaming data, simply querying an updated static index no longer suffices. Retrieval now involves dynamic index management and continuous knowledge assimilation.

In traditional RAG systems, designed for static data, preprocessing is slow, and unsuitable for real-time analysis [\[Zhu,](#page-20-0) [2025,](#page-20-0) [Sankaradas et al.,](#page-19-0) [2025\]](#page-19-0). How should we maintain data freshness and semantic coverage under memory budgets and high data ingestion rates? Methods such as *StreamingRAG* [\[Sankaradas et al.,](#page-19-0) [2025\]](#page-19-0) try to overcome these issues with knowledge graphs that evolve dynamically in real-time.

LLM agents for dynamic domains such as medicine, logistics and finance, must have access to real-time, continuously updated information. Static RAG would yield obsolete or inaccurate information in such fast-changing environments. In many agentic applications real-time RAG is necessary for advanced, timely, and reliable information.

# 12.7 Evaluation

Evaluating a retrieval-augmented system is harder than evaluating a standalone language model because both the retriever and the generator affect the final answer. A sound evaluation should assess each component separately and examine their combined behavior.

**Retrieval metrics.** For a query, compare the retrieved passages with a set of gold-standard documents. Common metrics include:

1. **Recall@k** (Equation [12.7](#page-15-0)) and **Precision@k** (Equation [12.8](#page-15-0)), which measure retrieval coverage and precision among the top \(k\) results.
2. **Mean Reciprocal Rank (MRR)** (Equation [12.9](#page-15-0)) and mean average precision, which measure ranking quality.
3. **Normalized Discounted Cumulative Gain (nDCG)** (Equation [12.10](#page-15-0)), which gives more weight to highly ranked relevant results.

BEIR contains 18 heterogeneous datasets covering question answering, fact checking, and entity retrieval. Its results show that late-interaction and re-ranking models can improve zero-shot performance, at additional cost [Thakur et al., 2021](#page-19-0). KILT evaluates knowledge-intensive tasks against a unified Wikipedia snapshot and measures both task performance and provenance, encouraging systems to cite the passages they use [Petroni et al., 2021](#page-19-0).

{15}------------------------------------------------

<span id="page-15-0"></span>

\[\operatorname{Recall}@k=\frac{|\{\text{relevant documents}\}\cap\{\text{top-}k\text{ retrieved}\}|}{|\{\text{relevant documents}\}|}\tag{12.7}\]

\[\operatorname{Precision}@k=\frac{|\{\text{relevant documents}\}\cap\{\text{top-}k\text{ retrieved}\}|}{k}\tag{12.8}\]

\[\operatorname{MRR}=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\operatorname{rank}_i}\tag{12.9}\]

Here, \(\operatorname{rank}_i\) is the rank of the first relevant document for query \(i\).

\[\operatorname{nDCG}@k=\frac{\operatorname{DCG}@k}{\operatorname{IDCG}@k}\tag{12.10}\]

where

\[\operatorname{DCG}@k=\sum_{i=1}^{k}\frac{2^{\operatorname{rel}_i}-1}{\log_2(i+1)}\]

and \(\operatorname{rel}_i\) is the graded relevance of the document at rank \(i\).

**Generation metrics.** After assembling the context, assess the quality of the generated answer. Automatic metrics such as BLEU, ROUGE, and BERTScore are often of limited use because there may be no single ground-truth answer. More suitable measures include *faithfulness* and *helpfulness*. Faithfulness can be measured with entailment models: for a generated output \(y\) and retrieved context \(x\), check whether each claim is supported by the cited passages, for example with natural-language inference or citation checking.

\[\operatorname{Faithfulness}=\frac{\text{outputs entailed by context}}{\text{total outputs}}\tag{12.11}\]

Helpfulness (or utility) measures whether the answer meets the user's information need. This often requires human evaluation or a specialized LLM evaluator. Latency and cost also matter in interactive settings. Helpfulness can be based on human or model scores:

\[\operatorname{Helpfulness}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{score}(y_i,\operatorname{query}_i)\tag{12.12}\]

Here, the score may be a Likert-scale rating or a pairwise preference judgment.

**Joint evaluation and newer metrics.** Retrieval and generation should be evaluated separately. A system may retrieve irrelevant documents yet guess the right answer, or retrieve ideal evidence but fail to use it. Recent work therefore proposes unified evaluation frameworks. The Auepora process recommends reporting separate retrieval and generation scores and combining them into an overall measure [Yu et al., 2024](#page-20-0). RAGAS and similar toolkits provide automated

{16}------------------------------------------------

pipelines that use small models to estimate context recall, answer correctness, and citation accuracy. When evaluating adaptive methods (Section [12.4](#page-5-0)), also measure the retrieval trigger: how often does the model search when it should, and how often does it skip retrieval unnecessarily?

**Attribution and freshness.** In many applications, models must cite sources and avoid outdated information. Attribution tests align claims with evidence using natural-language inference and check consistency between URLs and answers. Freshness metrics measure whether answers reflect the latest valid sources and can also quantify version drift. Attribution and freshness are especially important in time-sensitive domains such as finance and medicine.

Careful evaluation guides system design. High retrieval recall with low faithfulness may indicate a need for better re-ranking. High faithfulness with low helpfulness may point to insufficient context coverage. For error analysis and auditing, log queries, retrieved passages, and model outputs transparently.
# 12.8 Concluding Remarks

Retrieval is essential for agentic LLM systems that need current information. Language models are static snapshots of knowledge; RAG helps keep their answers current, transparent, and grounded. Effective RAG requires careful work across indexing, query planning, retrieval, re-ranking, context assembly, and grounded generation, as well as adaptive methods that let the model decide when and how to search. Retrieval can also extend beyond text to graphs, code, and multimodal data. Rigorous evaluation is necessary for building trustworthy systems.

**Operational considerations.** In production, a retriever is a continuously running service, not a one-off script. Crawlers, webhooks, and document pipelines should ingest new data continuously. Track versions and deprecations, and maintain timelines for entities so newer facts can be preferred. Apply access controls at retrieval time and redact personally identifiable information during indexing. To manage latency and cost, cache embeddings and top-\(k\) results, reuse retrieval results across conversational turns, compress context with extractive summaries, tune \(k\) and chunk sizes for each domain, and re-rank only the leading candidates.

**Practical suggestions.** Start with a hybrid system that combines a strong dense retriever with BM25 or SPLADE. Re-rank the top 100 results with a cross-encoder, then send six to twelve passages to the LLM. Chunk the corpus thoughtfully; for example, use 300–500-token windows with 10–15% overlap, preserve headings and tables, and attach timestamps or other metadata. Design queries carefully: rewrite complex questions,

{17}------------------------------------------------

<span id="page-17-0"></span>

decompose multi-hop questions, expand entity aliases, and apply domain filters. Require citations for transparency. If evidence is weak, ask a follow-up question or abstain.

Use evidence-based system design and measure what you build. Track retrieval recall, groundedness, freshness, and cost. Experiment with re-rankers, and log system traces for debugging.

#### Try it yourself!

To experiment with retrieval-augmented generation, try the [RAG tutorial in Google Colab](https://colab.research.google.com/github/Agentic-LLMs/Agentic-AI-tutorials/blob/main/rag_tutorial.ipynb).

**Integration into agentic AI.** Retrieval supports planning, self-reflection, and action in the world (see Chapter 13). Agentic pipelines can call a retriever dynamically based on the model’s confidence or plan, as in Self-RAG and Graph Toolformer. They can query different sources, including text, graphs, code, and images, drawing on the strengths of each. The boundary between a model and its *memory* therefore becomes less distinct: RAG becomes part of the agent’s cognitive loop.

**Further reading.** For deeper coverage, consult surveys on RAG, evaluation, and graph-based methods [Gao et al., 2023b](#page-18-0); [Huang et al., 2024](#page-18-0); [Yu et al., 2024](#page-20-0); [Sharma et al., 2025](#page-19-0); [Peng et al., 2024](#page-19-0). Foundational work includes REALM, DPR, and the original RAG formulation [Guu et al., 2020](#page-18-0); [Karpukhin et al., 2020](#page-18-0); [Lewis et al., 2020](#page-18-0). Research on GraphRAG, evaluation, and agentic RAG is developing quickly; follow new papers and benchmarks to stay current.
# Bibliography

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















