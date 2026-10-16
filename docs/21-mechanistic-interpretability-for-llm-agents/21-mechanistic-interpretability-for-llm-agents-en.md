<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 403-422; Marker split PDF pages 1-20. -->



{0}------------------------------------------------

## <span id="page-0-0"></span>Chapter 21

# \*Mechanistic Interpretability for LLM Agents

### What You Will Learn

In this chapter you will learn how to look inside the black box of large language models using tools from explainable artificial intelligence and mechanistic interpretability. Specifically, you will be able to:

- Distinguish between the four main interpretability paradigms: *behavioral*, *attributional*, *concept-based*, and *mechanistic interpretability*.
- Explain how hallucination relates to *faithfulness*, and describe how mechanistic methods can identify the internal circuits responsible for model failures.
- Describe the *residual stream* formulation of the transformer and explain how information is iteratively refined across layers.
- Apply the *logit lens* and tuned lens to visualize how a model's predictions evolve through its layers.
- Explain *polysemanticity and superposition*, and understand why individual neurons are poor units of interpretation.
- Use *sparse autoencoders* for dictionary learning to decompose model activations into monosemantic, human-interpretable features.
- Perform *circuit analysis and activation patching* to identify and causally verify the internal algorithms responsible for specific model behaviors.

## 21.1 Opening the LLM

How do LLMs work internally? Opening the black box of large neural networks is an important research goal. We want to understand how billions of neurons encode representations, how models reason, and how they reach conclusions. Explainable artificial intelligence (XAI) offers many methods for investigating these questions [\[Minh et al.,](#page-16-0) [2022,](#page-16-0) [Rios et al.,](#page-17-0) [2020,](#page-17-0) [Van Stein et al.,](#page-18-0) [2022,](#page-18-0) [Selvaraju et al.,](#page-17-0) [2017,](#page-17-0) [Ali et al.,](#page-14-0) [2022](#page-14-0).

{1}------------------------------------------------

<span id="page-1-0"></span>

Static XAI methods developed in classical machine learning have been successful [\[Molnar,](#page-16-0) [2020\]](#page-16-0). For example, feature maps can show how input images map to output classes [\[Ren et al.,](#page-17-0) [2016,](#page-17-0) [Kohonen,](#page-16-0) [1982,](#page-16-0) [Redmon et al.,](#page-17-0) [2016\]](#page-17-0). Counterfactual analysis [\[Karimi et al.,](#page-16-0) [2020,](#page-16-0) [Huang et al.,](#page-15-0) [2024\]](#page-15-0), LIME [\[Ribeiro et al.,](#page-17-0) [2016\]](#page-17-0), and SHAP [\[Lundberg and Lee,](#page-16-0) [2017\]](#page-16-0) help explain how inputs map to outputs in structured-data tasks. These methods belong to the *behavioral* or *attributional* interpretability paradigms, which treat the model as a black box (Figure 21.1). Attribution methods such as SHAP and LIME estimate how much each input feature contributes to the output. Behavioral methods, such as counterfactual analysis, reveal which features matter, though not necessarily by how much.

![](assets/_page_1_Diagram_3.jpeg)

Figure 21.1: Taxonomy of *Interpretability Paradigms*. Behavioral interpretability treats the model as a black box, mapping input-output relations ($f(x)=y$). Attributional methods quantify the influence of specific input features on the final output using gradients or saliency. Concept-based interpretability employs probes to identify high-level, human-understandable representations within hidden layers. Mechanistic interpretability aims to reverse-engineer the specific causal circuits and internal algorithms that govern model behavior.

Moving beyond surface-level input–output correlations, *concept-based* interpretability asks whether specific, human-understandable information is encoded in a model's internal states. Researchers commonly investigate this with *linear probing*: they train a simple supervised classifier on frozen hidden activations to detect semantic concepts such as parts of speech, sentiment, or factual entities [\[Alain and Bengio,](#page-14-0) [2016,](#page-14-0) [Belinkov,](#page-14-0) [2022\]](#page-14-0). Attribution methods highlight *which* input features influence an output; probing tests *what* information is represented at different layers (Figure 21.1). However, probing has a known *causal gap*: finding that a concept is

{2}------------------------------------------------

<span id="page-2-0"></span>represented in a layer does not necessarily mean the model uses that information for its final prediction.

More recently, dynamic methods have been developed to uncover patterns under the hood of large transformer models. These belong to the sub-domain of mechanistic interpretability (MI). The goal of MI is to uncover the mechanisms by which the model dynamically comes to conclusions [\[Nanda et al.,](#page-17-0) [2023,](#page-17-0) [Bereska and Gavves,](#page-14-0) [2024,](#page-14-0) [Ferrando et al.,](#page-15-0) [2024,](#page-15-0) [Rai et al.,](#page-17-0) [2024\]](#page-17-0). In other words, MI seeks a low-level, causal explanation of a model's behavior in terms of its internal activations and connections, analogous to decompiling a program's binary into readable code. Methods such as sparse autoencoders [\[Cunningham et al.,](#page-15-0) [2023,](#page-15-0) [Makelov et al.,](#page-16-0) [2024\]](#page-16-0), neural lenses [\[Black et al.,](#page-14-0) [2022\]](#page-14-0) and circuit discovery [\[Conmy et al.,](#page-15-0) [2023\]](#page-15-0) are being used to successfully enhance insight into how LLMs work, for example, in chess [\[Davis and](#page-15-0) [Sukthankar,](#page-15-0) [2024\]](#page-15-0).

XAI and MI in particular are active areas of research that will allow us to better understand how LLMs reason and come to conclusions [\[Sharkey et al.,](#page-17-0) [2025\]](#page-17-0). Once a better understanding is reached, LLMs can be improved accordingly, for example, to reduce hallucinations.

This chapter will dive deeper into the many facets of MI.

## 21.2 Hallucination and Faithfulness in Model Reasoning

A significant barrier to the deployment of LLMs in high-stakes environments is the phenomenon of *hallucination*: the generation of syntactically plausible but factually incorrect content. From an interpretability perspective, the challenge lies in distinguishing whether a model's output is the result of a robust *internal computation* or a *post-hoc justification*. This distinction is formalized as the study of *faithfulness*, the degree to which a model's stated reasoning (e.g., in chain of thought prompting) is causally responsible for its final output.

Mechanistic interpretability (MI) provides the tools necessary to move beyond behavioral metrics of accuracy and toward a causal understanding of model failure modes. For instance, research using sparse autoencoders (SAEs) has identified specific *honesty* or *knowledge-retrieval* circuits [\[Lindsey et al.,](#page-16-0) [2025\]](#page-16-0). In these studies, researchers identified latent features that activate when a model encounters unfamiliar entities; a failure in the inhibitory logic of these circuits, where the model proceeds to generate tokens despite low activation in the relevant *knowledge* feature, directly results in fabricated answers.

Furthermore, MI allows us to investigate the *causal necessity* of intermediate reasoning steps. While a model may produce a convincing chain of thought (CoT), it remains possible that the model has already calculated the final answer via a separate internal shortcut and is merely generating the CoT as a *stochastic parrot*.

To test this, [Chen et al.](#page-15-0) [\[2026\]](#page-15-0) utilized *causal mediation analysis* combined with SAEs to intervene on the latent thought process during arithmetic tasks. By injecting the latent activations from a CoT-enabled run into a standard zero-shot run, they 

{3}------------------------------------------------

<span id="page-3-0"></span>demonstrated that in sufficiently large models, these intermediate thoughts significantly boosted solution confidence. This provides empirical evidence that for larger architectures, the CoT is not merely cosmetic but represents a *faithful* causal link in the model's reasoning chain.

Conversely, the absence of such causal links in smaller models suggests that *faithful reasoning* may be an emergent property of scale, appearing only when the model possesses sufficient parameters to represent complex, multi-step algorithms. Developing robust metrics to quantify this faithfulness is a critical frontier in AI safety; by peeking into the model's activations, we can potentially develop hallucination detectors that trigger when a model's internal confidence features do not align with its external verbosity.

## 21.3 The Residual Stream and Iterative Refinement

Before exploring specific MI methods it is necessary to establish a formal mental model of the transformer architecture. Unlike classical feed-forward networks where each layer replaces the representation of the previous one, modern transformers operate on a *residual stream* [\[Elhage et al.,](#page-15-0) [2021\]](#page-15-0).

Formally, let $h_\ell\in\mathbb{R}^{d_{\mathrm{model}}}$ be the state of the hidden representation after layer $\ell$. The Transformer architecture is defined by the recurrence:

$$
h_\ell = h_{\ell-1} + F_\ell(h_{\ell-1}) \tag{21.1}
$$

where F<sup>ℓ</sup> represents the computation performed by the <sup>ℓ</sup>-th layer (comprising multihead attention and feed-forward blocks).

This structure implies that the final representation $h_L$ is a cumulative sum of the initial embedding $x_{\mathrm{embed}}$ and the contributions ($\Delta h$) from every subsequent layer:

$$
h_L = x_{\mathrm{embed}} + \sum_{\ell=1}^{L}\Delta h_\ell \tag{21.2}
$$

In this view, the residual stream acts as a high-dimensional *communication channel*. Individual layers do not transform the entire state; rather, they read information from the stream, perform a local computation and write their results back into the stream via vector addition. This additive nature ensures that information can persist across many layers, and more importantly, it justifies the linear decomposition of the model's final output.

## 21.4 Polysemanticity and Superposition

A primary challenge in reverse-engineering neural networks is *polysemanticity* (Figure [21.2](#page-4-0)): individual neurons often activate for multiple, semantically unrelated concepts. This is closely related to *superposition*, in which a model represents $n$ features in $m$ dimensions, with $n>m$.

{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Figure_2.jpeg)

Figure 21.2: The concept of polysemanticity in LLMs and other transformer based models

Under the *linear representation hypothesis*, models exploit the high-dimensional geometry of activation space to store features as almost-orthogonal vectors. While this allows for efficient information compression, it results in a basis mismatch where the fixed neuron basis does not align with the true feature basis.

A formalization was provided by [Elhage et al.](#page-15-0) [\[2022\]](#page-15-0), who demonstrated using toy models that superposition is a mathematically principled strategy for models to maximize information density when features are sparse. They identified *phase transitions* in feature encoding: as the sparsity of a concept increases, the model can pack more features into the same number of dimensions, albeit at the cost of interference or noise between those features.

To resolve this entanglement, researchers have turned to *dictionary learning*, most notably via *sparse autoencoders* (SAEs). The objective is to learn a linear transformation that maps the tangled neuron activations into a much higher-dimensional, sparse latent space. [Bricken et al.](#page-14-0) [\[2023\]](#page-14-0) successfully applied this to small-scale transform

{5}------------------------------------------------

<span id="page-5-0"></span>ers, demonstrating that the resulting SAE features are significantly more monosemantic and human-interpretable than the original neurons. This suggested that models are not inherently uninterpretable, but rather that we have been looking at them through the wrong coordinate system.

Subsequent work has explored whether architectural modifications can mitigate the emergence of polysemanticity. For instance, [Liu et al.](#page-16-0) [\[2023\]](#page-16-0) and others have investigated how different activation functions or regularization penalties (e.g., $L_1$ sparsity) can pressure the model into adopting a more aligned, interpretable internal basis. Despite these advances, polysemanticity remains a pervasive challenge in state-of-the-art models. The current consensus in MI is that interpreting a model requires moving beyond the "neuron-as-a-unit" paradigm and instead identifying the underlying *sparse feature basis* in which the model\'s causal logic is actually computed.

## 21.5 The Logit Lens

A fundamental intuition in mechanistic interpretability is that a transformer model does not compute its output in a single leap at the final layer. Instead, it iteratively refines a representation within the *residual stream*. The *logit lens*<sup>1</sup> provides a window into this process by decoding the hidden states of intermediate layers directly into the output vocabulary space.

### 21.5.1 Formal Mechanism

Consider a transformer with $L$ layers. Let $x_0$ be the initial embedding of a token, and $h_\ell$ be the state of the residual stream after layer $\ell\in\{1,\ldots,L\}$. In a standard forward pass, the final hidden state $h_L$ is converted into logits $y$ via the layer norm (LN) and the unembedding matrix $W_U\in\mathbb{R}^{d_{\mathrm{model}}\times|V|}$:

$$
y=\operatorname{Softmax}(\operatorname{LN}(h_L)W_U) \tag{21.3}
$$

The logit lens generalizes this by applying the *same* unembedding transformation to intermediate states $h_\ell$:

$$
\hat{y}_\ell=\operatorname{Softmax}(\operatorname{LN}(h_\ell)W_U) \tag{21.4}
$$

where $\hat{y}_\ell$ represents the model\'s tentative probability distribution at layer $\ell$. By visualizing these distributions across all layers, researchers can observe the *evolution of belief* as the model moves from vague semantic clusters to specific token predictions.

**Direct Logit Attribution.** While the logit lens provides an aggregate view of the residual stream $h_\ell$, we often wish to isolate the contribution of a *single* layer or head. Because the final output is a linear function of the cumulative sum of layer outputs,

<sup>1</sup>First introduced in the following blog post: [https://www.lesswrong.com/posts/](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) [AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). An updated implementation is provided by [\[Wang,](#page-18-0) [2025\]](#page-18-0).

{6}------------------------------------------------

<span id="page-6-0"></span>![](assets/_page_6_Diagram_2.jpeg)

Figure 21.3: Visualization of the *logit lens*. The residual stream is intercepted at an intermediate layer (e.g., Layer 2) and decoded using the model\'s final unembedding matrix ($W_U$). This reveals the current belief of the model at that specific step, though it may be noisy in early layers.

we can compute the *direct logit attribution* (DLA) of layer <sup>ℓ</sup> by projecting its specific output Δℎ<sup>ℓ</sup> directly into the vocabulary space:

$$
\operatorname{DLA}(\ell)=\operatorname{LN}(\Delta h_\ell)W_U \tag{21.5}
$$

DLA allows researchers to determine which specific components are writing the correct answer into the stream, rather than just identifying where the answer becomes visible. In the study of factual recall, DLA often reveals that a single knowledge-producing layer provides the vast majority of the probability boost for the correct token.

### 21.5.2 From the Logit Lens to the Tuned Lens

While the initial logit lens is highly informative for the final third of a network, it often yields gibberish or high-entropy distributions in early layers. This occurs because the residual stream at layer 2, for example, is not yet formatted for the output distribution; it may be storing low-level positional information that is mathematically orthogonal to the unembedding matrix $W_U$.

To bridge this gap, [Belrose et al.](#page-14-0) [\[2023\]](#page-14-0) introduced the *tuned lens*. Instead of using the static $W_U$, the tuned lens trains a layer-specific affine transformation $\phi_\ell$ to map $h_\ell$ to the logit space:

$$
y_{\mathrm{tuned},\ell}=\operatorname{Softmax}(\phi_\ell(h_\ell)) \tag{21.6}
$$

{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Diagram_1.jpeg)

Figure 21.4: Visualization of the *Tuned Lens*. Unlike the naive lens, this method uses a layer-specific, *learned* transformation ($\phi_\ell$) to decode the residual stream. This transformation is trained to align the intermediate representation with the final output distribution, providing a clearer picture of information emergence in early layers.

The transformation $\phi_\ell$ is trained to minimize the KL-divergence between $y_{\mathrm{tuned},\ell}$ and the model\'s actual final output $y$. This effectively translates the internal language of early layers into the output vocabulary, revealing that factual information often emerges much earlier in the network than the naive logit lens would suggest.

### 21.5.3 Applications and Interpretations

Lenses serve as a critical diagnostic tool for identifying *early exit* candidates and locating specific knowledge neurons. For instance, if the logit for a specific fact (e.g., Paris) jumps from the 100th to the 1st rank at layer 14, it suggests that the attention heads or MLP sub-layers in layer 14 are responsible for the factual retrieval.

Variants such as the *attention lens* [\[Sakarvadia et al.,](#page-17-0) [2023\]](#page-17-0) apply similar logic to the outputs of individual attention heads, allowing us to see what each head contributes to the prediction in isolation. Collectively, these tools shift our view of the transformer from a sequential pipeline to a *cumulative iterative refiner* of latent information.

## 21.6 Circuit Analysis

A central objective of MI is the identification of *circuits*: minimal subgraphs of a model's computation graph that are both necessary and sufficient to implement a specific high-level function. Unlike behavioral analysis, circuit discovery seeks to explain *how* information is transformed step-by-step, from input embeddings to the final logit.

{8}------------------------------------------------

### <span id="page-8-0"></span>21.6.1 The Prototypical Circuit: Induction Heads

The discovery of the *induction head* represents a landmark in transformer interpretability [\[Olsson et al.,](#page-17-0) [2022\]](#page-17-0). Induction heads are a specialized circuit consisting of at least two attention heads that implement a prefix-matching algorithm. Formally, given a sequence [A] [B] . . . [A], an induction head at the current position [A] attends to the previous occurrence of [A] and *copies* the subsequent token [B] into the residual stream (Figure 21.5).

The Mechanism of Composition To understand how the two heads in an induction circuit communicate, we must define the three types of *attention composition* [\[Elhage](#page-15-0) [et al.,](#page-15-0) [2021\]](#page-15-0):

- V-Composition: The output (Value) of a previous head is used as the input to a subsequent head.
- Q-Composition / K-Composition: The output of a previous head affects *where* a subsequent head looks by writing into the Query or Key vectors of a token.

An induction head typically relies on *K-Composition*: a previous token head in an early layer writes information about the preceding token into the residual stream. The induction head in a later layer then uses this information in its Key vector to match the current prefix, allowing it to attend to the correct historical token [B].

![](assets/_page_8_Diagram_7.jpeg)

Figure 21.5: *Mechanics of an Induction Head Circuit.* Induction heads implement a prefix-matching algorithm where the model identifies a previous occurrence of the current token (the prefix) and attends to the subsequent token to predict a repeat sequence.

[Olsson et al.](#page-17-0) [\[2022\]](#page-17-0) observed that these heads emerge during a specific phase transition in training, coinciding with a sharp increase in the model's in-context learning capabilities. This was one of the first instances where a complex capability was traced to a specific, discrete algorithm implemented across multiple layers.

### 21.6.2 Methodologies for Circuit Discovery

Manually identifying circuits is labor-intensive. The study of the *indirect object identification (IOI)* task by [Wang et al.](#page-18-0) [\[2022\]](#page-18-0) remains the standard for manual tracing. By utilizing *path patching*, they identified a network of 26 attention heads (classified into 

{9}------------------------------------------------

<span id="page-9-0"></span>roles such as *Name Movers*, *S-Inhibition Heads*, and *Duplicate Token Detectors*) that collectively perform coreference resolution. *Coreference resolution* refers to the process of identifying when different tokens in a text refer to the same underlying entity. For example, in the IOI task, the model must determine that a later mention of a name (the indirect object) refers back to a specific person introduced earlier in the sentence, a computation implemented by a specialized sub-network of attention heads.

However, scaling these findings to frontier models requires *automated circuit discovery (ACD)*.

- ACD: [Conmy et al.](#page-15-0) [\[2023\]](#page-15-0) introduced an algorithm that performs recursive ablation to prune edges in the computation graph, preserving only those whose removal significantly degrades model performance on a specific task.
- Sparse Feature Circuits: Moving beyond neurons, more recent frameworks integrate sparse autoencoders (SAEs) to trace circuits in *feature space* rather than neuron space [\[Marks et al.,](#page-16-0) [2024\]](#page-16-0). By treating SAE features as the nodes and the weights between them as edges, researchers can build attribution graphs that visualize the flow of human-understandable concepts (e.g., *refusal*, *translation*, or *grammar checking*).

Evaluation: Faithfulness, Completeness, and Minimality It is vital to distinguish between a *suggested* circuit and a *causal* circuit. A circuit is considered *faithful* if the model's performance is preserved when all components outside the circuit are ablated. It is *complete* if it captures all pathways the model uses to solve the task.

As models scale, we find that algorithms are often represented redundantly or in a distributed manner across the residual stream. Therefore, circuit tracing is not merely a search for a single wire, but an attempt to map the *causal mediation* of information through the high-dimensional latent space of the model.

## 21.7 Activation Patching and Causal Interventions

While observational tools such as the logit lens or probing can identify correlations between internal states and outputs, *activation patching* (also termed *causal tracing* or *interchange interventions*) allows researchers to test causal claims. This technique is a practical application of *causal mediation analysis* [\[Vig et al.,](#page-18-0) [2020\]](#page-18-0), designed to isolate the effect of a specific model component on downstream behavior (see [Figure 21.6](#page-10-0)).

**Formal framework.** The procedure involves two distinct forward passes:

1. **Clean run** ($x_{\mathrm{clean}}$): an input that produces the target behavior (e.g., *The Eiffel Tower is in Paris*).
2. **Corrupted or counterfactual run** ($x_{\mathrm{corr}}$): a modified input that disrupts the target behavior (e.g., *The Eiffel Tower is in Rome*) or adds Gaussian noise to the *Eiffel Tower* embedding.

{10}------------------------------------------------

<span id="page-10-0"></span>![](assets/_page_10_Diagram_2.jpeg)

Figure 21.6: *Activation patching.* Activations from a *clean run* (where the model succeeds) are swapped into a *corrupted run* (where it fails). If this interchange restores the correct output, the patched component is causally implicated in the task.

Let $z_{i,\ell}$ denote the activation of component $i$ at layer $\ell$. Activation patching replaces the corrupted-run activation $z^{\mathrm{corr}}_{i,\ell}$ with its clean-run counterpart $z^{\mathrm{clean}}_{i,\ell}$ during the corrupted forward pass. We then measure probability recovery:

$$
\operatorname{Effect}=\frac{P(y_{\mathrm{target}}\mid\mathrm{patched})-P(y_{\mathrm{target}}\mid\mathrm{corr})}{P(y_{\mathrm{target}}\mid\mathrm{clean})-P(y_{\mathrm{target}}\mid\mathrm{corr})} \tag{21.7}
$$

An effect score near 1 indicates that the patched component is a sufficient causal mediator for the information being studied.

**Locating knowledge and circuits.** Activation patching has led to significant architectural insights. Meng et al. (2022) used large-scale activation sweeps to show that factual associations are often localized in specific MLP blocks in a model's middle layers. They found that these knowledge neurons could cause a model to produce a specific fact even when the input context was corrupted.

Beyond localization, patching is the primary tool for *path patching*, which identifies interactions between components. In the indirect object identification study discussed earlier, Wang et al. (2022) patched specific attention heads to trace information flow from duplicate-token detectors to name movers. This reveals the model's computational graph and helps explain its internal algorithm, rather than merely locating information.

**Scalability and attribution patching.** A major challenge is the computational cost of activation patching: a full sweep requires a forward pass for every component in the model. To reduce this cost, researchers use *attribution patching* [Syed et al., 2024], which uses gradients to linearly approximate a patch's effect. This supports rapid hypothesis generation; promising causal candidates can then be validated with exact, but slower, activation patching. This precision makes patching indispensable for moving from speculative interpretation to rigorous causal evidence.

{11}------------------------------------------------

<span id="page-11-0"></span>

## 21.8 Sparse Autoencoders: Dictionary Learning for Feature Discovery

As discussed in the context of superposition, neural networks often represent more features than they have dimensions ($n>d$). To disentangle these representations, researchers use *sparse autoencoders* (SAEs), an unsupervised form of dictionary learning that projects activations into a high-dimensional, sparse latent space where features are approximately monosemantic.

![](assets/_page_11_Diagram_4.jpeg)

Figure 21.7: *Architecture of a sparse autoencoder.* The model projects lower-dimensional input activations ($x\in\mathbb{R}^{d}$) into a higher-dimensional latent space ($c\in\mathbb{R}^{N}$, where $N\gg d$). A sparsity penalty (e.g., $L_1$ regularization) encourages most hidden units to remain inactive, yielding a sparse set of interpretable features that are used to reconstruct the original input.

{12}------------------------------------------------

<span id="page-12-0"></span>

### Formal Architecture and Training Objective

An SAE consists of an encoder that maps a model activation $x\in\mathbb{R}^{d}$ to a high-dimensional hidden state $c\in\mathbb{R}^{N}$ (where $N\gg d$), and a decoder that attempts to reconstruct the original activation (Figure [21.7](#page-11-0)).

The encoder is typically defined as:

$$
c=\operatorname{ReLU}(W_{\mathrm{enc}}(x-b_{\mathrm{dec}})+b_{\mathrm{enc}}) \tag{21.8}
$$

And the reconstruction is given by:

$$
\hat{x}=W_{\mathrm{dec}}c+b_{\mathrm{dec}} \tag{21.9}
$$

To ensure the model learns a *sparse* dictionary rather than a simple identity mapping, we optimize a loss function combining reconstruction fidelity with a sparsityinducing penalty:

$$
L=\lVert x-\hat{x}\rVert_2^2+\lambda\lVert c\rVert_1 \tag{21.10}
$$

where $\lambda$ controls the trade-off between reconstruction accuracy and feature sparsity. Recent advancements, such as *JumpReLU SAEs* [\[Rajamanoharan et al.,](#page-17-0) [2024\]](#page-17-0), utilize a thresholded activation function to achieve near-binary on-off feature behavior, which more closely aligns with the theoretical ideal of discrete semantic features.

### 21.8.1 Monosemanticity and Shadow Models

The efficacy of SAEs has been validated across various scales. [Cunningham et al.](#page-15-0) [\[2023\]](#page-15-0) demonstrated that SAEs could recover latent features in GPT-2 representing specific grammatical concepts (e.g., numerical context), while [Gao et al.](#page-15-0) [\[2024\]](#page-15-0) successfully scaled this to billion-parameter models, identifying features for programming languages and sentiment.

A profound implication of this work is the ability to construct a *shadow model*: by replacing a model's polysemantic MLP layers with a sparse feature-based approximation, we can preserve model performance while rendering the internal computation legible. [Bricken et al.](#page-14-0) [\[2023\]](#page-14-0) showed that these SAE features are significantly more monosemantic than the original neurons, effectively untangling the superposed states into a basis that humans can reason about.

In the context of circuit tracing, researchers have moved beyond simple autoencoders to *transcoders* [\[Dunefsky et al.,](#page-15-0) [2024\]](#page-15-0). While a standard SAE reconstructs a single layer\'s activations ($x\rightarrow\hat{x}$), a transcoder is trained to map activations from layer $\ell$ to the *output* of layer $\ell$ (e.g., $x\rightarrow\operatorname{MLP}(x)$). This forces the latent features $c$ to represent the actual computational transformation performed by the layer, providing a direct map of how information is processed rather than just how it is stored.

**Evaluation and open challenges.** Evaluating SAEs requires metrics that balance *faithfulness* (how much performance is lost when using the features?) and *interpretability* (can people or automated LLM probes reliably describe a feature?). Despite their success, SAEs face open challenges such as feature splitting, in which one semantic concept is divided across redundant features, and the difficulty of interpreting very rare features.

{13}------------------------------------------------

## <span id="page-13-0"></span>21.9 Concluding Remarks

### 21.9.1 Outlook and Emerging Directions

Mechanistic interpretability has transitioned from a niche curiosity to a rigorous discipline that seeks to bridge the gap between statistical performance and scientific understanding. As the field matures, several frontier directions are defining the research agenda for the next decade.

The most pressing challenge remains the interpretability labor bottleneck. While manual circuit discovery on GPT-2 was a breakthrough, the sheer number of features in frontier models like GPT-4 or Claude 3 necessitates automated workflows. *Automated interpretability* [\[Bills et al.,](#page-14-0) [2023\]](#page-14-0) utilizes LLMs to generate and test hypotheses about the semantic meaning of millions of latent features discovered via SAEs. This recursive loop—using a model to interpret its own predecessors—is likely the only path toward the complete transparency of trillion-parameter systems.

To move beyond just-so stories, the field is adopting more rigorous validation frameworks. *Causal scrubbing* [\[Chan et al.,](#page-14-0) [2022\]](#page-14-0) represents the current state-ofthe-art for hypothesis testing. By systematically replacing components of a proposed circuit with their expected values under a specific hypothesis and measuring the performance drop, researchers can quantitatively measure the *faithfulness* of their explanations. This moves MI closer to a formal verification science, where one can prove that a model uses a specific algorithm.

Mechanistic interpretability is increasingly viewed as a prerequisite for robust *AI alignment*. By identifying the circuits responsible for deceptive alignment or situational awareness, researchers hope to build safety buffers that detect if a model is merely producing a user-pleasing answer while internally calculating a different, potentially harmful outcome. Furthermore, understanding the *training dynamics*, such as the *grokking* phenomenon where a model suddenly moves from memorization to generalization [\[Nanda et al.,](#page-17-0) [2023\]](#page-17-0), allows us to intervene during the training process to steer models toward more interpretable and safe representations.

While this chapter focused on language models, the principles of MI are being rapidly extended to other domains, for example to *time-series transformers* [\[Kalnare](#page-16-0) ¯ [et al.,](#page-16-0) [2026\]](#page-16-0). Time-series transformers present unique challenges, as the tokens lack discrete human semantics, requiring the development of new dictionary learning techniques for continuous signals. Similarly, interpreting vision transformers and multiagent reinforcement learning systems will require a synthesis of spatial, temporal and social interpretability lenses.

### 21.9.2 Further Reading

The journey from post-hoc behavioral observation to mechanistic circuit diagrams represents a paradigm shift in AI research. We are no longer satisfied with models that merely work; we demand to know *how* they work. By combining the microscope of sparse autoencoders with the scalpel of activation (and attention) patching, the community is building a comprehensive map of the transformer's mind, one circuit at a time.

{14}------------------------------------------------

#### <span id="page-14-0"></span>- Further Reading

For those interested in the foundational research that defined this field, we highly recommend the [Transformer Circuits Thread](https://transformer-circuits.pub/)*<sup>a</sup>* .

*a*<https://transformer-circuits.pub/>

# Bibliography

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. *arXiv preprint arXiv:1610.01644*, 2016. [404](#page-1-0)

Ameen Ali, Thomas Schnake, Oliver Eberle, Grégoire Montavon, Klaus-Robert Müller, and Lior Wolf. XAI for Transformers: Better Explanations through Conservative Propagation. In *International Conference on Machine Learning*, pages 435–451. PMLR, 2022. [403](#page-0-0)

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1):207–219, 2022. [404](#page-1-0)

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. *arXiv preprint arXiv:2303.08112*, 2023. [409](#page-6-0)

Leonard Bereska and Efstratios Gavves. Mechanistic interpretability for AI safety–a review. *arXiv preprint arXiv:2404.14082*, 2024. [405](#page-2-0)

Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models. [paper](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html), 2023. [416](#page-13-0)

Sid Black, Lee Sharkey, Leo Grinsztajn, Eric Winsor, Dan Braun, Jacob Merizian, Kip Parker, Carlos Ramón Guevara, Beren Millidge, Gabriel Alfour, et al. Interpreting neural networks through the polytope lens. *arXiv preprint arXiv:2211.12312*, 2022. [405](#page-2-0)

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, 2023. https://transformer-circuits.pub/2023/monosemanticfeatures/index.html. [407,](#page-4-0) [415](#page-12-0)

Lawrence Chan, Adria Garriga-Alonso, Nicholas Goldowsky-Dill, Ryan Greenblatt, Jenny Nitishinskaya, Ansh Radhakrishnan, Buck Shlegeris,

{15}------------------------------------------------

<span id="page-15-0"></span>and Nate Thomas. Causal scrubbing, a method for rigorously testing interpretability hypotheses. *AI Alignment Forum*, 2022. [Causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing). [416](#page-13-0)

Xi Chen, Aske Plaat, and Niki van Stein. How does chain of thought think? mechanistic interpretability of chain-of-thought reasoning with sparse autoencoding. In *AAAI2026, arXiv preprint arXiv:2507.22928*, 2026. [405](#page-2-0)

Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. *Advances in Neural Information Processing Systems*, 36:16318–16352, 2023. [405,](#page-2-0) [412](#page-9-0)

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. *arXiv preprint arXiv:2309.08600*, 2023. [405,](#page-2-0) [415](#page-12-0)

Austin L. Davis and Gita Sukthankar. Decoding chess mastery: A mechanistic analysis of a chess language transformer model. In *International Conference on Artificial General Intelligence*, pages 63–72. Springer, 2024. [405](#page-2-0)

Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. Transcoders find interpretable llm feature circuits. *Advances in Neural Information Processing Systems*, 37:24375–24410, 2024. [415](#page-12-0)

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. *Transformer Circuits Thread*, 1(1):12, 2021. [406,](#page-3-0) [411](#page-8-0)

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022. [407](#page-4-0)

Javier Ferrando, Gabriele Sarti, Arianna Bisazza, and Marta R. Costa-Jussà. A primer on the inner workings of transformer-based language models. *arXiv preprint arXiv:2405.00208*, 2024. [405](#page-2-0)

Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*, 2024. [415](#page-12-0)

Qi Huang, Sofoklis Kitharidis, Thomas Bäck, and Niki van Stein. TX-Gen: Multi-Objective optimization for sparse counterfactual explanations for time-series classification. In *Proceedings of the 1st International Conference on Explainable AI for Neural and Symbolic Methods - Volume 1: EXPLAINS*, pages 62–74. INSTICC, SciTePress, 2024. ISBN 978-989-758-720-7. doi: 10.5220/0013066400003886. [404](#page-1-0)

{16}------------------------------------------------

<span id="page-16-0"></span>Matīss Kalnāre, Sofoklis Kitharidis, Thomas Bäck, and Niki van Stein. Mechanistic interpretability for Transformer-Based Time-Series Classification. In Francesco Marcelloni, Kurosh Madani, Niki van Stein, and Joaquim Filipe, editors, *Computational Intelligence*, pages 249–270, Cham, 2026. Springer Nature Switzerland. ISBN 978-3-032-15638-9. [416](#page-13-0)

Amir-Hossein Karimi, Gilles Barthe, Borja Balle, and Isabel Valera. Model-agnostic counterfactual explanations for consequential decisions. In *International conference on artificial intelligence and statistics*, pages 895–905. PMLR, 2020. [404](#page-1-0)

Teuvo Kohonen. Self-organized formation of topologically correct feature maps. *Biological cybernetics*, 43(1):59–69, 1982. [404](#page-1-0)

Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah, and Joshua Batson. On the biology of a large language model. *Transformer Circuits Thread*, 2025. URL [https://transformer-circuits.pub/2025/](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) [attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). [405](#page-2-0)

Ziming Liu, Eric Gan, and Max Tegmark. Seeing is believing: Brain-inspired modular training for mechanistic interpretability. *Entropy*, 26(1):41, 2023. [408](#page-5-0)

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. *Advances in neural information processing systems*, 30, 2017. [404](#page-1-0)

Aleksandar Makelov, George Lange, and Neel Nanda. Towards principled evaluations of sparse autoencoders for interpretability and control. *arXiv preprint arXiv:2405.08366*, 2024. [405](#page-2-0)

Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. *arXiv preprint arXiv:2403.19647*, 2024. [412](#page-9-0)

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. *Advances in neural information processing systems*, 35: 17359–17372, 2022. [413](#page-10-0)

Dang Minh, H Xiang Wang, Y Fen Li, and Tan N Nguyen. Explainable Artificial Intelligence: A Comprehensive Review. *Artificial Intelligence Review*, pages 1–66, 2022. [403](#page-0-0)
C. Molnar. *Interpretable Machine Learning*. Leanpub, 2020. ISBN 9780244768522. URL <https://books.google.nl/books?id=jBm3DwAAQBAJ>. [404](#page-1-0)

{17}------------------------------------------------

<span id="page-17-0"></span>Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*, 2023. [405,](#page-2-0) [416](#page-13-0)

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. *arXiv preprint arXiv:2209.11895*, 2022. [411](#page-8-0)

Daking Rai, Yilun Zhou, Shi Feng, Abulhair Saparov, and Ziyu Yao. A practical review of mechanistic interpretability for transformer-based language models. *arXiv preprint arXiv:2407.02646*, 2024. [405](#page-2-0)

Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. Jumping ahead: Improving reconstruction ´ fidelity with JumpReLU sparse autoencoders. *arXiv preprint arXiv:2407.14435*, 2024. [415](#page-12-0)

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 779–788, 2016. [404](#page-1-0)

Shaoqing Ren, Kaiming He, Ross Girshick, Xiangyu Zhang, and Jian Sun. Object detection networks on convolutional feature maps. *IEEE transactions on pattern analysis and machine intelligence*, 39(7):1476–1481, 2016. [404](#page-1-0)

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “Why Should I Trust You?” Explaining the Predictions of Any Classifier. In *Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining*, pages 1135–1144, 2016. [404](#page-1-0)

Thiago Rios, Bas van Stein, Stefan Menzel, Thomas Bäck, Bernhard Sendhoff, and Patricia Wollstadt. Feature visualization for 3d point cloud autoencoders. In *2020 International Joint Conference on Neural Networks (IJCNN)*, pages 1–9. IEEE, 2020. [403](#page-0-0)

Mansi Sakarvadia, Arham Khan, Aswathy Ajith, Daniel Grzenda, Nathaniel Hudson, Andre Bauer, Kyle Chard, and Ian Foster. Attention lens: A tool for mechanistically interpreting the attention head information retrieval mechanism. *arXiv preprint arXiv:2310.16270*, 2023. [410](#page-7-0)

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In *Proceedings of the IEEE international conference on computer vision*, pages 618–626, 2017. [403](#page-0-0)

Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, et al. Open problems in mechanistic interpretability. *arXiv preprint arXiv:2501.16496*, 2025. [405](#page-2-0)

{18}------------------------------------------------

<span id="page-18-0"></span>Aaquib Syed, Can Rager, and Arthur Conmy. Attribution patching outperforms automated circuit discovery. In *Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP*, pages 407–416, 2024. [414](#page-11-0)

Bas Van Stein, Elena Raponi, Zahra Sadeghi, Niek Bouman, Roeland CHJ Van Ham, and Thomas Bäck. A comparison of global sensitivity analysis methods for explainable AI with an application in genomic prediction. *IEEE Access*, 10:103364–103381, 2022. [403](#page-0-0)

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. *Advances in neural information processing systems*, 33:12388– 12401, 2020. [412](#page-9-0)

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *arXiv preprint arXiv:2211.00593*, 2022. [411,](#page-8-0) [413](#page-10-0)

Zhenyu Wang. Logitlens4llms: Extending logit lens analysis to modern large language models, 2025. URL <https://arxiv.org/abs/2503.11667>. [408](#page-5-0)

{19}------------------------------------------------

