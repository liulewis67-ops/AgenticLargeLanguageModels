<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 83-110; Marker split PDF pages 1-28. -->



{0}------------------------------------------------

# <span id="page-0-0"></span>Chapter 4

# Scaling Laws and Efficient Large Language Models

#### What You Will Learn

In this chapter you will learn how performance of language models scales with data and training effort. Specifically, you will be able to:

- Explain how model performance relates to compute, data, and model size through *scaling laws*, and predict the effect of increasing one factor while holding the others fixed.
- Describe what techniques can be used to develop *smaller language models*, and analyze the tradeoff of using smaller models with respect to compute requirements and performance.
- Recognize the difference between *inference-time compute* and training-time compute.
- Reflect on *sustainable language models* and the environmental and privacy implications of large-scale deployment.

# 4.1 Scaling Laws: Compute, Data, Parameters

There are three primary factors involved in the training of LLMs: (1) the amount of compute, (2) the size of the dataset, and (3) the size of the model. Studies have shown that these three factors relate to the final performance of a model, measured as its loss on a held-out test set ($L$). This relation has been found to be both smooth and predictable. The relationship takes the form of a *power law*: as any of the three resources is increased, the loss decreases proportionally to a negative power of that resource. For instance, doubling the model size might reduce the loss by a factor of $2^{-\alpha}$ for some positive exponent $\alpha$. Because the behavior is predictable, we can estimate how much a model will improve (for instance, from doubling the training dataset size) *before* committing to an expensive training run.

{1}------------------------------------------------

<span id="page-1-0"></span>![](assets/_page_1_Figure_2.jpeg)

Figure 4.1: Loss on language modeling objective is a function of training compute, dataset size, and model size [\[Kaplan et al.,](#page-21-0) [2020\]](#page-21-0).

We define these three quantities ($C$, $D$, $N$) as follows:

- Compute ($C$): the total amount of arithmetic performed over the entire training run, measured in floating point operations (FLOPs). Note that this is a count of operations in total, not a rate per second. A widely used approximation relates it to the other two quantities as $C\approx6ND$, since processing a single token touches each of the $N$ parameters roughly six times across the forward (twice) and backward pass (four times).
- Dataset size ($D$): the total number of tokens the model is trained on. Note that a token is a subword unit defined by the tokenizer, so the same corpus corresponds to different values of $D$ under different tokenizers. The count of $D$ is the number of tokens as they are used during the training run. When data is repeated over several epochs, the repeated tokens still add to $D$ but may yield diminishing returns compared to fresh data.
- Model size ($N$): the number of trainable parameters (weights) in the model architecture. This count usually refers only to the parameters that increase when scaling model depth, and therefore excludes the embedding matrix.

Training large models comes at considerable cost, usually in the realm of tens of millions of dollars, and up to months of compute time. Scaling laws allow us to predict whether a model will train well.

#### 4.1.1 Kaplan Scaling Laws

There are different empirical results for scaling laws that are found by fitting a function to empirical test data. For instance, in Figure 4.1, every point is a trained model on a fixed dataset with a similar experimental setup, while only changing the three primary variables $C$, $D$, $N$. In early 2020, [Kaplan et al.](#page-21-0) [\[2020\]](#page-21-0) found how model performance is related to $C$, $D$, $N$. They estimated, following empirical observations, that the loss follows independent power laws related to the three main variables: $L(N)\propto N^{-\alpha_N}$ and $L(D)\propto D^{-\alpha_D}$, with $\alpha_N\approx0.076$ and $\alpha_D\approx0.095$.

Kaplan et al. combined the individual power laws into an optimizable equation (Equation [4.1](#page-2-0)). They assumed that the relationship would also apply to other generative modeling tasks [\[Kaplan et al.,](#page-21-0) [2020\]](#page-21-0). Under a compute-budget constraint, their derivation identifies an optimal model size $N$ and dataset size $D$ in terms of the optimal step size $S$ and batch size $B$.

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Figure_2.jpeg)

Figure 4.2: Larger models reach a given loss using fewer training tokens, illustrating greater sample efficiency [\[Kaplan et al.,](#page-21-0) [2020\]](#page-21-0).

The resulting loss function is:

$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}$ (4.1)

As the compute budget $C$ increases, Kaplan et al. concluded that model size $N$ should grow faster than dataset size $D$ ($N\propto C^{0.73}$ and $D\propto C^{0.27}$), a finding that favored large models such as GPT-3. The negative exponents reflect diminishing returns: doubling either model size or dataset size reduces loss by only a few percent.

At first, this recommendation seems to conflict with the exponents: because $\alpha_D>\alpha_N$, increasing data reduces loss faster than increasing model size for the same relative change. The explanation is that compute, rather than either resource alone, is constrained. Since $C\approx6ND$, increasing $D$ consumes compute just as increasing $N$ does. Model size therefore becomes an increasingly likely bottleneck. Once the compute cost of data is accounted for, allocating additional compute to $N$ yields a better return than allocating it to $D$, so the optimal allocation favors larger models.

**Sample efficiency.** Kaplan et al. also observed that larger models are more sample-efficient. For a fixed number of training tokens, a larger model reaches a lower loss; to reach a given target loss, it requires fewer tokens and fewer optimization steps than a smaller model. Figure 4.2 illustrates this effect.

<sup>1</sup> $B\propto C^{0.24},\ S\propto C^{0.03},\ D=B\cdot S\propto C^{0.27}$.

{3}------------------------------------------------

<span id="page-3-0"></span>

#### 4.1.2 Chinchilla Scaling Laws

In 2022, Hoffmann et al. [\[2022\]](#page-21-0) reevaluated Kaplan et al.’s findings with a model called Chinchilla and found that most large models at the time were severely undertrained. This discrepancy stemmed from Kaplan et al.’s methodology: their optimal allocation was extrapolated from the early, non-converged training dynamics of many models rather than from runs trained to convergence. This systematically biased their conclusions toward larger models trained for fewer steps.

The key insight from the Chinchilla study was that, for each increase in compute, model size $N$ and training-token count $D$ should grow in roughly equal proportions ($N\propto C^{0.5}$ and $D\propto C^{0.5}$). The resulting rule of thumb is to train on about 20 tokens for each trainable parameter to achieve compute optimality. This result has since been replicated by Besiroglu et al. (2024) and shown to be robust to sizable perturbations [\[Schaeffer et al.,](#page-24-0) [2025\]](#page-24-0).

Chinchilla models loss as a function of the parameter count $N$ and the number of training tokens $D$:

$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$ (4.2)

The term $E$ is the irreducible loss, corresponding to the entropy of natural text. The term $\frac{A}{N^{\alpha}}$ captures the suboptimality caused by a finite number of parameters, while $\frac{B}{D^{\beta}}$ captures the suboptimality caused by training on a finite number of tokens. Although $\alpha$ and $\beta$ play a role similar to Kaplan’s $\alpha_N$ and $\alpha_D$, they are jointly fitted in this two-term parameterization and therefore take different values. Fitting the Chinchilla scaling function to more than 400 models yielded these estimates:

$E\approx1.69,\quad A\approx406.4,\quad B\approx410.7,\quad \alpha\approx0.34,\quad \beta\approx0.28$ (4.3)

Optimizing this loss under a fixed compute budget $C$ (with $C\approx6ND$) gives the compute-optimal model size $N_{\mathrm{opt}}$ and dataset size $D_{\mathrm{opt}}$ for any budget:

$N_{\mathrm{opt}}(C)=G\left(\frac{C}{6}\right)^a,\qquad D_{\mathrm{opt}}(C)=\frac{1}{G}\left(\frac{C}{6}\right)^b$ (4.4)

Here, $G=\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}$ is a constant, $a=\frac{\beta}{\alpha+\beta}\approx0.45$, and $b=\frac{\alpha}{\alpha+\beta}\approx0.55$.

The exponents $a$ and $b$ are not independent: by construction, $a+b=1$. Taking logarithms of $C\approx6ND$ gives $\log C=\log6+\log N+\log D$. Substituting $N_{\mathrm{opt}}\propto C^a$ and $D_{\mathrm{opt}}\propto C^b$ shows that this relation can hold for every $C$ only when $a+b=1$. The same constraint applies to Kaplan’s exponents, since $0.73+0.27=1$. The difference is how the compute budget is divided between $N$ and $D$.

This rule is approximately linear because $N_{\mathrm{opt}}$ and $D_{\mathrm{opt}}$ grow at similar rates as the compute budget increases: their exponents, $a\approx0.45$ and $b\approx0.55$, are close,

{4}------------------------------------------------

<span id="page-4-0"></span>

| Model | Parameters ($N$) | Training tokens ($D$) | $D/N$ | Regime |
|---|---:|---:|---:|---|
| Gopher (Rae et al., 2021) | 280B | 300B | ≈1 | Undertrained |
| GPT-3 (Brown et al., 2020) | 175B | 300B | ≈1.7 | Undertrained |
| Chinchilla (Hoffmann et al., 2022) | 70B | 1.4T | 20 | Compute-optimal |
| LLaMA-1 7B (Touvron et al., 2023a) | 7B | 1.0T | ≈143 | Overtrained |
| LLaMA-2 7B (Touvron et al., 2023b) | 7B | 2.0T | ≈286 | Overtrained |
| LLaMA-3 8B (Grattafiori et al., 2024) | 8B | 15T | ≈1875 | Overtrained |

*Table 4.1: Token-to-parameter ratios of several well-known models relative to the Chinchilla rule of thumb $D\approx20N$.*

so the ratio $D_{\mathrm{opt}}/N_{\mathrm{opt}}$ stays near a constant value of roughly 20 across many orders of magnitude of compute. The value is best understood as an empirical calibration anchored on real compute-optimal runs rather than a number that results cleanly from the fitted coefficients. The Chinchilla model itself (70B parameters trained on 1.4T tokens) sits exactly at $D/N=20$. Table 4.1 shows several well-known models relative to this rule. Models below the ratio are *undertrained* (too large for their token budget), while models far above it are deliberately *overtrained* to obtain a small model that is cheap to run at inference time. Earlier flagship models such as GPT-3 and Gopher fall well below the line ($D/N\approx1$–$2$): they were made very large but trained on comparatively too few tokens. It is worth stressing that $D\approx20N$ is optimal only for minimizing the *training* loss at a fixed training budget. When a model will later serve many inference requests, it is often deliberately *over-trained* on far more than 20 tokens per parameter, trading additional training cost for a smaller model that is cheaper to run at inference time. LLaMA-3 8B, for instance, is trained on roughly 1900 tokens per parameter. This deliberate extreme over-training is chosen so that a small model still reaches a low loss, and is cheap to run during inference.

The picture so far is fairly clean: test loss falls smoothly, monotonically, and predictably with compute, data, and parameters. However, in practice, various factors influence our results. We discuss four phenomena that each probe a different edge of that picture: (1) compute-optimal curves, (2) emergent abilities, (3) inverse scaling, and (4) grokking.

#### 4.1.3 Compute-Optimal Curve

Computer hardware and energy are finite resources. A practical question is how to split a fixed compute budget $C$ between model size $N$ and dataset size $D$ to obtain the best possible model. Scaling laws answer this through the *compute-optimal frontier*: the allocation of $C$ that minimizes the loss.

This frontier is located empirically with the *IsoFLOP* method [\[Hoffmann et al.,](#page-21-0) [2022\]](#page-21-0), shown in Figure [4.3.](#page-5-0) For each model size, we choose the training-token count so that the final FLOP count is constant. We use a cosine annealing schedule whose cycle length matches the target FLOP count. The loss forms a clear valley, meaning that for a given FLOP budget there

{5}------------------------------------------------

<span id="page-5-0"></span>![](assets/_page_5_Figure_2.jpeg)

Figure 4.3: IsoFLOP curves [\[Hoffmann et al.,](#page-21-0) [2022\]](#page-21-0)

is an optimal model size to train (left). Using the location of these valleys, we project optimal model size and number of tokens for larger models (center and right). In green, we show the estimated number of parameters and tokens for an optimal model trained with the compute budget of Gopher. For a fixed FLOP budget we train several models of different sizes: a model that is too small underfits, while a model that is too large is forced to train on too few tokens, so the resulting curve of loss against model size is U-shaped. Its minimum identifies the compute-optimal model size $N_{\mathrm{opt}}$, from which the matching token count $D_{\mathrm{opt}}=C/(6N_{\mathrm{opt}})$ follows via $C\approx6ND$. Fitting a parabola to each such curve locates this minimum, and repeating the procedure across budgets finds how $N_{\mathrm{opt}}$ and $D_{\mathrm{opt}}$ grow with compute.

This parabola fitting must be done with care, and the simplification introduces systematic biases in the estimated optimum [\[Czech et al.,](#page-19-0) [2026\]](#page-19-0). These biases persist even on noise-free synthetic data and stem from three sources: (1) spacing the sampled model sizes too coarsely, (2) sampling that fails to locate the true minimum symmetrically, and (3) the genuine asymmetry of the loss surface, which a symmetric parabola cannot represent. At the scale of Llama-3, they estimate that such biases waste on the order of \$1.4M of compute. As a remedy, they recommend fitting the parametric loss model $L(N,D)=E+A/N^\alpha+B/D^\beta$ directly—Chinchilla Approach 3, the same form we used above to derive the $D\approx20N$ rule—with a numerically robust optimizer, rather than reading the optimum off fitted parabolas.

Complementary views on scaling [\[Paquette et al.,](#page-23-0) [2024\]](#page-23-0) study analytically solvable models (power-law random features trained with one-pass stochastic gradient descent) instead of fitting curves to trained models. They find that the compute-optimal exponent is not a single universal constant, but that the data-complexity/target-complexity plane instead splits into four primary phases (plus three sub-phases), each with its own exponent governing how $N_{\mathrm{opt}}$ grows with compute. The answer to the question which allocation rule to apply, and whether a fixed *20 tokens per parameter* is even the right shape, depends on the regime of the problem.

**Scaling for fine-tuning.** The preceding works studied how pretraining scales. Do these scaling laws also govern fine-tuning? [Zhang et al.](#page-26-0) [\[2024\]](#page-26-0) study both full-model fine-tuning, which updates all of a model's weights, and parameter-efficient fine-tuning (PEFT), which freezes the pretrained weights and trains only a

{6}------------------------------------------------

<span id="page-6-0"></span>![](assets/_page_6_Figure_2.jpeg)

Figure 4.4: Eight examples of emergence in the few-shot prompting setting, each point is a separate model [\[Wei et al.,](#page-25-0) [2022\]](#page-25-0).

small set of added parameters (as in LoRA) [\[Hu et al.,](#page-21-0) [2021\]](#page-21-0). The lesson most relevant to compute budgeting is that finetuning performance improves more from scaling the size of the *base model* than from collecting additional finetuning data, while adding more trainable PEFT parameters helps little. In practice, a task that underperforms is usually better served by starting from a larger pretrained model than by enlarging the adapter or gathering more finetuning examples.

#### 4.1.4 Emergent Abilities

Some of the most interesting phenomena in LLM scaling are *emergent abilities*: capabilities that are not present in smaller models but that appear suddenly as model size or compute cross a certain threshold [\[Wei et al.,](#page-25-0) [2022\]](#page-25-0). Examples include multi-step reasoning, few-shot translation, and specialized coding tasks, as shown in Figure 4.4. The ability to perform a task via few-shot prompting is emergent when a language model achieves random performance until a certain scale, after which performance significantly increases to well-above random.

However, whether these abilities appear suddenly has been contested. [Schaeffer et al.](#page-24-0) [\[2023\]](#page-24-0) argue that emergent abilities may be an illusion caused by the choice of evaluation metric. Figure [4.5](#page-7-0), for example, shows that discrete, non-smooth metrics such as accuracy or exact match can jump sharply, while underlying metrics such as cross-entropy loss or token-level probability follow a smooth, predictable power law across model scales.

{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Figure_2.jpeg)

Figure 4.5: Claimed emergent abilities evaporate upon changing the metric from a nonlinear (top) to a smooth (bottom) metric [\[Schaeffer et al.,](#page-24-0) [2023\]](#page-24-0)

#### 4.1.5 Inverse Scaling

While most tasks improve with scale, some exhibit *inverse scaling*, where larger models perform worse or develop stronger biases. This may happen in tasks where a distractor answer is more probable according to the language modeling objective than the correct answer, leading larger (and more confident) models to consistently choose the wrong option [\[McKenzie et al.,](#page-23-0) [2023,](#page-23-0) [Michaelov and Bergen,](#page-23-0) [2023\]](#page-23-0).

An example of inverse scaling is *hindsight neglect*, a task that tests whether a bet was worth taking based on its expected value. In case the expected value is positive, the model should respond with *yes*, and in case of a negative expected value it should respond with *no*. The situation is as follows. The model is only shown few-shot examples where the actual outcome always matches the expected value (that is, the bettor won money when the expected value was positive and lost money when the expected value was negative). The model is then asked a question about whether it was correct to take a bet where the expected value and the actual outcome do not match. Hindsight neglect tests whether models can correctly generalize from instructions when the few-shot examples are correctly-labeled but misleading. Repetitive algebra tests how models respond to simple algebra questions when given a specific pattern of correctlylabeled few-shot examples. Figure [4.6](#page-8-0) shows that models generally start around random performance, but fall below random performance at around 10<sup>22</sup> training FLOPs.

{8}------------------------------------------------

<span id="page-8-0"></span>![](assets/_page_8_Figure_2.jpeg)

Figure 4.6: Scaling behavior for the Hindsight Neglect (left) and Repetitive Algebra (right) tasks [\[McKenzie et al.,](#page-23-0) [2023\]](#page-23-0)

![](assets/_page_8_Figure_4.jpeg)

Figure 4.7: Early memorization of the training set (blue) versus late generalization (red: grokking) on a modular addition task [\[Nanda et al.,](#page-23-0) [2023\]](#page-23-0)

#### 4.1.6 Grokking

Grokking is a relatively recent paradox in statistical learning theory. This theory holds that, conventionally, when a large model is presented with a small dataset, it will overfit, due to memorization of the examples. Grokking, then, occurs when we continue to train this too-large model, and, after much more training, the model suddenly finds a generalizing solution [\[Power et al.,](#page-24-0) [2022\]](#page-24-0).

While originally observed on small algorithmic tasks, such as modular addition, the grokking phenomenon suggests that scaling the number of training steps (overtraining) can lead to qualitative shifts from memorization to understanding, even without increasing the dataset size $D$, see Figure 4.7.

The figure shows a data wall that initially limits scaling as the data is running out [\[Villalobos et al.,](#page-25-0) [2024\]](#page-25-0), but later, after much more training, the model groks a generalizing solution.

Grokking is relevant to language modeling since we are rapidly running out of high-quality human text on the internet to satisfy the requirements on data $D$ of Chinchilla 

{9}------------------------------------------------

<span id="page-9-0"></span>scaling for next-generation models. Because of these inherent limits, researchers are looking beyond simple scaling of $D$. As an alternative, we can look at a variety of these new frontiers: (1) better architectural efficiency (such as Mixture of Experts), (2) inference-time scaling (such as giving models more compute during inference to *think*, as seen in OpenAI\'s o-series), or (3) synthetic data generation (Chapter 19).

# 4.2 Small Language Models

As the requirements for compute and data continue to grow, there is an increasing interest in *small language models* (SLMs): models that are small enough to run on consumer hardware or even mobile or edge devices while maintaining high performance. These models are often developed using techniques that optimize for efficiency without sacrificing too much performance.

To understand the deployment landscape for these models, it is helpful to categorize common types of compute infrastructure:

- Datacenters: High-performance GPUs (such as NVIDIA H100s) and TPUs, primarily used for *pretraining* large models and hosting high-throughput inference APIs.
- Consumer Hardware: Standard desktop GPUs and high-end CPUs, where SLMs can be affordably *finetuned* or run locally by individual users.
- Edge Devices: Mobile phones, Internet-of-Things devices, and laptops, which are increasingly equipped with Neural Processing Units (NPUs) specifically designed for low-power, efficient on-device *inference*.

To fit models into the latter constrained environments, optimization techniques focus on speeding up throughput, reducing latency, reducing memory usage, and increasing sample efficiency. The techniques for achieving these goals can be grouped into two categories: (1) model-centric optimization, which targets the architecture of the models and their optimization approaches, and (2) data-centric optimization, which targets the data used to train the models. We will discuss both categories.

### 4.2.1 Model-centric Optimization

First we discuss the model-centric optimization techniques. We introduce the following techniques that allow for efficient training of smaller models, while still maintaining high task performance. Model-centric techniques can focus on pretraining/finetuning or inference.

#### 4.2.1.1 Reducing Pretraining Cost

Parameter-efficient fine-tuning (PEFT) methods such as *Low-Rank Adaptation* (LoRA) [\[Hu et al.,](#page-21-0) [2021\]](#page-21-0) adapt a large model to a new task by training only a tiny fraction of its parameters—often less than 1% [\[Lialin et al.,](#page-22-0) [2023\]](#page-22-0). The key observation is that the weight update learned during fine-tuning has a low *intrinsic rank*: although

{10}------------------------------------------------

<span id="page-10-0"></span>![](assets/_page_10_Diagram_2.jpeg)

Figure 4.8: Low-Rank Adaptation reparameterization [\[Hu et al.,](#page-21-0) [2021\]](#page-21-0). During fine-tuning, only $A$ and $B$ are trained.

a pretrained weight matrix $W_0\in\mathbb{R}^{d\times k}$ is full-rank, the change $\Delta W$ needed to adapt it to a new task can be approximated well by a low-rank matrix. Instead of updating $W_0$ directly, LoRA freezes it and represents the update as the product of two small matrices:

$W=W_0+\Delta W=W_0+BA,\qquad B\in\mathbb{R}^{d\times r},\quad A\in\mathbb{R}^{r\times k}$ (4.5)

where the rank $r\ll\min(d,k)$. For an input $x$, the modified forward pass becomes:

$h=W_0x+\frac{\alpha}{r}BAx$ (4.6)

Here, $\alpha$ is a constant scaling factor that controls the magnitude of the update, roughly like a learning rate in standard training. As Figure 4.8 illustrates, updating only $A$ and $B$ reduces the number of trainable parameters from $d\times k$ to $r(d+k)$. At initialization, $A$ is drawn from a random Gaussian and $B$ is set to zero, so $\Delta W=0$ and training starts from the pretrained model. Because $\Delta W$ can be merged directly back into $W_0$ after training, LoRA adds no inference latency, and one base model can serve many small, swappable task-specific adapters.

**Architectural innovations.** A second family of techniques changes the model architecture itself to make it cheaper to run, rather than compressing a fixed architecture after the fact. Two influential examples are Grouped-Query Attention and Mixture of Experts.

**Grouped-Query Attention (GQA)** [\[Ainslie et al.,](#page-18-0) [2023\]](#page-18-0) targets the memory-bandwidth bottleneck of autoregressive decoding. In standard multi-head attention (MHA), every query head has its own key and value head, all of which are used to compute attention scores. At the other extreme, multi-query attention (MQA) lets all query heads share one key/value head. This reduces the cache but can hurt expressivity and training stability. GQA interpolates between the two: query heads are divided into $g$ groups, and heads in each group share one key/value head. With $g=1$, GQA is equivalent to MQA; with $g$ equal to the number of query heads,

{11}------------------------------------------------

<span id="page-11-0"></span>![](assets/_page_11_Diagram_2.jpeg)

Figure 4.9: Illustration of the sparsely gated Mixture of Experts (MoE) architecture on top of a decoder-only transformer [\[Zhang et al.,](#page-26-0) [2025\]](#page-26-0)

it is equivalent to MHA, so $g$ trades quality against memory use. GQA is often combined with KV-cache optimizations such as those discussed in Section [4.2](#page-9-0), and it is now the default attention variant in many open-weight models.

*Mixture of Experts* (MoE) [\[Shazeer et al.,](#page-24-0) [2017\]](#page-24-0) decouples the total number of parameters from the compute spent per token. Instead of a single dense feed-forward block, each MoE layer contains many parallel "expert" subnetworks and a lightweight *gating network* (router) that activates only a few experts—typically the top one or two—for each token. Because only the selected experts compute, the cost per token stays close to that of one dense block, even as total parameter count and model capacity grow by an order of magnitude or more [\[Fedus et al.,](#page-20-0) [2022\]](#page-20-0).

This *sparse activation* lets models such as Mixtral and other recent frontier systems have hundreds of billions of parameters while activating only a fraction for each token. The trade-offs are that all experts still need to fit in memory, the workload must be balanced across experts (often with an auxiliary loss), and routing adds engineering complexity. Figure 4.9 shows a top-$k$ router with $k=2$: it selects the two highest-scoring FFN experts for each token, which then run in parallel and combine their outputs with a weighted sum [\[Zhang et al.,](#page-26-0) [2025\]](#page-26-0).

**Knowledge distillation.** A large teacher model trains a smaller student by encouraging it to match the teacher's output distribution [\[Gu et al.,](#page-21-0) [2023,](#page-21-0) [Xu et al.,](#page-26-0) [2024\]](#page-26-0). Crucially, the student learns from the teacher's full *soft* probability distribution (typically softened with a temperature), not just the single correct answer. These targets carry richer information, including which incorrect answers the teacher considers plausible. This lets the student approach the teacher's quality at a fraction of its size. The same idea underpins the synthetic-data generation discussed

{12}------------------------------------------------

<span id="page-12-0"></span>below, where a frontier model effectively acts as the teacher, generating high-quality training data for a smaller student model.

#### 4.2.1.2 Reducing Inference Cost

Once a model is trained, it can be further optimized for deployment at inference time. Whereas training-time techniques focus on reducing the cost of producing the model, inference-time techniques target the cost of *using* it: the throughput (tokens generated per second), the latency (time to the first and each subsequent token), and the memory footprint required to serve a request. We discuss several techniques that are commonly used to reduce inference cost, which can be applied to both large and small models.

**Key–Value (KV) caching.** During autoregressive generation, the model produces one token at a time, and each new token attends to all previous tokens. Recomputing the attention keys and values for the entire context at every step is quadratically expensive. However, when predicting the next token, the representations of the previous tokens do not change. The model can therefore store their key and value vectors in a *KV cache* and reuse them. This avoids recomputing the keys and values for the whole context at every step, making decoding practical. The cache itself grows linearly with sequence length and batch size, however, and can dominate memory use for long contexts. GQA and MQA reduce its size by sharing key/value heads across query heads. *PagedAttention* instead stores the cache in non-contiguous blocks, reducing memory fragmentation and supporting higher serving throughput [\[Kwon et al.,](#page-21-0) [2023\]](#page-21-0).

**Pruning.** Another way to reduce model size is to remove redundant or less important connections in the neural network [\[Ma et al.,](#page-22-0) [2023\]](#page-22-0). *Unstructured pruning* sets individual weights to zero, whereas *structured pruning* removes entire neurons, attention heads, or layers. Structured pruning can yield speedups on standard hardware; unstructured pruning requires sparsity-aware kernels to deliver practical benefits.

**Quantization.** Another way to reduce memory use during inference is to lower the precision of the weights loaded into memory—for example, from 16-bit to 8-bit or 4-bit values—without changing the matrix dimensions. This can reduce memory use and speed up inference, although lower precision may also reduce model quality. The same approach can be applied to the KV cache to keep longer contexts in memory. Quantization is widely used [\[Dettmers et al.,](#page-20-0) [2022,](#page-20-0) [Dettmers and](#page-19-0) [Zettlemoyer,](#page-19-0) [2023,](#page-19-0) [Xiao et al.,](#page-25-0) [2023\]](#page-25-0).

Alternatively, rather than casting all weights to the same precision, *mixed precision* modes have been proposed, with different levels of numerical precision for different parts of the computation, to balance speed and stability [\[Micikevicius et al.,](#page-23-0) [2017\]](#page-23-0). This approach may suffer from non-deterministic results due to numerical instabilities [\[Yuan et al.,](#page-26-0) [2025\]](#page-26-0).

{13}------------------------------------------------

<span id="page-13-0"></span>**Speculative decoding.** A small draft model quickly proposes several tokens, which a larger target model then verifies in parallel [\[Leviathan et al.,](#page-22-0) [2023\]](#page-22-0). This can speed up inference—by up to two or three times in some settings—without changing the target model's output distribution, combining models from different points on the scaling curve.

#### 4.2.2 Data-centric Optimization

In addition to model-centric approaches, the literature emphasizes that data quality and curation are often more important than the size of the dataset when training efficient models [\[Luo et al.,](#page-22-0) [2025\]](#page-22-0). Small models can perform well if they are trained on highly curated, high-quality data. Hence, it is important to create a good data recipe, in addition to an efficient model architecture. Data-centric techniques broadly fall into two camps: *selecting* the most useful parts from a broader pool of data, and *generating* new synthetic data where good data is scarce.

#### 4.2.2.1 Data Selection and Curation

Rather than training on all available text, a growing body of work shows that carefully choosing *which* data to train on, and in what proportions, yields better models for the same compute budget. For pretraining, NVIDIA's *CLIMB* framework [\[Diao](#page-20-0) [et al.,](#page-20-0) [2026\]](#page-20-0) embeds a large corpus into a semantic space, clusters it, and then iteratively searches for the data mixture that minimises proxy-model loss. The resulting *ClimbMix* dataset (400B tokens) lets a sub-billion-parameter model outperform comparable models trained on far larger but unfiltered corpora. The same principle applies at the finetuning stage: *LIMA* [\[Zhou et al.,](#page-26-0) [2023\]](#page-26-0) shows that supervised finetuning on only 1,000 carefully curated prompt–response pairs is enough to produce a strong instruction-following model, supporting the view that most knowledge is acquired during pretraining and that alignment mainly teaches the model a desired response *style*.

#### 4.2.2.2 Synthetic Data Generation

When suitable data is scarce or unavailable, it can be generated. The *Textbooks Are All You Need* approach [\[Gunasekar et al.,](#page-21-0) [2023\]](#page-21-0), used for models such as Microsoft's Phi series, showed that clean, synthetic, textbook-quality data can help a relatively small 1.3B-parameter model outperform models ten times its size.

Such pipelines typically use frontier models to generate examples, then apply *LLM-as-a-judge* scoring to retain accurate and educational material. Rigorous *decontamination* is also needed to prevent synthetic data from leaking into benchmark test sets and inflating reported performance. Synthetic data is especially useful for reasoning: training on step-by-step reasoning trajectories from larger models can distill complex problem-solving skills into a smaller architecture (see also Chapter 6) [\[Muennighoff et al.,](#page-23-0) [2025,](#page-23-0) [Arora and Zanette,](#page-19-0) [2026\]](#page-19-0).

{14}------------------------------------------------

#### <span id="page-14-0"></span>4.2.2.3 Sample-Efficient Learning

Beyond synthetic data, researchers are investigating how models can learn more efficiently from limited, naturalistic data. Initiatives such as the *BabyLM Challenge* [\[Warstadt et al.,](#page-25-0) [2023\]](#page-25-0) train models under strict data constraints, such as 10M to 100M tokens. This is comparable to the language input a human child receives during early development. The challenge encourages sample-efficient architectures and provides a testbed for studying cognitively plausible language acquisition.

### 4.2.3 Popular Models

Having surveyed improvement techniques, we now look at how selected large and small models apply them. Models are often grouped by openness: proprietary models expose an API; open-weight models release trained parameters; and open-source models also share training data and code. In practice, openness is a spectrum rather than a strict set of categories. An openness index can be used to assess transparency [\[Artificial Analysis,](#page-19-0) [2026\]](#page-19-0). Many open models are available through repositories such as Hugging Face.

The following examples are not exhaustive. They highlight influential open-weight models designed to run efficiently on consumer hardware and illustrate the range of efficiency techniques and model scales in use at the time of writing.

**Phi series** [\[Abdin et al.,](#page-18-0) [2024\]](#page-18-0) comprises capable SLMs that can run locally on mobile devices. Its core innovation is a data-centric training approach: Microsoft curates high-quality web data and combines it with synthetic data generated by larger models. As a result, smaller models such as the 3.8B-parameter Phi-3-mini perform well and can rival much larger models, including GPT-3.5 and Mixtral, on reasoning tasks.

**Gemma series** [\[Mesnard et al.,](#page-23-0) [2024\]](#page-23-0) is described by Google as inheriting Gemini's general capabilities, particularly in text, reasoning, and coding, while performing strongly within its size classes. Google has disclosed relatively little about Gemma's training methods and data.

**OLMo family** [\[Ettinger et al.,](#page-20-0) [2025\]](#page-20-0) was developed by the Allen Institute for AI with openness as a central goal. The project released its training pipeline, including the Dolma pretraining corpus, training and evaluation code, logs, and intermediate checkpoints. This transparency supports research into training dynamics, data effects, and algorithmic bias.

**Qwen series** [\[Bai et al.,](#page-19-0) [2023\]](#page-19-0) spans base and instruction-tuned models, from dense models with fewer than one billion parameters to large Mixture-of-Experts (MoE) models. It performs strongly on mathematics, coding, and logical-reasoning benchmarks,

{15}------------------------------------------------

<span id="page-15-0"></span>and also offers strong multilingual capabilities.

**OpenELM** [\[Mehta et al.,](#page-23-0) [2024\]](#page-23-0) is Apple's efficient language-model family for edge devices. Its layer-wise scaling strategy uses smaller latent dimensions in layers closer to the input and progressively wider dimensions toward the output, aiming for high accuracy with fewer parameters.

**gpt-oss** [\[Agarwal et al.,](#page-18-0) [2025\]](#page-18-0) consists of two open-weight reasoning models released by OpenAI. They use an efficient MoE architecture and are optimized for agentic capabilities.

**Mixtral series** [\[Jiang et al.,](#page-21-0) [2024\]](#page-21-0) helped popularize sparse MoE models in the open-weight ecosystem. For example, Mixtral 8×7B activates only part of its parameters for each token, combining a large total parameter count with lower inference compute. The models were released under a Responsible AI License.

**Llama series** [\[Grattafiori et al.,](#page-20-0) [2024\]](#page-20-0) is widely used. Llama 3 was trained on up to 15 trillion tokens, and the family ranges from smaller, edge-friendly models to a 405B-parameter frontier model. Llama helped establish the scale of open models and catalyzed an ecosystem of fine-tuned models and derivatives.

**DeepSeek** [\[Liu et al.,](#page-22-0) [2025\]](#page-22-0) V3 is a large MoE model with 671B total parameters and 37B activated per token. It combines Multi-head Latent Attention (MLA), which compresses the key–value cache, with auxiliary-loss-free expert load balancing. Trained using FP8 mixed precision, it achieved frontier-level performance at comparatively low compute cost. The family also includes the DeepSeek-R1 reasoning model.

**GLM** [\[Zeng et al.,](#page-26-0) [2026\]](#page-26-0) is developed by Zhipu AI. The General Language Model family builds on reasoning capabilities developed early in training and retained throughout the process. Recent versions target agentic workflows and software engineering, with an emphasis on tool use and code generation.

**Pythia** [\[Biderman et al.,](#page-19-0) [2023\]](#page-19-0) is a suite of research models from EleutherAI, ranging from 70M to 12B parameters. Each model is trained on the same data in the same order, and intermediate checkpoints are released. This setup enables controlled studies of how capabilities, memorization, and biases emerge during training.

**BLOOM** [\[Scao et al.,](#page-24-0) [2023\]](#page-24-0) is a 176B-parameter open-access model developed by the BigScience collaborative research effort. It was trained on the ROOTS corpus, which covers 46 natural languages and 13 programming languages, and was among the first large-scale models to prioritize multilingual coverage and openness. It was released under a Responsible AI License.

{16}------------------------------------------------

<span id="page-16-0"></span>**SmolLM** [\[Allal et al.,](#page-18-0) [2025\]](#page-18-0) is Hugging Face's family of compact models (135M, 360M, and 1.7B parameters) for efficient on-device use. Like the Phi series, its performance reflects careful data curation: the SmolLM-Corpus combines high-quality web data and Python code with textbooks and stories generated synthetically using Mixtral.
## 4.3 Inference-Time Compute

Traditional scaling laws describe the relationship between compute and performance during training. Recent work shifts some of that computational burden to inference, a practice known as *inference-time scaling* (or test-time compute) [\[Muennighoff et al.,](#page-23-0) [2025\]](#page-23-0). Instead of spending the entire compute budget during training, a model is allowed to deliberate longer at inference time. This budget is often measured by the number of tokens generated or reasoning paths explored in pursuit of a higher-quality answer. Throughput (tokens per second) therefore matters: the faster tokens are generated, the more reasoning the model can perform in a given time.

A larger inference-time token budget can improve model capabilities [\[Du et al.,](#page-20-0) [2024\]](#page-20-0). Recent work also suggests that, on some difficult reasoning tasks, carefully allocating test-time compute can allow smaller models to outperform larger ones [\[Snell et al.,](#page-24-0) [2024\]](#page-24-0). These gains are not guaranteed; they depend on the task and on the search or verification strategy used [\[Stechly et al.,](#page-24-0) [2023\]](#page-24-0). Beyond performance, this shift raises broader questions about what kind of systems LLMs are, which Chapter 6 explores in detail.

## 4.4 Sustainable Language Models

The large-scale training and deployment of LLMs have significant environmental and economic implications. Training and inference consume substantial energy, while cooling them places additional strain on water supplies [\[Li et al.,](#page-22-0) [2025,](#page-22-0) [Jegham et al.,](#page-21-0) [2025\]](#page-21-0). Standardized reporting of training energy use and carbon footprints has only recently begun [\[Ding and Shi,](#page-20-0) [2024\]](#page-20-0). Training a single large model has reportedly consumed as much energy as hundreds of households use in a year. For BLOOM, emissions exceeded 50 tonnes of CO₂ equivalent when both dynamic power consumption and the embodied carbon of hardware manufacturing were included [\[Luccioni et al.,](#page-22-0) [2023\]](#page-22-0).

As LLM use grows across science and society, inference increasingly dominates the total carbon footprint, especially for models used by millions of people each day [\[Özcan et al.,](#page-23-0) [2025,](#page-23-0) [Wu et al.,](#page-25-0) [2022,](#page-25-0) [Samsi et al.,](#page-24-0) [2023\]](#page-24-0). This creates an opportunity: spending more compute during training to produce a smaller, more efficient model (see Section [4.1](#page-0-0)) can reduce the long-term environmental cost of deployment.

As global inference demand grows to billions of queries per day, cumulative energy use and related financial costs can exceed the initial training investment [\[Faiz et al.,](#page-20-0) [2024\]](#page-20-0). Deploying large models at this scale requires substantial data
{17}------------------------------------------------

<span id="page-17-0"></span>center infrastructure and consumes large amounts of electricity and water. To support sustainable, large-scale use, there is a strong push toward deploying Small Language Models (SLMs). By reducing model size and quantizing its weights, an SLM can lower the memory bandwidth and compute required per generated token, offering a greener and more economically viable option.

**Privacy and data leakage.** Beyond environmental concerns, centralizing LLM services through cloud APIs creates privacy risks. When users query cloud-based models, sensitive personal data, corporate intellectual property, or protected health information must be sent to third-party servers. This creates a risk of data leakage through unauthorized interception, adversarial extraction, or the model's inadvertent memorization and later reproduction of private prompts [\[Aiyappa et al.,](#page-18-0) [2023,](#page-18-0) [Li et al.,](#page-22-0) [2024,](#page-22-0) [Chen et al.,](#page-19-0) [2024,](#page-19-0) [Maini et al.,](#page-22-0) [2024,](#page-22-0) [Liu et al.,](#page-22-0) [2022\]](#page-22-0).

To reduce these privacy risks and reliance on cloud infrastructure, developers can move from API-based to *local, on-device inference* [\[Qu et al.,](#page-24-0) [2025\]](#page-24-0):

- **API-based inference:** Relies on powerful remote servers, providing access to frontier models with hundreds of billions of parameters. It introduces network latency, requires an internet connection, and sends user data to the cloud provider.
- **Local inference:** Runs the model directly on the user's own hardware, such as a smartphone, laptop, or edge server. Prompts remain on the device, which can help protect privacy; local inference also avoids network round-trip latency and can work offline.

Local inference remains constrained by the physical limits of consumer and edge hardware, especially memory and bandwidth. LLM inference is typically memory-bound: storing model weights requires substantial RAM, while generating tokens requires high memory bandwidth. Many mobile devices also run on batteries, and sustained matrix operations can quickly drain them or trigger thermal throttling.

Making local deployment practical requires inference optimizations such as 4-bit quantization and speculative decoding. Specialized Neural Processing Units (NPUs) designed for low-power edge AI are also becoming more widely available.
# 4.5 Concluding Remarks

As LLM use becomes more ubiquitous and the demand for quick responses increases, the focus in LLM development is shifting towards more efficient solutions. Hence, the focus on scaling data, compute, and model size purely for raw performance is transitioning into a focus on scaling for *efficiency*.

{18}------------------------------------------------

<span id="page-18-0"></span>While the fundamental power laws still hold, the focus has shifted from simply training the largest possible model toward compute-optimal training, inference-time scaling, and initiatives to develop capable small models that run on consumer and edge hardware.

The directions that have been discussed in this chapter are complementary rather than competing: the model-centric techniques (LoRA, GQA, MoE, quantization, distillation) and the data-centric techniques (careful data selection and synthetic generation) are increasingly combined within a single training recipe.

As we move closer to the data wall, the role of synthetic data and architectural efficiency are likely to grow in importance, and the ability to do more with less is becoming a research focus.

# Bibliography

- Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sebastien Bubeck, Martin Cai, Qin Cai, ´ Vishrav Chaudhary, et al. Phi-3 technical report: A highly capable language model locally on your phone, 2024. URL <https://arxiv.org/abs/2404.14219>. [97](#page-14-0) Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, et al. gptoss-120b & gpt-oss-20b model card, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2508.10925) [2508.10925](https://arxiv.org/abs/2508.10925). [98](#page-15-0) Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 4895–4901, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.298. URL <https://aclanthology.org/2023.emnlp-main.298/>. [93](#page-10-0) Rachith Aiyappa, Jisun An, Haewoon Kwak, and Yong-yeol Ahn. Can we trust the evaluation on ChatGPT? In Anaelia Ovalle, Kai-Wei Chang, Ninareh Mehrabi, Yada Pruksachatkun, Aram Galystan, Jwala Dhamala, Apurv Verma, Trista Cao, Anoop Kumar, and Rahul Gupta, editors, *Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023)*, pages 47–54, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.trustnlp-1.
- 5. URL <https://aclanthology.org/2023.trustnlp-1.5/>. [100](#page-17-0) Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martin Blazquez, Guilherme Penedo, Lewis Tunstall, Andres Marafioti, Agust ´ ´ın Piqueres Lajar´ın, Hynek Kydl´ıcek, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan Son NGUYEN, ˇ

{19}------------------------------------------------

<span id="page-19-0"></span>Ben Burtenshaw, Clementine Fourrier, Haojun Zhao, Hugo Larcher, Mathieu Mor- ´ lon, Cyril Zakka, Colin Raffel, Leandro Von Werra, and Thomas Wolf. SmolLM2: When smol goes big — data-centric training of a fully open small language model. In *Second Conference on Language Modeling*, 2025. URL [https:](https://openreview.net/forum?id=3JiCl2A14H) [//openreview.net/forum?id=3JiCl2A14H](https://openreview.net/forum?id=3JiCl2A14H). [99](#page-16-0) Daman Arora and Andrea Zanette. Training language models to reason efficiently. *Advances in Neural Information Processing Systems*, 38:60770–60808, 2026. [96](#page-13-0) Artificial Analysis. Artificial analysis openness index, 2026. URL [https://artificialanalysis.ai/evaluations/](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index) [artificial-analysis-openness-index](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index). Accessed: 2026-03-16. [97](#page-14-0) Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, et al. Qwen technical report, 2023. URL [https:](https://arxiv.org/abs/2309.16609) [//arxiv.org/abs/2309.16609](https://arxiv.org/abs/2309.16609). [97](#page-14-0) Tamay Besiroglu, Ege Erdil, Matthew Barnett, and Josh You. Chinchilla scaling: A replication attempt. *arXiv preprint arXiv:2404.10102*, 2024. [86](#page-3-0) Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In *International Conference on Machine Learning*, pages 2397–2430. PMLR, 2023. [98](#page-15-0) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. [87](#page-4-0) Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, and Haixu Tang. The janus interface: How fine-tuning in large language models amplifies the privacy risks. In *Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security*, pages 1285– 1299, 2024. [100](#page-17-0) Eric Czech, Zhiwei Xu, Yael Elmatad, Yixin Wang, and William Held. Problems with chinchilla approach 2: Systematic biases in isoflop parabola fits, 2026. URL <https://arxiv.org/abs/2603.22339>. [88](#page-5-0) Tim Dettmers and Luke Zettlemoyer. The case for 4-bit precision: k-bit inference scaling laws. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 7750–7774. PMLR, 23–29 Jul 2023. URL

<https://proceedings.mlr.press/v202/dettmers23a.html>. [95](#page-12-0)

{20}------------------------------------------------

- <span id="page-20-0"></span>Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3.int8(): 8-bit matrix multiplication for transformers at scale. In
- S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems*, volume 35, pages 30318–30332. Curran Associates, Inc., 2022. URL [https:](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf) [//proceedings.neurips.cc/paper\\_files/paper/2022/file/](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf) [c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf). [95](#page-12-0) Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, et al. Nemotron-climb: Clustering-based iterative data mixture bootstrapping for language model pre-training. *Advances in Neural Information Processing Systems*, 38, 2026. [96](#page-13-0) Yi Ding and Tianyao Shi. Sustainable llm serving: Environmental implications, challenges, and opportunities. In *2024 IEEE 15th International Green and Sustainable Computing Conference (IGSC)*, pages 37–38. IEEE, 2024. [99](#page-16-0) Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In *Proceedings of the 41st International Conference on Machine Learning*, ICML'24. JMLR.org, 2024. [99](#page-16-0) Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, et al. Olmo 3, 2025. URL [https://arxiv.](https://arxiv.org/abs/2512.13961) [org/abs/2512.13961](https://arxiv.org/abs/2512.13961). [97](#page-14-0) Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Prateek Sharma, Fan Chen, and Lei Jiang. Llmcarbon: Modeling the end-to-end carbon footprint of large language models. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, *International Conference on Learning Representations*, volume 2024, pages 24727–24741, 2024. URL [https:](https://proceedings.iclr.cc/paper_files/paper/2024/file/6b4ac044095525631df38e20919b45d2-Paper-Conference.pdf) [//proceedings.iclr.cc/paper\\_files/paper/2024/file/](https://proceedings.iclr.cc/paper_files/paper/2024/file/6b4ac044095525631df38e20919b45d2-Paper-Conference.pdf) [6b4ac044095525631df38e20919b45d2-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/6b4ac044095525631df38e20919b45d2-Paper-Conference.pdf). [99](#page-16-0) William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022. URL [http://jmlr.org/papers/](http://jmlr.org/papers/v23/21-0998.html) [v23/21-0998.html](http://jmlr.org/papers/v23/21-0998.html). [94](#page-11-0) Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, et al. The llama 3 herd of models, 2024. URL <https://arxiv.org/abs/2407.21783>. [87,](#page-4-0) [98](#page-15-0)

{21}------------------------------------------------

<span id="page-21-0"></span>Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In *The Twelfth International Conference on Learning Representations*, 2023. [94](#page-11-0) Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar Teodoro Mendes, Allie Del ´ Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sebastien ´ Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need, 2023. URL <https://arxiv.org/abs/2306.11644>. [96](#page-13-0) Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre. Training compute-optimal large language models. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. [86,](#page-3-0) [87,](#page-4-0) [88](#page-5-0) Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. [89,](#page-6-0) [92,](#page-9-0) [93](#page-10-0) Nidhal Jegham, Marwan Abdelatti, Chan Young Koh, Lassad Elmoubarki, and Abdeltawab Hendawi. How hungry is ai? benchmarking energy, water, and carbon footprint of llm inference, 2025. URL <https://arxiv.org/abs/2505.09598>. [99](#page-16-0) Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep ´ Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, ´ Thibaut Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed. Mixtral of ´ experts, 2024. URL <https://arxiv.org/abs/2401.04088>. [98](#page-15-0) Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL [https://arxiv.org/abs/2001.](https://arxiv.org/abs/2001.08361) [<sup>08361</sup>](https://arxiv.org/abs/2001.08361). [84,](#page-1-0) [85,](#page-2-0) [86](#page-3-0) Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, SOSP '23, page 611–626, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702297. doi: 10.1145/3600006.3613165. URL [https://doi.org/10.1145/3600006.](https://doi.org/10.1145/3600006.3613165) [<sup>3613165</sup>](https://doi.org/10.1145/3600006.3613165). [95](#page-12-0)

{22}------------------------------------------------

<span id="page-22-0"></span>Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In *International Conference on Machine Learning*, pages 19274–19286. PMLR, 2023. [96](#page-13-0) Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. Making ai less 'thirsty'. *Commun. ACM*, 68(7):54–61, June 2025. ISSN 0001-0782. doi: 10.1145/3724499. URL <https://doi.org/10.1145/3724499>. [99](#page-16-0) Yucheng Li, Frank Guerin, and Chenghua Lin. An open source data contamination report for large language models, 2024. URL [https://arxiv.org/abs/2310.](https://arxiv.org/abs/2310.17589) [<sup>17589</sup>](https://arxiv.org/abs/2310.17589). [100](#page-17-0) Vladislav Lialin, Vijeta Deshpande, and Anna Rumshisky. Scaling down to scale up: A guide to parameter-efficient fine-tuning. *arXiv preprint arXiv:2303.15647*, 2023. [92](#page-9-0) Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, et al. Deepseek-v3 technical report, 2025. URL [https:](https://arxiv.org/abs/2412.19437) [//arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437). [98](#page-15-0) Lan Liu, Yi Wang, Gaoyang Liu, Kai Peng, and Chen Wang. Membership inference attacks against machine learning models via prediction sensitivity. *IEEE Transactions on Dependable and Secure Computing*, 20(3):2341–2347, 2022. [100](#page-17-0) Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of bloom, a 176b parameter language model. *Journal of Machine Learning Research*, 24(253):1–15, 2023. URL [http://jmlr.org/papers/](http://jmlr.org/papers/v24/23-0069.html) [v24/23-0069.html](http://jmlr.org/papers/v24/23-0069.html). [99](#page-16-0) Junyu Luo, Bohan Wu, Xiao Luo, Zhiping Xiao, Yiqiao Jin, Rong-Cheng Tu, Nan Yin, Yifan Wang, Jingyang Yuan, Wei Ju, and Ming Zhang. A survey on efficient large language model training: From data-centric perspectives. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 30904–30920, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1493. URL [https://aclanthology.org/](https://aclanthology.org/2025.acl-long.1493/) [2025.acl-long.1493/](https://aclanthology.org/2025.acl-long.1493/). [96](#page-13-0) Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. *Advances in neural information processing systems*, 36: 21702–21720, 2023. [95](#page-12-0) Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. Llm dataset inference: Did you train on my dataset? *Advances in Neural Information Processing Systems*, 37:124069–124092, 2024. [100](#page-17-0)

{23}------------------------------------------------

<span id="page-23-0"></span>Ian R. McKenzie, Alexander Lyzhov, Michael Martin Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Xudong Shen, Joe Cavanagh, Andrew George Gritsevskiy, Derik Kauffman, Aaron T. Kirtland, Zhengping Zhou, Yuhui Zhang, Sicong Huang, Daniel Wurgaft, Max Weiss, Alexis Ross, Gabriel Recchia, Alisa Liu, Jiacheng Liu, Tom Tseng, Tomasz Korbak, Najoung Kim, Samuel R. Bowman, and Ethan Perez. Inverse scaling: When bigger isn't better. *Transactions on Machine Learning Research*, 2023. ISSN 2835-8856. URL <https://openreview.net/forum?id=DwgRm72GQF>. Featured Certification. [90,](#page-7-0) [91](#page-8-0) Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, and Mohammad Rastegari. Openelm: An efficient language model family with open training and inference framework, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2404.14619) [2404.14619](https://arxiv.org/abs/2404.14619). [98](#page-15-0) Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Riviere, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, ` Leonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, ´ et al. Gemma: Open models based on gemini research and technology, 2024. URL <https://arxiv.org/abs/2403.08295>. [97](#page-14-0) James A. Michaelov and Benjamin K. Bergen. Emergent inabilities? inverse scaling over the course of pretraining. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 14607–14615, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.973. URL [https:](https://aclanthology.org/2023.findings-emnlp.973/) [//aclanthology.org/2023.findings-emnlp.973/](https://aclanthology.org/2023.findings-emnlp.973/). [90](#page-7-0) Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. *arXiv preprint arXiv:1710.03740*, 2017. [95](#page-12-0) Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori ` Hashimoto. s1: Simple test-time scaling. *arXiv preprint arXiv:2501.19393*, 2025. [96,](#page-13-0) [99](#page-16-0) Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*, 2023. [91](#page-8-0) Miray Ozcan, Philipp Wiesner, Philipp Weiß, and Odej Kao. Quantifying the energy ¨ consumption and carbon emissions of llm inference via simulations. *arXiv preprint arXiv:2507.11417*, 2025. [99](#page-16-0) Elliot Paquette, Courtney Paquette, Lechao Xiao, and Jeffrey Pennington. 4+3 phases of compute-optimal neural scaling laws. In *The Thirty-eighth Annual Conference on* 

{24}------------------------------------------------

*Neural Information Processing Systems*, 2024. URL [https://openreview.](https://openreview.net/forum?id=aVSxwicpAk)

<span id="page-24-0"></span>[net/forum?id=aVSxwicpAk](https://openreview.net/forum?id=aVSxwicpAk). [88](#page-5-0) Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022. URL <https://arxiv.org/abs/2201.02177>. [91](#page-8-0) Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin Huang. Mobile edge intelligence for large language models: A contemporary survey. *IEEE Communications Surveys & Tutorials*, 27(6):3820–3860, 2025. [100](#page-17-0) Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. *arXiv preprint arXiv:2112.11446*, 2021. [87](#page-4-0) Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William Bergeron, Jeremy Kepner, Devesh Tiwari, and Vijay Gadepally. From words to watts: Benchmarking the energy costs of large language model inference. In *2023 IEEE high performance extreme computing conference (HPEC)*, pages 1–9. IEEE, 2023. [99](#page-16-0) Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel ´ Hesslow, Roman Castagne, Alexandra Sasha Luccioni, Franc¸ois Yvon, Matthias ´ Galle, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, et al. ´ Bloom: A 176b-parameter open-access multilingual language model, 2023. URL <https://arxiv.org/abs/2211.05100>. [98](#page-15-0) Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? *Advances in neural information processing systems*, 36: 55565–55581, 2023. [89,](#page-6-0) [90](#page-7-0) Rylan Schaeffer, Noam Levi, Andreas Kirsch, Theo Guenais, Brando Miranda, Elyas Obbad, and Sanmi Koyejo. Evaluating the robustness of chinchilla compute-optimal scaling. *arXiv preprint arXiv:2509.23963*, 2025. [86](#page-3-0) Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparselygated mixture-of-experts layer, 2017. URL [https://arxiv.org/abs/1701.](https://arxiv.org/abs/1701.06538) [<sup>06538</sup>](https://arxiv.org/abs/1701.06538). [94](#page-11-0) Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024. [99](#page-16-0) Kaya Stechly, Matthew Marquez, and Subbarao Kambhampati. GPT-4 doesn't know it's wrong: An analysis of iterative prompting for reasoning problems. In *NeurIPS 2023 Foundation Models for Decision Making Workshop*, 2023. URL [https://](https://openreview.net/forum?id=PMtZjDYB68) [openreview.net/forum?id=PMtZjDYB68](https://openreview.net/forum?id=PMtZjDYB68). [99](#page-16-0)

{25}------------------------------------------------

<span id="page-25-0"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozi ´ ere, Naman Goyal, Eric Hambro, Faisal ` Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023a. URL [https:](https://arxiv.org/abs/2302.13971) [//arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971). [87](#page-4-0) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b. [87](#page-4-0) Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will we run out of data? limits of LLM scaling based on human-generated data. In *Forty-first International Conference on Machine Learning*, 2024. URL <https://openreview.net/forum?id=ViZcgDQjyG>. [91](#page-8-0) Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, and Ryan Cotterell. Findings of the BabyLM challenge: Sample-efficient pretraining on developmentally plausible corpora. In Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, and Ryan Cotterell, editors, *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning*, pages 1–34, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.conll-babylm.1. URL [https:](https://aclanthology.org/2023.conll-babylm.1/) [//aclanthology.org/2023.conll-babylm.1/](https://aclanthology.org/2023.conll-babylm.1/). [97](#page-14-0) Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*, 2022. [89](#page-6-0) Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Gloria Chang, Fiona Aga, Jinshi Huang, Charles Bai, Michael Gschwind, Anurag Gupta, Myle Ott, Anastasia Melnikov, Salvatore Candido, David Brooks, Geeta Chauhan, Benjamin Lee, Hsien-Hsin Lee, Bugra Akyildiz, Maximilian Balandat, Joe Spisak, Ravi Jain, Mike Rabbat, and Kim Hazelwood. Sustainable ai: Environmental implications, challenges and opportunities. In D. Marculescu, Y. Chi, and C. Wu, editors, *Proceedings of Machine Learning and Systems*, volume 4, pages 795–813, 2022. URL [https://proceedings.mlsys.org/paper\\_files/paper/](https://proceedings.mlsys.org/paper_files/paper/2022/file/462211f67c7d858f663355eff93b745e-Paper.pdf) [2022/file/462211f67c7d858f663355eff93b745e-Paper.pdf](https://proceedings.mlsys.org/paper_files/paper/2022/file/462211f67c7d858f663355eff93b745e-Paper.pdf). [99](#page-16-0) Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 38087–38099. PMLR, 23–29 Jul 2023. URL <https://proceedings.mlr.press/v202/xiao23c.html>. [95](#page-12-0)

{26}------------------------------------------------

<span id="page-26-0"></span>Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. A survey on knowledge distillation of large language models. *arXiv preprint arXiv:2402.13116*, 2024. [94](#page-11-0) Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi, Xia Hu, and Zirui Liu. Understanding and mitigating numerical sources of nondeterminism in LLM inference. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025. URL [https://openreview.](https://openreview.net/forum?id=Q3qAsZAEZw) [net/forum?id=Q3qAsZAEZw](https://openreview.net/forum?id=Q3qAsZAEZw). [95](#page-12-0) Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, et al. Glm-5: from vibe coding to agentic engineering, 2026. URL <https://arxiv.org/abs/2602.15763>. [98](#page-15-0) Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Firat. When scaling meets LLM finetuning: The effect of data, model and finetuning method. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://](https://openreview.net/forum?id=5HCnKDeTws) [openreview.net/forum?id=5HCnKDeTws](https://openreview.net/forum?id=5HCnKDeTws). [88](#page-5-0) Danyang Zhang, Junhao Song, Ziqian Bi, Xinyuan Song, Yingfang Yuan, Tianyang Wang, Joe Yeong, and Junfeng Hao. Mixture of experts in large language models, 2025. URL <https://arxiv.org/abs/2507.11181>. [94](#page-11-0) Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36:55006–55021, 2023. [96](#page-13-0)

{27}------------------------------------------------

