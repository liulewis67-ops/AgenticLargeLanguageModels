<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 181-188; Marker split PDF pages 1-8. -->

{0}------------------------------------------------

# Chapter 8: Metalearning

#### What You Will Learn

By the end of this chapter, you will be able to:

- Recognize the different levels of learning used in LLM pipelines.
- Explain the connection between in-context learning, few-shot learning, and metalearning.
- Describe a metalearning system with an inner learning loop and an outer learning loop.
- Explain the moving-target problem and why dynamic metalearning can be difficult to stabilize.

# 8.1 Why Metalearning?

As we have seen, self-reflective LLM approaches can be powerful, but getting their training processes to converge can be challenging. In agentic LLM systems, learning can occur at multiple levels, sometimes concurrently. In machine learning, algorithms that learn at multiple levels are said to perform metalearning. Metalearning is often described as *learning to learn*—for example, learning which prompt works best for a particular task. The different learning levels can also interfere with one another and prevent learning from converging.

To understand why multi-level learning algorithms for LLMs sometimes fail, we will look more closely at metalearning and discuss ways to improve the stability of dynamic, feedback-rich approaches.

Metalearning refers to algorithms that learn across multiple levels. Such algorithms learn how to adapt so that they can solve new tasks, for which they have not been explicitly trained, using only a small number of examples.

{1}------------------------------------------------

### <span id="page-1-0"></span>8.1.1 Deep Learning Is Slow

Metalearning has a long history in machine learning (see, for example, [Schmidhuber, 1987](#page-7-0); [Hospedales et al., 2021](#page-6-0); [Huisman et al., 2021](#page-6-0)). It draws inspiration from human learning: people are often said to learn how to learn. Speed is another motivation for its development.

Conventional deep learning is slow. Neural networks for typical tasks can have millions or billions of parameters, and training them requires large datasets, often containing millions of examples. Training a network from scratch can take hours or months. Deep learning became practical in part because specialized parallel hardware, such as GPUs, became available [Krizhevsky et al., 2012](#page-7-0).

Researchers therefore began exploring few-shot learning: methods that achieve high accuracy after training on only a small number of examples.

#### 8.1.2 Achieving Few-Shot Learning

Some machine-learning algorithms are better suited to particular problems than others. Most training algorithms also have settings that can be tuned to improve performance, such as the learning rate, momentum, parameter initialization, or—when an algorithm has multiple levels—the choice of the underlying training algorithm. These settings are called *hyperparameters*; they govern how an algorithm learns. Can we learn good hyperparameters from data?

Different problems may require different hyperparameter settings. For example, a high learning rate or a local-search algorithm may work well on a smooth dataset, while a lower learning rate or a global-search algorithm may work better on a rugged one. The best configuration depends on the task, and that configuration can itself be learned. Learning how to configure an algorithm across tasks is called *metalearning*. By contrast, learning the model's ordinary parameters is called *base learning*.

Metalearning aims to make algorithms learn new tasks more quickly. Instead of applying a generic algorithm from scratch, we first use a set of related tasks to learn how to configure the learning algorithm for the kinds of tasks it will encounter.

Once we have learned a suitable configuration, we may be able to train on just a few examples and still achieve high accuracy on examples from a new task. Metalearning supports few-shot learning by starting from related knowledge rather than from scratch.

Metalearning consists of an outer (meta-training) loop around an inner (base-training) loop. In classical machine learning, outer-loop choices might include the learning rate or the learning algorithm itself; different algorithms can be evaluated across training tasks [Brazdil et al., 2022](#page-6-0). In deep learning, the initialization of the neural-network weights is often used [Finn et al., 2017](#page-6-0). In agentic LLM systems, we also encounter learning at different levels, a topic we will return to in the next chapter. The metalearning literature offers useful theory and insights for studying these systems.

{2}------------------------------------------------

<span id="page-2-0"></span>

## 8.2 Inner/Outer Loop

Like other supervised-learning approaches, metalearning has a *meta-training* phase, in which the system trains on a range of related tasks, and a *meta-test* phase, in which it is evaluated on a new, unseen task. The goal is to improve generalization by learning a hyperparameter setting for the training algorithm that works well across many training tasks.

A model's parameters are learned through an optimization loop. A loss function measures the difference between the model's predictions and the training examples; the learning algorithm updates the parameters to reduce that loss. The loop processes the dataset and adjusts the parameters with the aim of minimizing the total loss.

Learning to learn uses two loops. The *inner loop* optimizes the model parameters for examples from one task. The *outer loop* optimizes the algorithm's hyperparameters across a set of tasks in an application domain. In other words, the inner loop learns from examples within a task, while the outer loop learns how the system should learn across tasks.

Let base-learning task \(\mathcal{T}_i\) have dataset \(D_i\) and loss function \(\mathcal{L}_i\), so that \(\mathcal{T}_i=(D_i,\mathcal{L}_i)\). The dataset contains input-label pairs, \(D_i=\{(x_j,y_j)\}\), and is split into training and test sets: \(D_i=\{D_{i,\mathrm{train}},D_{i,\mathrm{test}}\}\). On each task's training data, the model \(\hat f_{\theta_i}(D_{i,\mathrm{train}})\) is estimated by minimizing the loss \(\mathcal{L}_i(\theta_i,D_{i,\mathrm{train}})\). The model \(\hat f\) is defined by an algorithm with hyperparameters \(\omega\). Depending on the algorithm, these may include the learning rate \(\alpha\), initial parameters \(\theta_0\), and other constants.

The inner loop optimizes the base parameters \(\theta_i\); the outer loop optimizes hyperparameters \(\omega\) so that the resulting model performs well across the base tasks. In simplified form:

\[
\omega^* = \arg\min_{\omega}\; \mathcal{L}\!\left(\arg\min_{\theta_i}\mathcal{L}_{\omega}(\theta_i,D_{i,\mathrm{train}})\right).
\]

The inner optimization finds parameters \(\theta_i\) for each task's training data. The outer optimization evaluates the resulting learning procedure across tasks and selects \(\omega\). The result is an optimized set of hyperparameters, \(\omega^*\).

After metalearning, the tuned algorithm is expected to work well across a set of tasks in its application domain. We can then expect it to generalize to a new task after seeing only a few examples, achieving high accuracy on unseen examples.

{3}------------------------------------------------

<span id="page-3-0"></span>
### Transfer Learning

Metalearning is closely related to transfer learning. Transfer learning pretrains a neural network on one task and transfers its weights to a subsequent fine-tuning phase. It can therefore be viewed as one iteration of a network-initialization-based metalearning approach: metalearning with a single base-learning task.

# 8.3 LLMs and Metalearning

Metalearning appears in many parts of LLM development. One common approach initializes a model with previously learned knowledge so that it can learn a related task more quickly. This is transfer learning, used in the pretraining and fine-tuning phases of a standard LLM pipeline. Repeated fine-tuning can be viewed as a metalearning process, and LLMs provide a prominent example of a pipeline designed to support few-shot learning. In-context learning was another major breakthrough [Brown et al., 2020](#page-6-0).

In agentic LLM systems, few-shot adaptation appears in other forms as well. We will consider both static and dynamic metalearning. In static metalearning, the weights in one loop are not changed by the results of another loop. In dynamic metalearning, the loops can influence one another.

#### 8.3.1 Static LLM Metalearning

We begin with several examples of static metalearning in LLMs.

#### 8.3.1.1 Pretraining Pipeline

The standard LLM training pipeline is an elaborate, multi-phase transfer-learning scheme. It can achieve strong zero-shot performance: for example, it may answer a familiar question such as *Where was Mozart born?* using knowledge acquired during pretraining and fine-tuning.

The phases in the pipeline are inner optimization loops. Decisions about which weights to transfer and how to fine-tune in later phases can be viewed as outer-loop hyperparameter choices.

This is static metalearning because, once a phase has finished, its weights are not changed by the results of later phases. The weights remain fixed between phases, and training proceeds sequentially.

#### 8.3.1.2 In-Context Learning

Beyond zero-shot learning, in-context learning is a form of few-shot learning: the user supplies a small number of examples in the prompt, and the LLM uses them to produce a better answer. The pretrained model is the inner loop; the examples in the prompt guide the outer loop.

{4}------------------------------------------------

This is static metalearning because the examples in the prompt do not change the model's weights.

#### 8.3.1.3 Chain of Thought

Ordinary in-context learning uses *facts* as few-shot examples and focuses on content. Chain-of-thought prompting uses *steps in a process* as examples and focuses on the reasoning process.

Again, the pretrained model is the inner loop, the prompt examples guide the outer loop, and the model's weights remain fixed.

#### 8.3.1.4 Static Metalearning Is Stable

The standard LLM pipeline consists of sequential transfer-learning phases (inner loops). Once an inner training loop has converged, that phase is not run again. Fine-tuning results are not fed back into the pretrainer, and the pretrainer is not rerun with hyperparameters adjusted according to fine-tuning results. Because the phases do not feed into one another, they cannot destabilize one another through feedback.

Dynamic metalearning differs in that its iterations can influence one another.

### 8.3.2 Dynamic LLM Metalearning

Agentic LLM systems use several forms of dynamic metalearning. The iterations may run concurrently and destabilize one another's convergence.

In previous chapters, we discussed self-reflection and RLVR. In these approaches, an outer loop can change an inner loop, and the result of one inner loop can affect another. This creates a moving training target and can make training diverge. Results may also be brittle: small changes in inputs or hyperparameters—such as random seeds, prompt wording, or the duration of metalearning—can produce large differences. In addition, using the same model for different roles, such as generation and evaluation, may amplify the base model's biases and hallucinations.

The following sections examine examples of dynamic metalearning in agentic LLM systems.

#### 8.3.2.1 Self-Reflection

In self-reflection (Section 6.2.4.2), an agentic harness manages prompts and calls the LLM in different roles, such as evaluating a state or refining an answer.

In metalearning terms, the *model* is the inner learning loop and the *harness* is the outer loop. Although the model's weights remain fixed, the harness uses the model's answers to construct new prompts. The resulting feedback can make the outer-loop iterations interact in nonlinear and hard-to-predict ways.

When the same model is used in different roles, its imperfections—including bias and hallucinations—may also be amplified.

{5}------------------------------------------------

#### <span id="page-5-0"></span>8.3.2.2 RLVR

Reinforcement learning with verifiable rewards (RLVR; see Section 6.2.4.3) is a popular training approach for coding and mathematics. Reasoning results generated at inference time are used to fine-tune the model during training.

In metalearning terms, fine-tuning is the inner loop and inference-time reasoning is the outer loop. RLVR feeds outer-loop results into the inner-loop training phase, coupling the two learning processes. Care is needed to avoid amplifying inference-time errors by treating them as ground truth in a repeated fine-tuning and inference cycle.

# 8.4 Stabilizing Dynamic Metalearning

Training can become unstable when interference between inner and outer loops is under-damped. Oscillations between the loops may grow instead of fading away.

One way to mitigate this problem is to add damping or let the loops change at different rates. For example, we can reduce the learning rate of one loop or keep some inner-loop parameters fixed for longer than others. Another option is to use models of different types or sizes, as in a mixture-of-experts model [Jiang et al., 2024](#page-6-0). A student–teacher setup deliberately assigns different roles to a stronger and a weaker model. Using different models may reduce the amplification of hallucinations and interference between loops.

More generally, when metalearning behaves poorly, we can introduce diversity to encourage exploration and reduce greedy behavior that leads to local optima or divergent training. Developing useful remedies often takes experimentation and practice. Debugging agentic LLM systems can be challenging.

# 8.5 Concluding Remarks

LLM training can be understood as a form of metalearning. Standard LLM training uses transfer learning, while inference-time reasoning and training in agentic LLM systems draw on further metalearning ideas. Dynamic methods such as self-reflection are powerful, but feedback between inner and outer loops can destabilize training. This can happen even in agents that previously behaved reliably.

When a trained model is placed in an agent that interacts with the world, and the agent is also trained using feedback from those interactions, the model can be viewed as the base learner (inner loop) and the interaction process as the metalearner (outer loop). The same cautions that apply to metalearning within a training pipeline also apply outside it. In real-world interactions, however, errors may have more serious consequences.

Online machine learning, including deep reinforcement learning, also faces unstable training and moving targets. <span id="page-6-0"></span>Deep reinforcement learning has introduced stabilization techniques such as replay buffers and target networks [Mnih et al., 2015](#page-7-0); Rainbow is another example [Hessel et al., 2018]. These methods often decouple the timing of two learning processes, for example by slowing one of them. A similar idea appears in Hierarchical Reasoning Models (HRM) [Wang et al., 2025](#page-7-0), which use a two-level metalearning approach and stabilize learning by operating at two different speeds.

Understanding such stabilization methods is useful when designing agentic LLM systems. The next part surveys the different ways agentic LLMs interact with the world, from static to dynamic.
# Bibliography

Pavel Brazdil, Jan N Van Rijn, Carlos Soares, and Joaquin Vanschoren. *Metalearning: Applications to automated machine learning and data mining*. Springer Nature, 2022. [182](#page-1-0) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. [184](#page-3-0) Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *International conference on machine learning*, pages 1126–1135. PMLR, 2017. [183](#page-2-0) Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018. 187 Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Metalearning in neural networks: A survey. *IEEE transactions on pattern analysis and machine intelligence*, 44(9):5149–5169, 2021. [182](#page-1-0) Mike Huisman, Jan N Van Rijn, and Aske Plaat. A survey of deep meta-learning. *Artificial Intelligence Review*, 54(6):4483–4541, 2021. [182](#page-1-0) Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep ´ Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, ´ Thibaut Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed. Mixtral of ´ experts, 2024. URL <https://arxiv.org/abs/2401.04088>. [186](#page-5-0)

{7}------------------------------------------------

<span id="page-7-0"></span>Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012. [182](#page-1-0) Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533, 2015. [187](#page-6-0) Jurgen Schmidhuber. ¨ *Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook*. PhD thesis, Technische Universitat¨ Munchen, 1987. ¨ [182](#page-1-0) Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model. *arXiv preprint arXiv:2506.21734*, 2025. [187](#page-6-0)