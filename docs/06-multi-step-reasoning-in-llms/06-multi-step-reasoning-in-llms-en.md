<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 125-150; Marker split PDF pages 1-26. -->



{0}------------------------------------------------

## <span id="page-0-0"></span>Chapter 6

# Multi-step Reasoning in LLMs

#### What You Will Learn

In this chapter, you will learn how language models use *multi-step reasoning* to solve more complex problems. By the end, you will be able to:

- Explain *chain of thought*: how in-context learning can guide an LLM through intermediate steps and improve answers to complex queries.
- Describe how reasoning methods have expanded LLM capabilities and why reasoning is important for LLM agents acting in the real world.
- Build a first *agent harness*: a small program that calls an LLM with different prompts and manages its prompts and responses.
- Explain *self-reflection*: how a harness can call an LLM in different roles, such as evaluating an answer or generating subproblems, to improve its output.
- Explain the role of *memory*: how a harness tracks an agent's state and uses it to manage prompts.
- Describe *reinforcement learning with verifiable rewards*, in which reasoning results from inference-time tasks are used for fine-tuning.
## 6.1 In-Context Reasoning

As discussed in Chapter 3, large language models can learn from examples supplied in a prompt, a capability known as *in-context learning* (ICL), or *prompt learning* [\[Brown et al.,](#page-19-0) [2020\]](#page-19-0). This section introduces ICL, common reasoning benchmarks, and chain-of-thought methods for multi-step reasoning.

{1}------------------------------------------------

#### <span id="page-1-0"></span>6.1.1 In-Context Learning

In-context learning [\[Dong et al.,](#page-20-0) [2022\]](#page-20-0) takes place at inference time, not during training. The model receives examples in its prompt and uses them to answer a query. These examples can help a decoder model draw on relevant knowledge learned during pretraining. A model may perform a task accurately with only a few examples, which is why ICL is often described as *few-shot learning*. This depends on the model having acquired broad knowledge during training.

**Learning, not training.** Unlike pretraining or fine-tuning, ICL does not change the Transformer's parameters. It is therefore not parameter training, although the model learns from examples in the prompt. This also means that information from one LLM call is not retained in the parameters for the next call. To share results between calls, they must be added to a later prompt as examples or other context.

**Context window.** The part of an LLM that receives the prompt is called its *context window*. The prompt and all examples used to guide the answer must fit within it. Since context windows have finite capacity, only a limited amount of information can be provided this way. ICL can therefore be useful for learning from a few examples but is poorly suited to retaining information over long periods or across separate calls.

**Example.** An ICL prompt combines a few demonstrations with a query [\[Liu et al.,](#page-22-0) [2023\]](#page-22-0). The model then generates an answer in a forward pass. For emotion recognition, a prompt might pair *I missed the bus today* with *I felt so [ ]*, prompting the model to complete it with *bad*. For translation, the prompt could append *French: [ ]* to the same sentence. The examples provide context that helps the model infer the intended task.

**Few-shot learning and the learning ideal.** ICL works when a model has enough prior knowledge to generalize from the few examples in a prompt. The user supplies relevant examples to help the model select the background knowledge

{2}------------------------------------------------

<span id="page-2-0"></span>needed to answer the instruction. A prompt with a few examples enables few-shot learning; a prompt containing only an instruction and no examples enables zero-shot learning.

ICL attracted considerable interest as a form of few-shot learning. In traditional machine learning, a model $f$ is trained on a dataset, and complex tasks often require many examples and costly training. ICL can instead use a few examples and a single forward pass through a decoder model, relying on computation already spent during pretraining. The prospect of inexpensive generalization from a few examples has long been an ideal in machine learning.

**Practical considerations and prompt engineering.** ICL is convenient: users can try it with a commercial cloud model from a smartphone, without having to train a large model themselves. However, results can be sensitive to how the prompt is written and which examples it contains. This brittleness led to the field of *prompt engineering*, which studies how to formulate prompts that help LLMs perform the intended task [\[Radford et al.,](#page-22-0) [2019,](#page-22-0) [Wei et al.,](#page-24-0) [2022a,](#page-24-0) [Giray,](#page-21-0) [2023,](#page-21-0) [Sahoo et al.,](#page-23-0) [2024\]](#page-23-0).

Most LLM applications also use hidden system prompts to assign the model a role or set behavioral constraints. In chatbots, the context may additionally include a short history of previous interactions or search results to support a more coherent conversation.

ICL has prompted further research, including multi-step reasoning—the subject of this chapter. LLMs have historically struggled with reasoning tasks such as simple arithmetic. The following sections introduce benchmarks for these tasks and then examine an ICL method that substantially improved performance.

#### 6.1.2 Multi-step Reasoning Benchmarks

Progress in artificial intelligence is often measured with benchmarks. Research on multi-step LLM reasoning began with elementary-school math word problems. The GSM8K benchmark was created to study this type of reasoning [\[Cobbe et al.,](#page-20-0) [2021\]](#page-20-0). It contains 8,500 human-written problems and solutions, designed to be high-quality, diverse, and moderately difficult. Before chain-of-thought prompting, language models struggled to perform well on this dataset. Here is a typical example:

*Problem:* Romeo bakes four trays of cookies in a week, with two dozen cookies per tray. If the cookies are shared equally among 16 people, how many cookies does each person get?

{3}------------------------------------------------

<span id="page-3-0"></span>*Answer:* $4 \times 2 \times 12 / 16 = 6$.

Although GSM8K may look easy, it proved difficult for LLMs. When the benchmark was introduced, the state-of-the-art GPT-3 (175B) achieved only about 15% accuracy [\[Wei et al.,](#page-24-0) [2022b\]](#page-24-0). Related math word-problem benchmarks include SVAMP [\[Patel et al.,](#page-22-0) [2021\]], ASDiv [\[Miao et al.,](#page-22-0) [2021\]], AQuA [\[Ling et al.,](#page-21-0) [2017\]], and MAWPS [\[Koncel-Kedziorski et al.,](#page-21-0) [2016\]](#page-21-0).

Other benchmarks assess reasoning in additional domains, including advanced mathematics (AIME [\[Codeforces,](#page-20-0) [2024\]](#page-20-0)), code comprehension and translation between programming languages (Spider [\[Yu et al.,](#page-25-0) [2018\]], Rozière et al. [\[2020\]](#page-23-0)), embodied tasks such as household robotics and cooking (ALFWorld [\[Shridhar et al.,](#page-23-0) [2020\]], Ahn et al. [\[2022\]](#page-19-0)), puzzle solving (Game24 [\[Yao et al.,](#page-25-0) [2024\]](#page-25-0)), creative writing [\[Yao et al.,](#page-25-0) [2024\]](#page-25-0), board games (checkmate problems [\[Yang et al.,](#page-25-0) [2024\]](#page-25-0)) and Minecraft [\[Fan et al.,](#page-20-0) [2022\]](#page-20-0), and web navigation (WebShop [\[Yao et al.,](#page-25-0) [2022a\]](#page-25-0)). Other work studies games and planning problems [\[Ruoss et al.,](#page-23-0) [2024,](#page-23-0) [Paglieri et al.,](#page-22-0) [2025,](#page-22-0) [Wen et al.,](#page-24-0) [2025a\]](#page-24-0).

Benchmarks are intended to make it possible to compare approaches, but results can be difficult to compare when studies use different experimental conditions. See [Kamoi et al.](#page-21-0) [\[2024\]](#page-21-0) and [Shojaee et al.](#page-23-0) [\[2025\]](#page-23-0) for critical discussions of challenges in evaluating LLM reasoning.

#### 6.1.3 Chain-of-Thought Prompting

Around 2022, standard LLMs performed poorly on simple math word problems. Simply increasing model size, the approach discussed in Chapter 4, did not solve the problem [\[Rae et al.,](#page-22-0) [2021\]](#page-22-0). Researchers therefore explored a different approach.

A Transformer-based decoder generates one token at a time: LLMs are next-token predictors. When prompted for an answer, a model may respond immediately, but mathematical reasoning usually involves several steps. For example, solving the cookie problem requires calculating $4 \times 2 \times 12 / 16 = 6$. Instead of asking the model only for the answer, we can prompt it to work through a sequence of intermediate steps, much as a student is taught to break a complex problem into smaller steps and explain the reasoning: *Don't just tell me the answer; show how you arrived at it.* We should guide the model to
{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Diagram_2.jpeg)

Figure 6.1: Different chain-of-thought (CoT) prompting techniques [\[Wei et al.,](#page-24-0) [2022b,](#page-24-0) [Kojima et al.,](#page-21-0) [2022,](#page-21-0) [Gao et al.,](#page-20-0) [2023,](#page-20-0) [Plaat et al.,](#page-22-0) [2025\]](#page-22-0). Prompts are shown above the corresponding answers.

The language model should be guided to generate intermediate steps explicitly and then combine them. Its final answer should include those steps and arrive at a correct result.

A few-shot prompt should therefore include examples of intermediate reasoning, not just examples of final answers. [Wei et al.](#page-24-0) [\[2022b\]](#page-24-0), among others, studied this approach and introduced *chain of thought*. Figure 6.1 illustrates CoT and related methods.

One way to elicit intermediate reasoning is to write a prompt that demonstrates the steps for a task—for example, counting the balls in a container or working out how many pieces a pie was cut into. When [Wei et al.](#page-24-0) [\[2022b\]](#page-24-0) prompted an LLM to restate information from a question as intermediate reasoning steps, it performed much better than when asked to produce only a final answer. The paper's example is shown in Figure 6.1. With longer few-shot CoT examples, the model tends to follow the demonstrated reasoning pattern. On GSM8K, CoT increased accuracy from 16% to 47%.

This idea—that prompts can guide an LLM through step-by-step reasoning—sparked substantial research into multi-step reasoning. CoT has since become a standard prompting approach.

In the original work, researchers hand-wrote prompts for different problem types. [Kojima et al.](#page-21-0) [\[2022\]](#page-21-0) showed that adding the task-independent phrase *Let's think step by step* also improved performance on many problems. Because the phrase contains no task-specific information, this is a form of zero-shot learning. Figure 6.1 compares

{5}------------------------------------------------

<span id="page-5-0"></span>these approaches in its second and third columns. Experiments also found gains on arithmetic, symbolic, and logical reasoning benchmarks. On GSM8K, adding self-consistency (discussed next) with the newer PaLM model raised accuracy to 70% [\[Wang et al.,](#page-24-0) [2022\]](#page-24-0).

The CoT idea also draws on Scratchpad [\[Nye et al.,](#page-22-0) [2021\]](#page-22-0), earlier work that generated natural-language steps for arithmetic reasoning [\[Ling et al.,](#page-21-0) [2017,](#page-21-0) [Cobbe et al.,](#page-20-0) [2021\]](#page-20-0), and research using formal languages for reasoning [\[Roy and Roth,](#page-23-0) [2016,](#page-23-0) [Chiang and Chen,](#page-20-0) [2018,](#page-20-0) [Amini et al.,](#page-19-0) [2019,](#page-19-0) [Chen et al.,](#page-20-0) [2019\]](#page-20-0).

#### 6.1.3.1 Self-Evaluation

LLMs can make mistakes, and errors can compound across a sequence of reasoning steps. Evaluating the model and its intermediate steps can help limit this accumulation.

One approach is to have an LLM evaluate its own reasoning. Multi-step CoT makes a model more vulnerable to accumulated errors [\[Weng et al.,](#page-24-0) [2022,](#page-24-0) [Xiao et al.,](#page-24-0) [2023\]](#page-24-0). [Weng et al.](#page-24-0) [\[2022\]](#page-24-0) describe *self-verification*: the model uses its CoT conclusion as a condition for solving the original problem again, then compares the resulting answer with its conclusion. It can also generate variations of a candidate conclusion and select the one most consistent with the original problem.

A related approach, *self-consistency*, uses an ensemble of sampled reasoning paths [\[Wang et al.,](#page-24-0) [2022\]](#page-24-0). An ensemble combines multiple predictions to produce a stronger result [\[Sagi and Rokach,](#page-23-0) [2018,](#page-23-0) [Breiman,](#page-19-0) [2001\]](#page-19-0). Standard greedy decoding follows a single path by repeatedly choosing the token with the highest logit. Self-consistency instead samples diverse reasoning paths and selects the final answer that appears most often. This works because many reasoning problems can be solved through different valid paths. Figure [6.2](#page-6-0) illustrates the method.

Self-consistency has been evaluated on arithmetic, commonsense, and symbolic reasoning with models including GPT-3. It typically adds 10–20 percentage points to CoT performance and is often combined with CoT in modern LLMs.

{6}------------------------------------------------

<span id="page-6-0"></span>![](assets/_page_6_Diagram_2.jpeg)

Figure 6.2: Self-consistency, adapted from [\[Wang et al.,](#page-24-0) [2022\]](#page-24-0), illustrating majority voting.
{7}------------------------------------------------

#### <span id="page-7-0"></span>6.1.3.2 Code Generation for Validation

Another way to improve the reliability of multi-step reasoning is to use a formal language instead of natural language. Formal languages, such as Python or mathematical logic, are less ambiguous.

LLMs can generate code from natural-language prompts. Codex, introduced by [Chen et al.](#page-19-0) [\[2021\]](#page-19-0), was trained on publicly available code from GitHub and can generate programs from natural-language descriptions, including comments. This has made formal languages a popular tool for reasoning: an LLM can express steps in Python and have a compiler, debugger, or interpreter check them.

This resembles the way human programmers work: write code, execute it, inspect errors, and use the feedback to improve it. *Self-debugging* follows this cycle of code generation, execution, and explanation [\[Chen et al.,](#page-20-0) [2023\]](#page-20-0). The model can inspect execution results to identify mistakes in its code and explain the code in natural language.

Another method, *Program-Aided Language Models* (PAL), has the LLM generate Python code that an interpreter then executes [\[Gao et al.,](#page-20-0) [2023\]](#page-20-0). The problem is decomposed into small code snippets that the interpreter can run. The fourth column of Figure [6.1](#page-4-0) illustrates PAL. The interpreter checks the computation, leaving the LLM responsible for translating the natural-language problem into executable steps. Experiments across 13 mathematical, symbolic, and algorithmic reasoning tasks showed that PAL could outperform much larger models of the time, reaching 72% on GSM8K and 93% on Penguins [\[Gao et al.](#page-20-0) [2023\]](#page-20-0).
#### 6.1.3.3 Evolving Code-Generation Prompts

Research on formal-language verification has led to methods that automatically generate working code tailored to specific problems, without human feedback. [Romera-Paredes et al. (2024)](#page-23-0) introduced FunSearch, which combines formal methods and LLMs to support mathematical discovery and code generation. It maintains populations of candidate programs and evaluates them against a function derived from the problem specification. After scoring programs generated from an initial prompt, FunSearch builds a new prompt from selected candidates and asks the LLM to produce further programs.

Another approach uses evolutionary computation to generate and optimize evolutionary algorithms. [van Stein and Back (2024)](#page-23-0) introduced LLaMEA (Large Language Model Evolutionary Algorithm), which uses LLMs to design and optimize algorithms.

{8}------------------------------------------------

<span id="page-8-0"></span>LLaMEA generates initial algorithm structures and improves them through mutation and selection. The authors report an 11-percentage-point improvement on the Black-Box Optimization Benchmark (BBOB) [Hansen and Ros (2010)](#page-21-0).

## 6.2 Self-Reflection and State

In-context learning is a powerful few-shot method. When examples include reasoning steps, they can further improve an LLM’s problem-solving performance.

This section examines limits of in-context learning and ways to extend them. It first considers state and memory: how state can be passed between separate LLM calls, what longer context windows make possible, and how well LLMs solve problems that require them to track and update state.

#### 6.2.1 LLMs as Memoryless Functions

In-context few-shot learning uses the prompt to condition an LLM’s retrieval of learned information. The transformer produces the next token based on its parameters, often denoted by $(K,V)$. With chain-of-thought prompts, the model is also conditioned on the example reasoning steps included in the prompt.

In this setting, all information supplied to the LLM is part of the prompt. We can describe the model as a function $f$ that maps a prompt $x$ to an answer $y$:

$f(x) \rightarrow y$.

In principle, an LLM does not retain information between separate calls: its parameters do not change, and it has no persistent global variables that store the interaction history. If a follow-up request depends on state from an earlier call, that state must be included in the new prompt $x'$. The model itself does not remember its previous answer.

Repeating the full conversation in every new question would be cumbersome. Chatbots address this by including earlier turns in the prompt, often through system-managed context. The underlying model still receives that history as input each time; it does not retain conversational state across calls by itself. LLMs can produce associative next-token responses from the supplied prompt, but they are not general-purpose stateful programs that reliably remember values and update variables across calls.

{9}------------------------------------------------

<span id="page-9-0"></span>![](assets/_page_9_Picture_2.jpeg)

Figure 6.3: Tic-tac-toe is a simple board game in which keeping track of state matters.

#### 6.2.2 Retrieval Augmentation and Long Context Windows

The memory limitation goes beyond separate calls. All information needed to answer a query must fit in the context window, whose finite size limits how much can be provided at once. This becomes a challenge when summarizing a long text such as a book. Section 20.2.1 also discusses in-context reinforcement-learning methods that store complete episode histories in the context window. These cases motivate methods for supplying more relevant information to an LLM.

One early way to expand the information available at inference time is retrieval-augmented generation (RAG). RAG lets an LLM retrieve external information, for example from a company database or Wikipedia [Lewis et al. (2020)](#page-21-0). It can also help reduce factual errors and provide access to time-sensitive information such as financial or logistics data. Chapter 12 discusses RAG in more detail.

Longer context windows allow models to accept longer texts. With the standard Transformer architecture, computation grows quadratically with context length, so researchers are also developing linear-complexity architectures [Ren et al. (2024)](#page-22-0). Other approaches compress context, for example through semantic compression [Fei et al. (2024)](#page-20-0), chunking [Wang et al. (2023); Zhou et al. (2025)](#page-25-0), or recursive processing [Zhang et al. (2025)](#page-25-0). New memory architectures for LLMs remain an active research area [Yan et al. (2025)](#page-24-0).

#### 6.2.3 LLMs and Stateful Problems

LLMs often struggle with problems that require accurate state tracking [Wong et al. (2025)](#page-24-0). This includes planning tasks in which the current state must be remembered, even in simple board games such as the one in Figure 6.3. A model may confuse board states and give plausible-sounding moves that would be valid in familiar games but are wrong for the actual position [Paglieri et al. (2025); Ruoss et al. (2024)](#page-23-0).

Researchers have explored ways to improve reasoning over state, often drawing on reinforcement learning and planning algorithms [Wong et al. (2026)](#page-24-0). The memoryless nature of LLMs also contributes to a problem known as the *knowing-doing gap*. Other methods for stateful problems are discussed in Chapter 20.

{10}------------------------------------------------

<span id="page-10-0"></span>**Knowing-doing gap.** Beyond the lack of state between separate calls, LLMs can also fail to execute a task correctly even when they can describe the right procedure.

This gap describes cases in which an LLM can produce a correct textual solution—or even the right algorithm—but cannot reliably carry it out [Paglieri et al. (2025)](#page-22-0). Asked how to solve a problem, a model may provide correct executable Python code. Asked to solve the problem itself, however, it may fail to follow that algorithm. Errors often appear as the values of variables change from step to step [Paglieri et al. (2025); Ruoss et al. (2024)](#page-23-0). Faithfully executing an algorithm and tracking register values are not what attention-based Transformers were designed to do; they were designed to model language using associative patterns [La Malfa et al. (2024)](#page-21-0).<sup>1</sup>

A related state-management failure can occur when an LLM plays a board game against itself: pieces may appear or disappear between moves. The model may choose a move that would be reasonable in another position but does not fit the current board, then provide a plausible-sounding explanation. LLMs can lose or mix up individual values as a variable changes state. Adding instructions to track each value carefully may help, but it is cumbersome and error-prone [La Malfa et al. (2024)](#page-21-0).

Both the knowing-doing gap and the disappearing-pieces problem can be avoided by using an external interpreter to execute an algorithm generated by the LLM. Some commercial chatbots use this approach. A Python interpreter executes code accurately and more efficiently than asking an LLM to simulate each operation.

Chapter 20 discusses other approaches to state and memory in Transformer-based systems.

#### 6.2.4 Agentic Harnesses for Prompt Improvement

External code execution is one way to manage state. Another is to add small external algorithms that explicitly manage state and prompts. Such a control algorithm is called a *harness*.

<sup>1</sup>People can make similar mistakes: we may describe long division correctly but still miscalculate $12345 / 42$. Computers are good at this kind of exact arithmetic; people and LLMs can make mistakes.

{11}------------------------------------------------

<span id="page-11-0"></span>A harness can call the same LLM in different roles, using separate prompts for tasks such as generation or state evaluation. This section begins with search-based prompt management.

When an LLM is asked to critique or evaluate its own output, the approach is often called *self-reflection*. The term is somewhat anthropomorphic; whether the reflection is performed by the LLM or by the surrounding harness is open to debate.

#### 6.2.4.1 Search-Based Prompt Optimization

Search-based inference-time methods go beyond chain-of-thought prompting. A chain of thought follows one line of reasoning, with each step leading to a single next step. Search-based methods explore several alternatives at each step, forming a tree of possibilities.

Exploring such a tree requires tracking which steps have been visited, which answer currently looks best, and which branches remain to be explored. Conventional algorithms, such as small Python programs, can maintain this state more reliably than an LLM. An external algorithm can build a search tree of alternative steps and look ahead several steps (see Figure [6.6](#page-13-0) on page [138](#page-13-0)). It calls the LLM as a subroutine for tasks such as proposing actions or evaluating outcomes. The algorithm can then backtrack and try alternatives. This external controller tracks state and manages prompts; it is the harness around the LLM.

We begin with depth-first and breadth-first search [Cormen et al. (2022)](#page-20-0). Tree of Thoughts (ToT) uses these strategies to explore different reasoning paths dynamically [Yao et al. (2024)](#page-25-0). In ToT, the LLM proposes candidate thoughts and evaluates them through separate prompts. The external search procedure combines these steps to explore possible reasoning paths, look ahead, and backtrack (Figures [6.4](#page-12-0) and [6.5](#page-12-0)).

ToT has been applied to planning puzzles such as the Game of 24. Given four numbers, the task is to combine them with arithmetic operations (+, −, ×, ÷) to make 24. For example, with 4, 9, 10, and 13, one solution is $(10-4) \times (13-9)=24$. ToT first uses a *propose* prompt to generate possible next steps. The model then evaluates those candidates using an evaluation prompt that can include chain-of-thought examples. Figure [6.5](#page-12-0) shows an example prompt sequence from the original paper.

{12}------------------------------------------------

<span id="page-12-0"></span>Figure 6.4: Tree of Thoughts: breadth-first search (left) and depth-first search (right). The arrows indicate prompts to the LLM. Adapted from [Yao et al. (2024)](#page-25-0).

![](assets/_page_12_Diagram_4.jpeg)

Figure 6.5: Example Tree of Thoughts prompts for the Game of 24 [Yao et al. (2024)](#page-25-0).

{13}------------------------------------------------

<span id="page-13-0"></span>![](assets/_page_13_Diagram_2.jpeg)

Figure 6.6: Reasoning structures for (b) chain of thought, (c) self-consistency, and (d) Tree of Thoughts.

The authors compare ToT with linear chain-of-thought and self-consistency on the Game of 24, creative writing, and mini crossword tasks. ToT achieved 74% accuracy on the Game of 24; the other tasks were evaluated qualitatively. Figure 6.6 illustrates the different reasoning structures.<sup>3</sup>


<span id="page-14-0"></span>

#### 6.2.4.2 Self-Reflection

![](assets/_page_14_Diagram_2.jpeg)

Figure 6.7: Self-Refine architecture [Madaan et al. (2023)](#page-22-0).

These approaches use LLMs in different roles. Some draw on ideas from reinforcement learning to help a model improve its outputs.

**Self-Refine.** The first approach is Self-Refine [Madaan et al. (2023)](#page-22-0). The LLM generates an initial response, critiques it, and then revises it based on that feedback. The process uses three prompts—for generation, feedback, and refinement—and follows a greedy chain of reasoning: it tries one successor at a time rather than exploring a tree as ToT does. The authors report improvements over ordinary chain-of-thought prompting on tasks such as dialogue generation, code optimization, and constrained generation.

**ReAct.** A broader approach considers multiple possible actions in each state. ReAct combines reasoning traces with actions and observations [Yao et al. (2022b)](#page-25-0). Its framework can use external knowledge sources such as Wikipedia and can be adapted through fine-tuning or hand-written prompts. These mechanisms can ground responses and help reduce errors and hallucinations. On the ALFWorld and WebShop interactive decision-making benchmarks, ReAct achieved absolute success-rate gains of 34 and 10 percentage points, respectively, over the compared imitation-learning and reinforcement-learning methods.

**Reflexion.** Reflexion extends this line of work by having agents learn from failures and improve their results through reflection [Shinn et al. (2024)](#page-23-0).<sup>5</sup> It uses three LLM roles: an actor, an evaluator, and a reflector. The harness stores explicit memory between calls: short-term memory supports trajectory generation, while long-term memory stores experience. This external memory helps address the state-management limits of a standard LLM. In Reflexion, (1) the actor generates text and actions, (2) the evaluator scores the actor’s outputs, and (3) the reflector produces verbal feedback to help the actor improve.

<sup>4</sup><https://github.com/ysymyth/ReAct>

<sup>5</sup><https://github.com/noahshinn/reflexion>

{15}------------------------------------------------

<span id="page-15-0"></span>![](assets/_page_15_Diagram_2.jpeg)

Figure 6.8: Reflexion architecture [Shinn et al. (2024)](#page-23-0), which resembles the agent–environment structure used in reinforcement learning.

The actor can use chain-of-thought prompting or ReAct. Reflexion has been evaluated on decision-making, reasoning, and coding tasks; the authors report improvements of 10–20 percentage points [Shinn et al. (2024)](#page-23-0).

**Buffer of Thoughts.** Another approach, Buffer of Thoughts (BoT), moves toward metareasoning [Yang et al. (2024)](#page-25-0); [Gao et al. (2024)](#page-20-0). It stores high-level *thought templates* distilled from a range of tasks in a managed meta-buffer. A problem distiller extracts these templates from individual problems. Figure [6.9](#page-16-0) compares BoT with approaches such as chain of thought and Tree of Thoughts. The authors report that BoT outperforms the comparison methods on puzzles, with gains of 11% on the Game of 24 and 51% on checkmate tasks. Thought templates are related to metacognition, or “thinking about thinking” [Didolkar et al. (2024)](#page-20-0).

**Critiques and prompt sensitivity.** Self-reflection also has limitations. Asking a model to improve an answer raises the question of why it did not produce the better answer initially. Studies by [Huang et al. (2023)](#page-21-0) and [Shojaee et al. (2025)](#page-23-0) identify methodological concerns in some evaluations: labels may reveal the answer, refinement prompts may provide more information than the initial prompt, or the same LLM may be used as both agent and environment.

{16}------------------------------------------------

<span id="page-16-0"></span>![](assets/_page_16_Diagram_2.jpeg)

Figure 6.9: Buffer of Thoughts [Yang et al. (2024)](#page-25-0).

In one reproducibility study, self-refinement did not outperform chain-of-thought prompting with self-consistency [Huang et al. (2023)](#page-21-0). Results are also sensitive to how a prompt is phrased. Self-reflection remains an active area of research, but many questions remain open.

These approaches may require many LLM calls, making them slow and expensive. Finding a suitable method for a given problem can require substantial experimentation. Chapter 10 presents a practical case study of self-reflection methods; the preceding critiques should be kept in mind.

#### 6.2.4.3 Reinforcement Learning with Verifiable Rewards

This chapter focuses on in-context learning, which occurs at inference time without changing the model’s parameters. Fine-tuning, by contrast, happens during training and updates those parameters. This section briefly discusses methods that combine reasoning at inference time with subsequent model training.

Reinforcement learning with verifiable rewards (RLVR) trains language models on tasks whose outcomes can be checked automatically, such as mathematical problem-solving or code execution. For these tasks, a verifier can determine whether an answer is correct without human feedback.

RLVR replaces a learned reward model with a verification function for tasks with objectively checkable answers. The policy generates completions for a set of prompts, and a deterministic verifier checks them. Correct answers receive a positive reward; incorrect answers receive zero. The reward can then be used to update the policy with an algorithm such as proximal policy optimization (PPO) [Schulman et al. (2017)](#page-23-0), an approach also used in RLHF.

{17}------------------------------------------------

<span id="page-17-0"></span>An influential example is the development of DeepSeek-R1 [Guo et al. (2025)](#page-21-0). Unlike approaches that rely on an external reasoning controller at inference time, this line of work trains a model to generate its own reasoning steps. It combines data generation and reinforcement learning in an iterative process, reducing reliance on supervised examples. Starting from a base model, reinforcement learning can reinforce reasoning patterns that lead to verifiable outcomes. DeepSeek used group-relative policy optimization (GRPO), which estimates advantages from group-based comparisons [Shao et al. (2024)](#page-23-0).

The authors report that sophisticated behaviors, such as revisiting an approach or exploring alternatives, emerged during fine-tuning rather than being explicitly programmed [Mercer et al. (2025)](#page-22-0). For example, DeepSeek-R1-Zero learned to spend more computation on some problems by reevaluating an initial approach. This form of learning to reason may support more adaptive reasoning behavior.

GRPO is suited to tasks with verifiable rewards, where reinforcement-learning updates can be applied without relying on human preference judgments. RLVR methods are now commonly used when fine-tuning LLMs for such tasks [Lambert et al. (2024)](#page-21-0).

Methods such as GRPO generate answers to training prompts, score them with a verifier, and use the resulting rewards to update the model. DeepSeek’s reported results [Guo et al. (2025)](#page-21-0) have increased interest in GRPO [Shao et al. (2024)](#page-23-0) and other RLVR methods [Lambert et al. (2024); Guo et al. (2025); Wen et al. (2025b); Yue et al. (2025)](#page-25-0).

## 6.3 Concluding Remarks

This chapter has examined ways to improve LLM reasoning. Chain-of-thought prompting is a well-known approach: it provides examples of intermediate steps that guide a model through a multi-step problem. More elaborate methods use search, self-reflection, or reinforcement learning at inference time. Training methods such as RLVR use verifier feedback to fine-tune models for reasoning tasks.

{18}------------------------------------------------

#### <span id="page-18-0"></span>6.3.1 Brittleness

Many algorithms have been proposed for building agentic harnesses because the field is young and still faces substantial challenges.

These methods can also be brittle. A small prompt change—even a comma—may determine whether an approach works. Performance can also shift with hyperparameters such as temperature, the random seed, the harness’s memory strategy, or the choice of LLM.

This instability can be frustrating, but it is common in a developing field. Experiments should include multiple runs with different seeds, models, and hyperparameter settings to show how reliably a method works across conditions.

#### 6.3.2 Thinking, Fast and Slow

This chapter closes with a broader perspective on reasoning in LLMs. Two traditions in AI are symbolic AI and connectionism. Symbolic AI models intelligence as the manipulation of symbols according to logical rules. Connectionism studies how intelligent behavior can emerge from interactions among connected units, such as neurons, cells, ants, people, or robots. Symbolic AI draws on mathematics, while connectionism draws on biology. As discussed in Section 2.3, both traditions have produced useful methods and important results, from search and reasoning to deep learning and LLMs.

In *Thinking, Fast and Slow* (2011), Daniel Kahneman uses System 1 and System 2 to distinguish intuitive, fast thought from deliberate, slow thought [Kahneman (2011)](#page-21-0). These terms are also used in AI: fast thinking is associated with retrieval by learned associations, while slow thinking is associated with multi-step reasoning. A zero-shot LLM response at inference time is often compared to System 1; step-by-step prompting or an agentic harness is compared to System 2.

LLMs belong to the connectionist tradition. Tools used at inference time can add functions associated with symbolic AI, such as retrieval and explicit computation.

Researchers sometimes describe LLMs in human terms, but an LLM generates text by predicting the next token; it is not a human thinker. Producing intermediate tokens creates a longer path from the prompt to the final answer. One possible explanation for improved performance is that the model takes smaller steps, making a correct continuation more likely at each stage. This need not imply that its process resembles human reasoning.

{19}------------------------------------------------

<span id="page-19-0"></span>Reasoning can be viewed as narrowing the distribution of possible continuations until the correct answer becomes more likely, without assuming that the model’s process mirrors human cognition [Guo et al. (2025)](#page-21-0).

#### 6.3.3 Conclusion

This chapter has discussed how LLMs use next-token prediction to perform language tasks and how they can approach tasks that require multi-step reasoning. It has also covered methods for managing complex reasoning structures.

Most of the methods are in-context and relatively easy to inspect because their prompts are readable. Agentic harnesses can manage prompts and memory, while pretraining and fine-tuning can teach models to integrate multi-step reasoning into decision-making beyond next-token prediction alone.

Researchers are also developing richer models that combine language with visual reasoning, decision-making, and action. The next chapter examines pretrained and fine-tuned multimodal models.



## Bibliography

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. *arXiv preprint arXiv:2204.01691*, 2022. [128](#page-3-0) Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. *arXiv preprint arXiv:1905.13319*, 2019. [130](#page-5-0) Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(16): 17682–17690, 2024. [138](#page-13-0) Leo Breiman. Random forests. *Machine learning*, 45:5–32, 2001. [130](#page-5-0) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. [125](#page-0-0) Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021. [132](#page-7-0)

{20}------------------------------------------------

<span id="page-20-0"></span>Xinyun Chen, Chen Liang, Adams Wei Yu, Denny Zhou, Dawn Song, and Quoc V Le. Neural symbolic reader: Scalable integration of distributed and symbolic representations for reading comprehension. In *International Conference on Learning Representations*, 2019. [130](#page-5-0) Xinyun Chen, Maxwell Lin, Nathanael Scharli, and Denny Zhou. Teaching large lan- ¨ guage models to self-debug. *arXiv preprint arXiv:2304.05128*, 2023. [132](#page-7-0) Ting-Rui Chiang and Yun-Nung Chen. Semantically-aligned equation generation for solving and reasoning math word problems. *arXiv preprint arXiv:1811.00720*, 2018. [130](#page-5-0) Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021. [127,](#page-2-0) [130](#page-5-0) MAA Codeforces. American invitational mathematics examination-aime 2024, 2024, 2024. [128](#page-3-0) Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, and Clifford Stein. *Introduction to algorithms*. MIT press, 2022. [136](#page-11-0) Aniket Didolkar, Anirudh Goyal, Nan Rosemary Ke, Siyuan Guo, Michal Valko, Timothy Lillicrap, Danilo Rezende, Yoshua Bengio, Michael Mozer, and Sanjeev Arora. Metacognitive capabilities of llms: An exploration in mathematical problem solving. *arXiv preprint arXiv:2405.12205*, 2024. [140](#page-15-0) Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey on in-context learning. *arXiv preprint arXiv:2301.00234*, 2022. [126](#page-1-0) Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, et al. Minedojo: Building open-ended embodied agents with internet-scale knowledge. *Advances in Neural Information Processing Systems*, 35:18343–18362, 2022. [128](#page-3-0) Weizhi Fei, Xueyan Niu, Pingyi Zhou, Lu Hou, Bo Bai, Lei Deng, and Wei Han. Extending context window of large language models via semantic compression. In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 5169– 5181, 2024. [134](#page-9-0) Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In *International Conference on Machine Learning*, pages 10764–10799. PMLR, 2023. [129,](#page-4-0) [132](#page-7-0) Peizhong Gao, Ao Xie, Shaoguang Mao, Wenshan Wu, Yan Xia, Haipeng Mi, and Furu Wei. Meta reasoning for large language models. *arXiv preprint arXiv:2406.11698*, 2024. [140](#page-15-0)

{21}------------------------------------------------

<span id="page-21-0"></span>Louie Giray. Prompt engineering with chatgpt: a guide for academic writers. *Annals of biomedical engineering*, 51(12):2629–2633, 2023. [127](#page-2-0) Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025. [142,](#page-17-0) [144](#page-19-0) Nikolaus Hansen and Raymond Ros. Black-box optimization benchmarking of newuoa compared to bipop-cma-es: on the bbob noiseless testbed. In *Proceedings of the 12th annual conference companion on Genetic and evolutionary computation*, pages 1519–1526, 2010. [133](#page-8-0) Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. *arXiv preprint arXiv:2310.01798*, 2023. [140,](#page-15-0) [141](#page-16-0) Daniel Kahneman. *Thinking, fast and slow*. macmillan, 2011. [143](#page-18-0) Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. When can llms actually correct their own mistakes? a critical survey of self-correction of llms. *Transactions of the Association for Computational Linguistics*, 12:1417–1440, 2024. [128](#page-3-0) Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213, 2022. [129](#page-4-0) Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In *Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies*, pages 1152–1157, 2016. [128](#page-3-0) Emanuele La Malfa, Christoph Weinhuber, Orazio Torre, Fangru Lin, Samuele Marro, Anthony Cohn, Nigel Shadbolt, and Michael Wooldridge. Code simulation challenges for large language models. *arXiv preprint arXiv:2401.09074*, 2024. [135](#page-10-0) Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. *arXiv preprint arXiv:2411.15124*, 2024. [142](#page-17-0) Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨ aschel, et al. ¨ Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020. [134](#page-9-0) Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. *arXiv preprint arXiv:1705.04146*, 2017. [128,](#page-3-0) [130](#page-5-0)

{22}------------------------------------------------

<span id="page-22-0"></span>Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. *ACM Computing Surveys*, 55(9):1–35, 2023. [126](#page-1-0) Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Selfrefine: Iterative refinement with self-feedback. *Advances in Neural Information Processing Systems*, 36, 2023. [139](#page-14-0) Sarah Mercer, Samuel Spillard, and Daniel P Martin. Brief analysis of deepseek r1 and it's implications for generative ai. *arXiv preprint arXiv:2502.02523*, 2025. [142](#page-17-0) Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. *arXiv preprint arXiv:2106.15772*, 2021. [128](#page-3-0) Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. *arXiv preprint arXiv:2112.00114*, 2021. [130](#page-5-0) Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kucinski, Lerrel Pinto, Rob Fer- ´ gus, Jakob Nicolaus Foerster, Jack Parker-Holder, and Tim Rocktaschel. BAL- ¨ ROG: Benchmarking agentic LLM and VLM reasoning on games. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https:](https://openreview.net/forum?id=fp6t3F669F) [//openreview.net/forum?id=fp6t3F669F](https://openreview.net/forum?id=fp6t3F669F). [128,](#page-3-0) [134,](#page-9-0) [135](#page-10-0) Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? *arXiv preprint arXiv:2103.07191*, 2021. [128](#page-3-0) Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and Thomas Back. Multi-step reasoning with large language models, a survey. ¨ *ACM Computing Surveys, arXiv preprint arXiv:2407.11511*, 58(6):160:1–160:35, 2025. [129](#page-4-0) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. [127](#page-2-0) Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. *arXiv preprint arXiv:2112.11446*, 2021. [128](#page-3-0) Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. *arXiv preprint arXiv:2406.07522*, 2024. [134](#page-9-0)

{23}------------------------------------------------

<span id="page-23-0"></span>Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. *Nature*, 625(7995):468–475, 2024. [132](#page-7-0) Subhro Roy and Dan Roth. Solving general arithmetic word problems. *arXiv preprint arXiv:1608.01413*, 2016. [130](#page-5-0) Baptiste Roziere, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample. Unsupervised translation of programming languages. *Advances in neural information processing systems*, 33:20601–20611, 2020. [128](#page-3-0) Anian Ruoss, Fabio Pardo, Harris Chan, Bonnie Li, Volodymyr Mnih, and Tim Genewein. Lmact: A benchmark for in-context imitation learning with long multimodal demonstrations. *arXiv preprint arXiv:2412.01441*, 2024. [128,](#page-3-0) [134,](#page-9-0) [135](#page-10-0) Omer Sagi and Lior Rokach. Ensemble learning: A survey. *Wiley interdisciplinary reviews: data mining and knowledge discovery*, 8(4):e1249, 2018. [130](#page-5-0) Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and Aman Chadha. A systematic survey of prompt engineering in large language models: Techniques and applications. *arXiv preprint arXiv:2402.07927*, 2024. [127](#page-2-0) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017. [142](#page-17-0) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024. [142](#page-17-0) Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36, 2024. [139,](#page-14-0) [140](#page-15-0) Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity. *arXiv preprint arXiv:2506.06941*, 2025. [128,](#page-3-0) [140](#page-15-0) Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, ´ and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. *arXiv preprint arXiv:2010.03768*, 2020. [128](#page-3-0) Niki van Stein and Thomas Back. Llamea: A large language model evolutionary algo- ¨ rithm for automatically generating metaheuristics. *arXiv preprint arXiv:2405.20132*, 2024. [132](#page-7-0)

{24}------------------------------------------------

<span id="page-24-0"></span>Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. *Advances in Neural Information Processing Systems*, 36:74530–74543, 2023. [134](#page-9-0) Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*, 2022. [130,](#page-5-0) [131](#page-6-0) Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*, 2022a. [127](#page-2-0) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824– 24837, 2022b. [128,](#page-3-0) [129](#page-4-0) Pengcheng Wen, Jiaming Ji, Chi-Min Chan, Juntao Dai, Donghai Hong, Yaodong Yang, Sirui Han, and Yike Guo. Thinkpatterns-21k: A systematic study on the impact of thinking patterns in llms. *arXiv preprint arXiv:2503.12918*, 2025a. [128](#page-3-0) Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. *arXiv preprint arXiv:2506.14245*, 2025b. [142](#page-17-0) Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. *arXiv preprint arXiv:2212.09561*, 2022. [130](#page-5-0) Annie Wong, Thomas Back, Aske Plaat, Niki van Stein, and Anna V Kononova. Rea- ¨ soning capabilities of large language models on dynamic tasks. In *3rd International Conference on Foundation and Large Language Models (FLLM2025), 25-28 November, Vienna, Austria, arXiv preprint arXiv:2505.10543*, 2025. [134](#page-9-0) Annie Wong, Aske Plaat, Thomas Back, Niki van Stein, and Anna V Kononova. State ¨ design matters: How representations shape dynamic reasoning in large language models. *Transactions on Machine Learning Research*, 2026. [134](#page-9-0) Yisheng Xiao, Lijun Wu, Junliang Guo, Juntao Li, Min Zhang, Tao Qin, and Tie-yan Liu. A survey on non-autoregressive generation for neural machine translation and beyond. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2023. [130](#page-5-0) BY Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, and Zheng Liu. General agentic memory via deep research. *arXiv preprint arXiv:2511.18423*, 2025. [134](#page-9-0)

{25}------------------------------------------------

<span id="page-25-0"></span>Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. Buffer of thoughts: Thought-augmented reasoning with large language models. *arXiv preprint arXiv:2406.04271*, 2024. [128,](#page-3-0) [140,](#page-15-0) [141](#page-16-0) Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. *Advances in Neural Information Processing Systems*, 35:20744–20757, 2022a. [128](#page-3-0) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*, 2022b. [139](#page-14-0) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 36, 2024. [128,](#page-3-0) [136,](#page-11-0) [137](#page-12-0) Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, et al. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. In *Association for Computational Linguistics*, 2018. [128](#page-3-0) Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? *arXiv preprint arXiv:2504.13837*, 2025. [142](#page-17-0) Alex L Zhang, Tim Kraska, and Omar Khattab. Recursive language models. *arXiv preprint arXiv:2512.24601*, 2025. [134](#page-9-0) Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. *arXiv preprint arXiv:2506.15841*, 2025. [134](#page-9-0)
