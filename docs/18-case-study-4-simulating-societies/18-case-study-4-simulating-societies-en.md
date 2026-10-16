{0}------------------------------------------------

# <span id="page-0-0"></span>Chapter 18

# Case Study 4: Simulating Societies with LLM Agents

#### What You Will Learn

In this case study, you will become familiar with agent-based social simulations through the widely used Generative Agents framework [\[Park et al.,](#page-5-0) [2023\]](#page-5-0). Specifically, you will learn to:

- Run and configure a *multi-agent social simulation* powered by LLMs.
- Conduct a structured *machine-psychological* experiment.
- Compare two *cognitive architectures*: a standard memory system and one with a negativity bias.
- Investigate how *personality descriptions* and architectural choices shape emergent agent behavior.
- Critically assess *prompt fragility*, model dependence, and the limitations of LLM-driven simulations.

#### 18.1 Social Simulation

In social simulations, generative agents are LLM-driven characters that interact with an environment and with one another. They may plan activities, maintain memories, and reflect on their experiences. The Generative Agents framework, introduced by [Park et al.](#page-5-0) [\[2023\]](#page-5-0), showed that LLM-powered characters can collectively produce believable daily routines, social interactions, and even coordinated group activities, all emerging from a relatively simple and elegant cognitive architecture.

In this case study, you will use *two* cognitive architectures for your agents, run the same scenario with each, and compare their emergent behavior. One architecture treats all memories equally. The other introduces a *negativity bias*: negative experiences are encoded more strongly and are more likely to be retrieved and reflected on. We will see that this asymmetry, inspired by findings from cognitive psychology

{1}------------------------------------------------

<span id="page-1-0"></span>[\[Baumeister et al.,](#page-5-0) [2001\]](#page-5-0), measurably affects the agents in our experiment.

The goal is to understand how architectural choices—in this case, small changes to an agent's internal processing—can lead to emergent consequences. You will also see how fragile agent personas can be: a single word in a personality description or a subtle change in memory weighting can fundamentally alter how an agent behaves, plans, and reflects.

#### Course Materials

The course exercise materials are available at:

<https://github.com/DorotaMoravcikova/agentic-llm-course-excercises>

#### 18.2 Background

#### 18.2.1 Generative Agents

The Generative Agents framework, introduced by [Park et al.](#page-5-0) [\[2023\]](#page-5-0), showed that LLM-powered characters can produce believable daily routines and social interactions, and can even coordinate group activities.

Each agent has three core components:

1. **Memory stream:** A timestamped log of everything the agent has perceived, done, and thought.
2. **Retrieval function:** When the agent needs to act, it retrieves relevant memories based on recency, importance, and relevance to the current situation.
3. **Reflection mechanism:** The agent periodically synthesizes higher-level insights from its raw memories.

An agent's *personality* is defined by a short natural-language description stored in a *scratch* file. It specifies the character's name, age, learned and innate traits, and lifestyle. This description is included in every LLM prompt to anchor the agent's behavior.

#### 18.2.2 Bad Is Stronger Than Good

When an LLM-powered agent in the original Generative Agents simulation receives criticism, it forms a memory. If it is praised an hour later, it forms another. The agent retrieves these two memories—one negative and one positive—with equal priority, provided they are equally recent, relevant, and important.

Cognitive architectures for agent-based social simulations often draw on cognitive science to produce believable behavior. The recency, relevance, and importance variables are loosely inspired by findings about human memory retrieval. Importance captures something like the arousal or salience of an event. However, in the original simulation [\[Park et al.,](#page-5-0) [2022\]](#page-5-0), the memory system is effectively

{2}------------------------------------------------

<span id="page-2-0"></span>*valence-blind*: positive, negative, and neutral memories are encoded and retrieved with equal weight. A human, by contrast, might treat a negatively valenced event differently: encode it with sharper sensory detail, retrieve it more vividly, and be more likely to revisit it in later thoughts. Psychological studies find that people remember insults longer than compliments. A single bad interaction with a colleague can outweigh weeks of pleasant ones. In other words, in our psychology, *bad is stronger than good* [\[Baumeister et al.,](#page-5-0) [2001\]](#page-5-0)—an asymmetry that is especially consequential for memory.

In this case study, we test the NEVER (*Negative Emotional Valence Enhances Recapitulation*) model, which adds a more human-like mechanism for memory retrieval. In an agent augmented with NEVER, negative memories receive higher importance scores when encoded. As a result, they are more likely to be retrieved and to trigger reflections.

The architecture is otherwise unchanged. This makes it possible to conduct a controlled *machine-psychological* experiment comparing two architectural choices: valence-blind versus valence-sensitive memory. You will also use the structured setup to test how changes to agent initialization—particularly personality descriptions and system prompts—affect behavior.

#### 18.2.3 Why This Matters for Agent Design

This case study illustrates two important lessons:

1. **Architectural choices have emergent consequences.** A memory-weighting parameter that seems innocuous at the component level can produce dramatic changes in an agent's social behavior, emotional tone, and decision-making over time. These effects are difficult to predict from the architecture alone.
2. **Prompts are fragile.** Agent behavior is highly sensitive to personality descriptions and system prompts. Small changes can produce large behavioral shifts, a phenomenon sometimes called *prompt sensitivity* in the LLM literature.

#### 18.3 Implementation

##### 18.3.1 The Simulation Environment

The simulation uses the Smallville environment from [Park et al.](#page-5-0) [\[2023\]](#page-5-0): a small town rendered as a 2D tile map, where agents move between locations, interact with objects, and converse.

Our simulation uses a simplified scenario: a cafe with three agents:

- **Dolores:** A warm, empathetic, detail-oriented barista. She is the *experimental* agent and will receive the negativity bias.
{3}------------------------------------------------

<span id="page-3-0"></span>- **Maeve:** A barista with a matched personality profile. She is the *control* agent, using the standard architecture without a negativity bias.
- **Bernard:** The cafe manager, who is unpredictable, moody, and occasionally confrontational. He serves as a naturalistic *stressor*, creating social friction to which both baristas are exposed.

This three-agent setup, in which two agents have similar roles and positions (baristas working under Bernard's supervision), allows us to control for agent architecture and initialization. In the first experiment, Dolores and Maeve receive the same personality traits and encounter the same environment. The only difference is how Dolores processes memories: she uses the NEVER implementation, which assigns greater importance to negative memories.

##### 18.3.2 Agent Architecture

Each agent maintains:

- A *personality description* (the scratch file): a short text describing the agent's name, traits, occupation, relationships, and goals.
- A *memory stream*: a list of timestamped observations, actions, and reflections.
- A *current plan*: a short description of what the agent intends to do.

##### 18.3.3 The Cognitive Loop

At each simulation step, representing 10 seconds of in-game time, each agent executes the following cycle:

1. **Perceive:** Observe the current environment, including what other agents are doing, any ongoing conversations, and changes in the surroundings.
2. **Retrieve:** Query the memory stream for relevant past experiences.
3. **Plan:** Use the personality description, retrieved memories, and current observations to decide what to do next with the LLM.
4. **Act:** Execute the planned action in the environment.
5. **Reflect (periodically):** Every few hours of in-game time, synthesize higher-level reflections from recent memories.

Each step involves an LLM call. The prompt combines the agent's personality description, which remains constant, with the relevant retrieved memories and current observations, which change over time.

##### 18.3.4 Two Architecture Variants

You will use two variants of the retrieval function:

{4}------------------------------------------------

<span id="page-4-0"></span>**Variant A: Baseline memory retrieval.** Retrieval is based on a weighted combination of recency, relevance, and importance. This is the standard Generative Agents architecture described by [Park et al.](#page-5-0) [\[2023\]](#page-5-0).

**Variant B: Negativity-biased memory retrieval.** When a new memory is created, its emotional valence is assessed. The retrieval function is extended with a valence term. Absolute valence is normalized to [0, 1] and multiplied by 1.5 for negative memories, producing a retrieval hierarchy of negative > positive > neutral. The retrieval score becomes:

$$
\operatorname{score}=\alpha_{\mathrm{rec}}\cdot R+\alpha_{\mathrm{rel}}\cdot L+\alpha_{\mathrm{imp}}\cdot I+\alpha_{\mathrm{val}}\cdot V
\tag{18.1}
$$

All weights are set to $\alpha_i=1$. Thus, the only difference between the two variants is the addition of the valence term; everything else is identical.

##### 18.3.5 Setup and Installation

Clone the repository and follow the instructions in its README:

<https://github.com/DorotaMoravcikova/generative_agents/tree/course-exercises>

#### 18.4 Discussion

After running the experiments, reflect on the following questions:

1. **Prompt versus architecture:** You have seen that both the personality prompt and the memory architecture influence agent behavior. Which has a stronger effect? Which is more predictable? If a simulation were used for a consequential application, such as modeling policy adoption, which would you use to shape agent behavior, and why?
2. **Emergent behavior:** Did you observe any agent behaviors that surprised you—things not explicitly specified in the personality or architecture? In your view, what mechanisms produced them?
3. **Evaluation challenges:** How can you evaluate whether a generative agent is behaving well? The valence score is one metric, but it is itself an LLM judgment. What are the limitations of using an LLM to evaluate another LLM's output (or that of the same LLM)? Can you think of alternative evaluation approaches?
4. **Cognitive realism:** Negativity bias is one finding from cognitive psychology. What other cognitive biases or memory effects could be implemented architecturally? What emergent effects would you expect? Consider, for example, confirmation bias, the availability heuristic, and recency bias.

{5}------------------------------------------------

## <span id="page-5-0"></span>18.5 Concluding Remarks

In this chapter, we used a generative-agent simulation to explore two important phenomena in agentic AI: the emergent effects of cognitive-architecture choices (here, negativity bias) and the sensitivity of agent behavior to prompt content (prompt fragility).

We saw that a small change to memory weighting—assigning negative memories higher importance scores—can produce measurable changes in agent behavior over time. We also saw that agent personas are fragile: small changes to personality descriptions can produce large, sometimes unpredictable, effects on behavior.

These findings have practical implications for anyone building LLM-driven agent systems. A persona is not a reliable anchor for behavior. The architecture—how memories are stored, retrieved, and reflected on—shapes behavior in ways that may not be apparent from reading the code.

Generative-agent simulation is a young and evolving field. The exercises in this chapter provided hands-on experience with its core mechanisms and challenges. As the field matures, prompt robustness, cognitive-architecture design, and the evaluation of emergent behavior will remain central concerns.

#### Bibliography

Roy F. Baumeister, Ellen Bratslavsky, Catrin Finkenauer, and Kathleen D. Vohs. Bad is stronger than good. *Review of General Psychology*, 5(4):323–370, 2001. [354](#page-1-0), [355](#page-2-0).

Joon Sung Park, Lindsay Popowski, Carrie Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Social simulacra: Creating populated prototypes for social computing systems. In *Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology*, pages 1–18, 2022. [354](#page-1-0).

Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, pages 1–22, 2023. [353](#page-0-0), [354](#page-1-0), [355](#page-2-0), [357](#page-4-0).

<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 353-358; Marker split PDF pages 1-6. -->

