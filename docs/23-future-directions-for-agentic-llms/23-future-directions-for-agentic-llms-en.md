<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 461-468; the final conclusion sentence continues from page 469, which also begins the bibliography. Marker split PDF pages 1-8. -->



{0}------------------------------------------------


<span id="page-0-0"></span>
## Chapter 23

# Future Directions for Agentic LLMs

As we reach the end of this book, we provide a concise overview of the material we have discussed. First, we look back at how agentic LLMs are built. Then we summarize their applications and behavior in single-agent and multi-agent settings. Finally, in Section [23.2](#page-1-0), we examine the limitations of current approaches and what the future may bring.

## 23.1 The State of Agentic LLMs

As discussed in Parts I, II, and III, the foundations of agentic LLMs lie in language modeling, reasoning, tool use, and different degrees of autonomy.

#### 23.1.1 Technology

Agentic LLMs are reasoning models that act in the world and adapt their behavior based on feedback. After pretraining and fine-tuning (Chapter 3), they may access external information at inference time through retrieval-augmented generation (RAG; Chapter 12). They may also prompt themselves to verify results, enabling self-reflection and in-context reinforcement learning (Chapter 6). Agentic LLMs are increasingly multimodal, operating across language, vision, and action (Chapter 7). To support these capabilities, they can be equipped with pretrained world models that provide a rudimentary understanding of the physical world. Interactions with the environment may then improve these world models at test time.

The main machine-learning architecture is the Transformer. Models are pretrained through self-supervision (next-token prediction) and fine-tuned for specific tasks through supervised and reinforcement learning.

In-context chain-of-thought prompting has significantly improved models’ reasoning abilities. Inference-time methods now support fine-tuning (e.g., RLVR and GRPO) and self-reflection (e.g., ReAct and Reflexion; Chapter 6).

{1}------------------------------------------------

<span id="page-1-0"></span>A broad range of prompting styles for agent–environment interaction has also been developed (Chapter 9).

#### 23.1.2 Applications

Part IV introduced tools that make it easier to build agentic LLMs (Chapter 11). These systems have been designed for many applications. In e-commerce, they can help identify users’ preferences and make recommendations. In medicine, they can read and summarize diagnostic texts. In finance, they can analyze financial reports and advise traders. In science and education, they can support literature research, generate experiment scripts, analyze results, help write papers, and assist with reviewing them (Chapter 13).

#### 23.1.3 Behavior

In Part V, we shifted from a technical view of agentic LLMs to studying their behavior in single-agent and multi-agent settings. After introducing basic behavioral features of LLMs, Chapter 15 examined how LLM-driven agents perform on tasks adapted from human behavioral methods and frameworks developed for AI benchmarking. Chapter 16 focused on multi-agent interactions built around predefined roles and protocols. In Chapter 17, the focus shifted to open-ended settings, where researchers studied behavioral structures, norms, and conventions that emerged at the system level from small-scale interactions among agents.

Several recurring patterns emerged across the approaches and studies in Part V: concerns about construct validity (how well observations generalize to human settings); prompt sensitivity and cascading effects, especially in multi-agent systems; a knowing–doing gap, in which models’ ability to predict verbally did not reliably translate into action; benefits from structured reasoning, particularly when increased reasoning budgets followed structures inspired by human reasoning; and emergent properties that could be desirable (performance above compute-matched baselines) or undesirable (norm violations).

## 23.2 Limitations and Future Work

Agentic LLMs build on the capabilities of large language models and have greatly expanded their uses. They still face many limitations. We begin with more technical topics—memory, scale, explainability, and cognitive architectures—then turn to broader questions, including potential sentience or consciousness and the ethical implications of these properties. Earlier chapters have discussed possible approaches to some of these challenges.

{2}------------------------------------------------


<span id="page-2-0"></span>
#### 23.2.1 Memory and Self-reflection

Language models are functions: the output that they compute is only based on their input—they are memory-less between calls. In order for agents to act naturally in the world, to hold a conversation, to be able to learn from their actions by self-reflection, agents must be able to remember. For us, our identity hinges on our ability to remember. Agents require a form of memory.

One simple way to provide rudimentary memory is to feed past interactions back into the prompt, an approach used successfully by current chatbots. However, context windows remain limited despite progress in long-context modeling. Agent memory is an active research area, including how LLMs can best compress information and store it in short- or long-term memory.

Agents can access external information through retrieval-augmented generation (RAG), for example from corporate databases. Researchers are also developing agent memory architectures that provide longer-term storage, either as text or as token embeddings in vector databases (Section 9.3).

#### 23.2.2 Scale

As agents become able to self-reflect and to learn from interactions with their environment, we should discuss the topic of scaling of the underlying language models.

A major challenge is to continue improving LLM performance. Initially, model size was a key determinant—hence the term *large* language models. Only sufficiently large models trained on sufficiently large text corpora achieved strong benchmark results, and in-context learning was observed only in larger models. Scaling model size and training data therefore seemed a promising path. Further scaling is now increasingly difficult: much of the high-quality data available on the Internet has already been used to train current models, while training larger models for longer periods requires substantial time, money, and energy. Current methods may be reaching a plateau [Sutskever, 2024].

Further gains require challenging training data (Chapter 19). One approach is to use agent–environment interactions as training data. When such data is new, diverse, and high quality, it may support pretraining (see also Figure 1.1). Some inference-time methods, including GRPO and RLVR, already use results generated at inference time to fine-tune models [Guo et al., 2025]. Vision-language-action (VLA) models are also trained on self-generated action data [Black et al., 2024; Brohan et al., 2023] (Chapter 7).

##### Overthinking

Scaling studies of chain-of-thought reasoning also reveal limits to additional thinking effort. Chain of thought is more effective in some domains than in others [Sprague et al., 2024]. Deeper reasoning can lead to *overthinking*, in which the reasoning process becomes less effective as more effort is spent [Liu et al., 2024; Wu et al., 2025]. Ongoing studies are examining how chain of thought works [Chen et al., 2026b] and which mechanisms are effective [Chen et al., 2026a].

{3}------------------------------------------------


<span id="page-3-0"></span>
##### Open-world models and curriculum learning

For scaling to work, synthetic training data must be diverse and differ from existing data. Open-world models are being studied as a way to generate diverse training environments. Such models aim to create effectively unlimited datasets or environments in which agents can continue learning and improving [Bruce et al., 2024]. Open-world models draw inspiration from evolutionary algorithms, quality-diversity methods, and computer games [Vinyals et al., 2019; Mouret and Clune, 2015; Bakhtin et al., 2022].

Another way to improve performance is to develop better training methods. Curriculum learning builds on the observation that learning often improves when a system starts with easy tasks before moving on to harder ones. This applies to both human learning and artificial neural networks. Most domains have hierarchies of related concepts that build on one another; in school, for example, students learn addition before long division or logarithms. Researchers are exploring curriculum methods for both pretraining and post-training [Chen et al., 2025; Elgaar and Amiri, 2026].

#### 23.2.3 Hallucination and Explainability

Another limitation is hallucination: models often give plausible-sounding but factually incorrect answers, and their stated reasoning may not be causally responsible for their outputs—a failure of *faithfulness*.

Explainable AI (XAI) offers a range of techniques, from behavioral and attribution-based methods such as SHAP and LIME, which treat the model as a black box, to concept-based interpretability, which probes internal representations for human-understandable features. Mechanistic interpretability (MI), discussed in Chapter 21, goes further by aiming to reverse-engineer computational circuits inside Transformer models.

Tools such as sparse autoencoders, neural lenses, and activation patching have been used to investigate circuits associated with hallucinations and to test whether chain-of-thought reasoning can be a faithful, causally active part of model behavior in sufficiently large architectures. As agentic LLMs enter high-stakes settings, MI may help researchers develop hallucination detectors that flag discrepancies between a model’s internal confidence features and its external output.

Another direction is to compare human and Transformer cognitive architectures. Language models appear to produce natural language in ways that resemble humans. Does this reflect similar underlying architectures, coincidence, or some other explanation? Chapter 22 explored this question.

#### 23.2.4 Emotional Behavior and Consciousness

{4}------------------------------------------------


<span id="page-4-0"></span>
Humans tend to anthropomorphize: when inanimate objects behave like us, or sometimes even when they do not, we may attribute purposeful behavior and implicitly attribute consciousness to them. This can lead us to say that *the computer is thinking*, *the computer wants something*, or *the computer must not be feeling well*—or even that *this book loves to be read*. LLMs can elicit similar reactions, and their utterances may persuade even rational observers to believe they have purposes. The source text argues that they cannot have purposes because they are algorithms executed by mechanical devices, unlike humans, whom it describes as organic.<sup>1</sup>

Reports describe emotional relationships between chatbots and people, some of which have led to dangerous behavior and worse.

Consciousness is actively debated in AI philosophy. It is generally understood as awareness of oneself and the environment, together with subjective experiences, thoughts, and feelings [Velmans, 2009; Butlin et al., 2023]. But how can it be tested? A pragmatic view holds that behavior and its consequences matter to society, regardless of whether the acting entity is conscious. Bengio and Elmoznino [2025] argue that even the appearance of consciousness in an agent’s actions can have consequences, and that distinguishing conscious from non-conscious entities will become increasingly difficult. They argue that society must decide how to respond to agents acting in domains such as medicine, finance, and science. Credit and liability must be assigned fairly, and society must determine how to handle self-preserving AI systems.

#### 23.2.5 Social Behavior

We have discussed how LLMs can simulate human reasoning (Chapter 15) and how they can improve simulations of social behavior in multi-agent systems. However, these simulations remain heavily influenced by their prompt scripts. Some studies report realistic social behavior, while others identify substantial limitations.

Studying emergent behavior at scale also poses challenges (Chapter 17). As in any multi-agent simulation, it is difficult to validate how closely the simulated behavior reflects reality, although some studies report impressive results [Park et al., 2024].

#### 23.2.6 Legal and Ethical Aspects

Assistants’ actions in the real world have real-world consequences. Agentic LLMs therefore raise legal and ethical challenges, especially when imperfect models act. Questions of responsibility and liability should be addressed before deployment—for example, when a financial assistant hallucinates or a self-driving robot makes an incorrect inference.

<sup>1</sup>Note that the duck-typing point of view disagrees: this behaviorist view holds that if it walks like a duck and quacks like a duck, we should consider it a duck.

{5}------------------------------------------------


<span id="page-5-0"></span>
Technical safety is a major research priority. Securing agentic LLMs requires moving beyond prompt-based defenses to integrated, multilayer safeguards. In safe reinforcement learning, one key direction is to impose explicit safety constraints on agents’ actions and on their reasoning and planning processes [Gu et al., 2024].

Another approach is continual adversarial training and automated red-teaming to improve robustness against manipulation. Other priorities include mechanisms for self-regulation and risk awareness, so agents can detect and avoid unsafe actions, as well as rigorous, standardized safety benchmarks [Andriushchenko et al., 2024]. Together, these measures outline a roadmap for accountable and trustworthy deployment in high-stakes domains.

Questions about who is responsible for an agent’s actions remain unresolved. As agents begin to be used in the real world, addressing these issues becomes urgent. Many safety, ethical, and trust concerns must be addressed before the full potential of agentic LLMs can be realized. These topics will remain important research priorities and warrant dedicated surveys and books [Gan et al., 2024; Jiao et al., 2025; Raza et al., 2025].

#### 23.2.7 Crime

The discussion on ethics and consciousness becomes quite concrete when we consider criminal behavior. Technological developments can be used by people who act within the law, and those who act outside of the law. They can also be used for unethical behavior for which no effective laws yet exist. As agentic LLMs are able to display more and more believable behavior, it will become difficult to distinguish real and artificial behavior. This may lead to dangerous ethical situations, for example when criminals use the technology to influence the behavior of unsuspecting victims.

The boundary between lawful persuasion, exploitation of trust, and fraud can be unclear. Agentic LLMs may make social engineering more sophisticated across social networks, marketing, espionage, and cybercrime [Kshetri, 2025; Lupinacci et al., 2025]. They may also help combat crime [Challita and Parrend, 2025; Andriushchenko et al., 2024], but public awareness may take time to catch up with the risks. People less familiar with social networks and LLMs may be especially vulnerable to confidence tricks and agent-enabled cybercrime.

#### 23.2.8 Limitations of Applications

Agentic LLMs have a wide range of potential applications. Proofs of concept can be impressive at first glance, but closer examination reveals limitations that may be difficult to address and could have serious consequences when these systems are used in the real world.

#### 23.2.8.1 AI Science

{6}------------------------------------------------


<span id="page-6-0"></span>
Many agentic LLMs are being developed to support scientific research across the entire research pipeline. Literature-summarization tools support ideation, while other tools generate experiment scripts, support data interpretation and graph creation, assist with writing, and help review scientific manuscripts. As a result, it is becoming easier to produce papers that are well written and look polished on the surface.

In mathematics, LLM-based systems have achieved impressive results on olympiad-level problems in both geometry [Trinh et al., 2024] and algebra [Hubert et al., 2026]. These systems include Transformer networks but are otherwise specialized mathematics solvers that use automated theorem-proving tools such as Lean [De Moura et al., 2015] and reinforcement learning [Silver et al., 2017; Browne et al., 2012].

In 2025, a benchmark study evaluated AI science tools across several tasks in machine learning [Nathani et al., 2025]. Using the frontier models available at the time, the authors found no evidence that these systems could generate novel hypotheses, algorithms, or architectures, or make substantial scientific advances. They noted that *modern LLM agents can successfully tackle a diverse array of quantitative experiments, reflecting advanced skills and domain adaptability*, while also cautioning that *it is not yet clear if the notion of scientific novelty can be successfully automated or even formally defined in a form suitable for agents*.

#### 23.2.8.2 E-commerce

Liability is a major obstacle to deploying agentic LLMs in e-commerce. Security and privacy concerns may prevent users from sharing personal details and preferences with systems vulnerable to data leaks. This creates a catch-22 for recommender systems: when users withhold information about their preferences, recommendations may be poor, reducing trust and making users even less willing to share. Public adoption will depend on collecting high-quality preference information safely and securely, and on providing useful recommendations.

#### 23.2.8.3 Trading

Liability is paramount in financial trading. Studies report that LLMs can outperform human analysts at summarizing large volumes of financial reports, but the stakes are high: incorrect advice can cause direct financial losses. Because markets are unpredictable, it may take time to identify a faulty model; an individual loss may result from a model error or simply an unlucky trade. Models need thorough testing before they can be trusted. Automated trading systems are unlikely to replace human decision-making unless they prove exceptionally reliable.

LLMs have had the most success in finance when analyzing textual reports. Some studies of financial time-series analysis report positive results, but reliable and verifiable benchmarks remain difficult to establish.

{7}------------------------------------------------


<span id="page-7-0"></span>
Li et al. [2026] likewise question whether agentic LLMs can reliably predict financial-market outcomes.

#### 23.2.8.4 Medicine

In medicine, accuracy and especially explainability are essential for trust. Model predictions have direct implications for liability and the ethics of action, so medical agentic LLM systems must keep a human in the decision-making loop.

As in trading, the most promising medical applications of agentic LLMs involve analyzing and summarizing text, such as meeting reports or literature reviews. Results in other applications, including treatment planning and surgery, remain limited.

Medicine may be one of the most valuable—and most cautious—domains for agentic LLMs. In matters of life and death, we must proceed carefully, even as the potential benefits remain substantial.

## 23.3 Concluding Remarks

Agentic LLM technology is developing rapidly. This book has traced that development across three connected pillars: *reasoning*, *acting*, and *interacting*. Reasoning LLMs provide a cognitive foundation through chain of thought, self-reflection, and retrieval augmentation, enabling multi-step problem solving, literature synthesis, and complex decision support. Acting agents extend these capabilities into the real world through tools, APIs, and world models, assisting users in fields from robotics to finance and medicine. Interacting agents go further by engaging in multi-agent simulations that exhibit emergent social behaviors—such as cooperation, trust, and norm formation—that were not explicitly scripted in the prompts.

These three pillars are connected. As Figure 1.1 illustrates, they form a cycle: reasoning enables better action and interaction, while data generated by agents as they act and interact can be used to pretrain and fine-tune better reasoning models. This virtuous cycle may help move beyond plateaus in LLM scaling. Interaction data from agentic assistants, robotic systems, and multi-agent simulations could provide a continually replenished source of diverse, high-quality training signals.

Yet, significant challenges remain across all three pillars. Hallucination undermines trust in reasoning chains and threatens the stability of self-improving systems. Safety, liability, and ethical accountability are unresolved for acting agents in high-stakes domains. Emergent behavior in large-scale multi-agent systems remains difficult to validate and control. Further research in the fields of mechanistic interpretability, open world models, scalable simulation infrastructures, and robust safety benchmarks is required to overcome these challenges.

<span id="page-8-0"></span>

We are at an early and consequential moment. Proofs of concept are impressive, real-world deployments are beginning, and the theoretical foundations of memory, metacognition, consciousness, and collective intelligence remain open questions that connect artificial intelligence to philosophy, cognitive science, and law. This book has aimed to give you the technical grounding and conceptual vocabulary to contribute to these questions. The field needs researchers who understand both the power and the limits of what agentic LLMs can do today, and who can help shape what they will become tomorrow.
