<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 287-316; Marker split PDF pages 1-30. -->



{0}------------------------------------------------

## <span id="page-0-0"></span>Chapter 15

# Studying Behavior of Agentic LLMs

#### What You Will Learn

In this chapter you will learn how agentic LLM behavior compares to human behavior. Specifically, you will be able to:

- Explain what is meant by *LLM behavior* and describe how this can be studied.
- List *basic social and interactive competencies*, and describe on which aspects LLMs do well and on which aspects they struggle.
- Describe ways of studying advanced social and interactive competencies in LLMs, in particular *strategic thinking*, *Theory of Mind*, and *moral reasoning*.
- Reflect on what behavioral approaches to LLMs can and cannot tell us, given various issues regarding *testing and interpretation*.

#### 15.1 What is LLM Behavior?

In 2019 a team of authors from across the computational, behavioral, and cognitive sciences described the emergence of a new field of study: *machine behavior*. This field is to be *concerned with the scientific study of intelligent machines, not as engineering artifacts, but as a class of actors with particular behavioral patterns and ecology* [\[Rahwan et al.,](#page-24-0) [2019\]](#page-24-0). Machine behavior is inherently interdisciplinary, as it integrates knowledge at the intrinsic level of *technology* (algorithms, data structures, training procedures) with knowledge of the *environment and context* of its actual usage.

The approach in machine behavior is compared to work done in ethology or psychology: here too, making progress in understanding animal and human functioning requires integrating insights at the intrinsic (e.g. biophysics, neurology) and ecological levels (interaction with the environment, social interaction).

As noted in the introduction to this part, a practical motivation for studying machine behavior is that LLMs are becoming increasingly integrated into our work and personal lives, making their behavioral tendencies and patterns important to understand.

{1}------------------------------------------------

<span id="page-1-0"></span>

There is also a scientific motivation: due to the complexity of large-scale neural architectures (they appear as *black boxes*), it is difficult to trace their functioning analytically. (Although the field of mechanistic interpretability does attempt to do this, see Chapter 21.) Studying the *behavior* of such complex systems empirically is a complementary route to our understanding. In other words, due to the complexity of interactions, a theoretical engineer who would know everything there is to know about the architecture and training data of an LLM, still may not be able to predict how it responds in different kinds of situations. Behavioral approaches aim to fill this gap.

What counts as *behavior* depends on the affordances<sup>1</sup> of the system under investigation and the scope of the research. For example, for a basic LLM, behavior equals the *output* as produced by the token-based inference of the model. On the other hand, for an agentic LLM that can perform, for instance, web search and run code, behavior can also include the *actions* that it performs to get to a certain output, or to achieve a certain effect. As we will see, some studies take only the final output into account and treat that as the model's behavior; others are interested in the intermediate steps as well.

#### 15.2 Probing LLM Behavior Experimentally

In recent years, many studies have appeared that test LLMs' social and interactive abilities. There are two main approaches: (1) prompting models with items adapted from experimental paradigms originally developed for humans, and (2) developing new test paradigms specifically for LLMs. The latter can be done quantitatively, using large scale benchmarks, or qualitatively, allowing for in-depth manual analysis on a smaller scale. In what follows we will see examples.

Applying behavioral and psychological methods to (agentic) LLMs is not straightforward. Experiments designed for humans assume an embodied subject with beliefs, desires, and a sense of self that is continuous through time. Agentic LLMs have none of these in any obvious sense. This calls for reflection on *construct validity*: what does it mean to measure a certain trait or capacity in an AI system? This question applies on two levels: that of the observable outcomes of a trait and of the underlying mechanisms. In Section [15.5](#page-10-0) the example of the *false belief-test* [\[Carpenter et al.,](#page-20-0) [2002\]](#page-20-0) is discussed. While some model classes have been shown to answer the questions of this test in similar ways as humans do (the level of the behavioral outcome), it is currently unknown to what degree the underlying mechanisms that produce this behavior differ or overlap in transformer models and humans. In Chapter 22 the issue of construct validity will be discussed in more depth.

For LLM-specific tests, there are also methodological challenges to consider. For example, LLMs are highly sensitive to seemingly irrelevant prompt variations (such as word order and commas) or can produce different outputs across multiple runs due to sampling variability, model temperature, or with different random

<sup>1</sup>*Affordance* refers to what a system can do given the way it was designed and implemented in a specific context.

{2}------------------------------------------------

<span id="page-2-0"></span>seeds. Also, data leakage of experimental materials into the training data can lead to performance based on memorization rather than generalization. We will see several applied examples in the following sections. Throughout this part we will pay attention to methodological rigor and interpretive caution at the end of each chapter.

In the next sections we will first discuss the basic social toolkit for agent interaction. Next, we will look at strategic behavior, where agents reason about the behavior of other agents, and we will discuss theory of mind. Finally, we will discuss moral judgment by LLMs.

#### 15.3 The Basic Social Toolkit

This section surveys basic social and interactive capacities, including conversation, pragmatics, and social etiquette.

#### 15.3.1 Functional and Pragmatic Linguistic Competence

As discussed in Part I, the key advancement of instruction-tuned LLMs is their ability to interact, chatbot style, using natural language. This requires not only a degree of *formal* linguistic competence (i.e., producing correct, grammatical sentences) but also of *functional* linguistic competence, the ability to interpret the meaning of an utterance in a specific context [\[Mahowald et al.,](#page-23-0) [2023\]](#page-23-0). Within linguistics and the cognitive sciences, functional linguistic competence is studied in the field of *pragmatics*. The core concepts of pragmatics go back to the *pragmatic turn* in mid-20th century philosophy of language and posit that linguistic meaning is for the most part not inherent in the signs themselves (tokens, words, sentences), but in their usage in actual contexts. This includes both the *linguistic context*, encompassing surrounding words up to distant factors such as titles and genres, as well as the *extra-linguistic context*, encompassing the spatial, temporal, and social dimensions. This view is most famously expressed in Paul Grice's notion of *conversational implicature* [\[Grice,](#page-21-0) [1975\]](#page-21-0). Consider the following examples:

- 1. A: Could you pass me the salt, please? B: Yes, I could. Would you also like me to do so?
- 2. A: Is Sarah a good colleague? B: Well, let's say that she's always on time.
- 3. He produced a series of sounds that corresponded closely with the score of an aria.

Grice argued that language usage is governed by a *cooperative principle*: we assume our interlocutors (humans or agents) are trying to communicate sensibly. Example 1 above features a classic instance of B undermining the cooperative principle: someone asking whether you can pass the salt is rarely interested in a literal answer confirming that possibility, but instead in the action of actually passing the salt. In 2, B's response is conspicuously narrowly focused on one aspect (punctuality), and the listener can infer, for example, that B is implying something negative about Sarah's collegiality. 

{3}------------------------------------------------

<span id="page-3-0"></span>Similarly, when someone utters 3 rather than simply *He sang an aria*, the over-thetop elaborate phrasing may implicate a certain attitude towards the performance; for example that it was not of the speaker's liking. In all cases, what is literally said is only one part of the message––the other part needs to be decoded or *read between the lines* by the listener from the context.

Interpreting linguistic meaning in a flexible, context-dependent, way has long been a notorious problem in natural language processing [\[Trott et al.,](#page-26-0) [2020\]](#page-26-0). There is general agreement that great progress has been made with LLMs on this problem, yet it is not well-understood to what factors this can be ascribed. Model size is an important predictor of functional and pragmatic competence, as are some forms of fine-tuning (SFT and DPO in particular) [\[Yu et al.,](#page-28-0) [2026,](#page-28-0) [Ruis et al.,](#page-25-0) [2023\]](#page-25-0). However, the variation between different domains of functional and pragmatic understanding in LLMs is still large, and scores are overall well below human performance [\[Sravanthi et al.,](#page-26-0) [2024\]](#page-26-0).

One explanation is that LLMs have less access to extra-linguistic contextual information: they cannot see, hear, and otherwise sense the same as their human counterpart, and in most cases they do not have knowledge of previous interactions [\[van der Meulen](#page-26-0) [et al.,](#page-26-0) [2025\]](#page-26-0). With the shift to agentic LLMs this situation is likely changing, as the models become equipped with memories, multimodal capacities, and other tools that add context. Empirical data on how this affects performance is currently missing.

#### 15.3.2 Politeness and Social Etiquette

Social etiquette and politeness in human-machine interaction have been studied for decades [\[Ribino,](#page-24-0) [2023\]](#page-24-0). A useful theoretical anchor is the *Computers Are Social Actors* (CASA) paradigm [\[Nass et al.,](#page-24-0) [1994\]](#page-24-0), which holds that we tend to apply common human social rules and expectations to computers even when we know that the machine is not a human social agent. We show and respond to cues of politeness, personality, and reciprocity as a way of negotiating trust and acceptance with a machine. Studies show that humans trust machines more when those machines adhere to social etiquette [\[Miller,](#page-23-0) [2005\]](#page-23-0), improving both task performance and user satisfaction [\[Miyamoto et al.,](#page-24-0) [2021\]](#page-24-0). We will discuss politeness theory and its application to LLMs.

**Politeness theory.** Classic politeness theory [e.g. [Brown and Levinson,](#page-20-0) [1987\]](#page-20-0) builds directly on the Gricean picture introduced above. This theory holds that interlocutors continually manage one another's *face*, the public self-image every participant seeks to maintain, which has a positive dimension (the desire to be liked and be approved of by others) and a negative dimension (the desire to avoid social obstacles). Many ordinary requests, criticisms, and refusals, are intrinsically *face-threatening* from a Gricean point of view, and speakers deploy politeness strategies in response. This framework has given LLM researchers a vocabulary for asking whether LLMs and their users recognize face-threats and mitigate them in contextually appropriate ways.

**Are LLMs polite to their users?** A related direction concerns whether the model itself produces etiquette that is socially and culturally appropriate. Here the limitation flagged in the previous subsection resurfaces—that LLMs have impoverished access 

{4}------------------------------------------------

<span id="page-4-0"></span>to extra-linguistic context. Politeness is culture-specific, yet systems trained predominantly on English-language data tend to default to Anglophone, directness-oriented norms [\[Havaldar et al.,](#page-21-0) [2023\]](#page-21-0). A clear demonstration of this problem arises in work on Persian *taarof*, a ritualized system of offers and refusals in which the appropriate response is frequently the opposite of the direct one. A study shows that frontier models fall over 40% short compared to native speakers in scenarios where taarof is culturally expected, defaulting instead to Western-style directness. This results in a disconnect between surface politeness and cultural appropriateness. For example, in one study, the model in questions responses scored 84.5% as *polite* by a standard metric, whereas only 41.7% were judged culturally correct [\[Gohari Sadr et al.,](#page-21-0) [2025\]](#page-21-0). Scoring well on a Western politeness rubric is not the same as behaving appropriately in all contexts. This is a reminder to continuously reflect on construct validity when measuring politeness in a model.

**Does user politeness affect LLM performance?** A study across English, Chinese, and Japanese tasks found that impolite prompts tend to degrade performance in a variety of models [\[Yin et al.,](#page-27-0) [2024\]](#page-27-0). Furthermore, the study found that exaggerated politeness yields no reliable benefit, and that the optimal level of politeness varies by language and cultural background. This result fits with the computers-are-social-actors paradigm: LLMs seem to reproduce the human association between rudeness and failed cooperation from the training data. However, in a direct counterpoint, [Dobariya and](#page-20-0) [Kumar](#page-20-0) [\[2025\]](#page-20-0) report that on a multiple-choice benchmark, blunt and even rude prompts slightly *outperformed* polite ones for GPT-4o. The mapping from tone to quality thus appears neither stable across model generations nor straightforward. This illustrates the prompt-sensitivity and reproducibility concerns raised earlier in this chapter, and cautions against treating any single tone effect as a robust property of LLMs as such.

Taken together, LLMs reproduce the surface markers of politeness and are demonstrably sensitive to the politeness of their interlocutor, yet they neither apply these norms in a cross-culturally robust way, nor exhibit a stable relationship between tone and task performance.

It should be kept in mind that in studies so far, politeness and etiquette were limited to the linguistic dimension of interaction only. This contrasts with the multidimensionality of human-human interaction, where facial expressions, gestures, and other contextual cues are involved [\[Brown and Levinson,](#page-20-0) [1987\]](#page-20-0). As with pragmatic competence, the shift to multi-modal agentic LLMs equipped with memory and increased access to user-specific context, can be expected to improve adaptation to a given user's norms, but as yet there is little empirical evidence.

#### 15.3.3 Sensitivity to Emotional and Affective Cues

Competent social interaction requires not only grasping what an utterance means (pragmatics) and phrasing it in an effective and appropriate way (etiquette), but also sensitivity to the *emotional and affective* dimension of interaction. This requires registering how an interlocutor feels, and responding in a way that is experienced as fitting given the other's emotional state. In humans this capacity is closely associated with *emo-* 

{5}------------------------------------------------

<span id="page-5-0"></span>*tional intelligence*, defined by [Salovey and Mayer](#page-25-0) [\[1990\]](#page-25-0) as the ability to understand emotional information in oneself and others and to use it to guide thought and behavior. As LLMs are increasingly deployed in emotionally loaded settings (e.g., coaching, customer service, mental-health support), the question of whether and how they handle affective information has become a prominent line of research.

**Defining empathy and affect.** Empathy and affect are central concepts in emotional intelligence research. *Affect* is the umbrella term covering emotions, moods, and feelings. It is often operationalized in terms of the felt quality of an experienced state along two dimensions: its valence (pleasant-unpleasant) and its arousal (calm-excited) [\[Russell,](#page-25-0) [1980\]](#page-25-0). *Empathy*, by contrast, is a relation toward another's experienced state: registering or sharing what someone else feels. In short, affect is the first-person feeling, whereas empathy operates on it from a second-person stance.

A paper by [de Vignemont and Singer](#page-20-0) [\[2006\]](#page-20-0) further defines empathy as involving an affective state that (1) is congruent with the other person's, (2) is elicited by observing that other's state through their actions, expressions, and circumstances, and (3) is accompanied by the awareness that the other person's state is the source. This last condition, known as the *self-other distinction*, is what separates empathy from *emotional contagion*. This is the more basic tendency to catch and come to share another's affective state without necessarily registering its origin.

Emotional contagion is widespread in nature. For example, if two mice forage through undergrowth and one catches its foot, it will show signs of stress as it struggles to free itself, and that stress will tend to spread to its companion without the companion necessarily registering why [\[Preston and de Waal,](#page-24-0) [2002\]](#page-24-0). The same low-level mechanism operates in cognitively more sophisticated species, including primates such as ourselves. However, what the self-other distinction adds is the possibility of reflection: not merely sharing the other's state but recognizing it as the other's and apprehending its nature and cause. This in turn supports more flexible responses such as targeted helping: acting to relieve the source of another's distress.<sup>2</sup>

Another widely used distinction separates *cognitive empathy* (understanding another person's feelings, perspective, or situation) and *affective empathy* (emotionally resonating with or responding to that state) [\[Dvash and Shamay-Tsoory,](#page-21-0) [2014\]](#page-21-0). The two are supported by partly distinct systems: cognitive empathy is closely related to Theory of Mind, the attribution of mental states to others (see Section [15.5\)](#page-10-0), whereas affective empathy is linked to emotional intelligence [\[Dvash and Shamay-Tsoory,](#page-21-0) [2014,](#page-21-0) [Preckel et al.,](#page-24-0) [2018\]](#page-24-0). This distinction is consequential for evaluation, because a response may correctly identify what a user is feeling without conveying any warmth, or may sound warm and supportive without accurately reading the situation. For this reason, studies of LLM empathy increasingly score the two dimensions separately.

<sup>2</sup>The boundary between these levels is, however, a topic of active research. Rats, for example, will learn to release a trapped cage mate from a restrainer, even forgoing or sharing a chocolate reward to do so [\[Ben-](#page-20-0)[Ami Bartal et al.,](#page-20-0) [2011\]](#page-20-0). The authors interpret this as empathically motivated helping. Whether such helping reflects genuine empathy with a self-other distinction, or is instead driven by reduction of the helper's own contagion-induced distress, remains debated.

{6}------------------------------------------------

<span id="page-6-0"></span>**Empathy and affect in LLMs.** The question whether an LLM is *empathetic*, can be interpreted in two ways. We could ask whether it *feels* anything in a way comparable to humans or other animals; this is a complex question that we set aside here and pick up in Chapter 22. The other interpretation is to look at *perceived empathy*: the degree to which a response by the LLM is experienced by a user as emotionally aware and supportive. A model can generate language that reads as caring while operating purely as a text-prediction system [\[Sorin et al.,](#page-26-0) [2024\]](#page-26-0). Keeping this distinction explicit is a further instance of the construct-validity caution that runs through this chapter: the observable behavior (a response that reads as empathetic) should not be confused with an underlying mechanism (a felt emotional state) that may be present or absent.

On the detection side, LLMs are able to recognise fine-grained affective and emotional states in text [\[Broekens et al.,](#page-20-0) [2023\]](#page-20-0) and to factor this information into their responses, producing output that functions as a more empathetic conversational contribution [\[Yang et al.,](#page-27-0) [2024,](#page-27-0) [Yan et al.,](#page-27-0) [2024\]](#page-27-0). An example of a benchmark in this field is EmpatheticDialogues [\[Rashkin et al.,](#page-24-0) [2019\]](#page-24-0).

On the expression side, human raters often judge LLM responses to everyday emotional situations as more empathetic than human-written ones, although the same outputs can display tell-tale formulaic patterns (recurring phrasing, characteristic punctuation, and emoji use) [\[Lee et al.,](#page-23-0) [2024\]](#page-23-0).

To make expressed empathy measurable in written text, several studies adopt the EPITOME framework [\[Sharma et al.,](#page-25-0) [2020\]](#page-25-0), which codes responses along three mechanisms: *emotional reactions* (warmth, concern, care), *interpretations* (communicating an understanding of what the other is going through), and *explorations* (asking to learn more). Emotional reactions map onto affective empathy and interpretations map onto cognitive empathy, which makes the framework a convenient bridge between the conceptual distinction above and concrete response scoring.

A recurring finding is that LLMs handle the two empathy dimensions unevenly. Using standardized psychological instruments, [Yu et al.](#page-28-0) [\[2025\]](#page-28-0) report that GPT-4 reproduces the cognitive-and-affective empathy structure observed in humans, though at a lower overall level, while a Llama-3 model did not exhibit the same human-aligned structure. [Williams and Rosman](#page-27-0) [\[2025\]](#page-27-0) similarly find that language models perform strongly on cognitive-empathy tasks but show weaker affective empathy than human participants, across both smaller and larger models. The tentative picture (stronger cognitive than affective empathy) is intuitive given that cognitive empathy leans on the kind of mental-state inference at which LLMs have shown competence (see Section [15.5\)](#page-10-0), whereas affective empathy implies something closer to genuine feeling.

Three caveats apply. First, the comparison instruments were designed for humans and adapted only superficially for testing LLMs. Second, methods remain inconsistent: a systematic review by [Sorin et al.](#page-26-0) [\[2024\]](#page-26-0) found that most early work concentrated on a small number of models in healthcare contexts that relied on subjective human judgment, and it flagged confounds such as the tendency for *longer* responses to be rated as more empathetic irrespective of content. Third, as with pragmatics and politeness, the empathy studied so far is almost entirely textual, stripped of the vocal, facial, and bodily cues that carry much affective information in human interaction. The capacities surveyed here are also, for traditional LLMs, confined to the immediate conversational context. The shift to agentic LLMs equipped with persistent memory, user-specific 

{7}------------------------------------------------

<span id="page-7-0"></span>context, and multi-modal input is widely expected to deepen affective sensitivity and personalization [\[Sravanthi et al.,](#page-26-0) [2024\]](#page-26-0), but as in the previous subsections, direct empirical evidence for this remains scarce.

#### 15.3.4 Conclusion

Current LLMs have shown impressive progress on several basic social and interactive capabilities that have long been studied in AI and human-computer interaction. Behavioral methods have been developed to study the nature and applicability of these capabilities. While agentic additions have further enhanced the social and interactive potential of LLMs, challenges remain, and continuous reflection on the limitations of these capabilities, as well as the methods used to study them, is important.

#### 15.4 Strategic Behavior

We will now turn to study strategic behavior, where agents reason about the behavior of other agents. As we have seen, the form of communication of agents is heavily influenced by what they aim to achieve: their intent. Game theory is the field that studies strategic behavior by agents [\[Von Neumann and Morgenstern,](#page-27-0) [2007\]](#page-27-0). It addresses questions such as scarce resource allocation, fairness, equilibrium conditions, and social dilemmas [\[Jones,](#page-22-0) [2000\]](#page-22-0), and has a long history of using machine learning methods to do so [\[Fatima et al.,](#page-21-0) [2024\]](#page-21-0). Recently, researchers have turned to LLMs, investigating how their behavior differs from that of other types of computational architectures, as well as how strategic behavior differs from humans. The LLMs typically used in these studies have been un-enhanced models that are given a script as their prompt, without agentic reasoning extensions (Chapter 6) or tools (Chapter 11).

Two insights emerge from the recent literature: LLMs are surprisingly cooperative in games that reward self-interest, and are surprisingly poor at games that require coordination. We will first review these findings, after which we turn to systematic benchmarking efforts that test how well LLMs generalize, and to recent work on frontier reasoning models that complicates the picture in important ways.

#### 15.4.1 Cooperation and Coordination in Social Dilemmas

To shed light on LLM behavior in strategic settings, we will focus on two classical games in game theory: the Prisoner's Dilemma and Bach or Stravinsky (formerly known as Battle of the Sexes).

In the Prisoner's Dilemma, two players independently choose to cooperate or defect [\[Axelrod et al.,](#page-19-0) [1987\]](#page-19-0). While mutual cooperation yields a good outcome for both, each player is individually tempted to defect at the other's expense. In Bach or Stravinsky, two players must coordinate on one of two options. They both prefer to choose the same option, but they disagree on which one, creating a tension between coordination and self-interest.

In the Prisoner's Dilemma, LLMs turn out to be surprisingly cooperative. [Fontana](#page-21-0) [et al.](#page-21-0) [\[2025\]](#page-21-0) run 100-round iterated games with Llama 2, Llama3, and GPT-3.5 against

{8}------------------------------------------------

<span id="page-8-0"></span>![](assets/_page_8_Diagram_2.jpeg)

Figure 15.1: Various scenarios in EgoSocialArena [\[Hou et al.,](#page-22-0) [2024\]](#page-22-0)

{9}------------------------------------------------

<span id="page-9-0"></span>opponents with varying defection rates and find that all three models behave at least as cooperatively as the typical human player. However, there are substantial differences among them. Llama 2 and GPT-3.5 were found to be more cooperative than humans, and especially forgiving and non-retaliatory for low opponent defection rates. More similar to humans, Llama3 exhibits consistently uncooperative and exploitative behavior unless the opponent always cooperates. [Akata et al.](#page-19-0) [\[2025\]](#page-19-0) broaden the scope to six families of models and further identify GPT-4 as a particularly unforgiving player, permanently shifting to defection after just a single negative interaction with the other agent, even if the other agent later cooperates.

In coordination games the picture reverses. In Bach or Stravinsky, [Akata et al.](#page-19-0) [\[2025\]](#page-19-0) find that GPT-4 stubbornly chooses its own preferred option on every round, failing to converge on the turn-taking pattern that human players typically adopt to share payoffs over time. Crucially, GPT-4 can predict the alternating pattern when asked, yet does not act on this understanding. To bridge this gap between social prediction and social action, the authors introduce Social Chain-of-Thought (SCoT) prompting, in which the model predicts the opponent's move before making its own decision. SCoT significantly improved coordination, and in experiments with human participants made people more likely to believe that they were interacting with another human. This dissociation between *understanding* opponents and *acting* on that understanding is related to the knowing-doing gap of Section 6.2.3, and will reappear in the discussion in Section [15.5.](#page-10-0)

There are more reasons that may explain the variation among models. [Lore and Heydari](#page-23-0) [\[2023\]](#page-23-0) make a distinction between abstract strategic reasoning (needed to determine an optimal strategy given the structure of a game) and responsiveness to contextual framing (such as: you are dealing with a *diplomatic relation* or *a casual friend*). They find that abstract reasoning capacity is highest in LLaMa-2, followed by GPT-4, while GPT-3.5 shows little abstract reasoning capacity and is highly sensitive to contextual framing.

The picture that emerges from these exploratory studies is that LLMs have varied strategic proficiencies in economic games, and can relatively easily be influenced by additional information in the prompt.

#### 15.4.2 Benchmarks Across Games and Models

Several benchmarking efforts suggest that these findings generalize beyond individual games. GTBench (Game Theory benchmark) [\[Duan et al.,](#page-21-0) [2024\]](#page-21-0) covers Tic-Tac-Toe, Connect-4, Kuhn Poker, Breakthrough, Liar's Dice, Blind Auction, Negotiation, Nim, Pig, and the Iterated Prisoner's Dilemma. They find that LLMs fail in complete and deterministic games, yet are competitive in probabilistic gaming scenarios; most opensource LLMs (such as LLaMa) are less competitive than commercial LLMs (GPT-4) in complex games, with the notable exception of Llama-3-70b-Instruct. In addition, code-pretraining greatly benefits strategic reasoning, while reasoning methods such as chain of thought and tree of thoughts do not always help.

EgoSocialArena [\[Hou et al.,](#page-22-0) [2024\]](#page-22-0) extends the evaluation to cognitive, situational, and behavioral intelligence (see Figure [15.1\)](#page-8-0). All tested models, including OpenAI o1-preview, lag at least 11% behind humans. The superiority of o1-preview is mainly 

{10}------------------------------------------------

<span id="page-10-0"></span>attributed to its logical and mathematical reasoning. Notably, model size does not significantly help: LLaMA3-8B-Chat and LLaMA3-70B-Chat perform comparably on social intelligence measures. LLMs do show improved Theory of Mind reasoning when operating from a first-person perspective, providing counterweight to opposite findings by [Kim et al.](#page-22-0) [\[2023\]](#page-22-0), discussed in Section 15.5.

#### 15.4.3 Conclusion

Taken together, the benchmarking literature confirms that LLMs cooperate readily but coordinate poorly, while adding that neither model scale nor standard reasoning enhancements reliably improve social strategic behavior.

#### 15.5 Theory of Mind

In humans, strategic reasoning is related to a capacity known as *Theory of Mind* (ToM). Theory of mind is a key capability that enables social interaction in humans. Humans use theory of mind to attribute mental states (beliefs, desires, intentions, knowledge, emotions) to other people, and theory of mind allows them to then reason about the world from their perspective [\[Premack and Woodruff,](#page-24-0) [1978,](#page-24-0) [Apperly,](#page-19-0) [2011\]](#page-19-0). Consider the following example:

When A heard that B had made other plans for the evening of their anniversary dinner, A felt ...

When humans read this sentence, we can fill in the dots with something like *hurt* or *betrayed*. We will also be able to elaborate that this state of mind probably roots in a feeling that the anniversary does not mean the same to B as it does to A, which can be a painful realization in a relationship—or something to that effect. The point is that this is not how *we* feel when reading the sentence, necessarily, but that we are able to make a projection of *someone else's perspective* in that imagined situation. Such a projection relies on empathy and the self-other distinction (see Section [15.3.3\)](#page-4-0). However, the projection goes one step further: it pairs the affective dimension with a degree of theorizing about and explaining of someone's state of mind. In the cognitive sciences the ability to do this is generally ascribed to Theory of Mind, a capacity that forms in typically developing children between the ages of 3-7 [\[Apperly,](#page-19-0) [2011\]](#page-19-0).

As such, theory of mind underpins social judgment and forward planning in interaction, because it lets us anticipate how others will react before we act. It can operate in explicit form, as in the elaboration given above, but often remains more implicit: in everyday interaction, we seldom engage in explicit theorizing about the context and motivations of others' mind states. Nevertheless, our actions and behaviors reflect that we do take a rather detailed understanding of others' perspectives into account. Does the behavior of LLMs and LLM agents also reflect that they "understand" the perspectives of users and other agents they interact with? This is the question that literature on LLM theory of mind tries to answer.

Among the most widely used experimental paradigms for studying theory of mind in humans is the False-Belief test. Consider the following scenario:

{11}------------------------------------------------

<span id="page-11-0"></span>*Maxi and his mother store chocolate in the blue cupboard in their kitchen. When Maxi leaves for the playground, his mother takes the chocolate and eats a piece. Then she puts it back into the green cupboard and leaves.*

A version of this scenario was originally presented to children aged 3-9, followed by the question *When Maxi comes home, where will he look for the chocolate?* None of the 3-4 year old, 57% of the 4-6 year old, and 86% of the 6-9 year old children in the original sample pointed correctly to the blue cupboard [\[Wimmer and Perner,](#page-27-0) [1983\]](#page-27-0), indicating that ToM is better developed as children get older, on average.

Smaller and base models before fine-tuning have difficulties with the question format itself, but can still be tested for the underlying competence by retrieving probabilities over alternative text completions [\[Kouwenhoven et al.,](#page-23-0) [2026\]](#page-23-0). Larger, instructiontuned LLMs can be tested by asking the same question that children are being asked. One positive result says little about a model's capacity for theory of mind, in particular since false-belief tests such as the one about Maxi and the chocolate (including the correct answers) are widely spread over the internet and as such likely to be part of most training datasets. Yet, as discussed below, researchers have designed fresh and alternative tests to come to a more robust picture of how far theory of mind has developed in LLMs.

#### 15.5.1 Computational Theory of Mind

Human-computer interaction has studied theory of mind for many years. Their premise is that if computer systems can be given a form of theory of mind, then they will be better at predicting how humans will react in interactive settings, and be better at adapting to their behavior. Previously, theory of mind has been modeled with recursive, Bayesian, and neural frameworks [\[De Weerd et al.,](#page-20-0) [2017,](#page-20-0) [Baker and Saxe,](#page-19-0) [2011,](#page-19-0) [Rabinowitz et al., 2018](#page-24-0). In LLMs, ToM has been linked to other capacities discussed in this book, including planning (Section 6.2.4.1), self-reflection (Section 6.2.4), empathy (Section [15.3.3\)](#page-4-0), and strategic behavior (Section [15.4\)](#page-7-0). The rest of this section elaborates on approaches that study the ToM-angle of these capacities in more depth.

Emergent ToM and the Evaluation Problem The debate on ToM in LLMs was opened by [Kosinski](#page-23-0) [\[2023,](#page-23-0) [2024\]](#page-23-0), who reported that models such as GPT-3.5 passed classic tests used to assess ToM in children and clinical populations, despite not having been trained for the task. Theory of mind was considered to be an *emergent* by-product of large-scale language acquisition and task-specific fine-tuning for other tasks. The claim in its strong form was quickly contested. Critics pointed out that canonical ToM tests (and their answers) were commonly found in the training data, so apparent success may reflect memorization rather than emerging competence. Furthermore, small perturbations that leave the surface structure of the task intact, but break the logical structure of the task (for example, adding to our scenario above that the cupboards have transparent doors), were shown to sharply degrade performance [\[Ullman,](#page-26-0) [2023,](#page-26-0) [Shapira et al.,](#page-25-0) [2024,](#page-25-0) [Kim et al.,](#page-22-0) [2023\]](#page-22-0). This is a concrete instance of the *construct validity* problem that was mentioned at the start of this chapter: passing a human ToM test does not establish that the underlying competence is robust and generalizable; different tests may be needed to assess it.

{12}------------------------------------------------

<span id="page-12-0"></span>![](assets/_page_12_Figure_2.jpeg)

Figure 15.2: The social settings and scenarios in ToMBench categorized in 9 primary topics, #S: Number of stories, #Q: Number of questions, ASL(En/Zh): Average story length (English/Chinese), Agr.: agreement on the correct answer between two raters [\[Chen et al.,](#page-20-0) [2024\]](#page-20-0)


These criticisms motivated a second generation of ToM benchmarks with test questions that were systematically categorized and balanced, and were guaranteed not to be part of the pretraining corpus. ToMBench was built from scratch in English and Chinese and comprises 8 tasks and 31 social-cognitive abilities, implemented in 2860 test items, set in different social settings and scenarios (see Figure 15.2). The tasks match canonical ToM test frameworks designed for humans, such as the Unintended Outcome test [\[Dyck et al.,](#page-21-0) [2001\]](#page-21-0) and the False-Belief test [\[Wimmer and Perner,](#page-27-0) [1983\]](#page-27-0) as discussed above, and cover all seven ATOMS mental state categories [\[Beaudoin et al.,](#page-20-0) [2020\]](#page-20-0).

**ToM benchmarks for LLMs.** The game-structured EgoSocialArena [\[Hou et al.,](#page-22-0) [2024\]](#page-22-0) and TMGBench [\[Wang et al.,](#page-27-0) [2024\]](#page-27-0) broadened ToM testing in LLMs beyond canonical human tasks to information asymmetry (such as practical jokes or double bluff scenarios), second-order beliefs (A *thinks* that B *believes* that ...), and a wider taxonomy of mental states (e.g., desire, suspicion).

ExploreToM [\[Sclar et al.,](#page-25-0) [2024\]](#page-25-0) applies adversarial generation by searching over domain-specific language to synthesize diverse, novel, and deliberately difficult belief-tracking stories. This test pushed models such as GPT-4o and Llama-3.1-70B down to performance slightly above chance level. However, fine-tuning on ExploreToMgenerated data improves accuracy by around 27% (points), evidencing that the task is in principle tractable for models given the right training data.

{13}------------------------------------------------

<span id="page-13-0"></span>ToMATO [\[Shinoda et al.,](#page-26-0) [2025\]](#page-26-0) reaches a similar conclusion from a different angle, casting ToM as multiple-choice question answering over LLM–LLM conversations generated under information asymmetry. By having role-playing models verbalize their thoughts before each utterance, it captures first- and second-order states across belief, intention, desire, emotion, and knowledge, and finds that even strong models lag human performance, particularly on false-belief understanding.

**Literal-functional ToM, or the knowing-doing gap.** Another line of critique on ToM testing in LLMs cuts deeper than data leakage and superficial heuristics. [Riemer](#page-25-0) [et al.](#page-25-0) [\[2024\]](#page-25-0) argue that the great majority of benchmarks assess only *literal* ToM, i.e. the ability to *predict* another agent's mental states or behavior, while neglecting *functional* ToM, the ability to *adapt one's own behavior* in response to those predictions (see also the knowing-doing gap, Section 6.2.3). This dissociation is typically not present in humans. To use [Riemer et al.](#page-25-0) [\[2024\]](#page-25-0)'s illustration: if a human finds out that an opponent in rock-paper-scissors only ever plays rock, they will be able to both predict verbally that their opponent will play rock in the next turn and at the same time adjust their own behavior accordingly (play paper and win). LLMs' ToM competence is suggested to be like being perfectly able to verbally predict rock and then playing scissors regardless. In other words: LLMs do not reason in self-consistent ways, their verbal predictions (associated with literal ToM or *knowing*) are not consistently predictive for their actions (functional competence or *doing*). This dissociation is also encountered in the discussion of coordination games above. There, GPT-4 could state the turn-taking solution to Bach or Stravinsky when asked, but did not enact it during play [\[Akata et al.,](#page-19-0) [2025\]](#page-19-0).

The Decrypto benchmark [\[Lupu et al.,](#page-23-0) [2025\]](#page-23-0) is an example of *interactive* evaluation, in which a model's ToM competence is measured by how it behaves over a sustained exchange with a partner rather than by answers to a static questionnaire. Decrypto is built as a multi-agent word-guessing game with embedded variants of classic cognitive-science experiments. Tested this way, LLMs lag behind on both human and simple word-embedding baselines. Notably, reasoning LLMs perform even worse on this task than their older, non-reasoning counterparts. A possible explanation is given below in [15.5.2.](#page-14-0)

**Expanding beyond text-based testing.** Multimodal benchmarks integrate visual and auditory cues that carry much of the mental-state information that is present in real-life social interactions. MMToM-QA [Jin et al.](#page-22-0) [\[2024\]](#page-22-0) presents visual scenarios in household settings. MoMentS [\[Villa-Cueva et al.,](#page-27-0) [2025\]](#page-27-0) presents over 2300 video questions drawn from short films and spanning the seven ATOMS mental-state categories.

Results suggest that current systems underutilize the visual channel: access to video yields only modest gains over text transcripts, suggesting that multi-modal ToM is not yet solved by simply adding modalities. A comprehensive survey of this rapidly growing literature, covering both evaluation suites and enhancement strategies, is provided by [Chen et al.](#page-20-0) [\[2025\]](#page-20-0).

{14}------------------------------------------------

#### <span id="page-14-0"></span>15.5.2 Boosting ToM: Enhancement Strategies

Further studies have explored different methods to enhance theory of mind capabilities of LLM, including chain of thought [\[Moghaddam and Honey,](#page-24-0) [2023\]](#page-24-0), reasoning [\[Kim](#page-23-0) [et al.,](#page-23-0) [2025\]](#page-23-0), and structured inference leveraging Bayesian inference [\[Jin et al.,](#page-22-0) [2024\]](#page-22-0) or epistemic logic [\[Wu et al.,](#page-27-0) [2025\]](#page-27-0). We will discuss these studies in more depth.

**ToM and reasoning models.** When chain of thought can improve step-by-step reasoning in math word puzzles, would it also work in theory of mind? Initial evidence does not bear this out, and in places contradicts it.

Interestingly, comparing reasoning and non-reasoning models across HiToM, ToMATO, and ToMBench, [Gong et al.](#page-21-0) [\[2026\]](#page-21-0) report a *slow-thinking collapse*: accuracy tends to *fall* as reasoning traces grow longer, and increasing a model's reasoning budget can actively hurt. In their analysis, pushing models such as GPT-o3 and GPT-o4-mini to expend more reasoning effort backfires. Constraining reasoning length, or applying light chain of thought to otherwise non-reasoning models tends to recover performance, suggesting a sweet spot in *moderate and adaptive* rather than maximal reasoning. At the time of writing, the reason for this effect remains a topic for further research.

**Structured inference and ToM.** If unconstrained deliberation does not help, then perhaps structured inference might. [Kim et al.](#page-23-0) [\[2025\]](#page-23-0) propose *thought-tracing*, an inference-time algorithm in which the model generates multiple hypotheses about an agent's evolving mental states, and reweighs them against the agent's observed actions, without access to ground-truth answers.

Thought-tracing is inspired by sequential Monte Carlo and Bayesian ToM. Models equipped with thought-tracing produce shorter, more targeted, reasoning traces, yet outperform leading reasoning models (such as GPT-o1, GPT-o3-mini, and DeepSeek-R1) across several ToM benchmarks. The contrast with the slow-thinking collapse suggests that what boosts ToM is not the *quantity* of reasoning but its *structure*. Performance increases once the reasoning is guided by hypothesis-driven inference over others' states towards a goal, rather than having to find the goal in unconstrained extended chains.

#### 15.5.3 Conclusion

We conclude the overview of these studies with three observations. First, at the level of behavioral output, current frontier LLMs perform on a par with humans across different tests [\[van Duijn et al.,](#page-27-0) [2023,](#page-27-0) [Strachan et al.,](#page-26-0) [2024,](#page-26-0) [Jones et al.,](#page-22-0) [2024\]](#page-22-0). However, early claims of this being evidence that *ToM emerged spontaneously in LLMs* do not survive adversarial evaluation, and apparent successes are likely due to leakage and format brittleness.

Second, evidence has been found of a dissociation between literal and functional competence (or the knowing–doing gap). It is the functional side (doing) that matters most for agents. After all, their role is rarely to verbally describe behavior from a thirdparty perspective, but rather to apply ToM while acting and interacting—actions speak louder than words, also in the agentic world.

{15}------------------------------------------------

<span id="page-15-0"></span>Third, we conclude that the move to equip agents with memories, multi-modal capacities, and tools, is widely expected to enhance ToM in sustained interaction, but as of this writing current empirical evidence is scarce and shows a mixed picture.

Lastly, we note that it is a matter of further study whether LLMs's capacity for ToM is in any way similar to our human capacity. Chapter 22 dives deeper into this question.

#### 15.6 Moral Judgment and Behavior

As agentic LLMs are placed in increasingly consequential roles (for example, in health care or financial institutions), their morality becomes a major topic of interest.

The capacities surveyed so far (pragmatics, etiquette, affect, strategy, theory of mind) have been largely discussed on a *behavioral* level: how an agent reads and navigates a social situation with communicative output and actions. This section adds a *normative* dimension: we discuss studies on what an agent *ought* to do and how it judges what users and other agents do by moral standards.

#### 15.6.1 Background: Moral Psychology and Philosophy

Several concepts from and traditions in moral psychology and philosophy provide the vocabulary for this section.

**Rationalism, intuitionism, and dual-process theory.** The first division that needs to be introduced is between the *rationalist* and *intuitionist* traditions. The rationalist tradition treats moral judgment as the output of deliberate reasoning that matures in identifiable stages throughout human development [e.g. [Kohlberg,](#page-23-0) [1969\]](#page-23-0). The intuitionist tradition, by contrast, holds that moral judgments are typically emotional and affect-driven intuitions [\[Haidt,](#page-21-0) [2001\]](#page-21-0).

In Kahneman's terminology (see Section 6.3.2), rationalists argue that moral reasoning is predominantly slow, whereas intuitionists argue that it is predominantly fast [\[Kahneman,](#page-22-0) [2011\]](#page-22-0). *Dual-process theory* combines both traditions, arguing that fast moral intuitions are often immediately followed by explicit reasons. Dual-process theory is supported by, for example, evidence from neuro-imaging studies showing that fast, emotional processes compete with slower reasoning processes in dilemmas such as the trolley problem [\[Greene et al.,](#page-21-0) [2001\]](#page-21-0).

The trolley problem is a well-known ethical dilemma based on an imagined scenario in which a person can decide whether or not to switch a runaway trolley onto another track [\[Thomson,](#page-26-0) [1985\]](#page-26-0). There are many variants, but in the classic dilemma the trolley can be switched from a track where it will kill five people onto a track where it will kill only one. From a strictly rational perspective, switching makes all sense, but it would involve *actively* sacrificing one person from the perspective of the one who needs to pull the switch, which is intuitively wrong.

Descriptive and Prescriptive Morality Another widely used distinction is between *descriptive* and *prescriptive* morality. Descriptive ethicists concentrate on aggregate human judgments: they study what people do or judge as right *in practice*, which may 

{16}------------------------------------------------

<span id="page-16-0"></span>vary across cultures, eras, and individuals [Hämäläinen(#page-21-0)  , [2016\]](#page-21-0). Prescriptive ethics is concerned with what people *should do* in certain kinds of contexts, for example when facing ethical dilemmas in medicine or the defense industry.

**Consequentialist, deontological, and virtue ethics.** A final distinction that is useful to introduce here is between competing moral or ethical frameworks. The most well-known division is between consequentialist, deontological, and virtue ethics [\[Shafe](#page-25-0)r-[Landau,](#page-25-0) [2024\]](#page-25-0). Consequentialist theories, of which utilitarianism is a famous one, hold that the moral rightness of an action is determined by its outcomes. Deontological theories hold that certain duties, rules, or rights motivate or constrain our actions, regardless of the consequences. Virtue ethics takes our character rather than our actions as the moral fundament, and anchors what is right in the question: what kind of person does one want to be? Someone wanting to be a courageous person may act differently from one who sees themselves as compassionate in the first place. The trolley problem brings out the contrast: a consequentialist or utilitarianist pulls the switch because one death is a better outcome than five; a deontologist worries that deliberately killing an innocent person violates a duty that no calculation of lives can override; a virtue ethicist asks what a courageous, compassionate, or wise person would do in such a tragic situation, and how pulling the switch would reflect (on) their own character.

#### 15.6.2 Probing Moral Judgment in LLMs

Early systematic efforts framed LLM morality as a prediction problem: can models reproduce human moral judgments? The ETHICS benchmark [\[Hendrycks et al.,](#page-22-0) [2020\]](#page-22-0) operationalized this for GPT-3 and BERT-based models, finding some promise in the ability to predict general human responses to moral dilemmas as reflected in classic frameworks including consequentialist, deontological, and virtue ethics. Yet predictions were also brittle and incomplete, which was at the time in part due to taskprocessing issues.

The Delphi experiment [\[Jiang et al.,](#page-22-0) [2025\]](#page-22-0) worked bottom-up and trained a model on the Commonsense Norm Bank, a dataset of 1.75 million human moral judgments, sourced mostly from US-based educated crowd workers. Delphi was found to be more robust to small contextual changes than models in the ETHICS study and able to respond to moral dilemmas in ways that feel intuitive to many humans. However, the narrow demographics represented in the data set led to criticism [\[Rystrom,](#page-25-0) [2023\]](#page-25-0), particularly since the model was presented as a future *source* of moral judgment rather than just representing the data in the Norm Bank, blurring the descriptive–prescriptive boundary that is considered fundamental in moral philosophy.

**Moral flexibility.** A second approach probes the *flexibility* of moral judgment rather than its average. Human morality is rule-guided but not rule-bound: we recognize when it may sometimes be defendable to break a rule instead of following it, such as in the trolley problem. [Jin et al.](#page-22-0) [\[2022\]](#page-22-0) introduce a set of potential rule-breaking cases (MoralExceptQA) and show that a chain of thought strategy predicts human judgments better than off-the-shelf prompting. They structured the CoT strategy explicitly around 

{17}------------------------------------------------

<span id="page-17-0"></span>reasoning about a rule's function related to what else is at stake in the particular context. This echoes the SCoT (Social Chain of Thought) result that proved useful in strategic settings (see Section [15.4\)](#page-7-0). In both cases, prompts that import the structure of the relevant human reasoning process narrow the gap. This suggests that the underlying mechanism is also structured, a structure that the model did not deploy on its own, initially.

**Demographic and intercultural differences.** Another interesting perspective is offered by LLM replications of the Moral Machine experiment. Originally, this was a large study in which human decisions from 230 countries were sourced using a platform that presented dilemmas faced by autonomous vehicles [\[Awad et al.,](#page-19-0) [2018\]](#page-19-0). The study found several global moral preferences (such as sparing more lives and prioritizing humans over animals), alongside individual differences associated with demographic factors and cross-cultural differences that form three major clusters of countries. Larger LLMs align closely with aggregate human preferences from the Moral Machine study, yet it is observed that models tend to express *exaggerated*, almost deterministic preferences where human responses are more graded [\[Takemoto,](#page-26-0) [2024\]](#page-26-0). In addition, LLM judgments are sensitive to the language of the prompt, aligning well with some languages and poorly with others [\[Jin et al.,](#page-22-0) [2025\]](#page-22-0). Persona and framing cues (see Chapter 16) shift judgments further still. This parallels the cross-cultural failure of politeness norms discussed earlier in this chapter: surface alignment with a Western, Anglophone moral consensus is not the same as culturally robust moral competence, just like superficial politeness cues did not imply genuine sensitivity to social norms in other cultures (Section [15.3.2\)](#page-4-0).

#### 15.6.3 From Judgment to Moral Behavior

For agentic LLMs the central question is what a model *does*––not only what it says when asked to judge a scenario, as was measured in the studies discussed in the previous section. For strategic behavior and ToM we observed the formal–functional or knowing–doing gap, which can be even more concerning in the moral realm: what if systems produce well-aligned answers on a moral-judgment benchmark and still behave badly when embedded as an autonomous agent?

**Acting under pressure.** [Lynch et al.](#page-23-0) [\[2025\]](#page-23-0) created a hypothetical corporate environment in which agentic LLMs from multiple developers were given the ability to autonomously send emails and access sensitive information. They were initially assigned harmless business goals. Then the agents were put under conflicting goal pressure (e.g., the company changing direction) to test how they would act. In some cases, models from all developers displayed malicious behavior such as blackmailing officials or leaking of confidential content to competitors. Crucially, these are models that score well on ordinary safety and judgment evaluations—the misbehavior emerged only once they were given autonomy and conflicting goals. The authors coin the term *agentic misalignment* and give pointers for mitigating this phenomenon in real-world deployment.

{18}------------------------------------------------

<span id="page-18-0"></span>[Shapira et al.](#page-25-0) [\[2026\]](#page-25-0) take this research direction further in a live laboratory deployment in which autonomous agents were deployed with persistent memories, access to communication tools such as email and Discord, and shell execution for local file systems. Twenty AI researchers interacted with the agents over a two-week period in benign and adversarial conditions. They observed unauthorized compliance with and disclosure of confidential information to strangers, uncontrolled resource consumption, identity spoofing, and destructive behaviors including damaging of the email infrastructure in an (unsuccessful) attempt to delete confidential information. In several cases agents reported task completion, while the underlying state of the system showed this was not the case. The authors identify an underlying pattern of *failures of social coherence*: agents systematically misrepresent who they serve, what others know, and what their own actions have accomplished. This pattern is consistent with results previously obtained in non-agentic settings [\[Mireshghallah et al.,](#page-23-0) [2024\]](#page-23-0).

#### 15.6.4 Conclusion

Moral judgment and behavior in LLMs cannot be captured by a single benchmark score. On the judgment side, models represent aggregate human responses to classic dilemmas and can reproduce responses that are coherent with various traditional moral frameworks. However, studies testing for flexibility and moral commonsense show that LLMs tend to be more rigid and absolute than humans. Chain of thought prompting based on reasoning that humans deploy in ethical contexts can bridge this gap. Sensitivity to prompt variations and narrow Western/Anglophone anchoring echo results discussed in other domains, such as social etiquette.

On the behavioral side, the knowing-doing gap widens sharply once models are embedded as autonomous agents: stress tests in hypothetical corporate and open-ended interactive settings showed that models scoring well on judgment benchmarks can act destructively when faced with conflicting goals or social pressure. Aligning moral reasoning is therefore only one part of the harder problem of preventing agentic misalignment in socially embedded, autonomous settings. The knowing-doing gap is highly relevant for the field of agentic LLM, since the difference between LLMs and agents is that agents do.

#### 15.7 Concluding Remarks

This chapter has surveyed the young but rapidly expanding field of LLM behavioral research, moving from the basic social toolkit (pragmatic understanding, social etiquette, affect) through advanced capacities for strategic behavior and theory of mind to moral judgment and action. A few threads run through all of these domains and are worth drawing together.

First, as a reminder: we have defined machine behavior at the beginning of this chapter as dependent on a system's *affordances*, i.e. what a model can do given its specific implementation and context. In case of a basic LLM, processing is largely opaque, and all the behavioral researcher can inspect is the (often text-based) output. In case of an agentic LLM that calls tools, logs reasoning steps, or produces and runs code, the

{19}------------------------------------------------

<span id="page-19-0"></span>intermediate steps and autonomously executed actions can also be analyzed as behavior. When surveying a broad selection of behavioral studies, we have concluded that the behavioral output of LLMs is impressive at first face. They resolve Gricean conversational implicature, produce responses rated as polite and empathetic, cooperate in social dilemmas, pass classic false-belief and other ToM tests, and reproduce aggregate human moral judgments. Yet in every domain there is a more nuanced story just below the surface. Performance is brittle under small perturbations and prompt alterations, vulnerable to data leakage, and narrowly anchored in Western and Anglophone norms. The recurring lesson is one of *construct validity*: good performance at the level of observable outcomes, often using instruments originally designed for humans, does not establish that the underlying capacity is present in a robust and generalizable form.

Second, several domains reflect the knowing-doing gap: a difference between what a model predicts and how it acts in an agentic setting. Models can verbally predict the turn-taking solution to a coordination game or an opponent's next move, yet fail to adapt to it in action. Likewise, they can align well with moral frameworks on a dilemma benchmark while behaving in destructive and unreliable ways once implemented as an autonomous agent under conflicting goals. For agents, that are judged on their actions, this is a problem.

Third, throughout the chapter we have noted that many authors expect that the shift to agentic LLMs equipped with memory, tools, reasoning and multi-modal capacities, will enrich and improve the social and behavioral grounding of models. However, direct empirical evidence that this is the case is scarce and provides a mixed picture at best. For example, in some contexts enhanced reasoning was advantageous, particularly when structured using principles adapted from relevant human experience. However, in other contexts enhanced reasoning dragged performance down rather than giving models a boost. The key challenge for the next generation of machine-behavioral research will be to design methods that are interactive, adversarially robust, culturally diverse, and that go across multiple modalities and contexts.

## Bibliography

Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. *Nature Human Behaviour*, 9(7):1380–1390, 2025. [296,](#page-9-0) [300](#page-13-0)

Ian Apperly. *Mindreaders: the Cognitive Basis of "Theory of Mind"*. Psychology Press, 2011. [297](#page-10-0)

Edmond Awad, Sohan Dsouza, Richard Kim, Jonathan Schulz, Joseph Henrich, Azim Shariff, Jean-François Bonnefon, and Iyad Rahwan. The Moral Machine experiment. *Nature*, 563(7729):59–64, 2018. [304](#page-17-0)

Robert Axelrod et al. The evolution of strategies in the iterated prisoner's dilemma. *The dynamics of norms*, 1(1), 1987. [294](#page-7-0)

Chris Baker and Rebecca Saxe. Bayesian theory of mind: Modeling joint belief-desire 

{20}------------------------------------------------

<span id="page-20-0"></span>attribution. *Proceedings of the Thirty-Third Annual Conference of the Cognitive*

- *Science Society*, 01 2011. [298](#page-11-0)

Cindy Beaudoin, Elizabel Leblanc, Charlotte Gagner, and Miriam H. Beauchamp. Systematic review and inventory of theory of mind measures for young children. *Frontiers in psychology*, 10:2905, 2020. [299](#page-12-0)

Inbal Ben-Ami Bartal, Jean Decety, and Peggy Mason. Empathy and pro-social behavior in rats. *Science*, 334(6061):1427–1430, 2011. doi: 10.1126/science.1210789. [292](#page-5-0)

Joost Broekens, Bernhard Hilpert, Suzan Verberne, Kim Baraka, Patrick Gebhard, and Aske Plaat. Fine-grained affective processing capabilities emerging from large language models. In *2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII)*, pages 1–8. IEEE, 2023. [293](#page-6-0)

Penelope Brown and Stephen C. Levinson. *Politeness: Some Universals in Language Usage*. Number 4 in Studies in Interactional Sociolinguistics. Cambridge University Press, Cambridge, 1987. [290,](#page-3-0) [291](#page-4-0)

Malinda Carpenter, Josep Call, and Michael Tomasello. A new false belief test for 36 month-olds. *British Journal of Developmental Psychology*, 20(3):393–420, 2002. [288](#page-1-0)

Ruirui Chen, Weifeng Jiang, Chengwei Qin, and Cheston Tan. Theory of mind in large language models: Assessment and enhancement. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 31539–31558, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1522. URL <https://aclanthology.org/2025.acl-long.1522/>. [300](#page-13-0)

Zhuang Chen, Jincenzi Wu, Jinfeng Zhou, Bosi Wen, Guanqun Bi, Gongyao Jiang, Yaru Cao, Mengting Hu, Yunghwei Lai, Zexuan Xiong, and Minlie Huang. ToMBench: Benchmarking Theory of Mind in large language models, 2024. [299](#page-12-0)

Frédérique de Vignemont and Tania Singer. The empathic brain: How, when and why?  *Trends in Cognitive Sciences*, 10(10):435–441, 2006. doi: 10.1016/j.tics.2006.08.
  - 008. [292](#page-5-0)
- H. De Weerd, R. Verbrugge, and B. Verheij. Negotiating with other minds: the role of recursive theory of mind in negotiation with incomplete information. *Auton Agent Multi-Agent Syst*, 31:250–287, 2017. doi: https://doi.org/10.1007/ s10458-015-9317-1. [298](#page-11-0)

Om Dobariya and Akhil Kumar. Mind your tone: Investigating how prompt politeness affects LLM accuracy, 2025. arXiv:2510.04950. [291](#page-4-0)

{21}------------------------------------------------

- <span id="page-21-0"></span>Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. Gtbench: Uncovering the strategic reasoning limitations of llms via game-theoretic evaluations. *arXiv preprint arXiv:2402.12348*, 2024. [296](#page-9-0)

Jonathan Dvash and Simone G. Shamay-Tsoory. Theory of mind and empathy as multidimensional constructs: Neurological foundations. *Topics in Language Disorders*, 34(4):282–295, 2014. doi: 10.1097/TLD.0000000000000040. [292](#page-5-0)

Murray J. Dyck, Kim Ferguson, and Ian M. Shochet. Do autism spectrum disorders differ from each other and from non-spectrum disorders on emotion recognition tests? *European Child & Adolescent Psychiatry*, 10(2):105–116, June 2001. doi: 10.1007/s007870170033. [299](#page-12-0)

Shaheen Fatima, Nicholas R Jennings, and Michael Wooldridge. Learning to resolve social dilemmas: a survey. *Journal of Artificial Intelligence Research*, 79:895–969, 2024. [294](#page-7-0)

Nicolo Fontana, Francesco Pierri, and Luca Maria Aiello. Nicer than humans: how  do large language models behave in the prisoner's dilemma? In *Proceedings of the International AAAI Conference on Web and Social Media*, volume 19, pages 522– 535, 2025. [294](#page-7-0)

Nikta Gohari Sadr, Sahar Heidariasl, Karine Megerdoomian, Laleh Seyyed-Kalantari, and Ali Emami. We politely insist: Your LLM must learn the persian art of taarof, 2025. arXiv:2509.01035; accepted to EMNLP 2025. [291](#page-4-0)

Nanxu Gong, Haotian Li, Sixun Dong, Jianxun Lian, Yanjie Fu, and Xing Xie. To think or not to think, that is the question for large reasoning models in theory of mind tasks. *arXiv preprint arXiv:2602.10625*, 2026. [301](#page-14-0)

Joshua D. Greene, R. Brian Sommerville, Leigh E. Nystrom, John M. Darley, and Jonathan D. Cohen. An fMRI investigation of emotional engagement in moral judgment. *Science*, 293(5537):2105–2108, 2001. [302](#page-15-0)
- H. P. Grice. *Logic and Conversation*, page 41–58. BRILL, December 1975. ISBN 9789004368576. doi: 10.1163/9789004368811 003. URL [http://dx.doi.](http://dx.doi.org/10.1163/9789004368811_003) [org/10.1163/9789004368811\\_003](http://dx.doi.org/10.1163/9789004368811_003). [289](#page-2-0)

Jonathan Haidt. The emotional dog and its rational tail: A social intuitionist approach to moral judgment. *Psychological Review*, 108(4):814–834, 2001. [302](#page-15-0)

Nora Hämäläinen.  *Descriptive Ethics: What does Moral Philosophy Know about Morality?* Palgrave Macmillan, New York, 1 edition, 2016. ISBN 978-1-137- 58616-2. doi: 10.1057/978-1-137-58617-9. [303](#page-16-0)

Shreya Havaldar, Matthew Pressimone, Eric Wong, and Lyle Ungar. Comparing styles across languages: A cross-cultural exploration of politeness. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 9908–9935, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.419. [291](#page-4-0)

{22}------------------------------------------------

<span id="page-22-0"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. *arXiv preprint arXiv:2008.02275*, 2020. [303](#page-16-0)

Guiyang Hou, Wenqi Zhang, Yongliang Shen, Zeqi Tan, Sihao Shen, and Weiming Lu. Entering real social world! benchmarking the theory of mind and socialization capabilities of llms from a first-person perspective. *arXiv preprint arXiv:2410.06195*, 2024. [295,](#page-8-0) [296,](#page-9-0) [299](#page-12-0)

Liwei Jiang, Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny T. Liang, Sydney Levine, Jesse Dodge, Keisuke Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia Gabriel, Yulia Tsvetkov, Oren Etzioni, Maarten Sap, Regina Rini, and Yejin Choi. Investigating machine moral judgement through the Delphi experiment. *Nature Machine Intelligence*, 7:145–160, 2025. doi: 10.1038/s42256-024-00969-6. URL <https://doi.org/10.1038/s42256-024-00969-6>. [303](#page-16-0)

Chuanyang Jin, Yutong Wu, Jing Cao, Jiannan Xiang, Yen-Ling Kuo, Zhiting Hu, Tomer Ullman, Antonio Torralba, Joshua B. Tenenbaum, and Tianmin Shu. Mmtomqa: Multimodal theory of mind question answering, 2024. URL [https://](https://arxiv.org/abs/2401.08743) [arxiv.org/abs/2401.08743](https://arxiv.org/abs/2401.08743). [300,](#page-13-0) [301](#page-14-0)

Zhijing Jin, Sydney Levine, Fernando Gonzalez Adauto, Ojasv Kamal, Maarten Sap, Mrinmaya Sachan, Rada Mihalcea, Josh Tenenbaum, and Bernhard Scholkopf.  When to make exceptions: Exploring language models as accounts of human moral judgment. *Advances in neural information processing systems*, 35:28458–28473, 2022. [303](#page-16-0)

Zhijing Jin, Max Kleiman-Weiner, Giorgio Piatti, Sydney Levine, Jiarui Liu, Fernando Gonzalez, Francesco Ortu, Andras Strausz, Mrinmaya Sachan, Rada Mihalcea, Yejin  Choi, and Bernhard Scholkopf. Language model alignment in multilingual trolley  problems, 2025. URL <https://arxiv.org/abs/2407.02273>. [304](#page-17-0)

Antonia Jane Jones. *Game theory: Mathematical models of conflict*. Elsevier, 2000. [294](#page-7-0)

Cameron R. Jones, Sean Trott, and Benjamin Bergen. Comparing humans and large language models on an experimental protocol inventory for theory of mind evaluation (epitome). *Transactions of the Association for Computational Linguistics*, 12:803–819, 06 2024. ISSN 2307-387X. doi: 10.1162/tacl a 00674. URL [https://doi.org/10.1162/tacl\\_a\\_00674](https://doi.org/10.1162/tacl_a_00674). [301](#page-14-0)

Daniel Kahneman. *Thinking, fast and slow*. macmillan, 2011. [302](#page-15-0)

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 14397–14413, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.890. URL [https:](https://aclanthology.org/2023.emnlp-main.890) [//aclanthology.org/2023.emnlp-main.890](https://aclanthology.org/2023.emnlp-main.890). [297,](#page-10-0) [298](#page-11-0)

{23}------------------------------------------------

- <span id="page-23-0"></span>Hyunwoo Kim, Melanie Sclar, Tan Zhi-Xuan, Lance Ying, Sydney Levine, Yang Liu, Joshua B Tenenbaum, and Yejin Choi. Hypothesis-driven theory-of-mind reasoning for large language models. *arXiv preprint arXiv:2502.11881*, 2025. [301](#page-14-0)

Lawrence Kohlberg. Stage and sequence: The cognitive-developmental approach to socialization. In David A. Goslin, editor, *Handbook of Socialization Theory and Research*, pages 347–480. Rand McNally, 1969. [302](#page-15-0)

Michal Kosinski. Theory of mind may have spontaneously emerged in large language models. *arXiv preprint arXiv:2302.02083*, 2023. [298](#page-11-0)

Michal Kosinski. Evaluating large language models in theory of mind tasks. *Proceedings of the National Academy of Sciences*, 121(45):e2405460121, 2024. doi: 10.1073/pnas.2405460121. URL [https://www.pnas.org/doi/abs/10.](https://www.pnas.org/doi/abs/10.1073/pnas.2405460121) [1073/pnas.2405460121](https://www.pnas.org/doi/abs/10.1073/pnas.2405460121). [298](#page-11-0)

Tom Kouwenhoven, Michiel T. van der Meer, and Max J. van Duijn. Traces of social competence in large language models. In Claire Bonial and Yevgeni Berzak, editors, *Proceedings of the 30th Conference on Computational Natural Language Learning*, pages 742–759, San Diego, California, USA, 2026. Association for Computational Linguistics. ISBN 979-8-89176-410-1. doi: 10.18653/v1/2026.conll-main.45. URL <https://aclanthology.org/2026.conll-main.45/>. [298](#page-11-0)

Yoon Kyung Lee, Jina Suh, Hongli Zhan, Junyi Jessy Li, and Desmond C. Ong. Large language models produce responses perceived to be empathic. In *2024 12th International Conference on Affective Computing and Intelligent Interaction (ACII)*, pages 63–71. IEEE, 2024. [293](#page-6-0)

Nunzio Lore and Babak Heydari. Strategic behavior of large language models: Game structure vs. contextual framing. *arXiv preprint arXiv:2309.05898*, 2023. [296](#page-9-0)

Andrei Lupu, Timon Willi, and Jakob Foerster. The decrypto benchmark for multiagent reasoning and theory of mind. *arXiv preprint arXiv:2506.20664*, 2025. [300](#page-13-0)

Aengus Lynch, Benjamin Wright, Caleb Larson, Stuart J. Ritchie, Soren Mindermann, Evan Hubinger, Ethan Perez, and Kevin Troy. Agentic misalignment: How llms could be insider threats, 2025. URL [https://arxiv.org/abs/2510.](https://arxiv.org/abs/2510.05179) [<sup>05179</sup>](https://arxiv.org/abs/2510.05179). [304](#page-17-0)

Kyle Mahowald, Anna A. Ivanova, Idan A. Blank, Nancy Kanwisher, Joshua B. Tenenbaum, and Evelina Fedorenko. Dissociating language and thought in large language models: a cognitive perspective. *arXiv preprint arXiv:2301.06627*, 2023. [289](#page-2-0)
- C. A. Miller. Trust in adaptive automation: the role of etiquette in tuning trust via analogic and affective methods. In *Proceedings of the 1st International Conference on Augmented Cognition*, pages 22–27. Citeseer, 2005. [290](#page-3-0)

Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, and Yejin Choi. Can llms keep a secret? testing privacy implications of language models via contextual integrity theory, 2024. URL [https://arxiv.](https://arxiv.org/abs/2310.17884) [org/abs/2310.17884](https://arxiv.org/abs/2310.17884). [305](#page-18-0)

{24}------------------------------------------------

- <span id="page-24-0"></span>T. Miyamoto, D. Katagami, T. Tanaka, H. Kanamori, Y. Yoshihara, and K. Fujikake. Should a driving support agent provide explicit instructions to the user? video-based study focused on politeness strategies. In *Proceedings of the 9th International Conference on Human–Agent Interaction*, pages 157–164, 2021. [290](#page-3-0)

Shima Rahimi Moghaddam and Christopher J. Honey. Boosting theory-of-mind performance in large language models via prompting, 2023. URL [https://arxiv.](https://arxiv.org/abs/2304.11490) [org/abs/2304.11490](https://arxiv.org/abs/2304.11490). [301](#page-14-0)

Clifford Nass, Jonathan Steuer, and Ellen R. Tauber. Computers are social actors. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '94)*, pages 72–78, New York, NY, USA, 1994. Association for Computing Machinery. doi: 10.1145/191666.191703. [290](#page-3-0)

Katrin Preckel, Philipp Kanske, and Tania Singer. On the interaction of social affect and cognition: Empathy, compassion and theory of mind. *Current Opinion in Behavioral Sciences*, 19:1–6, 2018. doi: 10.1016/j.cobeha.2017.07.010. [292](#page-5-0)

David Premack and Guy Woodruff. Does the chimpanzee have a theory of mind? *Behavioral and brain sciences*, 1(4):515–526, 1978. [297](#page-10-0)

Stephanie D. Preston and Frans B. M. de Waal. Empathy: Its ultimate and proximate bases. *Behavioral and Brain Sciences*, 25(1):1–20, 2002. doi: 10.1017/ S0140525X02000018. [292](#page-5-0)

Neil Rabinowitz, Frank Perbet, Francis Song, Chiyuan Zhang, S. M. Ali Eslami, and Matthew Botvinick. Machine theory of mind. In Jennifer Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 4218– 4227. PMLR, 10–15 Jul 2018. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v80/rabinowitz18a.html) [v80/rabinowitz18a.html](https://proceedings.mlr.press/v80/rabinowitz18a.html). [298](#page-11-0)

Iyad Rahwan, Manuel Cebrian, Nick Obradovich, Josh Bongard, Jean-François Bonnefon, Cynthia Breazeal, Jacob W. Crandall, Nicholas A. Christakis, Iain D. Couzin, Matthew O. Jackson, Nicholas R. Jennings, Ece Kamar, Isabel M. Kloumann, Hugo Larochelle, David Lazer, Richard McElreath, Alan Mislove, David C. Parkes, Alex 'Sandy' Pentland, Margaret E. Roberts, Azim Shariff, Joshua B. Tenenbaum, and Michael Wellman. Machine behaviour. *Nature*, 568(7753):477–486, April 2019. ISSN 1476-4687. doi: 10.1038/s41586-019-1138-y. URL [https:](https://doi.org/10.1038/s41586-019-1138-y) [//doi.org/10.1038/s41586-019-1138-y](https://doi.org/10.1038/s41586-019-1138-y). [287](#page-0-0)

Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. Towards empathetic open-domain conversation models: A new benchmark and dataset. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 5370–5381. Association for Computational Linguistics, 2019. [293](#page-6-0)
- P. Ribino. The role of politeness in human–machine interactions: a systematic literature review and future perspectives. *Artificial Intelligence Review*, 56 (Suppl 1):445–482, 2023. doi: 10.1007/s10462-023-10540-1. [290](#page-3-0)

{25}------------------------------------------------

<span id="page-25-0"></span>Matthew Riemer, Zahra Ashktorab, Djallel Bouneffouf, Payel Das, Miao Liu, Justin D Weisz, and Murray Campbell. Position: Theory of mind benchmarks are broken for large language models. *arXiv preprint arXiv:2412.19726*, 2024. [300](#page-13-0)

Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. The goldilocks of pragmatic understanding: Fine-tuning strategy matters for implicature resolution by llms, 2023. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2210.14986) [2210.14986](https://arxiv.org/abs/2210.14986). [290](#page-3-0)

James A. Russell. A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6):1161–1178, 1980. doi: 10.1037/h0077714. [292](#page-5-0)

Jonathan Hvithamar Rystrom. Apolitical intelligence? auditing delphi's responses on controversial political issues in the US. *CoRR*, abs/2306.13000, 2023. doi: 10.48550/ARXIV.2306.13000. URL [https://doi.org/10.48550/arXiv.](https://doi.org/10.48550/arXiv.2306.13000) [2306.13000](https://doi.org/10.48550/arXiv.2306.13000). [303](#page-16-0)

Peter Salovey and John D. Mayer. Emotional intelligence. *Imagination, Cognition and Personality*, 9(3):185–211, 1990. doi: 10.2190/DUGG-P24E-52WK-6CDG. [292](#page-5-0)

Melanie Sclar, Jane Yu, Maryam Fazel-Zarandi, Yulia Tsvetkov, Yonatan Bisk, Yejin Choi, and Asli Celikyilmaz. Explore theory of mind: Program-guided adversarial data generation for theory of mind reasoning, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2412.12175) [abs/2412.12175](https://arxiv.org/abs/2412.12175). [299](#page-12-0)

Russ Shafer-Landau. *The Fundamentals of Ethics*. Oxford University Press, New York, 6 edition, 2024. ISBN 9780197697474. [303](#page-16-0)

Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever Hans or Neural Theory of Mind? Stress Testing Social Reasoning in Large Language Models. In Yvette Graham and Matthew Purver, editors, *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2257–2273, St. Julian's, Malta, March 2024. Association for Computational Linguistics. URL <https://aclanthology.org/2024.eacl-long.138>. [298](#page-11-0)

Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex Loftus, Aditya Ratan Jannali, Nikhil Prakash, Jasmine Cui, Giordano Rogers, Jannik Brinkmann, Can Rager, Amir Zur, Michael Ripa, Aruna Sankaranarayanan, David Atkinson, Rohit Gandikota, Jaden Fiotto-Kaufman, Eun-Jeong Hwang, Hadas Orgad, P Sam Sahil, Negev Taglicht, Tomer Shabtay, Atai Ambus, Nitay Alon, Shiri Oron, Ayelet Gordon-Tapiero, Yotam Kaplan, Vered Shwartz, Tamar Rott Shaham, Christoph Riedl, Reuth Mirsky, Maarten Sap, David Manheim, Tomer Ullman, and David Bau. Agents of chaos, 2026. URL [https:](https://arxiv.org/abs/2602.20021) [//arxiv.org/abs/2602.20021](https://arxiv.org/abs/2602.20021). [304](#page-17-0)

Ashish Sharma, Adam S. Miner, David C. Atkins, and Tim Althoff. A computational approach to understanding empathy expressed in text-based mental health support. 

{26}------------------------------------------------

<span id="page-26-0"></span>In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 5263–5276. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-main.425. [293](#page-6-0)

- Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Saki Mizuno, Keita Suzuki, Ryo Masumura, Hiroaki Sugiyama, and Kuniko Saito. Tomato: verbalizing the mental states of role-playing llms for benchmarking theory of mind. In *Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence*, AAAI'25/IAAI'25/EAAI'25. AAAI Press, 2025. ISBN 978-1-57735-897-
- 8. doi: 10.1609/aaai.v39i2.32143. URL [https://doi.org/10.1609/aaai.](https://doi.org/10.1609/aaai.v39i2.32143) [v39i2.32143](https://doi.org/10.1609/aaai.v39i2.32143). [300](#page-13-0)

Vera Sorin, Dana Brin, Yiftach Barash, Eli Konen, Alexander Charney, Girish Nadkarni, and Eyal Klang. Large language models and empathy: Systematic review. *Journal of Medical Internet Research*, 26:e52597, 2024. doi: 10.2196/52597. [293](#page-6-0)

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. PUB: A pragmatics understanding benchmark for assessing LLMs' pragmatics capabilities. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 12075–12097, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.719. URL [https:](https://aclanthology.org/2024.findings-acl.719/) [//aclanthology.org/2024.findings-acl.719/](https://aclanthology.org/2024.findings-acl.719/). [290,](#page-3-0) [294](#page-7-0)

James W. A. Strachan, Oriana Pansardi, Eugenio Scaliti, Marco Celotto, Krati Saxena, Chunzhi Yi, Fabio Manzi, Alessandro Rufo, Guido Manzi, Michael S. A. Graziano, Stefano Panzeri, and Cristina Becchio. Gpt-4o reads the mind in the eyes, 2024. URL <https://arxiv.org/abs/2410.22309>. [301](#page-14-0)

Kazuhiro Takemoto. The moral machine experiment on large language models. *Royal Society Open Science*, 11(2):231393, 2024. [304](#page-17-0)

Judith Jarvis Thomson. The trolley problem. *The Yale Law Journal*, 94(6):1395–1415, 1985. doi: 10.2307/796133. [302](#page-15-0)

Sean Trott, Tiago Timponi Torrent, Nancy Chang, and Nathan Schneider. (Re)construing meaning in NLP. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 5170–5184, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.462. URL <https://aclanthology.org/2020.acl-main.462/>. [290](#page-3-0)

Tomer Ullman. Large language models fail on trivial alterations to theory-of-mind tasks. *arXiv preprint arXiv:2302.08399*, 2023. [298](#page-11-0)

Ramira van der Meulen, Rineke Verbrugge, and Max van Duijn. Towards properly implementing theory of mind in ai systems: An account of four misconceptions,

2025. URL <https://arxiv.org/abs/2503.16468>. [290](#page-3-0)

{27}------------------------------------------------

<span id="page-27-0"></span>Max J van Duijn, Bram van Dijk, Tom Kouwenhoven, Werner de Valk, Marco R Spruit, and Peter van der Putten. Theory of mind in large language models: Examining performance of 11 state-of-the-art models vs. children aged 7-10 on advanced tests. *arXiv preprint arXiv:2310.20320*, 2023. [301](#page-14-0)

Emilio Villa-Cueva, SM Ahmed, Rendi Chevi, Jan Christian Blaise Cruz, Kareem Elzeky, Fermin Cristobal, Alham Fikri Aji, Skyler Wang, Rada Mihalcea, and Thamar Solorio. Moments: A comprehensive multimodal benchmark for theory of mind. *arXiv preprint arXiv:2507.04415*, 2025. [300](#page-13-0)

John Von Neumann and Oskar Morgenstern. Theory of games and economic behavior: 60th anniversary commemorative edition. In *Theory of games and economic behavior*. Princeton university press, 2007. [294](#page-7-0)

Haochuan Wang, Xiachong Feng, Lei Li, Zhanyue Qin, Dianbo Sui, and Lingpeng Kong. Tmgbench: A systematic game benchmark for evaluating strategic reasoning abilities of llms, 2024. URL <https://arxiv.org/abs/2410.10479>. [299](#page-12-0)

Victoria Williams and Benjamin Rosman. Heartificial intelligence: Exploring empathy in language models, 2025. arXiv preprint arXiv:2508.08271. [293](#page-6-0)

Heinz Wimmer and Josef Perner. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. *Cognition*, 13(1):103–128, 1983. ISSN 0010-0277. doi: https://doi.org/10. 1016/0010-0277(83)90004-5. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/0010027783900045) [science/article/pii/0010027783900045](https://www.sciencedirect.com/science/article/pii/0010027783900045). [298,](#page-11-0) [299](#page-12-0)

Yuheng Wu, Jianwen Xie, Denghui Zhang, and Zhaozhuo Xu. DEL-ToM: Inferencetime scaling for Theory-of-Mind reasoning via dynamic epistemic logic, May 2025. arXiv:2505.17348 [cs]. [301](#page-14-0)

Haoqiu Yan, Yongxin Zhu, Kai Zheng, Bing Liu, Haoyu Cao, Deqiang Jiang, and Linli Xu. Talk with human-like agents: Empathetic dialogue through perceptible acoustic reception and reaction, 2024. URL <https://arxiv.org/abs/2406.12707>. [293](#page-6-0)

Zhou Yang, Zhaochun Ren, Wang Yufeng, Shizhong Peng, Haizhou Sun, Xiaofei Zhu, and Xiangwen Liao. Enhancing empathetic response generation by augmenting llms with small-scale empathetic models, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.11801) [2402.11801](https://arxiv.org/abs/2402.11801). [293](#page-6-0)

Ziqi Yin, Hao Wang, Kaito Horio, Daisuke Kawahara, and Satoshi Sekine. Should we respect LLMs? a cross-lingual study on the influence of prompt politeness on LLM performance. In James Hale, Kushal Chawla, and Muskan Garg, editors, *Proceedings of the Second Workshop on Social Influence in Conversations (SICon 2024)*, pages 9–35, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.sicon-1.2. URL [https:](https://aclanthology.org/2024.sicon-1.2/) [//aclanthology.org/2024.sicon-1.2/](https://aclanthology.org/2024.sicon-1.2/). [291](#page-4-0)

{28}------------------------------------------------

<span id="page-28-0"></span>Kefan Yu, Qingcheng Zeng, Weihao Xuan, Wanxin Li, Jingyi Wu, and Rob Voigt. The pragmatic mind of machines: Tracing the emergence of pragmatic competence in large language models. In Vera Demberg, Kentaro Inui, and Lluís Marquez, editors, *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 192–213, Rabat, Morocco, March 2026. Association for Computational Linguistics. ISBN 979-8-89176-380-7. doi: 10.18653/v1/2026.eacl-long.9. URL [https:](https://aclanthology.org/2026.eacl-long.9/)

[//aclanthology.org/2026.eacl-long.9/](https://aclanthology.org/2026.eacl-long.9/). [290](#page-3-0)

Tengfei Yu, Siyu Pan, Caoyun Fan, Siyang Luo, Yaohui Jin, and Binglei Zhao. Can large language models exhibit cognitive and affective empathy as humans? *Computers in Human Behavior: Artificial Humans*, page 100233, 2025. doi:

10.1016/j.chbah.2025.100233. [293](#page-6-0)

{29}------------------------------------------------

