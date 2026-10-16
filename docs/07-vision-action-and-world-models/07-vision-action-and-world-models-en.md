<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 151-180; Marker split PDF pages 1-30. -->



{0}------------------------------------------------

# <span id="page-0-0"></span>Chapter 7

# Vision, Action, and World Models

#### What You Will Learn

Agents act in a world that involves more than language. This chapter explains how transformer-based models can represent information in modalities beyond text. You will learn to:

- Describe the architecture of *vision-language models* (VLMs), which typically combine a language component and a vision component. VLMs can, for example, generate realistic images and video clips.
- Explain how *vision-language-action* (VLA) models work and how training on robotic actions can support more general robot behavior.
- Explain how *world models* can help agents account for physical features of their surroundings, such as walls that cannot be walked through.

# 7.1 Multimodal Vision-Language Models

Many decisions involve vision. As this section shows, transformer architectures can also learn patterns in visual tokens. Models pretrained on both language and visual data are called vision-language models, or VLMs. They typically include a language component, a vision component, and a component that combines their representations.

We will discuss several VLM architectures, describing the tasks they address, how they are structured, and how they are trained.

Computer vision has a long history of success in AI. Deep learning achieved strong results in image recognition well before the breakthroughs in natural language processing [Krizhevsky et al. (2012)](#page-26-0). Can these two successful fields be combined?

{1}------------------------------------------------

<span id="page-1-0"></span>![](assets/_page_1_Picture_2.jpeg)

Figure 7.1: A visual question-answering task might ask: *What is the mustache made of?*

Combining them is not straightforward. Image processing has traditionally used convolutional neural networks, which work well with two-dimensional pixel grids; language processing has used transformers, which work well with one-dimensional token sequences. Researchers have developed architectures that bring the two fields together, and most frontier models are now VLMs.

### 7.1.1 Tasks

VLMs are evaluated on several kinds of tasks. In visual question answering (VQA), a model answers questions about an image. In image captioning, it generates a textual description. Models can also generate images from prompts.

For Figure 7.1, a VQA system might be asked what the mustache is made of and answer “bananas.” Other questions might ask the model to count objects of a particular shape and color in a region of an image—for example, whether there is a red block in the upper corner of a table. Benchmarks cover a wide range of visual questions, including MOSAIC [Burda-Lassen et al. (2025)](#page-24-0), GLOBAL-RG [Bhatia et al. (2024)](#page-24-0), CultureVQA [Nayak et al. (2024)](#page-27-0), FoodieQA [Li et al. (2024)](#page-27-0), and MaRVL [Liu et al. (2021)](#page-27-0).

In image captioning, a model could describe the scene in Figure [7.2](#page-2-0) as “a happy dog standing in the ocean.” Other captions might also be valid, such as “a wet dog,” “a cold dog,” or “a dog with a red collar standing in the ocean.” Popular benchmarks include MS-COCO [Lin et al. (2014)](#page-27-0), nocaps,<sup>1</sup> MSR-VTT for video [Xu et al. (2016)](#page-29-0), and VATEX for conventional short video captions.

<sup>1</sup><https://nocaps.org>

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Picture_2.jpeg)

Figure 7.2: An example image-captioning task, with the caption: *A happy dog is standing in the ocean.*

MSR-VTT and VATEX have been used in video-caption evaluation [Wang et al. (2019)](#page-29-0). Newer benchmarks focus on detailed, multi-faceted descriptions, including DetailCaps [Dong et al. (2024)](#page-25-0), VDC [Chai et al. (2024)](#page-24-0), and CapArena [Cheng et al. (2025)](#page-25-0).

Typical VLM applications include summarization, image generation, search and retrieval, image segmentation, and object detection. The Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark (MMMU) is a comprehensive benchmark for evaluating vision-language models on expert-level tasks [Yue et al. (2024)](#page-29-0).

### 7.1.2 Architecture

How can we design a model to perform these tasks?

Language represented as a token sequence is a one-dimensional problem, while vision represented as a pixel grid is two-dimensional. The corresponding deep-learning architectures have therefore differed: transformers model sequential patterns in language, while convolutional neural networks model spatial patterns in images. It was not initially clear what architecture a multimodal VLM should use—or whether such a model would work at all.

Early research asked whether transformers could also handle images, alongside conventional convolutional neural networks (CNNs). Specialized vision transformers and tokenization schemes proved effective for image processing [Dosovitskiy et al. (2020)](#page-25-0).

{3}------------------------------------------------

<span id="page-3-0"></span>![](assets/_page_3_Diagram_2.jpeg)

Figure 7.3: VLM architecture, adapted from LLaVA [Liu et al. (2023)](#page-27-0); image from Hugging Face.

A VLM also needs to combine image and language representations and generate text. A typical architecture connects an image encoder—either a CNN or a vision transformer—to a text-processing component. A projector aligns the visual representation with the language model’s embedding space, and a text decoder generates the response. Figure 7.3 illustrates this design using LLaVA as an example [Liu et al. (2023)](#page-27-0). The projector, often a multilayer perceptron (MLP), maps visual embeddings into a space the language model can use for joint reasoning.

More elaborate VLMs can contain several transformer modules. Figure [7.4](#page-4-0) shows the generative Stable Diffusion architecture [Esser et al. (2024)](#page-25-0).

#### 7.1.3 Learning Algorithms

VLM training must align and combine information from visual and language encoders. Once the representations are connected, the model can relate an image to text, interpret a query, and produce an answer. Researchers have developed several training approaches.

OpenAI’s CLIP—*Contrastive Language–Image Pretraining*—uses contrastive learning [Radford et al. (2021)](#page-28-0). This form of self-supervised learning creates a training signal without manually assigned labels. It brings related examples closer together in an embedding space and pushes unrelated examples apart. For example, it increases similarity for a matching image–text pair and decreases similarity for mismatched pairs.

{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Diagram_2.jpeg)

Figure 7.4: Stable Diffusion architecture [Esser et al. (2024)](#page-25-0).

Another approach uses masking. The VLM learns to predict portions of text or images that have been randomly obscured, much as language models learn from masked inputs. In image training, the model can learn to reconstruct hidden pixels using the visible image and its caption. FLAVA (Foundational Language and Vision Alignment) is one example; it uses a vision transformer for image encoding and transformers for language encoding and multimodal projection [Singh et al. (2022)](#page-28-0).

VLMs can also generate images from text prompts. Many text-to-image systems use diffusion models [Ho et al. (2022); Yang et al. (2023)](#page-29-0). Examples include Google’s Imagen [Baldridge et al. (2024)](#page-24-0), Midjourney, OpenAI’s DALL·E [Betker et al. (2023)](#page-24-0), and Stability AI’s Stable Diffusion [Esser et al. (2024)](#page-25-0). DALL·E 2 uses a diffusion model conditioned on CLIP image embeddings. Figure [7.5](#page-5-0) shows its well-known surreal image of avocado armchairs.

Training a VLM from scratch is expensive. Another option is to combine existing pretrained models: an LLM, a vision encoder, and a mapping network that projects visual embeddings into the language model’s input space. LLaVA (Large Language and Vision Assistant), for example, combines a Vicuna language decoder with a CLIP ViT vision encoder and uses a linear projector to align their outputs [Liu et al. (2023)](#page-27-0).

Many VLMs have been developed, and they often perform well on benchmarks and support decision-making in agents. Some current open and commercial models are multimodal VLMs, including DeepSeek-VL2, Gemini 2.0 Flash, GPT-4o, Llama 3.2, NVLM, and Qwen 2.5-VL.

{5}------------------------------------------------

<span id="page-5-0"></span>![](assets/_page_5_Picture_2.jpeg)

Figure 7.5: DALL·E: avocado armchairs.

![](assets/_page_5_Picture_4.jpeg)

Figure 7.6: Examples from the FLAVA image–text pretraining dataset [Singh et al. (2022)](#page-28-0).

### 7.1.4 Results

How well do VLMs perform at captioning and image generation? Research has reported results across visual question answering, object detection, image captioning, and text-to-image generation. Figure 7.6 shows examples from the dataset used by FLAVA. DALL·E 3, the successor to DALL·E 2, was trained to produce images that better reflect detailed prompts [Betker et al. (2023)](#page-24-0). Figure [7.7](#page-6-0) shows an example of prompt following with DALL·E 3, and Figure [7.8](#page-6-0) shows an example from Stable Diffusion based on a rectified-flow diffusion model.


<span id="page-6-0"></span>![](assets/_page_6_Picture_2.jpeg)

Figure 7.7: DALL·E 3 examples of following text-to-image prompts [Betker et al. (2023)](#page-24-0).

![](assets/_page_6_Picture_4.jpeg)

Figure 7.8: Stable Diffusion examples [Esser et al. (2024)](#page-25-0).

{7}------------------------------------------------

<span id="page-7-0"></span>![](assets/_page_7_Picture_2.jpeg)

Figure 7.9: “VLA” and “Vlas” mean different things in different languages; one of the images hints at which language is shown.

# 7.2 Decisions and Vision-Language-Action Models

Agents must weigh options before acting. One way to model this process is to train transformers on sequences of states, actions, and rewards, $(s,a,r)$ (see Section 20.1.2). Vision-language-action models (VLAs) are trained on sequences that combine language, visual observations, and robot actions. They produce robot API commands—a specialized language for controlling movement.

A VLA interprets a visual scene and a text instruction, then selects an action for a robot arm based on what its camera observes. Such models have shown that robots can generalize to unfamiliar situations and still complete tasks.

### 7.2.1 Tasks

A VLA should follow instructions in the real world and adapt as its environment changes. For example, if a manipulation model has learned to pick up red pencils, it should also be able to pick up blue pencils without requiring a new training set and training run from scratch. It should recognize objects when their orientation or position changes, or when lighting conditions differ, and know how to use a robot arm to perform the requested manipulation. Figure [7.10](#page-8-0) shows an example from CLIPort [Shridhar et al. (2022)](#page-28-0).

The hardest problems combine scenes, objects, and actions in ways that differ from the combinations seen during training. VLAs must generalize to these unseen combinations.

{8}------------------------------------------------

<span id="page-8-0"></span>![](assets/_page_8_Picture_2.jpeg)

Figure 7.10: CLIPort example: *Place the gray E-shaped block in the left-hand E-shaped slot* [Shridhar et al. (2022)](#page-28-0).

![](assets/_page_8_Picture_4.jpeg)

Figure 7.11: Another CLIPort example: *Sweep the beans into the blue zone* [Shridhar et al. (2020)](#page-28-0).

{9}------------------------------------------------

<span id="page-9-0"></span>Long-horizon navigation through complex, changing environments—even outdoors—remains especially challenging [Shridhar et al. (2022)](#page-28-0).

Collecting more demonstrations is often impractical in real-world human–robot interaction. To perform high-level tasks such as the one in Figure [7.11](#page-8-0), a VLA must recognize objects, their poses, and individual object regions. It must also understand language and connect linguistic meaning to visual objects and robot actions.

Language offers an intuitive way to specify goals and transfer semantic concepts across tasks. Research on VLMs has also shown how different kinds of embeddings can be connected semantically.

### 7.2.2 Architecture

VLA architectures can be complex. Transformers model sequential patterns well, which suggests they can also represent sequential decision problems of the kind often studied in reinforcement learning for robotics (see Sections 2.2.3 and 20.2.1). In these problems, an action changes the current state, and the resulting state determines what actions are available next.

Early work on Decision Transformer (DT) showed that GPT-style models can learn policies for offline reinforcement-learning tasks [Chen et al. (2021); Janner et al. (2021)](#page-26-0). Decision-pretrained transformers (DPTs) extend this idea by pretraining or fine-tuning transformer architectures on decision traces [Lee et al. (2023); Wang et al. (2024a)](#page-29-0). Section 20.1.2 discusses decision traces in more detail.

Following CLIPort [Shridhar et al. (2020, 2022)](#page-28-0), many VLAs are built on VLMs and represent robot commands as language tokens. Two examples illustrate different ways to combine language, vision, and navigation: Mobility VLA combines a long-context VLM with topological maps [Chiang et al. (2024)](#page-25-0), while LM-Nav composes several pretrained models to plan from observations and free-form instructions [Shah et al. (2023)](#page-28-0).

{10}------------------------------------------------

<span id="page-10-0"></span>![](assets/_page_10_Diagram_2.jpeg)

Figure 7.12: Mobility VLA architecture, combining a long-context VLM and topological maps [Chiang et al. (2024)](#page-25-0).

![](assets/_page_10_Diagram_4.jpeg)

Figure 7.13: LM-Nav’s three-model architecture [Shah et al. (2023)](#page-28-0).

Mobility VLA addresses *multimodal instruction navigation with demonstration tours* (MINT). In this setting, a previously recorded demonstration video provides prior information about the environment. The model has two levels: a long-context VLM identifies a goal frame from the video, and a lower-level navigation policy traverses a topological graph to generate actions at each time step.

LM-Nav uses three existing models: an LLM extracts landmarks, a VLM grounds them in visual observations, and a visual navigation model (VNM) executes the plan. Together, the models let LM-Nav follow text instructions using observations from the target environment, without fine-tuning.

Google’s RT-2 takes another approach [Brohan et al. (2023)](#page-24-0). It fine-tunes a VLM on both robot-trajectory data and large-scale vision-language tasks, such as VQA, so that knowledge learned from the web can inform robot control. Robot actions are represented as text tokens and included in training alongside natural language (Figure [7.14](#page-11-0)). Magma follows a related approach [Yang et al. (2025)](#page-29-0), pretraining on heterogeneous image, video, and robotics data. It uses Set-of-Mark tokens to identify actionable objects and Trace-of-Mark tokens to label movements, such as the trajectories of human hands or robot arms, supporting action grounding and planning (Figure [7.15](#page-11-0)).

Another model, $\pi^0$, uses a pretrained late-fusion VLM and a diverse dataset of robot-manipulation tasks [Black et al. (2024)](#page-24-0). Its design is inspired by Transfusion [Zhou et al. (2024)](#page-29-0).

{11}------------------------------------------------

<span id="page-11-0"></span>![](assets/_page_11_Diagram_2.jpeg)

Figure 7.14: RT-2 architecture: a VLM trained with robot actions represented as text tokens [Brohan et al. (2023)](#page-24-0).

![](assets/_page_11_Picture_3.jpeg)

Figure 7.15: Magma pretraining pipeline [Yang et al. (2025)](#page-29-0).


<span id="page-12-0"></span>![](assets/_page_12_Diagram_2.jpeg)

Figure 7.16: Architecture of $\pi^0$, a VLM-based model trained with cross-embodiment robot data [Black et al. (2024)](#page-24-0).

The dataset covers a range of manipulation tasks. Inspired by Transfusion [Zhou et al. (2024)](#page-29-0), $\pi^0$ adds a separate action expert that generates robot actions through flow matching, supporting smooth manipulation. The model can perform tasks directly from prompts or be fine-tuned on higher-quality data for complex, multi-stage tasks such as folding laundry or assembling a box (Figure 7.16).

#### 7.2.3 Learning Algorithms

Most VLA architectures build on VLMs and may add navigation maps. Robot commands are often represented as ordinary text tokens. VLA pretraining and fine-tuning commonly use self-supervised methods similar to those used for LLMs, VLMs, and decision transformers. Some systems reuse pretrained models to reduce training costs.

#### 7.2.4 Results

Several VLAs have shown zero-shot generalization to situations not seen during training. Mobility VLA, RT-2, and Magma are examples of models that perform navigation and object manipulation; $\pi^0$ has also been used for tasks such as folding laundry [Chiang et al. (2024); Brohan et al. (2023); Yang et al. (2025); Black et al. (2024)](#page-29-0).

{13}------------------------------------------------

<span id="page-13-0"></span>The results are best appreciated in video demonstrations. Videos are available for $\pi^0$<sup>2</sup> and RT-2.<sup>3</sup>

# 7.3 World Models, Physics, and Planning

An image presented to a VLM is a pattern of pixels; it does not inherently tell the model what is physically possible. Agents that interact with the world need knowledge of physical constraints so they do not, for example, attempt to walk through a wall.

World models can help agents act coherently in a physical environment. LLMs are trained primarily on language and acquire some knowledge about the world from text, but they are not specifically trained to build a broad, grounded understanding of physical reality. Some researchers argue that progress toward capable agents and general intelligence requires more research on world models [Sutton et al. (2022); Matsuo et al. (2022); LeCun et al. (2022); Klindt et al. (2026)](#page-28-0).

Robots and other agents need at least a basic model of physical processes to connect observations and actions to real outcomes. One approach is to learn a model of the environment from feedback on the agent’s actions.

The term *world model* has several uses in AI. In model-based reinforcement learning, it refers to a learned model of the environment [Ha and Schmidhuber (2018); Moerland et al. (2023); Sutton (1991)](#page-28-0). The model uses interaction data to build a local representation of the environment and improve the agent’s policy. It is often a compact, learned representation [Hafner et al. (2019); Plaat et al. (2020)](#page-27-0), also called an abstract or *latent* representation. World models may encode physical laws or spatial properties that constrain which actions an agent can take. What a model represents depends on its domain.

A language model learns statistical patterns that support tasks such as question answering and translation. A world model instead represents causal relationships that can be used to predict or simulate future states. Building agents that connect language models to such grounded models requires additional methods.

This section reviews how world models are used in agentic LLM systems.

<sup>2</sup><https://www.physicalintelligence.company/blog/pi0>

<sup>3</sup><https://robotics-transformer2.github.io>

{14}------------------------------------------------

#### <span id="page-14-0"></span>7.3.1 Tasks

A world model should encode enough knowledge for an agent to take meaningful actions grounded in its environment. A robot controller should account for the limits of an arm’s joints. A navigation agent should represent gravity, distance, and walls as solid obstacles. A financial agent should understand the consequences of a negative account balance. People often take such facts for granted, so an agent’s failure to represent them can be surprising. A VLM trained only on pixels may not recognize whether a texture represents clothing, smoke, or a wall.

Standard pretrained LLMs already contain some useful world knowledge. Fine-tuning can also encourage desired behavior: RLHF uses human feedback, while RLVR uses automatically verifiable outcomes for tasks such as mathematics and coding. Some early approaches use pretrained LLMs as world models without further fine-tuning. Agents can also learn from interaction, using reward signals to reinforce physically appropriate behavior.

In model-based reinforcement learning, a world model can improve sample efficiency by letting an agent plan in a learned model instead of repeatedly sampling from the real environment. This is especially useful when real-world interaction is more costly than simulation. The LLM world-model approaches discussed here draw on model-based reinforcement learning (Section [7.3.5](#page-20-0)). They address different tasks; the chapter focuses on three examples: WorldGPT, WebDreamer, and WorldCoder.

**WorldGPT.** WorldGPT aims to solve visual question-answering and embodied-planning tasks, including audio-video question answering [Ge et al. (2024)](#page-25-0). It learns a world-aware VLM from videos and uses a latent-space representation with external knowledge bases and long-term memory.

**WebDreamer.** WebDreamer aims to make learning web-interaction tasks, such as e-commerce, more sample-efficient [Gu et al. (2024)](#page-25-0). It lets an agent simulate actions before executing them on a live website, reducing the need to gather every experience from the real web. The method adapts model-based reinforcement learning to web environments and draws on Dreamer, a family of latent-dynamics models [Hafner et al. (2019, 2020a, 2023)](#page-26-0). Typical tasks include finding products on e-commerce sites. A smaller model distilled from a larger one supports efficient simulation.

**WorldCoder.** WorldCoder also aims to improve sample efficiency, but it generates an interpretable dynamics model—the transition and reward functions—as executable Python code [Tang et al. (2024)](#page-28-0). The code can then be used by a standard model-based reinforcement-learning system.

{15}------------------------------------------------

<span id="page-15-0"></span>![](assets/_page_15_Picture_2.jpeg)

Figure 7.17: Sokoban task used with WorldCoder [Tang et al. (2024)](#page-28-0).

![](assets/_page_15_Picture_4.jpeg)

Figure 7.18: ALFWorld task used with WorldCoder [Tang et al. (2024)](#page-28-0).

{16}------------------------------------------------

<span id="page-16-0"></span>![](assets/_page_16_Diagram_2.jpeg)

Figure 7.19: WorldGPT architecture [Ge et al. (2024)](#page-25-0).

The generated Python code is more efficient to execute than repeated LLM inference. WorldCoder has been evaluated on grid-based tasks such as Sokoban (Figure [7.17](#page-15-0)) and household planning tasks in ALFWorld (Figure [7.18](#page-15-0)).

#### 7.3.2 Architecture

A world model adds meaning to observations by representing how actions change the environment—something LLMs and VLMs may not capture reliably on their own. For example, it should distinguish walkable floor from a wall that blocks movement. A reward function indicates which outcomes are desirable. World-model systems therefore often combine an environment model with a policy learned from reward feedback. One useful perspective is to view a world model as a VLM combined with a policy learner.

WorldGPT, WebDreamer, and WorldCoder use different architectures.

**WorldGPT.** WorldGPT is built on the Vicuna-7B-v0 LLM.<sup>4</sup> It performs reasoning in a latent space and aims to model world dynamics, including transition and reward functions. Its architecture combines an LLM with multimodal encoders and a decoder to synthesize multimodal instruction examples. A cognitive component supports specialized or long-horizon tasks through memory, knowledge retrieval, and context reflection. The context-reflection module uses a Q-former [Li et al. (2023)](#page-26-0); Figure 7.19 shows the architecture.

<sup>4</sup><https://huggingface.co/lmsys/vicuna-7b-v1.3>, <https://lmsys.org>

{17}------------------------------------------------

<span id="page-17-0"></span>Figure 7.20: WebDreamer pseudocode, using an LLM for the simulation and scoring functions [Gu et al. (2024)](#page-25-0).

![](assets/_page_17_Diagram_4.jpeg)

Figure 7.21: WorldCoder architecture [Tang et al. (2024)](#page-28-0).

**WebDreamer.** WebDreamer targets web interaction. It distills a large Qwen-70B model [Bai et al. (2025)](#page-24-0) into a smaller 7B model, which simulates actions in a model of the web before they are tried on the live site. It uses model predictive control (MPC) to choose website actions and replan after each interaction [Garcia et al. (1989); Schwenzer et al. (2021)](#page-28-0). The LLM serves as both the world model and value function: it estimates the result of an action (the *sim* function) and scores candidate outcomes (the *score* function). It can also generate candidate actions [Koh et al. (2024)](#page-26-0). The authors use two stages: sample top-$k$ actions, then use the LLM to refine the candidates before simulation.

**WorldCoder.** WorldCoder is a model-based agent that represents its knowledge of the environment as an executable Python program (Figure 7.21). It uses program synthesis to generate a state representation and the task’s action, transition, and reward functions. In reinforcement learning, these components define a Markov decision process (MDP): state representation $S$, action function $A$, transition function $T$, and reward function $R$ (Section [7.3.5](#page-20-0)). The generated code is checked and improved through iterative self-debugging [Chen et al. (2023)](#page-25-0), as described in Section 6.1.3.2.

{18}------------------------------------------------

<span id="page-18-0"></span>This process repeats until the program produces the required output. WorldCoder’s learned model is symbolic: its transition function is executable code, whereas a conventional deep reinforcement-learning model is typically a neural network. At inference time, WorldCoder can be efficient because it runs the Python transition and reward functions without repeatedly calling a large LLM.

In many model-based LLM approaches, the LLM simulates the world at inference time. WorldCoder instead uses the LLM to build a simulator that can be executed later. A Python interpreter applies the transition function; in ReAct, by contrast, the LLM reasons over the world state directly. The symbolic approach can be better suited to planning tasks because LLMs may struggle to plan reliably at inference time (Section 6.2.3).

#### 7.3.3 Learning Algorithms

This section reviews the pretraining and fine-tuning methods used by the three approaches.

**WorldGPT.** WorldGPT learns world dynamics by pretraining and fine-tuning on millions of videos from different domains, using Vicuna-7B-v0 as its base model. A language-centric encoder creates a unified representation of multimodal states, and special vocabulary tokens support multimodal decoding. Training progresses from unimodal to cross-modal data. Fine-tuning uses a cognitive architecture and a dataset of more than 10 million videos annotated with fine-grained state transitions. The cognitive components—memory, knowledge retrieval, and context reflection—support specialized knowledge acquired from synthetic multimodal instruction data.

**WebDreamer.** WebDreamer aims to learn dynamics for model-based reinforcement learning. It initializes Dreamer-7B from the pretrained Qwen2-VL model [Wang et al. (2024b)](#page-29-0), then fine-tunes it on 3.1 million interaction examples. Its data pipeline includes (1) random walks in which the model interacts with web pages by searching and clicking, and (2) synthesis, in which Qwen2-VL-72B describes state changes from visual snapshots taken before and after an action. For each action, WebDreamer simulates possible future states over a finite horizon and replans using MPC.

**WorldCoder.** WorldCoder uses program synthesis to learn a transition function that predicts the environment’s next state. It begins with random exploration to collect interaction data, synthesizes and updates its world model, then plans using the generated code.

{19}------------------------------------------------

<span id="page-19-0"></span>Different planners can be used, from depth-limited value iteration in simple domains to Monte Carlo tree search (MCTS) in more complex ones [Browne et al. (2012)](#page-24-0).

### 7.3.4 Results

We now compare the results reported for the three approaches.

**WorldGPT.** WorldGPT is evaluated on a multimodal state-transition prediction benchmark. The authors report that it predicts the dynamics of complex real-world kitchen and cooking scenarios. In one experiment, it was 30 times faster than traditional diffusion-based methods on audio-video question-answering tasks [Ge et al. (2024)](#page-25-0).

**WebDreamer.** WebDreamer is evaluated on web-interaction tasks. The authors report performance above GPT-4o on some benchmarks and improvements over reactive baselines. It performs well in the sandbox environment VisualWebArena and on real-world websites in Online-Mind2Web and Mind2Web-Live. On the online benchmarks, the Dreamer-7B approach is reported to be four to five times more sample-efficient than tree search. On live websites, WebDreamer outperforms the reactive baseline by 42% and 24%, respectively.

**WorldCoder.** Results are reported on grid worlds and household planning tasks such as ALFWorld. WorldCoder is more sample-efficient than model-free reinforcement learning and more compute-efficient than ReAct-style agents that use an LLM to plan in context.

Representing world knowledge as code has several advantages. Prior knowledge can be embedded in a program, and solutions to published problems may already be familiar from pretraining. Code can also be transferred across tasks by reusing program fragments. It is human-readable, so people can inspect the model’s representation. However, WorldCoder inherits the computational difficulty of planning: problems that are hard for dedicated solvers may also be hard for WorldCoder. An LLM-generated transition function may also be less reliable than one designed by an expert. Synthesized code currently aims to reproduce transition and reward functions; future work on self-reflection and metareasoning may improve results.

{20}------------------------------------------------

<span id="page-20-0"></span>![](assets/_page_20_Diagram_2.jpeg)

Figure 7.22: Standard reinforcement-learning loop. The agent learns a parameterized policy $\pi_\theta$ from reward feedback from the environment [Sutton and Barto (2018)](#page-28-0).

#### 7.3.5 Model-Based Reinforcement Learning

World models are used in agentic LLM systems as well as in model-based reinforcement learning. This section briefly reviews the latter to clarify how its ideas are applied to LLM world models.

An influential early model-based approach is Dyna [Sutton (1991)](#page-28-0). More recent overviews include [Ha and Schmidhuber (2018)](#page-26-0), [Hafner et al. (2020b)](#page-26-0), [Moerland et al. (2023)](#page-27-0), and [Plaat et al. (2023)](#page-27-0).

In reinforcement learning, an agent learns how to act by interacting with an environment and receiving rewards.

#### 7.3.5.1 Model-Free Reinforcement Learning

In model-free reinforcement learning, the agent learns a policy directly from interaction with the environment; it does not build an explicit model of the environment (Figure 7.22). A Markov decision process (MDP) is often described by four components: a state space $S$, an action function $A$, a transition function $T(s,a)\rightarrow s'$, and a reward function $R(s)\rightarrow r$. Taking action $a$ in state $s$ leads to a new state $s'$ and a reward $r$.

The goal is to learn an optimal policy $\pi^*(s)\rightarrow a^*$ that selects the best action for each state and maximizes expected cumulative reward. In model-free learning, the agent does not have direct access to the environment’s transition and reward functions; it learns from sampled outcomes. This direct approach can require many samples: policy improvement depends on interaction, and large MDPs may need substantial data before learning converges.

{21}------------------------------------------------

<span id="page-21-0"></span>![](assets/_page_21_Diagram_2.jpeg)

Figure 7.23: In model-based reinforcement learning, environment feedback improves both the policy $\pi$ and the agent’s local dynamics model $\hat{M}=(\hat{T},\hat{R})$, which can be used to plan.

#### 7.3.5.2 Model-Based Reinforcement Learning

Direct reinforcement learning does not retain sampled experience; each policy update requires new interaction. But interacting with the real world—for example, a robot whose joints wear out—can be expensive. A straightforward improvement is to store past samples and reuse them. A more advanced approach also learns a local model $\hat{M}=(\hat{T},\hat{R})$ of the environment dynamics. Once learned, this model can generate additional simulated samples at lower cost (Figure 7.23).

These are indirect methods: the agent learns a local dynamics model from the environment, then uses that model to improve its policy. The model’s estimates $\hat{T}$ and $\hat{R}$ approximate the true transition and reward functions $T$ and $R$, which are only directly known to the environment. Model-based reinforcement learning has been successful in tasks such as Go [Silver et al. (2016)], Diplomacy [Bakhtin et al. (2022)], and StarCraft [Vinyals et al. (2019)], and in learning across domains [Hafner et al. (2020b)](#page-26-0).

Model-based reinforcement learning also connects to planning, a classical area of symbolic AI. A learned latent representation lets an agent simulate possible futures efficiently without acting in the real world.

#### 7.3.5.3 Dimension Reduction

In deep model-based reinforcement learning, the learned local dynamics model is often called a world model [Ha and Schmidhuber (2018)](#page-26-0). It can compress high-dimensional observations into lower-dimensional state and reward representations, which can then support more efficient policy learning.

{22}------------------------------------------------

<span id="page-22-0"></span>Many environments contain details that matter little to an agent’s decisions—for example, trees in a video-game background that never move and do not affect the value of its actions. A world model can abstract away such details and reduce the effective dimensionality of the state space.

#### 7.3.5.4 Joint Embedding Predictive Architecture

The joint embedding predictive architecture (JEPA), proposed by LeCun et al. (2022) and further developed by Assran et al. (2023), predicts future states in a compressed latent representation rather than reconstructing raw sensory details or pixels. Like dimension-reduction methods in model-based reinforcement learning, JEPA learns a compact representation of the world.

Model-based reinforcement learning learns a local transition model supervised by rewards. JEPA instead learns from self-supervised observations, such as videos or images, by mapping context embeddings to target embeddings. It does not explicitly require actions, rewards, or policy optimization. The two approaches can be connected: JEPA-style representations can serve as perception or world-model components in reinforcement-learning or planning systems [Lim et al. (2026)](#page-26-0), for example to support robot control [Destrade et al. (2025)](#page-25-0).

#### 7.3.5.5 Agentic World Models

In model-based reinforcement learning, an LLM can be used in two ways. A pretrained, static LLM can act as a world model $\hat{T},\hat{R}$ within a model-based reinforcement-learning system, such as LLM4WM [Liu et al. (2024)](#page-27-0). The LLM assists planning, for example through tree search or Monte Carlo tree search (MCTS).

Alternatively, model-based reinforcement learning can train an LLM world model (MBRL4LLM) using feedback from interaction with the world. The LLM then becomes adaptive through an external harness or by post-training the model itself. This research area explores how agents can build more realistic models of their environments and act more effectively within them [Zuo et al. (2026)](#page-29-0).<sup>5</sup>

## 7.4 Concluding Remarks

Transformers can be trained on tokenized language and on data from other modalities. This chapter has reviewed several architectures for combining modalities, although they differ substantially. Most are transformer-based, with tokenization schemes adapted to the data they process. The following overview highlights their shared features.

{23}------------------------------------------------

<span id="page-23-0"></span>

# Bibliography

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from

{24}------------------------------------------------

<span id="page-24-0"></span>images with a joint-embedding predictive architecture. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 15619–15629. IEEE, 2023. [173](#page-22-0) Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [168](#page-17-0) Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, et al. Human-level play in the game of diplomacy by combining language models with strategic reasoning. *Science*, 378(6624):1067–1074, 2022. [172](#page-21-0) Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Lluis Castrejon, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, et al. Imagen 3. *arXiv preprint arXiv:2408.07009*, 2024. [155](#page-4-0) James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. *Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf*, 2(3):8, 2023. [155,](#page-4-0) [156,](#page-5-0) [157](#page-6-0) Mehar Bhatia, Sahithya Ravi, Aditya Chinchure, Eunjeong Hwang, and Vered Shwartz. From local concepts to universals: Evaluating the multicultural understanding of vision-language models. *arXiv preprint arXiv:2407.00263*, 2024. [152](#page-1-0) Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. $\pi^0$: A vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024. [161,](#page-10-0) [163](#page-12-0) Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. *arXiv preprint arXiv:2307.15818*, 2023. [161,](#page-10-0) [162,](#page-11-0) [163](#page-12-0) Cameron B Browne, Edward Powley, Daniel Whitehouse, Simon M Lucas, Peter I Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A survey of monte carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in games*, 4(1):1–43, 2012. [170](#page-19-0) Olena Burda-Lassen, Aman Chadha, Shashank Goswami, and Vinija Jain. How culturally aware are vision-language models? In *2025 IEEE 6th International Conference on Image Processing, Applications and Systems (IPAS)*, pages 1–6. IEEE, 2025. [152](#page-1-0) Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. *arXiv preprint arXiv:2410.03051*, 2024. [153](#page-2-0)

{25}------------------------------------------------

<span id="page-25-0"></span>Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. *Advances in neural information processing systems*, 34:15084–15097, 2021. [160](#page-9-0) Xinyun Chen, Maxwell Lin, Nathanael Scharli, and Denny Zhou. Teaching large lan- ¨ guage models to self-debug. *arXiv preprint arXiv:2304.05128*, 2023. [168](#page-17-0) Kanzhi Cheng, Wenpo Song, Jiaxin Fan, Zheng Ma, Qiushi Sun, Fangzhi Xu, Chenyang Yan, Nuo Chen, Jianbing Zhang, and Jiajun Chen. Caparena: Benchmarking and analyzing detailed image captioning in the llm era. *arXiv preprint arXiv:2503.12329*, 2025. [153](#page-2-0) Hao-Tien Lewis Chiang, Zhuo Xu, Zipeng Fu, Mithun George Jacob, Tingnan Zhang, Tsang-Wei Edward Lee, Wenhao Yu, Connor Schenck, David Rendleman, Dhruv Shah, et al. Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs. *arXiv preprint arXiv:2407.07775*, 2024. [160,](#page-9-0) [161,](#page-10-0) [163](#page-12-0) Matthieu Destrade, Oumayma Bounou, Quentin Le Lidec, Jean Ponce, and Yann LeCun. Value-guided action planning with jepa world models. *arXiv preprint arXiv:2601.00844*, 2025. [173](#page-22-0) Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption. *arXiv preprint arXiv:2405.19092*, 2024. [153](#page-2-0) Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. [154](#page-3-0) Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry ¨ Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024. [154,](#page-3-0) [155,](#page-4-0) [157](#page-6-0) Carlos E Garcia, David M Prett, and Manfred Morari. Model predictive control: Theory and practice—a survey. *Automatica*, 25(3):335–348, 1989. [168](#page-17-0) Zhiqi Ge, Hongzhe Huang, Mingze Zhou, Juncheng Li, Guoming Wang, Siliang Tang, and Yueting Zhuang. Worldgpt: Empowering llm as multimodal world model. In *Proceedings of the 32nd ACM International Conference on Multimedia*, pages 7346– 7355, 2024. [165,](#page-14-0) [167,](#page-16-0) [170](#page-19-0) Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, et al. Is your llm secretly a world model of the internet? model-based planning for web agents. *arXiv preprint arXiv:2411.06559*, 2024. [165,](#page-14-0) [168](#page-17-0)

{26}------------------------------------------------

<span id="page-26-0"></span>David Ha and Jurgen Schmidhuber. World models. ¨ *arXiv preprint arXiv:1803.10122*, 2018. [164,](#page-13-0) [171,](#page-20-0) [172](#page-21-0) Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In *International Conference on Machine Learning*, pages 2555–2565, 2019. [164,](#page-13-0) [165](#page-14-0) Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In *International Conference on Learning Representations*, 2020a. [165](#page-14-0) Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. *arXiv preprint arXiv:2010.02193*, 2020b. [171,](#page-20-0) [172](#page-21-0) Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*, 2023. [165](#page-14-0) Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. *arXiv preprint arXiv:2212.10071*, 2022. [155](#page-4-0) Michael Janner, Qiyang Li, and Sergey Levine. Offline reinforcement learning as one big sequence modeling problem. *Advances in neural information processing systems*, 34:1273–1286, 2021. [160](#page-9-0) David Klindt, Yann LeCun, and Randall Balestriero. When does lejepa learn a world model? *arXiv preprint arXiv:2605.26379*, 2026. [164](#page-13-0) Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. *arXiv preprint arXiv:2407.01476*, 2024. [168](#page-17-0) Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012. [151](#page-0-0) Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. *Open Review*, 62(1):1–62, 2022. [164,](#page-13-0) [173](#page-22-0) Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, and Emma Brunskill. Supervised pretraining can learn in-context reinforcement learning. *Advances in Neural Information Processing Systems*, 36:43057–43083, 2023. [160](#page-9-0) Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference on machine learning*, pages 19730–19742. PMLR, 2023. [167](#page-16-0)

{27}------------------------------------------------

<span id="page-27-0"></span>Wenyan Li, Crystina Zhang, Jiaang Li, Qiwei Peng, Raphael Tang, Li Zhou, Weijia Zhang, Guimin Hu, Yifei Yuan, Anders Søgaard, Daniel Hershcovich, and Desmond Elliott. FoodieQA: A multimodal dataset for fine-grained understanding of Chinese food culture. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 19077–19095, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.1063. URL <https://aclanthology.org/2024.emnlp-main.1063/>. [152](#page-1-0) Jing Yu Lim, Rushi Shah, Zarif Ikram, Samson Yu, Haozhe Ma, Tze-Yun Leong, and Dianbo Liu. Jedi: Joint embedding diffusion world model for online model-based reinforcement learning. *arXiv preprint arXiv:2605.13013*, 2026. [173](#page-22-0) Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects ´ in context. In *European conference on computer vision*, pages 740–755. Springer, 2014. [152](#page-1-0) Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. Visually grounded reasoning across languages and cultures. *arXiv preprint arXiv:2109.13238*, 2021. [152](#page-1-0) Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in neural information processing systems*, 36:34892–34916, 2023. [154,](#page-3-0) [155](#page-4-0) Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-language-action models for embodied ai. *arXiv preprint arXiv:2405.14093*, 2024. [163](#page-12-0) Yutaka Matsuo, Yann LeCun, Maneesh Sahani, Doina Precup, David Silver, Masashi Sugiyama, Eiji Uchibe, and Jun Morimoto. Deep learning, reinforcement learning, and world models. *Neural Networks*, 152:267–275, 2022. [164](#page-13-0) Thomas M Moerland, Joost Broekens, Aske Plaat, Catholijn M Jonker, et al. Modelbased reinforcement learning: A survey. *Foundations and Trends® in Machine Learning*, 16(1):1–118, 2023. [164,](#page-13-0) [171](#page-20-0) Shravan Nayak, Kanishk Jain, Rabiul Awal, Siva Reddy, Sjoerd Van Steenkiste, Lisa Anne Hendricks, Aishwarya Agrawal, et al. Benchmarking vision language models for cultural understanding. *arXiv preprint arXiv:2407.10920*, 2024. [152](#page-1-0) Aske Plaat, Walter Kosters, and Mike Preuss. Deep model-based reinforcement learning for high-dimensional problems, a survey. *arXiv preprint arXiv:2008.05598*, 2020. [164](#page-13-0) Aske Plaat, Walter Kosters, and Mike Preuss. High-accuracy model-based reinforcement learning, a survey. *Artificial Intelligence Review*, 56(9):9541–9573, 2023. [171](#page-20-0) 

{28}------------------------------------------------

<span id="page-28-0"></span>Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [154](#page-3-0) Max Schwenzer, Muzaffer Ay, Thomas Bergs, and Dirk Abel. Review on model predictive control: An engineering perspective. *The International Journal of Advanced Manufacturing Technology*, 117(5):1327–1349, 2021. [168](#page-17-0) Dhruv Shah, Błazej Osi ˙ nski, Sergey Levine, et al. Lm-nav: Robotic navigation with ´ large pre-trained models of language, vision, and action. In *Conference on robot learning*, pages 492–504. PMLR, 2023. [160,](#page-9-0) [161](#page-10-0) Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, ´ and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. *arXiv preprint arXiv:2010.03768*, 2020. [159,](#page-8-0) [160](#page-9-0) Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In *Conference on robot learning*, pages 894–906. PMLR, 2022. [158,](#page-7-0) [159,](#page-8-0) [160](#page-9-0) David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016. [172](#page-21-0) Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 15638–15650, 2022. [155,](#page-4-0) [156](#page-5-0) Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. *ACM Sigart Bulletin*, 2(4):160–163, 1991. [164,](#page-13-0) [171](#page-20-0) Richard S Sutton and Andrew G Barto. *Reinforcement Learning*. Adaptive Computation and Machine Learning series. Bradford Books, Cambridge, MA, 2 edition, November 2018. [171](#page-20-0) Richard S Sutton, Michael Bowling, and Patrick M Pilarski. The alberta plan for ai research. *arXiv preprint arXiv:2208.11173*, 2022. [164](#page-13-0) Hao Tang, Darren Key, and Kevin Ellis. Worldcoder, a model-based llm agent: Building world models by writing code and interacting with the environment. *Advances in Neural Information Processing Systems*, 37:70148–70212, 2024. [165,](#page-14-0) [166,](#page-15-0) [168](#page-17-0) Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michael Mathieu, Andrew ¨ Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. *Nature*, 575(7782):350–354, 2019. [172](#page-21-0)

{29}------------------------------------------------

<span id="page-29-0"></span>Jiuqi Wang, Ethan Blaser, Hadi Daneshmand, and Shangtong Zhang. Transformers learn temporal difference methods for in-context reinforcement learning. *arXiv preprint arXiv:2405.13861*, 2024a. [160](#page-9-0) Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024b. [169](#page-18-0) Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-andlanguage research. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 4581–4591, 2019. [153](#page-2-0) Lilian Weng. What are diffusion models? *lilianweng. github. io*, page 21, 2021. [154](#page-3-0) Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 5288–5296, 2016. [152](#page-1-0) Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. *arXiv preprint arXiv:2502.13130*, 2025. [161,](#page-10-0) [162,](#page-11-0) [163](#page-12-0) Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. *ACM computing surveys*, 56(4):1–39, 2023. [154,](#page-3-0) [155](#page-4-0) Ruibin Yuan, Hanfeng Lin, Shuyue Guo, Ge Zhang, Jiahao Pan, Yongyi Zang, Haohe Liu, Yiming Liang, Wenye Ma, Xingjian Du, et al. Yue: Scaling open foundation models for long-form music generation. *arXiv preprint arXiv:2503.08638*, 2025. [174](#page-23-0) Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9556–9567, 2024. [153](#page-2-0) Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. *arXiv preprint arXiv:2408.11039*, 2024. [163](#page-12-0) Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang Su, Qingfeng Lan, et al. Qwen-agentworld: Language world models for general agents. *arXiv preprint arXiv:2606.24597*, 2026. [173](#page-22-0)
