<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 211-222; Marker split PDF pages 1-12. -->



{0}------------------------------------------------

## <span id="page-0-0"></span>Chapter 10

# Case Study 2: BALROG for LLM Reasoning and Reflection

#### What You Will Learn

In this case study you will gain hands-on experience building self-reflective agents for reasoning tasks. Specifically, you will be able to:

- Run a multi-step *reasoning agent* on a benchmark of puzzles and games.
- Adapt a reasoning agent with your own ideas.
- Implement your own *self-reflective* agent.
- Critically reflect on the *performance and stability* of a learning agent in the domain of puzzles and games.

## 10.1 BALROG Overview

This case study uses BALROG, a benchmark suite for reasoning tasks, to explore an LLM agent that performs self-reflection.<sup>1</sup> BALROG stands for *Benchmarking Agentic LLM and VLM Reasoning On Games*. The BALROG benchmark consists of a mix of easy and hard games, some of which are difficult for LLMs to play well, as shown in the BALROG paper [\[Paglieri et al.,](#page-10-0) [2025\]](#page-10-0).

Initially, the BALROG games are benchmarked with standard LLMs. The code, however, is quite open and amenable to writing your own agent approaches, to see if you can do better. The software is written for you to write your own agent code for in-context (reinforcement) learning.

The [BALROG website](https://balrogai.com) provides the code and a leaderboard. This chapter guides you through running the games and writing your own agent.

<sup>1</sup>A Balrog is a mythical demon from Middle-earth, featured in the fantasy works of J.R.R. Tolkien. A Balrog is a match even for Gandalf the Grey, the most powerful wizard of all.

{1}------------------------------------------------

<span id="page-1-0"></span>

| Skill or measure | BabyAI | TextWorld | Crafter | Baba Is AI | MiniHack | NLE |
|---|---:|---:|---:|---:|---:|---:|
| Navigation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Exploration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Resource management | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Complex credit assignment | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Inferring environment dynamics | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Long-term planning | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Turns to complete | 10¹ | 10² | 10³ | 10² | 10² | 10⁴–10⁵ |
| Time for a human to master | Seconds | Minutes | Hours | Hours | Hours | Years |

*Figure 10.1: Games and capabilities represented in BALROG.* [Paglieri et al., 2025](#page-10-0)

The games range from easy to difficult: BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack, and NetHack Learning Environment (NLE). Some are text-based and some include visual input, but all require reasoning. NetHack is particularly difficult; high scores are not expected without substantial expertise. Achieving one without using a guide might be publishable research.

BALROG challenges in-context agents with games that require reasoning, planning, and long sequences of actions. Figure 10.1 summarizes the skills and difficulty represented across its six environments.

#### 10.1.1 Puzzles and Games

BALROG consists of the six puzzles and games shown in Figure 10.1. The games run from easy grid world puzzles to hard text-based dungeon and dragons strategy games, with long credit assignment episodes and flat reward landscapes.

The BALROG authors provide a codebase for experimenting with agents and evaluating custom designs. To perform well on the benchmarks in BALROG, the agents must be good at reasoning and planning, and, most likely, be good at adapting and reflecting on their own behavior.

BabyAI is a simple grid-world game, where the agent has to solve a navigation task. A basic reinforcement-learning algorithm can solve this game. Tasks are given in natural language, such as: *go to the blue ball, then pick up the grey key*. Of course, the challenge for the agentic LLM is to perform the planning part of the task. It is expected that the LLM will be able to do well on this simple game (see the BALROG paper).

NetHack, on the other hand, is a highly challenging game, also for human players. NetHack is a text-based dungeons and dragons-like game. It includes both long-term strategic planning, in the sense that winning a game can involve hundreds of thousands of steps, and short-term tactical fights to fend off the usual hordes of monsters. Credit assignment is a challenge in NetHack, because actions taken earlier may contribute to eventual success or failure.

{2}------------------------------------------------

<span id="page-2-0"></span>![](assets/_page_2_Diagram_2.jpeg)

*Figure 10.2: Architecture of BALROG.* [Paglieri et al., 2025](#page-10-0)

Actions in one area may have unexpected consequences much later elsewhere, making the game difficult for reinforcement-learning agents. Human players may spend years mastering NetHack without relying on guides.

### 10.1.2 BALROG Architecture

We will now look at the architecture of the BALROG framework. Figure 10.2 shows the architecture described in the BALROG paper [Paglieri et al., 2025](#page-10-0).

Please familiarize yourself with the BALROG system. Read the paper [Paglieri et al.](#page-10-0) [\[2025\]](#page-10-0), which you can find on <https://arxiv.org/abs/2411.13543>, or go to the BALROG website at <https://balrogai.com>.

If you have not done so yet, install BALROG on your system with the git clone command. Go to <https://balrogai.com>, or directly to the Github repository at <https://github.com/balrog-ai/BALROG>. Install the repository on your local computer.

Decide which LLM you want to use. Please note that typical BALROG agents make many LLM calls, at each step of the way. If you want to run the LLM locally, install it, or use it from Huggingface. If you want to use a (commercial) cloud-based LLM, acquire an API-key and install the API-key in your environment. You may want to consider applying for a free student-tier, if that is offered. The use of frontier LLMs can be expensive, but usually older and smaller LLMs, some of which are still relatively capable, are much cheaper.

The BALROG repository README explains how to install the system and configure an API key. The following environment-variable example uses placeholders. For the first run, any supported provider will do; a later exercise compares multiple LLMs.

{3}------------------------------------------------

<span id="page-3-0"></span>

```bash
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
```

*Listing 10.1: Setting API keys in the environment.*
#### 10.1.3 Agent

Please read the [BALROG agent tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/agents.md). In BALROG, agents are small pieces of Python code that call an LLM with a prompt to provide an answer how to determine the next step. The agent code is responsible for receiving answers from the model (observations) and for selecting the next action. The LLM is used to execute various prompts.

BALROG follows interaction style D from Figure 9.7, where the prompt is built up from the history of interactions.

#### 10.1.3.1 Prompt Builder

The agent builds a prompt based on previous actions, based on observation/action pairs from previous interactions. Thus, in reinforcement learning terms, the agent performs the function of the policy, selecting the next action to perform based on the incoming observation.

Go to the prompt\_builder directory to change the way that prompts are constructed from the history of actions.

#### 10.1.3.2 Agents

Agents are responsible for the memory function: managing the observation/action histories. The agents must keep a record of past observations and actions to support context-aware decision-making, and this must be implemented by us, external to the LLM.

The agent code in BALROG queries the LLM: agents send observations to the LLM, receive responses, and use those responses to decide on actions to take in the game.

BALROG comes with a number of pre-written agents, which you should study to design your own. The repository has a naive agent and a chain of thought agent. Also provided are more robust versions for handling LLM responses.

When you build your own agent, you will most likely build on the custom agent. The BALROG agent tutorial provides an example for a simple planning agent.

{4}------------------------------------------------

```bash
python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

*Listing 10.2: Running the standard evaluation.*
## 10.2 Implementation Steps

We will now provide some exercises to help you explore building your agents.

#### 10.2.1 Reproduce Paper

We start by trying to reproduce the results in the BALROG paper. Please have a look at the BALROG code at Github <https://github.com/balrog-ai/BALROG>. Read the Evaluation Guide, the Agent Development tutorial, and the instructions how to run Few Shot Learning experiments (in-context learning). Install the relevant tools on your computer, and the right Python version (and put it in your PATH). Check the version by starting the Python interpreter from the command prompt: \$ python.

First, run the standard evaluation, to make sure you installed things correctly, see Listing 10.2.

#### 10.2.2 Different LLMs

Compare the standard agent across benchmarks using several models, including hosted and open-weight options such as models from OpenAI, Anthropic, DeepSeek, OLMo, and Qwen. Especially the open version from OpenAI, Claude, DeepSeek, OLMo and Qwen may work well.

Running these models is important, and will lead to more insight into the relative strengths of the various models.

#### 10.2.3 Different Puzzles

Next, to get a better understanding of the benchmark, you should, after running the LLM agent in the different puzzle problems, have a closer look at the prompts and the answers of LLM. See if you can see where the LLM agent fails, where the weakness of the agent lies. By close inspection of the individual actions, and a deeper understanding of the puzzle, you might be able to think of a way to improve your agent.

#### 10.2.4 Computational Cost

BALROG agents call an LLM for each step that the agent makes, often with a long prompt. In these puzzles and games many such steps are made. Running different 

{5}------------------------------------------------

LLMs on the full benchmark, for different seeds, will easily require many tokens.

In your experiments, first try a small and cheap LLM, such as Qwen-3B. Start with the easiest benchmarks (BabyAI and Crafter). If you wish to expand, please go ahead, but be mindful of the computational cost. You want insights, but you also want your experiments to finish in a reasonable time for a reasonable cost.

#### 10.2.5 Implement Own Reasoning Agent

Now it is time to implement your own learning agent. Go to the agents directory in balrog, and edit the custom.py file. Listing [10.3](#page-6-0) reproduces the code for Custom Agent from the BALROG documentation. The code can be run with the command in Listing [10.4](#page-7-0) (specifying the custom agent, and running with fewer workers). Of course, you can specify your own LLM of choice in the command.

Start with the provided code, run it, see how it works. Then, you can try to make small improvements. Try making changes to the prompt for the LLM, and to the way in which the Python code handles the learning in the agent, either with the history prompt builder, in the plan, or in the learning algorithm.

## 10.3 Evaluation

The performance of your agent will be reported by BALROG as a progression percentage. Record the progression numbers for each benchmark, and for the agent types that you try. Do this in a structured way, you may want to write a Python or shell script to automate the running of the experiments and the logging of the results, perhaps also plotting the progression in a graph. Consider using wandb.ai.

What agent types work well, and on what benchmark do they work well? Make a matrix in which you log progression for the agent types and the benchmark types that you try.

Note that LLM experiments are inherently non-deterministic: two runs with the same prompt may give different results. Therefore, perform multiple runs of each experiment. If you get five identical results in a row, you may start to trust these results. If they vary widely, you may want to investigate. Is the reason purely statistical? Does turning down the temperature solve the problem, or is there another problem?

Do not blindly trust the progression numbers that are reported. Since you are using code on a computer, there may be bugs, or other issues. The BALROG system facilitates delving deeper into the runs. It logs csv files that you can look at with a spreadsheet. Have a look at some of the runs, and decide if it makes sense what you see, or if the agents are exhibiting degenerative behavior. If they do, try to find the reason. Did you do something wrong? Is there unexpected interference between the benchmark and the agent behavior? Is the reward landscape too flat, so that there is no signal for the agent to go after, and all it does is random actions running in circles in a way that makes no sense at all? Is there something that you can do about this?

Please note that research on large language models, especially agentic LLMs, is still developing, and our insights and our software frameworks have not yet stabilized. Some unexpected results are to be expected, and some challenges or frustration as well.

{6}------------------------------------------------

<span id="page-6-0"></span>

```python
from balrog.agents.base import BaseAgent
import re


class CustomAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = (
            f"Current Plan:\n{self.plan}\n"
            if self.plan
            else "You have no plan yet.\n"
        )
        planning_instructions = """
Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan. Then, provide the next action to take.
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan." if the current plan is good>
ACTION: <your next action>
""".strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n\n" + planning_instructions

        response = self.client.generate(messages)
        plan, action = self._extract_plan_and_action(response.completion)
        if plan != "No changes to the plan.":
            self.plan = plan
        response = response._replace(reasoning=plan, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        plan_match = re.search(
            r"PLAN:\s*(.*?)(?=\nACTION:|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL,
        )
        action_match = re.search(
            r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL
        )
        plan = (
            plan_match.group(1).strip()
            if plan_match
            else "No changes to the plan."
        )
        action = action_match.group(1).strip() if action_match else None
        return plan, action
```

*Listing 10.3: Custom-agent code in BALROG.*
{7}------------------------------------------------

<span id="page-7-0"></span>

```bash
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=64 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

*Listing 10.4: Running the custom-agent evaluation.*

![](assets/_page_7_Picture_4.jpeg)

*Figure 10.3: ReAct example—performing a search.* [Yao et al., 2022](#page-10-0)

A structured experiment plan and consistent result logging make practical agentic-LLM experiments easier to run, evaluate, and improve.
## 10.4 Self-Reflection

After reproducing the benchmark results and experimenting with the agents, you can take on a larger challenge: extending a BALROG agent with self-reflection. The LLM can serve different roles, such as selecting actions, evaluating outcomes, and generating reflections. Because an adaptive system can become unstable when it optimizes a changing target, first make sure the base learner from the previous section works and converges predictably.

Two approaches to improving LLM agents are ReAct [Yao et al., 2022](#page-10-0), which focuses on prompting and tool use, and Reflexion [Shinn et al., 2024](#page-10-0), which adds short- and long-term memory to the agent algorithm.

{8}------------------------------------------------

<span id="page-8-0"></span>

![](assets/_page_8_Diagram_2.jpeg)

*Figure 10.4: Reflexion architecture.* [Shinn et al., 2024](#page-10-0)

**ReAct.** ReAct uses prompts to interact with the world and encourages agents to use tools to retrieve information (Figure [10.3](#page-7-0)). See the [ReAct website](https://react-lm.github.io) and [code repository](https://github.com/ysymyth/ReAct). ReAct prompting is also implemented in frameworks such as LangChain<sup>2</sup> and BeeAI<sup>3</sup>.

Review these frameworks to understand ReAct's prompt-based interaction style, then try implementing it in BALROG. Does it improve performance? Which games benefit from external information lookup?

**Reflexion.** Unlike prompt-based ReAct, Reflexion uses Python-coded agents to support more elaborate behavior. It uses the same LLM in different contexts with different prompts, based on the idea that critiquing an existing approach may be easier than inventing a successful one from scratch. See the [Reflexion repository](https://github.com/noahshinn/reflexion) and [a discussion of the method](https://nanothoughts.substack.com/p/reflecting-on-reflexion).

Figure 10.4 illustrates the Reflexion architecture, its prompts, and memory types. Figure [10.5](#page-9-0) presents the algorithm, with separate LLM calls for the actor \(M_a\), evaluator \(M_e\), and self-reflection module \(M_{sr}\). The algorithm is pseudocode; details such as memory management, prompt construction, and convergence criteria are left open. Consult the repository for implementation details.

<sup>2</sup> See <https://docs.langchain.com/oss/python/langchain/overview>.

<sup>3</sup> See the [BeeAI ReAct example](https://github.com/i-am-bee/beeai-framework/blob/main/python/examples/agents/react.py).
{9}------------------------------------------------

<span id="page-9-0"></span>

```text
Algorithm 1: Reinforcement via self-reflection

Initialize actor, evaluator, and self-reflection modules: M_a, M_e, M_sr
Initialize policy π_θ(a_i | s_i), with θ = {M_a, memory}
Generate initial trajectory τ_0 using π_θ
Evaluate τ_0 using M_e
Generate initial self-reflection s_r0 using M_sr
Set memory = {s_r0}
Set t = 0
while M_e does not pass, or t < maximum trials:
    Generate trajectory τ_t = (a_0, o_0, ..., a_t, o_t) using π_θ
    Evaluate τ_t using M_e
    Generate self-reflection s_rt using M_sr
    Append s_rt to memory
    Increment t
end while
return
```

*Figure 10.5: Reflexion algorithm.*

Reflexion may also lead to instability or loops. Its simple memory management can exceed the model's context window. You could explore chunking or other ways to compress memory, implemented directly in Python or with an LLM.

Try the Reflexion repository on tasks such as ALFWorld and HotpotQA.

**Extensions.** Implement a Reflexion agent in BALROG and run it on the benchmark. What problems arise? Does Reflexion improve performance over the standard agents? Then explore other reflection methods, applications, and LLMs.
## 10.5 Concluding Remarks

This case study explored how an LLM agent can perform self-reflection. We used BALROG, an open benchmark suite of games ranging from easy to difficult. Its open design lets us write custom agents and run them on the games.

This has been an important practical case study. You have explored writing an agent for in-context reasoning and extending it beyond base learning through self-reflection. You have also examined the stability of self-reflection across different LLMs and challenging benchmarks.

{10}------------------------------------------------

## <span id="page-10-0"></span>Bibliography

Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kucinski, Lerrel Pinto, Rob Fer- ´ gus, Jakob Nicolaus Foerster, Jack Parker-Holder, and Tim Rocktaschel. BAL- ¨ ROG: Benchmarking agentic LLM and VLM reasoning on games. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https:](https://openreview.net/forum?id=fp6t3F669F) [//openreview.net/forum?id=fp6t3F669F](https://openreview.net/forum?id=fp6t3F669F). [211,](#page-0-0) [212,](#page-1-0) [213](#page-2-0) Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36, 2024. [218,](#page-7-0) [219,](#page-8-0) [220](#page-9-0) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*, 2022. [218](#page-7-0)

{11}------------------------------------------------



