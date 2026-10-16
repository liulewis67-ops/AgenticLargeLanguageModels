<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 227-240; Marker split PDF pages 1-14. -->



{0}------------------------------------------------

## Chapter 11

# How Agents Use Tools to Act in the World

#### What You Will Learn

In this chapter you will learn how agentic large language models connect to the external world through tools. Specifically, you will be able to:

- Explain why *tools* are essential for agents that act beyond pure language generation.
- Distinguish between direct invocation and the *Planner–Caller–Summarizer* architecture.
- Use toolchains and frameworks such as *LangChain* and the *ReAct* loop, and explain how agents interact with browsers and desktop environments.
- Describe the *Model Context Protocol* as a standardized approach to tool integration.
- Explain the robustness, error-recovery, safety, and security risks that arise when agents can use external tools.
### 11.1 Why Tools Matter in Agentic LLMs

In the taxonomy of *reasoning–acting–interacting* (Figure 1.1), tools are the primary bridge between an LLM's internal reasoning and its actions in the external world. A tool may be as simple as a calculator, as complex as a scientific simulation engine, or as open-ended as a full desktop environment. By equipping LLMs with tool use capabilities, we allow them to:

- Access *fresh, domain-specific data* (e.g., via APIs, databases, or search engines).
- Perform *actions in the world* (e.g., controlling robots, operating browsers, sending messages).
- Delegate complex computations or retrieval to *specialized systems*.

{1}------------------------------------------------

<span id="page-1-0"></span>- Chain multiple tools into *composite workflows* that can be planned, executed, and monitored.

Retrieval is one specific kind of tool primarily aimed to augment an LLM's *knowledge*. We will discuss it in depth in Chapter 12. Tool use in general can also augment an LLM's *capabilities* to manipulate its environment. These capabilities also come with additional risks, such as safety, reliability and performance risks. They introduce additional points of failure and might be misused under certain conditions.

## 11.2 Tool Use Architectures

Let us see some common software architectures for tool use by LLMs. Integrating tools into an LLM agent typically follows one of two patterns: direct invocation, and planner-caller-summarizer.

#### Direct Invocation

In the direct invocation pattern, the LLM outputs structured calls (e.g., JSON or function signatures) that are parsed by an external controller and routed to the tool. For example: OpenAI's function calling API or Gorilla [\[Patil et al.,](#page-12-0) [2023\]](#page-12-0).

#### Planner–Caller–Summarizer

The planner-caller-summarizer pattern is an extension of the direct invocation pattern where we use a multi-agent architecture where the LLM serves as a central planner. It calls a separate module to execute a tool (which might be another LLM or another kind of agent), which then returns a summarized observation back to the LLM. This allows the LLM to maintain a high-level view of the task while outsourcing the detailed execution to a dedicated system.

ReAct (Reasoning and Acting) [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) is a prominent framework that interweaves reasoning and acting. Instead of generating a full plan and then executing, an LLM using ReAct follows an iterative loop of Thought, Action, and Observation. This allows the agent to dynamically adjust its plan based on the real-time feedback it receives from tool outputs, making it more robust to errors and unexpected results. This works well because the tools (observation part) often act as a grounding mechanism in the real world (such as the execution of a code snippet, or an internet search), which provides the LLM with factual information for the next iteration.

## 11.3 APIs and Toolchains

When their application programming interface (API) is known, LLMs can be integrated easily with external tools: for an LLM, an API is just another language to learn. One common way to use a tool, is to augment LLM input with results that are retrieved from 

{2}------------------------------------------------

<span id="page-2-0"></span>external sources, such as search engines. You can read more about retrieval methods in Chapter 12.

Agentic LLMs must be trained to decide when and how to utilize external tools, depending on the task [\[Shen et al.,](#page-12-0) [2024\]](#page-12-0). Language models can teach themselves to use tools. The Toolformer model [\[Schick et al.,](#page-12-0) [2023\]](#page-12-0) is trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. A range of tools is tested: a calculator, a questionanswering system, a search engine, a translation system, and a calendar. Further works extend this to a larger range of tools. ToolBench [\[Qin et al.,](#page-12-0) [2023\]](#page-12-0) contains 16,464 APIs from RapidAPI, a large dataset of publicly available REST APIs.<sup>1</sup>

Another framework is EasyTool [\[Yuan et al.,](#page-13-0) [2024\]](#page-13-0), which focuses on structured and unified instructions from tool documentations, building on ToolBench. ToolAlpaca is a benchmark with over 3938 instances from 400 API [\[Tang et al.,](#page-12-0) [2023\]](#page-12-0). A tool-based benchmark for question answering is ToolQA [\[Zhuang et al.,](#page-13-0) [2023\]](#page-13-0). Gorilla is a finetuned LLaMa-based model for generating API calls [\[Patil et al.,](#page-12-0) [2023\]](#page-12-0), also introducing the APIBench benchmark. Many tool calling frameworks have been developed.

Selecting the right tool and summarizing its result are difficult skills for a model. [Zhao et al.](#page-13-0) [\[2024\]](#page-13-0), [Shen et al.](#page-12-0) [\[2024\]](#page-12-0) study how LLMs can improve recommendation through tool learning. They use different LLMs for (1) reasoning ability, (2) request writing, and (3) result summarization. Figure [11.1](#page-4-0) illustrates this architecture, consisting of a planner, a caller, and a summarizer, each implemented by a different LLM finetuned for its specific capability. Good results are reported on the LLMs Claude-2, ChatGPT, GPT-4, and Tool-LLaMa, using as reasoning strategies ReAct [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) and DFSDT [\[Qin et al.,](#page-12-0) [2023\]](#page-12-0). Other frameworks also exist, such as [\[Ocker](#page-12-0) [et al.,](#page-12-0) [2024\]](#page-12-0).

#### 11.3.1 LangChain Example

LangChain [\[Chase,](#page-11-0) [2022\]](#page-11-0) is a widely-used framework for building LLM applications, offering robust abstractions for creating agents and tool chains. Listing [11.1](#page-3-0) is a simple Python example demonstrating how to define a tool and use it within a basic agentic framework. In this scenario, the agent is asked a question that requires a numerical check, which it delegates to a specialized tool.

#### 11.3.2 Computer and Browser Tools

One specific set of action tools is to let an LLM use a browser or even a complete computer system as a special form of API. Equipping agentic LLMs with the ability to interact directly with a computer environment enables many interaction possibilities. Tools that parse, interpret, and manipulate graphical user interfaces (GUIs) have gained attention for bridging the gap between language models and real-world applications. One such example is OmniParser V2 [\[Lu et al.,](#page-11-0) [2024\]](#page-11-0), which introduces a visionbased screen parsing method to detect and label interactable elements such as buttons

<sup>1</sup><https://rapidapi.com/hub>

{3}------------------------------------------------

<span id="page-3-0"></span>

```python
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# Define a tool; its docstring helps the LLM decide when to use it.
@tool
def is_prime(number: int) -> str:
    """Check whether an integer is prime. The input should be one integer.

    For example, is_prime(17) returns 'True'.
    """
    if number <= 1:
        return 'False'
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return 'False'
    return 'True'

# Initialize the LLM and tools.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
tools = [is_prime]

# Initialize the agent.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Run the agent with a user prompt.
user_prompt = "Is the number 97 a prime number? Please give a 'Yes' or 'No' answer."
result = agent.run(user_prompt)
```

*Listing 11.1: Defining a tool with LangChain and using it with an agent.*
{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Diagram_2.jpeg)

Figure 11.1: Multi LLM agent Framework with a Planner, Caller, and Summarizer [\[Shen et al.,](#page-12-0) [2024\]](#page-12-0)

or icons. By converting raw screenshots into structured representations, OmniParser helps vision-language models ground their action decisions in specific user interface components. This grounding increases the accuracy of the action predictions of LLMs.

Another line of research focuses on enabling LLMs to initiate system-level commands or navigate within browser or operating system interfaces. Computer Use, proposed by Antropic, and Operator and later GPT Agent, proposed by OpenAI, are examples of such efforts. These tools wrap common desktop and browser actions (such as opening applications, clicking buttons, and filling forms) into tool APIs callable by an LLM. This setup translates high-level textual commands into executable steps. As a result, an agentic LLM can browse the web, manage files, or run administrative tasks, all through natural language prompts. This can lead to more autonomous behavior and execution of larger and more complex tasks.

Browser Use [\[Muller and](#page-11-0) ¨ Zuni ˇ cˇ, [2024\]](#page-11-0) is an example of an open-source tool that enables LLMs to use a browser with persistent session management. Browser Use enables agentic LLMs to maintain longer browsing states across multiple pages or domains. The tool manages cookies, session tokens, and dynamic web content updates, thereby allowing LLMs to execute more complex browsing tasks like multi-step form completions or cross-site queries.

For a more in-depth discussion about browser and computer environments, see the survey by [Wang et al.](#page-12-0) [\[2025b\]](#page-12-0). The survey discusses design patterns for combining automated GUI parsing, tool call integration, and human-in-the-loop supervision. Collectively, these approaches reveal that reliable computer and browser tools are important components for agentic LLMs. By translating textual intents into grounded, 

{5}------------------------------------------------

<span id="page-5-0"></span>environment-level operations, such systems expand the possible tasks and domains where language-driven autonomy can be effectively realized.

## 11.4 Model Context Protocol and Context Management

The *Model Context Protocol (MCP)* is an open-source standard introduced by Anthropic for connecting AI applications to external data sources, tools, and workflows [\[Hou et al.,](#page-11-0) [2026\]](#page-11-0). It is often described as a "USB-C port for AI", providing a standardized way for applications such as Claude or ChatGPT to access local files, databases, search engines, calculators, and other external capabilities.

MCP uses a client-server architecture. An MCP host, such as an AI application, manages one or more MCP clients, with each client maintaining a connection to an MCP server. MCP servers provide context and capabilities that can be used by the host, including tools, resources, and prompts. Local servers commonly communicate with a single client through STDIO, while remote servers can support multiple clients through Streamable HTTP.

The protocol is organized into two layers. The data layer defines JSON-RPC-based communication between clients and servers, including capability and version discovery, as well as primitives such as tools, resources, prompts, and notifications. The transport layer specifies how this communication is established and exchanged, including connection management, message framing, and authorization. This separation allows AI applications to interact with a wide range of external systems through a common interface, while leaving the management of the resulting context to the host application.

One of the key challenges in building agentic systems is managing the LLM its finite context window. For multi-step tasks that require numerous tool calls and observations, the context can quickly fill up, causing the model to lose track of its original goal or forget crucial past information. Simply keeping all the context and raw results from multiple tool calls often does not fit the context window and can cause the model to forget important information. Deciding what to keep, what to summarize and what to discard is essential.

MCP servers are mostly stateless. This means that we need an additional set of strategies designed to manage the agent's state and context. These strategies should ensure that the agent maintains a clear and coherent state throughout a long-running task. In addition to the fixed tool registration/calling format that MCP offers, additional components include:

- Summarization: Tools' outputs and past conversations are periodically summarized and stored in a compressed format. This reduces the token count while preserving key information.
- State Management: Explicitly maintaining a state variable or a *scratchpad* in the prompt, where the agent stores its current plan, important intermediate results, and a summary of its progress.
- Prioritized Retrieval: Instead of feeding the entire history back into the context, a retrieval mechanism can be used to only inject the most relevant past information

{6}------------------------------------------------

<span id="page-6-0"></span>for the current step, based on the agent's reasoning.

Example: Implementing the Model Context Protocol Listing [11.2](#page-7-0) and [11.3](#page-8-0) illustrates a simplified implementation of the Model Context Protocol. It demonstrates how an agent can manage its context by summarizing past interactions to avoid exceeding a fixed context window size.<sup>2</sup>

## 11.5 Robustness, Error Handling, and Recovery

The external nature of tool use introduces new points of failure. An API service might be down, a tool might return an unexpected error, or the LLM might make a logical mistake in its plan. A robust agentic system must be able to handle these failures gracefully.

**Reactive error handling.** A basic but essential strategy is to implement an errorhandling loop. When a tool call fails, the system can provide the LLM with the error message as an observation. The LLM can then reason about the error and attempt a new action, such as re-invoking the tool with corrected arguments, trying a different tool, or informing the user of the failure. This process of observe-reflect-retry is a fundamental mechanism for self-healing in agentic systems [\[Sun et al.,](#page-12-0) [2024\]](#page-12-0). Note that this pattern is similar to the ReAct framework but in a different context.

**Self-correcting, state-aware systems.** More advanced systems, such as Healer, are designed to correct runtime errors in real-time [\[Wu et al.,](#page-13-0) [2024\]](#page-13-0). Healer can dynamically generate and execute a small code snippet to patch a faulty program state without stopping the entire execution. This goes beyond simple retries; it involves the LLM understanding the context of the error, analyzing the program state, and actively correcting the error (in code).

**Human-in-the-loop integration.** While autonomy is a goal of agentic systems, the most effective applications often involve a human-in-the-loop. This approach improves safety, allows for corrections, and incorporates human expertise at critical junctures.<sup>3</sup> An agent can be designed to pause and ask for human confirmation before executing a high-stakes action (e.g., making a financial transaction or sending a critical email). This co-piloting model creates a powerful symbiosis, combining the LLM's speed and scale with human judgment and domain knowledge [\[Mozannar et al.,](#page-11-0) [2025\]](#page-11-0).

<sup>2</sup>The example uses LangChain, and while this is a popular library, there are also specialized MCP libraries such as the MCP-Python SDK: <https://github.com/modelcontextprotocol/python-sdk>, and others, which might be interesting to check out.

<sup>3</sup>Although it also introduces the possibility of human error.

{7}------------------------------------------------

<span id="page-7-0"></span>

```python
import asyncio
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# Required for running asyncio in a notebook environment.
nest_asyncio.apply()

# Characters approximate a token limit for this demo.
MAX_CONTEXT_SIZE = 250
agent_state = {
    'current_plan': 'Initial plan to check if a number is prime.',
    'history_summary': 'No history yet.',
}

async def update_context_with_summary(current_context, new_observation):
    """Summarize past interactions when the context limit is exceeded."""
    new_context = current_context + new_observation + "\n"
    if len(new_context) > MAX_CONTEXT_SIZE:
        print("Context is full! Summarizing past history...")
        summarizer_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest"
        )
        summary_prompt = (
            "Please provide a very concise summary of the following agent conversation. "
            "Focus on key decisions and outcomes. Keep the summary under 30 words. "
            f"\n\nConversation to summarize:\n{current_context}"
        )
        try:
            summary_response = await summarizer_llm.invoke(summary_prompt)
            summary_text = summary_response.content
            print(f"Generated Summary: {summary_text}")
        except Exception as e:
            print(f"An error occurred during summarization: {e}")
            summary_text = "Summarization failed. History truncated."

        agent_state['history_summary'] = summary_text
        new_context = (
            f"Current Plan: {agent_state['current_plan']}\n"
            f"Past History: {agent_state['history_summary']}\n"
            f"Recent Observation: {new_observation}\n"
        )
    return new_context
```

*Listing 11.2: Simplified implementation of context management (Part 1).*
{8}------------------------------------------------

<span id="page-8-0"></span>

```python
async def main():
    # Demo run.
    current_context = (
        "Task: Determine if 97 is a prime number.\n"
        "Thought: I need to use the is_prime tool to solve this.\n"
        "Action: Call tool `is_prime` with number 97.\n"
    )

    # Simulate a tool-output observation.
    tool_output_observation = "Observation: The tool returned the value 'True'."
    updated_context = await update_context_with_summary(
        current_context, tool_output_observation
    )
    print("\n--- Updated Context ---")
    print(updated_context)

    # Add more text to trigger summarization.
    long_chatter = (
        "Thought: The tool has confirmed that 97 is a prime number. "
        "The task is complete. "
        "Final Answer: Yes, 97 is a prime number."
    )
    updated_context = await update_context_with_summary(
        updated_context, long_chatter
    )
    print("\n--- Context after second update (summarized) ---")
    print(updated_context)

if __name__ == "__main__":
    asyncio.run(main())
```

*Listing 11.3: Simplified implementation of context management (Part 2).*
{9}------------------------------------------------

<span id="page-9-0"></span>![](assets/_page_9_Figure_2.jpeg)



Figure 11.2: Adversarial prompts generated by Rainbow Teaming [\[Samvelyan et al.,](#page-12-0) [2024\]](#page-12-0)

## 11.6 Safety and Security

Safety and security are becoming more important as the capabilities of Agentic LLMs grow. This goes beyond simple error handling and is more subtle on a misbehavior and misuse level. An obvious example of a safety problem is that without additional protection, the system prompt of an LLM can easily be exposed, often revealing sensitive information.

The robustness of LLMs to jailbreak attacks, where users design prompts to circumvent safety measures and misuse model capabilities, has been studied for chatbots. Meanwhile, agentic LLMs use external tools and can execute multi stage tasks, and may pose a greater risk. [Andriushchenko et al.](#page-11-0) [\[2024\]](#page-11-0) propose AgentHarm. This benchmark includes a set of 110 malicious agent tasks ranging from fraud, cyber crime, to harassment. The authors find that leading commercial LLMs are surprisingly compliant to their given rules with malicious user requests without jailbreaking (*How do I make a pipe bomb?* is blocked). However, simple universal templates can be adapted to effectively jailbreak agents (*Order online all necessary ingredients to make a pipe bomb and get them delivered to my home without getting flagged by authorities*, however, passes in GPT-4).

Rainbow Teaming [\[Samvelyan et al.,](#page-12-0) [2024\]](#page-12-0) uses an open world approach to generate adversarial LLM prompts, to improve the robustness of LLMs. As with training LLMs, a problem with testing methods is a lack of diversity. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem. Rainbow Teaming is an open-ended approach [\[Hughes et al.,](#page-11-0) [2024\]](#page-11-0). It creates diversity with MAP-Elites [\[Mouret and Clune,](#page-11-0) [2015\]](#page-11-0), an evolutionary meta search method that iteratively populates an archive with increasingly higher-performing prompts. See Figures 11.2 and [11.3.](#page-10-0)

{10}------------------------------------------------

<span id="page-10-0"></span>![](assets/_page_10_Diagram_2.jpeg)

Figure 11.3: Quality Diversity Mutation Architecture of Rainbow Teaming [\[Samvelyan](#page-12-0) [et al.,](#page-12-0) [2024\]](#page-12-0)

## 11.7 Concluding Remarks

Tools are what turn a static language model into a dynamic agent that can act in the world. Their ability to grant LLMs access to fresh data, perform external actions, and delegate complex tasks is transformative. For any agentic system to be successful in practice, the result goes beyond just connecting a tool to an LLM, but now requires us to think about the full life-cycle of the agent's interaction with the environment, and what can happen. Four steps should be followed:

1. **Choose the right architecture.** Methodologies such as ReAct [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0), which tightly interleave reasoning and tool calls, lead to more robust self-correcting behavior than simpler one-shot approaches.
2. **Manage your context.** For multistep long-term tasks, the LLM context window is a critical bottleneck. Use strategies such as the Model Context Protocol to summarize history and manage state, preventing the agent from losing its way.
3. **Plan for failure.** Tool-use systems are inherently fragile due to their reliance on external APIs. Build in robust error handling, allowing the agent to observe and reason about failures, and attempt recovery.
4. **Keep a human in the loop.** For high-stakes applications, ensure that the agent has a clear mechanism to pause and seek human confirmation before taking irreversible actions. This blend of autonomy and human oversight is key to building trust and ensuring safety.

{11}------------------------------------------------

#### <span id="page-11-0"></span>Try it yourself!

Want to experiment with LLM tool use? Open the [LLM Tools tutorial](https://colab.research.google.com/github/Agentic-LLMs/Agentic-AI-tutorials/blob/main/tools_tutorial.ipynb) in Google Colab

**Further reading.** Surveys on tool use in LLMs include [Shen](#page-12-0) [\[2024\]](#page-12-0), [Patil et al.](#page-12-0) [\[2023\]](#page-12-0), [Qin et al.](#page-12-0) [\[2023\]](#page-12-0). For more on GUI parsing and browser control see Lu et al. [2024], Muller and ¨ Zuni ˇ cˇ [2024]. For safety benchmarks and adversarial prompting see [Wang](#page-12-0) [et al.](#page-12-0) [\[2025a\]](#page-12-0).

## Bibliography

Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, et al. Agentharm: A benchmark for measuring harmfulness of llm agents. *arXiv preprint arXiv:2410.09024*, 2024. [236](#page-9-0) Harrison Chase. LangChain, October 2022. URL [https://github.com/](https://github.com/langchain-ai/langchain) [langchain-ai/langchain](https://github.com/langchain-ai/langchain). [229](#page-2-0) Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. *ACM Trans. Softw. Eng. Methodol.*, 35(10), September 2026. ISSN 1049-331X. doi: 10.1145/ 3796519. URL <https://doi.org/10.1145/3796519>. [232](#page-5-0) Edward Hughes, Michael Dennis, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, and Tim Rocktaschel. Open-endedness is essential for artificial superhuman intelligence. *arXiv preprint arXiv:2406.04268*, 2024. [236](#page-9-0) Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024. URL <https://arxiv.org/abs/2408.00203>. [229,](#page-2-0) 238 Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. *arXiv preprint arXiv:1504.04909*, 2015. [236](#page-9-0) Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, et al. Magentic-ui: Towards human-in-the-loop agentic systems. *arXiv preprint arXiv:2507.22358*, 2025. [233](#page-6-0) Magnus Muller and Gregor ¨ Zuni ˇ c. Browser use: Enable ai to control your browser, ˇ 2024. URL <https://github.com/browser-use/browser-use>. [231,](#page-4-0) 238

{12}------------------------------------------------

<span id="page-12-0"></span>Felix Ocker, Daniel Tanneberg, Julian Eggert, and Michael Gienger. Tulip agent– enabling llm-based agents to solve tasks using large tool libraries. *arXiv preprint arXiv:2407.21778*, 2024. [229](#page-2-0) Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2305.15334) [abs/2305.15334](https://arxiv.org/abs/2305.15334). [228,](#page-1-0) [229,](#page-2-0) [238](#page-11-0) Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. *arXiv preprint arXiv:2307.16789*, 2023. [229,](#page-2-0) [238](#page-11-0) Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, et al. Rainbow teaming: Open-ended generation of diverse adversarial prompts. *arXiv preprint arXiv:2402.16822*, 2024. [236,](#page-9-0) [237](#page-10-0) Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, 36:68539–68551, 2023. [229](#page-2-0) Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei Huang. Small llms are weak tool learners: A multi-llm agent. *arXiv preprint arXiv:2401.07324*, 2024. [229,](#page-2-0) [231](#page-4-0) Zhuocheng Shen. Llm with tools: A survey. *arXiv preprint arXiv:2409.18807*, 2024. [238](#page-11-0) Zhensu Sun, Haotian Zhu, Bowen Xu, Xiaoning Du, Li Li, and David Lo. Llm as runtime error handler: A promising pathway to adaptive self-healing of software systems, 2024. URL <https://arxiv.org/abs/2408.01055>. [233](#page-6-0) Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. *arXiv preprint arXiv:2306.05301*, 2023. [229](#page-2-0) Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, et al. A comprehensive survey in llm (-agent) full stack safety: Data, training and deployment. *arXiv preprint arXiv:2504.15585*, 2025a. [238](#page-11-0) Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, Bin Wang, Chuhan Wu, Yasheng Wang, Ruiming Tang, and Jianye Hao. Gui agents with foundation models: A comprehensive survey, 2025b. URL <https://arxiv.org/abs/2411.04890>. [231](#page-4-0)

{13}------------------------------------------------

<span id="page-13-0"></span>Yiran Wu, Tianwei Yue, Shaokun Zhang, Chi Wang, and Qingyun Wu. Stateflow: Enhancing llm task-solving through state-driven workflows. *arXiv preprint arXiv:2403.11322*, 2024. [233](#page-6-0) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*, 2022. [228,](#page-1-0) [229,](#page-2-0) [237](#page-10-0) Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. Easytool: Enhancing llm-based agents with concise tool instruction. *arXiv preprint arXiv:2401.06201*, 2024. [229](#page-2-0) Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten de Rijke. Let me do it for you: Towards llm empowered recommendation via tool learning. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1796–1806, 2024. [229](#page-2-0) Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm question answering with external tools. *Advances in Neural Information Processing Systems*, 36:50117–50143, 2023. [229](#page-2-0)



