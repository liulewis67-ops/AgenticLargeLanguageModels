<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 227-240; Marker split PDF pages 1-14. -->



{0}------------------------------------------------

## 第11章

# 智能体如何使用工具在世界中行动

#### 你将学到的内容

本章将介绍智能体大语言模型如何借助工具连接外部世界。学完本章后，你将能够：

- 解释为什么*工具*对于超越纯文本生成的代理至关重要。
- 区分直接调用与*规划器–调用器–总结器*架构。
- 使用 *LangChain*、*ReAct* 等工具链与框架，并说明代理如何与浏览器和桌面环境交互。
- 说明*模型上下文协议*如何为工具集成提供标准化方式。
- 分析代理使用外部工具时面临的稳健性、错误恢复、安全与安保风险。
### 11.1 工具在智能体大语言模型中为何重要

在“推理–行动–互动”的分类法中（图1.1），工具是LLM内部推理与其在外部世界中行为之间的主要桥梁。工具可以像计算器一样简单，也可以像科学模拟引擎一样复杂，或者像完整的桌面环境一样开放。通过赋予LLM 工具使用的能力，我们让它们能够：

- 访问*新鲜的、特定领域的数据*（例如，通过 API、数据库或搜索引擎）。
- 在世界中执行*动作*（例如，控制机器人、操作浏览器、发送消息）。
- 将复杂的计算或检索委托给*专门系统*。

{1}------------------------------------------------

<span id="page-1-0"></span>- 将多个工具组合成*复合工作流*，并进行规划、执行和监控。

检索是一种特定类型的工具，主要用于增强 LLM 的*知识*。我们将在第12章详细讨论它。一般来说，工具使用也可以增强 LLM 操作其环境的*能力*。这些能力也带来了额外的风险，例如安全性、可靠性和性能风险。它们引入了额外的故障点，并可能在特定条件下被滥用。

## 11.2 工具使用架构

让我们看看 LLM 的工具使用的一些常见软件架构。将工具集成到 LLM 代理中通常遵循两种模式之一：直接调用，以及规划者-调用者-总结者。

#### 直接调用

在直接调用模式中，LLM 输出结构化调用（例如，JSON 或函数签名），这些调用由外部控制器解析并路由到工具。例如：OpenAI 的函数调用 API 或 Gorilla [\[Patil et al.,](#page-12-0) [2023\]](#page-12-0)。

#### 规划器–调用器–总结器

规划者-调用者-总结者模式是直接调用模式的扩展，在这种模式中，我们使用多智能体架构，其中 LLM 作为中央规划者。它调用一个独立的模块来执行工具（该工具可能是另一个 LLM 或其他类型的代理），然后将总结后的观察结果返回给 LLM。这使得 LLM 能够保持任务的高级视图，同时将详细的执行外包给专用系统。

ReAct（推理与行动）[\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) 是一种将推理与行动交织在一起的突出框架。与先生成完整计划再执行不同，使用 ReAct 的 LLM 遵循一个思考、行动和观察的迭代循环。这使得智能体能够根据从工具输出中获得的实时反馈动态调整其计划，从而在面对错误和意外结果时更加稳健。这效果很好，因为工具（观察部分）通常充当现实世界中的基础机制（例如执行代码片段或进行互联网搜索），为下一次迭代提供 LLM 的事实信息。

## 11.3 APIs 和工具链

当他们的应用程序接口（API）已知时，LLM 可以轻松与外部工具集成：对于一个 LLM，API 只是另一种需要学习的语言。使用工具的一种常见方法是，用从 
 获取的结果增加 LLM 的输入。
{2}------------------------------------------------

<span id="page-2-0"></span>外部来源，例如搜索引擎。您可以在第12章中阅读有关检索方法的更多内容。

智能体大语言模型必须经过训练，以决定何时以及如何使用外部工具，这取决于任务 [\[Shen et al.,](#page-12-0) [2024\]](#page-12-0)。语言模型可以自学使用工具。Toolformer 模型 [\[Schick et al.,](#page-12-0) [2023\]](#page-12-0) 经过训练以决定调用哪些 API、何时调用、传递哪些参数，以及如何将结果最好地融入未来的令牌预测中。测试了一系列工具：计算器、问答系统、搜索引擎、翻译系统和日历。进一步的工作将其扩展到更多的工具。ToolBench [\[Qin et al.,](#page-12-0) [2023\]](#page-12-0) 包含 RapidAPI 提供的 16,464 个 API，这是一个大型的公开 REST API 数据集。<sup>1</sup>

另一个框架是 EasyTool [\[Yuan et al.,](#page-13-0) [2024\]](#page-13-0)，它专注于来自工具文档的结构化和统一指令，基于 ToolBench 构建。ToolAlpaca 是一个基准，包含来自 400 个 API 的超过 3938 个实例 [\[Tang et al.,](#page-12-0) [2023\]](#page-12-0)。用于问答的基于工具的基准是 ToolQA [\[Zhuang et al.,](#page-13-0) [2023\]](#page-13-0)。Gorilla 是一个基于 LLaMa 微调的用于生成 API 调用的模型 [\[Patil et al.,](#page-12-0) [2023\]](#page-12-0)，同时引入了 APIBench 基准。已经开发了许多工具调用框架。

为模型选择合适的工具并总结其结果是困难的技能。[Zhao et al.](#page-13-0) [\[2024\]](#page-13-0)、[Shen et al.](#page-12-0) [\[2024\]](#page-12-0) 研究了 LLM 如何通过工具学习来改进推荐。他们使用不同的 LLM 来进行 (1) 推理能力、(2) 请求编写和 (3) 结果总结。图 [11.1](#page-4-0) 展示了这一架构，包括一个规划器、一个调用器和一个总结器，每个模块都由针对其特定能力微调的不同 LLM 实现。在 LLM Claude-2、ChatGPT、GPT-4 和 Tool-LLaMa 上报告了良好的结果，使用的推理策略包括 ReAct [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) 和 DFSDT [\[Qin et al.,](#page-12-0) [2023\]](#page-12-0)。其他框架也存在，例如 [\[Ocker](#page-12-0) [et al.,](#page-12-0) [2024\]](#page-12-0)。

#### 11.3.1 LangChain 示例

LangChain [\[Chase,](#page-11-0) [2022\]](#page-11-0) 是一个用于构建 LLM 应用程序的广泛使用的框架，提供了用于创建代理和工具链的强大抽象。清单 [11.1](#page-3-0) 是一个简单的 Python 示例，演示如何定义一个工具并在基本的代理框架中使用它。在这种情况下，代理被问到一个需要数值检查的问题，它将该问题委托给一个专门的工具。

#### 11.3.2 计算机和浏览器工具

一组特定的操作工具是让 LLM 使用浏览器甚至完整的计算机系统作为一种特殊形式的 API。赋予智能体大语言模型直接与计算机环境交互的能力，使得许多交互可能性成为可能。解析、解释和操作图形用户界面（GUI）的工具因弥合语言模型与现实世界应用之间的差距而受到关注。一个这样的例子是 OmniParser V2 [\[Lu et al.,](#page-11-0) [2024\]](#page-11-0)，它引入了一种基于视觉的屏幕解析方法，用于检测和标记可交互元素，例如按钮 

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

图 11.1：具有规划器、呼叫器和总结器的多 LLM 代理框架 [\[Shen et al.,](#page-12-0) [2024\]](#page-12-0)

或图标。通过将原始截图转换为结构化表示，OmniParser 帮助 vision-语言模型将其操作决策建立在特定的用户界面组件上。这种基础增加了 LLM 操作预测的准确性。

另一条研究路线集中于使 LLM 能够启动系统级命令或在浏览器或操作系统界面中导航。由 Antropic 提出的 Computer Use，以及由 OpenAI 提出的 Operator 和后来的 GPT Agent，就是这类努力的例子。这些工具将常见的桌面和浏览器操作（例如打开应用程序、点击按钮和填写表单）封装到可由 LLM 调用的工具 API 中。这种设置将高级文本命令转化为可执行的步骤。因此，智能体大语言模型可以通过自然语言提示来浏览网页、管理文件或执行管理任务。这可以导致更自主的行为和执行更大、更复杂任务的能力。

浏览器使用 [\[Muller and](#page-11-0) ¨ Zuni ˇ cˇ，[2024\]](#page-11-0) 是一个开源工具示例，它允许 LLM 使用具有持久会话管理的浏览器。浏览器使用使智能体大语言模型能够在多个页面或域之间维持更长的浏览状态。该工具管理 Cookie、会话令牌和动态网页内容更新，从而允许 LLM 执行更复杂的浏览任务，如多步骤表单填写或跨站查询。

有关浏览器和计算机环境的更深入讨论，请参阅 [Wang et al.](#page-12-0) [\[2025b\]](#page-12-0) 的调查。该调查讨论了将自动化 GUI 解析、工具调用集成以及人工参与监督相结合的设计模式。总体而言，这些方法表明，可靠的计算机和浏览器工具是智能体大语言模型的重要组成部分。通过将文本意图转化为具体的 

{5}------------------------------------------------

<span id="page-5-0"></span>环境级操作，这样的系统扩展了语言驱动自主性可以有效实现的可能任务和领域。

## 11.4 模型上下文协议与上下文管理

*模型上下文协议（MCP）* 是由 Anthropic 推出的一个开源标准，用于将 AI 应用程序连接到外部数据源、工具和工作流 [\[Hou et al.,](#page-11-0) [2026\]](#page-11-0)。它通常被描述为 AI 的“USB-C接口”，为像 Claude 或 ChatGPT 这样的应用程序提供了一种访问本地文件、数据库、搜索引擎、计算器和其他外部功能的标准化方式。

MCP 使用客户端-服务器架构。MCP 主机，例如 AI 应用程序，管理一个或多个 MCP 客户端，每个客户端与 MCP 服务器保持连接。MCP 服务器提供主机可使用的上下文和功能，包括工具、资源和提示。本地服务器通常通过 STDIO 与单个客户端通信，而远程服务器则可以通过可流式 HTTP 支持多个客户端。

该协议分为两层。数据层定义了客户端与服务器之间基于 JSON-RPC 的通信，包括能力和版本的发现，以及工具、资源、提示和通知等原语。传输层则规定了如何建立和交换这种通信，包括连接管理、消息帧和授权。这种分离使 AI 应用能够通过一个通用接口与各种外部系统进行交互，同时将生成的上下文管理留给主机应用程序。

构建自主系统的一个关键挑战是管理 LLM 的有限上下文窗口。对于需要多次工具调用和观察的多步骤任务，上下文很快就会填满，导致模型失去对其原始目标的跟踪或者忘记关键的过去信息。仅仅保留所有上下文和多次工具调用的原始结果通常无法适应上下文窗口，因此会导致遗忘。决定保留什么、总结什么以及舍弃什么至关重要。

MCP 服务器大多是无状态的。这意味着我们需要一套额外的策略来管理代理的状态和上下文。这些策略应确保代理在长时间运行的任务中保持清晰且连贯的状态。除了 MCP 提供的固定工具注册/调用格式之外，还应包括的额外组件有：

- 总结：工具的输出和过去的对话会定期被总结并以压缩格式存储。这在保留关键信息的同时减少了令牌数量。
- 状态管理：在提示中显式维护一个状态变量或*备忘录*，在其中代理存储其当前计划、重要的中间结果以及进展概要。
- 优先检索：与将整个历史信息反馈到上下文中不同，可以使用检索机制仅注入最相关的过去信息

{6}------------------------------------------------

<span id="page-6-0"></span>对于当前步骤，基于代理的推理。
  
 示例：实现模型上下文协议清单 [11.2](#page-7-0) 和 [11.3](#page-8-0) 展示了模型上下文协议的简化实现。它演示了代理如何通过总结过去的交互来管理其上下文，以避免超过固定的上下文窗口大小。<sup>2</sup> 

## 11.5 稳健性、错误处理与恢复

工具使用的外部性质引入了新的故障点。API 服务可能会宕机，一个工具可能会返回意外错误，或者 LLM 可能在其计划中犯逻辑错误。一个健壮的自治系统必须能够优雅地处理这些故障。

反应式错误处理一个基本但至关重要的策略是实现一个错误处理循环。当工具调用失败时，系统可以将错误信息作为观察结果提供给 LLM。然后，LLM 可以根据错误进行推理，并尝试新的操作，例如使用更正后的参数重新调用工具、尝试不同的工具，或者通知用户失败情况。这个观察-反思-重试的过程是具有代理系统自我修复的基本机制 [\[Sun et al.,](#page-12-0) [2024\]](#page-12-0)。请注意，这种模式与 ReAct 框架类似，但应用于不同的场景。

自我纠正与状态感知系统更高级的系统，如 Healer，被设计用于实时纠正运行时错误 [\[Wu et al.,](#page-13-0) [2024\]](#page-13-0)。Healer 可以动态生成并执行一个小的代码片段，以修复有故障的程序状态，而无需停止整个执行过程。这超越了简单的重试；它涉及 LLM 理解错误的上下文，分析程序状态，并主动纠正错误（在代码中）。

人类参与的整合尽管自主性是智能代理系统的目标，但最有效的应用通常涉及人类参与。这种方法提高了安全性，允许进行纠正，并在关键环节融入人类专业知识。<sup>3</sup> 代理可以被设计为在执行高风险操作之前暂停并请求人类确认（例如，进行金融交易或发送重要电子邮件）。这种共同操作模式创造了强大的共生关系，将LLM的速度和规模与人类的判断力和领域知识相结合 [\[Mozannar et al.,](#page-11-0) [2025\]](#page-11-0)。

<sup>2</sup>这个示例使用了 LangChain，虽然这是一个流行的库，但也有一些专门的 MCP 库，例如 MCP-Python SDK: <https://github.com/modelcontextprotocol/python-sdk>，以及其他一些库，可能值得查看。
  
 <sup>3</sup>尽管它也引入了人为错误的可能性。

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
  
  
  
 图 11.2：彩虹团队生成的对抗性提示 [\[Samvelyan et al.,](#page-12-0) [2024\]](#page-12-0) 

## 11.6 安全与防护

随着智能体大语言模型功能的增强，安全性和保障变得越来越重要。这不仅仅是简单的错误处理，还涉及对不当行为和误用的更细微层面的处理。安全问题的一个明显例子是，如果没有额外的保护，LLM 的系统提示信息很容易被暴露，经常会泄露敏感信息。

已经对 LLM 抵御越狱攻击的稳健性进行了研究，这类攻击中用户设计提示以规避安全措施并滥用模型能力，同时用于聊天机器人。与此同时，智能体大语言模型使用外部工具，能够执行多阶段任务，可能带来更大的风险。[Andriushchenko et al.](#page-11-0) [\[2024\]](#page-11-0) 提出了 AgentHarm。该基准包含一套 110 个恶意代理任务，范围涵盖欺诈、网络犯罪到骚扰。作者发现，领先的商业 LLM 对恶意用户请求表现出令人惊讶的遵守其设定规则的能力，即使在不越狱的情况下（*如何制造管炸弹？* 被阻止）。然而，简单的通用模板可以被改编来有效地绕过代理 (*在线订购制造土制炸弹的所有必要原料并送到我家而不被当局发现*，不过，在 GPT-4 中可以通过)。

彩虹团队[\[Samvelyan et al.,](#page-12-0) [2024\]](#page-12-0) 使用一种开放世界的方法生成对抗性LLM提示，以提高LLM的鲁棒性。与训练LLM一样，测试方法存在的问题是缺乏多样性。彩虹团队将对抗性提示生成视为一个质量-多样性问题。彩虹团队是一种开放式方法[\[Hughes et al.,](#page-11-0) [2024\]](#page-11-0)。它通过MAP-Elites [\[Mouret and Clune,](#page-11-0) [2015\]](#page-11-0)创建多样性，这是一种进化元搜索方法，通过迭代地用性能越来越高的提示填充档案。见图11.2和[11.3.](#page-10-0)

{10}------------------------------------------------

<span id="page-10-0"></span>![](assets/_page_10_Diagram_2.jpeg) 
  
 图11.3：彩虹团队质量多样性突变架构 [\[Samvelyan](#page-12-0) [et al.,](#page-12-0) [2024\]](#page-12-0) 
 
## 11.7 结语

工具是将一个静态的语言模型转变为能够在世界中行动的动态代理的关键。它们赋予 LLM 访问新数据、执行外部操作以及委派复杂任务的能力具有变革性。对于任何一个代理系统在实践中取得成功，其结果不仅仅是将工具连接到 LLM，而是现在要求我们考虑代理与环境交互的整个生命周期，以及可能发生的情况。应遵循四个步骤：

- 1. 选择合适的架构：像 ReAct [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) 这样将推理与工具调用紧密交织的方法，比简单的一次性方法更能产生更稳健的自我纠正行为。
- 2. 管理你的上下文：对于多步骤的长期任务，LLM 上下文窗口是一个关键瓶颈。使用诸如模型上下文协议之类的策略来总结历史并管理状态，防止代理迷失方向。
- 3. 接纳失败：工具使用系统由于依赖外部 API，本质上是脆弱的。构建强大的错误处理机制，使代理能够观察和推理失败，并尝试恢复。
- 4. 保持人工干预：对于高风险应用，确保代理有明确的机制在执行不可逆的操作前暂停并寻求人类确认。这种自主性与人工监管的结合是建立信任和确保安全的关键。

{11}------------------------------------------------

#### <span id="page-11-0"></span>动手试试！

想要尝试使用 LLM 工具吗？在 Google Colab 打开我们的：Ð [LLM Tools tutorial](https://colab.research.google.com/github/Agentic-LLMs/Agentic-AI-tutorials/blob/main/tools_tutorial.ipynb) 
 
 进一步阅读关于工具使用在 LLM 上的调查包括 [Shen](#page-12-0) [\[2024\]](#page-12-0)、[Patil et al.](#page-12-0) [\[2023\]](#page-12-0)、[Qin et al.](#page-12-0) [\[2023\]](#page-12-0)。有关 GUI 解析和浏览器控制的更多内容，请参见 Lu 等人 [2024]、Muller 和 ¨ Zuni ˇ cˇ [2024]。有关安全基准和对抗性提示，请参见 [Wang](#page-12-0) [et al.](#page-12-0) [\[2025a\]](#page-12-0)。

## 参考文献

Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, et al. Agentharm: A benchmark for measuring harmfulness of llm agents. *arXiv preprint arXiv:2410.09024*, 2024. [236](#page-9-0) Harrison Chase. LangChain, October 2022. URL [https://github.com/](https://github.com/langchain-ai/langchain) [langchain-ai/langchain](https://github.com/langchain-ai/langchain). [229](#page-2-0) Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. *ACM Trans. Softw. Eng. Methodol.*, 35(10), September 2026. ISSN 1049-331X. doi: 10.1145/ 3796519. URL <https://doi.org/10.1145/3796519>. [232](#page-5-0) Edward Hughes, Michael Dennis, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, and Tim Rocktaschel. Open-endedness is essential for artificial superhuman intelligence. *arXiv preprint arXiv:2406.04268*, 2024. [236](#page-9-0) Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024. URL <https://arxiv.org/abs/2408.00203>. [229,](#page-2-0) 238 Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. *arXiv preprint arXiv:1504.04909*, 2015. [236](#page-9-0) Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, et al. Magentic-ui: Towards human-in-the-loop agentic systems. *arXiv preprint arXiv:2507.22358*, 2025. [233](#page-6-0) Magnus Muller and Gregor ¨ Zuni ˇ c. Browser use: Enable ai to control your browser, ˇ 2024. URL <https://github.com/browser-use/browser-use>. [231,](#page-4-0) 238

{12}------------------------------------------------

<span id="page-12-0"></span>Felix Ocker, Daniel Tanneberg, Julian Eggert, and Michael Gienger. Tulip agent– enabling llm-based agents to solve tasks using large tool libraries. *arXiv preprint arXiv:2407.21778*, 2024. [229](#page-2-0) Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2305.15334) [abs/2305.15334](https://arxiv.org/abs/2305.15334). [228,](#page-1-0) [229,](#page-2-0) [238](#page-11-0) Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. *arXiv preprint arXiv:2307.16789*, 2023. [229,](#page-2-0) [238](#page-11-0) Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, et al. Rainbow teaming: Open-ended generation of diverse adversarial prompts. *arXiv preprint arXiv:2402.16822*, 2024. [236,](#page-9-0) [237](#page-10-0) Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, 36:68539–68551, 2023. [229](#page-2-0) Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei Huang. Small llms are weak tool learners: A multi-llm agent. *arXiv preprint arXiv:2401.07324*, 2024. [229,](#page-2-0) [231](#page-4-0) Zhuocheng Shen. Llm with tools: A survey. *arXiv preprint arXiv:2409.18807*, 2024. [238](#page-11-0) Zhensu Sun, Haotian Zhu, Bowen Xu, Xiaoning Du, Li Li, and David Lo. Llm as runtime error handler: A promising pathway to adaptive self-healing of software systems, 2024. URL <https://arxiv.org/abs/2408.01055>. [233](#page-6-0) Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. *arXiv preprint arXiv:2306.05301*, 2023. [229](#page-2-0) Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, et al. A comprehensive survey in llm (-agent) full stack safety: Data, training and deployment. *arXiv preprint arXiv:2504.15585*, 2025a. [238](#page-11-0) Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, Bin Wang, Chuhan Wu, Yasheng Wang, Ruiming Tang, and Jianye Hao. Gui agents with foundation models: A comprehensive survey, 2025b. URL <https://arxiv.org/abs/2411.04890>. [231](#page-4-0)

{13}------------------------------------------------

<span id="page-13-0"></span>Yiran Wu, Tianwei Yue, Shaokun Zhang, Chi Wang, and Qingyun Wu. Stateflow: Enhancing llm task-solving through state-driven workflows. *arXiv preprint arXiv:2403.11322*, 2024. [233](#page-6-0) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*, 2022. [228,](#page-1-0) [229,](#page-2-0) [237](#page-10-0) Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. Easytool: Enhancing llm-based agents with concise tool instruction. *arXiv preprint arXiv:2401.06201*, 2024. [229](#page-2-0) Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten de Rijke. Let me do it for you: Towards llm empowered recommendation via tool learning. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1796–1806, 2024. [229](#page-2-0) Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm question answering with external tools. *Advances in Neural Information Processing Systems*, 36:50117–50143, 2023. [229](#page-2-0)




