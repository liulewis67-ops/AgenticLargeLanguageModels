<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 227-240; Marker split PDF pages 1-14. -->



{0}------------------------------------------------

## 第 11 章

# 智能体如何使用工具在世界中行动

#### 你将学到的内容

本章将介绍智能体大语言模型如何借助工具连接外部世界。学完本章后，你将能够：

- 解释为何**工具**是智能体超越文本生成、在外部环境中执行任务的关键。
- 区分**直接调用**与**规划器—调用器—总结器**两类工具调用架构。
- 了解 **LangChain**、**ReAct** 等工具链与框架，说明智能体如何操作浏览器与桌面环境。
- 说明**模型上下文协议（MCP）**如何提供标准化的工具集成接口。
- 分析智能体使用外部工具时面临的可靠性、错误恢复、安全性与安全防护问题。
## 11.1 工具在智能体大语言模型中为何重要

在本书的“推理—行动—交互”分类框架中（见图 1.1），**工具连接了 LLM 的内部推理与外部环境中的实际操作**。工具既可以是简单的计算器，也可以是复杂的科学仿真引擎，甚至是完整的桌面操作环境。借助工具，LLM 能够：

- **获取最新的领域数据**：通过 API、数据库或搜索引擎访问模型参数之外的信息。
- **执行外部操作**：例如控制机器人、操作浏览器或发送消息。
- **委托专用系统处理任务**：将复杂计算或信息检索交由相应工具完成。

{1}------------------------------------------------

<span id="page-1-0"></span>

- **组合工作流**：把多个工具组织成可规划、执行和监控的连续任务。

检索是工具使用的一种特殊形式，主要用于补充 LLM 可获取的**知识**（详见第 12 章）；而一般意义上的工具使用还扩展了模型改变外部环境的**行动能力**。不过，外部工具也引入了新的故障点，并增加了安全、可靠性和性能方面的风险，某些能力还可能被滥用。

## 11.2 工具使用架构

将外部工具集成到智能体大语言模型中，常见的组织方式包括**直接调用**与**规划器—调用器—总结器**两种架构。

#### 直接调用

在直接调用模式下，LLM 生成结构化的工具调用请求（如 JSON 参数或函数调用描述），由外部控制器解析并分派给相应工具。OpenAI 的函数调用接口和 Gorilla [\[Patil et al.,](#page-12-0) [2023\]](#page-12-0) 都体现了这一思路。

#### 规划器–调用器–总结器

规划器—调用器—总结器架构在直接调用的基础上进一步分离职责：规划器负责统筹任务，调用器负责执行工具请求（也可以由另一个 LLM 或其他智能体承担），总结器则把执行结果压缩为供规划器继续决策的观察信息。这样，系统便能在保持整体任务视图的同时，把具体操作交给专用模块。

**ReAct（Reasoning and Acting）**[\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) 将推理与行动交替组织成“思考—行动—观察”的循环。它不要求智能体一次性生成完整计划，而是允许系统依据每次工具执行的结果调整后续步骤。例如，运行代码或网络检索得到的观察结果，可以为下一轮推理提供外部依据，从而提高面对错误和意外情况时的适应能力。

## 11.3 APIs 和工具链

只要外部工具提供已知的应用程序编程接口（API），LLM 就可以根据接口描述生成相应调用。一种常见的工具使用方式是将外部来源返回的信息加入模型的上下文。
{2}------------------------------------------------

<span id="page-2-0"></span>

例如，系统可以通过搜索引擎检索资料，再将结果提供给 LLM。检索方法将在第 12 章详细讨论。

智能体需要根据任务判断何时调用工具、选择哪个工具以及如何传递参数 [\[Shen et al.,](#page-12-0) [2024\]](#page-12-0)。研究表明，语言模型可以通过训练学习这些行为。**Toolformer** [\[Schick et al.,](#page-12-0) [2023\]](#page-12-0) 让模型学习选择 API、确定调用时机和参数，并将返回结果用于后续词元预测；其研究覆盖计算器、问答、搜索、翻译和日历等工具。后续工作进一步扩展了工具范围。例如，**ToolBench** [\[Qin et al.,](#page-12-0) [2023\]](#page-12-0) 整理了来自 RapidAPI 的 16,464 个 API，形成面向真实 REST 接口的评测资源。<sup>1</sup>

**EasyTool** [\[Yuan et al.,](#page-13-0) [2024\]](#page-13-0) 在 ToolBench 的基础上，尝试从工具文档中提炼结构化、统一的调用说明。**ToolAlpaca** [\[Tang et al.,](#page-12-0) [2023\]](#page-12-0) 提供了来自约 400 个 API 的 3,938 个以上示例；**ToolQA** [\[Zhuang et al.,](#page-13-0) [2023\]](#page-13-0) 面向需要调用外部工具的问答任务。**Gorilla** [\[Patil et al.,](#page-12-0) [2023\]](#page-12-0) 则通过微调 LLaMA 来生成 API 调用，并提出了 APIBench 基准。

工具选择、参数组织和结果总结都不是简单的任务。[Zhao et al.](#page-13-0) [\[2024\]](#page-13-0) 与 [Shen et al.](#page-12-0) [\[2024\]](#page-12-0) 研究了如何借助工具学习改进推荐任务，并分别使用不同的 LLM 承担（1）规划推理、（2）工具请求生成和（3）结果总结。图 [11.1](#page-4-0) 展示了这一多模型架构：规划器、调用器和总结器分别针对自身职责进行微调。相关实验使用了 Claude-2、ChatGPT、GPT-4、Tool-LLaMA 等模型，以及 ReAct [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) 和 DFSDT [\[Qin et al.,](#page-12-0) [2023\]](#page-12-0) 等推理策略。其他工具编排方法还包括 [Ocker et al.](#page-12-0) [\[2024\]](#page-12-0) 的研究。

### 11.3.1 LangChain 示例

**LangChain** [\[Chase,](#page-11-0) [2022\]](#page-11-0) 是常用的 LLM 应用开发框架，提供构建智能体及连接工具的抽象接口。代码清单 [11.1](#page-3-0) 演示了如何在 Python 中定义工具，并让智能体把需要数值判断的问题交给专用工具处理。

### 11.3.2 计算机和浏览器工具

浏览器和完整的计算机系统也可以作为工具，供 LLM 通过接口操作。要让智能体完成这类任务，系统通常需要先识别图形用户界面（GUI）中的可交互元素，再把自然语言指令转换为具体操作。**OmniParser V2** [\[Lu et al.,](#page-11-0) [2024\]](#page-11-0) 是这一方向的例子，它使用视觉方法解析屏幕，识别并标注按钮

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

*代码清单 11.1：使用 LangChain 定义工具并让智能体调用。*
{4}------------------------------------------------

<span id="page-4-0"></span>![](assets/_page_4_Diagram_2.jpeg)

图 11.1：包含规划器、调用器与总结器的多 LLM 智能体架构 [\[Shen et al.,](#page-12-0) [2024\]](#page-12-0)

或图标。通过把屏幕截图转换为结构化表示，OmniParser 帮助视觉—语言模型将操作决策关联到具体界面元素，从而改善操作定位的准确性。

另一条研究路线关注如何让 LLM 调用系统命令，或在浏览器与操作系统界面中导航。Anthropic 的 Computer Use、OpenAI 的 Operator 及其后续产品 GPT Agent，都是这一方向的代表。它们把启动应用、点击按钮、填写表单等操作封装为模型可调用的工具，让系统得以将高层自然语言指令分解为可执行步骤，进而完成浏览网页、管理文件等较复杂的任务。

开源项目 **browser-use** [\[Müller and Žunič, 2024\]](#page-11-0) 展示了另一种实现方式：它为 LLM 提供可保持会话状态的浏览器操作能力，并处理 Cookie、会话令牌及动态网页内容，支持跨页面操作、多步骤表单填写和跨站查询。

关于浏览器和计算机环境中的智能体，可参阅 [Wang et al.](#page-12-0) [\[2025b\]](#page-12-0) 的综述，其中讨论了 GUI 解析、工具调用与人工监督等设计模式。总体来看，可靠的计算机操作工具能够把文本意图转化为具体行动。

{5}------------------------------------------------

<span id="page-5-0"></span>

这种能力拓展了语言驱动的智能体所能完成的任务范围。

## 11.4 模型上下文协议与上下文管理

**模型上下文协议（Model Context Protocol，MCP）** 是由 Anthropic 提出的开放协议，用于将 AI 应用连接到外部数据、工具和工作流 [\[Hou et al.,](#page-11-0) [2026\]](#page-11-0)。它有时被比作 AI 应用的“USB-C 接口”：通过统一协议，宿主应用可以接入本地文件、数据库、搜索引擎和其他外部能力。

MCP 采用客户端—服务器架构。**宿主（Host）**通常是 AI 应用，负责管理一个或多个 MCP 客户端；每个客户端与相应的 MCP 服务器建立连接。服务器向宿主提供工具、资源和提示等能力。本地服务器常通过标准输入输出（stdio）通信，远程服务器则可以使用 Streamable HTTP。

协议可以分为两层：**数据层**规定基于 JSON-RPC 的消息与操作，包括能力协商、版本协商，以及工具、资源、提示和通知等机制；**传输层**负责连接建立、消息传输及相关授权流程。这种分层使不同外部系统能够通过统一接口接入 AI 应用，而具体的上下文组织仍由宿主负责。

在需要持续调用工具的长任务中，有限的上下文窗口是重要瓶颈。若将全部对话和工具返回结果原样累积，输入可能很快超出窗口限制，模型也可能丢失任务目标或关键中间信息。因此，系统需要明确决定：**哪些信息保留原文，哪些信息摘要压缩，哪些信息可以舍弃**。

MCP 解决的是外部能力的标准化接入，并不自动解决智能体的长期状态管理问题。应用还需要自行设计任务状态与上下文维护机制，常见做法包括：

- **摘要压缩**：定期总结工具输出和历史对话，在保留关键事实的同时减少上下文占用。
- **显式状态管理**：维护任务计划、关键中间结果和当前进度等结构化状态或备忘录。
- **按需检索**：不将全部历史记录塞回上下文，而是只检索与当前步骤最相关的信息。

{6}------------------------------------------------

<span id="page-6-0"></span>

**示例：上下文管理。** 代码清单 [11.2](#page-7-0) 和 [11.3](#page-8-0) 展示了一个简化的上下文摘要机制：当历史长度接近设定上限时，系统压缩先前交互，以便保留任务所需信息。该示例使用 LangChain 演示应用层的状态管理思想，**并非 MCP 协议本身的完整实现**。<sup>2</sup> 

## 11.5 稳健性、错误处理与恢复

工具执行发生在模型之外，因此会引入新的故障类型：API 可能暂时不可用，返回的数据可能不符合预期，模型也可能制定错误计划。可靠的智能体系统必须能够识别这些情况，并采取合理的恢复措施。

**反应式错误处理。** 一种基本策略是将工具调用失败的信息作为新的观察结果反馈给 LLM，使它据此调整参数、改用其他工具，或在无法恢复时告知用户。这种“观察—反思—重试”循环构成了智能体自我恢复的基础 [\[Sun et al.,](#page-12-0) [2024\]](#page-12-0)。它与 ReAct 的循环形式相似，但重点在于异常处理。

**自我纠正与状态感知。** 更进一步的系统（如 Healer）尝试实时修复运行时错误 [\[Wu et al.,](#page-13-0) [2024\]](#page-13-0)。这类方案让 LLM 分析故障上下文和程序状态，再生成并执行修复代码，而不仅仅是重试原操作。

**人在回路（Human-in-the-loop）。** 对高风险任务，系统可以在不可逆操作前暂停，并请求用户确认，例如执行金融交易或发送重要邮件。人工参与有助于引入领域判断、发现模型疏漏并改善安全性，但也可能带来人为错误。<sup>3</sup> 相关协同设计见 [Mozannar et al.](#page-11-0) [\[2025\]](#page-11-0)。

<sup>2</sup>示例使用 LangChain；若要实现 MCP 连接，可参考专门的 SDK，例如 [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)。
  
 <sup>3</sup>人工参与同样可能引入人为错误。

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

*代码清单 11.2：上下文管理的简化实现（第一部分）。*
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

*代码清单 11.3：上下文管理的简化实现（第二部分）。*
{9}------------------------------------------------

<span id="page-9-0"></span>![](assets/_page_9_Figure_2.jpeg) 
  
  
  
 图 11.2：Rainbow Teaming 生成的对抗性提示 [\[Samvelyan et al.,](#page-12-0) [2024\]](#page-12-0) 

## 11.6 安全与防护

智能体能够调用外部工具后，安全问题便不再局限于回答质量或一般的执行错误，还包括越权操作、敏感信息泄露与恶意使用等风险。例如，系统提示或工具返回的数据如果处理不当，可能暴露不应公开的信息。

越狱研究主要考察攻击者能否通过特制提示绕过模型安全限制；而具备工具权限的智能体还可能将不当指令转化为多步骤外部操作，风险因此更加复杂。**AgentHarm** [Andriushchenko et al.](#page-11-0) [\[2024\]](#page-11-0) 提供了包含 110 项有害智能体任务的评测集，覆盖欺诈、网络犯罪和骚扰等类别。论文指出，模型在单轮问答中拒绝某些请求，并不意味着以工具调用形式表达的相关任务同样会被可靠拦截。

**Rainbow Teaming** [\[Samvelyan et al.,](#page-12-0) [2024\]](#page-12-0) 探索了自动生成多样化对抗性提示的方法，以缓解安全测试样本覆盖不足的问题。它将提示生成建模为**质量—多样性优化**，借鉴开放式搜索 [\[Hughes et al.,](#page-11-0) [2024\]](#page-11-0)，并利用 MAP-Elites [\[Mouret and Clune,](#page-11-0) [2015\]](#page-11-0) 维护具有不同特征的候选提示集合。相关示意见图 [11.2](#page-9-0) 和 [11.3](#page-10-0)。

{10}------------------------------------------------

<span id="page-10-0"></span>![](assets/_page_10_Diagram_2.jpeg) 
  
 图 11.3：Rainbow Teaming 的质量—多样性变异架构 [\[Samvelyan](#page-12-0) [et al.,](#page-12-0) [2024\]](#page-12-0) 
 
## 11.7 结语

工具是让语言模型从静态问答走向实际行动的关键：它们使模型能够获取最新信息、执行外部操作，并把复杂任务委托给专用系统。但构建可靠的智能体，远不止把工具接口接到 LLM 上；还需要考虑任务从规划到执行、反馈和恢复的完整生命周期。本章强调四项工程原则：

1. **选择合适的架构。** ReAct [\[Yao et al.,](#page-13-0) [2022\]](#page-13-0) 等循环式方法让模型依据观察结果不断调整行动，比一次性生成并执行固定计划更便于处理意外情况。
2. **维护上下文与状态。** 面对长期、多步骤任务，应通过摘要、显式状态和检索控制上下文规模；MCP 负责工具连接，不能代替应用层状态管理。
3. **将失败纳入设计。** 外部 API 可能出错或不可用，系统应记录异常、分析原因，并在适当情况下安全重试或停止。
4. **保留必要的人工监督。** 涉及资金、敏感数据或不可逆操作时，应有明确的审批节点和可追溯的执行记录。

{11}------------------------------------------------

#### <span id="page-11-0"></span>动手试试！

如果想动手实践，可以在 Google Colab 中打开原作者提供的 [LLM 工具使用教程](https://colab.research.google.com/github/Agentic-LLMs/Agentic-AI-tutorials/blob/main/tools_tutorial.ipynb)。 
 
 进一步阅读可参考工具使用综述 [Shen](#page-12-0) [\[2024\]](#page-12-0)、[Patil et al.](#page-12-0) [\[2023\]](#page-12-0) 和 [Qin et al.](#page-12-0) [\[2023\]](#page-12-0)。GUI 解析与浏览器控制方面可参阅 Lu et al. [2024] 和 Müller 与 Žunič [2024]；安全评测与对抗提示研究则可参考 [Wang et al.](#page-12-0) [\[2025a\]](#page-12-0)。

## 参考文献

- Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, et al. Agentharm: A benchmark for measuring harmfulness of llm agents. *arXiv preprint arXiv:2410.09024*, 2024. [236](#page-9-0) 

- Harrison Chase. LangChain, October 2022. URL [https://github.com/](https://github.com/langchain-ai/langchain) [langchain-ai/langchain](https://github.com/langchain-ai/langchain). [229](#page-2-0) 

- Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. *ACM Trans. Softw. Eng. Methodol.*, 35(10), September 2026. ISSN 1049-331X. doi: 10.1145/ 3796519. URL <https://doi.org/10.1145/3796519>. [232](#page-5-0) 

- Edward Hughes, Michael Dennis, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, and Tim Rocktaschel. Open-endedness is essential for artificial superhuman intelligence. *arXiv preprint arXiv:2406.04268*, 2024. [236](#page-9-0) 

- Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024. URL <https://arxiv.org/abs/2408.00203>. [229,](#page-2-0) 238 

- Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. *arXiv preprint arXiv:1504.04909*, 2015. [236](#page-9-0) 

- Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, et al. Magentic-ui: Towards human-in-the-loop agentic systems. *arXiv preprint arXiv:2507.22358*, 2025. [233](#page-6-0) 

- Magnus Muller and Gregor ¨ Zuni ˇ c. Browser use: Enable ai to control your browser, ˇ 2024. URL <https://github.com/browser-use/browser-use>. [231,](#page-4-0) 238

{12}------------------------------------------------

<span id="page-12-0"></span>

- Felix Ocker, Daniel Tanneberg, Julian Eggert, and Michael Gienger. Tulip agent– enabling llm-based agents to solve tasks using large tool libraries. *arXiv preprint arXiv:2407.21778*, 2024. [229](#page-2-0) 

- Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2305.15334) [abs/2305.15334](https://arxiv.org/abs/2305.15334). [228,](#page-1-0) [229,](#page-2-0) [238](#page-11-0) 

- Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. *arXiv preprint arXiv:2307.16789*, 2023. [229,](#page-2-0) [238](#page-11-0) 

- Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, et al. Rainbow teaming: Open-ended generation of diverse adversarial prompts. *arXiv preprint arXiv:2402.16822*, 2024. [236,](#page-9-0) [237](#page-10-0) 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, 36:68539–68551, 2023. [229](#page-2-0) 

- Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei Huang. Small llms are weak tool learners: A multi-llm agent. *arXiv preprint arXiv:2401.07324*, 2024. [229,](#page-2-0) [231](#page-4-0) 

- Zhuocheng Shen. Llm with tools: A survey. *arXiv preprint arXiv:2409.18807*, 2024. [238](#page-11-0) 

- Zhensu Sun, Haotian Zhu, Bowen Xu, Xiaoning Du, Li Li, and David Lo. Llm as runtime error handler: A promising pathway to adaptive self-healing of software systems, 2024. URL <https://arxiv.org/abs/2408.01055>. [233](#page-6-0) 

- Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. *arXiv preprint arXiv:2306.05301*, 2023. [229](#page-2-0) 

- Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, et al. A comprehensive survey in llm (-agent) full stack safety: Data, training and deployment. *arXiv preprint arXiv:2504.15585*, 2025a. [238](#page-11-0) 

- Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, Bin Wang, Chuhan Wu, Yasheng Wang, Ruiming Tang, and Jianye Hao. Gui agents with foundation models: A comprehensive survey, 2025b. URL <https://arxiv.org/abs/2411.04890>. [231](#page-4-0)

{13}------------------------------------------------

<span id="page-13-0"></span>

- Yiran Wu, Tianwei Yue, Shaokun Zhang, Chi Wang, and Qingyun Wu. Stateflow: Enhancing llm task-solving through state-driven workflows. *arXiv preprint arXiv:2403.11322*, 2024. [233](#page-6-0) 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*, 2022. [228,](#page-1-0) [229,](#page-2-0) [237](#page-10-0) 

- Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. Easytool: Enhancing llm-based agents with concise tool instruction. *arXiv preprint arXiv:2401.06201*, 2024. [229](#page-2-0) 

- Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten de Rijke. Let me do it for you: Towards llm empowered recommendation via tool learning. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1796–1806, 2024. [229](#page-2-0) 

- Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm question answering with external tools. *Advances in Neural Information Processing Systems*, 36:50117–50143, 2023. [229](#page-2-0)

