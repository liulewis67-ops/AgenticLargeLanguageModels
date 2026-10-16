<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 277-282; Marker split PDF pages 1-6. -->



{0}------------------------------------------------

# <span id="page-0-0"></span>Chapter 14

# Case Study 3: Building a Multi-Agent Diagnosis Framework

#### What You Will Learn

In this case study you will gain hands-on experience building a multi-agent system for a high-stakes application domain. Specifically, you will be able to:

- Implement a *hospital diagnostic workflow* with planner and executor agents communicating through a shared context.
- Integrate trained machine learning *classifiers as tools* within an agentic pipeline.
- Replace rule-based planners with a language model and *evaluate* the impact on routing and diagnostic accuracy.
- Assess system performance using classification and routing metrics, separating planner errors from executor errors.
- Critically reflect on the *limitations, risks, and ethical* considerations of deploying agentic systems in medical settings.

#### 14.1 Scenario Overview

This case study focuses on building and extending a simplified multi-agent hospital simulation using a shared context inspired by the Model Context Protocol (MCP, Chapter 11) [\[Hou et al.,](#page-5-0) [2026\]](#page-5-0). The case study is designed to provide hands-on experience with agentic architectures in a high-risk domain, highlighting coordination, planning, evaluation, and safety considerations.

You will simulate a diagnostic workflow in a hospital setting. Patient cases are processed by agents that interact exclusively through a shared context:

- *Planner agents* decide which diagnostic pathway(s) should be followed.

{1}------------------------------------------------

- *Executor agents* apply trained ML classifiers to produce diagnoses.
- *Shared context* stores cases, tasks, intermediate decisions and results.

The simulation runs in discrete steps and logs how cases move through the system. As with the earlier case studies, the goal is *not* to build a clinically valid system. The goal is for you to understand how agentic architectures behave in a high-risk domain: how agents coordinate through shared state, how a language model performs when placed in a triage role, and where the limitations and safety concerns of such a system arise.<sup>1</sup>

#### 14.2 Environment Setup

As in the earlier case studies, we recommend working inside a dedicated virtual environment using Python 3.10 or higher (see Listing 14.1 for setup instructions).

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Listing 14.1: Setting Up the Case Study 3 Environment

#### 🚀 View on GitHub!

The baseline code, data files, and trained models are available in the [Assignment 3 repository](https://github.com/Agentic-LLMs/Agentic-AI-tutorials/tree/main/case3).

#### 14.3 Implementation Steps

##### 14.3.1 Step 0: Training Diagnostic Models

You are provided with a script, `train_models.py`, that trains simple classifiers (random forests) on three public datasets:

- **Diabetes:** the Pima Indians Diabetes dataset.
- **Breast cancer:** the Wisconsin breast cancer dataset, available through `sklearn.datasets`.

<sup>1</sup>This case study is partly based on a tutorial by Ayushman Pranav, *Modeling Hospital Intelligence with MCP: A Multi-Agent Diagnostic Simulation*, available at [https://medium.com/@AyushmanPranav/](https://medium.com/@AyushmanPranav/modeling-hospital-intelligence-with-mcp-a-multi-agent-diagnostic-simulation-afeff0dd885d) [modeling-hospital-intelligence-with-mcp-a-multi-agent-diagnostic-simulation-afeff0dd885d](https://medium.com/@AyushmanPranav/modeling-hospital-intelligence-with-mcp-a-multi-agent-diagnostic-simulation-afeff0dd885d).

{2}------------------------------------------------

- **Heart disease:** the UCI heart disease dataset.

Execute the provided training script and store the resulting models for later use; the executor agents in Steps 1 and 2 load these trained models as tools rather than retraining them on every run.

**Deliverables:** trained model files (or a clear, reproducible script to regenerate them) for all three diseases.

##### 14.3.2 Step 1: Understanding and Running the Baseline

The baseline implementation, `basic_mcp.py`, uses three rule-based planner agents, each hard-coded to a specific disease. Before running anything, read the code carefully. It implements three components that recur throughout this case study:

- Shared context (MCP-like layer). A central data structure that stores incoming patient cases, assigned diagnostic tasks, and diagnostic results. All agents interact *only* through this shared context, never directly with one another, mirroring the shared-state coordination pattern discussed for the Model Context Protocol in Section 11.
- Planner agents. Each rule-based planner reads unassigned cases, decides (using a fixed rule) whether its disease domain applies, and creates a task in the shared context.
- Executor agents. Each executor reads tasks assigned to its specialty, runs the corresponding trained ML classifier from Step 0, produces a prediction, and writes the result back into the shared context.

Run the baseline simulation and observe how a single patient case flows through the system: from arrival, through planning and task creation, to execution and logging of the result.

**Analysis questions:** assess how realistic this baseline setup is. How well is performance evaluated? What assumptions are implicit in the way patients and diagnoses are handled?

##### 14.3.3 Step 2: Replacing the Rule-Based Planners with an LLM Agent

In the second part, `llm_mcp.py`, the three rule-based planners are replaced by a single LLM-based planner agent. This is a substantially harder task for the planner: each patient case now combines features drawn from *all three* datasets into one record, and is converted into a textual description using the provided make case text function, which you are not permitted to modify. The LLM planner must read this combined, textual case description and decide, without hard-coded rules, which diagnostic pathway or pathways apply.

Your tasks are to:

{3}------------------------------------------------

- implement a function that calls a real LLM (e.g., via Ollama for a local model, or the OpenAI or Gemini APIs) to obtain the planner's decision for a given case;
- implement the step() method of the LLMPlannerAgent, so that it selects one or more diagnostic pathways together with a primary diagnosis for each case;
- handle malformed or unparseable LLM outputs gracefully, rather than letting the simulation crash on a single bad response; and
- log every planner decision, together with the case it was made for, so that planner behavior can be analyzed after the run.

Because the planner now has to infer relevance from unstructured text rather than from a fixed rule, this step should give you direct experience with the reliability problems that arise when an LLM is placed in a decision-making role: parsing failures, inconsistent formatting, and disagreement between the LLM's routing decision and the dataset's ground-truth label.

**Deliverables:** a working `LLMPlannerAgent`, a short note on which LLM you used and how you accessed it (local vs. API), and the logged planner decisions for a full simulation run.

#### 14.4 Evaluation

You must evaluate both components of the system, and keep the two evaluations separate:

- Model (executor) performance: assess executor predictions using confusion matrices and standard classification metrics (accuracy, precision, recall, F1) for each disease.
- Planner performance: evaluate the LLM's triage quality using precision and recall of routing decisions (did the case get sent to the right disease pathway or pathways?) and primary-diagnosis accuracy (did the planner correctly identify the main concern?).

In your report, clearly separate errors due to incorrect routing (a planner error) from errors due to incorrect model predictions (an executor error). A case can fail for either reason independently: a well-routed case can still be misclassified by the executor, and a correctly-classifiable case can be missed entirely because the planner never routed it to the right pathway. Reporting only an end-to-end accuracy number would hide this distinction and is not sufficient for this case study.

#### 14.5 Reflection Questions

The following questions are intended to guide a short written report on your implementation. You do not need to write a narrative essay; concise, direct answers are sufficient, and you should be prepared to elaborate on your reasoning verbally.

{4}------------------------------------------------

#### A. Baseline architecture

- Describe, in at most five sentences, the flow of a single patient case through the baseline system.
- Each baseline case is sampled from a dataset and therefore belongs to exactly one disease domain by construction. What does this imply about how the rule-based planner "decides," and is this a realistic model of triage?
- The classifiers in Step 0 are trained and evaluated on the same data distribution used to generate simulation cases. Why is this methodologically problematic, and what would a fairer evaluation protocol look like?

#### B. LLM planner design

- Which LLM did you use, and how did you access it (local model, API)? Briefly justify the choice.
- The make case text function presents a patient as a combination of features drawn from three separate, unrelated datasets. Why is this medically unrealistic, and does that matter for the purposes of this exercise?
- Describe one concrete design choice in your step() implementation—for example, how you handle parsing failures, abstentions, or cases routed to multiple diseases—and explain why you made it.

#### C. Results

- Report triage precision and recall for each disease, along with confusion matrices for the executor models. What do these numbers tell you about how well the overall system works?
- The LLM planner can route a single case to multiple diseases simultaneously. Give one example scenario where this is beneficial, and one where it could be harmful.

#### D. Reflection

- Identify one safety risk that is specific to using an LLM as a planner in a medical triage context, and describe how it could be mitigated.
- Is it fair to evaluate triage performance by comparing the LLM's routing decisions against the ground-truth labels of the original, single-disease datasets? Explain any reservations you have.

{5}------------------------------------------------

#### <span id="page-5-0"></span>Optional Extensions

As an extra, you may extend the system by making patient sampling more realistic (e.g., mixed or comorbid cases sampled with more principled priors). You can also add another disease from the UCI repository, or introduce an evaluator or verifier agent that checks the planner's decisions before they reach the executors. Another idea is simulating human-in-the-loop decision making, or incorporating uncertainty thresholds and abstention mechanisms so that the planner can decline to route a case when it is not confident. Finally, you can integrate retrieval from external knowledge sources (Chapter 12) so the planner can ground its routing decisions in clinical reference material rather than relying purely on parametric knowledge.

#### Notes

This assignment is educational and non-clinical. Do not claim medical validity for any system you build here. Emphasis is placed on clarity, reproducibility, and critical reflection rather than raw performance. A system that routes conservatively, abstains when uncertain, and is transparent about its limitations is, in this setting, a better outcome than one that maximizes accuracy while hiding its failure modes.

#### 14.6 Concluding Remarks

This case study brought together several threads from earlier parts of the book: agent roles and coordination (Part III), tool use and the Model Context Protocol (Chapter 11), and the evaluation methodology needed to separate planning errors from execution errors in a compound agentic system. By first building a rule-based baseline and then replacing its planners with a single LLM-based planner, you have seen directly how much harder planning becomes once it depends on an LLM's interpretation of unstructured, textual case descriptions rather than a fixed rule.

The medical framing was chosen deliberately: it is a domain where the cost of an overconfident, unexplainable error is high, and where questions of abstention, human oversight, and auditability are core design requirements. The same coordination pattern applies well beyond healthcare, to any agentic system where a language model is responsible for routing decisions that other, more specialized components then act on.

## Bibliography

Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. *ACM Trans. Softw. Eng. Methodol.*, 35(10), September 2026. ISSN 1049-331X. doi: 10.1145/3796519. URL <https://doi.org/10.1145/3796519>. [277](#page-0-0)