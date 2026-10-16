<!-- Source: AgenticLargeLanguageModels.pdf; physical PDF pages 111-120; Marker split PDF pages 1-10. -->



{0}------------------------------------------------

## Chapter 5

# Case Study 1: Building LLMs from Scratch

#### What You Will Learn

In this case study, you will learn how to train your own model. By the end, you will be able to:

- Explain the end-to-end *training pipeline* of a ChatGPT-style LLM and distinguish the purpose and mechanics of tokenization, pretraining, mid-training, and supervised fine-tuning.
- Configure and train a Byte Pair Encoding (BPE) *tokenizer* and explain the trade-offs involved in choosing its vocabulary size.
- Describe the *Transformer architecture* in mechanistic terms, including attention heads, layer depth, and embedding dimensions, and investigate how these hyperparameters interact with model capacity and compute budget.
- Evaluate a trained language model on standard *benchmarks* (ARC-Easy, ARC-Challenge, and GSM-8K) and critically interpret the results, including failure modes.
- Critically reflect on the ethical, environmental, and accessibility dimensions of LLM training, including cost as a barrier to entry and data curation as a value-laden choice.
- Find and use information about open-access LLMs, including model cards, training reports, training code, and other types of *Hub entries*.
## 5.1 Building a Chatbot

We will work through a practical case study covering the full lifecycle of a modern ChatGPT-style language model. The project uses *nanochat*<sup>1</sup>, a minimal, single-codebase, full-stack implementation of an instruction-following LLM released by Andrej Karpathy in April

<sup>1</sup><https://github.com/karpathy/nanochat>

{1}------------------------------------------------

<span id="page-1-0"></span>2025. Unlike his earlier *nanoGPT*, which covered only pretraining, nanochat implements the complete training pipeline: a Byte Pair Encoding (BPE) tokenizer, pretraining on web text followed by supervised fine-tuning (SFT) on benchmark datasets, an inference server, and a chat web interface.

The code is written mostly in Python with few dependencies, making it easy to use and extend. The codebase contains roughly 8,000 lines of Python and a small amount of Rust.<sup>2</sup>

A striking result is that nanochat can train a GPT-2-capable LLM for about $48—roughly two hours on a node with eight H100 GPUs. For comparison, training GPT-2 cost about $43,000 in 2019 [\[Radford et al.,](#page-9-0) [2019\]](#page-9-0). Although this hardware is beyond the reach of many student projects, the experiment can be scaled down for a hands-on course assignment. Every gradient update and design decision is visible, from raw bytes to a working chat assistant. The case study walks through four tasks in the nanochat pipeline.

Course GPU budgets are limited, so you will train a *small* model (depth 4; use depth 2 only if necessary; see Section [5.3](#page-2-0)). You will then use theory and published nanochat benchmarks to reflect on what may happen at larger scales.

## 5.2 Background: Karpathy's nanochat

This case study provides hands-on experience with each stage of a modern LLM training pipeline, from input text to a deployed chat assistant. It connects the concepts from Chapters 2–4 to concrete, debuggable code.

Out of the box, nanochat is configured to train a compute-optimal model using a single complexity setting: --depth, which specifies the number of Transformer layers. Other hyperparameters—including embedding width, number of attention heads, learning-rate schedule, training horizon, and weight decay—are derived automatically using near-optimal settings. This keeps the configuration space manageable while preserving the educational value of understanding each parameter.

The training pipeline follows familiar stages, briefly described below.

**Pretraining.** The model is trained on approximately 6 billion tokens from CLIMBMix [\[Diao et al.,](#page-8-0) [2026\]](#page-8-0), a data-mixture corpus built through iterative, clustering-based bootstrapping. A depth-24 model (about 1.4B parameters) reaches a validation loss of 2.41 (0.717 bits per byte) in roughly four hours on a compute cluster with four H100 GPUs.

**Supervised fine-tuning (SFT).** Starting from the pretrained checkpoint, the model is fine-tuned on a mixture of instruction-following and reasoning data: SmolTalk [\[Ben Allal et al.,](#page-8-0) [2024\]](#page-8-0) (460K general-purpose conversations), MMLU (multiple-choice knowledge questions, 3 epochs), GSM-8K (grade-school mathematics, 4 epochs), and identity

<sup>2</sup> If that number seems large, note that comparable proprietary systems contain millions of lines of code. Nanochat is designed as a small, self-contained version.

{2}------------------------------------------------

<span id="page-2-0"></span>conversations and synthetic spelling tasks—about 1.1 million training examples in total. Cross-entropy loss is computed *only on assistant tokens*, not on user prompts, following the InstructGPT training procedure [\[Ouyang et al.,](#page-9-0) [2022\]](#page-9-0).

Before each training stage, a BPE tokenizer is trained on 2 billion characters using a Rust implementation. It produces a vocabulary of 32,768 tokens in under a minute.

A web inference server and lightweight JavaScript chat interface are included, so you can interact with the trained model immediately.

**Supplementary resources.** Two freely available courses offer complementary perspectives on this case study. The *Hugging Face LLM Course* [\[Hugging Face,](#page-9-0) [2022\]](#page-9-0) introduces the Transformers, Datasets, Tokenizers, and Accelerate libraries, as well as the Hugging Face Hub, where many open-weight models are distributed and evaluated. Mathieu Labonne's *LLM Course* [\[Labonne,](#page-9-0) [2023\]](#page-9-0) provides a practitioner-oriented roadmap covering pretraining data curation, parameter-efficient fine-tuning (LoRA and QLoRA), preference alignment (DPO, PPO, and GRPO), quantization (GGUF and GPTQ), and production inference serving (vLLM and TGI). These resources place nanochat in the broader open LLM ecosystem and are referenced in Tasks 3 and 4. You may also consult other sources; cite them in your report where appropriate.

## 5.3 Experiment Preparation

**Repository.** Clone the course repository and read its README carefully before writing any code.

Listing 5.1: Cloning the nanochat repository

    git clone git@github.com:karpathy/nanochat.git

**Compute.** No dedicated compute resource is required. You may use a laptop with a GPU or a more powerful computer. Each experiment is designed to take at most about eight hours. For pretraining, use --depth 4; --depth 2 is acceptable only if your hardware cannot support the larger model. Do *not* attempt --depth 12 or greater unless you have substantial compute resources.

**Logging.** nanochat can log training loss, validation loss, and bits per byte to the terminal at configurable intervals. It also provides tools for reporting and saving information during a run. Enable structured logging to a .jsonl file from the start; retrofitting logging after a run is painful. You may experiment with different logging formats. The chapter recommends trying a service such as Weights & Biases before training full models.

{3}------------------------------------------------

## 5.4 Task 0: Model Hub Warm-Up

**Description.** Before starting the tasks, explore common parameter and training settings across different models by navigating the Hugging Face Hub. For each of the three model IDs in Table 5.1, find and record:

1. The number of training tokens or steps.
2. The model architecture, including the number of layers, hidden dimension, attention heads, and trainable parameters.
3. The training data sources, if available.

| Model ID | Organization |
|---|---|
| allenai/OLMo-1B | Allen Institute for AI |
| HuggingFaceTB/SmolLM2-360M | Hugging Face |
| Qwen/Qwen2.5-0.5B | Alibaba Cloud |

Table 5.1: Model IDs for the Hub warm-up exercise

Pay attention to where you find each piece of information. Some fields appear on the model card, others require inspecting config.json under the Files tab, and some are available only in the accompanying paper. Include a short table with your findings in the report and cite the source for each field.

## 5.5 Task 1: Tokenization

#### 5.5.1 What to Do

Train a BPE tokenizer using nanochat's Rust tokenizer trainer on a 500-million-character sample of CLIMBMix. Run two experiments, with vocabulary sizes of *8,192* and *32,768* tokens.

#### 5.5.2 Deliverables

- A short script that reproduces your tokenizer training run.
- A report section (at most one page) answering the following questions with evidence from your tokenizer outputs:
  1. What is BPE, and how does it differ from word-level and character-level tokenization? Draw a merge tree for a concrete example from the data or one of your own. Do not reuse an existing example, and include the merge counts.

<sup>3</sup><https://huggingface.co/models>

{4}------------------------------------------------

<span id="page-4-0"></span>2. How does vocabulary size affect the *compression ratio* (tokens per character) and *sequence length* for a fixed text sample? Show measurements from both tokenizers on English text.
3. Why does vocabulary size matter for the downstream language model? Consider the size of the embedding matrix, the softmax denominator, and the difficulty of learning reliable representations for rare tokens.
4. What artifacts or failure cases do you observe? How does your tokenizer handle numbers, source code, or non-English text?

Pay attention to the correctness of your BPE explanation, the quality of your empirical evidence, the depth of your analysis of vocabulary trade-offs, and your discussion of failure cases.

## 5.6 Task 2: Pretraining

#### 5.6.1 What to Do

Pretrain a Transformer on CLIMBMix using nanochat with --depth 4, or --depth 2 if your computer is not fast enough. Log training loss, validation loss, and bits per byte (bpb) at regular intervals.

#### 5.6.2 Deliverables

- Training and validation bpb curves, presented as plots in your report. Use bpb rather than raw loss.
- A saved model checkpoint.
- A report section (approximately two pages) covering:
  1. **The depth parameter:** Explain what --depth controls and how the codebase derives the other hyperparameters from it, including embedding width, number of attention heads, learning-rate schedule, and training horizon. For your chosen depth, report the number of layers and attention heads, embedding dimension, total trainable parameters, and total training tokens. Why is a single-dial design useful for this codebase, and what are its limitations compared with tuning each hyperparameter independently?
  2. **Scaling laws:** Use the Chinchilla framework [\[Hoffmann et al.,](#page-9-0) [2022\]](#page-9-0) to calculate the compute-optimal number of training tokens for your model's parameter count. How does nanochat's default training horizon compare? What does it mean for a model to be *under-trained* or *over-trained*?
  3. **Loss analysis:** Plot bpb over training and explain the curve. When does it flatten? What does the gap between training and validation curves tell you?

{5}------------------------------------------------

- 4. **Qualitative inspection:** Sample five completions from your pretrained model without a system prompt (raw next-token prediction). What can the model do at this stage, and why?
  
Pay attention to the correctness of your depth-parameter and auto-configuration explanations, the accuracy of your architecture description, the scaling-law calculations, the quality of your loss analysis, and the insightfulness of your qualitative assessment.

## 5.7 Task 3: Mid-Training and Supervised Fine-Tuning

#### 5.7.1 What to Do

nanochat performs two training stages: pretraining and SFT. The book also discusses how additional mid-training can improve a model. In this task, you will add a mid-training stage by modifying scripts/chat_sft.py. The goal is to separate *reasoning adaptation* (mid-training) from *instruction following* (SFT), then measure the effect.

**Stage 1 — Mid-training.** Modify the train_tasks list in scripts/chat_sft.py so it contains only reasoning and knowledge tasks: MMLU and GSM-8K. Remove all conversational and spelling data. Run chat_sft.py from the pretrained base checkpoint and save the resulting checkpoint.

**Stage 2 — SFT.** Modify scripts/chat_sft.py again so train_tasks contains only SmolTalk. Make a small change so the script loads your Stage 1 checkpoint instead of the base pretrained model. Inspect the load_model call in chat_sft.py for guidance.<sup>4</sup> Run chat_sft.py from the Stage 1 checkpoint to perform SFT after mid-training.

After each stage, evaluate the model on ARC-Easy, ARC-Challenge, and GSM-8K using scripts/chat_eval.py.

**Sanity check.** Before Stage 1, inspect the data by printing the first few rows of the MMLU and SmolTalk datasets. Determine how many training examples each dataset contributes and what a single example looks like.

#### 5.7.2 Deliverables

- Two checkpoints, one for each stage (mid-training and SFT).
- A table of ARC-Easy, ARC-Challenge, and GSM-8K evaluation scores after each stage.
- A report section (approximately three pages) covering:

<sup>4</sup> Hint: Read nanochat/checkpoint_manager.py to understand how checkpoints are located.

{6}------------------------------------------------

<span id="page-6-0"></span>1. **Implementation:** Describe the code changes that allow training in stages and list the hyperparameters used in your experiments.
2. **Loss masking:** Why is SFT loss computed only on assistant tokens? Give a concrete token-level example showing the loss with and without masking.
3. **Dataset analysis:** For each stage, describe the dataset, its approximate size, and why it was chosen. What values or potential biases are embedded in the dataset splits, and how might they shape model behavior?
4. **Benchmark scores:** Report scores after each stage in a table. What changed between Stages 1 and 2, and why? Compare both with the pretrained model so your analysis covers all three stages. Note any surprising results.
5. **Parameter-efficient fine-tuning:** nanochat uses full fine-tuning, updating every parameter during SFT. Explain how LoRA (Low-Rank Adaptation) and its quantized variant, QLoRA, differ from full fine-tuning. What are the trade-offs in compute, memory, and model quality? Would you recommend PEFT at the model depths used here?

Pay attention to the clarity of your implementation description, correctness of the loss-masking explanation, depth of your dataset analysis, accuracy and completeness of the benchmark results and interpretation, and conceptual accuracy of the LoRA/QLoRA comparison.

## 5.8 Task 4: Inference, Deployment, and Critical Reflection

#### 5.8.1 Part A: Inference and Deployment

Launch nanochat's inference server and web UI, and interact with your trained model. Then address the following in your report (approximately one page):

1. **The inference loop:** Explain autoregressive sampling, including how the key–value (KV) cache avoids redundant computation across decoding steps. Experiment with different decoding parameters: What is temperature, and how do values of 0.1, 0.7, and 1.5 affect output quality and diversity? Show and discuss five representative examples from your model.
2. **Production hardening:** Identify three concrete engineering changes needed before deploying nanochat's inference server at production scale.

{7}------------------------------------------------

<span id="page-7-0"></span>3. **Capability comparison and scaling:** Find published ARC-Easy, ARC-Challenge, and GSM-8K scores for at least two publicly available models: one comparable in size to your nanochat model and one substantially larger. Report the evaluation conditions (for example, zero-shot or few-shot), and cite each source. Put your own results alongside the published scores in a table. Where does your model stand relative to similarly sized models? What does the gap between your model and a well-trained model of the same size reveal about data quality, training duration, and alignment?

#### 5.8.2 Part B: Critical Reflection

Write a short discussion (approximately 0.5–1 page) responding to *one* of the following prompts.

*Prompt 1 — Access and democratization.* nanochat's headline claim is that “the best ChatGPT that $100 can buy” brings LLM training within reach of individuals. Critically assess this claim. Who still cannot access this pipeline, and why?

*Prompt 2 — Data and values.* nanochat pretrains on CLIMBMix, a clustering-based, iteratively bootstrapped data-mixture corpus. What decisions shaped this dataset, and whose values do they reflect? How might a different pretraining corpus lead to a qualitatively different model?

*Prompt 3 — Environmental cost.* nanochat's “affordable” training run uses about four hours on a node with eight H100 GPUs. Estimate the carbon cost of your own run using an established method, such as the ML CO₂ Impact calculator.<sup>5</sup> Situate the result in the broader context of LLM environmental impact. Does *cheap* mean *low-impact*?

Part A: Pay attention to the accuracy of your inference-loop description, the practicality of your production-hardening proposals, and the completeness and relevance of your benchmark comparison.

Part B: Pay attention to analytical depth, use of evidence, and willingness to challenge assumptions.

## 5.9 Recommended Reading

- [Vaswani et al.](#page-9-0) [\[2017\]](#page-9-0) — The original Transformer paper; essential reading before Task 2.
  
<sup>5</sup><https://mlco2.impact.compute.org>

{8}------------------------------------------------

<span id="page-8-0"></span>- [Hoffmann et al.](#page-9-0) [\[2022\]](#page-9-0) — Chinchilla scaling laws; required for the scaling-law question in Task 2.
- [Ouyang et al.](#page-9-0) [\[2022\]](#page-9-0) — InstructGPT; foundational to the SFT and RLHF discussion in Task 3.
- [Diao et al.](#page-9-0) [\[2026\]](#page-9-0) — CLIMBMix / Nemotron-CLIMB; read about its data-curation methodology before the Task 2 dataset analysis.
- [Bender et al.](#page-9-0) [\[2021\]](#page-9-0) — A critical perspective on large language models; relevant to the Task 4 reflection prompts.
- [Hugging Face](#page-9-0) [\[2022\]](#page-9-0) — The Hugging Face LLM Course; covers Transformers, Tokenizers, Datasets, and the Hugging Face Hub; useful for Tasks 1, 3, and 4.
- [Labonne](#page-9-0) [\[2023\]](#page-9-0) — Labonne's *LLM Course*; a practitioner roadmap covering LoRA/QLoRA, DPO, GGUF/GPTQ quantization, and vLLM/TGI serving; recommended before Tasks 3 and 4.

## 5.10 Concluding Remarks

This chapter used nanochat as a window into the full LLM training pipeline. You trained a BPE tokenizer, observed a Transformer learn from web text, adapted it to follow instructions, evaluated it on standard benchmarks, and deployed it as a chat assistant.

The gap between your depth-4 model and frontier models will be apparent because you are unlikely to have datacenter-scale compute. Even so, the experiments are scalable in principle: rerunning them at greater depth with more data should produce more capable models, as scaling laws predict. After completing Assignment 1, you will have trained a full LLM and be better prepared for future experiments that require greater scale.
## Bibliography

Loubna Ben Allal et al. SmolTalk, 2024. URL [https://huggingface.co/](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) [datasets/HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). [112](#page-1-0) Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, FAccT '21, page 610–623, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445922. URL <https://doi.org/10.1145/3442188.3445922>. 119 Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, et al. Nemotron-climb: Clustering-based

{9}------------------------------------------------

<span id="page-9-0"></span>iterative data mixture bootstrapping for language model pre-training. *Advances in Neural Information Processing Systems*, 38, 2026. [112,](#page-1-0) [119](#page-8-0) Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre. Training compute-optimal large language models. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. [115,](#page-4-0) [119](#page-8-0) Hugging Face. The hugging face LLM course, 2022. URL [https://](https://huggingface.co/learn/llm-course) [huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course). [113,](#page-2-0) [119](#page-8-0) Mathieu Labonne. LLM course, 2023. URL [https://github.com/mlabonne/](https://github.com/mlabonne/llm-course) [llm-course](https://github.com/mlabonne/llm-course). [113,](#page-2-0) [119](#page-8-0) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022. [113,](#page-2-0) [119](#page-8-0) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. [112](#page-1-0) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. [118](#page-7-0)