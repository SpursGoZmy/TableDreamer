# TableDreamer Synthetic Data
[![paper](https://img.shields.io/badge/Paper-ACL_2025_Finding-red)](https://arxiv.org/abs/2506.08646) [![synthetic_dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/SpursgoZmy/TableDreamer-27K) [![model](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow)](https://huggingface.co/SpursgoZmy/TableDreamer-Llama3.1-8B-Instruct) [![Llama_factory](https://img.shields.io/badge/Code_Base-Llama_Factory-yellow)](https://github.com/hiyouga/LLaMA-Factory)

## 1. Introduction

<img src="./Visualization/data_volume_and_performance.png" width = "350" height = "270" align=right />

LLM-based synthetic data has played an important role in recent development of powerful LLMs. Lots of effort has been dedicated to synthesize training data for different NLP tasks like math, coding, information extraction and so on, but data synthesis for table instruction tuning has not been thoroughly investigated. Recent LLM-based data synthesis methods face several limitations in generating table instruction tuning data. (1) they can not thoroughly explore the vast input space of table understanding tasks, which consists of diverse tables and task instructions, leading to limited data diversity. (2) they ignore the underlying weaknesses in table understanding ability of the target LLM and may blindly pursue the increase of data quantity, resulting in suboptimal data efficiency. (3) synthetic training data with poor diversity could improve table understanding ability but at the huge cost of LLMs' general capacity. In this paper, we introduce a data synthesis pipeline for generating table instruction tuning data (i.e., table, instruction and response), aiming to improve data diversity and efficiency, as well as maintain models' general capacities.

## 2. Synthetic Data and Fine-tuned Model
The 27K TableDreamer synthetic instruction tuning data is available at the [huggingface dataset](https://huggingface.co/datasets/SpursgoZmy/TableDreamer-27K). We synthesize table titles, tables and instructions, and then randomly select one prompt template from multiple candidates to organize them into the final input prompt. The data has been converted into the Alpaca data format and can be directly used to fine-tune LLMs with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) codebase. 

Data schema:
```python
{
  "instruction": "Table caption:\nHistorical Patterns of Military Alliances and Their Influence ..." # Synthetic input prompt including synthetic table, table title and instruction
  "input": "Not used"
  "output": "The military strategies used in the 'Thirty Years' War' were primarily focused on ..." # Synthetic output response from teacher LLMs like GPT-4o or Llama3.1-70B-instruct
}
```

We use 27K GPT-4o synthetic data to fine-tune Llama3.1-8B-Instruct and the saved model checkpoint by LLaMA-Factory is available at [huggingface model](https://huggingface.co/SpursgoZmy/TableDreamer-Llama3.1-8B-Instruct/tree/main), which can be directly used with transformers and vllm inference. We use the recommended hyperparameters from this paper [Rethinking Table Instruction Tuning](https://arxiv.org/abs/2501.14693) during fine-tuning.

## TODOs
- [x] Synthetic data and fine-tuned models
- [ ] Scripts of data synthesis pipeline.
- [ ] The code for model inference.
- [ ] Evaluation data and scripts.


