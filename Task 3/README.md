# Task 3: Abstractive Text Summarization using Phi-2 & QLoRA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-4.36%2B-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)

## 📌 Overview
This project focuses on **Abstractive Text Summarization**, where the model generates a concise summary that captures the core meaning of a source text, rather than just extracting sentences. 

We utilize **Microsoft Phi-2**, a powerful Small Language Model (SLM) with 2.7 Billion parameters. To fine-tune this model on a consumer-grade GPU (Google Colab T4), we implement **QLoRA (Quantized Low-Rank Adaptation)**.

## 📂 Dataset
* **Name:** [XSum (Extreme Summarization)](https://huggingface.co/datasets/EdinburghNLP/xsum)
* **Content:** BBC News articles and their single-sentence summaries.
* **Task:** The model reads the news article and generates a one-sentence summary.

## 🛠️ Methodology

### 1. Model Architecture
* **Base Model:** `microsoft/phi-2`
* **Quantization:** 4-bit Normal Float (NF4) using `bitsandbytes`. This reduces VRAM usage significantly (from ~6GB to ~3GB for model loading).

### 2. Fine-Tuning Strategy (PEFT)
Instead of retraining all 2.7B parameters, we attach **LoRA Adapters** to specific layers of the model:
* **Target Modules:** `Wqkv`, `fc1`, `fc2`, `dense`.
* **Rank (r):** 32
* **Alpha:** 64
* **Trainable Parameters:** Less than 2% of the total model size, enabling fast and efficient training.

### 3. Prompt Engineering
We format the data into an instruction-following structure to guide the model:

```text
### Instruction:
Summarize the news article below concisely.

### Input:
{Full News Article}

### Response:
{Summary}
