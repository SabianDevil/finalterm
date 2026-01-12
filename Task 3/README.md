# Fine-Tuning Phi-2 for Abstractive Summarization 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30%2B-yellow)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)

## 📖 Project Overview
This repository contains the implementation for **Task 3: Abstractive Text Summarization**. We utilize **Microsoft Phi-2** (a 2.7B parameter Small Language Model) and fine-tune it to generate concise summaries of news articles.

To train this model on consumer-grade hardware (Google Colab T4 GPU), we employ **QLoRA (Quantized Low-Rank Adaptation)**, allowing us to achieve high-performance results with significantly reduced memory usage.

## 📂 Dataset
* **Name:** [XSum (Extreme Summarization)](https://huggingface.co/datasets/EdinburghNLP/xsum)
* **Source:** BBC News Articles.
* **Goal:** Generate a one-sentence summary (abstractive) that captures the essence of the article.

## 🛠️ Technical Approach

### 1. Model Architecture
* **Base Model:** `microsoft/phi-2`
* **Quantization:** 4-bit Normal Float (NF4) via `bitsandbytes`.
* **Training Method:** Supervised Fine-Tuning (SFT) with LoRA adapters.

### 2. QLoRA Configuration (Optimized)
Unlike standard configurations, we target **all linear layers** to ensure stable convergence:
* **Rank (r):** 32
* **Alpha:** 64
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `dense`, `fc1`, `fc2`

### 3. Hardware Optimization (T4 Fix)
Specific patches were applied to ensure stability on NVIDIA T4 GPUs:
* Forced `float32` precision for LoRA adapters and LayerNorm modules to prevent `grad_scaler` errors common with mixed precision on older GPUs.

## 🚀 Installation & Usage

### Prerequisites
Install the required libraries for quantization and fine-tuning:

```bash
pip install -q -U torch transformers datasets peft bitsandbytes trl accelerate
