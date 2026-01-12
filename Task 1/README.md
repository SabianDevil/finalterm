# Natural Language Inference (NLI) with DistilBERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30%2B-yellow)
![Task](https://img.shields.io/badge/Task-Text_Classification-red)

## 📌 Overview
This project focuses on **Natural Language Inference (NLI)**, a task to determine the logical relationship between two sentences: a *premise* and a *hypothesis*. The model classifies the relationship into three categories:
1.  **Entailment:** The hypothesis logically follows from the premise.
2.  **Neutral:** There is no logical relationship.
3.  **Contradiction:** The hypothesis conflicts with the premise.

We fine-tune **DistilBERT**, a lighter and faster version of BERT, to achieve high accuracy with lower computational cost.

## 📂 Dataset
* **Name:** [MNLI (Multi-Genre Natural Language Inference)](https://huggingface.co/datasets/glue/viewer/mnli)
* **Source:** GLUE Benchmark.
* **Size:** subset of 4,000 training samples used for demonstration.

## 🛠️ Methodology
* **Model Architecture:** `distilbert-base-uncased`
* **Objective:** Sequence Classification (3 labels).
* **Optimization:** AdamW optimizer with a learning rate of 2e-5.

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install transformers datasets evaluate accelerate scikit-learn
    ```
2.  **Run Notebook:** Open `Task1_NLI_DistilBERT.ipynb` and execute cells.

## 📊 Results
The model successfully learned to distinguish logical relationships.
* **Validation Accuracy:** ~80% (on subset)
* **Inference Example:**
    * *Premise:* "A soccer player is running."
    * *Hypothesis:* "A person is moving."
    * *Prediction:* **Entailment**

---
**Author:** [Nama Anda]
**NIM:** [NIM Anda]
