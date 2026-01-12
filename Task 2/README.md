# Generative Question Answering with T5

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-Seq2Seq-green)
![Model](https://img.shields.io/badge/Model-T5_Small-orange)

## 📌 Overview
This project implements a **Generative Question Answering** system. Unlike extractive models that just highlight text, this model generates natural language answers based on a given context.

We utilize **T5 (Text-to-Text Transfer Transformer)**, specifically the `t5-small` variant, treating the QA task as a sequence-to-sequence generation problem.

## 📂 Dataset
* **Name:** [SQuAD v1.1 (Stanford Question Answering Dataset)](https://huggingface.co/datasets/squad)
* **Structure:** Context paragraph, Question, and Answer.

## 🛠️ Methodology
* **Model:** `t5-small`
* **Input Format:** `question: [Question] context: [Context]`
* **Output Format:** `[Answer text]`
* **Training:** Fine-tuned using `Seq2SeqTrainer`.

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install transformers datasets evaluate accelerate sentencepiece
    ```
2.  **Run Notebook:** Open `Task2_QA_T5.ipynb` and execute cells.

## 📊 Results
* **Capability:** The model can accurately locate and generate answers from unseen contexts.
* **Example:**
    * *Context:* "Borobudur is a 9th-century Mahayana Buddhist temple in Magelang..."
    * *Question:* "Where is Borobudur?"
    * *Answer:* "Magelang"

---
**Author:** [Nama Anda]
**NIM:** [NIM Anda]
