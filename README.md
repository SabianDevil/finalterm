# Hands-On End-to-End Deep Learning Models (NLP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow)

## 📌 Project Overview
This repository serves as the submission for the **Deep Learning Final Term Exam**. The project focuses on "Fine-Tuning Hugging Face Models," covering three major paradigms in modern Natural Language Processing (NLP):
1.  **Encoder Models:** For Sequence Classification (Logic/Inference).
2.  **Encoder-Decoder Models:** For Generative Question Answering.
3.  **Decoder-Only Models (LLM):** For Abstractive Text Summarization using Parameter-Efficient Fine-Tuning (PEFT).

---

## 👥 Student & Group Identification
* **Class:** [TK-46-02]

| Name | NIM | Role/Task |
| :--- | :--- | :--- |
| **Muhammad Sabian Aziz** | **1103223236** | **All Tasks Implementation** |

---

## 📂 Project Structure
```bash
├── notebooks/
│   ├── Task1_MNLI_DistilBERT.ipynb       # Natural Language Inference
│   ├── Task2_SQuAD_T5.ipynb              # Question Answering
│   └── Task3_XSum_Phi2.ipynb             # Text Summarization (QLoRA)
├── reports/
│   ├── Report_Task1.md
│   ├── Report_Task2.md
│   └── Report_Task3.md
├── README.md
└── requirements.txt
```

📝 Task Breakdown
🔹 Task 1: Natural Language Inference (NLI)
Determining the logical relationship (Entailment, Contradiction, Neutral) between a premise and a hypothesis.
Model Architecture: distilbert-base-uncased (Encoder).
Dataset: GLUE Benchmark - MNLI (Multi-Genre Natural Language Inference).
Key Approach: Fine-tuning a lightweight DistilBERT model to perform sequence-pair classification efficiently.

🔹 Task 2: Generative Question Answering
Generating a textual answer based on a given context paragraph and a question.
Model Architecture: t5-small (Encoder-Decoder / Seq2Seq).
Dataset: SQuAD (Stanford Question Answering Dataset).
Key Approach: Modeling QA as a text-to-text generation problem (question: ... context: ... -> answer).

🔹 Task 3: Abstractive Text Summarization
Generating concise summaries from news articles using a Large Language Model (LLM).
Model Architecture: microsoft/phi-2 (2.7B Parameters - Decoder Only).
Dataset: XSum (Extreme Summarization).
Key Approach: Using QLoRA (Quantized Low-Rank Adaptation) to fine-tune the massive model in 4-bit precision on a consumer GPU (T4).

🛠️ Installation & Setup
To replicate the experiments, install the required dependencies. Note that Task 3 requires specific libraries for quantization.

# General dependencies
pip install torch transformers datasets evaluate scikit-learn matplotlib

# Specific dependencies for Task 3 (Phi-2 QLoRA)
pip install peft bitsandbytes accelerate trl einops

🚀 Usage & Inference
Running Task 1 (MNLI)

from transformers import pipeline
classifier = pipeline("text-classification", model="./models/distilbert-mnli")
classifier("A soccer player is running.", "A person is moving.")
# Output: Entailment

from transformers import pipeline
classifier = pipeline("text-classification", model="./models/distilbert-mnli")
classifier("A soccer player is running.", "A person is moving.")
# Output: Entailment

Running Task 2 (QA)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("./models/t5-squad")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/t5-squad")
input_text = "question: Who is the president? context: The president is..."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))

Running Task 3 (Summarization)
Note: Requires GPU for 4-bit loading.

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base_model = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, "./models/phi2-xsum-adapter")

📊 Results Summary
DistilBERT achieved faster convergence compared to BERT-Base while maintaining competitive accuracy on the MNLI subset.

T5-Small successfully learned to extract and rephrase information from contexts, proving the effectiveness of the Text-to-Text framework.

Phi-2 with QLoRA demonstrated that high-quality abstractive summarization is possible on free-tier GPUs by training only a fraction of parameters (<5%).

🤝 Acknowledgments
Course: Deep Learning (Final Term)

Datasets provided by Hugging Face Hub (glue, squad, xsum).
