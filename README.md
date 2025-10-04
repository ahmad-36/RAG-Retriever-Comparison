# RAG-Retriever-Comparison
## Overview

This repository contains code and experimental results for analyzing the effect of different retrieval strategies — **BM25**, **fine-tuned DPR**, and **Hybrid (BM25+DPR)** — on the **accuracy** and **hallucination rate** of Retrieval-Augmented Generation (RAG) systems.
The experiments are based on the **SQuAD v1/v2** datasets and include fine-tuning of DPR models, retriever evaluation, and integration with a lightweight LLM (GPT-Neo 125M).

---

## 🔍 Repository Structure

```
RAG-Retrieval-Experiments/
│
├── fine_tune_dpr.py                # Fine-tunes DPR encoders on SQuAD v1
├── bm25_dpr_hybrid_retrievers.py   # Implements BM25, DPR, and Hybrid retrieval
├── retriever_evaluation.py         # Evaluates retrievers using semantic similarity
├── rag_llm_evaluation.py           # RAG pipeline evaluation with LLM (GPT-Neo)
│
├── retrievers_v1.pkl               # Serialized retriever index (BM25, DPR, Hybrid)
├── retriever_results_semantic.csv  # Retriever-only results
├── retriever_llm_results_light.csv # End-to-end RAG results
│
├── README.md                       # Documentation
└── requirements.txt                # Required dependencies
```

---

## ⚙️ Installation

1️⃣ **Clone the repository**

```
git clone https://github.com/yourusername/RAG-Retrieval-Experiments.git
cd RAG-Retrieval-Experiments
```

2️⃣ **Install dependencies**

```
pip install -r requirements.txt
```

If running in Google Colab, dependencies will be installed automatically within the scripts.

---

## 📘 Experiments Summary

### 1. Fine-Tuning DPR on SQuAD

File: `fine_tune_dpr.py`

* Uses **DPR (Dense Passage Retriever)** encoders from Facebook.
* Fine-tunes on 20k SQuAD v1 examples using contrastive learning.
* Optimizer: Adam, LR=1e-5, 2 epochs, batch size 8.
* Saves models as:

  * `/content/dpr_question_encoder_squad_20k_gpu`
  * `/content/dpr_ctx_encoder_squad_20k_gpu`

---

### 2. BM25, Fine-Tuned DPR, and Hybrid Retriever

File: `bm25_dpr_hybrid_retrievers.py`

* Implements three retrievers:

  * **BM25Retriever** (sparse)
  * **DPRRetriever** (dense, fine-tuned)
  * **HybridRetriever** (weighted fusion)
* Uses SQuAD passages for indexing.
* Saves retriever objects (`retrievers_v1.pkl`) for reuse.

---

### 3. Retriever Evaluation

File: `retriever_evaluation.py`

* Evaluates retriever performance using:

  * **Precision**, **Recall**, **F1**, **Accuracy**
  * **Average Semantic Similarity**
* Semantic evaluation uses `sentence-transformers/all-MiniLM-L6-v2`
* Saves results to:

  * `retriever_results_semantic.csv`
  * `retriever_metrics_semantic.csv`

---

### 4. RAG (Retriever + LLM) Evaluation

File: `rag_llm_evaluation.py`

* Integrates retrievers with **GPT-Neo 125M**.
* Evaluates **generated answer quality** via:

  * **F1**, **BLEU**, **ROUGE-L**, **Semantic Similarity**
  * **Hallucination Rate**
* Context limited to 500 tokens per query.
* Results saved as:

  * `retriever_llm_results_light.csv`
  * `retriever_llm_metrics_light.csv`

---

## 📊 Key Findings

* **BM25** performs best on small, coherent datasets.
* **Dense DPR** improves with fine-tuning and large datasets.
* **Hybrid retrieval** yields the best balance between accuracy and factual grounding.
* Proper **retrieval parameter tuning (Top-K)** and **semantic evaluation** reduce hallucinations significantly.

---

## 🧠 Requirements

* Python 3.9+
* PyTorch with CUDA (optional for GPU)
* Required packages:

  * `transformers`, `datasets`, `torch`, `tqdm`
  * `sentence-transformers`, `scikit-learn`, `nltk`
  * `rouge-score`, `sacrebleu`, `faiss-gpu` (optional)

Example installation:

```
pip install transformers datasets torch sentence-transformers scikit-learn nltk rouge-score sacrebleu tqdm faiss-gpu
```

---

## 🧩 Notes

* The repository supports running on both **CPU** and **GPU**.
* Results are reproducible but can vary slightly depending on random seeds and batch sampling.
* Fine-tuned DPR weights are saved locally (not pushed to GitHub due to size).

---

## 👩‍💻 Author

**[Ahmad Abdullah]**
University of Trier – Department of Computerlinguistik und Digital Humanities
Module: [Natural Language Processing]
Term Paper: *The Impact of Retrieval Strategies on Hallucination and Accuracy in RAG-Based Question Answering*

---

## 📄 License

This project is released for academic and educational purposes only.
All rights reserved © [Your Name], 2025.

---
