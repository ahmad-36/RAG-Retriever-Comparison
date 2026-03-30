# RAG Retriever Comparison

Evaluating retrieval strategies for Retrieval-Augmented Generation (RAG) on the SQuAD dataset. The project compares **BM25**, **DPR** (Dense Passage Retrieval), and **Hybrid** retrievers across two experiments:

1. **Retriever-only evaluation** — measures retrieval quality (Hit@K, MRR) without any LLM generation.
2. **LLM + Retriever evaluation** — end-to-end pipeline that retrieves passages, generates answers via an LLM (Llama 3), and evaluates faithfulness, hallucination, and answer quality.

---

## Project Structure

```
├── requirements.txt
├── Retriever-only/
│   ├── retriever_only_eval.py        # Retriever-only evaluation script
│   └── results/                      # Evaluation results (Hit@K, MRR)
└── LLM+Retriever/
    ├── retriever_llm_faithfulness.py  # Full RAG pipeline with faithfulness evaluation
    ├── quick_model_compare.py         # Compare multiple LLM models side-by-side
    ├── download_models.py             # Download required models
    └── results/LLM_results/           # LLM generation and faithfulness results
```

---

## Experiments

### Experiment 1: Retriever-Only

Evaluates retrieval accuracy across BM25, DPR (`all-MiniLM-L6-v2`), and Hybrid on SQuAD v1 and v2 (20,000 examples each).

| Retriever | Dataset | K=3 Hit@K | K=5 Hit@K | K=9 Hit@K |
|-----------|---------|-----------|-----------|-----------|
| BM25 | SQuAD v1 | 0.699 | 0.748 | 0.798 |
| DPR | SQuAD v1 | 0.857 | 0.904 | 0.940 |
| Hybrid | SQuAD v1 | 0.856 | 0.896 | 0.931 |
| BM25 | SQuAD v2 | 0.706 | 0.755 | 0.804 |
| DPR | SQuAD v2 | 0.859 | 0.905 | 0.940 |
| Hybrid | SQuAD v2 | 0.860 | 0.899 | 0.933 |

### Experiment 2: LLM + Retriever (Faithfulness)

End-to-end RAG evaluation using Llama 3 with NLI-based faithfulness scoring and claim-level hallucination detection on SQuAD v1 (100 examples).

**Best configuration: Hybrid (alpha=0.2, K=10)**
- Faithfulness: **0.94**
- Hallucination Rate: **0.06**
- Retrieval Hit@K: **0.99**
- ROUGE-L: **0.72**
- Semantic Similarity: **0.83**

---

## Key Findings

- **DPR** significantly outperforms BM25 in retrieval accuracy across all K values.
- **Hybrid retrieval** (alpha=0.2, favoring DPR) achieves the best end-to-end faithfulness (94%) with the lowest hallucination rate (6%).
- Increasing **Top-K** improves retrieval hit rate but does not always improve generation quality — the sweet spot is K=5 to K=10.
- Most hallucinations are **intrinsic** (rephrasing errors) rather than extrinsic (fabricated facts).

---

## Metrics

- **Retriever-only**: Hit@K, MRR (Mean Reciprocal Rank)
- **LLM + Retriever**: F1, BLEU, ROUGE-L, Semantic Similarity, Faithfulness, Hallucination Rate (extrinsic vs intrinsic)

---

## Setup

```bash
pip install -r requirements.txt
```

### Running Retriever-Only Evaluation

```bash
python Retriever-only/retriever_only_eval.py --datasets both --top-k 3 5 9 --limit 20000
```

### Running LLM + Retriever Evaluation

Requires an Ollama server running locally with the desired model:

```bash
python LLM+Retriever/retriever_llm_faithfulness.py
```

Environment variables can configure the run (dataset selection, top-k, hybrid alpha, etc.). See the script header for details.

### Comparing Multiple LLMs

```bash
python LLM+Retriever/quick_model_compare.py
```

---

## Models Used

- **Dense Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Llama 3 (via Ollama)
- **NLI Model**: `facebook/bart-large-mnli`
- **Sparse Retriever**: BM25 (rank-bm25)

---

## Notes

- The repository supports running on both CPU and GPU.
- Results may vary slightly depending on random seeds and batch sampling.
- Fine-tuned model weights are not included due to size constraints.

---

## License

This project is released for academic and educational purposes only.
All rights reserved. Ahmad Abdullah, 2025.
