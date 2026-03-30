import os
import re
import time
import json
import nltk
import torch
import requests
import numpy as np
import pandas as pd
import sacrebleu
from tqdm.auto import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================
# ENV CONFIG HELPERS
# =========================================
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default

def _env_list(name: str, default, cast=float):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [cast(x) for x in vals]

# =========================================
# NLTK
# =========================================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# =========================================
# CONFIG
# =========================================
DATASET_SELECTION = os.getenv("DATASET_SELECTION", "v1")  # "v1", "v2", "both"
NUM_VALIDATION_EXAMPLES = _env_int("NUM_VALIDATION_EXAMPLES", 500)

TOP_K_LIST = _env_list("TOP_K_LIST", [3, 5, 10], cast=int)
MAX_TOP_K = max(TOP_K_LIST)

HYBRID_ALPHA_LIST = _env_list("HYBRID_ALPHA_LIST", [0.2, 0.5, 0.8], cast=float)

MAX_CONTEXT_TOKENS = _env_int("MAX_CONTEXT_TOKENS", 500)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 60)
MAX_ANSWER_WORDS = _env_int("MAX_ANSWER_WORDS", 16)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://gpu-l4-02.research.tib.eu:11434").rstrip("/")
NLI_MODEL = "facebook/bart-large-mnli"
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HF_DATASET_CACHE = None
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results_faithfulness_multi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENTAILMENT_THRESHOLD = 0.60
CONTRADICTION_THRESHOLD = 0.60
CLAIM_RELEVANCE_THRESHOLD = float(os.getenv("CLAIM_RELEVANCE_THRESHOLD", "0.35"))
CLAIM_MIN_LEXICAL_OVERLAP = _env_int("CLAIM_MIN_LEXICAL_OVERLAP", 1)
MIN_CLAIM_WORDS = _env_int("MIN_CLAIM_WORDS", 3)
SPLIT_CLAIMS_ON_CONJUNCTIONS = os.getenv("SPLIT_CLAIMS_ON_CONJUNCTIONS", "false").lower() in ("1", "true", "yes")
SHORT_CLAIM_TOKEN_HEURISTIC_MAX = _env_int("SHORT_CLAIM_TOKEN_HEURISTIC_MAX", 4)

CORPUS_LIMIT = None

COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "that",
    "the", "their", "theirs", "them", "there", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "whom", "why", "with", "you",
    "your", "yours",
}

# =========================================
# GPU / SYSTEM INFO
# =========================================
def build_corpus():
    corpus_data = []

    if DATASET_SELECTION in ["v1", "both"]:
        ds = load_dataset("squad", split="validation", cache_dir=HF_DATASET_CACHE)
        if CORPUS_LIMIT is not None:
            ds = ds.select(range(min(CORPUS_LIMIT, len(ds))))
        corpus_data.extend([("v1", item["context"]) for item in ds])

    if DATASET_SELECTION in ["v2", "both"]:
        ds = load_dataset("squad_v2", split="validation", cache_dir=HF_DATASET_CACHE)
        if CORPUS_LIMIT is not None:
            ds = ds.select(range(min(CORPUS_LIMIT, len(ds))))
        corpus_data.extend([("v2", item["context"]) for item in ds])

    unique_contexts = {}
    for ds_name, ctx in corpus_data:
        key = (ds_name, normalize_text(ctx))
        unique_contexts[key] = normalize_text(ctx)

    corpus_by_dataset = {"v1": [], "v2": []}
    for (ds_name, _), ctx in unique_contexts.items():
        corpus_by_dataset[ds_name].append(ctx)

    return corpus_by_dataset

def build_ranked_docs(documents, scores, top_k):
    order = np.argsort(scores)[::-1][:top_k]
    return [{"document": documents[i], "score": float(scores[i])} for i in order]

def evaluate_config(
    dataset_name,
    question,
    gold_answer,
    retriever_type,
    top_k,
    alpha,
    docs,
    generator,
    nli_tokenizer,
    nli_model,
    semantic_model,
    rouge,
):
    context = build_context_from_docs(docs, MAX_CONTEXT_TOKENS)
    mean_retrieval_score = float(np.mean([r["score"] for r in docs])) if docs else 0.0
    retrieval_hit = answer_in_context(gold_answer, docs)

    llm_answer = generate_answer(generator, context, question)
    faith = faithfulness_report(
        nli_tokenizer,
        nli_model,
        llm_answer,
        context,
        question=question,
        semantic_model=semantic_model,
    )

    f1 = token_f1(gold_answer, llm_answer)
    bleu = sacrebleu.corpus_bleu([llm_answer], [[gold_answer]]).score
    rouge_l = rouge.score(gold_answer, llm_answer)["rougeL"].fmeasure

    try:
        gold_emb = semantic_model.encode([gold_answer], convert_to_numpy=True)
        ans_emb = semantic_model.encode([llm_answer], convert_to_numpy=True)
        sem_sim = cosine_similarity(gold_emb, ans_emb)[0][0]
    except Exception:
        sem_sim = 0.0

    em = exact_match(gold_answer, llm_answer)

    return {
        "Dataset": dataset_name,
        "RetrieverType": retriever_type,
        "Retriever": f"{retriever_type}_{dataset_name}+LLM",
        "TopK": top_k,
        "HybridAlpha": alpha if retriever_type == "hybrid" else np.nan,
        "Question": question,
        "Gold Answer": gold_answer,
        "LLM Answer": llm_answer,
        "Retrieved Context Preview": context[:500],
        "Mean Retrieval Score": mean_retrieval_score,
        "Retrieval Hit@K": retrieval_hit,
        "Exact Match": em,
        "F1": f1,
        "BLEU": bleu,
        "ROUGE-L": rouge_l,
        "Semantic Similarity": sem_sim,
        "Num Claims": faith["num_claims"],
        "Supported Claims": faith["supported"],
        "Unsupported Claims": faith["unsupported"],
        "Contradicted Claims": faith["contradicted"],
        "Relevant Claims": faith["relevant_claims"],
        "Off-topic Claims": faith["off_topic_claims"],
        "Off-topic Claim Rate": faith["off_topic_rate"],
        "Faithfulness": faith["faithfulness"],
        "Extrinsic Hallucination Rate": faith["extrinsic_rate"],
        "Intrinsic Hallucination Rate": faith["intrinsic_rate"],
        "Hallucination Rate": faith["hallucination_rate"],
        "Focused Faithfulness": faith["focused_faithfulness"],
        "Focused Extrinsic Hallucination Rate": faith["focused_extrinsic_rate"],
        "Focused Intrinsic Hallucination Rate": faith["focused_intrinsic_rate"],
        "Focused Hallucination Rate": faith["focused_hallucination_rate"],
    }, faith

def print_system_info():
    print("=" * 80)
    print("SYSTEM / GPU INFO")
    print("=" * 80)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch):", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
    else:
        print("No CUDA GPU visible to PyTorch.")
    print("=" * 80)

device = 0 if torch.cuda.is_available() else -1
device_str = "cuda" if device == 0 else "cpu"

# =========================================
# HELPERS
# =========================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenize(text: str):
    return normalize_text(text).lower().split()

def token_f1(gold: str, pred: str) -> float:
    gold_tokens = simple_tokenize(gold)
    pred_tokens = simple_tokenize(pred)

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    common = {}
    for tok in set(gold_tokens):
        common[tok] = min(gold_tokens.count(tok), pred_tokens.count(tok))

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(gold: str, pred: str) -> int:
    return int(normalize_text(gold).lower() == normalize_text(pred).lower())

def content_tokens(text: str):
    toks = re.findall(r"[A-Za-z0-9']+", normalize_text(text).lower())
    return [t for t in toks if len(t) > 2 and t not in COMMON_STOPWORDS]

def lexical_overlap_count(question: str, claim: str) -> int:
    q = set(content_tokens(question))
    c = set(content_tokens(claim))
    return len(q & c)

def compact_answer(answer: str, max_words: int = MAX_ANSWER_WORDS) -> str:
    answer = normalize_text(answer)
    if not answer:
        return ""

    answer = re.sub(
        r"^(according to (the )?context|based on (the )?context|from (the )?context)\s*[:,.-]?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    sents = nltk.sent_tokenize(answer)
    if sents:
        answer = sents[0]

    answer = re.split(
        r"\b(?:but|however|although|though|whereas)\b",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;:-")

    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words]).rstrip(" ,;:-")

    return answer

def question_claim_relevance(question: str, claim: str, semantic_model) -> float:
    question = normalize_text(question)
    claim = normalize_text(claim)
    if not question or not claim:
        return 0.0

    try:
        q_emb = semantic_model.encode([question], convert_to_numpy=True)
        c_emb = semantic_model.encode([claim], convert_to_numpy=True)
        return float(cosine_similarity(q_emb, c_emb)[0][0])
    except Exception:
        return 0.0

def split_into_claims(answer: str):
    answer = normalize_text(answer)
    if not answer:
        return []

    sents = nltk.sent_tokenize(answer)
    claims = []

    for sent in sents:
        parts = [sent]
        if SPLIT_CLAIMS_ON_CONJUNCTIONS:
            parts = re.split(r"\b(?:and|but|while|whereas)\b", sent, flags=re.IGNORECASE)
        for p in parts:
            p = normalize_text(p)
            if len(p.split()) >= MIN_CLAIM_WORDS:
                claims.append(p)

    seen = set()
    final_claims = []
    for c in claims:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            final_claims.append(c)

    # Keep short direct answers (e.g., "Paris", "1998") evaluable as one claim.
    if len(final_claims) == 0 and answer:
        final_claims = [answer]

    return final_claims

def chunk_context(context: str, chunk_words: int = 120, overlap: int = 30):
    words = context.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += max(1, chunk_words - overlap)
    return chunks

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def minmax_normalize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def answer_in_context(gold_answer: str, docs):
    gold = normalize_text(gold_answer).lower()
    if not gold:
        return 0
    for doc in docs:
        if gold in normalize_text(doc["document"]).lower():
            return 1
    return 0

def build_context_from_docs(docs, max_context_tokens):
    context = " ".join([d["document"] for d in docs])
    return " ".join(context.split()[:max_context_tokens])

# =========================================
# RETRIEVERS
# =========================================
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [simple_tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def score_all(self, query):
        tokenized_query = simple_tokenize(query)
        return np.array(self.bm25.get_scores(tokenized_query), dtype=np.float32)

class DenseRetriever:
    def __init__(self, documents, model_name):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def score_all(self, query):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.doc_embeddings)[0]
        return np.array(sims, dtype=np.float32)

# =========================================
# MODEL LOADERS
# =========================================
def load_generation_pipeline():
    print(f"\nConnecting to Ollama server at {OLLAMA_HOST} using model {LLM_MODEL}.")

    def generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, return_full_text=True):
        options = {"num_predict": int(max_new_tokens)}
        options["temperature"] = 0.7 if do_sample else 0.0

        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        body = resp.json()
        generated = body.get("response", "")
        if return_full_text:
            return [{"generated_text": f"{prompt}{generated}"}]
        return [{"generated_text": generated}]

    return generator

def load_nli_model():
    print("Loading NLI model:", NLI_MODEL)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    if device == 0:
        nli_model = nli_model.cuda()
    nli_model.eval()
    return nli_tokenizer, nli_model

def nli_scores(nli_tokenizer, nli_model, premise: str, hypothesis: str):
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    if device == 0:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = nli_model(**inputs).logits[0].detach().cpu().numpy()

    probs = softmax_np(logits)
    return {
        "contradiction": float(probs[0]),
        "neutral": float(probs[1]),
        "entailment": float(probs[2]),
    }

def generate_answer(generator, context: str, question: str):
    prompt = (
        "Use only the context below to answer the question.\n"
        "Return one short direct answer only (max 12 words).\n"
        "Do not add explanation or extra facts.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    out = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=True,
    )

    text = out[0]["generated_text"]
    answer = text.split("Answer:")[-1].strip()
    answer = answer.split("\n")[0].strip()
    return compact_answer(answer, MAX_ANSWER_WORDS)

# =========================================
# FAITHFULNESS
# =========================================
def verify_claim_against_context(nli_tokenizer, nli_model, claim: str, context: str):
    claim_toks = content_tokens(claim)
    context_norm = normalize_text(context).lower()
    if 0 < len(claim_toks) <= SHORT_CLAIM_TOKEN_HEURISTIC_MAX:
        if all(tok in context_norm for tok in claim_toks):
            return {
                "claim": claim,
                "label": "supported",
                "best_entailment": 1.0,
                "best_contradiction": 0.0,
                "evidence": claim,
            }

    chunks = chunk_context(context, chunk_words=120, overlap=30)

    best_ent = 0.0
    best_con = 0.0
    best_chunk_ent = ""
    best_chunk_con = ""

    for chunk in chunks:
        scores = nli_scores(nli_tokenizer, nli_model, chunk, claim)

        if scores["entailment"] > best_ent:
            best_ent = scores["entailment"]
            best_chunk_ent = chunk

        if scores["contradiction"] > best_con:
            best_con = scores["contradiction"]
            best_chunk_con = chunk

    if best_ent >= ENTAILMENT_THRESHOLD and best_ent > best_con:
        label = "supported"
        evidence = best_chunk_ent
    elif best_con >= CONTRADICTION_THRESHOLD and best_con > best_ent:
        label = "contradicted"
        evidence = best_chunk_con
    else:
        label = "unsupported"
        evidence = ""

    return {
        "claim": claim,
        "label": label,
        "best_entailment": best_ent,
        "best_contradiction": best_con,
        "evidence": evidence,
    }

def faithfulness_report(
    nli_tokenizer,
    nli_model,
    answer: str,
    context: str,
    question: str = "",
    semantic_model=None,
    relevance_threshold: float = CLAIM_RELEVANCE_THRESHOLD,
):
    claims = split_into_claims(answer)

    if len(claims) == 0:
        return {
            "claims": [],
            "num_claims": 0,
            "supported": 0,
            "unsupported": 0,
            "contradicted": 0,
            "faithfulness": 0.0,
            "extrinsic_rate": 0.0,
            "intrinsic_rate": 0.0,
            "hallucination_rate": 0.0,
            "relevant_claims": 0,
            "off_topic_claims": 0,
            "off_topic_rate": 0.0,
            "focused_faithfulness": 0.0,
            "focused_extrinsic_rate": 0.0,
            "focused_intrinsic_rate": 0.0,
            "focused_hallucination_rate": 0.0,
        }

    checked = []
    for claim in claims:
        claim_eval = verify_claim_against_context(nli_tokenizer, nli_model, claim, context)
        overlap = lexical_overlap_count(question, claim) if question else 0
        relevance = question_claim_relevance(question, claim, semantic_model) if semantic_model is not None else 1.0
        if semantic_model is not None:
            is_relevant = int(
                relevance >= relevance_threshold
                or overlap >= CLAIM_MIN_LEXICAL_OVERLAP
            )
        else:
            is_relevant = int(overlap >= CLAIM_MIN_LEXICAL_OVERLAP) if question else 1
        claim_eval["question_relevance"] = relevance
        claim_eval["question_overlap"] = overlap
        claim_eval["question_relevant"] = is_relevant
        checked.append(claim_eval)

    # For short direct answers (single claim), avoid over-filtering relevance.
    if len(checked) == 1:
        checked[0]["question_relevant"] = 1

    supported = sum(1 for x in checked if x["label"] == "supported")
    unsupported = sum(1 for x in checked if x["label"] == "unsupported")
    contradicted = sum(1 for x in checked if x["label"] == "contradicted")
    total = len(checked)

    relevant_checked = [x for x in checked if x["question_relevant"] == 1]
    relevant_total = len(relevant_checked)
    relevant_supported = sum(1 for x in relevant_checked if x["label"] == "supported")
    relevant_unsupported = sum(1 for x in relevant_checked if x["label"] == "unsupported")
    relevant_contradicted = sum(1 for x in relevant_checked if x["label"] == "contradicted")
    off_topic_claims = total - relevant_total

    return {
        "claims": checked,
        "num_claims": total,
        "supported": supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "faithfulness": supported / total if total else 0.0,
        "extrinsic_rate": unsupported / total if total else 0.0,
        "intrinsic_rate": contradicted / total if total else 0.0,
        "hallucination_rate": (unsupported + contradicted) / total if total else 0.0,
        "relevant_claims": relevant_total,
        "off_topic_claims": off_topic_claims,
        "off_topic_rate": off_topic_claims / total if total else 0.0,
        "focused_faithfulness": relevant_supported / relevant_total if relevant_total else 0.0,
        "focused_extrinsic_rate": relevant_unsupported / relevant_total if relevant_total else 0.0,
        "focused_intrinsic_rate": relevant_contradicted / relevant_total if relevant_total else 0.0,
        "focused_hallucination_rate": (relevant_unsupported + relevant_contradicted) / relevant_total if relevant_total else 0.0,
    }

# =========================================
# MAIN
# =========================================
def main():
    total_start = time.time()
    print_system_info()
    print("Using device:", device_str)

    print("\nLoading semantic model for evaluation:", SEMANTIC_MODEL)
    semantic_model = SentenceTransformer(SEMANTIC_MODEL)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print("\nLoading evaluation data...")
    val_data = load_eval_data()  # Call the newly added function here
    print("Validation examples:", len(val_data))

    print("\nBuilding corpus for retrievers...")
    corpus_by_dataset = build_corpus()

    print("\nBuilding retriever backends...")
    retrievers = {}
    documents_by_dataset = {}

    if DATASET_SELECTION in ["v1", "both"] and len(corpus_by_dataset["v1"]) > 0:
        documents_by_dataset["v1"] = corpus_by_dataset["v1"]
        print("Building v1 BM25 backend...")
        bm25_v1 = BM25Retriever(corpus_by_dataset["v1"])
        print("Building v1 Dense backend...")
        dense_v1 = DenseRetriever(corpus_by_dataset["v1"], SEMANTIC_MODEL)
        retrievers["v1"] = {"bm25": bm25_v1, "dpr": dense_v1}

    if DATASET_SELECTION in ["v2", "both"] and len(corpus_by_dataset["v2"]) > 0:
        documents_by_dataset["v2"] = corpus_by_dataset["v2"]
        print("Building v2 BM25 backend...")
        bm25_v2 = BM25Retriever(corpus_by_dataset["v2"])
        print("Building v2 Dense backend...")
        dense_v2 = DenseRetriever(corpus_by_dataset["v2"], SEMANTIC_MODEL)
        retrievers["v2"] = {"bm25": bm25_v2, "dpr": dense_v2}

    generator = load_generation_pipeline()  # Uses Gemma3 via Ollama
    nli_tokenizer, nli_model = load_nli_model()

    results = []
    claim_rows = []

    print("\nStarting multi-config evaluation...")
    eval_start = time.time()
    total_loops = len(val_data) * len(TOP_K_LIST) * (2 + len(HYBRID_ALPHA_LIST))
    pbar = tqdm(total=total_loops, desc="Retriever+LLM multi-eval", dynamic_ncols=True)

    for dataset_name, example in val_data:
        question = example["question"]
        gold_answer = example["answers"]["text"][0] if example["answers"]["text"] else ""

        docs = documents_by_dataset[dataset_name]
        bm25_backend = retrievers[dataset_name]["bm25"]
        dpr_backend = retrievers[dataset_name]["dpr"]

        # Compute score vectors once per query
        bm25_scores = bm25_backend.score_all(question)
        dpr_scores = dpr_backend.score_all(question)

        bm25_norm = minmax_normalize(bm25_scores)
        dpr_norm = minmax_normalize(dpr_scores)

        # Precompute rankings
        bm25_top_docs_max = build_ranked_docs(docs, bm25_scores, MAX_TOP_K)
        dpr_top_docs_max = build_ranked_docs(docs, dpr_scores, MAX_TOP_K)
        hybrid_docs_by_alpha = {}
        for alpha in HYBRID_ALPHA_LIST:
            hybrid_scores = alpha * bm25_norm + (1.0 - alpha) * dpr_norm
            hybrid_docs_by_alpha[alpha] = build_ranked_docs(docs, hybrid_scores, MAX_TOP_K)

        # Evaluation steps
        for retriever_type, max_docs in [("bm25", bm25_top_docs_max), ("dpr", dpr_top_docs_max)]:
            for top_k in TOP_K_LIST:
                config_start = time.time()
                docs_k = max_docs[:top_k]

                row, faith = evaluate_config(
                    dataset_name=dataset_name,
                    question=question,
                    gold_answer=gold_answer,
                    retriever_type=retriever_type,
                    top_k=top_k,
                    alpha=np.nan,
                    docs=docs_k,
                    generator=generator,
                    nli_tokenizer=nli_tokenizer,
                    nli_model=nli_model,
                    semantic_model=semantic_model,
                    rouge=rouge,
                )
                row["Step Time Sec"] = time.time() - config_start
                results.append(row)

                for c in faith["claims"]:
                    claim_rows.append({
                        "Dataset": dataset_name,
                        "RetrieverType": retriever_type,
                        "Retriever": row["Retriever"],
                        "TopK": top_k,
                        "HybridAlpha": np.nan,
                        "Question": question,
                        "Gold Answer": gold_answer,
                        "LLM Answer": row["LLM Answer"],
                        "Claim": c["claim"],
                        "Claim Label": c["label"],
                        "Question Relevance": c["question_relevance"],
                        "Question Overlap": c["question_overlap"],
                        "Question Relevant": c["question_relevant"],
                        "Entailment Score": c["best_entailment"],
                        "Contradiction Score": c["best_contradiction"],
                        "Evidence Preview": c["evidence"][:300],
                    })

                done = len(results)
                elapsed = time.time() - eval_start
                avg_time = elapsed / done if done else 0.0
                remaining = total_loops - done
                eta_sec = avg_time * remaining

                pbar.set_postfix({
                    "done": done,
                    "avg_s": f"{avg_time:.2f}",
                    "eta_min": f"{eta_sec / 60:.1f}",
                })
                pbar.update(1)

        for alpha in HYBRID_ALPHA_LIST:
            max_docs = hybrid_docs_by_alpha[alpha]
            for top_k in TOP_K_LIST:
                config_start = time.time()
                docs_k = max_docs[:top_k]

                row, faith = evaluate_config(
                    dataset_name=dataset_name,
                    question=question,
                    gold_answer=gold_answer,
                    retriever_type="hybrid",
                    top_k=top_k,
                    alpha=alpha,
                    docs=docs_k,
                    generator=generator,
                    nli_tokenizer=nli_tokenizer,
                    nli_model=nli_model,
                    semantic_model=semantic_model,
                    rouge=rouge,
                )
                row["Step Time Sec"] = time.time() - config_start
                results.append(row)

                for c in faith["claims"]:
                    claim_rows.append({
                        "Dataset": dataset_name,
                        "RetrieverType": "hybrid",
                        "Retriever": row["Retriever"],
                        "TopK": top_k,
                        "HybridAlpha": alpha,
                        "Question": question,
                        "Gold Answer": gold_answer,
                        "LLM Answer": row["LLM Answer"],
                        "Claim": c["claim"],
                        "Claim Label": c["label"],
                        "Question Relevance": c["question_relevance"],
                        "Question Overlap": c["question_overlap"],
                        "Question Relevant": c["question_relevant"],
                        "Entailment Score": c["best_entailment"],
                        "Contradiction Score": c["best_contradiction"],
                        "Evidence Preview": c["evidence"][:300],
                    })

                done = len(results)
                elapsed = time.time() - eval_start
                avg_time = elapsed / done if done else 0.0
                remaining = total_loops - done
                eta_sec = avg_time * remaining

                pbar.set_postfix({
                    "done": done,
                    "avg_s": f"{avg_time:.2f}",
                    "eta_min": f"{eta_sec / 60:.1f}",
                })
                pbar.update(1)

    pbar.close()

    df_results = pd.DataFrame(results)
    df_claims = pd.DataFrame(claim_rows)

    summary = (
        df_results
        .groupby(["Dataset", "RetrieverType", "TopK", "HybridAlpha"], dropna=False, as_index=False)
        .agg({
            "Retrieval Hit@K": "mean",
            "Exact Match": "mean",
            "F1": "mean",
            "BLEU": "mean",
            "ROUGE-L": "mean",
            "Semantic Similarity": "mean",
            "Num Claims": "mean",
            "Supported Claims": "mean",
            "Unsupported Claims": "mean",
            "Contradicted Claims": "mean",
            "Relevant Claims": "mean",
            "Off-topic Claims": "mean",
            "Off-topic Claim Rate": "mean",
            "Faithfulness": "mean",
            "Extrinsic Hallucination Rate": "mean",
            "Intrinsic Hallucination Rate": "mean",
            "Hallucination Rate": "mean",
            "Focused Faithfulness": "mean",
            "Focused Extrinsic Hallucination Rate": "mean",
            "Focused Intrinsic Hallucination Rate": "mean",
            "Focused Hallucination Rate": "mean",
            "Step Time Sec": "mean",
        })
        .sort_values(["Dataset", "RetrieverType", "TopK", "HybridAlpha"], na_position="first")
    )

    summary_for_pivot = summary.copy()
    summary_for_pivot["HybridAlpha"] = summary_for_pivot["HybridAlpha"].astype(object)
    summary_for_pivot.loc[
        summary_for_pivot["RetrieverType"] != "hybrid", "HybridAlpha"
    ] = "none"

    # Keep only observed config combinations (avoid cartesian expansion with empty rows).
    pivot_summary = (
        summary_for_pivot
        .groupby(["RetrieverType", "TopK", "HybridAlpha"], as_index=False, dropna=False)
        .agg({
            "Retrieval Hit@K": "mean",
            "F1": "mean",
            "BLEU": "mean",
            "ROUGE-L": "mean",
            "Semantic Similarity": "mean",
            "Faithfulness": "mean",
            "Hallucination Rate": "mean",
            "Extrinsic Hallucination Rate": "mean",
            "Intrinsic Hallucination Rate": "mean",
            "Focused Faithfulness": "mean",
            "Focused Hallucination Rate": "mean",
            "Focused Extrinsic Hallucination Rate": "mean",
            "Focused Intrinsic Hallucination Rate": "mean",
            "Off-topic Claim Rate": "mean",
        })
        .sort_values(["RetrieverType", "TopK", "HybridAlpha"])
    )

    best_configs = summary.sort_values(
        by=["Focused Faithfulness", "Focused Hallucination Rate", "F1"],
        ascending=[False, True, False]
    ).head(15)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(OUTPUT_DIR, f"multi_results_{timestamp}.csv")
    claims_path = os.path.join(OUTPUT_DIR, f"multi_claims_{timestamp}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"multi_summary_{timestamp}.csv")
    pivot_path = os.path.join(OUTPUT_DIR, f"multi_pivot_summary_{timestamp}.csv")
    best_path = os.path.join(OUTPUT_DIR, f"multi_best_configs_{timestamp}.csv")
    meta_path = os.path.join(OUTPUT_DIR, f"multi_run_meta_{timestamp}.json")

    df_results.to_csv(results_path, index=False)
    df_claims.to_csv(claims_path, index=False)
    summary.to_csv(summary_path, index=False)
    pivot_summary.to_csv(pivot_path, index=False)
    best_configs.to_csv(best_path, index=False)

    total_runtime = time.time() - total_start
    meta = {
        "dataset_selection": DATASET_SELECTION,
        "num_validation_examples": NUM_VALIDATION_EXAMPLES,
        "top_k_list": TOP_K_LIST,
        "hybrid_alpha_list": HYBRID_ALPHA_LIST,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_answer_words": MAX_ANSWER_WORDS,
        "claim_relevance_threshold": CLAIM_RELEVANCE_THRESHOLD,
        "claim_min_lexical_overlap": CLAIM_MIN_LEXICAL_OVERLAP,
        "split_claims_on_conjunctions": SPLIT_CLAIMS_ON_CONJUNCTIONS,
        "min_claim_words": MIN_CLAIM_WORDS,
        "short_claim_token_heuristic_max": SHORT_CLAIM_TOKEN_HEURISTIC_MAX,
        "llm_model": LLM_MODEL,
        "nli_model": NLI_MODEL,
        "semantic_model": SEMANTIC_MODEL,
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "total_runtime_sec": total_runtime,
        "output_files": {
            "results": results_path,
            "claims": claims_path,
            "summary": summary_path,
            "pivot_summary": pivot_path,
            "best_configs": best_path,
        },
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Total runtime: {total_runtime / 60:.2f} minutes")
    print("Meta JSON:", meta_path)
    
    
def load_eval_data():
    val_data = []

    if DATASET_SELECTION in ["v1", "both"]:
        squad_v1_val = load_dataset(
            "squad",
            split=f"validation[:{NUM_VALIDATION_EXAMPLES}]",
            cache_dir=HF_DATASET_CACHE,
        )
        val_data.extend([("v1", item) for item in squad_v1_val])

    if DATASET_SELECTION in ["v2", "both"]:
        squad_v2_val = load_dataset(
            "squad_v2",
            split=f"validation[:{NUM_VALIDATION_EXAMPLES}]",
            cache_dir=HF_DATASET_CACHE,
        )
        val_data.extend([("v2", item) for item in squad_v2_val])

    return val_data




if __name__ == "__main__":
    main()
