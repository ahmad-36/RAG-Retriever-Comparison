#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


DATASET_DISPLAY_NAMES = {
    "v1": "SQuAD v1",
    "v2": "SQuAD v2",
}

DATASET_HF_NAMES = {
    "v1": "squad",
    "v2": "squad_v2",
}

DEFAULT_SPLITS = {
    "v1": "train",
    "v2": "train",
}

DEFAULT_LIMITS = {
    "v1": 20_000,
    "v2": 20_000,
}

DEFAULT_TOP_K_LIST = (3, 5, 9)

DEFAULT_DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    source_index: int
    question: str
    gold_answers: tuple[str, ...]


@dataclass
class DatasetBundle:
    dataset_key: str
    dataset_name: str
    split: str
    answerable_only: bool
    selected_examples: int
    scanned_examples: int
    unique_documents: int
    documents: list[str]
    document_match_texts: list[str]
    examples: list[EvalExample]


class BM25Retriever:
    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)
        self.tokenized_docs = [tokenize_bm25(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def score_all(self, query: str) -> np.ndarray:
        return np.asarray(
            self.bm25.get_scores(tokenize_bm25(query)),
            dtype=np.float32,
        )


class DenseRetriever:
    def __init__(
        self,
        documents: Sequence[str],
        model: SentenceTransformer,
        batch_size: int,
        show_progress: bool,
    ):
        self.documents = list(documents)
        self.model = model
        self.doc_embeddings = self.model.encode(
            self.documents,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        return self.model.encode(
            list(queries),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def score_query_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        return np.dot(self.doc_embeddings, query_embedding).astype(np.float32)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Retriever-only evaluation for BM25, DPR, and Hybrid on SQuAD."
    )
    parser.add_argument(
        "--dataset-selection",
        choices=["v1", "v2", "both"],
        default="both",
        help="Which datasets to evaluate.",
    )
    parser.add_argument(
        "--v1-limit",
        type=int,
        default=DEFAULT_LIMITS["v1"],
        help="Number of SQuAD v1 examples to evaluate.",
    )
    parser.add_argument(
        "--v2-limit",
        type=int,
        default=DEFAULT_LIMITS["v2"],
        help="Number of SQuAD v2 answerable examples to evaluate.",
    )
    parser.add_argument(
        "--v1-split",
        default=DEFAULT_SPLITS["v1"],
        help="SQuAD v1 split to evaluate.",
    )
    parser.add_argument(
        "--v2-split",
        default=DEFAULT_SPLITS["v2"],
        help="SQuAD v2 split to evaluate.",
    )
    parser.add_argument(
        "--top-k-list",
        default="3,5,9",
        help="Comma-separated K values to evaluate for all datasets.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 in hybrid fusion. DPR gets (1 - alpha).",
    )
    parser.add_argument(
        "--dpr-model-name",
        default=DEFAULT_DENSE_MODEL_NAME,
        help="Dense retriever backend. Defaults to the Experiment 2 dense model.",
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=128,
        help="Batch size for document encoding.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=128,
        help="Batch size for query encoding.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "results_retriever_only"),
        help="Directory for per-query CSV, summary CSV, and metadata JSON.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached datasets/models.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def normalize_match_text(text: str) -> str:
    return str(text).lower().strip()


def parse_int_list(raw_value: str) -> list[int]:
    values = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("K values must be positive integers.")
        values.append(value)
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("At least one K value is required.")
    return unique_values


def tokenize_bm25(text: str) -> list[str]:
    return normalize_text(text).lower().split()


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = OrderedDict()
    for value in values:
        if value not in seen:
            seen[value] = None
    return list(seen.keys())


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-8:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    k = min(k, scores.size)
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    candidate_idx = np.argpartition(scores, -k)[-k:]
    sorted_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    return sorted_idx.astype(np.int64)


def extract_gold_answers(raw_answers: Sequence[str]) -> tuple[str, ...]:
    answers = []
    for answer in raw_answers:
        if normalize_match_text(answer):
            answers.append(str(answer).strip())
    return tuple(unique_preserve_order(answers))


def document_contains_answer(
    document_index: int,
    document_match_texts: Sequence[str],
    gold_answers: Sequence[str],
) -> bool:
    document_text = document_match_texts[int(document_index)]
    return any(
        normalize_match_text(answer) in document_text
        for answer in gold_answers
        if normalize_match_text(answer)
    )


def first_hit_rank(
    doc_indices: Sequence[int],
    document_match_texts: Sequence[str],
    gold_answers: Sequence[str],
) -> int | None:
    normalized_answers = [
        normalize_match_text(answer)
        for answer in gold_answers
        if normalize_match_text(answer)
    ]
    if not normalized_answers:
        return None

    for rank, doc_index in enumerate(doc_indices, start=1):
        if any(answer in document_match_texts[int(doc_index)] for answer in normalized_answers):
            return rank
    return None


def select_dataset_keys(dataset_selection: str) -> list[str]:
    if dataset_selection == "both":
        return ["v1", "v2"]
    return [dataset_selection]


def load_examples_and_corpus(
    dataset_key: str,
    split: str,
    limit: int,
    cache_dir: str | None,
    answerable_only: bool,
    local_files_only: bool,
    disable_progress: bool,
) -> DatasetBundle:
    dataset_name = DATASET_DISPLAY_NAMES[dataset_key]
    hf_name = DATASET_HF_NAMES[dataset_key]
    download_config = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset(
        hf_name,
        split=split,
        cache_dir=cache_dir,
        download_config=download_config,
    )

    documents: list[str] = []
    document_match_texts: list[str] = []
    doc_id_by_text: dict[str, int] = {}
    examples: list[EvalExample] = []
    scanned_examples = 0

    progress = tqdm(
        total=limit,
        desc=f"Selecting {dataset_name}",
        disable=disable_progress,
        dynamic_ncols=True,
    )

    for source_index, item in enumerate(dataset):
        scanned_examples = source_index + 1
        gold_answers = extract_gold_answers(item["answers"]["text"])
        if answerable_only and not gold_answers:
            continue

        question = normalize_text(item["question"])
        context = normalize_text(item["context"])
        if not question or not context:
            continue

        if context not in doc_id_by_text:
            doc_id_by_text[context] = len(documents)
            documents.append(context)
            document_match_texts.append(normalize_match_text(item["context"]))

        examples.append(
            EvalExample(
                example_id=str(item.get("id", f"{dataset_key}-{source_index}")),
                source_index=source_index,
                question=question,
                gold_answers=gold_answers,
            )
        )
        progress.update(1)

        if len(examples) >= limit:
            break

    progress.close()

    return DatasetBundle(
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        split=split,
        answerable_only=answerable_only,
        selected_examples=len(examples),
        scanned_examples=scanned_examples,
        unique_documents=len(documents),
        documents=documents,
        document_match_texts=document_match_texts,
        examples=examples,
    )


def build_summary_row(dataset: str, k: int, retriever: str, stats: dict[str, float]) -> dict[str, float | int | str]:
    count = int(stats["count"])
    return {
        "Dataset": dataset,
        "K": k,
        "Retriever": retriever,
        "Hit@K": stats["hit_sum"] / count if count else 0.0,
        "MRR": stats["mrr_sum"] / count if count else 0.0,
    }


def main() -> None:
    args = parse_args()

    if args.local_files_only:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    per_query_path = output_dir / f"retriever_only_per_query_{timestamp}.csv"
    summary_path = output_dir / f"retriever_only_summary_{timestamp}.csv"
    metadata_path = output_dir / f"retriever_only_metadata_{timestamp}.json"

    dataset_config = {
        "v1": {
            "split": args.v1_split,
            "limit": args.v1_limit,
            "answerable_only": False,
        },
        "v2": {
            "split": args.v2_split,
            "limit": args.v2_limit,
            "answerable_only": True,
        },
    }
    top_k_list = parse_int_list(args.top_k_list)

    started_at = time.time()
    selected_dataset_keys = select_dataset_keys(args.dataset_selection)
    summary_stats: Dict[tuple[str, int, str], dict[str, float]] = {}
    dataset_meta: dict[str, dict[str, object]] = {}

    print(
        f"Loading dense retriever backend {args.dpr_model_name} on "
        f"{'cuda' if torch.cuda.is_available() else 'cpu'}..."
    )
    dense_model = SentenceTransformer(
        args.dpr_model_name,
        local_files_only=args.local_files_only,
    )

    per_query_fieldnames = [
        "Dataset",
        "Split",
        "K",
        "Retriever",
        "ExampleId",
        "SourceIndex",
        "Question",
        "GoldAnswers",
        "Hit@K",
        "MRR",
        "FirstHitRank",
        "TopKPassageIds",
        "TopKScores",
        "TopKMatchFlags",
    ]

    with per_query_path.open("w", newline="", encoding="utf-8") as per_query_file:
        writer = csv.DictWriter(per_query_file, fieldnames=per_query_fieldnames)
        writer.writeheader()

        for dataset_key in selected_dataset_keys:
            config = dataset_config[dataset_key]
            bundle = load_examples_and_corpus(
                dataset_key=dataset_key,
                split=config["split"],
                limit=config["limit"],
                cache_dir=args.hf_cache_dir,
                answerable_only=config["answerable_only"],
                local_files_only=args.local_files_only,
                disable_progress=args.disable_progress,
            )

            if not bundle.examples or not bundle.documents:
                dataset_meta[dataset_key] = {
                    "dataset_name": bundle.dataset_name,
                    "split": bundle.split,
                    "answerable_only": bundle.answerable_only,
                    "selected_examples": bundle.selected_examples,
                    "scanned_examples": bundle.scanned_examples,
                    "unique_documents": bundle.unique_documents,
                    "k_list": top_k_list,
                }
                continue

            print(f"\nBuilding BM25 for {bundle.dataset_name} with {bundle.unique_documents} passages...")
            bm25_retriever = BM25Retriever(bundle.documents)

            print(
                f"Building dense retriever for {bundle.dataset_name} "
                f"with model {args.dpr_model_name}..."
            )
            dense_retriever = DenseRetriever(
                documents=bundle.documents,
                model=dense_model,
                batch_size=args.doc_batch_size,
                show_progress=not args.disable_progress,
            )

            print(f"Encoding {bundle.selected_examples} queries for {bundle.dataset_name}...")
            query_embeddings = dense_retriever.encode_queries(
                [example.question for example in bundle.examples],
                batch_size=args.query_batch_size,
                show_progress=not args.disable_progress,
            )

            progress = tqdm(
                bundle.examples,
                desc=f"Evaluating {bundle.dataset_name}",
                disable=args.disable_progress,
                dynamic_ncols=True,
            )
            for query_index, example in enumerate(progress):
                bm25_scores = bm25_retriever.score_all(example.question)
                dpr_scores = dense_retriever.score_query_embedding(query_embeddings[query_index])

                # Hybrid retrieval reuses the Experiment 2 score-level fusion pattern.
                hybrid_scores = (
                    args.hybrid_alpha * minmax_normalize(bm25_scores)
                    + (1.0 - args.hybrid_alpha) * minmax_normalize(dpr_scores)
                )

                for retriever_name, score_vector in (
                    ("BM25", bm25_scores),
                    ("DPR", dpr_scores),
                    ("Hybrid (BM25 + DPR)", hybrid_scores),
                ):
                    for k in top_k_list:
                        top_indices = top_k_indices(score_vector, k)
                        top_scores = [round(float(score_vector[idx]), 6) for idx in top_indices]
                        first_rank = first_hit_rank(
                            top_indices,
                            bundle.document_match_texts,
                            example.gold_answers,
                        )
                        hit_at_k = 1 if first_rank is not None else 0
                        mrr = 1.0 / first_rank if first_rank is not None else 0.0
                        match_flags = [
                            1 if document_contains_answer(idx, bundle.document_match_texts, example.gold_answers) else 0
                            for idx in top_indices
                        ]

                        writer.writerow(
                            {
                                "Dataset": bundle.dataset_name,
                                "Split": bundle.split,
                                "K": k,
                                "Retriever": retriever_name,
                                "ExampleId": example.example_id,
                                "SourceIndex": example.source_index,
                                "Question": example.question,
                                "GoldAnswers": json.dumps(list(example.gold_answers), ensure_ascii=True),
                                "Hit@K": hit_at_k,
                                "MRR": round(mrr, 6),
                                "FirstHitRank": first_rank if first_rank is not None else "",
                                "TopKPassageIds": json.dumps([int(idx) for idx in top_indices]),
                                "TopKScores": json.dumps(top_scores),
                                "TopKMatchFlags": json.dumps(match_flags),
                            }
                        )

                        key = (bundle.dataset_name, k, retriever_name)
                        stats = summary_stats.setdefault(
                            key,
                            {"count": 0.0, "hit_sum": 0.0, "mrr_sum": 0.0},
                        )
                        stats["count"] += 1
                        stats["hit_sum"] += hit_at_k
                        stats["mrr_sum"] += mrr

            progress.close()
            dataset_meta[dataset_key] = {
                "dataset_name": bundle.dataset_name,
                "split": bundle.split,
                "answerable_only": bundle.answerable_only,
                "selected_examples": bundle.selected_examples,
                "scanned_examples": bundle.scanned_examples,
                "unique_documents": bundle.unique_documents,
                "k_list": top_k_list,
            }

    summary_rows = [
        build_summary_row(dataset, k, retriever, stats)
        for (dataset, k, retriever), stats in sorted(summary_stats.items())
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["Dataset", "K", "Retriever", "Hit@K", "MRR"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    finished_at = time.time()
    metadata = {
        "started_at_epoch": started_at,
        "finished_at_epoch": finished_at,
        "runtime_sec": finished_at - started_at,
        "dataset_selection": args.dataset_selection,
        "top_k_list": top_k_list,
        "datasets": dataset_meta,
        "hybrid_alpha": args.hybrid_alpha,
        "dpr_model_name": args.dpr_model_name,
        "dpr_backend_note": "Dense backend reused from Experiment 2.",
        "doc_batch_size": args.doc_batch_size,
        "query_batch_size": args.query_batch_size,
        "hf_cache_dir": args.hf_cache_dir,
        "local_files_only": args.local_files_only,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_files": {
            "per_query_csv": str(per_query_path),
            "summary_csv": str(summary_path),
            "metadata_json": str(metadata_path),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("\nSummary")
    print("=" * 72)
    for row in summary_rows:
        print(
            f"{row['Dataset']} | K={row['K']} | {row['Retriever']} | "
            f"Hit@K={row['Hit@K']:.4f} | MRR={row['MRR']:.4f}"
        )
    print("=" * 72)
    print(f"Per-query CSV: {per_query_path}")
    print(f"Summary CSV:   {summary_path}")
    print(f"Metadata JSON: {metadata_path}")


if __name__ == "__main__":
    main()
