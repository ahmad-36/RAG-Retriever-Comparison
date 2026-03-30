#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quickly compare multiple Ollama LLMs on small RAG faithfulness runs."
    )
    parser.add_argument(
        "--script-path",
        default="retriever_llm_faithfulness.py",
        help="Path to retriever_llm_faithfulness.py",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names. If empty, discover from `ollama list`.",
    )
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--dataset-selection", default="v1")
    parser.add_argument("--top-k-list", default="3")
    parser.add_argument("--hybrid-alpha-list", default="0.5")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=500)
    parser.add_argument("--output-dir", default="results_faithfulness_model_compare")
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://gpu-l4-02.research.tib.eu:11434"),
    )
    parser.add_argument(
        "--apptainer-sif",
        default="/nfs/data/env/cuda_ollama_251030.sif",
        help="Used only for model discovery via `ollama list`.",
    )
    return parser.parse_args()


def clean_host_for_ollama_cli(host: str) -> str:
    host = host.strip()
    host = re.sub(r"^https?://", "", host)
    return host


def discover_models(host: str, sif_path: str):
    cmd = [
        "apptainer",
        "exec",
        "--env",
        f"OLLAMA_HOST={clean_host_for_ollama_cli(host)}",
        sif_path,
        "ollama",
        "list",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return []

    models = []
    for ln in lines[1:]:
        parts = ln.split()
        if not parts:
            continue
        name = parts[0]
        # Skip non-generation models for this benchmark.
        if "embed" in name.lower():
            continue
        models.append(name)
    return models


def best_row_from_summary(summary_path: Path):
    with summary_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def _num(row, key):
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")

    rows.sort(
        key=lambda r: (
            _num(r, "Faithfulness"),
            -_num(r, "Hallucination Rate"),
            _num(r, "F1"),
        ),
        reverse=True,
    )
    return rows[0]


def run_one_model(args, model_name: str, run_root: Path):
    model_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    model_output_dir = run_root / model_tag
    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_output_dir / "run.log"

    env = os.environ.copy()
    env["OLLAMA_HOST"] = args.ollama_host
    env["LLM_MODEL"] = model_name
    env["DATASET_SELECTION"] = args.dataset_selection
    env["NUM_VALIDATION_EXAMPLES"] = str(args.num_examples)
    env["TOP_K_LIST"] = args.top_k_list
    env["HYBRID_ALPHA_LIST"] = args.hybrid_alpha_list
    env["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    env["MAX_CONTEXT_TOKENS"] = str(args.max_context_tokens)
    env["OUTPUT_DIR"] = str(model_output_dir)

    cmd = [sys.executable, args.script_path]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_sec = time.time() - t0

    log_path.write_text(
        f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
    )

    meta_files = sorted(model_output_dir.glob("multi_run_meta_*.json"))
    if proc.returncode != 0 or not meta_files:
        return {
            "model": model_name,
            "status": "failed",
            "return_code": proc.returncode,
            "wall_time_sec": wall_sec,
            "meta_path": "",
            "best_retriever": "",
            "best_topk": "",
            "best_alpha": "",
            "best_faithfulness": "",
            "best_hallucination_rate": "",
            "best_f1": "",
        }

    meta_path = meta_files[-1]
    meta = json.loads(meta_path.read_text())
    summary_path = Path(meta["output_files"]["summary"])
    best = best_row_from_summary(summary_path)

    return {
        "model": model_name,
        "status": "ok",
        "return_code": 0,
        "wall_time_sec": round(wall_sec, 2),
        "run_runtime_sec": round(float(meta.get("total_runtime_sec", 0.0)), 2),
        "meta_path": str(meta_path),
        "best_retriever": best.get("RetrieverType", "") if best else "",
        "best_topk": best.get("TopK", "") if best else "",
        "best_alpha": best.get("HybridAlpha", "") if best else "",
        "best_faithfulness": best.get("Faithfulness", "") if best else "",
        "best_hallucination_rate": best.get("Hallucination Rate", "") if best else "",
        "best_f1": best.get("F1", "") if best else "",
    }


def main():
    args = parse_args()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"compare_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models(args.ollama_host, args.apptainer_sif)

    if not models:
        raise SystemExit("No models found. Provide --models or check ollama connectivity.")

    print(f"Comparing {len(models)} models with {args.num_examples} examples each.")
    print(f"Run root: {run_root}")

    rows = []
    for idx, model in enumerate(models, start=1):
        print(f"[{idx}/{len(models)}] Running model: {model}")
        row = run_one_model(args, model, run_root)
        rows.append(row)
        print(
            f"  -> {row['status']} | wall={row.get('wall_time_sec', '')}s | "
            f"faith={row.get('best_faithfulness', '')} | hall={row.get('best_hallucination_rate', '')}"
        )

    out_csv = run_root / "model_comparison.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    out_json = run_root / "model_comparison.json"
    out_json.write_text(json.dumps(rows, indent=2))

    print("\nDone.")
    print(f"Comparison CSV : {out_csv}")
    print(f"Comparison JSON: {out_json}")


if __name__ == "__main__":
    main()
