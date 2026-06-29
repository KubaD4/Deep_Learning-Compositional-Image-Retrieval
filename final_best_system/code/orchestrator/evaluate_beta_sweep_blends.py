#!/usr/bin/env python3
"""Sweep vector-delta beta values for the strongest learned gate checkpoint.

No training happens here. For each official JSON query and source image, this
evaluates:

    q = normalize(q_model + beta * (q_sum - source))

for both generic and tuned arithmetic sums. The goal is to determine whether a
fixed beta is enough, or whether different queries need an adaptive beta head.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
SCRIPTS = ROOT / "scripts"
ORCH = ROOT / "orchestrator"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORCH))
os.environ.setdefault("DL_PROJECT_ROOT", str(PACKAGE_ROOT))

import evaluate_orchestrated_router as router  # noqa: E402
import evaluate_sum_model_blends as blends  # noqa: E402
from learned_gate_core import load_model_checkpoint, load_prompt_embedding_cache  # noqa: E402
from project_core import (  # noqa: E402
    ARTIFACTS_DIR,
    EMBEDDING_DIR,
    MODEL_ID,
    TOP_KS,
    atomic_json_dump,
    choose_device,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p").replace("-", "m")


def method_name(family: str, beta: float) -> str:
    return f"model_plus_{family}_delta_beta_{beta_tag(beta)}"


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ARTIFACTS_DIR / "results" / "beta_sweep" / f"beta_sweep_{stamp}"


def find_default_checkpoint() -> Path:
    local_best = ROOT / "final_best_system" / "weights" / "best_val_official_like_at10.pt"
    if local_best.exists():
        return local_best
    return router.find_best_official_checkpoint()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_all_method_frames(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    per_queries = []
    for method_dir in sorted(output_root.glob("*")):
        if not method_dir.is_dir() or method_dir.name == "comparison":
            continue
        summary_path = method_dir / "summary.csv"
        per_query_path = method_dir / "per_query_metrics.csv"
        if summary_path.exists() and per_query_path.exists():
            summary = pd.read_csv(summary_path)
            per_query = pd.read_csv(per_query_path)
            summary["method"] = method_dir.name
            per_query["method"] = method_dir.name
            summaries.append(summary)
            per_queries.append(per_query)
    return pd.concat(summaries, ignore_index=True), pd.concat(per_queries, ignore_index=True)


def add_beta_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    family = []
    beta = []
    for method in frame["method"].astype(str):
        if "generic_delta_beta_" in method:
            family.append("generic")
            beta.append(float(method.rsplit("beta_", 1)[1].replace("p", ".")))
        elif "tuned_delta_beta_" in method:
            family.append("tuned")
            beta.append(float(method.rsplit("beta_", 1)[1].replace("p", ".")))
        elif method == "model_only":
            family.append("model_only")
            beta.append(0.0)
        elif method.endswith("_sum_only"):
            family.append(method.replace("_sum_only", ""))
            beta.append(float("nan"))
        else:
            family.append("other")
            beta.append(float("nan"))
    frame["family"] = family
    frame["beta"] = beta
    return frame


def plot_outputs(output_root: Path) -> None:
    summary, perq = load_all_method_frames(output_root)
    summary = add_beta_metadata(summary)
    perq = add_beta_metadata(perq)
    comp = output_root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    summary.to_csv(comp / "combined_summary.csv", index=False)
    perq.to_csv(comp / "combined_per_query_metrics.csv", index=False)

    sweep = summary[summary["family"].isin(["generic", "tuned"])].copy()
    sweep = sweep.sort_values(["family", "beta"])
    sweep[[
        "method",
        "family",
        "beta",
        "macro_Recall@1",
        "macro_Recall@5",
        "macro_Recall@10",
        "micro_Recall@10",
        "macro_Precision@10",
        "micro_Precision@10",
    ]].to_csv(comp / "overall_beta_sweep.csv", index=False)

    plt.figure(figsize=(9, 5))
    for family, color in [("generic", "#2f6f73"), ("tuned", "#b76e22")]:
        sub = sweep[sweep["family"] == family].sort_values("beta")
        plt.plot(sub["beta"], sub["macro_Recall@10"], marker="o", label=f"{family} macro R@10", color=color)
        plt.plot(sub["beta"], sub["micro_Recall@10"], marker="x", linestyle="--", label=f"{family} micro R@10", color=color)
    plt.xlabel("beta")
    plt.ylabel("Recall@10")
    plt.title("Official JSON beta sweep")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(comp / "overall_beta_sweep.png", dpi=180)
    plt.close()

    best_summary = sweep.sort_values("macro_Recall@10", ascending=False).iloc[0]
    (comp / "BEST_OVERALL_BETA.txt").write_text(
        "\n".join(
            [
                f"method={best_summary['method']}",
                f"family={best_summary['family']}",
                f"beta={best_summary['beta']}",
                f"macro_Recall@10={best_summary['macro_Recall@10']}",
                f"micro_Recall@10={best_summary['micro_Recall@10']}",
                f"macro_Precision@10={best_summary['macro_Precision@10']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = []
    for (query_id, family), frame in perq[perq["family"].isin(["generic", "tuned"])].groupby(["query_id", "family"]):
        best = frame.sort_values("Recall@10", ascending=False).iloc[0]
        rows.append(
            {
                "query_id": int(query_id),
                "query": best["query"],
                "family": family,
                "best_beta": float(best["beta"]),
                "best_Recall@10": float(best["Recall@10"]),
                "method": best["method"],
            }
        )
    best_per_query = pd.DataFrame(rows).sort_values(["query_id", "family"])
    best_per_query.to_csv(comp / "per_query_best_beta.csv", index=False)

    plt.figure(figsize=(12, 7))
    labels = []
    values = []
    colors = []
    for _, row in best_per_query.iterrows():
        labels.append(f"{row['query']} | {row['family']} beta={row['best_beta']:.2f}")
        values.append(row["best_Recall@10"])
        colors.append("#2f6f73" if row["family"] == "generic" else "#b76e22")
    order = list(range(len(labels)))[::-1]
    plt.barh([labels[i] for i in order], [values[i] for i in order], color=[colors[i] for i in order])
    plt.xlabel("Best Recall@10")
    plt.title("Best beta per query and sum family")
    plt.tight_layout()
    plt.savefig(comp / "per_query_best_beta.png", dpi=180)
    plt.close()

    for family in ["generic", "tuned"]:
        heat = perq[perq["family"] == family].pivot_table(
            index="query",
            columns="beta",
            values="Recall@10",
            aggfunc="first",
        )
        heat.to_csv(comp / f"per_query_recall10_heatmap_{family}.csv")
        plt.figure(figsize=(10, 7))
        plt.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.01, float(heat.max().max())))
        plt.colorbar(label="Recall@10")
        plt.xticks(range(len(heat.columns)), [f"{c:.2f}" for c in heat.columns], rotation=45)
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xlabel("beta")
        plt.title(f"Per-query Recall@10 beta sweep ({family})")
        plt.tight_layout()
        plt.savefig(comp / f"per_query_recall10_heatmap_{family}.png", dpi=180)
        plt.close()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    started = time.monotonic()
    device = choose_device(args.device)
    checkpoint = args.checkpoint or find_default_checkpoint()
    output_root = args.output_root or default_output_root()
    if args.force and output_root.exists():
        for path in sorted(output_root.glob("*")):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                for child in sorted(path.glob("**/*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)

    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, prompt_v1_bank = router.load_text_banks(device)
    attributes, _, _ = read_attribute_table()
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations_all = load_evaluation()
    query_ids = list(range(len(annotations_all))) if args.query_ids is None else args.query_ids
    gallery_cache = load_torch(EMBEDDING_DIR / "test_image_embeddings.pt")
    gallery = blends.normalize(gallery_cache["embeddings"]).to(device)

    methods = ["model_only", "generic_sum_only", "tuned_sum_only"]
    for beta in args.betas:
        methods.append(method_name("generic", beta))
        methods.append(method_name("tuned", beta))
    records = {method: {} for method in methods}

    atomic_json_dump(
        {
            "checkpoint": str(checkpoint),
            "model_id": MODEL_ID,
            "betas": args.betas,
            "methods": methods,
            "notes": "No-training beta sweep over model_plus_generic_delta_beta and model_plus_tuned_delta_beta.",
        },
        output_root / "experiment_config.json",
    )

    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations_all[query_id]
            conditions = parse_query(item["query"])
            source_indices = [int(index) for index in item["ground_truth"]]
            for start in range(0, len(source_indices), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices))
                batch_indices = source_indices[start:end]
                idx = torch.tensor(batch_indices, device=device)
                source = gallery[idx]
                q_model = blends.model_query(
                    model,
                    source,
                    conditions,
                    attribute_to_index,
                    prompt_cache,
                    str(config.get("condition_mode", "signed_prompt")),
                    device,
                )
                q_generic = blends.generic_sum_query(source, conditions, attribute_to_index, text_bank)
                q_tuned = blends.tuned_sum_query(source, conditions, attribute_to_index, text_bank, prompt_v1_bank)
                query_vectors = {
                    "model_only": q_model,
                    "generic_sum_only": q_generic,
                    "tuned_sum_only": q_tuned,
                }
                for beta in args.betas:
                    query_vectors[method_name("generic", beta)] = blends.vector_delta(q_model, q_generic, source, beta)
                    query_vectors[method_name("tuned", beta)] = blends.vector_delta(q_model, q_tuned, source, beta)

                rows = torch.arange(len(batch_indices), device=device)
                for method, query in query_vectors.items():
                    scores = query @ gallery.T
                    scores[rows, idx] = -torch.inf
                    rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    rows_for_method = records[method].setdefault(query_id, [])
                    rows_for_method.extend(
                        {
                            "method": method,
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top10": ranking,
                        }
                        for source_index, ranking in zip(batch_indices, rankings)
                    )

                print(f"query {query_id + 1}/{len(annotations_all)} sources {end}/{len(source_indices)}")
                if STOP_REQUESTED or (
                    args.time_budget_seconds and time.monotonic() - started >= args.time_budget_seconds
                ):
                    raise SystemExit("Time budget reached before completion")

    for method in methods:
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({"method": method, "checkpoint": str(checkpoint)}, method_dir / "config.json")
        blends.finalize_method(annotations_all, records[method], method_dir, checkpoint)

    if args.query_ids is None:
        plot_outputs(output_root)
    print(f"Beta sweep results saved in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
