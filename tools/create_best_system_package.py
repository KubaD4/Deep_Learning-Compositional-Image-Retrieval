#!/usr/bin/env python3
"""Create a local package with the current best system and comparison plots.

The package is intentionally report-friendly: it contains the winning result
CSVs, a compact code snapshot, text/prompt embeddings used by CLIP arithmetic,
and plots comparing the assignment baseline, strongest CLIP-only baseline, and
the best current system.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/kuba/deep_learning")
RESULTS = ROOT / "cluster" / "artifacts" / "results"
EMBEDDINGS = ROOT / "cluster" / "data" / "celeba" / "embeddings" / "openai_clip_vit_b32"
OUT = ROOT / "final_best_system"

ASSIGNMENT_BASELINE_DIR = RESULTS / "baselines" / "01_direct_sum"
STRONG_BASELINE_DIR = RESULTS / "baselines" / "04_contrastive_sequential"
BEST_DIR = RESULTS / "sum_model_blends" / "model_plus_generic_delta_100"
BEST_MICRO_DIR = RESULTS / "sum_model_blends" / "model_plus_tuned_delta_050"
COMPARISON_DIR = RESULTS / "sum_model_blends" / "comparison"

REPORT_METRICS = [
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "Precision@1",
    "Precision@5",
    "Precision@10",
]

SUMMARY_METRICS = [
    "macro_Recall@1",
    "macro_Recall@5",
    "macro_Recall@10",
    "macro_Precision@1",
    "macro_Precision@5",
    "macro_Precision@10",
    "micro_Recall@1",
    "micro_Recall@5",
    "micro_Recall@10",
    "micro_Precision@1",
    "micro_Precision@5",
    "micro_Precision@10",
]


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def label_query(query: str, max_len: int = 42) -> str:
    if len(query) <= max_len:
        return query
    return query[: max_len - 3] + "..."


def make_per_query_table(out_dir: Path) -> list[dict]:
    assignment_rows = {
        int(r["query_id"]): r for r in read_csv(ASSIGNMENT_BASELINE_DIR / "per_query_metrics.csv")
    }
    strong_rows = {int(r["query_id"]): r for r in read_csv(STRONG_BASELINE_DIR / "per_query_metrics.csv")}
    best_rows = {int(r["query_id"]): r for r in read_csv(BEST_DIR / "per_query_metrics.csv")}
    rows = []
    for query_id in sorted(assignment_rows):
        vanilla = assignment_rows[query_id]
        strong = strong_rows[query_id]
        s = best_rows[query_id]
        vanilla_value = float(vanilla["Recall@10"])
        strong_value = float(strong["Recall@10"])
        best = float(s["Recall@10"])
        rows.append(
            {
                "query_id": query_id,
                "query": vanilla["query"],
                "assignment_baseline_direct_sum_Recall@10": vanilla_value,
                "strong_clip_baseline_contrastive_sequential_Recall@10": strong_value,
                "final_system_model_plus_generic_delta_100_Recall@10": best,
                "final_minus_assignment_baseline_abs": best - vanilla_value,
                "final_minus_assignment_baseline_rel_percent": (
                    (best - vanilla_value) / vanilla_value * 100.0
                )
                if vanilla_value
                else "",
                "final_minus_strong_clip_baseline_abs": best - strong_value,
                "final_minus_strong_clip_baseline_rel_percent": (
                    (best - strong_value) / strong_value * 100.0
                )
                if strong_value
                else "",
            }
        )
    write_csv(out_dir / "per_query_recall10_three_systems.csv", rows)
    return rows


def make_overall_table(out_dir: Path) -> list[dict]:
    assignment = read_one_csv(ASSIGNMENT_BASELINE_DIR / "summary.csv")
    strong = read_one_csv(STRONG_BASELINE_DIR / "summary.csv")
    best = read_one_csv(BEST_DIR / "summary.csv")
    rows = []
    for metric in SUMMARY_METRICS:
        vanilla = float(assignment[metric])
        strong_value = float(strong[metric])
        val = float(best[metric])
        rows.append(
            {
                "metric": metric,
                "assignment_baseline_direct_sum": vanilla,
                "strong_clip_baseline_contrastive_sequential": strong_value,
                "final_system_model_plus_generic_delta_100": val,
                "final_minus_assignment_baseline_abs": val - vanilla,
                "final_minus_assignment_baseline_rel_percent": (
                    (val - vanilla) / vanilla * 100.0
                )
                if vanilla
                else "",
                "final_minus_strong_clip_baseline_abs": val - strong_value,
                "final_minus_strong_clip_baseline_rel_percent": (
                    (val - strong_value) / strong_value * 100.0
                )
                if strong_value
                else "",
            }
        )
    write_csv(out_dir / "overall_metrics_three_systems.csv", rows)
    write_csv(out_dir / "overall_metrics_macro.csv", [row for row in rows if row["metric"].startswith("macro_")])
    write_csv(out_dir / "overall_metrics_micro.csv", [row for row in rows if row["metric"].startswith("micro_")])
    return rows


def plot_per_query(rows: list[dict], out_dir: Path) -> None:
    queries = [label_query(r["query"]) for r in rows]
    y = list(range(len(rows)))
    assignment = [float(r["assignment_baseline_direct_sum_Recall@10"]) for r in rows]
    strong = [float(r["strong_clip_baseline_contrastive_sequential_Recall@10"]) for r in rows]
    best = [float(r["final_system_model_plus_generic_delta_100_Recall@10"]) for r in rows]
    deltas_assignment = [b - a for a, b in zip(assignment, best)]
    deltas_strong = [b - a for a, b in zip(strong, best)]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [2.2, 1.0]})
    ax = axes[0]
    ax.scatter(assignment, y, label="Assignment baseline: direct sum", color="#94a3b8", s=42)
    ax.scatter(strong, y, label="Strong CLIP baseline: contrastive sequential", color="#6b7280", s=45)
    ax.scatter(best, y, label="Final system: model + generic delta x1.0", color="#0f766e", s=55)
    for yi, a, b in zip(y, assignment, best):
        ax.plot([a, b], [yi, yi], color="#dbeafe", linewidth=2, zorder=0)
    for yi, a, b in zip(y, strong, best):
        ax.plot([a, b], [yi, yi], color="#cbd5e1", linewidth=1.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(queries)
    ax.invert_yaxis()
    ax.set_xlabel("Recall@10")
    ax.set_title("Per-query Recall@10: baselines vs final system")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    height = 0.36
    colors_assignment = ["#0f766e" if d >= 0 else "#dc2626" for d in deltas_assignment]
    colors_strong = ["#14b8a6" if d >= 0 else "#f87171" for d in deltas_strong]
    ax2.barh([i - height / 2 for i in y], deltas_assignment, height=height, color=colors_assignment, label="Final - direct sum")
    ax2.barh([i + height / 2 for i in y], deltas_strong, height=height, color=colors_strong, label="Final - contrastive seq.")
    ax2.axvline(0, color="#111827", linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Absolute improvement in Recall@10")
    ax2.set_title("Final system improvements")
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "per_query_recall10_three_systems.png", dpi=180)
    plt.close(fig)


def plot_overall(rows: list[dict], out_dir: Path) -> None:
    metrics = [r["metric"] for r in rows]
    assignment = [float(r["assignment_baseline_direct_sum"]) for r in rows]
    strong = [float(r["strong_clip_baseline_contrastive_sequential"]) for r in rows]
    best = [float(r["final_system_model_plus_generic_delta_100"]) for r in rows]
    rel_assignment = [float(r["final_minus_assignment_baseline_rel_percent"]) for r in rows]
    rel_strong = [float(r["final_minus_strong_clip_baseline_rel_percent"]) for r in rows]

    short = [m.replace("macro_", "Macro ").replace("micro_", "Micro ") for m in metrics]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2.0, 1.2]})
    ax = axes[0]
    width = 0.25
    ax.bar([i - width for i in x], assignment, width, label="Assignment baseline: direct sum", color="#cbd5e1")
    ax.bar(x, strong, width, label="Strong CLIP baseline: contrastive sequential", color="#94a3b8")
    ax.bar([i + width for i in x], best, width, label="Final system", color="#0f766e")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Overall official metrics: assignment baseline, strong baseline, final system")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax2 = axes[1]
    ax2.bar([i - width / 2 for i in x], rel_assignment, width, label="Final vs direct sum", color="#0f766e")
    ax2.bar([i + width / 2 for i in x], rel_strong, width, label="Final vs contrastive seq.", color="#14b8a6")
    ax2.axhline(0, color="#111827", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=35, ha="right")
    ax2.set_ylabel("Relative improvement (%)")
    ax2.set_title("Relative improvement over both baselines")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "overall_metrics_three_systems.png", dpi=180)
    plt.close(fig)


def _normalize_2d(vector):
    norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    if norm == 0:
        return vector
    return [vector[0] / norm, vector[1] / norm]


def _arrow(ax, start, end, color, label, width=0.012, alpha=0.9):
    ax.arrow(
        start[0],
        start[1],
        end[0] - start[0],
        end[1] - start[1],
        width=width,
        head_width=0.055,
        head_length=0.065,
        length_includes_head=True,
        color=color,
        alpha=alpha,
    )
    ax.text(end[0] + 0.025, end[1] + 0.02, label, color=color, fontsize=11)


def plot_toy_vector_explanations(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    source = _normalize_2d([1.0, 0.14])
    d_smiling = [0.02, 0.14]
    d_eyeglasses = [-0.03, 0.16]
    minus_d_young = [-0.05, 0.08]
    q_sum = _normalize_2d(
        [
            source[0] + d_smiling[0] + d_eyeglasses[0] + minus_d_young[0],
            source[1] + d_smiling[1] + d_eyeglasses[1] + minus_d_young[1],
        ]
    )
    q_model = _normalize_2d([0.82, 0.58])
    target = _normalize_2d([0.59, 0.81])
    q_final = _normalize_2d(
        [
            q_model[0] + (q_sum[0] - source[0]),
            q_model[1] + (q_sum[1] - source[1]),
        ]
    )

    # Euclidean view: useful for explaining why "adding a correction" looks
    # unintuitive if we think in absolute coordinates.
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Toy coordinate view: model + CLIP arithmetic delta")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    _arrow(ax, [0, 0], source, "#111827", "source")
    _arrow(ax, [0, 0], q_model, "#2563eb", "q_model")
    _arrow(ax, [0, 0], q_sum, "#f97316", "q_sum")
    _arrow(ax, [0, 0], q_final, "#7c3aed", "q_final")
    _arrow(ax, [0, 0], target, "#16a34a", "target")
    _arrow(ax, source, [source[0] + d_smiling[0], source[1] + d_smiling[1]], "#0ea5e9", "d_smiling", width=0.006)
    _arrow(ax, source, [source[0] + d_eyeglasses[0], source[1] + d_eyeglasses[1]], "#14b8a6", "d_eyeglasses", width=0.006)
    _arrow(ax, source, [source[0] + minus_d_young[0], source[1] + minus_d_young[1]], "#f59e0b", "-d_young", width=0.006)
    ax.plot(
        [q_model[0], q_final[0]],
        [q_model[1], q_final[1]],
        linestyle="--",
        color="#7c3aed",
        alpha=0.7,
    )
    ax.text(0.72, 0.86, "delta = q_sum - source", color="#7c3aed", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "toy_vector_correction_euclidean.png", dpi=180)
    plt.close(fig)

    # Cosine view: CLIP retrieval ranks by angle after normalization, so
    # vectors with different coordinate magnitudes can be equivalent.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Toy CLIP/cosine view: rank by angle after normalization")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-0.25, 1.15)
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#cbd5e1", linewidth=1.3, alpha=0.55)
    ax.add_patch(circle)
    _arrow(ax, [0, 0], source, "#111827", "source")
    _arrow(ax, [0, 0], q_model, "#2563eb", "q_model")
    _arrow(ax, [0, 0], q_sum, "#f97316", "q_sum")
    _arrow(ax, [0, 0], q_final, "#7c3aed", "q_final")
    _arrow(ax, [0, 0], target, "#16a34a", "target")
    _arrow(ax, source, [source[0] + d_smiling[0], source[1] + d_smiling[1]], "#0ea5e9", "d_smiling", width=0.005)
    _arrow(ax, source, [source[0] + d_eyeglasses[0], source[1] + d_eyeglasses[1]], "#14b8a6", "d_eyeglasses", width=0.005)
    _arrow(ax, source, [source[0] + minus_d_young[0], source[1] + minus_d_young[1]], "#f59e0b", "-d_young", width=0.005)
    ax.plot(
        [q_model[0], q_final[0]],
        [q_model[1], q_final[1]],
        linestyle="--",
        color="#7c3aed",
        alpha=0.75,
    )
    ax.text(0.72, 0.79, "delta = q_sum - source", color="#7c3aed", fontsize=11)
    ax.text(
        -0.22,
        -0.18,
        "CLIP retrieval uses cosine similarity: after normalization, angle matters most.",
        color="#374151",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "toy_vector_correction_clip_cosine.png", dpi=180)
    plt.close(fig)


def make_manifest(overall_rows: list[dict], per_query_rows: list[dict]) -> None:
    assignment = read_one_csv(ASSIGNMENT_BASELINE_DIR / "summary.csv")
    strong = read_one_csv(STRONG_BASELINE_DIR / "summary.csv")
    best = read_one_csv(BEST_DIR / "summary.csv")
    best_micro = read_one_csv(BEST_MICRO_DIR / "summary.csv")
    checkpoint_path = best["checkpoint"]
    local_checkpoint = OUT / "weights" / "best_val_official_like_at10.pt"
    manifest = {
        "best_primary_system": {
            "name": "model_plus_generic_delta_100",
            "reason": "Highest Macro Recall@10 among all current systems.",
            "formula": "q_final = normalize(q_model + 1.0 * (q_generic_sum - source))",
            "result_dir": str(BEST_DIR),
            "macro_Recall@10": float(best["macro_Recall@10"]),
            "micro_Recall@10": float(best["micro_Recall@10"]),
        },
        "best_micro_system": {
            "name": "model_plus_tuned_delta_050",
            "reason": "Highest Micro Recall@10 among all current systems.",
            "formula": "q_final = normalize(q_model + 0.5 * (q_tuned_sum - source))",
            "result_dir": str(BEST_MICRO_DIR),
            "macro_Recall@10": float(best_micro["macro_Recall@10"]),
            "micro_Recall@10": float(best_micro["micro_Recall@10"]),
        },
        "assignment_vanilla_baseline": {
            "name": "direct_sum",
            "reason": "Vanilla zero-shot baseline requested by the assignment: unmodified CLIP plus naive latent arithmetic.",
            "formula": "q = normalize(source + sum(sign * positive_text_embedding(attribute)))",
            "result_dir": str(ASSIGNMENT_BASELINE_DIR),
            "macro_Recall@10": float(assignment["macro_Recall@10"]),
            "micro_Recall@10": float(assignment["micro_Recall@10"]),
        },
        "strongest_zero_shot_clip_baseline": {
            "name": "contrastive_sequential",
            "reason": "Strongest no-training CLIP-only baseline among our arithmetic variants.",
            "formula": "q_i = normalize(q_{i-1} + sign * contrastive_text_direction(attribute))",
            "result_dir": str(STRONG_BASELINE_DIR),
            "macro_Recall@10": float(strong["macro_Recall@10"]),
            "micro_Recall@10": float(strong["micro_Recall@10"]),
        },
        "checkpoint": {
            "local_status": "present" if local_checkpoint.exists() else "missing",
            "local_path": str(local_checkpoint) if local_checkpoint.exists() else "",
            "expected_remote_path": checkpoint_path,
            "note": "The final system uses this learned gate checkpoint plus the CLIP arithmetic delta branch.",
        },
        "assignment_metrics": {
            "primary": "Recall@K for K=1,5,10 averaged across valid source images.",
            "secondary": "Precision@K for K=1,5,10.",
            "reported_here": "macro metrics average per-query values equally; micro metrics average over all source-query cases.",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Final Best System Package

This folder collects the assignment baseline, strongest CLIP-only baseline, final system, and clean report plots.

## Winner For Primary Report Number

`model_plus_generic_delta_100`

Formula:

```text
q_model = learned_gate(source, query)
q_sum = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

Why this wins: it has the best current Macro Recall@10, which is the cleanest query-balanced summary of the assignment's primary Recall@K metric.

## Baselines

Assignment vanilla baseline:

```text
direct_sum
q = normalize(v_ref + t_A - t_B + ...)
```

Strongest zero-shot CLIP-only baseline:

```text
contrastive_sequential
q_i = normalize(q_{{i-1}} + d_i)
```

## Current Values

```text
Assignment baseline Macro Recall@10:   {float(assignment['macro_Recall@10']):.6f}
Strong CLIP baseline Macro Recall@10:  {float(strong['macro_Recall@10']):.6f}
Final system Macro Recall@10:          {float(best['macro_Recall@10']):.6f}

Final vs assignment baseline:          {(float(best['macro_Recall@10']) - float(assignment['macro_Recall@10'])) / float(assignment['macro_Recall@10']) * 100:.2f}%
Final vs strong CLIP baseline:         {(float(best['macro_Recall@10']) - float(strong['macro_Recall@10'])) / float(strong['macro_Recall@10']) * 100:.2f}%

Assignment baseline Micro Recall@10:   {float(assignment['micro_Recall@10']):.6f}
Strong CLIP baseline Micro Recall@10:  {float(strong['micro_Recall@10']):.6f}
Final system Micro Recall@10:          {float(best['micro_Recall@10']):.6f}

Final vs assignment baseline:          {(float(best['micro_Recall@10']) - float(assignment['micro_Recall@10'])) / float(assignment['micro_Recall@10']) * 100:.2f}%
Final vs strong CLIP baseline:         {(float(best['micro_Recall@10']) - float(strong['micro_Recall@10'])) / float(strong['micro_Recall@10']) * 100:.2f}%
```

## Important Files

- `results/clean_report/overall_metrics_three_systems.csv`
- `results/clean_report/overall_metrics_macro.csv`
- `results/clean_report/overall_metrics_micro.csv`
- `results/clean_report/per_query_recall10_three_systems.csv`
- `results/clean_report/overall_metrics_three_systems.png`
- `results/clean_report/per_query_recall10_three_systems.png`
- `results/assignment_baseline_direct_sum/summary.csv`
- `results/strong_clip_baseline_contrastive_sequential/summary.csv`
- `results/best_system_model_plus_generic_delta_100/summary.csv`
- `code/evaluate_sum_model_blends.py`

## Macro vs Micro

Macro average:

```text
1. compute the metric inside each query;
2. average the 14 query-level scores equally.
```

Micro average:

```text
pool all source-query cases together, so queries with more source images count more.
```

For the report, Macro Recall@K is the cleanest query-balanced number. Micro Recall@K is still useful because it shows performance over all evaluated source cases.

## Checkpoint Note

If present, the best learned-gate checkpoint is stored at:

```text
weights/best_val_official_like_at10.pt
```

If that file is missing after cloning, use `weights/fetch_best_weights.sh` from a machine with cluster access.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    report_notes = """# Report Notes

Use this folder as the clean source of final report assets.

## Required Cosine Figure

Include this image when explaining why the vector-delta correction is valid for
CLIP retrieval:

```text
final_best_system/explanations/toy_vector_correction_clip_cosine.png
```

Required nearby explanation:

```text
Stessa direzione. Per cosine similarity sono praticamente uguali.
Il punto chiave: CLIP retrieval non chiede "quanto sono vicino come coordinate assolute?", ma:
"qual è l'immagine con embedding che ha angolo/cosine più alto rispetto a q_final?"
```

## Final System Formula

```text
q_model = learned_gate(source, query)
q_sum = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

## Report Metrics

Report all assignment metrics:

```text
Recall@1, Recall@5, Recall@10
Precision@1, Precision@5, Precision@10
```

Use `results/clean_report/overall_metrics_macro.csv` and
`results/clean_report/overall_metrics_micro.csv` for the final tables.
"""
    (OUT / "REPORT_NOTES.md").write_text(report_notes, encoding="utf-8")


def main() -> int:
    preserved_weight = OUT.parent / ".tmp_best_val_official_like_at10.pt"
    previous_weight = OUT / "weights" / "best_val_official_like_at10.pt"
    if previous_weight.exists():
        shutil.copy2(previous_weight, preserved_weight)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Result snapshots.
    copy_tree(ASSIGNMENT_BASELINE_DIR, OUT / "results" / "assignment_baseline_direct_sum")
    copy_tree(STRONG_BASELINE_DIR, OUT / "results" / "strong_clip_baseline_contrastive_sequential")
    copy_tree(BEST_DIR, OUT / "results" / "best_system_model_plus_generic_delta_100")
    copy_tree(BEST_MICRO_DIR, OUT / "results" / "best_micro_model_plus_tuned_delta_050")
    copy_tree(COMPARISON_DIR, OUT / "results" / "sum_model_blends_comparison")

    # Code snapshot needed to reproduce the blend evaluation.
    code_files = [
        ROOT / "cluster" / "orchestrator" / "evaluate_sum_model_blends.py",
        ROOT / "cluster" / "orchestrator" / "evaluate_orchestrated_router.py",
        ROOT / "cluster" / "scripts" / "learned_gate_core.py",
        ROOT / "cluster" / "scripts" / "project_core.py",
        ROOT / "cluster" / "scripts" / "train_gate_model.py",
        ROOT / "cluster" / "jobs" / "48_evaluate_sum_model_blends_short.sh",
        ROOT / "cluster" / "requirements.txt",
    ]
    for src in code_files:
        copy_file(src, OUT / "code" / src.name)

    # Small text/prompt embedding caches used by arithmetic directions.
    for name in [
        "attribute_text_embeddings.pt",
        "signed_attribute_prompt_embeddings.pt",
        "signed_attribute_prompt_embeddings_v2_photo_templates.pt",
    ]:
        copy_file(EMBEDDINGS / name, OUT / "embeddings" / name)

    # Configs/prompts that define the signed condition embeddings.
    for name in ["attribute_prompts.json", "attribute_prompts_v2_photo_templates.json"]:
        copy_file(ROOT / "cluster" / "configs" / name, OUT / "configs" / name)

    checkpoint_path = read_one_csv(BEST_DIR / "summary.csv")["checkpoint"]
    weights_dir = OUT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    local_weight = weights_dir / "best_val_official_like_at10.pt"
    if preserved_weight.exists():
        shutil.copy2(preserved_weight, local_weight)
        preserved_weight.unlink()

    clean_report = OUT / "results" / "clean_report"
    per_query_rows = make_per_query_table(clean_report)
    overall_rows = make_overall_table(clean_report)
    plot_per_query(per_query_rows, clean_report)
    plot_overall(overall_rows, clean_report)
    plot_toy_vector_explanations(OUT / "explanations")
    make_manifest(overall_rows, per_query_rows)

    missing_note = weights_dir / "WEIGHTS_MISSING.txt"
    if local_weight.exists():
        if missing_note.exists():
            missing_note.unlink()
        (weights_dir / "CHECKPOINT_INFO.txt").write_text(
            "Best learned-gate checkpoint is present locally.\n\n"
            "Local path:\n"
            "final_best_system/weights/best_val_official_like_at10.pt\n\n"
            "Original cluster path:\n"
            f"{checkpoint_path}\n",
            encoding="utf-8",
        )
    else:
        (weights_dir / "WEIGHTS_MISSING.txt").write_text(
            "The learned-gate checkpoint is not present in this local copy.\n\n"
            "Expected best checkpoint on cluster:\n"
            f"{checkpoint_path}\n\n"
            "Suggested command from the Mac:\n"
            f"scp kuba.diquattro@baldo.disi.unitn.it:{checkpoint_path} "
            "/Users/kuba/deep_learning/final_best_system/weights/\n",
            encoding="utf-8",
        )
    fetch_script = weights_dir / "fetch_best_weights.sh"
    fetch_script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        f"scp kuba.diquattro@baldo.disi.unitn.it:{checkpoint_path} "
        "/Users/kuba/deep_learning/final_best_system/weights/\n"
        "echo \"Copied best checkpoint into /Users/kuba/deep_learning/final_best_system/weights/\"\n",
        encoding="utf-8",
    )
    fetch_script.chmod(0o755)

    print(f"Created {OUT}")
    print(f"Clean report: {clean_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
