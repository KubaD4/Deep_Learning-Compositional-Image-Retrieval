#!/usr/bin/env python3
"""Focused probe architecture fine sweep.

This script reuses the v3 probe-sweep pipeline but replaces the search space
with 50 variants centred on the best v3 candidates:

- deep MLP probes with ASL/BCE/focal losses;
- residual MLP probes;
- label-wise probes;
- compact, medium, and larger label-transformer probes.

The frozen retrieval system is unchanged. Each probe is trained, calibrated on
validation attributes, and evaluated on the official JSON with the same methods
as v3. The script also writes training-curve PNGs from the per-config
``train_metrics.csv`` files so we can inspect loss/selection-score dynamics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import probe_arch_sweep_v3 as v3


Config = v3.ProbeArchConfig


def focused_configs(profile: str) -> list[Config]:
    if profile == "short":
        return [
            Config(
                "v4_smoke_mlp_deep_asl",
                "mlp",
                loss="asl",
                epochs=1,
                max_steps=4,
                batch_size=512,
                hidden_dims=(1024, 512),
                lr=2e-4,
                dropout=0.05,
                notes="Smoke test for v4 fine sweep.",
            )
        ]

    configs: list[Config] = []

    # MLP variants around p05/p06/p03/p02.
    mlp_specs = [
        ("m01_deep_asl_lr2e4_d05", "asl", (1536, 1024, 512), 2e-4, 0.05, True, 0.0, 4.0),
        ("m02_deep_asl_lr1e4_d05", "asl", (1536, 1024, 512), 1e-4, 0.05, True, 0.0, 4.0),
        ("m03_deep_asl_lr3e4_d05", "asl", (1536, 1024, 512), 3e-4, 0.05, True, 0.0, 4.0),
        ("m04_deep_asl_lr2e4_d00", "asl", (1536, 1024, 512), 2e-4, 0.00, True, 0.0, 4.0),
        ("m05_deep_asl_lr2e4_d15", "asl", (1536, 1024, 512), 2e-4, 0.15, True, 0.0, 4.0),
        ("m06_wide_asl_2048", "asl", (2048, 1024, 512), 2e-4, 0.05, True, 0.0, 4.0),
        ("m07_wide_asl_1536x2", "asl", (1536, 1536, 768), 2e-4, 0.05, True, 0.0, 4.0),
        ("m08_mid_asl_1024x2", "asl", (1024, 1024, 512), 2e-4, 0.05, True, 0.0, 4.0),
        ("m09_xwide_asl_2048_1536", "asl", (2048, 1536, 768), 1e-4, 0.10, True, 0.0, 4.0),
        ("m10_deep_asl_no_posw", "asl", (1536, 1024, 512), 2e-4, 0.05, False, 0.0, 4.0),
        ("m11_deep_bce_lr2e4", "bce", (1536, 1024, 512), 2e-4, 0.05, True, 0.0, 4.0),
        ("m12_deep_bce_lr1e4", "bce", (1536, 1024, 512), 1e-4, 0.05, True, 0.0, 4.0),
        ("m13_deep_focal_lr2e4", "focal_bce", (1536, 1024, 512), 2e-4, 0.05, True, 0.0, 4.0),
        ("m14_asl_gamma2", "asl", (1536, 1024, 512), 2e-4, 0.05, True, 0.0, 2.0),
        ("m15_asl_gamma6", "asl", (1536, 1024, 512), 2e-4, 0.05, True, 0.0, 6.0),
        ("m16_asl_noise005", "asl", (1536, 1024, 512), 2e-4, 0.05, True, 0.005, 4.0),
        ("m17_asl_noise015", "asl", (1536, 1024, 512), 2e-4, 0.05, True, 0.015, 4.0),
        ("m18_lowdrop_lr1e4", "asl", (1024, 1024, 512), 1e-4, 0.05, True, 0.0, 4.0),
        ("m19_lowdrop_lr3e4", "asl", (1024, 1024, 512), 3e-4, 0.05, True, 0.0, 4.0),
        ("m20_deep_asl_longer", "asl", (1536, 1024, 512), 2e-4, 0.05, True, 0.0, 4.0),
    ]
    for name, loss, hidden, lr, dropout, use_pos_weight, noise, gamma_neg in mlp_specs:
        configs.append(
            Config(
                f"v4_{name}",
                "mlp",
                loss=loss,
                epochs=16 if name.endswith("longer") else 14,
                hidden_dims=hidden,
                lr=lr,
                dropout=dropout,
                use_pos_weight=use_pos_weight,
                noise_std=noise,
                asym_gamma_neg=gamma_neg,
            )
        )

    # Residual MLP variants around p08/p09.
    residual_specs = [
        ("r01_1024_b2_asl_lr2e4", "asl", 1024, 2, 2e-4, 0.05, 0.0),
        ("r02_1024_b2_asl_lr3e4", "asl", 1024, 2, 3e-4, 0.05, 0.0),
        ("r03_1024_b3_asl_lr2e4", "asl", 1024, 3, 2e-4, 0.05, 0.0),
        ("r04_1536_b2_asl_lr2e4", "asl", 1536, 2, 2e-4, 0.05, 0.0),
        ("r05_1536_b3_asl_lr2e4", "asl", 1536, 3, 2e-4, 0.10, 0.0),
        ("r06_2048_b2_asl_lr1e4", "asl", 2048, 2, 1e-4, 0.10, 0.0),
        ("r07_1024_b2_bce_lr3e4", "bce", 1024, 2, 3e-4, 0.05, 0.0),
        ("r08_1536_b2_bce_lr2e4", "bce", 1536, 2, 2e-4, 0.05, 0.0),
        ("r09_1024_b3_asl_noise005", "asl", 1024, 3, 2e-4, 0.05, 0.005),
        ("r10_1536_b3_asl_noise005", "asl", 1536, 3, 2e-4, 0.05, 0.005),
    ]
    for name, loss, hidden_dim, blocks, lr, dropout, noise in residual_specs:
        configs.append(
            Config(
                f"v4_{name}",
                "residual_mlp",
                loss=loss,
                epochs=14,
                hidden_dim=hidden_dim,
                blocks=blocks,
                lr=lr,
                dropout=dropout,
                noise_std=noise,
            )
        )

    # Label-wise variants around p12/p13.
    labelwise_specs = [
        ("l01_256_asl_lr2e4", "asl", 256, 2e-4, 0.05),
        ("l02_384_asl_lr2e4", "asl", 384, 2e-4, 0.05),
        ("l03_512_asl_lr2e4", "asl", 512, 2e-4, 0.05),
        ("l04_768_asl_lr1e4", "asl", 768, 1e-4, 0.10),
        ("l05_512_asl_lr1e4", "asl", 512, 1e-4, 0.05),
        ("l06_512_bce_lr2e4", "bce", 512, 2e-4, 0.05),
        ("l07_512_focal_lr2e4", "focal_bce", 512, 2e-4, 0.05),
        ("l08_384_focal_lr2e4", "focal_bce", 384, 2e-4, 0.05),
    ]
    for name, loss, label_dim, lr, dropout in labelwise_specs:
        configs.append(
            Config(
                f"v4_{name}",
                "labelwise",
                loss=loss,
                epochs=14,
                label_dim=label_dim,
                lr=lr,
                dropout=dropout,
            )
        )

    # Transformer variants around p14/p16 plus larger variants. There are only
    # 40 attribute tokens, so big transformers may overfit, but we include them
    # explicitly to test whether extra label-attention capacity helps.
    transformer_specs = [
        ("t01_128_l1_asl_lr2e4", "asl", 128, 1, 4, 2e-4, 0.05),
        ("t02_128_l2_asl_lr2e4", "asl", 128, 2, 4, 2e-4, 0.05),
        ("t03_160_l2_asl_lr2e4", "asl", 160, 2, 4, 2e-4, 0.05),
        ("t04_192_l2_asl_lr2e4", "asl", 192, 2, 4, 2e-4, 0.05),
        ("t05_256_l2_asl_lr1e4", "asl", 256, 2, 4, 1e-4, 0.10),
        ("t06_256_l4_asl_lr1e4", "asl", 256, 4, 4, 1e-4, 0.10),
        ("t07_384_l2_asl_lr1e4", "asl", 384, 2, 8, 1e-4, 0.10),
        ("t08_384_l3_asl_lr1e4", "asl", 384, 3, 8, 1e-4, 0.10),
        ("t09_384_l4_asl_lr8e5", "asl", 384, 4, 8, 8e-5, 0.10),
        ("t10_512_l2_asl_lr8e5", "asl", 512, 2, 8, 8e-5, 0.10),
        ("t11_512_l3_asl_lr8e5", "asl", 512, 3, 8, 8e-5, 0.12),
        ("t12_512_l4_focal_lr8e5", "focal_bce", 512, 4, 8, 8e-5, 0.12),
    ]
    for name, loss, label_dim, layers, heads, lr, dropout in transformer_specs:
        configs.append(
            Config(
                f"v4_{name}",
                "label_transformer",
                loss=loss,
                epochs=14,
                label_dim=label_dim,
                layers=layers,
                heads=heads,
                lr=lr,
                dropout=dropout,
            )
        )

    assert len(configs) == 50, f"Expected 50 configs, got {len(configs)}"
    return configs


def _read_metric_csv(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def write_live_ranked_systems(output_root: Path, rows: list[dict[str, Any]]) -> None:
    """Write a human-readable live leaderboard after every completed config."""
    method_to_mode = {
        "query_only_fill_accuracy": "only_query",
        "hamming_only_fill_accuracy": "only_hamming",
        "query_hamming_fill_accuracy": "both",
    }
    live_rows: list[dict[str, Any]] = [
        {
            "system": "previous_best_model_and_probe_both",
            "config_id": "previous_best",
            "arch": "probe_v2",
            "loss": "calibrated",
            "filter": "both",
            "R@1": 0.12136112854412583,
            "R@5": 0.3503086273282216,
            "R@10": 0.47297151547368665,
            "P@1": 0.12136112854412583,
            "P@5": 0.10076000495632169,
            "P@10": 0.08570375640709439,
            "kept": 32.70156117632821,
        }
    ]

    for row in rows:
        mode = method_to_mode.get(str(row.get("method", "")))
        if mode is None:
            continue
        live_rows.append(
            {
                "system": f"model_and_{row['config_id']}_{mode}",
                "config_id": row["config_id"],
                "arch": row["arch"],
                "loss": row["loss"],
                "filter": mode,
                "R@1": float(row["macro_Recall@1"]),
                "R@5": float(row["macro_Recall@5"]),
                "R@10": float(row["macro_Recall@10"]),
                "P@1": float(row["macro_Precision@1"]),
                "P@5": float(row["macro_Precision@5"]),
                "P@10": float(row["macro_Precision@10"]),
                "kept": float(row["avg_kept_in_pool"]),
            }
        )

    live_rows.sort(key=lambda item: (float(item["R@10"]), float(item["P@10"])), reverse=True)

    csv_path = output_root / "live_ranked_systems.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(live_rows[0]))
        writer.writeheader()
        writer.writerows(live_rows)

    lines = [
        f"{'rank':<5} {'system':<62} {'arch':<18} {'loss':<12} {'filter':<13} "
        f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'P@1':>7} {'P@5':>7} {'P@10':>7} {'kept':>8}",
        "-" * 170,
    ]
    for rank, row in enumerate(live_rows, start=1):
        lines.append(
            f"{rank:<5} "
            f"{str(row['system']):<62} "
            f"{str(row['arch']):<18} "
            f"{str(row['loss']):<12} "
            f"{str(row['filter']):<13} "
            f"{float(row['R@1']):>7.4f} "
            f"{float(row['R@5']):>7.4f} "
            f"{float(row['R@10']):>7.4f} "
            f"{float(row['P@1']):>7.4f} "
            f"{float(row['P@5']):>7.4f} "
            f"{float(row['P@10']):>7.4f} "
            f"{float(row['kept']):>8.1f}"
        )
    (output_root / "live_ranked_systems.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_training_curves(output_root: Path) -> None:
    if v3.plt is None:
        return
    curve_dir = output_root / "training_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_config: dict[str, list[dict[str, float | str]]] = {}
    for metrics_path in sorted((output_root / "probes").glob("*/train_metrics.csv")):
        rows = _read_metric_csv(metrics_path)
        if rows:
            metrics_by_config[metrics_path.parent.name] = rows

    if not metrics_by_config:
        return

    for metric, filename, ylabel in [
        ("train_loss", "train_loss_all_configs.png", "Train loss"),
        ("selection_score", "selection_score_all_configs.png", "Validation selection score"),
        ("acc_pct_hamming_le2", "valid_hamming_le2_all_configs.png", "Validation % images with <=2 attr errors"),
        ("f1_macro_f1", "valid_macro_f1_all_configs.png", "Validation macro F1"),
    ]:
        v3.plt.figure(figsize=(13, 8))
        for config_id, rows in metrics_by_config.items():
            xs = [float(row["epoch"]) for row in rows if metric in row]
            ys = [float(row[metric]) for row in rows if metric in row]
            if xs and ys:
                alpha = 0.35 if len(metrics_by_config) > 12 else 0.75
                v3.plt.plot(xs, ys, linewidth=1.1, alpha=alpha, label=config_id)
        v3.plt.xlabel("Epoch")
        v3.plt.ylabel(ylabel)
        v3.plt.title(f"Probe v4 fine sweep: {ylabel}")
        if len(metrics_by_config) <= 15:
            v3.plt.legend(fontsize=8)
        v3.plt.tight_layout()
        v3.plt.savefig(curve_dir / filename, dpi=160)
        v3.plt.close()

    # Focused readable plot for the current top JSON configs.
    aggregate = output_root / "aggregate_summary.csv"
    if aggregate.exists():
        with aggregate.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("method") == "query_hamming_fill_accuracy"]
        top_ids = [row["config_id"] for row in sorted(rows, key=lambda r: float(r["macro_Recall@10"]), reverse=True)[:10]]
        if top_ids:
            v3.plt.figure(figsize=(12, 7))
            for config_id in top_ids:
                rows = metrics_by_config.get(config_id, [])
                xs = [float(row["epoch"]) for row in rows if "selection_score" in row]
                ys = [float(row["selection_score"]) for row in rows if "selection_score" in row]
                if xs and ys:
                    v3.plt.plot(xs, ys, linewidth=2.0, label=config_id)
            v3.plt.xlabel("Epoch")
            v3.plt.ylabel("Validation selection score")
            v3.plt.title("Probe v4 fine sweep: top-10 JSON configs training curves")
            v3.plt.legend(fontsize=8)
            v3.plt.tight_layout()
            v3.plt.savefig(curve_dir / "selection_score_top_json_configs.png", dpi=180)
            v3.plt.close()


_original_plot_global_summary = v3.plot_global_summary


def plot_global_summary_with_curves(output_root: Path, rows: list[dict[str, Any]]) -> None:
    _original_plot_global_summary(output_root, rows)
    write_live_ranked_systems(output_root, rows)
    plot_training_curves(output_root)


def main() -> int:
    v3.default_configs = focused_configs
    v3.plot_global_summary = plot_global_summary_with_curves
    return v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
