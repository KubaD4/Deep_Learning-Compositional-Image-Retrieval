#!/usr/bin/env python3
"""Create a toy figure explaining broad-pool retrieval vs top-10 precision."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "explanations" / "top_pool_retrieval_diagnostic.png"


def main() -> None:
    rng = np.random.default_rng(7)
    q_final = np.array([0.0, 0.0])
    source = np.array([-1.55, -0.35])

    # Toy projection: many retrieved candidates lie inside the broad top-500
    # neighbourhood. Some official-valid targets are in that region, but the
    # nearest ten by cosine/rank are not guaranteed to be the official-valid ten.
    other = rng.normal(loc=[0.18, -0.02], scale=[0.52, 0.38], size=(135, 2))
    top10 = np.array(
        [
            [-0.13, 0.12],
            [0.09, 0.06],
            [0.15, -0.09],
            [-0.05, -0.16],
            [0.23, 0.14],
            [0.30, -0.03],
            [-0.25, 0.02],
            [0.05, 0.25],
            [-0.18, -0.20],
            [0.31, 0.23],
        ]
    )
    valid = np.array(
        [
            [0.50, 0.44],
            [0.72, 0.22],
            [0.64, -0.31],
            [-0.55, 0.28],
            [-0.42, -0.36],
            [0.02, 0.56],
            [0.40, -0.55],
            [0.86, -0.02],
        ]
    )

    fig, ax = plt.subplots(figsize=(10.8, 7.0), dpi=180)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")

    top500_circle = Circle(q_final, 1.05, facecolor="#4c9aff", alpha=0.08, edgecolor="#2c6ed5", linewidth=2.2)
    top10_circle = Circle(q_final, 0.38, facecolor="#ff9f43", alpha=0.08, edgecolor="#e67e22", linewidth=2.0, linestyle="--")
    ax.add_patch(top500_circle)
    ax.add_patch(top10_circle)

    ax.scatter(other[:, 0], other[:, 1], s=30, c="#c6ccd8", alpha=0.7, label="other candidates in broad pool")
    ax.scatter(top10[:, 0], top10[:, 1], s=78, c="#e4572e", edgecolor="white", linewidth=0.9, label="our top-10 by q_final cosine")
    ax.scatter(valid[:, 0], valid[:, 1], s=95, c="#23a455", marker="D", edgecolor="white", linewidth=0.9, label="official-valid targets in pool")
    ax.scatter([q_final[0]], [q_final[1]], s=160, c="#263238", marker="*", label="q_final prediction")
    ax.scatter([source[0]], [source[1]], s=110, c="#276ef1", marker="s", label="source image embedding")

    ax.annotate(
        "",
        xy=q_final,
        xytext=source,
        arrowprops=dict(arrowstyle="->", linewidth=2.4, color="#276ef1"),
    )
    ax.text(-1.50, -0.48, "source", color="#276ef1", fontsize=11, weight="bold")
    ax.text(0.05, -0.08, "q_final", color="#263238", fontsize=11, weight="bold")
    ax.text(-0.98, 0.96, "top-500 candidate region\\ncontains many official-valid targets", color="#2c6ed5", fontsize=12, weight="bold")
    ax.text(0.32, 0.35, "top-10 by cosine\\ncan include near-but-invalid faces", color="#c2410c", fontsize=11, weight="bold")
    ax.text(0.45, -0.78, "reranking/filtering should select\\nthe green targets inside the pool", color="#16803c", fontsize=11, weight="bold")

    ax.set_title("Why broad-pool retrieval helps: q_final reaches the right region, top-10 selection is the bottleneck", fontsize=14, pad=16)
    ax.set_xlabel("toy CLIP dimension 1")
    ax.set_ylabel("toy CLIP dimension 2")
    ax.set_xlim(-1.9, 1.35)
    ax.set_ylim(-1.15, 1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e8e1d8", linewidth=0.9)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_color("#b8b0a6")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
