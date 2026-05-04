"""Static decision-tree figure for diagnosing a failing RL run.

This figure has no data dependency.  It renders a flowchart with
matplotlib primitives that walks a practitioner from "my run looks bad"
to one of four diagnoses (sample complexity, coverage breakage,
confounding, partial observability), using the same metric thresholds
as the regime-map figure.

Lives in the plotting package so it is generated alongside the
data-driven figures by ``scripts/make_all_figures.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from causal_rl.plotting.style import apply_style


def make_diagnosis_flowchart(
    output_dir: Path,
    *,
    gap_threshold: float = 5.0,
    delta_tv_threshold: float = 0.04,
    ess_threshold: float = 0.5,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Helper for a rounded box centered at (x, y) with text inside.
    def box(
        x: float,
        y: float,
        text: str,
        *,
        width: float = 2.4,
        height: float = 0.9,
        face: str = "white",
        edge: str = "black",
    ) -> tuple[float, float]:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.08",
                fc=face,
                ec=edge,
                linewidth=1.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, wrap=True)
        return x, y

    def arrow(x1: float, y1: float, x2: float, y2: float, *, label: str = "") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                label,
                fontsize=7,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"},
            )

    # Layout (rows top to bottom).
    box(5.0, 9.4, "Read eval.csv (final checkpoint)", face="#E3F2FD")
    box(5.0, 8.2, f"gap_to_oracle ≥ {gap_threshold}?", face="#FFFDE7")
    box(2.5, 7.0, "Done.\nResidual is sample complexity.", face="#E8F5E9")
    box(7.5, 7.0, f"Δ_TV ≥ {delta_tv_threshold}?", face="#FFFDE7")
    box(5.0, 5.6, f"ESS ≥ {ess_threshold}\nand min_prop ≥ 0.01?", face="#FFFDE7")
    box(9.0, 5.6, "Z observed (expose_z)?", face="#FFFDE7")

    # Leaf diagnoses.
    box(2.5, 4.0, "Coverage-broken.\nCollect with higher\naction diversity\nor more samples.", face="#FFEBEE")
    box(5.0, 4.0, "Sample complexity.\nMore samples,\ntighter algorithm.", face="#E8F5E9")
    box(8.0, 4.0, "Confounding visible,\nrecoverable.\nFit IPW or CQL.", face="#FFF8E1")
    box(9.5, 2.6, "Partial observability.\nExpose more state.", face="#FCE4EC")

    # Edges.
    arrow(5.0, 8.95, 5.0, 8.65)
    arrow(4.4, 7.95, 3.0, 7.4, label="no")
    arrow(5.6, 7.95, 7.0, 7.4, label="yes")
    arrow(7.0, 6.55, 5.5, 6.05, label="no")
    arrow(8.0, 6.55, 9.0, 6.05, label="yes")
    arrow(4.4, 5.2, 2.8, 4.45, label="no")
    arrow(5.0, 5.15, 5.0, 4.45, label="yes")
    arrow(8.6, 5.2, 8.0, 4.45, label="yes")
    arrow(9.4, 5.15, 9.5, 2.95, label="no")

    fig.suptitle(
        "Diagnosing a failing RL run", fontsize=12, fontweight="bold", y=0.995
    )
    pdf = output_dir / "diagnosis_flowchart.pdf"
    png = output_dir / "diagnosis_flowchart.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
