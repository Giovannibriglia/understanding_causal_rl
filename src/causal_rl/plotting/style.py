from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def apply_style() -> None:
    style_path = Path(__file__).resolve().parents[3] / "style.mplstyle"
    plt.style.use(str(style_path))
    try:
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$\Delta_{\mathrm{TV}}$")
        fig.canvas.draw()
        plt.close(fig)
    except Exception:
        plt.rcParams["text.usetex"] = False
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Computer Modern Roman" not in available:
        plt.rcParams["font.serif"] = ["DejaVu Serif"]
    # Use a colorblind-friendly palette
    try:
        from causal_rl.plotting.colors import get_palette

        palette = get_palette()
    except Exception:
        palette = [
            "#0173b2",
            "#de8f05",
            "#029e73",
            "#d55e00",
            "#cc78bc",
            "#ca9161",
            "#fbafe4",
            "#949494",
        ]
    plt.rcParams["axes.prop_cycle"] = cycler(color=palette)


def figure(
    width: Literal["single", "double"] = "single", height_ratio: float = 0.6
) -> tuple[Figure, Axes]:
    w = 3.25 if width == "single" else 6.75
    h = max(1.8, w * height_ratio)
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax
