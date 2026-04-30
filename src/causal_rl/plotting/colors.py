from __future__ import annotations

from typing import Dict, Tuple, List

import hashlib
import colorsys

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Base colorblind-friendly palette (Matplotlib tab20)
_PALETTE = [mcolors.to_hex(c) for c in plt.get_cmap("tab20").colors]

# Preferred algorithm ordering to assign stable base colors
_ALGO_ORDER = ["ppo", "a2c", "dqn", "ddpg", "cql", "confounded_dqn", "td3", "sac"]

# Line styles and markers
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = [None, "o", "s", "^", "D", "v", "P", "X"]

# Caches
_STYLE_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}
_COLOR_CACHE: Dict[str, str] = {}


def get_palette() -> list[str]:
    return list(_PALETTE)


def _hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0, s, v)


def _hsv_to_hex(h_deg: float, s: float, v: float) -> str:
    h = (h_deg % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return mcolors.to_hex((r, g, b))


def _base_color_for_algo(algo: str) -> str:
    algo_l = str(algo).lower() if algo is not None else ""
    if algo_l in _COLOR_CACHE:
        return _COLOR_CACHE[algo_l]
    if algo_l in _ALGO_ORDER:
        idx = _ALGO_ORDER.index(algo_l) % len(_PALETTE)
    else:
        digest = hashlib.md5(algo_l.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(_PALETTE)
    col = _PALETTE[idx]
    _COLOR_CACHE[algo_l] = col
    return col


def get_style(algo: str | None, behaviour: str | None = None) -> Dict[str, object]:
    """Return plotting style for (algorithm, behaviour).

    Color is derived from algorithm base color, with small deterministic
    hue/saturation/value perturbations for different behaviours so that
    each (algo, behaviour) pair has a distinct color while remaining
    visually related to the algorithm base hue.
    """
    key = (str(algo) if algo is not None else "", str(behaviour) if behaviour is not None else "")
    if key in _STYLE_CACHE:
        return _STYLE_CACHE[key]

    base_hex = _base_color_for_algo(algo or "")
    base_h, base_s, base_v = _hex_to_hsv(base_hex)

    if behaviour is None or behaviour == "":
        style = {"color": base_hex, "linestyle": _LINESTYLES[0], "marker": None}
        _STYLE_CACHE[key] = style
        return style

    # Deterministic variant index from behaviour string
    digest = hashlib.md5(str(behaviour).encode("utf-8")).hexdigest()
    hval = int(digest, 16)
    VARIANTS = 9
    OFFSETS = [-48, -36, -24, -12, 0, 12, 24, 36, 48]
    idx = hval % VARIANTS

    # Hue perturbation around base hue
    hue_deg = (base_h + OFFSETS[idx]) % 360.0

    # Slightly vary saturation and value to increase distinctiveness
    center = VARIANTS // 2
    s_delta = (idx - center) * 0.03
    v_delta = -(idx - center) * 0.02
    s = min(0.95, max(0.35, base_s + s_delta))
    v = min(0.95, max(0.45, base_v + v_delta))

    color_hex = _hsv_to_hex(hue_deg, s, v)
    ls = _LINESTYLES[idx % len(_LINESTYLES)]
    mk = _MARKERS[idx % len(_MARKERS)]

    style = {"color": color_hex, "linestyle": ls, "marker": mk}
    _STYLE_CACHE[key] = style
    return style


def parse_algo_beh_from_label(label: str) -> Tuple[str | None, str | None]:
    if label is None:
        return None, None
    s = str(label)
    if "/" in s:
        parts = s.split("/")
        algo = parts[0]
        beh = parts[1] if len(parts) > 1 else None
        return algo, beh
    if " (" in s and s.endswith(")"):
        algo, rest = s.split(" (", 1)
        beh = rest[:-1]
        return algo, beh
    parts = s.split()
    if parts:
        return parts[0], None
    return None, None
