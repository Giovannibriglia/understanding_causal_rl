from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellConfig:
    """Configuration for one cell in the 2×2 causal-RL factorial.

    Axes
    ----
    expose_z    Whether the mediator Z is included in the agent's observation
                (partial-observability axis).
    pi_b_known  Whether the offline behaviour-policy log-probs are logged with
                the dataset (propensity-knowledge axis).

    The hidden confounder U is controlled at dataset-collection time via the
    ``alpha_conf`` environment knob and the behaviour policy's ``depends_on_u``
    flag — not by the cell.
    """

    expose_z: bool
    pi_b_known: bool


# C1 – mdp_known:     fully observed, propensity known   → easiest case
# C2 – mdp_unknown:   fully observed, propensity unknown → needs propensity estimation
# C3 – pomdp_known:   partial observation, propensity known  → needs history
# C4 – pomdp_unknown: partial observation, propensity unknown → both at once
CELL_CONFIGS: dict[int, CellConfig] = {
    1: CellConfig(expose_z=True, pi_b_known=True),
    2: CellConfig(expose_z=True, pi_b_known=False),
    3: CellConfig(expose_z=False, pi_b_known=True),
    4: CellConfig(expose_z=False, pi_b_known=False),
}

CELL_SHORTHANDS: dict[int, str] = {
    1: "mdp_known",
    2: "mdp_unknown",
    3: "pomdp_known",
    4: "pomdp_unknown",
}


def export_cell_configs() -> dict[str, dict[str, bool]]:
    return {
        str(cell): {"expose_z": c.expose_z, "pi_b_known": c.pi_b_known}
        for cell, c in CELL_CONFIGS.items()
    }
