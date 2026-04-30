from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CellConfig:
    expose_z: bool
    expose_u: bool
    pi_b_known: bool
    on_policy: bool


CELL_CONFIGS: dict[int, CellConfig] = {
    1: CellConfig(expose_z=True, expose_u=False, pi_b_known=True, on_policy=True),
    2: CellConfig(expose_z=False, expose_u=False, pi_b_known=True, on_policy=True),
    3: CellConfig(expose_z=True, expose_u=False, pi_b_known=True, on_policy=False),
    4: CellConfig(expose_z=False, expose_u=False, pi_b_known=True, on_policy=False),
    5: CellConfig(expose_z=True, expose_u=False, pi_b_known=False, on_policy=False),
    6: CellConfig(expose_z=False, expose_u=False, pi_b_known=False, on_policy=False),
    7: CellConfig(expose_z=True, expose_u=True, pi_b_known=False, on_policy=False),
    8: CellConfig(expose_z=False, expose_u=True, pi_b_known=False, on_policy=False),
}


def export_cell_configs() -> dict[str, dict[str, bool]]:
    return {str(cell): asdict(config) for cell, config in CELL_CONFIGS.items()}
