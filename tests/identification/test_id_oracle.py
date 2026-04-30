from __future__ import annotations

import pytest

from causal_rl.identification.id_oracle import get_id_status


ENV = "tabular-sepsis-v0"

# Truth table: (cell, alpha_conf, beh_depends_on_u) -> expected status
TRUTH_TABLE: list[tuple[int, float, bool, str]] = [
    # C1 (expose_z=T, pi_b_known=T): always id
    (1, 0.0, False, "id"),
    (1, 2.0, False, "id"),
    (1, 0.0, True, "id"),
    (1, 2.0, True, "id"),
    # C2 (expose_z=T, pi_b_known=F): id unless confounded
    (2, 0.0, False, "id"),
    (2, 2.0, False, "id"),
    (2, 0.0, True, "id"),   # beh_depends_on_u=T but alpha_conf=0 → not confounded
    (2, 2.0, True, "partial_id"),
    # C3 (expose_z=F, pi_b_known=T): always partial_id
    (3, 0.0, False, "partial_id"),
    (3, 2.0, False, "partial_id"),
    (3, 0.0, True, "partial_id"),
    (3, 2.0, True, "partial_id"),
    # C4 (expose_z=F, pi_b_known=F): partial_id unless confounded → non_id
    (4, 0.0, False, "partial_id"),
    (4, 2.0, False, "partial_id"),
    (4, 0.0, True, "partial_id"),  # beh_depends_on_u=T but alpha_conf=0 → not confounded
    (4, 2.0, True, "non_id"),
]


@pytest.mark.parametrize("cell,alpha_conf,beh_depends_on_u,expected", TRUTH_TABLE)
def test_get_id_status(cell: int, alpha_conf: float, beh_depends_on_u: bool, expected: str) -> None:
    result = get_id_status(ENV, cell, alpha_conf=alpha_conf, beh_depends_on_u=beh_depends_on_u)
    assert result == expected, (
        f"cell={cell}, alpha_conf={alpha_conf}, beh_depends_on_u={beh_depends_on_u}: "
        f"expected {expected!r}, got {result!r}"
    )


def test_invalid_cell_returns_non_id() -> None:
    assert get_id_status(ENV, 99) == "non_id"


def test_default_args_cell1_is_id() -> None:
    assert get_id_status(ENV, 1) == "id"


def test_default_args_cell3_is_partial_id() -> None:
    assert get_id_status(ENV, 3) == "partial_id"
