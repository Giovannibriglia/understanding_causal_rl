"""Graphical identifiability oracle for the 4-cell factorial.

Classifies the target query P(R | do(A), S) as one of:
  ``id``         – point-identifiable from the observational margin in this cell.
  ``partial_id`` – partially identifiable: bounds can be computed but not the
                   exact value.
  ``non_id``     – non-identifiable: the query is not recoverable from the
                   observational distribution alone for any sample size.

Rules are derived from graphical criteria (Pearl's do-calculus / the ID
algorithm) and depend on three runtime inputs:
  - ``cell``             – determines (expose_z, pi_b_known) from CELL_CONFIGS.
  - ``alpha_conf``       – > 0 means U actively biases rewards.
  - ``beh_depends_on_u`` – True when the behaviour policy conditions on U,
                           creating a spurious A ↔ R path through U.

Truth table
-----------
C1 (expose_z=T, pi_b_known=T):   always id.
  Z blocks every backdoor; IPW with known π_b identifies the query.

C2 (expose_z=T, pi_b_known=F):
  – beh_depends_on_u=F or alpha_conf=0 → id.   Z still blocks the backdoor.
  – beh_depends_on_u=T and alpha_conf>0 → partial_id.
    A bidirected A↔R edge appears (U drives both A and R); the causal effect
    is not point-identified, but the Bareinboim natural bounds apply.

C3 (expose_z=F, pi_b_known=T):   always partial_id.
  Z is hidden; IPW over the marginal P(A|S) is biased by Z, but bounds via
  marginalisation over Z are available.

C4 (expose_z=F, pi_b_known=F):
  – beh_depends_on_u=F or alpha_conf=0 → partial_id.   Missing Z still
    leaves bounds, but no additional A↔R confounder.
  – beh_depends_on_u=T and alpha_conf>0 → non_id.
    Both Z is hidden AND there is an unmeasured A↔R confounder through U;
    no observational distribution pins down the interventional one.
"""

from __future__ import annotations

from causal_rl.envs.cell_config import CELL_CONFIGS

# Ordered values for numeric encoding in regressions.
ID_STATUS_ORDER: dict[str, int] = {"id": 0, "partial_id": 1, "non_id": 2}


def get_id_status(
    env_name: str,
    cell: int,
    alpha_conf: float = 0.0,
    beh_depends_on_u: bool = False,
) -> str:
    """Return the identifiability status for a (cell, runtime-knobs) combination.

    Parameters
    ----------
    env_name:
        Environment identifier (used only for forward-compatibility; the
        rules are the same for both supported envs).
    cell:
        Cell number 1–4 as defined in ``CELL_CONFIGS``.
    alpha_conf:
        Confounding strength passed to the environment (``>0`` means U
        actively distorts the observational reward distribution).
    beh_depends_on_u:
        ``True`` when the behaviour policy conditions on U, creating a
        spurious A↔R correlation through the hidden confounder.

    Returns
    -------
    ``"id"``, ``"partial_id"``, or ``"non_id"``.  Falls back to ``"non_id"``
    for unregistered cell numbers (fail-safe rather than silent).
    """
    cfg = CELL_CONFIGS.get(cell)
    if cfg is None:
        return "non_id"

    confounded = beh_depends_on_u and alpha_conf > 0.0

    if cfg.expose_z and cfg.pi_b_known:  # C1
        return "id"

    if cfg.expose_z and not cfg.pi_b_known:  # C2
        return "partial_id" if confounded else "id"

    if not cfg.expose_z and cfg.pi_b_known:  # C3
        return "partial_id"

    # C4: expose_z=False, pi_b_known=False
    return "non_id" if confounded else "partial_id"


def get_id_status_ordinal(
    env_name: str,
    cell: int,
    alpha_conf: float = 0.0,
    beh_depends_on_u: bool = False,
) -> int:
    """Return an ordinal encoding: id=0, partial_id=1, non_id=2."""
    return ID_STATUS_ORDER.get(get_id_status(env_name, cell, alpha_conf, beh_depends_on_u), 2)
