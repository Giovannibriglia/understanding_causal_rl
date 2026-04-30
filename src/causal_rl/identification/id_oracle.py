"""Graphical identifiability oracle for the 8-cell factorial.

Classifies the target query P(R | do(A), S) as one of:
  ``id``         – point-identifiable from the observational margin in this cell.
  ``partial_id`` – partially identifiable: bounds can be computed but not the
                   exact value.
  ``non_id``     – non-identifiable: the query is not recoverable from the
                   observational distribution alone for any sample size.

The classification is a hand-coded mapping for the eight cells used in the paper,
verified against the standard identification criteria (Pearl's do-calculus / the
ID algorithm):

  - A query is identifiable when the causal diagram has no unblocked backdoor
    path from A to R through hidden variables, OR when such paths are blocked by
    observed covariates.
  - With known π_b and no hidden confounders on A→R, IPW identifies the query.
  - With unknown π_b and a hidden confounder (U) creating a bidirected A↔R edge,
    the query is not point-identifiable (partial or non-id depending on whether
    observable covariates bound the effect).
"""

from __future__ import annotations

# (env_name, cell) → id_status
# env_name is a prefix match (both envs share the same structure).
ID_STATUS_MAP: dict[tuple[str, int], str] = {
    # On-policy cells (cells 1-2): no off-policy confounding; ID depends on Z.
    ("tabular-sepsis-v0", 1): "id",  # Z observed, π_b = π_e → no gap
    ("tabular-sepsis-v0", 2): "partial_id",  # Z hidden → partial bounds
    ("tabular-sepsis-v0", 3): "id",  # Z observed, π_b known → IPW identifies
    ("tabular-sepsis-v0", 4): "partial_id",  # Z hidden, π_b known → bounds, IPW biased
    ("tabular-sepsis-v0", 5): "partial_id",  # Z observed, π_b unknown → selection bias
    ("tabular-sepsis-v0", 6): "non_id",  # Z hidden, π_b unknown, U hidden → non-id
    ("tabular-sepsis-v0", 7): "partial_id",  # U observed but bidirected A↔R from unknown π_b
    ("tabular-sepsis-v0", 8): "non_id",  # Z hidden + U exposed does not recover Z-gap
    ("continuous-ward-v0", 1): "id",
    ("continuous-ward-v0", 2): "partial_id",
    ("continuous-ward-v0", 3): "id",
    ("continuous-ward-v0", 4): "partial_id",
    ("continuous-ward-v0", 5): "partial_id",
    ("continuous-ward-v0", 6): "non_id",
    ("continuous-ward-v0", 7): "partial_id",
    ("continuous-ward-v0", 8): "non_id",
}

# Ordered values for numeric encoding in regressions.
ID_STATUS_ORDER = {"id": 0, "partial_id": 1, "non_id": 2}


def get_id_status(env_name: str, cell: int) -> str:
    """Return the identifiability status string for ``(env_name, cell)``.

    Falls back to ``"non_id"`` for unregistered combinations so that new
    cells fail conservatively rather than silently.
    """
    key = (env_name, cell)
    if key in ID_STATUS_MAP:
        return ID_STATUS_MAP[key]
    # Try prefix match (e.g. "tabular-sepsis-v0-modified" → "tabular-sepsis-v0")
    for (ename, ecell), status in ID_STATUS_MAP.items():
        if env_name.startswith(ename) and ecell == cell:
            return status
    return "non_id"


def get_id_status_ordinal(env_name: str, cell: int) -> int:
    """Return an ordinal encoding: id=0, partial_id=1, non_id=2."""
    return ID_STATUS_ORDER.get(get_id_status(env_name, cell), 2)
