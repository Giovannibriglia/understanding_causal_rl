"""Per-cell causal diagrams for the 4-cell factorial.

Each diagram is a directed acyclic graph (DAG) plus bidirected edges that
represent unobserved common causes (hidden confounders).  The ``observed``
set records which nodes are in the agent's observation at each cell.

Node vocabulary (shared across envs):
  S_t     – observable state at time t
  A_t     – action at time t
  R_t     – reward at time t
  S_tp1   – next observable state S_{t+1}
  Z       – latent mediator / treatment modifier (time-invariant or slow)
  U       – unobserved confounder (affects reward; may also affect action
            selection depending on the behaviour policy)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalDiagram:
    """Minimal representation of a causal diagram for one (env, cell) pair."""

    nodes: frozenset[str]
    directed: frozenset[tuple[str, str]]
    # Bidirected edges represent unobserved common causes
    bidirected: frozenset[tuple[str, str]]
    # Nodes in the agent's observation at this cell
    observed: frozenset[str]


def _base_directed() -> frozenset[tuple[str, str]]:
    """Edges shared by all cells."""
    return frozenset(
        {
            ("S_t", "A_t"),
            ("S_t", "R_t"),
            ("A_t", "R_t"),
            ("S_t", "S_tp1"),
            ("A_t", "S_tp1"),
            ("Z", "S_tp1"),
            ("Z", "R_t"),
            ("U", "R_t"),
        }
    )


def _make_diagram(expose_z: bool, pi_b_known: bool) -> CausalDiagram:
    nodes = frozenset({"S_t", "A_t", "R_t", "S_tp1", "Z", "U"})
    directed = _base_directed()

    # When π_b is unknown the behaviour policy may depend on U, creating a
    # spurious A↔R path.  We record this as a bidirected edge in the ADMG.
    if not pi_b_known:
        bidirected: frozenset[tuple[str, str]] = frozenset({("A_t", "R_t")})
    else:
        bidirected = frozenset()

    obs_nodes = {"S_t", "A_t", "R_t"}
    if expose_z:
        obs_nodes.add("Z")

    return CausalDiagram(
        nodes=nodes,
        directed=directed,
        bidirected=bidirected,
        observed=frozenset(obs_nodes),
    )


# Pre-built diagrams indexed by cell number (same structural SCM for both envs).
_DIAGRAMS: dict[int, CausalDiagram] = {
    1: _make_diagram(expose_z=True, pi_b_known=True),    # C1: mdp_known
    2: _make_diagram(expose_z=True, pi_b_known=False),   # C2: mdp_unknown
    3: _make_diagram(expose_z=False, pi_b_known=True),   # C3: pomdp_known
    4: _make_diagram(expose_z=False, pi_b_known=False),  # C4: pomdp_unknown
}


def get_diagram(env_name: str, cell: int) -> CausalDiagram:
    """Return the causal diagram for the given (env_name, cell) pair."""
    if cell not in _DIAGRAMS:
        msg = f"No diagram registered for cell {cell}"
        raise KeyError(msg)
    return _DIAGRAMS[cell]
