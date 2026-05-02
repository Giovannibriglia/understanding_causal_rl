from __future__ import annotations


def checkpoint_ticks(total: int, n_checkpoints: int) -> set[int]:
    """Return the set of ticks at which to log, including 1 and ``total``.

    Distributes ``n_checkpoints - 2`` intermediate ticks uniformly between 1
    and ``total`` (inclusive).  Always returns at least ``{1, total}``; clamps
    requests below 2.  Intermediate ticks are also clamped into ``[2, total-1]``
    to prevent collisions with the first and last entries.

    Args:
        total: The final tick (must be >= 1).
        n_checkpoints: Requested number of checkpoints.  Values < 2 are
            promoted to 2.

    Returns:
        The set of integer ticks at which to log.
    """
    if total < 1:
        return {1}
    if n_checkpoints < 2:
        n_checkpoints = 2
    if total == 1 or n_checkpoints == 2:
        return {1, total}
    inner: set[int] = set()
    for i in range(1, n_checkpoints - 1):
        frac = i / (n_checkpoints - 1)
        tick = int(round(frac * total))
        tick = max(2, min(total - 1, tick))
        inner.add(tick)
    return {1, total, *inner}
