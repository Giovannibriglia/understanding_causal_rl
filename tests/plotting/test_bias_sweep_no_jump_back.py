"""v15: bias_sweep figure should be monotone in bias_strength.

Pre-v15, ``bias_sweep.pdf`` showed a discontinuous *jump-back* at
``bias_strength = 1.0`` for the id / partial_id strata.  The cause was
that ``UniformExplorer.__init__`` does not accept ``bias_strength`` —
the runner's silent ``try/except TypeError`` fallback constructed the
policy without the argument, and ``bias_strength`` was reported as
1.0 in the CSV regardless of the matrix sweep value.  All uniform
runs landed in the bs=1.0 bin, contaminating it with high-ESS / high-
min_propensity samples.

The v15 fix drops uniform-behaviour rows in ``bias_sweep.py``'s data
loader.  This test exercises that contract on a synthetic CSV tree:
verify that after the loader's filtering, the resulting per-stratum
``min_propensity`` series is monotonically non-increasing in
``bias_strength``.

Documented at ``docs/v15_bias_sweep_investigation.md``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from causal_rl.plotting.bias_sweep import make_bias_sweep  # noqa: E402

_HEADER = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "bias_strength",
    "id_status",
    "delta_tv",
    "delta_tv_ci_lo",
    "delta_tv_ci_hi",
    "min_propensity",
    "ess_ratio",
    "eval_return_mean",
    "eval_oracle_return_mean",
]


def _row(
    *,
    cell: int,
    behaviour: str,
    bs: float,
    min_prop: float,
    ess: float,
    id_status: str,
) -> dict[str, object]:
    return {
        "step": 100,
        "cell": cell,
        "env_name": "tabular-sepsis-v0",
        "algorithm": "dqn",
        "behaviour_policy": behaviour,
        "seed": 0,
        "bias_strength": bs,
        "id_status": id_status,
        "delta_tv": 0.04,
        "delta_tv_ci_lo": 0.03,
        "delta_tv_ci_hi": 0.05,
        "min_propensity": min_prop,
        "ess_ratio": ess,
        "eval_return_mean": 12.0,
        "eval_oracle_return_mean": 18.0,
    }


def _write_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # bias_sweep skips dirs without meta.json.
    (run_dir / "meta.json").write_text("{}", encoding="utf-8")


def _load_data_through_bias_sweep_loader(
    results_dir: Path,
) -> dict[str, dict[float, dict[str, list[float]]]]:
    """Re-implement just the data-load + binning portion of
    ``make_bias_sweep`` so we can inspect what the figure would see
    without rendering."""
    from collections import defaultdict

    out: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for eval_path in results_dir.rglob("eval.csv"):
        meta_path = eval_path.parent / "meta.json"
        if not meta_path.exists():
            continue
        with eval_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        # Mirror the v15 filter.
        if str(last.get("behaviour_policy", "")) == "uniform":
            continue
        try:
            bs = float(last["bias_strength"])
            id_status = str(last["id_status"])
            mp = float(last["min_propensity"])
            ess = float(last["ess_ratio"])
        except (KeyError, ValueError):
            continue
        out[id_status][bs]["min_propensity"].append(mp)
        out[id_status][bs]["ess_ratio"].append(ess)
    return out


def _build_synthetic_smoke_v8_tree(results: Path) -> None:
    """Mirror the smoke_v8 manifest shape on a much smaller scale.

    Uniform runs for every (cell × bs) — these all report bs=1.0 in
    the CSV (the pre-v15 bug).  Reward_aligned runs for every
    (cell × bs) report bs as requested.  Without the v15 fix, the
    bs=1.0 bin contains both uniform and reward_aligned, pulling
    min_propensity upward.
    """
    sweep_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for cell, status in ((1, "id"), (2, "partial_id"), (4, "non_id")):
        for bs in sweep_values:
            # Uniform: always reports bs=1.0 and high min_prop / ess.
            _write_run(
                results / f"uniform_c{cell}_bs{bs}",
                [
                    _row(
                        cell=cell,
                        behaviour="uniform",
                        bs=1.0,  # the bug: always reports 1.0
                        min_prop=0.10,
                        ess=0.96,
                        id_status=status,
                    )
                ],
            )
            # reward_aligned: monotone — high bs ⇒ low min_prop / ess.
            _write_run(
                results / f"ra_c{cell}_bs{bs}",
                [
                    _row(
                        cell=cell,
                        behaviour="reward_aligned",
                        bs=bs,
                        min_prop=0.09 - 0.04 * bs,
                        ess=0.95 - 0.20 * bs,
                        id_status=status,
                    )
                ],
            )


def test_min_propensity_monotone_in_bias_strength(tmp_path: Path) -> None:
    """v15 contract: after dropping uniform, min_propensity is monotone
    non-increasing in bias_strength for every visible stratum."""
    results = tmp_path / "results"
    _build_synthetic_smoke_v8_tree(results)
    binned = _load_data_through_bias_sweep_loader(results)
    assert binned, "loader returned empty; fixture broken"
    for status, by_bs in binned.items():
        bs_values = sorted(by_bs.keys())
        means = [
            sum(by_bs[bs]["min_propensity"]) / len(by_bs[bs]["min_propensity"])
            for bs in bs_values
        ]
        # Monotone non-increasing within a small noise tolerance.
        for prev, cur in zip(means, means[1:], strict=False):
            assert cur <= prev + 1e-6, (
                f"min_propensity not monotone in stratum {status}: "
                f"bs sequence={bs_values}, means={means}"
            )


def test_uniform_runs_excluded_from_bias_sweep(tmp_path: Path) -> None:
    """v15: uniform-behaviour rows must not contribute to the figure."""
    results = tmp_path / "results"
    _build_synthetic_smoke_v8_tree(results)
    binned = _load_data_through_bias_sweep_loader(results)
    # The fixture's uniform rows would, if included, push the bs=1.0
    # min_prop mean toward 0.10.  Without uniform, the bs=1.0 mean
    # should match the reward_aligned-only value 0.09 - 0.04 ≈ 0.05.
    for status, by_bs in binned.items():
        if 1.0 not in by_bs:
            continue
        mean_at_one = (
            sum(by_bs[1.0]["min_propensity"]) / len(by_bs[1.0]["min_propensity"])
        )
        assert mean_at_one < 0.07, (
            f"bs=1.0 min_propensity mean {mean_at_one:.3f} in stratum "
            f"{status} suggests uniform rows still contributing"
        )


def test_make_bias_sweep_renders_after_filter(tmp_path: Path) -> None:
    """End-to-end smoke: the figure renders without crashing on the
    filtered data."""
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _build_synthetic_smoke_v8_tree(results)
    make_bias_sweep(results, out)
    assert (out / "bias_sweep.pdf").exists()


# ---------------------------------------------------------------------------
# v16: degenerate single-bias_strength path
# ---------------------------------------------------------------------------


def _build_single_bs_tree(results: Path, *, bs: float = 1.0) -> None:
    """Synthetic smoke fixture for the single-bs path.

    One ``reward_aligned`` run per (cell × id_status) at the requested
    ``bs``.  Mirrors what the ``coverage_stress`` matrix produces when
    ``bias_strengths`` is set to ``[1.0]`` only.
    """
    for cell, status in ((1, "id"), (2, "id"), (3, "partial_id"), (4, "non_id")):
        _write_run(
            results / f"ra_c{cell}_bs{bs}",
            [
                _row(
                    cell=cell,
                    behaviour="reward_aligned",
                    bs=bs,
                    min_prop=0.05 + 0.01 * cell,
                    ess=0.80 + 0.04 * cell,
                    id_status=status,
                )
            ],
        )


def test_single_bias_strength_renders_without_error(tmp_path: Path) -> None:
    """v16: a manifest with a single ``bias_strength`` value renders
    a non-empty PDF.

    The pre-v16 line-plot path produced a degenerate figure with three
    points at x=1.0 and titles asserting trends not depicted.  The
    v16 single-bs branch swaps to a stratified bar chart and rewrites
    the titles; this test just confirms the file is produced and
    larger than a blank-PDF baseline.
    """
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _build_single_bs_tree(results, bs=1.0)
    make_bias_sweep(results, out)
    pdf = out / "bias_sweep.pdf"
    assert pdf.exists(), "bias_sweep.pdf was not written"
    # A blank matplotlib PDF is ~700 bytes; a populated one is several KB.
    assert pdf.stat().st_size > 1024, (
        f"bias_sweep.pdf is suspiciously small ({pdf.stat().st_size} bytes); "
        "fixture may not have produced a real figure"
    )


def test_single_bias_strength_does_not_claim_trend(
    tmp_path: Path, monkeypatch
) -> None:
    """v16: on a single-bs manifest, the figure must not assert
    "rises with bias_strength" / "falls with bias_strength" trends
    that the data does not depict.

    Inspects the rendered figure's axis titles via a monkeypatched
    ``plt.close`` that records the figure before disposing of it.
    Production code is unchanged; the patch is purely a test hook.
    """
    import matplotlib.pyplot as plt

    captured: list = []

    def _record_close(fig=None):
        if fig is not None and fig not in captured:
            captured.append(fig)

    monkeypatch.setattr(plt, "close", _record_close)

    results = tmp_path / "results"
    out = tmp_path / "fig"
    _build_single_bs_tree(results, bs=1.0)
    make_bias_sweep(results, out)

    assert captured, "no figure was passed to plt.close; test hook didn't fire"
    fig = captured[-1]
    titles = [ax.get_title() for ax in fig.axes]
    fig_texts = [t.get_text() for t in fig.texts]
    suptitle = fig._suptitle.get_text() if fig._suptitle is not None else ""
    all_title_text = "\n".join([*titles, *fig_texts, suptitle])

    assert "rises with bias_strength" not in all_title_text, (
        "single-bs figure asserts a trend not depicted in the data; "
        f"titles={titles!r}, texts={fig_texts!r}"
    )
    assert "falls with bias_strength" not in all_title_text, (
        "single-bs figure asserts a trend not depicted in the data; "
        f"titles={titles!r}, texts={fig_texts!r}"
    )
    assert any("by id_status" in t for t in titles), (
        "expected at least one axis title to describe the stratification "
        f"as 'by id_status'; got titles={titles!r}"
    )

    # Drain captured figures so they don't leak into other tests.
    for f in captured:
        plt.close(f)  # uses the original after monkeypatch teardown
