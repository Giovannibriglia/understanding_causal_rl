# v15 — bias_sweep jump-back at bs=1.0: investigation

## Symptom

`bias_sweep.pdf` panels 2 (ESS / N) and 3 (min propensity) show a
discontinuous **jump-back** at `bias_strength = 1.0` for the `id` and
`partial_id` strata. `min_propensity` falls from ~0.092 at bs=0.0
through ~0.063 at bs=0.75, then rises back to ~0.085 at bs=1.0. Same
shape on ESS. The `non_id` stratum is monotone — it falls through
bs=1.0 without the bounce.

The v11 fix to `behaviour/base.py::_mix_with_uniform` removed the
`bias_strength >= 1.0` early return that the v11 brief blamed. Reading
the post-PR-14 source confirms the early return is gone, so v11's
explanation cannot be the current cause. Yet the jump persists.

## Hypothesis

`runner.py::_make_behaviour_with_bias` silently falls back to
constructing the behaviour policy *without* `bias_strength` when the
constructor doesn't accept it:

```python
def _make_behaviour_with_bias(self, name: str, **kwargs: object) -> object:
    try:
        return make_behaviour(name, bias_strength=self.config.bias_strength, **kwargs)
    except TypeError:
        return make_behaviour(name, **kwargs)
```

`UniformExplorer.__init__` does **not** accept `bias_strength` and
unconditionally calls `super().__init__(bias_strength=1.0)`. The eval-row
writers at `runner.py:301` and `runner.py:389` then read
`getattr(self.behaviour_policy, "bias_strength", 1.0)` — which returns
1.0 for every uniform run, regardless of the matrix sweep value.

`bias_sweep.py:49` groups by the CSV's `bias_strength` column. So all
uniform runs land in the bs=1.0 bin regardless of the requested sweep
value, while `reward_aligned` and `reward_misaligned` runs (which *do*
accept `bias_strength`) populate the bs=0.0/0.25/0.5/0.75 bins normally.

The result: the bs=1.0 bin is the only bucket that contains uniform
runs. Uniform produces high ESS / high min_propensity by construction,
so its presence in the 1.0 bucket pulls the mean back up — the
"jump-back". The `non_id` stratum is monotone because non_id runs
require `beh_depends_on_u=True`, which uniform doesn't satisfy
(`UniformExplorer.depends_on_u = False`), so non_id's bs=1.0 bin
contains no uniform runs.

## Verification

Three steps from the brief.

### Step 1 — uniform falls back; reward_aligned does not

```pycon
>>> from causal_rl.behaviour.registry import make as make_behaviour
>>> make_behaviour('uniform', n_actions=8, bias_strength=0.25)
TypeError: UniformExplorer.__init__() got an unexpected keyword argument 'bias_strength'
>>> make_behaviour('uniform', n_actions=8).bias_strength
1.0
>>> make_behaviour('reward_aligned', n_actions=8, bias_strength=0.25).bias_strength
0.25
```

Confirms the `try`/`except TypeError` fallback. UniformExplorer
silently drops the requested bs and reports 1.0.

### Step 2 — non-uniform behaviours record the requested bs continuously

The `reward_aligned` family accepts `bias_strength` and records the
requested value verbatim. No CSV-level discontinuity at bs=1.0 for
reward_aligned runs — what `bias_sweep.py` sees as "bs=1.0" for
reward_aligned is the actual requested value.

### Step 3 — uniform runs collapse to the bs=1.0 bin

For every uniform-behaviour run in the smoke_v8 manifest, the
``eval.csv`` ``bias_strength`` column reads 1.0 regardless of which
sweep value the matrix runner requested. `bias_sweep.py:49` therefore
buckets every uniform run into the bs=1.0 group, contaminating that
bucket with high-ESS / high-min_propensity samples that pull the mean
upward.

The non_id stratum is unaffected because uniform runs cannot reach
non_id (uniform's `depends_on_u=False`); the non_id bs=1.0 bucket
contains only `reward_aligned` and `reward_misaligned`, both of which
correctly record the swept value.

**Hypothesis confirmed.** The jump-back is a binning artefact, not a
real metric phenomenon.

## Alternatives ruled out

* **v11 early-return regression** — verified gone in the post-PR-14
  source; the function flows through the mixing path at bs=1.0 as
  designed.
* **Coverage-restriction interaction** — coverage_stress.yaml is a
  separate config; smoke_v8 doesn't set ``forbidden_actions`` so the
  collector never invokes the v11 mask.
* **Numerical instability at bs=1.0** — the mixing coin sampler uses
  ``torch.rand < bs``; at bs=1.0 every coin draws 1, identical
  semantics to the limit of bs→1 from below. No numerical edge case.

## Fix

**Fix A (chosen).** Drop uniform from the `bias_sweep` figure at the
data-loading stage. One-line patch in `bias_sweep.py:49`. Add a caption
note explaining the omission.

Justification:
* Uniform doesn't have a meaningful bias_strength axis — the policy is
  uniform regardless of bs. Plotting it against a swept bs is a
  category error.
* The figure's claim is "bias_strength sweep changes coverage": that
  claim only applies to behaviour policies that *have* a bias-strength
  knob.
* One-line change, minimal blast radius.

Fix B (record `bias_strength_requested` separately and group on it)
was considered. It's the more correct structural fix, but it requires
a schema change, a runner change, a re-rendering, and removes the
informative TypeError signal. The silent `try/except TypeError` in
`_make_behaviour_with_bias` is itself a code smell, but fixing that
properly is a separate concern from the bias_sweep figure correctness.
Fix A unblocks the figure today; Fix B (cleaning up the silent
fallback) is left for a future PR if needed.

## Acceptance

* `tests/plotting/test_bias_sweep_no_jump_back.py` asserts that
  `min_propensity` is monotonically non-increasing in `bias_strength`
  for all visible strata after the fix.
* Re-rendering `bias_sweep.py` on the existing smoke_v8 results (no
  re-run needed) produces a figure where the jump-back disappears.
