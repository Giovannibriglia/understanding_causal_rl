# Reading results

- Per-run files are written under `results/<config>_<timestamp>/...`.
- Core files:
  - `train.csv`
  - `eval.csv`
  - `eval_perturbed.csv` (if perturbation evaluation is on)
  - `meta.json`
- Aggregated figures are written under `outputs/<config>_<timestamp>/`.

For figure semantics, see `docs/figures.md`.
For metric details, see `docs/metric.md`.
