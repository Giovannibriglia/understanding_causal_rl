# Behaviour Policies (4-cell setup)

## Uniform (baseline)

Samples each action with equal probability. This is the reference for ideal coverage and no strategic data collection bias.

## Reward-aligned (Family 1)

Uses a coarse baseline value model:
\[
\pi_b(a \mid s) \propto \exp(\beta \cdot V_{\text{base}}(s,a)).
\]
`beta=0` recovers uniform. Confounding appears when `beh_depends_on_u=True` and `alpha_conf>0`.

### Reward-misaligned (nocive variant)

Uses the same family template but with anti-aligned/misspecified scores and a risky-action bias term:
\[
\pi_b(a \mid s) \propto \exp(\beta \cdot \tilde V(s,a)).
\]
This policy is intentionally harmful for benchmark stress-testing and can depend on latent `U` when `beh_depends_on_u=True` and `alpha_conf>0`.

## Information-seeking (Family 2)

Uses predictive entropy under a frozen dynamics model:
\[
\pi_b(a \mid s) \propto \exp(\gamma \cdot H(\hat P(S' \mid S=s, A=a))).
\]
Higher `gamma` emphasises high-uncertainty actions.

### Certainty-seeking (nocive variant)

Flips the information objective to prefer low-entropy transitions:
\[
\pi_b(a \mid s) \propto \exp(-\gamma \cdot H(\hat P(S' \mid S=s, A=a))).
\]
This reduces action coverage and typically worsens offline robustness.

## Curiosity (Family 3)

Uses intrinsic surprise under an online forward model:
\[
\pi_b(a \mid s) \propto \exp(\delta \cdot \|\hat S' - S'\|^2).
\]
Tabular mode uses pseudocount novelty instead of Euclidean prediction error.

### Novelty-trap (nocive variant)

Starts from curiosity but adds a risk-seeking bias that over-selects unstable/high-risk actions. This policy is designed to produce lower return and larger train-eval mismatch under confounding.

## `bias_strength` interpolation

All families support `bias_strength`:
- `bias_strength=1.0`: pure family policy,
- `bias_strength=0.0`: uniform behaviour,
- in-between values: stochastic mixture between family and uniform actions.
