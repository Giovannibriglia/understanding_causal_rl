# Behaviour Policies

## Uniform (baseline)

Samples each action with equal probability. This is the reference for ideal coverage and no strategic data collection bias.

## Reward-aligned (Family 1)

Uses a coarse baseline value model:
\[
\pi_b(a \mid s) \propto \exp(\beta \cdot V_{\text{base}}(s,a)).
\]
`beta=0` recovers uniform. In cells 7–8 the score can depend on latent `U`.

### Reward-misaligned (nocive variant)

Uses the same family template but with anti-aligned/misspecified scores and a risky-action bias term:
\[
\pi_b(a \mid s) \propto \exp(\beta \cdot \tilde V(s,a)).
\]
This policy is intentionally harmful for benchmark stress-testing and can also depend on latent `U` in cells 7–8.

## Information-seeking (Family 2)

Uses predictive entropy under a frozen dynamics model:
\[
\pi_b(a \mid s) \propto \exp(\gamma \cdot H(\hat P(S' \mid S=s, A=a))).
\]
Higher `gamma` emphasises high-uncertainty actions. Cells 7–8 can condition entropy on `U`.

### Certainty-seeking (nocive variant)

Flips the information objective to prefer low-entropy transitions:
\[
\pi_b(a \mid s) \propto \exp(-\gamma \cdot H(\hat P(S' \mid S=s, A=a))).
\]
This reduces action coverage and typically worsens offline generalisation.

## Curiosity (Family 3)

Uses intrinsic surprise under an online forward model:
\[
\pi_b(a \mid s) \propto \exp(\delta \cdot \|\hat S' - S'\|^2).
\]
Tabular mode uses pseudocount novelty instead of Euclidean prediction error.

### Novelty-trap (nocive variant)

Starts from curiosity but adds a risk-seeking bias that over-selects unstable/high-risk actions. This policy is designed to produce lower return and larger train-eval mismatch under confounding.
