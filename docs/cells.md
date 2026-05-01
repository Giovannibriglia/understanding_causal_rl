# Four cells

| cell | shorthand | expose_Z | pi_b_known | typical id_status |
| --- | --- | --- | --- | --- |
| 1 | `mdp_known` | True | True | `id` |
| 2 | `mdp_unknown` | True | False | `id` (no confounder) / `partial_id` (with confounder) |
| 3 | `pomdp_known` | False | True | `partial_id` |
| 4 | `pomdp_unknown` | False | False | `partial_id` (no confounder) / `non_id` (with confounder) |

`expose_U` and on/off-policy are runtime knobs, not cell axes.
