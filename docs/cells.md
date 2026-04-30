# Eight Cells

Each cell corresponds to a tuple `(expose_Z, expose_U, pi_b_known, on_policy)`:

1. `(1,0,1,1)` fully observed on-policy baseline.
2. `(0,0,1,1)` hidden mediators in on-policy data.
3. `(1,0,1,0)` perfect offline archive.
4. `(0,0,1,0)` hidden mediators offline, known behaviour.
5. `(1,0,0,0)` behaviour unknown offline.
6. `(0,0,0,0)` hidden mediators + unknown behaviour.
7. `(1,1,0,0)` latent confounding in offline data.
8. `(0,1,0,0)` hidden mediators + latent confounding.

Worked examples:

- Cell 1: PPO on tabular sepsis with uniform behaviour gives near-zero TV gap under long training.
- Cell 2: Same setup but hidden `Z` increases generalisation gap.
- Cell 3: CQL with known `pi_b` converges with low bias.
- Cell 4: CQL must compensate for hidden `Z`; TV gap rises.
- Cell 5: Unknown `pi_b` weakens off-policy correction.
- Cell 6: Unknown `pi_b` and hidden `Z` compound errors.
- Cell 7: Latent `U` drives action selection and confounding-aware methods become necessary.
- Cell 8: Hardest regime with both hidden mediators and latent confounding.
