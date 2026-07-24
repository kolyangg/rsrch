# E12 projection-split plus schedule matching

Purpose: add only the audited train/inference timestep correction to E04,
without changing its optimizer routing.

- Forward and inference: exact NN3a_new1.
- Optimizer groups and LR multipliers: exact E04 projection split.
- Training timestep mode: `inference_ba_region`.
- With the canonical 15/50 BA schedule, training samples diffusion timesteps
  `0..699` rather than `0..999`.
- Loss: E04/E00 `masked_alternating`.

Re-run:

```bash
./run_architecture.sh E12_projection_split_schedule
```

Run this only if E04 preserves geometry better than E01/E02. The comparison
with E04 then isolates whether avoiding BA-inactive noisy timesteps improves
learning efficiency and checkpoint stability.
