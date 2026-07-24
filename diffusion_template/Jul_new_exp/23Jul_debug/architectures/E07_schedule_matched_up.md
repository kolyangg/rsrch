# E07 schedule-matched active-up route

Purpose: remove the audited train/inference mismatch in which NN3a updates BA
at very noisy timesteps where inference has not yet enabled BA.

- Forward and inference: exact NN3a_new1.
- Trainable scope: active up blocks.
- Training timestep mode: `inference_ba_region`.
- With the canonical 15/50 BA schedule, training samples diffusion timesteps
  `0..699` rather than `0..999`.
- Noise LR scale: `0.15`; reference LR remains `5e-5`.
- Loss: E00 `masked_alternating`.

Re-run:

```bash
./run_architecture.sh E07_schedule_matched_up
```

Expected signature: better face/detail learning per step and less coarse
layout pressure. A face improvement limited to the earlier-BA validation
stream is diagnostic, not sufficient for promotion.
