# E13 projection-split plus always-anchored loss

Purpose: add only the loss-preservation mechanism to E04, keeping its
projection routing and LR multipliers unchanged.

- Forward and inference: exact NN3a_new1.
- Optimizer groups and LR multipliers: exact E04 projection split.
- Loss: `0.80 * full_image_MSE + 0.20 * face_MSE` on every step.

Re-run:

```bash
./run_architecture.sh E13_projection_split_blended20
```

Promote over E04 only if the face remains distinctly branched-attention-like
while checkpoint-to-checkpoint geometry and non-face content are more stable.
