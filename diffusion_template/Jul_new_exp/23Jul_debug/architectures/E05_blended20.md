# E05 always-anchored blended loss

Purpose: isolate the effect of replacing heterogeneous alternating face/full
updates with a preservation term on every optimizer step.

- Forward and trainable scope: exact E00.
- Loss: `0.80 * full_image_MSE + 0.20 * face_MSE` on every step.
- Reference/noise LR: unchanged from E00.

Re-run:

```bash
./run_architecture.sh E05_blended20
```

Promotion signal: less checkpoint-to-checkpoint geometry drift and lower
outside/ring error without erasing the BA-vs-PhotoMaker face difference.
