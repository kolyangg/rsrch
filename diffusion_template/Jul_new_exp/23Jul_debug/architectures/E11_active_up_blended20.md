# E11 active-up plus blended loss

Purpose: combine the two lowest-risk training-only changes if their isolated
arms show complementary benefits.

- Forward: exact NN3a_new1.
- Trainable scope: active up blocks.
- Loss: 80/20 full/face blended loss.
- Reference LR: `5e-5`.
- Noise LR scale: `0.15`.

Re-run:

```bash
./run_architecture.sh E11_active_up_blended20
```

This is a combination candidate, not a substitute for the isolated E01 and
E05 comparisons.
