# E04 projection-split active-up route

Purpose: reduce coordinate distortion by limiting up0/reference-query/noise
movement while accelerating the direct reference K/V content path.

- Forward: identical to E00.
- Trainable scope: active up blocks only.
- Reference K/V LR: `1.5 × 5e-5`.
- Reference Q LR: `0.5 × 5e-5`.
- Noise Q/K/V LR: `0.1 × 5e-5`.
- Additional up0 multiplier: `0.35`.
- Loss: E00 `masked_alternating`.

Re-run:

```bash
./run_architecture.sh E04_projection_split
```

This arm is deliberately downstream of E01/E02: it tests whether the useful
reference-value route can learn without allowing coarse queries and the
ordinary-noise clone to dominate geometry.
