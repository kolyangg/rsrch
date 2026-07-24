# E03 staged up1 → up0

Purpose: learn local face detail before allowing limited coarse shape
adaptation.

- Forward: exact NN3a_new1 throughout.
- Steps 0–99: only up1 optimizer groups have nonzero LR.
- Steps 100–600: up1 stays at full LR; up0 is enabled at `0.35×`.
- Reference Q/K/V scale: `1.0`.
- Noise Q/K/V scale: `0.15`.
- Loss: E00 `masked_alternating`.

The stage is implemented in the experiment-local trainer and is based on the
global optimizer step, so checkpoint resume preserves the schedule. Every
transition logs the exact enabled optimizer group names.

Re-run:

```bash
./run_architecture.sh E03_staged_up1_up0
```
