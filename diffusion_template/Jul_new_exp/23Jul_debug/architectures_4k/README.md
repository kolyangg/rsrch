# 4k architecture cards

All six arms keep the exact NN3a_new1 branched-attention forward and differ
only in dataset profile, optimizer membership/rates, and loss. The full
resolved definitions are in `architecture_registry.json`; the experiment
matrix and rationale are in `4K_EXPERIMENT_PLAN_AND_SCHEDULE.md`.

- `L4_O1` / `L4_C1`: projection split, masked alternating.
- `L4_O2` / `L4_C2`: projection split, always-anchored 80/20 loss.
- `L4_O3` / `L4_C3`: reference V only, Q/K/noise frozen, anchored loss.
