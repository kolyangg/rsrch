# Subject-v2 recovery and priority backfill

This eight-GPU wave recovers the three unfinished runs from failed job
`lm-mpi-job-44b99a20-a6ad-4023-b3c6-f249b1abe83d` and starts CL10-CL14 ahead
of E14-E22.

| Worker | Chain |
|---:|---|
| 0 | CL6 resume |
| 1 | BC_E13_ds3 resume |
| 2 | CL8 |
| 3 | CL10 r2 → E14 r6 → E19 r2 |
| 4 | CL11 r1 → E15 r2 → E20 r2 |
| 5 | CL12 r1 → E16 r2 → E21 r2 |
| 6 | CL13 r1 → E17 r5 → E22 r2 |
| 7 | CL14 r1 → E18 r4 |

All workers share the original staging root. Completed runs return immediately;
complete checkpoint manifests are hash-validated and reused. Incomplete step
directories are moved under `incomplete_recovery/` before being regenerated.
Idempotent Comet reads and downloads use eight bounded attempts with exponential
backoff and atomic destination replacement.

No Comet mutation occurs until every checkpoint of a run is staged. Every
written run must finish with `replacement_verified.json` after downloading and
hash-checking the replacement assets from the original immutable Comet ID.
