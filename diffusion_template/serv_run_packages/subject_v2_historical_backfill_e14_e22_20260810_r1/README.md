# Subject-v2 historical validation backfill: CL10-CL14, then E14-E22

Selected immutable completed revisions:

| Chain | Runs | Checkpoints |
|---:|---|---:|
| 0 | CL10 r2 → E14 r6 → E19 r2 | 36 |
| 1 | CL11 r1 → E15 r2 → E20 r2 | 36 |
| 2 | CL12 r1 → E16 r2 → E21 r2 | 36 |
| 3 | CL13 r1 → E17 r5 → E22 r2 | 36 |
| 4 | CL14 r1 → E18 r4 | 24 |

The first attempt to start E14 and E15 immediately as a two-GPU wave was
rejected before job creation with `PROJECT_GPU_LIMIT_REACHED_ONLY_1_FREE`.
It is not retried. The five completed CL10-CL14 runs were subsequently placed
first, one per worker, ahead of all nine E14-E22 runs in the delayed wave.

`schedule_e16_e22_after_current.sh` watches current backfill job
`lm-mpi-job-44b99a20-a6ad-4023-b3c6-f249b1abe83d`. It requires that job to
finish successfully, waits 20 minutes, recomputes this project's Running plus
Pending GPU count, and makes exactly one submission attempt when adding five
GPUs stays within the user-authorized ceiling of ten. A companion monitor
records when the delayed MLS job reaches Running and then exits.

The CL10-CL14 and E19-E22 active auto-bbox maps are sealed at SHA-256
`b33cf026...`; E14-E18 use `4db6344d...`. Each run has 12 weights checkpoints
and one immutable Comet record. The same exact-replay, dry-stage, full-96
rescore, transactional Comet replacement, and download/hash-verification gates
used by the active E13/CL/BC backfill apply here.

At measured throughput of roughly 15–20 minutes per checkpoint, CL10-CL14
should finish about 3–4 hours after the delayed wave starts. The four
three-run chains set a whole-wave estimate of roughly 9–12 hours after startup.
Startup is driven by actual current-job completion rather than a stale fixed
clock ETA.
