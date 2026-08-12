# Subject-v2 historical validation backfill: 11 runs / 5 A100 workers

This is a single Serv/MLS submission. `workers: 5` allocates one A100 on the
binary job's mpimaster and on four mpiworker nodes. The dispatcher starts one
process on each allocated node; an atomic NFS claim assigns one non-overlapping
sequential chain. A Comet run is never split across workers because its metric
histories and selected assets are replaced as one replay-gated transaction.

| Worker | Sequential allocation | Saved checkpoints | Initial chain ETA |
|---:|---|---:|---:|
| 0 | E13, CL4, CL6 | 36 | 12 h |
| 1 | BC_E13, CL5 | 24 | 8 h |
| 2 | BC_E13_ds1, CL7 | 24 | 8 h |
| 3 | BC_E13_ds3, CL8 | 24 | 8 h |
| 4 | CL9, BC_E13_ds2 | 24 | 8 h |

The estimate starts conservatively at 20 minutes per checkpoint and updates
from observed checkpoint durations. The measured Eddie contract-v2 run took
about 11.7 minutes per checkpoint before the additional full-96 rescoring used
here, so expected wall time is approximately 9–12 hours after scheduling.

For every run, the launcher:

1. verifies the immutable Comet key, exact 12 weights checkpoints, subject-v2
   artifacts, legacy embedding, sealed manual bbox seed, and each immutable
   runtime's actual derived `pm96_bboxes_new_auto.json` generation map;
2. runs the tool in dry-run mode over every safe 2k–24k checkpoint;
3. requires exactly all 12 validation steps in the staging manifest;
4. reruns with `--reuse-staging --write`, replacing Eddie images and rescoring
   the merged full-96 panel (including the corrected active-box Chef/Lex score);
5. requires the final Comet replacement audit and all-step hash verification.

The active generation maps are intentionally run-specific: E13/BC_E13-family
snapshots use SHA-256 `4db6344d...`, while CL4-CL9 snapshots use
`b33cf026...`. The canonical manual seed is `a39645e2...`; passing that seed as
the active map changes all 12 Eddie outputs and is rejected by the exact replay
gate.

Rolling `BACKFILL_ETA` messages report the current checkpoint and remaining
time for its run. `BACKFILL_CHAIN_PLAN` reports the full worker-chain estimate.
Per-worker logs and status files are written below:

```text
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/logs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/status/
```

Submit the deployed YAML once:

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/run_subject_v2_historical_backfill_11runs_5workers.yaml \
  --comment "Replay-gated subject-v2 historical validation replacement for E13, BC_E13/ds1-ds3 and CL4-CL9 using five A100 workers under the scoped ten-GPU exception"
```

The reusable tool itself is documented in `TOOLS.md` and
`tools/comet/README.md`; this package is the fixed audited allocation for the
requested historical experiments.
