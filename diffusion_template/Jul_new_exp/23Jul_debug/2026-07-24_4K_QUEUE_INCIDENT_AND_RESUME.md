# 4k queue incident and resume

## User-visible symptom

The GPU was idle from approximately 2026-07-24 03:35 UTC until 08:30 UTC.
The queue had stopped, even though its two training pairs had produced valid
checkpoints and validation artifacts.

## Completed training before the resume

Four runs completed all 4,000 optimizer steps:

- priority 1: `L4_O1_oneid_projection_alt` and
  `L4_C1_large_projection_alt`;
- priority 4: `L4_O4_oneid_projection_schedule` and
  `L4_C4_large_projection_schedule`.

Each run has eight checkpoints and four canonical images plus similarity,
landmark, and text-CLIP metrics at all nine stages from step 0 through 4000.
Priority 4 also created its local PDF before the queue stopped.

## Causes

1. Priority 1 training completed, but both arm wrappers returned status 2
   because the already-running watcher read a partially changed shell source
   and reported an unmatched quote. The current watcher source passes
   `bash -n`; this was a live-file read race, not lost training.
2. Priority 4 training and local report generation completed, but its final
   Comet-unity audit saw the same immutable Comet experiment key at a page
   boundary twice. It interpreted two API observations of one key as two
   separate experiments and failed closed.
3. Both schedulers intentionally stop on any nonzero arm status, so no later
   pair launched.

## Repairs

- `audit_comet_unity.py` now deduplicates exact-name observations by immutable
  Comet key. Two different keys with the same run name still fail the audit.
- `repair_completed_4k_reports.sh` regenerates any missing local report,
  reruns the strict Comet audit, and uploads the report to the original
  training experiment for all four completed runs.
- `schedule_4k_resume_missing.sh` contains only the genuinely missing priority
  pairs: 2, 3, then 5 through 13. It checks for a completed epoch-8 checkpoint
  before launching each pair, preventing accidental retraining.

## Resume

Priority pair 2 (`projection_blend20`) started at
2026-07-24 08:30:49 UTC. Its OneID and CosmicLarge arms are running
concurrently. Priority 3 is next, followed by priorities 5 through 13.

Live scheduler state is in `scheduler_4k/QUEUE_RESUME_STATUS.md`; per-run logs
and the CPU repair status remain in `scheduler_4k/`.
