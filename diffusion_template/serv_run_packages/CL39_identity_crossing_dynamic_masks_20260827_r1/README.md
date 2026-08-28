# CL39 seed-specific PhotoMaker-mask crossing correction

This validation-only package supersedes the seed-1/2/3 crossing whose shared
automatic bbox cache contained only seed-0 records. Each one-A100 job now uses
an isolated `bbox_mask_gen` base path. Its A arm performs the existing
PhotoMaker-only pass, writes exactly 96 automatic boxes with the declared seed,
and saves 96 diagnostic overlays. A fail-closed InsightFace gate requires zero
missing faces, at most two unowned A-arm cells and mean fixed-box overlap of at
least 0.50. B/C/D reuse the exact accepted cache and record its SHA-256.

No model, checkpoint, prompt, reference, scheduler, CFG, inference-step,
identity-shift, branched-attention, `pose_adapt_ratio`, or CA-mixing behavior is
changed. The three jobs request one A100 each and are within the user's explicit
ten-GPU exception when combined with the five running project training jobs.

## Acceleration

After eight one-GPU project jobs were verified Running with platform error
code zero, two isolated acceleration workers were prepared under the user's
ten-GPU exception. Each has its own manifest-verified source tree and distinct
run/output names. Worker 1 was accepted as
`lm-mpi-job-d2a306cb-e84d-4eff-af6a-c374f59f12c5` and runs seed-1 D followed
by seed-3 C. Worker 2 was rejected before job creation with
`WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE`; it must not be retried without a
new user request.
