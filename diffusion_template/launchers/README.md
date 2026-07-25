# Launchers

- `active/` contains the supported April RHCA replay launchers.
- `archive/slurm/` contains older February/March SLURM jobs for provenance.

`run_rhca_apr2026_one_id_1gpu.sh` preserves the historical training-seen
`51.jpg` validation reference. For a leak-free companion control, use
`run_rhca_apr2026_one_id_holdout51_1gpu.sh`; it keeps validation fixed while
removing `51.jpg` from both training-target and training-reference sampling.

For the CosmicLarge next-step controls:

- `run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh` disables
  branched cross-attention while retaining historical branched self-attention.
- `run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k_1gpu.sh`
  additionally freezes target/noise projection copies; run it only after the
  CA-off trigger is met.
- `run_rhca_controlled_identity_factorial_4k_1gpu.sh` selects the
  manifest-backed `multi_full`, `single_full`, or `multi_cosref` arm through
  `FACTORIAL_ARM`.
- `run_rhca_cosmic_one_id_reference_policy_4k_1gpu.sh` runs the post-Task-D
  `margin40` or `canvas1024` data-only controls without changing the sealed
  validation package.
- `run_rhca_cosmic_large_adapted_1gpu.sh` uses the isolated loader for the
  real full-Cosmic `face_paths` manifest. `EXPERIMENT_ARM` selects an exact
  crop/canvas/caption policy and step budget.
- `launchers/neb/start_rhca_cosmic_experiment.sh` supplies Neb's fixed dataset
  and environment paths. Launch it through the `nohup setsid` procedure in
  `LOCAL_NEB_SERVER_OPERATIONS.md`; Neb must run only one GPU job at a time.

The new Cosmic launchers seed
`saved/<run_name>/comet_experiment.json` from the matching JSON under
`experiments/cosmic_large_adaptation/`. `CometMLWriter` preserves the plan and
fills the immutable experiment key during startup. A non-empty existing output
directory or an already-registered Comet key is a hard error.

Run active launchers from any working directory; they resolve the project root
from their own location. Archived SLURM jobs expect submission from
`diffusion_template` and may reference historical configs.

All Comet launchers require `COMET_API_KEY` from the environment. No API key is
stored in these files.
