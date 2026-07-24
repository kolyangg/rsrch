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

Run active launchers from any working directory; they resolve the project root
from their own location. Archived SLURM jobs expect submission from
`diffusion_template` and may reference historical configs.

All Comet launchers require `COMET_API_KEY` from the environment. No API key is
stored in these files.
