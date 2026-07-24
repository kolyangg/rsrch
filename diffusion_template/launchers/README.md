# Launchers

- `active/` contains the supported April RHCA replay launchers.
- `archive/slurm/` contains older February/March SLURM jobs for provenance.

`run_rhca_apr2026_one_id_1gpu.sh` preserves the historical training-seen
`51.jpg` validation reference. For a leak-free companion control, use
`run_rhca_apr2026_one_id_holdout51_1gpu.sh`; it keeps validation fixed while
removing `51.jpg` from both training-target and training-reference sampling.

Run active launchers from any working directory; they resolve the project root
from their own location. Archived SLURM jobs expect submission from
`diffusion_template` and may reference historical configs.

All Comet launchers require `COMET_API_KEY` from the environment. No API key is
stored in these files.
