# Launchers

- `active/` contains the supported April RHCA replay launchers.
- `archive/slurm/` contains older February/March SLURM jobs for provenance.

Run active launchers from any working directory; they resolve the project root
from their own location. Archived SLURM jobs expect submission from
`diffusion_template` and may reference historical configs.

All Comet launchers require `COMET_API_KEY` from the environment. No API key is
stored in these files.
