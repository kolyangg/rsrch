# Branched-attention clean training workspace

The `clean_full` branch contains the supported `aug-large-ds` training stack,
the sealed fixed-96 validation inputs, and config-selected dataset/model
variants. External model mirrors, old launchers, sealed source snapshots,
generated reports, and unsupported configs remain available in Git history but
are intentionally absent here.

Run all Hydra and training commands from `diffusion_template/`. See
[`diffusion_template/README.md`](diffusion_template/README.md) for the supported
config list and launcher contract.
