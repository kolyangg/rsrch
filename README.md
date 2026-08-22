# E13-family clean branch

This branch is the concise, auditable implementation of the ten selected
E13-family recipes. The shared architecture is explicit target-query /
reference-K/V branched self-attention; experiment differences live in small
Hydra leaves and isolated extension modules.

| Recipe | Difference from shared E13 |
|---|---|
| E13 | Large Dataset baseline. |
| BC_E13 | E13 trained on sealed BigCelebs. |
| CL14 | Corrected Cosmic policy plus a two-cell training-mask feather. |
| CL14_CA | CL14 plus bounded residual identity-token CA in `up_blocks.0/1`. |
| CL18 | CL14 plus training-only same-ID cross-view consistency. |
| CL19 | CL14 with a two-cell cosine router over full BA messages. |
| CL20 | CL14 with the sealed Cosmic/BigCelebs hard-case curriculum. |
| CL23 | CL19 plus fixed denoising-progress low/high frequency gains. |
| CL27 | CL23 plus training-only frequency-surface supervision. |
| CL39 | CL27 plus parameter-free entropy abstention to native target SA. |

The historical attention equations remain byte-identical to 2 June in
[`attn_processor_cleanest.py`](diffusion_template/src/model/photomaker_branched/attn_processor_cleanest.py).
The selected rank-128 specialization is
[`e13_attn_processor.py`](diffusion_template/src/model/photomaker_branched/e13_attn_processor.py).
Later routing equations are isolated in
[`hardcase_attn_processor.py`](diffusion_template/src/model/photomaker_branched/hardcase_attn_processor.py),
and training-only objectives are isolated in
[`e13_objectives.py`](diffusion_template/src/model/photomaker_branched/e13_objectives.py).
The exact shared-file hooks and possible later bloat are listed in the
[`2 June key-file rebase record`](diffusion_template/analysis/2026-08-22_june2_key_file_rebase_implementation.md).

Use the single supported launcher,
[`run_e13_family_24k_1gpu.sh`](diffusion_template/launchers/active/run_e13_family_24k_1gpu.sh),
through the exact Serv packages listed in
[`serv_run_packages/README.md`](diffusion_template/serv_run_packages/README.md).
That README records config names, path assumptions, `.env` requirements, and
submission gates. No job is submitted merely by these files.

Architecture and reproducibility details are in
[`docs/architecture/`](diffusion_template/docs/architecture/) and the current
session handoff is
[`docs/handoffs/LATEST.md`](diffusion_template/docs/handoffs/LATEST.md).
