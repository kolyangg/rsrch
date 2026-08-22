# Current handoff: clean_full

Updated 22 August 2026. Read this file before changing code, interpreting
results, or launching a job from `diffusion_template`.

## Current code boundary

`clean_full` is the supported unified branch. It intentionally contains one
active trainer/model/pipeline implementation, one Serv launcher, a small Hydra
config inheritance closure, and four training dataset implementations. The
support set is PM0; CL14, CL19, CL23, CL27, CL39; CL40-CL45; E13; BC_E13; and
the BC_E13 ds1-ds3 dataset arms. The allowlist and immutable historical Comet
keys are in `src/configs/clean_full_runs.json`.

The traditional core paths (`train.py`, `lora2.py`,
`attn_processor_cleanest.py`, both trainer files, and their model/runtime/
pipeline helpers) are exact copies of the local 2 June `main_clean` baseline,
commit `2157eada14824d14019e80f9416e6d736c837306`. They are reference code, not
the supported runtime. Allowlisted configs target `clean_full_*` modules and
the launcher starts `train_clean_full.py`. CL39-CL44 attention behavior and
CL45 PCGrad are isolated in defaults-off extension modules. Do not switch a
supported config back to the June targets: the audited hard-v1 ownership,
schema-v2 checkpoint, batched-conditioning, and fixed-96 contracts replaced
too much of those classes for inheritance to be safe.

Within the active model, `clean_full_model_contract.py` owns preparation,
optimizer ownership, manifests, and checkpoint I/O. Legacy disabled
frozen-teacher/predicted-x0 objectives are quarantined in
`clean_full_excluded_objectives.py`. Recent attention-arm helpers and CL45
PCGrad are likewise outside the shared model/processor/trainer bodies.

`PhotomakerBranchedLora.__init__()` now exposes one `clean_full_config`
mapping after `num_inference_steps`, instead of 226 flat experiment-era
arguments. Only 29 of those old values differed from their Python defaults in
the support set, and only 18 varied across the 17 configs. The closed schema in
`clean_full_model_config.py` separates common runtime/contract/architecture
settings from mask feathering, hard-case routing, frequency surface, and one
mutually exclusive recent extension. It recreates the retained internal
attributes so downstream processor/checkpoint behavior stays stable; all 17 x
226 resolved values matched the pre-refactor snapshots exactly. Do not add new
flat constructor keywords. Add a narrowly scoped schema field and validator
rule only when a supported config genuinely needs it.

Use only:

```bash
python tools/validate_clean_full_config.py --list
CONFIG_NAME=<allowlisted-config> RUN_NAME=<new-name> \
  bash launchers/active/run_clean_full_config_1gpu.sh
```

The config selects scientific behavior. Ad-hoc Hydra overrides are rejected.
The launcher selects a dataset preflight from the resolved config, seeds the
canonical run record, verifies the live Comet key, and runs deferred fixed-96
face quality only after Accelerate succeeds.

The full file/class/function and exclusion map is
`analysis/2026-08-22_clean_full_code_structure_and_run_inventory.md`.

## Scientific invariants

- Run Hydra from `diffusion_template/`; sibling `../dataset_full` paths are
  intentional.
- Training and validation use hard-replacement branched self-attention with
  reference K/V and target Q. Branched cross-attention is disabled.
- `pipeline.pose_adapt_ratio=0` and `pipeline.ca_mixing_for_face=false` are
  mandatory.
- Validation is the sealed 96-image `manual_val` panel at step 0 and every
  2,000 optimizer steps, one generated image per item. PM0 is validation-only
  at step 0 and disables branched attention.
- Identity curves are `id_sim_best_legacy` and mask-matched
  `id_sim_subject_v2`; the latter uses the sealed subject-v2 embeddings.
- All new training must preserve the optimized processor lookup/collector
  pipeline documented in
  `analysis/2026-08-16_training_pipeline_processor_lookup_fix.md`.
- Immutable Comet keys, not display names, identify historical runs.

## Current result context

The completed CL38-CL45 comparison found CL39 (null-key confidence router) to
be the strongest current system and CL44 (semantic window gate) the secondary
candidate. CL38 is deliberately excluded from the clean support set because
its recovery history is more complicated and it was not requested as a main
target. CL40-CL45 are retained as the latest six requested configs even where
their result did not beat CL39.

Canonical keys for all supported runs are recorded in
`src/configs/clean_full_runs.json`. The 21 August code/visual/metric synthesis
remains available from branch `test`; its decision-relevant conclusion is
summarized here so generated figure assets do not re-enter `clean_full`.

## Machine and credentials

Neb is unavailable. Do not access it, use it as a proxy, or submit work to it.
Use the local machine or Serv as authorized. Before submitting on Serv, inspect
this project's running and pending jobs and respect the normal six-A100
concurrent request ceiling.

Machine paths and credentials belong only in `.env`; never commit them. Every
new Comet run must produce `saved/<run_name>/comet_experiment.json` with a live
32-character experiment key during startup.

## Historical recovery

Removed configs, external model mirrors, per-job sealed snapshots, launchers,
generated reports/assets, and alternate training families remain recoverable
from branch `test` at base commit
`97e0364d6fa6ee6b1b8c3d99aa547805b18ad47f`. Do not copy them back into
`clean_full`; use a historical checkout when exact replay of an excluded run is
required.
