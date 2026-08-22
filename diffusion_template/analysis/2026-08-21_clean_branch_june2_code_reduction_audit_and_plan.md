# Clean-branch code reduction audit and implementation plan

**Date:** 21 August 2026  
**Branch audited:** `clean` at `8d56caf179994dcc12a125d05361e910d6f056af`  
**Exact comparison base:** `main_clean@2157eada14824d14019e80f9416e6d736c837306` (2 June 2026)  
**Scope:** the ten recipes listed in `serv_run_packages/README.md`, launched through `launchers/active/run_e13_family_24k_1gpu.sh`

## Executive conclusion

The clean branch has already removed the large historical `_old*` and backup
trees, but it is not yet a minimal selected-experiment branch. Three different
kinds of excess remain. **[code] [filesystem]**

1. **Clearly unreachable legacy surface:** 59 of 87 Hydra YAMLs, 23 Python
   files totalling 6,054 lines, all of `compare/`, both legacy reporting
   folders, old standalone inference, two release ZIPs, and 114 of 136 tracked
   `dataset_full/` paths are outside the ten-recipe launch/validation closure.
2. **Active configuration built on stale scaffolding:** every clean leaf still
   inherits `one_id_09Feb_testing.yaml`, which composes old datasets, metrics,
   writers, and controls before overriding them. The final selected values are
   mostly correct, but their provenance is unnecessarily hard to audit.
3. **Real retained experiment logic placed in shared June files:** CL19/23/27/39
   attention equations, CL18/27 objectives, E13 checkpoint ownership, and
   alternate-base validation are valid code, but much of it is embedded in
   `lora2.py`, `attn_processor_cleanest.py`, `branched_runtime.py`, and
   `base_trainer.py`. This is the main reason those files are difficult to
   compare with 2 June.

The recommended end state is therefore:

- delete files that have no selected-runtime or operational role; do not retain
  them under `_not_used`, because Git and the other branches already preserve
  them;
- replace the old one-ID inheritance chain with one direct, compact clean
  family configuration;
- restore the shared June model/attention files to a near-June shape and move
  retained experiment-specific equations into explicitly named active modules;
- remove duplicated checks, inactive switches, and always-zero telemetry while
  preserving exact routing, ownership, data, validation, checkpoint, and
  required throughput contracts.

No training, validation, Serv job, code implementation, or new test was run or
created for this audit. **[record]**

## 1. Evidence boundary and method

### 1.1 Repositories and worktree state

- `/home/kolyangg/rsrch_2Jun` is at the exact requested base commit
  `2157eada14824d14019e80f9416e6d736c837306`.
- The base commit is an ancestor of the audited clean HEAD.
- All line-delta counts in this report use the committed comparison
  `2157eada...HEAD`; they do not include unrelated local edits.
- The clean worktree was already dirty before this report:
  - `src/model/attn_procs/attn_processor.py` has a local one-line typo repair,
    `phaimport torch` to `import torch`;
  - two architecture PDFs under `analysis/assets/` are untracked.
- Those pre-existing changes were not modified. The locally repaired processor
  file is itself outside the selected closure; if deletion is approved, its
  one-line local change must be acknowledged rather than silently discarded.

### 1.2 Dependency tracing

The closure was derived from all ten Hydra leaves, their defaults chains, the
launcher's forced `writer=cometml` override, every `_target_` instantiated by
`train.py`, recursive local Python imports, and the validator/preflight/finalizer
commands invoked by the active launcher. `rg` was then used to cross-check
dynamic and lazy imports. **[code]**

This is a selected-workflow audit, not a claim about arbitrary manual imports.
Anything reachable only through an unsupported legacy config or a false flag is
reported separately from code that is not referenced at all.

### 1.3 Classification used below

- **Remove:** duplicated, inactive, always-zero, compatibility-only, or outside
  the declared ten-recipe scope.
- **Move:** scientifically or operationally required, but currently placed in a
  shared key file where it obscures the June baseline.
- **Stay:** concise shared behavior or experiment behavior that directly defines
  one of the retained recipes/contracts.

## 2. Supported configuration surface

The supported leaves are:

1. `E13_large_ds_joint_shadow_sa128_24k`
2. `BC_E13_big_celebs_joint_shadow_sa128_24k`
3. `CL14_cosmic_joint_shadow_sa128_softmask_24k`
4. `CL14_CA_cosmic_residual_identity_ca_24k`
5. `CL18_cosmic_crossview_spatial_consistency_24k`
6. `CL19_cosmic_true_soft_fullquery_router_24k`
7. `CL20_cosmic_bigcelebs_hardcase_curriculum_24k`
8. `CL23_cosmic_temporal_frequency_router_24k`
9. `CL27_cosmic_frequency_surface_energy_24k`
10. `CL39_cosmic_null_key_confidence_router_24k`

All ten use the same active launcher. Their final instantiated targets are:

| Role | Selected target(s) |
|---|---|
| Trainer | `src.trainer.sdxl_trainers.PhotomakerLoraTrainer` |
| Model | `src.model.photomaker_branched.lora2.PhotomakerBranchedLora` |
| Pipeline | `photomaker_branched_clean` for E13/BC/CL14; the subject-v2 wrapper for the seven corrected-validation descendants |
| Loss | `MaskedDiffusionLoss`, with BA auxiliary terms used only by CL18/CL27/CL39 descendants |
| Train datasets | `large_dataset`, `big_celebs`, `cosmic_large_adapted`, or `cl20_hardcase_curriculum` |
| Validation dataset | fixed 96-item `manual_val` only |
| Metrics | `clip_ts` plus legacy or subject-v2 identity similarity, and deferred seven-curve face quality |
| Scheduler | `WarmupHoldCosineLR` |
| Writer | `CometMLWriter` forced by the launcher |

### 2.1 Current config composition problem

There are 87 YAML files under `src/configs/`. Twenty-eight appear in the
current ten-leaf composition/load chain, but four of those are scaffolding with
no surviving selected value: `writer/console.yaml`,
`lr_scheduler/custom_linear.yaml`, `metrics/all_metrics_oneid.yaml`, and their
owning `one_id_09Feb_testing.yaml` defaults chain. Fifty-nine YAMLs have no path
from any retained leaf. **[config]**

The direct clean family should instead compose only the active groups. After
that change, the complete supported surface is 24 YAMLs:

- the ten leaves;
- `e13_family_24k.yaml` and `cl18_cl20_extension_24k.yaml`;
- one file each for trainer, model, pipeline, writer, DDP, transforms,
  dataloaders, datasets, and 24k LR schedule;
- the two metric maps actually needed by baseline and subject-v2 validation.

`one_id_09Feb_testing.yaml` should not remain the semantic parent of clean
experiments.

## 3. Runtime-used files

### 3.1 Key architecture and objective files

These files directly define trainable ownership, Q/K/V routing, residual
merges, objectives, checkpoint state, or validation architecture and must
remain active in some form:

| Area | Files | Why active |
|---|---|---|
| Model entry and base | `src/model/photomaker_branched/lora2.py`, `src/model/sdxl/original.py`, `src/model/photomaker_path.py` | SDXL/PhotoMaker construction and training forward |
| PhotoMaker identity encoder | `model_v2_NS.py`, `resampler.py`, `insightface_package.py` | PMv2 token/face conditioning |
| Shared hard BA | `attn_processor_cleanest.py`, `branched_runtime.py`, `branch_helpers.py`, `lora2_helpers.py` | target/reference doubled batch, target Q, reference K/V, masks, and processor installation |
| Ownership/checkpoints | `e13_contract.py` | exact trainable allowlist, optimizer groups, architecture manifest, schema-v2 save/load |
| CL14_CA | `residual_identity_ca_processor_v3.py` | selected rank-64 residual identity-token CA |
| Pipelines | `photomaker_branched_clean.py`, `br_pipeline_helpers.py`, `photomaker_branched_cl18_cl20.py` | fixed CL14 generation and subject-v2 identity selection |
| Loss/training | `loss/diffusion_loss.py`, `trainer/base_trainer.py`, `trainer/sdxl_trainers.py`, `train.py` | primary/auxiliary loss, alternate-base validation, metrics, optimizer loop |
| Subject-v2 validation | `face_subject_selector.py`, `metrics/id_sim_metric.py` | declared-face identity binding and score |

### 3.2 Used data modules

- `src/datasets/base_dataset.py`
- `src/datasets/collate.py`
- `src/datasets/data_utils.py`
- `src/datasets/large_dataset.py`
- `src/datasets/big_celebs.py`
- `src/datasets/cosmic_large_adapted.py`
- `src/datasets/cl20_hardcase_curriculum.py`
- `src/datasets/manual_val.py`
- `src/datasets/reference_frame.py`
- `src/datasets/reference_policy.py`

`src/datasets/bc_e13_schedule_policy.py` is used by startup/preflight rather
than the model training loop.

### 3.3 Used validation, logging, and infrastructure modules

- `src/logger/cometml.py`, `src/logger/logger.py`, and
  `src/logger/logger_config.json`;
- `src/lr_schedulers/lr_schedulers.py`;
- `src/metrics/aligner.py`, `base_metric.py`, `text_sim.py`, `tracker.py`,
  `id_sim_metric.py`, and `face_quality_validation.py`;
- `src/utils/auto_bbox_gen.py`, `id_utils.py`, `init_utils.py`, `io_utils.py`,
  and `model_utils.py` under the current cached-auto-bbox implementation;
- the active launcher, its two Serv wrappers, all ten Serv YAMLs, and the
  launcher's config validators, dataset preflights, CL20 schedule builder,
  Comet finalizer, and face-quality scorer.

The project `docs/`, `analysis/`, `setup/`, active report/Dropbox tools, and
environment snapshots are not imported during a training step, but they are
operational/provenance assets and should not be called dead code.

## 4. Files and folders outside the selected closure

### 4.1 Delete rather than move to `_not_used`

The following are fully recoverable from Git and other branches. Moving them to
`_not_used` would keep search noise and repository weight, so deletion is the
cleaner final state.

#### Whole top-level or project folders

| Path | Evidence | Proposed action |
|---|---|---|
| `compare/` | 131 files, 12,518,793 bytes; referenced only by the obsolete root README, old inference configs, and comparison scripts | Delete whole folder |
| `release/` | two stale source archives plus checksums/README, 18,841,540 bytes; Git already preserves the source | Delete whole folder |
| `diffusion_template/comet_utils/` | superseded by `tools/comet/`; no launcher reference | Delete whole folder |
| `diffusion_template/infer_tools/` | legacy PDF/metric tooling; no retained validation reference | Delete whole folder |
| `src/configs/inference/` | all 22 files are outside the ten-leaf launch path | Delete whole folder |
| `src/configs/pdf_output/` | all three JSONs serve removed legacy PDF tooling | Delete whole folder |

#### Standalone legacy entry points and documentation

- delete `diffusion_template/infer.py`;
- delete `diffusion_template/scripts/run_infer_combinations.py` and
  `run_infer_config_4Apr.json`;
- replace the obsolete root `README.md` with the clean-family README, update it
  for CL39, then remove duplicate `README_E13_FAMILY_CLEAN.md`;
- delete `diffusion_template/README_ba.md` and `README_ba_new.md`.

Keep the Cosmic archive/setup scripts and manual-validation ID-embedding script;
they are selected-data operational utilities, even though the launcher does not
invoke them on every run.

#### Unused Python modules

The following 23 files total 6,054 lines and are not imported or instantiated
by the ten recipes/startup closure:

- datasets:
  - `src/datasets/cosmic.py`
  - `src/datasets/cosmic_new_example.py`
  - `src/datasets/dreambooth.py`
  - `src/datasets/man_datasets.py`
- loggers:
  - `src/logger/clearml.py`
  - `src/logger/console.py`
  - `src/logger/utils.py`
  - `src/logger/wandb.py`
- metrics:
  - `src/metrics/dino.py`
- old/non-selected model implementations:
  - `src/model/attn_procs/attn_processor.py`
  - `src/model/photomaker/id_encoder.py`
  - `src/model/photomaker/lora.py`
  - `src/model/photomaker_branched/heatmap_utils.py`
  - `src/model/photomaker_branched/lora.py`
  - `src/model/sdxl/lora.py`
- non-selected pipelines/trainer:
  - `src/pipelines/photomaker.py`
  - `src/pipelines/sdxl.py`
  - `src/trainer/inferencer.py`
- inactive custom transforms:
  - `src/transforms/__init__.py`
  - `src/transforms/normalize.py`
  - `src/transforms/scale.py`
- utilities/entry points:
  - `bbox_utils/resize_with_json.py`
  - `infer.py`

The old `photomaker_branched/lora.py` even imports a backup path already
removed from the clean branch; it cannot be a supported implementation.

#### Inactive Hydra YAMLs

The 59 YAMLs with no selected-leaf path total 4,302 lines:

- all 22 files in `src/configs/inference/`;
- top-level historical bases:
  - `all_09Feb.yaml`, `all_09Feb_noise.yaml`, `all_09Feb_ref.yaml`,
    `all_id_br_attn1.yaml`, `all_id_br_attn1_lr4.yaml`;
  - `one_id_09Feb_testing_1model.yaml`, `one_id_09Feb_testing_all.yaml`,
    `one_id_09Feb_testing_idemb.yaml`, `one_id_09Feb_testing_idemb2.yaml`,
    `one_id_09Feb_testing_noise.yaml`, `one_id_09Feb_testing_ref.yaml`;
  - `one_id_br_attn1.yaml`, `one_id_br_attn1_local.yaml`,
    `one_id_br_attn1_lr3.yaml`, `one_id_br_attn1_lr4.yaml`,
    `one_id_br_attn1_lr4_new.yaml`, `one_id_br_attn1_step0.yaml`,
    `one_id_br_attn1_step0_new.yaml`, and
    `one_id_origv2_train_lora_local.yaml`;
- `datasets/all_datasets_local.yaml` and `lr_scheduler/constant.yaml`;
- all model configs except `model/photomaker_branched_lora2.yaml`;
- all pipeline configs except `pipeline/pm_br_09Feb_testing.yaml`;
- `trainer/sdxl.yaml`;
- `transforms/batch_transforms/example.yaml` and
  `transforms/instance_and_batch.yaml`;
- `writer/clearml.yaml`, `writer/cometml_local.yaml`, and `writer/wandb.yaml`.

After direct clean composition is introduced, also delete the four
composition-only remnants `one_id_09Feb_testing.yaml`,
`metrics/all_metrics_oneid.yaml`, `lr_scheduler/custom_linear.yaml`, and
`writer/console.yaml`.

### 4.2 Prune aggregate config files instead of retaining stale entries

`datasets/all_datasets.yaml` should contain only:

- train: `large_dataset`, `big_celebs`, `cosmic_large_adapted`, and
  `cl20_hardcase_curriculum`;
- validation: `manual_val`.

This removes target strings for DreamBooth, old Cosmic/one-ID variants,
machine-specific Large/Cosmic variants, `manual_val_two`, and old validation
datasets. `dataloaders/all_dataloaders.yaml` should retain only `train` and
`manual_val`. Metric maps should retain only the four selected definitions:
`clip_ts`, `id_sim_best`, `id_sim_best_legacy`, and `id_sim_subject_v2`.

### 4.3 Dataset repository cleanup

The launcher requires external, hash-pinned training manifests and image roots.
Within tracked `dataset_full/`, only the fixed manual-validation panel is used.
The exact retained set is 22 paths, approximately 2.96 MB:

- `val_dataset/references/*` (12 identity images);
- `val_dataset/prompts_10.txt`;
- `val_dataset/classes_ref.json`;
- `val_dataset/ref_bboxes.json`;
- `val_dataset/id_embeds_manual_val.pth`;
- both protocol directories, each retaining its README,
  `pm96_bboxes_new.json`, and `pm96_bboxes_new_auto.json`.

The other 114 tracked paths occupy approximately 331.4 MB and are not selected:
old one-ID images/metadata, obsolete Cosmic/Large manifests, analysis notebooks,
old bbox maps, alternate reference folders, helper scripts, a broken `LAION-5B`
symlink, and a tracked `.pyc`. Delete them after the clean dataset config no
longer names their fallbacks.

### 4.4 Loaded or reachable only through false/legacy branches

These are not as strong as the immediate deletions above:

| Path | Current reason it is reachable | Recommendation |
|---|---|---|
| `photomaker_branched/model.py` | eager import for a PMv1 branch; selected pipeline fixes `pm_version="v2"` | Remove PMv1 branch/import and then delete file |
| `create_mask_ref.py` | lazy `auto_mask_ref` path; every selected config sets it false | Remove the false branch and file after config narrowing |
| `utils/auto_bbox_gen.py`, `bbox_utils/generate_bboxes.py`, `visualize_bboxes.py`, and YOLO weight | current trainer loads the sealed `*_auto.json` through the generator/cache abstraction | Point `manual_val.bbox_mask_gen` directly to the exact sealed `*_auto.json`, set generation off, then remove regeneration/overlay fallback |
| `debug_helpers.py` | imported by shared runtime, but `val_debug=false` and branch debug outputs are disabled | Retain during the first pass to avoid a large unrelated June-base deletion; reconsider only in a separate scope-pruning pass |

The direct-auto-JSON change is not a metric/protocol change: the validation
dataset already supports prompt/ID keyed `face_crop_new` records, and the two
exact auto JSONs are tracked. It removes only an unselected regeneration
fallback. Nevertheless, it should be made as a separate, reviewable change.

## 5. Exact key-file differences from 2 June

### 5.1 Review-size summary

| File | June lines | Current lines | Diff | Primary disposition |
|---|---:|---:|---:|---|
| `train.py` | 403 | 499 | `+96 / -0` | remove duplicate checker; retain one contract call |
| `lora2.py` | 688 | 1,218 | `+553 / -23` | remove inactive/duplicate controls; move objectives and conditioning blocks; retain small hooks |
| `lora2_helpers.py` | 315 | 610 | `+295 / -0` | move collectors/variant objectives; retain one selected conditioning path and strict installer |
| `attn_processor_cleanest.py` | 757 | 1,195 | `+439 / -1` | move CL19/23/27/39 processor; remove false telemetry; retain hard-v1 base/cache |
| `branched_runtime.py` | 653 | 883 | `+298 / -68` | move factories/validation; retain sealed doubled-batch and timestep behavior |
| `e13_contract.py` | 0 | 564 | `+564 / -0` | keep ownership/checkpoint core; simplify settings and compatibility |
| `residual_identity_ca_processor_v3.py` | 0 | 330 | `+330 / -0` | stay; already correctly isolated |
| `base_trainer.py` | 968 | 1,150 | `+206 / -24` | move E13 validation construction/copy; keep face-quality integration |
| `sdxl_trainers.py` | 822 | 843 | `+25 / -4` | stay, except fields removed with dead telemetry |
| `diffusion_loss.py` | 80 | 104 | `+26 / -2` | remove ownership-zero plumbing; keep concise auxiliary loss |
| `br_pipeline_helpers.py` | 1,148 | 1,062 | `+51 / -137` | stay; mostly a sealed simplification |
| `photomaker_branched_clean.py` | 1,333 | 1,337 | `+10 / -6` | stay; already close to June |
| `photomaker_branched_cl18_cl20.py` | 0 | 165 | `+165 / -0` | stay subject-v2 behavior; rename/generalize and centralize transfer |

### 5.2 `train.py`

**Remove/move**

- `_assert_expected_trainable_contract` duplicates ownership and optimizer
  membership checks in `e13_contract.assert_trainable_contract` and duplicates
  numeric totals stored in YAML and validator scripts.
- Move the two accepted numeric profiles (ordinary E13 family and CL14_CA) into
  the contract owner, then delete the 96-line generic checker and YAML category
  trees.

**Stay**

- one short `model.assert_trainable_contract(optimizer)` call after optimizer
  creation.

### 5.3 `lora2.py`

**Remove as bloat**

- `conditioning_cache_enabled`: it is stored and validated but has no runtime
  consumer;
- repeated assignment/validation of the same CL settings in `lora2.py`,
  `e13_contract.py`, `branched_runtime.py`, configs, and validators;
- the always-zero `ba_ownership_loss` output;
- defaults-off `ba_hardcase_telemetry_enabled` plumbing;
- legacy option branches that none of the ten configs can select after direct
  composition.

**Move**

- the 40-plus E13/CL constructor parameters and the second full forwarding list
  into one compact contract/settings object owned by `e13_contract.py`;
- the CL18 alternate-reference teacher/student block and CL27 occluder/surface
  aggregation into an active `e13_objectives.py` module;
- actual InsightFace-session provider validation into
  `insightface_package.py` (the launcher-level ORT check remains separate);
- batched prompt/PhotoMaker/VAE conditioning methods into the selected
  conditioning helper, while deleting the inactive scalar-vs-batched switch.

**Stay**

- strict contract, optimizer, checkpoint-save/load, and post-eval delegates;
- CL14's small two-cell training-mask feather;
- the active batched frozen-conditioning algorithm. It materially improves
  throughput and is not bit-identical to unbatched BF16 GEMMs, so it must be
  preserved rather than silently reverted to June;
- concise forwarding of CL18/27 batch data to the objective helper.

### 5.4 `lora2_helpers.py`

**Remove as bloat**

- the inactive fallback conditioning implementation once the clean branch is
  explicitly selected-path only;
- mode selectors whose value is fixed across all ten leaves.

**Move**

- CL14_CA and CL27/39 telemetry/loss collectors to `e13_objectives.py`;
- any remaining variant-specific preparation out of the generic installer.

**Stay**

- fail-closed processor installation before optimizer construction;
- exactly one batched conditioning-preparation path;
- explicit `reference_noise` forwarding required by CL18;
- the optimized-pipeline invariant that `unet.attn_processors` is resolved
  once before each selected per-layer collector loop, with disabled collectors
  returning before lookup.

### 5.5 `attn_processor_cleanest.py`

**Remove as bloat**

- `hardcase_telemetry_enabled`, its setter, propagation, and low/high-scale
  reductions: every retained CL23/27/39 config fixes it false;
- legacy branched cross-attention support once direct config hardcodes native
  CA plus the separate CL14_CA residual. No retained recipe instantiates
  `BranchedCrossAttnProcessor`.

**Move**

- `_normalized_halves`, cosine soft routing, Gaussian frequency split,
  frequency-surface objective state, and null-key confidence into a dedicated
  `hardcase_attn_processor.py` subclass used only by CL19/23/27/39.

**Stay**

- the June hard-v1 target/reference processor and its explicit target-Q /
  reference-KV route;
- the small prepared-mask cache, preferably unconditional for this branch. It
  reuses an exact tensor and is part of the active throughput profile;
- a tiny common interface for masks and processor state.

This change should make `attn_processor_cleanest.py` visually close to the June
file, while the real later equations remain explicit in one active extension
file rather than being hidden.

### 5.6 `branched_runtime.py`

**Remove as bloat**

- repeated validation of fixed E13 values already owned by the contract;
- generic SA/CA/top-k branches unavailable to selected configs;
- false hard-case telemetry propagation.

**Move**

- hard-case processor construction to `hardcase_attn_processor.py`;
- CL14_CA processor selection/token-index setup beside
  `residual_identity_ca_processor_v3.py` or in one small active processor
  factory;
- variant runtime-setting transfer to the validation wrapper/helper.

**Stay**

- the sealed CL14 single-spatial-reference batch layout;
- explicit optional `reference_noise` for CL18;
- real scheduler-timestep progress for CL23/27/39;
- target/reference doubled UNet inputs and the merged-target return;
- the short early return that avoids unused branch-debug tensors.

The `-68` June deletions and revised batch/noise behavior are correctness
changes used to reproduce sealed CL14 generation, not cleanup candidates.

### 5.7 `e13_contract.py`

**Remove as bloat**

- duplicated inactive performance flags;
- repeated scalar assignments that can be represented once in a compact
  settings map;
- the duplicate unreachable `raise` in `_processor_prefixes`;
- compatibility projection for unrelated inert E14-E24 manifest fields if the
  approved scope is fresh clean runs/checkpoints only.

**Move/consolidate**

- all exact selected settings and the two numeric ownership profiles into this
  one owner;
- variant-specific manifest fragments behind one small extension hook rather
  than a long set of `getattr` branches.

**Stay**

- exact parameter-role allowlist;
- freezing/configuration of trainables;
- optimizer object-membership check;
- complete schema-v2 trainable checkpoint save/load;
- architecture, name, shape, and dtype manifest checks.

Checkpoint compatibility is the one scope decision that must be explicit.
Every supported leaf starts fresh (`continue_run=false`, no selected resume
checkpoint), so source-r2/E14-E24 projection is not needed by the launcher. If
the clean branch is also intended for ad-hoc loading of those historical
checkpoints, retain only the exact required compatibility cases and document
them.

### 5.8 CL14_CA and later attention files

`residual_identity_ca_processor_v3.py` should stay. It is one selected
architecture, already isolated, and owns its target-Q/active-PhotoMaker-ID-KV,
bounded gate, face-mask merge, and trainables. Its telemetry collector can move
out, but the processor should not be folded into the June shared attention
file.

The CL19/23/27/39 equations likewise stay, but in their own active processor
module. Moving them is organizational reduction in key files, not removal of
BA functionality.

### 5.9 Trainers, loss, and validation wrapper

**`base_trainer.py`: remove/move**

- move strict processor copy, PhotoMaker-default shadow/restore, and the long
  model-to-pipeline attribute tuple into a validation helper;
- remove the CL20 resume-position hook from the selected launcher path unless
  resume becomes a supported clean operation;
- remove compatibility toggles around fixed one-GPU behavior after old configs
  are deleted.

**`base_trainer.py`: stay**

- step-zero/every-2k validation;
- fixed-96 iteration and deferred face-quality session;
- gradient norms only when their log consumer requests them;
- no disabled auxiliary collector work.

**`sdxl_trainers.py`: stay**

- model telemetry propagation;
- one-GPU stacked diagnostic aggregation;
- exact generated/reference boxes passed to subject-v2 metrics.

**`diffusion_loss.py`: remove/stay**

- remove `ba_ownership_loss` input/output because every model return sets it to
  zero and no retained objective owns it;
- keep a concise `ba_aux_loss` addition and detached CL18/CL27 diagnostic
  outputs.

**Validation wrapper: move/stay**

- retain declared-face subject-v2 embedding selection;
- rename `photomaker_branched_cl18_cl20.py` to reflect that it now serves
  CL14_CA and CL23/27/39 too;
- replace its second long attribute-copy tuple with the same centralized
  runtime-settings transfer used by alternate-base validation.

### 5.10 Pipelines

The changes in `br_pipeline_helpers.py` and
`photomaker_branched_clean.py` should stay. They are already concise relative
to June and encode the sealed one-reference CL14 trajectory: first spatial
reference reuse, exact reference noise, no `branched_attn_end_step`, and GPU
InsightFace. Reverting them to reduce diff size would alter retained validation
behavior.

## 6. Proposed target layout

The active model directory should read as a small base plus named scientific
extensions:

```text
src/model/photomaker_branched/
  lora2.py                         # short model orchestration and delegates
  lora2_helpers.py                 # one selected conditioning/forward path
  attn_processor_cleanest.py       # shared hard-v1 SA, close to June
  hardcase_attn_processor.py       # CL19/23/27/39 equations
  residual_identity_ca_processor_v3.py  # CL14_CA
  e13_contract.py                  # ownership, exact settings, checkpoint
  e13_objectives.py                # CL18/27 objectives and active telemetry
  branched_runtime.py              # doubled-batch execution and dispatch
  branch_helpers.py
  insightface_package.py
  model_v2_NS.py
  resampler.py
```

This adds no architecture. It relocates code already active and permits the
shared files to approach their June shape. The new active modules replace many
more deleted legacy modules, so both source count and total lines still fall.

## 7. Implementation plan after approval

### Phase 0 — preserve the audit boundary

1. Re-check branch, HEAD, and dirty status.
2. Preserve the two untracked PDFs.
3. Record that the local edit to the unused old processor is typo-only before
   deleting that file; do not discard unrelated user work.
4. Take pre-change composed-config snapshots for all ten leaves using the exact
   launcher overrides. These are temporary comparison artifacts, not new test
   files.

### Phase 1 — make configuration selected-scope only

1. Rewrite `e13_family_24k.yaml` to compose direct active group files instead
   of `one_id_09Feb_testing.yaml`.
2. Prune datasets, dataloaders, metrics, model, pipeline, writer, transforms,
   and LR groups to selected entries.
3. Remove inactive config switches and the always-zero ownership loss name.
4. Keep the ten leaf deltas scientifically unchanged.
5. Compare every resolved scientific field against the Phase-0 snapshots:
   model target/ranks/routes, loss weights, data paths/policies, seeds,
   validation images/prompts/boxes, scheduler, steps, and metrics must match.

### Phase 2 — delete isolated unreachable surface

1. Delete the 59 unreachable YAMLs and three PDF JSON configs.
2. Delete the 23 unused Python files listed above.
3. Delete `compare/`, `release/`, legacy reporting/inference folders and
   entrypoints, and obsolete READMEs.
4. Reduce `dataset_full/` to the exact 22-file validation panel.
5. Remove generated/tracked bytecode and local `__pycache__` directories.
6. Keep operations/provenance files (`docs`, `analysis`, `setup`, active tools,
   Serv packages) intact.

### Phase 3 — slim key files without changing selected behavior

1. Consolidate selected settings, exact ownership totals, and checkpoint
   contract in `e13_contract.py`.
2. Delete the duplicate 96-line `train.py` checker, leaving one model-owned
   assertion.
3. Move CL18/27 objective and active telemetry code from `lora2.py` and
   `lora2_helpers.py` into `e13_objectives.py`.
4. Keep only the active batched conditioning path; remove its unused selector,
   unused cache flag, and scalar fallback.
5. Move CL19/23/27/39 attention code into
   `hardcase_attn_processor.py`; restore the shared hard-v1 processor to a
   near-June form.
6. Remove legacy branched CA and fixed-off generic route branches; retain
   CL14_CA only through its separate residual processor.
7. Reduce `branched_runtime.py` to shared doubled-batch execution plus concise
   processor dispatch.
8. Centralize validation model/pipeline state transfer outside
   `base_trainer.py` and the subject-v2 wrapper.
9. Remove always-zero ownership-loss plumbing and false hard-case telemetry.

### Phase 4 — remove selected-inactive fallback assets

1. Pin each leaf directly to its exact protocol `*_auto.json` generation boxes
   and disable generation of replacement boxes.
2. Remove `auto_bbox_gen.py`, detector/overlay helpers, and the YOLO weight once
   composition confirms no remaining target/import.
3. Remove the PMv1 encoder branch/file and automatic-reference-mask branch/file.
4. Leave broad debug-helper removal out of this pass unless a later request
   explicitly prioritizes further baseline pruning; it would create a large
   deletion diff in otherwise shared June code.

### Phase 5 — documentation and existing verification only

1. Update the single root clean README, `serv_run_packages/README.md`,
   architecture references, and `TOOLS.md` paths affected by deletion/moves.
2. Update `docs/handoffs/LATEST.md` with the final source layout and exact
   selected contract; do not rewrite experiment evidence.
3. Run only existing/repository-standard checks—no new tests:
   - Hydra composition and all existing config validators for all ten leaves;
   - `verify_cl14_generation_parity.py`;
   - shell syntax for the active launcher and Serv wrappers;
   - compile/import checks for the active closure;
   - processor installation, exact ownership totals, schema-v2 round trip, and
     alternate-base pipeline copy through the existing checks/startup path;
   - `rg` for deleted module/config names and forbidden per-layer
     `unet.attn_processors` property lookups.
4. Do not launch training, submit Serv work, commit, or push without separate
   user authorization.

## 8. Acceptance criteria

The cleanup is complete only when all of the following hold:

- all ten launcher config names still compose, with no stale target/path;
- no selected scientific setting differs from the pre-change snapshot;
- `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, native CA, explicit
  target-Q/reference-KV SA, ranks, ownership totals, masks, seeds, prompts,
  scheduler, 24k steps, and fixed-96 metrics remain unchanged;
- training and validation install the same selected processors and checkpoint
  settings;
- required optimized-pipeline behavior remains: disabled collectors are
  skipped, processor maps are resolved once per collector, gradients are
  measured only when consumed, and undeclared full-activation telemetry stays
  off;
- `train.py` has only a concise contract call rather than a second ownership
  implementation;
- `lora2.py` contains orchestration, not complete CL18/27 algorithms;
- `attn_processor_cleanest.py` contains shared hard-v1 behavior, not all later
  experiment equations;
- there is no legacy branched-CA implementation reachable by selected configs;
- tracked source/config line count and repository size are lower, not merely
  redistributed into `_not_used`;
- no new feature or test suite has been added.

## 9. Confidence and items not established

| Claim | Confidence | Basis |
|---|---|---|
| The 59 YAMLs and listed 23 Python files are outside the ten-recipe launcher closure | High | defaults/target tracing plus recursive import and `rg` cross-check |
| The 114 dataset paths are not selected by the supported launcher | High | launcher requires external training paths; composed validation paths resolve to the listed fixed-96 panel |
| CL19/23/27/39 and CL14_CA code must remain | High | selected leaf values directly invoke those equations/processors |
| Batched conditioning should be preserved | High | active in all ten leaves; documented material throughput gain and measured BF16 non-bit-identity versus scalar execution |
| Direct sealed-auto-JSON loading can replace regeneration fallback without changing boxes | High | `ManualPhotoMakerValDataset` supports the same prompt/ID keyed records; exact JSONs are tracked |
| Historical E14-E24/source-r2 checkpoint compatibility can be deleted | Medium | no supported leaf resumes from such a checkpoint; ad-hoc external usage was not established |
| Every dormant debug branch can be deleted safely | Not established | selected configs disable it, but it is interwoven with June pipeline code and was intentionally left outside the first pass |

The recommended first implementation should therefore execute Phases 0-3,
then Phase 4's explicit fixed-cache simplification, while retaining general
debug helpers. That gives a materially smaller repository and much cleaner key
file diffs without inventing functionality or altering the ten selected
experiments.
