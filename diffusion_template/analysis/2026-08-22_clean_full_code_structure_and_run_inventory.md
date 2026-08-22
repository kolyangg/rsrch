# `clean_full`: unified training code structure and run inventory

- Date: 22 August 2026
- Branch: `clean_full`
- Comet project: `aug-large-ds`
- Historical source boundary: branch `test` at
  `97e0364d6fa6ee6b1b8c3d99aa547805b18ad47f`
- Evidence: resolved Hydra configurations, source inspection, canonical local
  experiment records, shell syntax checks, import/compile checks, and the
  `clean_full` allowlist validator

## Finding

`clean_full` supports 17 selected experiment configurations from one checkout,
one active training entry point, one active trainer/model/pipeline stack, and
one Serv launcher. Scientific behavior is selected by the Hydra config; the
launcher accepts no ad-hoc Hydra arguments. The selected set covers the plain
PhotoMaker control, CL14/19/23/27/39, the latest six CL40-CL45 configs, E13,
BigCelebs E13, and the three scheduled BigCelebs dataset arms. [code][record]

The familiar core filenames are no longer the accumulating implementation.
`train.py`, `lora2.py`, `attn_processor_cleanest.py`, `branched_runtime.py`,
`lora2_helpers.py`, `base_trainer.py`, `sdxl_trainers.py`, and
`br_pipeline_helpers.py` are exact byte copies of `main_clean` commit
`2157eada14824d14019e80f9416e6d736c837306` in the local `rsrch_2Jun`
checkout. Supported configs instead target explicitly named `clean_full_*`
modules. Recent CL39-CL44 operations and CL45 PCGrad are further isolated in
small defaults-off extension modules. [code]

The old branch mixed active code with sealed per-job packages, archived
implementations, external model mirrors, generated reports, and broad Hydra
registries. This branch removes 2,263 tracked paths from the runtime checkout:
541 external-model paths, 695 sealed job-package paths, 182 config paths, 73
launcher paths, 67 other `src/` paths, 504 analysis/experiment/generated
evidence paths, 61 tool paths, and 140 miscellaneous paths. They remain
recoverable from Git history and branch `test`; they are not copied into the
new unified clone. [code]

Configuration composition and object-selection parity are verified for all 17
configs. A GPU forward/backward replay against every historical sealed source
snapshot was not run, so bitwise or numerical trajectory parity is **not
established**. [not established]

## 2 June compatibility boundary

The refactor uses two visible source layers. This is deliberate: an engineer
can open any traditional filename and see the exact 2 June implementation,
while an allowlisted run always enters the audited unified layer. [code]

| 2 June file retained byte-for-byte | Active supported replacement | Why the 2 June implementation is not the active base |
|---|---|---|
| `train.py` | `train_clean_full.py` | The active entry point must enforce the resolved trainable contract, validation-only PM0 path, and optimized collector controls before construction. |
| `src/model/photomaker_branched/lora2.py` | `clean_full_model.py` | The active class has fundamentally different hard-v1 ownership, batched reference conditioning, schema-v2 checkpoint manifests, alternate-base validation synchronization, telemetry/objective collection, and CL39-CL44 state. Subclassing the June class would hide replacements of `__init__`, training preparation, checkpointing, and most of `forward()` behind fragile overrides. |
| `src/model/photomaker_branched/attn_processor_cleanest.py` | `clean_full_attention_processor.py` | The supported processor uses audited target-Q/reference-KV hard routing, full-query soft routing, temporal-frequency routing, surface loss, and later independent arms; the June `__call__()` and batch/routing semantics are not a safe reusable core. |
| `src/model/photomaker_branched/branched_runtime.py` | `clean_full_branched_runtime.py` | Processor selection, install-time ownership, config propagation, and validation-time processor restoration changed as one contract. |
| `src/model/photomaker_branched/lora2_helpers.py` | `clean_full_model_helpers.py` | Trainable-name ownership, batched conditioning/cache, optimized processor lookup, telemetry, and auxiliary-loss collection changed together. |
| `src/trainer/base_trainer.py` | `clean_full_base_trainer.py` | The active loop adds step-zero/every-2,000 fixed-96 scheduling, validation-only execution, deferred face-quality handling, and strict checkpoint/Comet behavior. |
| `src/trainer/sdxl_trainers.py` | `clean_full_trainers.py` | The active trainer consumes BA telemetry/objectives, role-aware gradients, and the CL45 hook; its main `process_batch()` is no longer the June method plus a small local edit. |
| `src/pipelines/br_pipeline_helpers.py` | `clean_full_pipeline_helpers.py` | The active validation helper must install and synchronize the same audited processor/checkpoint semantics used by training. |

These are the cases where retaining the June implementation as an active
superclass would not be honest or maintainable. Exact copies remain importable
for historical investigation, but the clean launcher and config validator do
not select them. `model_v2_NS.py` remains unchanged from 2 June, and
`photomaker_branched_clean.py` remains close to its June structure because its
pipeline class contract is still reusable; only its helper import is redirected
to the active clean validation module. [code]

The recent config-specific code is split as follows:

| New module | Selected behavior |
|---|---|
| `clean_full_model_config.py` | Closed-schema expansion of one nested `clean_full_config` constructor input into shared hard-v1 attributes; excluded experiment defaults are quarantined here and cannot be selected by YAML. |
| `clean_full_model_contract.py` | Shared training preparation, trainable-role ownership, manifest construction, and schema-v2 checkpoint save/load. |
| `clean_full_excluded_objectives.py` | Quarantine mixin for older frozen-teacher, boundary-distillation, and predicted-x0 identity helpers; no allowlisted config enables them. |
| `clean_full_attention_extension_config.py` | Fail-closed selection and constructor wiring for mutually exclusive CL39-CL44 arms; also contains the retained-but-forbidden CL38 wiring. |
| `clean_full_attention_extensions.py` | Parameter-free CL39 null-key confidence, CL41 landmark canonical K/V, CL42 component-token memory, CL44 semantic-window gate, and the shared ramp used by CL40/43. |
| `clean_full_identity_conditioning.py` | CL40 `IdentityMotionProjector`, CL43 `IDAdaptiveModulation`, and CL41 landmark similarity-grid construction. |
| `src/trainer/clean_full_pcgrad.py` | CL45-only BA PCGrad projection; all other configs return before graph surgery. |

CL14/19/23/27 are the shared hard-v1 progression and remain in the active
processor/model stack. Moving that progression into subclasses without a
fixed-batch parity harness would risk changing checkpoint parameter names or
floating-point operation order; such parity is **not established**, so this
revision does not claim that deeper split. [code][not established]

## Model constructor input audit

The previous active `PhotomakerBranchedLora.__init__()` declared 226 flat
arguments after `num_inference_steps`. They were not all needed as independent
inputs. Across the 17 resolved configs, only 29 differed from their historical
Python defaults and only 18 varied between supported runs. The refactor
replaces all 226 parameters with one `clean_full_config` mapping and reduces
`clean_full_model.py` from 2,695 to 1,503 lines. [code]

The mapping has seven closed sections: `runtime`, `contract`, `architecture`,
`training_mask_feather`, `hardcase`, `frequency_surface`, and
`recent_extension`. Unknown sections/settings fail before model weights load.
The resolver recreates the same 226 flat values internally so existing runtime
attributes, processor wiring, manifests, and checkpoint names do not change.
All 17 pre/post resolved option maps matched exactly. [code]

| Usage across supported configs | Inputs |
|---|---|
| Common to all 17 | optimized conditioning flags; strict install/trainable-v2 contract; hard-v1/reference-only architecture; effective generic and PhotoMaker adapters; rank-128 hard-v1 branch |
| 12 configs (PM0 and CL14-CL45 ancestry) | two-cell training-mask feather; E13/BigCelebs arms keep zero |
| PM0 and CL19 | `soft_router` hard-case mode |
| CL23, CL27, CL39-CL45 | temporal-frequency hard-case mode |
| CL27 and CL39-CL45 | frequency-surface objective |
| CL39, CL40, CL41, CL42, CL43, or CL44 only | respectively one null-key, motion-projector, landmark-K/V, component-memory, adaptive-modulation, or semantic-window extension |
| CL45 only | PCGrad, which remains a trainer option rather than a model constructor input |
| No allowlisted config | identity auxiliary, boundary teacher, low-band contrastive/positive, attention ownership, shared learnable schedule, CL38 visibility ownership, alternate BA-v2/v3/v4, or identity-CA experiments |

The last group remains as fixed internal compatibility defaults because
retained inactive methods still read those attributes. It is no longer part of
Hydra's supported model surface; reviving one requires an explicit schema and
validator change rather than adding another constructor keyword. [code]

## Supported run allowlist

`src/configs/clean_full_runs.json` is the machine-readable authority. It pins
the config, dataset class, selected scientific feature, trainable-parameter
contract, canonical historical run, and immutable Comet key. `latest six` is
interpreted as CL40-CL45 because CL39 was named separately in the requested
scope. CL38 is excluded: it has a complicated multi-recovery history and is not
needed to reproduce a requested main configuration. [record]

| Config | Training data | Config-selected behavior | Canonical immutable Comet key |
|---|---|---|---|
| `PM0_original_photomaker_CL19_full96` | Cosmic config ancestry, no optimizer step | validation-only plain PhotoMaker V2; BA off | `74efd227d3f8488a98e83d815c77c07c` |
| `CL14_cosmic_joint_shadow_sa128_softmask_24k` | Cosmic Large Adapted | legacy hard route; two-cell training mask feather | `6fe0028be92242c38056b3d36665fdd6` |
| `CL19_cosmic_true_soft_fullquery_router_24k` | Cosmic Large Adapted | full-query `soft_router` | `cfeda7b55c174b3c83e8d40537ebb6dd` |
| `CL23_cosmic_temporal_frequency_router_24k` | Cosmic Large Adapted | temporal-frequency low/high route | `a9ec9c59d1624c68acb98737dcd65298` |
| `CL27_cosmic_frequency_surface_energy_24k` | Cosmic Large Adapted | CL23 plus frequency-surface loss and semantic occlusion | `dbfbf40c3bdd4f70bedc58bda3dfb9cd` |
| `CL39_cosmic_null_key_confidence_router_24k` | Cosmic Large Adapted | CL27 plus parameter-free null-key confidence | `b1ca0b3da679401c85b991f1bbdf0b2a` |
| `CL40_cosmic_identity_motion_projector_24k` | Cosmic Large Adapted | CL27 plus trainable rank-32 identity-motion projector | `1c2e0ac2fcae433db18f55de663b59ef` |
| `CL41_cosmic_landmark_canonical_kv_24k` | Cosmic Large Adapted | CL27 plus parameter-free landmark-canonical K/V | `b40179ef6a9d4dd6954f6d06d148069c` |
| `CL42_cosmic_component_token_memory_24k` | Cosmic Large Adapted | CL27 plus parameter-free component-token memory | `9613ca23f49f469b9bc0fda89055483d` |
| `CL43_cosmic_id_adaptive_modulation_24k` | Cosmic Large Adapted | CL27 plus trainable ID-adaptive modulation | `d29cbfa7927547c9ac71a8da0b583e33` |
| `CL44_cosmic_semantic_window_gate_24k` | Cosmic Large Adapted | CL27 plus parameter-free semantic-window gate | `42928f13f7ee41448d3d715231f8bb32` |
| `CL45_cosmic_ba_pcgrad_24k` | Cosmic Large Adapted | CL27 plus BA-only PCGrad optimization | `bfb129031773494f881ea629ced3fe60` |
| `E13_large_ds_joint_shadow_sa128_24k` | Large Dataset | joint BA/generic/PhotoMaker shadow-coadapter training | `1cc0a02371094b24a6a02a4cc649f10c` |
| `BC_E13_big_celebs_joint_shadow_sa128_24k` | BigCelebs v2 | E13 model; dataset-only transfer | `c138db7c41ae435c8a7560f40cf5f58d` |
| `BC_E13_ds1_repeatdepth_balanced_24k` | BigCelebs schedule ds1 | repeat-depth balanced schedule | `b5b23b0ca4b449bc8f4703d6a7334be1` |
| `BC_E13_ds2_scene_target_canonical_ref_24k` | BigCelebs schedule ds2 | ds1 plus scene-rich targets and canonical references | `5db54d7d4557487e94251656736843db` |
| `BC_E13_ds3_large_anchor_2to1_24k` | Large Dataset + BigCelebs ds2 | deterministic 2:1 Large-to-BigCelebs schedule | `43adf33cf7174e89b8fde1cdd640a052` |

All training configs select 24,000 optimizer steps as 12 epochs of 2,000
steps. Validation is step 0 and every 2,000 steps on the fixed 96-image
`manual_val` panel, one image per item, DDIM50, CFG 5. PM0 has only the step-0
validation event. Every supported BA run requires hard-v1 target Q/reference
K/V self-attention, `disable_branched_ca=true`, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`. [code]

## How a run starts

Run from `diffusion_template/`:

```bash
python tools/validate_clean_full_config.py --list

CONFIG_NAME=CL39_cosmic_null_key_confidence_router_24k \
RUN_NAME=CL39_clean_full_example \
bash launchers/active/run_clean_full_config_1gpu.sh
```

The execution flow is:

```text
run_clean_full_config_1gpu.sh
  -> validate_clean_full_config.py (allowlist + resolved-config contract)
  -> dataset-specific fail-closed preflight
  -> train_clean_full.py --config-name=<selected config>
  -> clean_full_trainers.PhotomakerLoraTrainer
  -> clean_full_model.PhotomakerBranchedLora
  -> MaskedDiffusionLoss + BranchedAttnProcessor
  -> fixed manual_val96 validation pipeline
  -> CometMLWriter immutable-key record
  -> deferred face-quality scorer after successful Accelerate exit
```

The launcher uses the resolved `train_dataset_name` to choose the Cosmic,
Large Dataset, BigCelebs, or scheduled BigCelebs preflight. It writes a
config-only plan to `saved/<run>/comet_experiment.json`; `CometMLWriter`
atomically augments that record with the live Comet key. The launcher polls the
same record and refuses to proceed silently when a 32-character key is not
registered. [code]

Operational paths and credentials are selected through the gitignored `.env`.
They are not scientific Hydra overrides. Before an operator submits the
launcher through Serv/MLS, they must inspect running and pending jobs and obey
the repository's A100 request ceiling. [code]

## Hydra structure

The 17 leaf configs share this retained ancestry:

```text
one_id_09Feb_testing.yaml
  -> one_id_rhca_apr2026_replay.yaml
  -> cosmic_large_initial_usage_rhca.yaml
  -> large_dataset_rhca_40k.yaml
  -> large_dataset_rhca_hard_v1_audited_20k.yaml
  -> large_dataset_joint_r128_24k.yaml
  -> E13_large_ds_joint_shadow_sa128_24k.yaml
  -> CL4_cosmic_joint_shadow_sa128_hygiene_24k.yaml
  -> CL9_cosmic_joint_shadow_sa128_refscale_24k.yaml
  -> CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml
```

CL19 adds `CL15_CL20_hardcase_base_24k.yaml`; CL23 inherits CL19; CL27
inherits CL23; CL39-CL45 are independent leaves over CL27. PM0 is a
validation-only leaf over CL19. BC_E13 and ds1-ds3 are dataset-only leaves over
E13. [code]

The only retained grouped fragments are:

- `trainer/photomaker_lora.yaml`;
- `model/photomaker_branched_lora2.yaml`;
- `writer/cometml.yaml`;
- `pipeline/pm_br_09Feb_testing.yaml`;
- `lr_scheduler/warmup_hold_cosine.yaml`;
- `transforms/only_instance.yaml` and
  `transforms/instance_transforms/image_1024.yaml`;
- `ddp/accelerate.yaml`;
- `datasets/clean_full_datasets.yaml`;
- `dataloaders/clean_full_dataloaders.yaml`;
- `metrics/clean_full_metrics.yaml`.

The dataset and metric registries contain only selectable objects for this
support boundary. The earlier all-dataset/all-metric registries were removed.
[code]

## Common files, classes, and functions used by training runs

The table names the executed public methods and the private methods that define
scientific or operational behavior. Small tensor/Python helpers called only by
the named methods are grouped with their owner rather than repeated as separate
rows. [code]

| File | Selected classes/functions | Runs |
|---|---|---|
| `train_clean_full.py` | `main()`, `_assert_expected_trainable_contract()`, `_print_trainable_summary()` | all; PM0 takes validation-only trainer branch |
| `tools/validate_clean_full_config.py` | `load_manifest()`, `compose_and_validate()`, `selected()`, `write_run_record()`, `main()` | all launcher starts |
| `launchers/active/run_clean_full_config_1gpu.sh` | config resolution, dataset dispatch, Accelerate launch, Comet polling, deferred finalization | all |
| `src/datasets/data_utils.py` | `get_dataloaders()`, `move_batch_transforms_to_device()`, `inf_loop()` | all |
| `src/datasets/collate.py` | `collate_fn()`, `collate_fn_val()` | all |
| `src/datasets/base_dataset.py` | `BaseDataset.__init__()`, `__len__()`, `preprocess_data()`, `_shuffle_and_limit_index()` | all training datasets |
| `src/trainer/clean_full_base_trainer.py` | `BaseTrainer.train()`, `_train_process()`, `_train_epoch()`, `_should_run_periodic_validation()`, `_evaluation_epoch()`, `_log_per_image_id_sim_table()`, checkpoint/logging methods; `_validate_only()` and `_validation_only_schedule()` for PM0 | all |
| `src/trainer/clean_full_trainers.py` | `PhotomakerLoraTrainer.__init__()`, `process_batch()`, `process_evaluation_batch()`, `_record_active_gradient_norms()`; inherited `SDXLTrainer._log_batch()` | all; training methods bypassed by PM0 |
| `src/trainer/clean_full_pcgrad.py` | `apply_ba_pcgrad_surrogate()` | CL45 only; imported but immediately inactive for every other config |
| `src/model/sdxl/original.py` | `SDXL.__init__()`, `compute_time_ids()`, `encode_prompt()` | all |
| `src/model/photomaker_path.py` | `resolve_photomaker_path()` | all |
| `src/model/photomaker_branched/clean_full_model.py` | `PhotomakerBranchedLora.__init__()`, `forward()`, `_sample_training_timesteps()`, prompt/bbox/reference-latent helpers | all; `forward()` only training configs |
| `src/model/photomaker_branched/clean_full_model_contract.py` | `CleanFullModelContractMixin.prepare_for_training()`, `get_trainable_params()`, `assert_trainable_contract()`, architecture manifest, and schema-v2 save/load methods | all |
| `src/model/photomaker_branched/clean_full_model_helpers.py` | trainable-name/role selection, `configure_branched_trainables()`, `assert_branched_trainable_contract()`, processor installation, `prepare_branched_training_inputs()`, `_prepare_branched_training_inputs_batched()`, `run_branched_forward_pass()`, telemetry and selected auxiliary collection | all BA training configs |
| `src/model/photomaker_branched/clean_full_branched_runtime.py` | `select_branched_processor_names()`, `patch_unet_attention_processors()`, `two_branch_predict()` | all BA training and BA validation paths |
| `src/model/photomaker_branched/clean_full_attention_processor.py` | `BranchLoRALinear`, `_clone_effective_linear()`, `BranchedAttnProcessor.__init__()`, `init_from_attention()`, `__call__()`, Q/K/V helpers, mask/router helpers | all BA configs |
| `src/model/photomaker_branched/clean_full_attention_extension_config.py` | `resolve_attention_extensions()`, `attention_extension_kwargs()` | all processor installation calls; only CL39-CL44 returns an enabled recent arm |
| `src/model/photomaker_branched/clean_full_attention_extensions.py` | `RecentAttentionExtensionsMixin` selected helpers | CL39-CL44 according to config |
| `src/model/photomaker_branched/clean_full_identity_conditioning.py` | `IdentityMotionProjector`, `IDAdaptiveModulation`, `similarity_grid_from_landmarks()` | CL40, CL43, and CL41 respectively |
| `src/model/photomaker_branched/model_v2_NS.py` | `PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken`, `QFormerPerceiver`, `FacePerceiverResampler`, `PerceiverAttention`, `FuseModule`, `MLP` and forwards | all |
| `src/model/photomaker_branched/resampler.py` | `FacePerceiverResampler`, `PerceiverAttention`, `FeedForward()`, `reshape_tensor()` | all |
| `src/model/photomaker_branched/insightface_package.py` | `FaceAnalysis2`, `create_face_analyzer()`, `analyze_faces()` | PhotoMaker identity conditioning in all |
| `src/face_subject_selector.py` | `select_subject_face()`, `face_bbox()`, `bbox_iou()`, `SubjectSelection.to_dict()` | training conditioning and validation subject selection |
| `src/loss/diffusion_loss.py` | `MaskedDiffusionLoss.forward()`, `_masked_face_mse()` | all training configs |
| `src/lr_schedulers/lr_schedulers.py` | `WarmupHoldCosineLR.__init__()` plus inherited `step()` | all training configs |
| `src/metrics/tracker.py` | `MetricTracker.reset()`, `update()`, `result()` | training and validation |
| `src/logger/logger.py` | `setup_logging()` | all |
| `src/logger/cometml.py` | `CometMLWriter.__init__()`, `_write_experiment_record()`, `set_step()`, scalar/image/table logging and config flattening | all |
| `src/utils/init_utils.py` | random/worker seeds, saving and logging setup | all |
| `src/utils/model_utils.py` | `import_model_class_from_model_name_or_path()`, `cos_sim()` | startup and ID metrics |
| `src/utils/io_utils.py` | `read_json()` | startup logging config |

`MaskedDiffusionLoss` is the only permitted loss class. `train_clean_full.py` rejects any
other `loss_kind`. The trainable contract is 2,240 tensors / 219,217,920
parameters except CL40 (2,348 / 223,272,960) and CL43 (2,384 / 222,596,736),
whose selected modules add parameters. [code]

## Dataset-specific files, classes, and functions

| Dataset/config family | Files and executed objects |
|---|---|
| CL14-CL45 | `tools/datasets/preflight_cosmic_cl.py`: `main()`, `short_side()`, `letterbox_scale()`, `area_fraction()`, `percentile()`; `src/datasets/cosmic_large_adapted.py`: `CosmicLargeAdaptedTrain` construction/item methods plus `build_cosmic_prompt()`, image/target/accept-list helpers; `reference_frame.compose_target_frame_reference()` and `reference_policy.valid_bbox()` |
| E13 | `tools/datasets/preflight_large_dataset.py`: `main()`; `src/datasets/large_dataset.py`: `LargeDatasetTrain.__init__()`, `_load_image()`, `__getitem__()` |
| BC_E13 | `tools/datasets/preflight_big_celebs.py`: `main()` plus hash/bbox/path/decode helpers; `src/datasets/big_celebs.py`: `BigCelebsTrain.__init__()` and inherited Large Dataset item path |
| BC_E13 ds1-ds3 | BigCelebs preflight above plus `preflight_bc_e13_dataset_schedule.py`; `BigCelebsE13ScheduledTrain` construction, schedule validation, resume-position validation, image load and item methods; `bc_e13_schedule_policy.py` hash, manifest, bbox/face, directionality and canonical-reference helpers |
| BC_E13 ds3 only | scheduled path above plus Large Dataset preflight and Large Dataset manifest/image lookup for its 2:1 mixture |
| all validation | `ManualPhotoMakerValDataset.__init__()`, `_resolve_prompt()`, `__len__()`, `__getitem__()` |

CL27 and CL39-CL45 enable Cosmic's deterministic semantic-occlusion branch.
CL14/CL19/CL23 have zero semantic-occlusion probability. Dataset schedule
rows, not run-name shell cases, select the ds1/ds2/ds3 target/reference source.
[code]

## Config-specific model and optimization functions

| Config(s) | Additional executed attention/model/trainer path |
|---|---|
| CL14, E13, BC_E13 arms | `BranchedAttnProcessor._call_legacy()` and `_prepare_mask()` |
| CL19 | `_call_hardcase()`, `_normalized_halves()`, `_binary_mask()`, `_soft_router_mask()`, `_full_target_lanes()`, `_reference_target_out()`, `_finish_full_router()` |
| CL23 | CL19 functions plus `_gaussian_split()` and `_progress()` |
| CL27 | CL23 functions plus `_frequency_surface_loss()`, `_masked_mean_square()` and `collect_frequency_surface_aux_loss()` |
| CL39 | CL27 plus `RecentAttentionExtensionsMixin._null_key_confidence()` in `up_blocks.0/1` |
| CL40 | CL27 plus `IdentityMotionProjector.forward()` and the extension mixin's `_step_ramp()` in `up_blocks.0/1` |
| CL41 | CL27 plus the extension mixin's `_landmark_rows()`/`_canonical_reference_out()` and `similarity_grid_from_landmarks()` |
| CL42 | CL27 plus the extension mixin's `_landmark_rows()` and `_component_memory_correction()` |
| CL43 | CL27 plus `IDAdaptiveModulation.forward()` and the extension mixin's `_step_ramp()` |
| CL44 | CL27 plus `RecentAttentionExtensionsMixin._semantic_window_scale()` |
| CL45 | CL27 model path plus `clean_full_pcgrad.apply_ba_pcgrad_surrogate()` and trainer `_gradient_norm()` |
| PM0 | no branched processor call and no optimizer step; validation pipeline uses the PhotoMaker V2 path |

The recent extension validator permits only the selected CL39-CL44 extension
for its leaf config; it rejects accidental combinations. PCGrad is allowed
only for CL45. CL40 and CL43 construct trainable modules; CL39/41/42/44 are
parameter-free extensions. [code]

## Validation and post-training files, classes, and functions

| File | Used objects |
|---|---|
| `src/pipelines/photomaker_branched_clean.py` | `PhotomakerBranchedPipeline.from_pretrained()`, `PhotoMakerStableDiffusionXLPipeline.__call__()`, `encode_prompt_with_trigger_word()`, `retrieve_timesteps()` |
| `src/pipelines/clean_full_pipeline_helpers.py` | pipeline construction; face analyzer/ID setup; reference latent/mask, generated mask and ID preparation; validation U-Net adapter mode; branched setup/step; denoising step; cleanup |
| `src/model/photomaker_branched/branch_helpers.py` | `prepare_mask4()` |
| `src/utils/auto_bbox_gen.py` | `AutoGenBboxStore.get()` and `ensure()` when an automatic bbox cache entry is missing |
| `bbox_utils/generate_bboxes.py`, `bbox_utils/visualize_bboxes.py` | YOLO face detector load, face record, and annotated bbox output for an automatic-bbox miss |
| `src/metrics/text_sim.py` | `TextSimMetric` construction, device movement and call |
| `src/metrics/id_sim_metric.py` | `IDSimBest` and `IDSimMaskMatched` construction/calls |
| `src/metrics/aligner.py`, `src/utils/id_utils.py` | InsightFace metric detection/embedding path |
| `src/metrics/face_quality_validation.py` | `FaceQualityValidationSession.add_batch()` and deferred `finalize()` staging |
| `tools/comet/finalize_deferred_face_quality.py` | record/manifests/hashes, scorer and backfill subprocess orchestration |
| `tools/inference/calculate_face_quality_metrics.py` | canonical face detection/crop and seven face-quality measures |
| `tools/comet/backfill_face_quality_metrics.py` | Comet metric/table/asset logging and verification |

Whether a historical validation event generated a new bbox or used its
snapshot-local cache cannot be determined solely from current code. That
per-event automatic-bbox branch is **not established**. [not established]

## Files and code not used by supported runs

### Removed from the branch

The following are absent from `clean_full` and therefore cannot be imported by
a supported run:

- all tracked `serv_run_packages/**` sealed source copies;
- `_other_models/**`, `_old/**`, and `compare/**` external/reference trees;
- archived and per-suite launchers, leaving only the unified launcher active;
- 182 historical, ablation, smoke, inference, and alternate-writer configs;
- source backup trees (`_backup`, `_old*`), old pipelines/trainers, PhotoMaker
  V1, alternate SDXL/PhotoMaker models, alternative loss files, ClearML/W&B/
  console writers, alternative processor-family files, unused datasets, and
  generic transform modules;
- generated outputs, report assets, Comet downloads, old run packages, and
  unrelated experiment JSONs;
- old dataset variants and unused dataset registries;
- old one-off validators, dataset builders, inference helpers, and report
  snapshots.

The removed source remains in Git history. “Removed” means out of this runtime
branch, not destroyed historical evidence. [code]

### Retained 2 June reference modules

The eight exact 2 June files in the compatibility table are not selected by
any supported config or by the unified launcher. They are retained at their
traditional import paths as readable historical reference code, not as a
second supported runtime. Directly invoking `train.py` is therefore outside
the `clean_full` support contract; use `train_clean_full.py` only through the
validated launcher. [code]

### Active modules with inactive branches

Some scientific history remains inside the large active model and processor
modules because extracting it while guaranteeing checkpoint and numerical
parity is a separate architecture refactor. These branches are not selected by
any of the 17 allowlisted configs:

- `BranchedCrossAttnProcessor` in `clean_full_attention_processor.py`; all configs
  require unchanged Diffusers cross-attention;
- residual-v2, anchored-v3, and query-adaptive-v4 constructor branches in
  `clean_full_branched_runtime.py`; the runtime rejects every architecture except
  `hard_replace_v1` before processor selection;
- CL38 visibility-ownership-v2 and earlier CL28-CL37 helpers within
  `BranchedAttnProcessor`: learnable frequency schedule, low-band contrastive/
  positive objectives, attention ownership, ROI teacher, visibility-balanced
  routing, clean memory, and high-resolution/anchored ROI routes;
- `clean_full_excluded_objectives.ExcludedObjectiveCompatibilityMixin`
  boundary-teacher, frozen-CL19 prediction, native-PhotoMaker prediction,
  low-noise identity reward, ArcFace/DINO/patch identity auxiliary,
  wrong-reference, and related predicted-x0 helpers. The active class retains
  this mixin only so fail-closed compatibility branches still resolve;
- `clean_full_model_helpers.py` collectors for visibility ownership, schedule anchor,
  low-band contrastive/positive, attention ownership, and ROI teacher;
- inactive generic methods inherited from `SDXL` and `SDXLTrainer` that are
  overridden by PhotoMaker subclasses;
- `BaseTrainer` resume/from-pretrained paths for canonical cold starts, and
  training-loop methods for PM0;
- pipeline wrapper methods `_prepare_*`, `_select_*`, `_run_*`, and
  `_save_step_previews()`; the active call path uses the equivalent free
  functions in `clean_full_pipeline_helpers.py`;
- `rescale_noise_cfg()` because `guidance_rescale=0`;
- debug-preview work in `debug_helpers.py` because `val_debug=false`;
- in-process PyIQA scoring because every config selects deferred scoring;
- Comet audio/text/histogram/general-asset convenience methods not called by
  training;
- `io_utils.write_json()`, `data_utils.get_bigger_crop()`/
  `get_crop_values()`, and unused reference-policy transformations;
- data-dependent optional branches such as Cosmic target body crops and cached
  automatic bbox generation.

The overwritten `CustomLinearLR`, unselected `IDSimMax`, and generic
non-PhotoMaker `Resampler` were removed from their active modules. [code]

### Retained operational tools that do not execute in a training job

`tools/comet/comet_experiment.py`, export/download/report builders,
`tools/reports/publish_report.py`, and `tools/dropbox/upload_to_dropbox.py` are
operator tools. They are retained for inspecting canonical experiments and
publishing reports, but neither the training entry point nor the unified
launcher imports them (except the explicit deferred finalizer listed above).
Documentation and `.env.example` are also non-executable. [code]

## Unified-codebase plan for sealed historical jobs

Historical jobs used immutable sealed snapshots. The unified approach keeps
immutability as provenance while removing source forks:

1. **One reviewed source commit.** Every new supported job uses the same
   `clean_full` commit. A job no longer selects a copied Python package.
2. **Config-only scientific selection.** `CONFIG_NAME` must appear in
   `clean_full_runs.json`; its leaf YAML contains the scientific delta.
   Run-name dispatch and ad-hoc Hydra overrides are rejected.
3. **Fail-closed composition.** `validate_clean_full_config.py` checks the
   exact object targets, dataset class, trainable ownership, BA invariants,
   validation contract, exclusive extension, scheduler, optimizer, writer,
   and loss before dataset or Comet work.
4. **Sealed inputs remain sealed.** `.env` selects machine-local paths, while
   preflights verify manifest/content hashes, image shape, identity/reference
   constraints, schedule mode, and the fixed subject-v2 embedding hash.
5. **Immutable run provenance.** The config-only plan and live Comet key share
   `saved/<run>/comet_experiment.json`. A follow-up hardening step should also
   upload the resolved Hydra config, source commit/archive hash, environment
   lock hash, model hashes, dataset seals, installed processor map, and
   trainable-name manifest as a Comet asset.
6. **Parity gates before claiming historical equivalence.** For each config,
   compare the sealed historical package with `clean_full`: resolved config,
   selected classes, installed processors, trainable names/counts, one fixed
   batch of conditioning/masks/timesteps/loss/gradients/update, checkpoint
   save/load, and fixed-seed step-zero validation. A full rerun must retain the
   standard step-0/every-2,000 manual_val96 contract.
7. **Keep config-specific extensions explicit.** CL39-CL44 helper operations,
   their install-time selection, CL40/43 trainable identity modules, and CL45
   optimizer projection are now separate defaults-off modules. A later phase
   may move the CL27 core and delete older inactive branches only after a
   fixed-batch parity harness is approved and passes without changing
   checkpoint parameter names or operation order.
8. **Rollback remains Git, not copied runtime trees.** Exact excluded-run
   recovery uses branch `test` or the recorded historical commit. It does not
   reintroduce sealed packages into `clean_full`.

The first four steps and the recent-arm part of step 7 are implemented in this
branch. Step 5 is implemented for the core record/key but not the proposed
expanded provenance asset. GPU parity gates and the deeper CL27 extraction
remain future work. [code][not established]

## Verification performed

From `diffusion_template/`:

```bash
bash -n launchers/active/run_clean_full_config_1gpu.sh

/home/kolyangg/anaconda3/envs/photomaker/bin/python -m py_compile \
  train.py train_clean_full.py \
  $(find src tools bbox_utils -type f -name '*.py' -print)

for path in \
  train.py \
  src/model/photomaker_branched/lora2.py \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/model/photomaker_branched/branched_runtime.py \
  src/model/photomaker_branched/lora2_helpers.py \
  src/trainer/base_trainer.py \
  src/trainer/sdxl_trainers.py \
  src/pipelines/br_pipeline_helpers.py; do
  cmp "$path" "/home/kolyangg/rsrch_2Jun/diffusion_template/$path"
done

while IFS= read -r config; do
  /home/kolyangg/anaconda3/envs/photomaker/bin/python \
    tools/validate_clean_full_config.py --config-name "$config"
done < <(/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  tools/validate_clean_full_config.py --list)
```

All eight compatibility files matched the local 2 June checkout byte-for-byte.
The method bodies moved into the model-contract, excluded-objective, and recent
attention mixins also matched their pre-refactor `clean_full` sources after
normalizing trailing blank lines.
All 17 configurations returned `status: ok`; their resolver outputs matched
the pre-refactor 226-option snapshots exactly (17 x 226 comparisons), and the
resolved options passed the model-level contract checks on a lightweight
object. The constructor signature exposes only `clean_full_config` after
`num_inference_steps`; unknown sections and settings were also confirmed to
fail closed. Shell syntax, active-module imports, and Python compilation
succeeded. These checks do not download model weights, instantiate a full GPU
U-Net, process a real dataset batch, submit an MLS job, or compare generated
images. [code][not established]

## Confidence and limitations

| Claim | Confidence | Basis |
|---|---|---|
| Support allowlist, canonical run names, and immutable keys | High | machine-readable manifest and experiment records |
| Resolved object targets, dataset classes, invariants, trainable counts, and validation cadence | High | all 17 configs composed and passed exact checks |
| Current source import/syntax integrity | High | retained-tree compile plus targeted imports |
| Traditional core files equal the 2 June local baseline | High | byte-for-byte `cmp` against `rsrch_2Jun` commit `2157eada...` |
| Constructor/config refactor preserves supported inputs | High | exact 17 x 226 pre/post resolved-option comparison and model-contract smoke check |
| Recent-arm selection remains mutually exclusive and config-driven | High | resolved-config validator plus separate defaults-off modules |
| Common and config-specific call paths | High | direct source inspection and config predicates |
| Historical sealed-snapshot numerical equivalence | Not established | no paired GPU replay was run |
| Exact data-dependent optional branch frequency | Not established | requires dataset/cache/runtime trace |
| Performance improvement from pruning | Not established | no throughput or disk-startup benchmark was run |
