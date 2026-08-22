# 2 June key-file rebase: implementation record

**Date:** 22 August 2026  
**Branch/worktree:** `clean`, `/home/kolyangg/rsrch_clean`  
**Baseline:** `2157eada14824d14019e80f9416e6d736c837306`  
**Scope:** the ten retained E13/BC_E13/CL14/CL14_CA/CL18/CL19/CL20/CL23/CL27/CL39 recipes

## Outcome

The shared key files now use the 2 June source as their literal baseline. Old
comments, debug code, generic branches, and apparent bloat were retained. New
experiment logic is either in a named extension module or behind a short
`e13_family_contract` dispatch.

Against the 2 June commit, the ten-file key set now has only `+123/-0` lines.
The previous clean-branch state was `+1,999/-265`; the intermediate aggressive
cleanup was `+1,076/-2,771` and did not satisfy the requirement to preserve the
old source text.

Five high-value shared/compatibility files are byte-identical to 2 June:

| File | SHA-256 (2 June and current) |
|---|---|
| `src/model/photomaker_branched/lora2.py` | `b1bb786ec2c6c50f2157b0fe3861ab88cb696fe2ac7f3ae71a5798d7563011c0` |
| `src/model/photomaker_branched/lora2_helpers.py` | `73c0c57995d1bfe1bb29bf865614a3b740c13a18de8096f0cbb10a274ff6b579` |
| `src/model/photomaker_branched/attn_processor_cleanest.py` | `5333ecea010ac1e0e2b28f9a4afd23112f6e6a3e9e6d407461fef035045033c4` |
| `src/pipelines/photomaker_branched_clean.py` | `9c7197bdfa5ae39917d27c9a87d483efd38a113fcabeb208230898f817ebdaec` |
| `src/model/photomaker_branched/model.py` | `f143276b205721c0cd49a839555158aa9d837e683a9295721d548d58ea34a161` |

## Exact additions in shared key files

These are the only differences in the ten shared key files. Line numbers refer
to the current file; function names are the stable audit locator.

| Shared file | Diff vs 2 June | New code location | Purpose |
|---|---:|---|---|
| `train.py` | `+2/-0` | `main`, lines 297-298 | Ask a model that owns a strict contract to verify optimizer membership. Legacy models are unchanged. |
| `lora2.py` | `+0/-0` | none | Exact 2 June file. Selected configs no longer target it. |
| `lora2_helpers.py` | `+0/-0` | none | Exact 2 June file. Selected configs no longer import it. |
| `attn_processor_cleanest.py` | `+0/-0` | none | Exact 2 June attention implementation, including old comments and branched CA. |
| `branched_runtime.py` | `+30/-0` | `patch_unet_attention_processors`, lines 50-61; `two_branch_predict`, lines 293-326 | Dispatch only an object marked `e13_family_contract` to the selected runtime. The complete 2 June generic runtime remains below each dispatch. |
| `base_trainer.py` | `+44/-0` | imports 11-18; training guards 376-396 and 816-818; alternate-base validation 516-552; face-quality calls 600-708 | Additive calls to the isolated validation helper. Every June line remains unchanged; selected-only trainer attributes are supplied by `e13_trainer.py`. |
| `sdxl_trainers.py` | `+5/-0` | `PhotomakerLoraTrainer.process_batch`, lines 289-291; metric sample construction, lines 796-797 | Flatten selected telemetry into the loss batch and bind subject-v2 metrics to the exact generated/reference boxes. |
| `diffusion_loss.py` | `+13/-0` | `MaskedDiffusionLoss.forward`, lines 60-72 | Add an auxiliary loss only when the selected model supplies it; the old return and all old loss classes remain. |
| `br_pipeline_helpers.py` | `+29/-0` | `ensure_face_analyzer`, lines 148-158; `run_branched_setup`, lines 509-523; `build_pipeline_from_pretrained`, lines 1155-1157 | GPU face analysis, the sealed one-spatial-reference/bbox route, and selected runtime-setting transfer. June branches remain intact below the guards. |
| `photomaker_branched_clean.py` | `+0/-0` | none | Exact 2 June file, including `branched_attn_end_step`, mask switches, helper methods, and comments. |

## Where the experiment code now lives

The selected config target is changed at
`src/configs/model/photomaker_branched_lora2.yaml:1` from the historical
`lora2.PhotomakerBranchedLora` to
`e13_model.PhotomakerBranchedLora`. This is the single model-routing change.
The trainer target similarly changes at
`src/configs/trainer/photomaker_lora.yaml:1` to the small selected subclass
`e13_trainer.E13PhotomakerLoraTrainer`.

| Extension file | Exact active region | Experiment responsibility |
|---|---|---|
| `src/model/photomaker_branched/e13_model.py` | class `PhotomakerBranchedLora`, lines 34-252 | Concise selected model construction and forward orchestration. |
| `src/model/photomaker_branched/e13_training_helpers.py` | lines 18-422 | Fail-closed processor installation; batched frozen text/PhotoMaker/VAE conditioning; one selected branched forward. |
| `src/model/photomaker_branched/e13_attn_processor.py` | class `E13BranchedAttnProcessor`, lines 13-72 | Pins the unchanged June equations to independent rank-128 LoRA Q/K/V and caches exact resized masks. |
| `src/model/photomaker_branched/e13_runtime.py` | processor map lines 15-206; sealed two-branch prediction lines 251-554 | Native CA plus E13 SA installation, optional CL14_CA processors, paired reference noise, true scheduler progress, and doubled-batch UNet execution. |
| `src/model/photomaker_branched/e13_contract.py` | lines 104-525 | Leaf settings, exact trainable ownership, optimizer groups, manifest, and schema-v2 checkpoint save/load. |
| `src/model/photomaker_branched/e13_objectives.py` | lines 20-268 | CL18 cross-view loss, CL27 surface loss, and enabled-only telemetry collection. |
| `src/model/photomaker_branched/hardcase_attn_processor.py` | lines 15-481 | CL19 soft router, CL23 temporal-frequency route, CL27 surface state, and CL39 null-key confidence. |
| `src/model/photomaker_branched/residual_identity_ca_processor_v3.py` | lines 37-330 | CL14_CA's bounded rank-64 residual identity-token cross-attention. |
| `src/trainer/e13_trainer.py` | class `E13PhotomakerLoraTrainer`, lines 6-18 | Consumes the three selected trainer settings without changing the June constructor. |
| `src/trainer/validation_model_helpers.py` | lines 8-129 | Disabled parameter-touch source, strict alternate-base processor transfer, PhotoMaker-default shadow/restore, and deferred face-quality session lifecycle. |
| `src/pipelines/photomaker_branched_subject_v2.py` | lines 21-127 | Declared-face PhotoMaker conditioning and subject-v2 identity metric binding. |
| `src/datasets/cl20_hardcase_curriculum.py` | complete module | CL20's training-only curriculum; no model-architecture change. |

`src/model/photomaker_branched/model.py` is also restored byte-for-byte from
2 June. The selected PMv2 path does not instantiate it, but the unchanged June
pipeline imports `PhotoMakerIDEncoder` eagerly, so removing the module would
make the selected pipeline itself unimportable.

Two other historical branches remain visible but selected-inactive:
`prepare_ref_mask` lazily imports removed `create_mask_ref.py` only when
`auto_mask_ref=true`, and the trainer lazily imports removed
`utils/auto_bbox_gen.py` only when `automatic_bboxes=true`. The E13 pipeline
guard forces direct bbox masks and all retained configs leave automatic bbox
generation off. These branches document June behavior but are intentionally
not supported by the clean recipe closure.

### Recipe-to-code map

| Recipe | Code beyond common E13 |
|---|---|
| E13 / BC_E13 | No leaf architecture extension. Dataset selection differs. |
| CL14 | `e13_settings.ba_training_mask_feather=2`, consumed by `e13_training_helpers._bbox_to_target_mask`. |
| CL14_CA | `ResidualIdentityCrossAttnProcessorV3`; installed by `e13_runtime.patch_unet_attention_processors`. |
| CL18 | Alternate reference data plus `e13_objectives._crossview_consistency_loss`. |
| CL19 | `HardcaseBranchedAttnProcessor` in `soft_router` mode. |
| CL20 | `CL20HardcaseCurriculumTrain`; attention/model remain CL14. |
| CL23 | `HardcaseBranchedAttnProcessor` in `temporal_frequency` mode. |
| CL27 | CL23 processor plus enabled surface terms in `e13_objectives`. |
| CL39 | CL27 processor plus parameter-free null-key confidence in the same hard-case module. |

## Historical bloat deliberately retained for later work

This section is advisory only. None of it should be removed as part of this
rebase; doing so would again make the key files diverge from the requested
baseline.

- `train.py`: Cosmic-specific config mutation, legacy BA selector plumbing,
  dynamic loss-kind replacement, and verbose trainable summaries.
- `lora2.py`: the large generic constructor, duplicated schedule controls,
  scalar conditioning helpers, debug comments, legacy checkpoint paths, and
  multiple training modes.
- `lora2_helpers.py`: broad trainability fallbacks, scalar per-sample
  conditioning, broad exception handling, and generic processor assumptions.
- `attn_processor_cleanest.py`: legacy branched cross-attention, strict-routing
  alternatives, repeated mask checks, reference-lane output computation, and
  old debug/TODO comments.
- `branched_runtime.py`: top-k processor selection, legacy SA/CA toggles,
  ID side channels, full-debug blocks, branch-preview outputs, processor
  restoration, and Gaussian-mask utility.
- `base_trainer.py`: zero-valued DDP parameter touching, generic alternate-base
  copy fallbacks, verbose validation image logging, and broad exception guards.
  Selected runs skip the expensive touch and unrequested gradient norms via
  the two concise guards; the old behavior remains the default.
- `sdxl_trainers.py`: automatic-bbox generation/regeneration, image debug
  output, broad compatibility paths, and per-loss synchronization.
- `diffusion_loss.py`: `DiffusionLoss` and `BlendedMaskedDiffusionLoss` are not
  used by the ten retained recipes but are kept exactly.
- `br_pipeline_helpers.py` and `photomaker_branched_clean.py`: automatic
  reference masks, dynamic generation masks, imported mask files,
  `branched_attn_end_step`, pose-forcing branches, debug previews, and the
  duplicated class helper API.
- `model.py`: the PMv1 compatibility implementation is selected-inactive, but
  is retained while the exact June pipeline keeps its eager PMv1 import.

If a later cleanup is approved, each item can be removed in a separate commit
after this June-baseline checkpoint. It should not be mixed with experiment
architecture changes.

## Verification

- AST parsing and isolated-cache `compileall` pass for all 55 retained Python
  files (including `train.py`), without creating repository bytecode files.
- All 24 retained Hydra YAMLs parse; all 22 local defaults edges exist; and all
  15 local `_target_` values resolve statically to their source definitions.
- Every retained local import resolves except the two selected-inactive lazy
  June fallbacks documented above. All three retained launcher/package shell
  scripts pass `bash -n`, and `git diff --check` passes.
- `tools/verify_cl14_generation_parity.py` passes. It seals the exact fixed-96
  inputs, the two June-based pipeline files plus their small hooks, and the
  selected `e13_runtime.two_branch_predict`; that selected function retains the
  prior seal `57bc575b...e17a8f01`.
- The five byte-identical files above were checked by independent SHA-256 of
  `git show 2157...:<path>` and the working-tree file.
- Full Hydra validators and Torch processor fixtures could not be rerun in this
  local shell because neither configured `photomaker` environment is present.
  They had passed immediately before this source-layout rebase; no training or
  validation job was launched.
