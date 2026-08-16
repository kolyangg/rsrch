# Serv `nasilaev/` can recover about 1.56 TB by retaining one checkpoint per inactive run and removing generated images

**Date:** 13 August 2026  
**Scope:** Read-only inspection of `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/` only.  
**Evidence cutoff:** Live sizes and MLS status on 13 August 2026. No Serv file was created, changed, or deleted.

## Executive conclusion

[measured] Serv's NFS mount had only `12.48 GB` available (`100%` used) at the updated cutoff. Measured content under `nasilaev/` is about `1.891 TB` (`1.761 TiB`; the active runs are still growing). [measured] Across 120 inactive checkpoint-bearing run directories, checkpoint files occupy `1,259.10 GB`. Keeping only the newest full resumable checkpoint in each directory retains `83.82 GB` and recovers **`1,175.28 GB`**. Removing all identified generated validation/derived images recovers another **`383.21 GB`**. These sets do not overlap, so the directly measured combined recovery is **`1,558.49 GB` (`1,451.46 GiB`)**.

Do **not** touch the active CL14_CA or CL21-CL26 runtime directories listed below. The earlier `159.56 GB` high-confidence cleanup list is largely a subset of the new checkpoint/image totals and must not be added again. Comet stores metrics/images, not a guaranteed restorable copy of checkpoint files; checkpoint pruning is therefore irreversible unless the files are archived elsewhere.

## Exact checkpoint-retention calculation

The calculation searched inactive `saved/` and `saved_continuations/` trees for `checkpoint*.pth` and `weights*.pth`. It excluded the eight live scientific roots. “Last” is the newest file timestamp within each run directory; every one of the 120 directories has at least one full checkpoint.

| Retention policy for inactive runs | Files before pruning | Retained | Recovered |
|---|---:|---:|---:|
| Keep newest full `checkpoint*.pth` only; remove all other full and weights files | 2,388 files / `1,259.10 GB` | 120 files / `83.82 GB` | **`1,175.28 GB` (`1,094.57 GiB`)** |
| Keep newest full checkpoint **and** newest weights-only file | 2,388 files / `1,259.10 GB` | 240 files / `113.50 GB` | `1,145.60 GB` (`1,066.92 GiB`) |

The first policy answers the requested “keep only last saved checkpoint” scenario. It preserves optimizer/scheduler state for one endpoint per run but removes the smaller inference-only weights copy. If convenient inference loading matters, the second policy costs only `29.68 GB` more across all 120 directories.

| Main location | Inactive checkpoint data | Recovered if one full checkpoint is retained |
|---|---:|---:|
| `runtime_sources_cl1_cl3_v1/` | `295.40 GB` | `276.94 GB` |
| `rsrch/` | `281.56 GB` | `264.48 GB` |
| `runtime_worktrees/` | `253.16 GB` | `236.83 GB` |
| `rsrch_test/` | `151.08 GB` | `136.51 GB` |
| `runtime_sources_cl15_cl20_v1/` | `129.17 GB` | `121.09 GB` |
| `runtime_sources_e19_e24_v3/` | `127.63 GB` | `119.65 GB` |
| Redundant CL14 seed under `runtime_sources_cl14_ca_v1/` | `21.10 GB` | `19.78 GB` |

## Generated validation and derived images

The image total includes PNG/JPEG/WebP files in inactive run `saved/` trees and known generated-output/staging locations. It excludes datasets, reference inputs, report source assets, and all eight active scientific roots.

| Image location/type | Files | Recoverable size |
|---|---:|---:|
| Images under inactive `saved/` / `saved_continuations/` trees | 168,490 | `252.88 GB` |
| └ canonical `val_images/` panels | 116,554 | `175.20 GB` |
| └ face-quality image copies | 44,736 | `66.96 GB` |
| └ other images inside saved run outputs | 7,200 | `10.71 GB` |
| Completed subject-v2 backfill staging (`analysis_jobs/`) | 66,177 | `100.29 GB` |
| Historical `face_quality_staging/` | 4,992 | `7.49 GB` |
| Completed `analysis_sidecars/` | 3,780 | `4.62 GB` |
| Historical `full_validation_results/` and the two large `ppr_*` result folders | 4,448 | `5.09 GB` |
| Inactive `hm_debug/` and `outputs/` image files | 9,244 | `12.83 GB` |
| `analysis_contact_sheets/` | 20 | `0.02 GB` |
| **Total** | **257,151** | **`383.21 GB` (`356.89 GiB`)** |

The images are already represented by immutable Comet assets or completed local reports according to the handoff, but deleting every raw panel removes convenient offline visual review and exact-pixel replay inputs. Preserve any selected report figures and reproduction seals locally before wholesale removal.

## What actually consumes the space

The Python source is not the problem. The canonical `rsrch_test` checkout excluding `saved/`, `saved_continuations/`, and debug output is under roughly `0.5 GB` including its small validation/reference tree. The large footprint is:

| Category | Approximate size | Interpretation |
|---|---:|---|
| Inactive full + weights checkpoint files | `1,259.10 GB` | Dominant cost: repeated optimizer states and weights at many epochs, often copied again into isolated runtime trees. |
| Identified generated validation/derived images | `383.21 GB` | Fixed-panel PNGs, face-quality copies, backfill staging, sidecars, and debug/validation output. |
| Core datasets | `154.17 GB` | `bigcelebs` `136.72 GB`; `dataset_full` `17.45 GB`. Keep. |
| Eight currently active scientific roots | `34.74 GB` and growing | CL14_CA r7/r11 and CL21-CL26. Do not touch. |
| Duplicated code/runtime metadata remaining in the seven heaviest experiment trees after subtracting checkpoints/images | about `35.62 GB` | Many isolated copies of the repo, `.git`, docs, small reference data, manifests, and miscellaneous outputs. Removable after provenance is retained and jobs are finished. |
| Conda environment | `10.96 GB` | Keep while this environment is the supported runtime. |
| Shared pretrained/metric caches | `4.93 GB` | `checkpoints/` plus `metric_cache/`; keep to avoid breaking model startup and scoring. |
| Non-image portion of completed subject-v2 staging | about `12.21 GB` | Tables, manifests, and other staging intermediates; verified publication means most can be removed after retaining the compact final audit records. |

In principle, after all current jobs finish, nearly everything except the following can be removed or archived: one canonical code checkout, datasets, Conda, shared model/metric caches, one chosen checkpoint per run, immutable Comet/config/manifests, compact logs, and selected report/reproduction evidence. Do not delete whole roots blindly: the large trees mix disposable checkpoints/images with the small provenance files needed to interpret them.

## Delete first: high-confidence redundant data

| Path | Size | What it is | Why it is reasonable to delete |
|---|---:|---|---|
| `analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/staging/` | `111.70 GB` | Per-checkpoint regenerated images/tables for 25 subject-v2 backfills | The handoff records all 25 replacements as transactionally verified on their immutable Comet experiments. Keep the tiny package/status/manifests if desired. |
| `analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/staging_failed_manual_seed_af9c6f65/` | `0.80 GB` | Failed staging made with the wrong bbox seed | Explicitly rejected evidence; zero Comet writes were made from it. |
| `face_quality_staging/2026-07-27/` | `7.54 GB` | Four downloaded 1,248-image face-quality staging sets | All four immutable Comet runs have the seven verified curves and API CSV; this is a completed transfer/work cache. |
| `runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/` | `24.88 GB` | A copied CL14 seed run inside the first failed CL14_CA package | Redundant with the canonical CL14 tree under `runtime_sources_cl1_cl3_v1/`. Both trees contain the same 2,553 relative filenames and sizes; final weights and full-checkpoint SHA-256 hashes also match exactly (`0de10e...` and `7ee388...`). |
| `runtime_sources_cl14_ca_v2` through `v7`, and `v9` through `v22` | `11.62 GB` | Superseded CL14_CA startup/smoke source versions | These contain failed, stopped, or superseded pre-r7/pre-r11 attempts. Active scientific runs use v8 and v23. Preserve small failure logs separately only if wanted. |
| `runtime_sources_cl14_ca_v8/CL14_CA_skipval_smoke_r5/` and `runtime_sources_cl14_ca_v23/CL14_CA_optimized_speed_smoke_r12/` | `1.12 GB` | Finished smoke outputs beside active scientific roots | Non-scientific smoke data. Wait until the sibling active jobs finish before modifying their parent trees. |
| Failed CL21/22/25/26 roots in `runtime_sources_cl21_cl26_v1/`: `CL21_*_r1`, `CL22_*_r1`, `CL25_*_r1`, `CL26_*_r1`, `CL26_*_r2` | `0.78 GB` | Superseded startup failures | No selected scientific result; active replacements are CL21 r2, CL22 r2, CL25 r2, and CL26 r3. Prefer waiting for the active sibling jobs to finish. |
| The six `*_r1` roots in `runtime_sources_cl15_cl20_v1/` | `0.11 GB` | CL15-CL20 packaging failures | Non-scientific startup failures; completed results are in the six `*_r2` roots. |
| `runtime_sources/` and `runtime_sources_e19_e24_v2/` | `0.85 GB` | Failed E19-E24 r1 source packages | Superseded by successful r2 trees in `runtime_sources_e19_e24_v3/`. |
| `e19_e24_source_bundle_20260806*.tar.gz` (three top-level archives) | `0.16 GB` | Three successive source bundles | Redundant once the successful v3 source tree and local source revision are retained. Low priority because they are small. |

The rows total `159.56 GB`. Excluding the two smoke directories inside currently active parent trees still yields about `158.44 GB`. This table is a conservative no-checkpoint-policy wave and overlaps substantially with the checkpoint/image totals above.

## Larger source-tree cleanup after applying retention

| Path | Current size | What dominates it | Suggested retention rule before deletion |
|---|---:|---|---|
| `runtime_sources_cl1_cl3_v1/` | `353.10 GB` | Completed CL0-CL14 roots, usually 12 full + 12 weights checkpoints and local validation panels per run | Keep config, immutable Comet record, logs, and selected best/final weights. Keep a full optimizer checkpoint only for runs that may be resumed. CL14 is important, but its final seed is also copied into active CL14_CA roots. |
| `runtime_sources_cl15_cl20_v1/` | `153.06 GB` | Six completed CL15-CL20 runs | Preserve CL19 best/final weights and provenance because current follow-ups derive from it. Intermediate full optimizer checkpoints and Comet-uploaded panels are the main pruning candidates. |
| `runtime_sources_e19_e24_v3/` | `152.56 GB` | Six completed E19-E24 runs | Retain selected best/final weights plus configs/records; prune optimizer states and uploaded panels if no resume/revalidation is planned. |
| `runtime_worktrees/` | `315.48 GB` | Completed E7-E18 and BC_E13-family worktrees; `saved/` is almost all of the size | Archive selected weights/checkpoints first, then remove obsolete worktrees. Largest single tree is `rsrch_test_BC_E13_dataset_20260809` at `75.74 GB`. |
| `rsrch/diffusion_template/saved/` | `346.59 GB` | Older non-`test` experiment checkpoints, including several 28-54 GB runs | The checkout is not the current default, but much of this may be unique historical evidence. Inventory best/final checkpoints before deleting the old checkout or its `saved/` tree. |
| `rsrch_test/diffusion_template/saved/` and `saved_continuations/` | `184.99 GB` + `7.90 GB` | Historical Cosmic/Large Dataset checkpoints and validation outputs | Do not bulk-delete. Apply the same best/final-weights rule run by run after checking whether any diagnostic still consumes a checkpoint. |
| `analysis_sidecars/` | `4.66 GB` | Completed CL9 intervention and Eddie replay outputs | Likely removable after confirming all desired raw outputs exist locally; the PDFs/reports and key results are already local. |

A representative completed 24k run stores twelve full checkpoints of about `1.319 GB` each, twelve weights-only checkpoints of about `0.440 GB` each, and `3.7-3.8 GB` of local validation/face-quality images. The cross-tree calculation above is preferable to multiplying this representative estimate because historical runs vary in length and are duplicated across several roots.

## Keep / currently in use

- `runtime_sources_cl14_ca_v8/CL14_CA_r7/` (`18.02 GB` inside the tree): MLS `lm-mpi-job-244ef7b2-3943-4998-a82e-ae1be2208169` was running.
- `runtime_sources_cl14_ca_v23/CL14_CA_optimized_r11/` (`14.03 GB`): MLS `lm-mpi-job-26dc8f54-1b96-4129-9151-a4fb066a7ff7` was running.
- The six selected CL21-CL26 roots in `runtime_sources_cl21_cl26_v1/`: all six corresponding MLS jobs were running.
- `datasets/` (`154.17 GB`): active scientific inputs (`bigcelebs` `136.72 GB`, `dataset_full` `17.45 GB`), not redundant output.
- `conda_env/` (`10.96 GB`), `metric_cache/` (`2.49 GB`), and `checkpoints/` (`2.44 GB`): shared runtime/model dependencies; deleting them risks breaking active jobs and future evaluation.
- Source, experiment JSONs, `comet_experiment.json`, configs, manifests, logs, and final/best weights are small relative to the generated checkpoint/panel data and should be retained.

## Caveats

- Sizes are allocated bytes from read-only `du -x -B1`; the shared mount's free space can change independently.
- Checkpoint and image totals use logical file sizes from `find`; sparse-file allocation differences, if any, may make actual freed blocks differ slightly.
- Only the duplicate CL14 final weights/full checkpoint pair was byte-hash compared; historical checkpoint trees were not exhaustively deduplicated by content.
- Comet publication makes local validation panels/staging redundant, but does not by itself replace checkpoint files.
- No deletion should occur in a tree read by a running job. Recheck MLS status and open file/process paths immediately before any cleanup.

### Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Listed sizes and `12.48 GB` free-space snapshot | High | Direct read-only `du`/`df` measurements. |
| One-last-full-checkpoint recovery is `1,175.28 GB` | High | Direct file inventory over 120 inactive run directories; eight live roots explicitly pruned. |
| Identified generated image recovery is `383.21 GB` | High for listed locations | Direct extension-based file inventory; datasets, source/reference assets, reports, and live roots excluded. |
| Subject-v2 and face-quality staging are completed publication caches | High | Current handoff records transactional Comet verification for all affected runs. |
| CL14 copied seed tree is redundant | High | Both trees have identical 2,553-file name/size inventories; final weights and full-checkpoint SHA-256 values match. |
| Failed/smoke runtime roots are not scientific results | High | Run names, handoff failure ledger, and live replacement-job status agree. |
| Completed experiment roots can be heavily pruned | Medium | Their metrics/panels are published, but checkpoint retention is a research-policy choice. |

### Not established

- That every historical checkpoint has an off-Serv copy; do not remove unique weights without selecting an archive/retention policy.
- That every file in similarly named runtime trees is byte-identical; only the stated CL14 final pair was hashed.
- That every image anywhere under `nasilaev/` is disposable; the `383.21 GB` total deliberately targets known generated-output locations and excludes dataset/reference/report assets.
- That the shared mount's free-space value will remain unchanged after this audit.
