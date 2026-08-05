# E10 dynamic-mask checkpoint revalidation and in-place Comet replacement

**Date:** 5 August 2026  
**Source run:** `E10_large_ds_pmdefault_effective_r64_20k_full96_r1`  
**Source MLS job:** `lm-mpi-job-01a36932-2be9-413c-8cb3-cadcca9ae5ad` (completed)  
**Immutable Comet key:** `0375f172f75c482f840317ec5ae41c05`  
**Sidecar job name:** `E10V_large_ds_dynamicmask_reval_2k20k_full96_r1`

## Decision

Revalidate the ten saved E10 checkpoints at steps 2,000 through 20,000. For
each checkpoint and each of the fixed 96 validation items:

1. load that checkpoint with E10's exact resolved architecture and historical
   `legacy_full_copy` validation processor behavior;
2. generate a BA-disabled image using the same prompt, reference image, seed,
   scheduler, guidance scale, and inference-step count;
3. detect the face on that checkpoint-current BA-disabled image on CPU;
4. run hard BA with the newly detected target box;
5. calculate ID similarity, text similarity, and the normal face-quality
   outputs for the resulting 96 BA images.

Step 0 is not regenerated. It is already identical across E7-E10 and predates
the learned E10 layout drift, so its cached mask is not stale.

## Comet replacement protocol

The job does not write to Comet while images are being generated. Every step
is staged under
`saved/E10V_large_ds_dynamicmask_reval_2k20k_full96_r1/` and must contain:

- exactly 96 readable images with the canonical output-key set;
- exactly 96 fresh automatic bbox records;
- an exact-row per-image ID-sim CSV;
- the aggregate ID/text metrics;
- a 96-row face-quality CSV and complete face-quality JSON.

Only after all ten steps pass those checks does the publisher mutate Comet.
It backs up the original asset IDs and all nine affected metric histories,
then replaces only:

- the 96 image assets at each step from 2k through 20k;
- `manual_val/id_sim` and `manual_val/text_sim` at those steps;
- the seven `face_quality/` values at those steps;
- the per-step ID-sim dataframe and face-quality CSV assets.

Comet can delete metrics only as a full named series. Therefore the publisher
deletes each affected series and immediately reconstructs its complete
11-point history from the untouched original step-0 value plus the ten new
values. Training metrics, parameters, source code, step-0 images/tables, run
name, and original experiment identity are not changed. The publisher then
verifies one value per metric/step, 96 image assets per validation step, exact
filenames, and table SHA-256 metadata. A rerun uses the saved pre-replacement
manifest and is safe after a partial publication failure.

This intentionally follows the user's instruction to replace the misleading
fixed-mask E10 validation rather than retain it as the primary view. The local
source run directory still contains the original images and tables for audit.

## Serv files

- MLS YAML:
  `serv_run_packages/E10V_large_ds_dynamicmask_reval_2k20k_full96_r1/run_E10V_large_ds_dynamicmask_reval_2k20k_full96_r1_1gpu.yaml`
- Entrypoint:
  `serv_run_packages/E10V_large_ds_dynamicmask_reval_2k20k_full96_r1/start_E10V_large_ds_dynamicmask_reval_2k20k_full96_r1_1gpu.sh`
- Fail-closed publisher:
  `tools/comet/replace_checkpoint_validation.py`
- Source checkpoints on Serv:
  `runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch{1..10}.pth`

The YAML requests one A100. At pre-submission audit this project's live count
was one A100 (E12), so submission raises it to two and remains below the normal
six-GPU ceiling. `CUDA_LAUNCH_BLOCKING` is explicitly rejected.

## First submission and corrected resume

The first MLS submission, `lm-mpi-job-733198c2-2c48-4f37-a67c-4f9d0f663610`,
successfully generated all 96 step-2k images, 96 fresh boxes, and the per-image
ID table, then failed before publication because the isolated face-quality
subprocess reported that CUDA was unavailable. The publication gate therefore
left Comet completely unchanged.

The corrected launcher reuses that intact step-2k staging, calculates its
face-quality files on CPU, and uses CPU face-quality scoring after later GPU
generation steps. The main image generation and ID/text scoring remain on the
A100. The publisher also accepts and cross-checks both console scalar formats:
the immediate `Step N: metric = value` line and the validation-only summary
line. Conflicting duplicated values remain fatal.

## Interpretation after completion

The resulting E10 curves answer: “How well does each E10 checkpoint perform
when BA follows the face produced by its own learned non-BA/default-adapter
path?” They do not answer whether E10 preserved the original composition; it
did not. Any comparison must note that E10 now uses a dynamic-mask validation
protocol while E0/E1-E9 retain the canonical fixed-mask protocol.
