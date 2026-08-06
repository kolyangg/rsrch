# E19-E24 implementation and Serv launch record

**Date:** 6 August 2026  
**Status:** corrected `r2` jobs running on Serv; startup and immutable Comet
registration verified.

## Implemented experiment arms

| Arm | Config | Intended delta from E13 |
|---|---|---|
| E19 | `src/configs/E19_large_ds_joint_shadow_sa128_multiref_24k.yaml` | E18's deterministic 48k identity-balanced multi-reference package; reference 0 remains the sole spatial BA lane |
| E20 | `src/configs/E20_large_ds_joint_shadow_sa128_branchout_r32_24k.yaml` | rank-32 output LoRA local to the hard self-attention reference branch |
| E21 | `src/configs/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k.yaml` | E19 and E20 combined as the fourth 2x2 corner |
| E22 | `src/configs/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k.yaml` | intended-box predicted-x0 identity loss through the exact frozen Buffalo-L recognition graph, with bounded online gradient calibration |
| E23 | `src/configs/E23_large_ds_joint_shadow_sa128_earlydecay_24k.yaml` | cosine LR decay begins at 8k instead of 14k |
| E24 | `src/configs/E24_large_ds_joint_shadow_sa128_alternating_24k.yaml` | exact even/odd face-only versus full-latent MSE, with audited component and gradient telemetry |

All arms preserve the E13 24k/batch-two/fixed-full96 contract,
`pose_adapt_ratio=0`, `ca_mixing_for_face=false`, shadow pretrained-default
validation, and the unchanged canonical `IDSimBest` metric.

## E22 implementation and verification

`src/model/photomaker_branched/arcface_identity_aux.py` executes the exact
`w600k_r50.onnx` graph with differentiable PyTorch operators while registering
all recognizer weights as frozen buffers. The configured model SHA-256 is
`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
On Serv, deterministic ONNX Runtime versus PyTorch verification produced:

- embedding cosine: `1.0`;
- maximum absolute embedding error: `3.725290298461914e-06`;
- input-gradient norm: `0.11236049234867096`.

The auxiliary uses scheduler-aware predicted-x0 reconstruction, differentiable
ROIAlign on the intended target box, and a detached normalized centroid of the
current target plus a distinct same-ID reference. It begins after 4k, ramps to
6k, applies on eligible `t <= 300` samples every two steps, and calibrates its
weight toward a 7.5% identity/diffusion gradient ratio with a hard `0.05` cap.
The historical PhotoMaker-CLIP auxiliary remains the default backend.

## Source and launch controls

- Active launcher: `launchers/active/run_E19_E24_large_ds_24k_1gpu.sh`.
- Serv startup template:
  `serv_run_packages/_sources/start_E19_E24_large_ds_24k_1gpu.sh`.
- Each `r2` job runs from a separate tree under
  `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/`.
- The shared snapshot has 846 hash-verified source files and identifies local
  base revision
  `d903b2c9e92ce1a6f3db7a1f8fccf82c0d1ab21f+e19-e24-snapshot-v3-20260806`.
- The fixed validation dataset link and required reference/embedding assets
  are checked before launch. Machine `.env` remains outside the manifest and
  is copied with mode `0600`.
- All six Hydra/spec gates pass. Runtime ownership gates pass exactly at
  2,240 tensors / 219,217,920 parameters for E19/E22/E23/E24 and 2,380 tensors
  / 224,542,720 parameters for E20/E21.

## Active corrected runs

| Arm | Run | Serv job | Immutable Comet key |
|---|---|---|---|
| E19 | `E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2` | `lm-mpi-job-8ad95723-ea0a-4bfb-a920-6904d91eb993` | `3280232a45ef4ea2ae68c8deff3b81c1` |
| E20 | `E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2` | `lm-mpi-job-51cd67d6-c28c-4185-9595-b37a273e71c1` | `4084c35600ae4ad3904446e5f4d2de92` |
| E21 | `E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2` | `lm-mpi-job-e11a4015-6493-4313-8e82-4c6525e02fec` | `3ef78907f60a4f5cbd7727fc5be7143e` |
| E22 | `E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2` | `lm-mpi-job-69206471-725e-4a97-b33f-a088e8fb6576` | `5a91be0df76f4966be5c77eee26cfc29` |
| E23 | `E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2` | `lm-mpi-job-48c7efd6-517d-400d-9eac-d77cba398853` | `9b6942c0ee6740c7aa4d3fe74effee93` |
| E24 | `E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2` | `lm-mpi-job-6f9ec18e-2c47-4b1e-ad97-4a29f16a31b5` | `5b64f84f134441b791e7c3ffbd6fe4f7` |

At the final check, all six were `Running`, had registered the listed Comet
keys, passed their trainable-ownership audits, and had entered the standard
step-zero validation phase. Together with still-running E17 and E18, project
usage was exactly eight A100s, matching the user-authorized exception.

## Failed first attempts

The six `r1` jobs passed source/config/data-manifest preflight and registered
Comet, but failed before model construction because `../dataset_full` pointed
to the training-only dataset mirror, which lacks fixed-96 reference images.
No optimizer step ran. Their JSON records preserve their job IDs, Comet keys,
and failure status; `r2` uses new run names and keys rather than rewriting
those immutable records.

During diagnosis, MLS visibility lag made the original E23/E24 submissions
appear absent and two duplicates were submitted. The exact duplicate jobs
`lm-mpi-job-858d7147-87ea-471d-b503-0ecae6c0dbef` and
`lm-mpi-job-b0d235dd-f84c-4993-bea7-5adb853ab0a7` were immediately deleted.
The corrected suite was submitted only after live project usage returned to
two A100s.
