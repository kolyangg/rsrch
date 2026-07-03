# BA Debug Runbook — v1 (03 Jul 2026)

Companion to `ba_debug_plan_v1.md`. Everything below is implemented and smoke-tested
(config parse, dataset load, override merges, module imports). The GPU matrix itself
must run where the model weights are cached (vast instance or any machine with
RealVisXL/SDXL-base/PhotoMaker-V2 in the HF cache) — the local WSL box has no weights
and only 23 GB free disk.

## What was implemented today

| File | Purpose |
|------|---------|
| `src/configs/inference/ba_2k_debug.yaml` | infer.py config mirroring the cosm_new1_vast training run (RealVis trunk, rank 32, noise_and_ref/lora, gs 5, auto bboxes). Flat/self-contained (infer.py does not resolve hydra defaults). |
| `serv_new_runs/run_ba_debug_matrix.sh` | One-command T0–T7 A/B matrix on a saved checkpoint (each test = separate output dir + log). |
| `scripts/inspect_ba_checkpoint.py` | CPU probe of BranchLoRA delta norms per site/branch/projection. Verified on the 2k ckpt. |
| `scripts/crop_refs_to_face.py` | Pre-crops val references with the exact training crop (square face + 20% margin). Already ran locally → `../dataset_full/val_dataset/references{,_two}_cropped` + `ref_bboxes{,_two}_cropped.json`. |
| `scripts/idsim_report.py` | Post-hoc InsightFace id-sim + face-detection-rate report over output dirs. |
| `debug_planning_03Jul/prompts_debug4.txt` | 4 prompts (Reading paper / Drumming / Kickboxing / Skiing) matching the saved step-0/step-2000 panels. |
| **Fix F1** (`branched_runtime.py` + flag plumbing in `lora2.py`, `br_pipeline_helpers.py`, `infer.py`) | `ba_uncond_face_fix`: under CFG, the uncond half of the face prompt keeps plain negative-prompt embeds instead of being masked by the cond prompt's ID-token positions. Default **off** (legacy) so T0 reproduces the bug; T7 turns it on. |
| **Fix F9** (`lora2_helpers.py`) | `ensure_branched_after_eval` now re-attaches the *same trained processor instances* instead of rebuilding fresh clones. Without this, any run with `pretrained_model_for_validation_name_or_path=null` silently **reset BA weights and detached the optimizer after every validation**. Required for the R-runs below. |
| `serv_new_runs/start_ba_cosm_new1_vast_Rx.sh` | Parameterized 2k rerun variants R1/R2/R3/R1R3 (all include F1 + validate-on-training-base). |

## 0) One-time prep on the GPU machine

```bash
cd /workspace/rsrch/diffusion_template          # adjust to your layout
git pull                                        # or rsync the repo
conda activate <training env>                   # env with torch/diffusers/insightface
export PM_PATH=/path/to/photomaker-v2.bin       # same as training .env
# regenerate cropped refs if ../dataset_full/val_dataset/references_two_cropped is absent:
python scripts/crop_refs_to_face.py \
  --images-dir ../dataset_full/val_dataset/references_two \
  --bbox-json  ../dataset_full/val_dataset/ref_bboxes.json \
  --out-dir    ../dataset_full/val_dataset/references_two_cropped \
  --out-json   ../dataset_full/val_dataset/ref_bboxes_two_cropped.json
```

The checkpoint path defaults to `saved/03Jul_start_ba_cosm_new1_vast/weights-epoch1.pth`
(on vast: wherever `saved/cosm_new1_vast/weights-epoch1.pth` lives — set `CKPT=`).

## 1) Run the matrix (2k checkpoint)

Full matrix (12 runs × 8 images each; T0 also does the PhotoMaker bbox pass):

```bash
bash serv_new_runs/run_ba_debug_matrix.sh
```

Priority subset if GPU time is tight (in this order):

```bash
bash serv_new_runs/run_ba_debug_matrix.sh T0 T0b T1_gs1 T7_uncondfix T4_noca T3_refcrop T2_sdxl
```

Notes:
- **Run T0 first** — it populates the shared auto-bbox store
  (`outputs/ba_debug/bbox_gen_auto_realvis.json`) so all RealVis tests use *identical*
  gen masks (and skip their own PhotoMaker pass). T2 uses its own store by design.
- Each test writes images to `outputs/ba_debug/<ID>/` and a log to `outputs/ba_debug/<ID>.log`.
- `CKPT=saved/.../weights-epochN.pth bash serv_new_runs/run_ba_debug_matrix.sh ...` to point at another checkpoint.

## 2) Score + read out

```bash
python scripts/idsim_report.py --refs-dir ../dataset_full/val_dataset/references_two outputs/ba_debug/T*
python scripts/inspect_ba_checkpoint.py saved/03Jul_start_ba_cosm_new1_vast/weights-epoch1.pth
```

Interpretation:

| Observation | Confirmed cause | Action |
|---|---|---|
| T1_gs1 (and/or T7_uncondfix) much cleaner than T0 | CFG amplification through trained branches / garbage uncond face prompt | Keep `ba_uncond_face_fix=true` everywhere (R-runs already do); optionally lower guidance during BA steps |
| T3_refcrop much cleaner than T0 | Ref-domain gap (tight blurry crops vs full sharp photos) | Crop refs at val (make it a pipeline option), and/or diversify training ref context |
| T4_noca much cleaner than T0 | CA branch drift (matches probe: largest deltas in attn2 noise_to_v) | `train_branched_ca_lora=false` (R1) or much lower CA lr |
| T2_sdxl much cleaner than T0 | SDXL→RealVis trunk mismatch | Validate on training base while iterating (R-runs do); align bases for the final model |
| T0b ≈ step-0 panel, T0 degraded | (sanity) trained deltas are the trigger — expected | — |
| T5_pmonly degraded | something outside BA leaks | investigate before anything else |
| T6_top50 cleaner | artifacts come from late-depth SA sites | consider `ba_patch_top_k<1` / `ba_train_top_k<1` |

Baseline references for comparison: `saved/03Jul_start_ba_cosm_new1_vast/val_images/manual_val_two/step_{0,2000}_batch_*/`
(same prompts/seeds as `prompts_debug4.txt`, jensen + keanu).

## 3) 2k rerun variants (config changes that might already help)

```bash
bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R1R3   # recommended first: CA frozen + clip/wd/lr↓
bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R1     # CA branches frozen only
bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R2     # blended masked loss (λ_face=0.15)
bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R3     # clip 1.0 + wd 1e-2 + lr 3e-5
```

All variants add `+model.ba_uncond_face_fix=true` and `pretrained_model_for_validation_name_or_path=null`
(validation on the training base — **only safe with today's F9 fix**; without it, every
validation silently reset the trained processors and detached the optimizer).
Compare each run's `step_2000` panel + id_sim against the original `cosm_new1_vast`.

## 4) When the ~20k checkpoint arrives

1. Rerun the matrix against it: `CKPT=saved/<run>/weights-epochN.pth bash serv_new_runs/run_ba_debug_matrix.sh T0 T0b T1_gs1 T7_uncondfix T4_noca T3_refcrop T2_sdxl` — effects will be gross and unambiguous.
2. Drift trajectory over all epoch checkpoints:
   `python scripts/inspect_ba_checkpoint.py saved/<run>/weights-epoch*.pth` — watch which
   group's norms grow fastest and correlate with the epoch where val visually breaks.
3. If gs=1/T7 is clean at 20k but gs=5 is broken → the CFG pathway is the fix target
   (F1 + possibly guidance handling); if broken even at gs=1 → prioritize ref-domain
   (T3) and mechanism hardening (additive attention masks — plan v1 Stage 2).

## Caveats

- infer.py rebuilds processors on the configured base and loads only LoRA A/B, while
  in-training validation *copies* the training-time `base_weight` buffers into the val
  model (`update_proc_weights_val=true`). So T0 approximates but does not bit-match the
  training-val pipeline. If T0 fails to reproduce the artifact at all, that buffer copy
  is the next thing to replicate.
- `ba_2k_debug.yaml` must stay flat: infer.py loads it with `OmegaConf.load`, hydra
  `defaults:` lists are ignored there.
- Results should be written up as `debug_planning_03Jul/ba_debug_results_v1.md`
  (one row per test: image panel path, mean id_sim, no-face count, verdict).
