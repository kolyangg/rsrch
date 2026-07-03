# BA Debug Runbook — v2 (03 Jul 2026, local execution)

Supersedes v1 for the *local* workflow (v1 remains valid for vast). Changes vs v1:
weights download script, local disk/VRAM constraints, epoch-2 checkpoint as the default
test subject, `EXTRA=` pass-through in the matrix driver, `enable_vae_tiling` flag.

## 0) One-time local setup (WSL, conda env `photomaker`, RTX 4090 Laptop 16GB)

```bash
conda activate photomaker
cd ~/rsrch/diffusion_template

# Weights (exactly the fp32 component files the SDXL model class loads):
bash scripts/download_ba_debug_weights.sh              # RealVis + PhotoMaker-V2 (~15GB)
# bash scripts/download_ba_debug_weights.sh --with-sdxl  # + SDXL-base (~13GB) for T2 only
```

Status 03 Jul: RealVis + PhotoMaker-V2 are in the local HF cache. SDXL-base was downloaded
and then **removed to free disk** (disk hit 100%; now ~16GB free). Re-fetch takes ~90s when
T2 is due. InsightFace `buffalo_l` auto-downloads on first run (~0.3GB); YOLO face model is
in the repo (`bbox_utils/yolov8n-face.pt`). No PM_PATH needed locally — `resolve_photomaker_path`
finds the bin via the HF-cache glob.

Checkpoints available: `saved/03Jul_start_ba_cosm_new1_vast/weights-epoch{1,2}.pth`
(2k and 4k steps). **Use epoch2 for A/Bs** (stronger degradation signal, deltas ~2× epoch1).

## 1) Smoke test (1 sample, ~5–10 min incl. model load)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python infer.py --config-name=inference/ba_2k_debug \
  saved_checkpoint=saved/03Jul_start_ba_cosm_new1_vast/weights-epoch2.pth \
  dataset.limit=1 output_dir=outputs/ba_debug/smoke \
  bbox_mask_gen_path=outputs/ba_debug/bbox_smoke.json enable_vae_tiling=true
```

## 2) Priority matrix on the 4k checkpoint

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=saved/03Jul_start_ba_cosm_new1_vast/weights-epoch2.pth \
EXTRA="enable_vae_tiling=true" \
bash serv_new_runs/run_ba_debug_matrix.sh T0 T0b T1_gs1 T7_uncondfix T4_noca T3_refcrop
```

Then the second tier (same env vars): `T1_gs2 T1_gs3 T4b_nosa T5_pmonly T6_top50`,
and finally T2 after re-fetching SDXL-base:

```bash
bash scripts/download_ba_debug_weights.sh --with-sdxl
CKPT=... EXTRA="enable_vae_tiling=true" bash serv_new_runs/run_ba_debug_matrix.sh T2_sdxl
rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0   # optional reclaim
```

Notes:
- **T0 must run first** (populates the shared bbox store so all RealVis tests get identical
  gen masks and skip their PhotoMaker pass).
- 8 images per test (jensen+keanu × 4 prompts, seed 0). Expect roughly 10–20 min per test
  on the laptop 4090; T0 longer (extra PhotoMaker pass for bboxes).
- Each test: images in `outputs/ba_debug/<ID>/`, log in `outputs/ba_debug/<ID>.log`.

## 3) Scoring and drift probes

```bash
python scripts/idsim_report.py --refs-dir ../dataset_full/val_dataset/references_two outputs/ba_debug/T*
python scripts/inspect_ba_checkpoint.py saved/03Jul_start_ba_cosm_new1_vast/weights-epoch1.pth \
                                        saved/03Jul_start_ba_cosm_new1_vast/weights-epoch2.pth
```

Drift so far (mean Frobenius of LoRA deltas, epoch1 → epoch2 = 2k → 4k steps):
attn2 noise_to_v 0.84→1.22 (largest), attn1 noise q 0.40→0.73, attn1 ref v 0.61→0.77 —
roughly linear doubling after warmup ended; same top sites (down_blocks.2 attn2 noise_to_v).

Interpretation table and baselines: see runbook v1 §2 (unchanged). Compare against
`saved/03Jul_start_ba_cosm_new1_vast/val_images/manual_val_two/step_{0,2000,4000}_batch_*`.

## 4) Rerun variants (unchanged from v1)

```bash
bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R1R3    # on vast
```

## Local caveats

- Disk: keep ≥3–4GB free; big reclaim candidates are `saved/` (15G), `comet_data/` (3.5G),
  `outputs/` (3.1G), `hm_debug/` (1.7G) — user's data, clean manually.
- VRAM: ~14GB free of 16GB; runs use bf16 + `enable_vae_tiling=true` +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. If a UNet-forward OOM still occurs,
  fall back to running that test on vast (runbook v1).
- RAM: fp32 model load peaks ~14GB CPU-side; close heavy apps if the load gets killed.
