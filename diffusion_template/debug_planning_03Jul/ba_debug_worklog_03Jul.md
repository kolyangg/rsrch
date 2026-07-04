# BA Debug Worklog — started 03 Jul 2026

Running log of actions, intermediate results, and decisions. Plans: `ba_debug_plan_v1.md`;
commands: `ba_debug_runbook_v1.md` (vast) / `ba_debug_runbook_v2.md` (local execution).

## Session 1 — 03 Jul (afternoon)

### Analysis (details in plan v1)
- Traced the full BA stack (attn_processor_cleanest / branched_runtime / lora2* /
  br_pipeline_helpers / trainer / cosmic dataset / manual_val).
- Ranked causes for the face-paint artifacts: (1) CFG through trained branches with a
  garbage uncond face prompt (val-only pathway), (2) ref-domain gap (train: tight blurry
  ~256px face crops; val: sharp full photos) + zero-K/V attention-sink dilution,
  (3) global CA drift (face-masked loss trains whole-image text pathway),
  (4) SDXL(train)→RealVis(val) trunk mismatch, (5) no grad clip / wd at lr 1e-4.
- Git archaeology: last known-good artifacts (early Feb) predate the ID-only face prompt
  (5273edd, Feb 18) and noise_and_ref separate LoRA weights (Mar 17).

### Code delivered (all smoke-tested; nothing committed yet)
- `src/configs/inference/ba_2k_debug.yaml` — infer.py config mirroring cosm_new1_vast.
- `serv_new_runs/run_ba_debug_matrix.sh` — T0–T7 A/B driver (per-test dirs/logs, shared
  auto-bbox store, `EXTRA=` pass-through for e.g. `enable_vae_tiling=true`).
- `scripts/inspect_ba_checkpoint.py`, `scripts/crop_refs_to_face.py`, `scripts/idsim_report.py`.
- **Fix F1** `ba_uncond_face_fix` (branched_runtime + lora2 + br_pipeline_helpers + infer.py):
  uncond half of face prompt keeps plain negative embeds under CFG. Default off (legacy).
- **Fix F9** (lora2_helpers): `ensure_branched_after_eval` re-attaches the *same trained*
  processor instances; previously, validating on the training base rebuilt fresh clones →
  silent training reset + detached optimizer. Prerequisite for the Rx reruns.
- `serv_new_runs/start_ba_cosm_new1_vast_Rx.sh` — R1/R2/R3/R1R3 rerun variants
  (all include F1 + validate-on-training-base).
- `infer.py`: `enable_vae_tiling` config flag (16GB-GPU relief).
- Cropped ref sets generated: `../dataset_full/val_dataset/references{,_two}_cropped`
  + `ref_bboxes{,_two}_cropped.json` (training-style square face crops +20% margin;
  e.g. jensen 600×337 → 177×177).

### Evidence so far
1. **step-0 val clean, step-2000 shows onset** (waxy red mottling, washed hairline —
   `saved/03Jul_start_ba_cosm_new1_vast/val_images/.../Drumming_m_jensen.png`), id_sim 0.204.
   → trained deltas are the trigger; failure progressive.
2. **Weight probe epoch1 (2k steps):** no blowup (delta abs-max ≈ 0.01); largest group =
   attn2 `noise_to_v` (global text pathway), esp. down_blocks.2. 4/840 deltas exactly zero.
3. **Weight probe epoch2 (4k steps): norms ≈ doubled vs 2k** (attn2 noise_to_v mean fro
   0.84→1.22, attn1 noise q 0.40→0.73; same top sites). Linear growth, no saturation —
   warmup ended at 2k, so drift accelerates; extrapolates to the gross 20k artifacts.
4. **step-4000 panel:** Drumming/jensen face more coherent than step-2000 in this sample but
   identity drifting + pasted/oversharpened look — per-sample degradation is not monotonic,
   another reason to score with id_sim over the full panel rather than single images.

### Local environment
- conda env `photomaker` has the full stack (torch/diffusers 0.29.1/peft/insightface/hydra).
- GPU: RTX 4090 Laptop 16GB (≈14GB free); RAM 30GB.
- Disk was the constraint: downloads (RealVis 13G + SDXL 13G + PM-V2 1.7G, ~5 min on this
  connection) filled the disk to 100%. **Removed SDXL-base cache** (re-downloadable in ~90s)
  → 16GB free. T2 (SDXL swap) will be run last after a re-fetch, or on vast.
  Reclaim candidates if more space is needed (user's call): `saved/` 15G, `comet_data/` 3.5G,
  `outputs/` 3.1G, `hm_debug/` 1.7G.
- Downloader: `scripts/download_ba_debug_weights.sh` (fetches exactly the fp32 component
  files that `src/model/sdxl/original.py` loads; `--with-sdxl` for the T2 base).

### In progress
- Smoke run (1 sample, epoch-2 ckpt, VAE tiling) — then the priority matrix on epoch2:
  `T0 T0b T1_gs1 T7_uncondfix T4_noca T3_refcrop`, scoring via idsim_report.
- Matrix runs use `CKPT=saved/03Jul_start_ba_cosm_new1_vast/weights-epoch2.pth` (stronger
  4k-step signal); epoch1 kept for drift probes.

### Smoke run (epoch-2 ckpt, 1 sample "Reading paper"/jensen, gs=5) — PASS + new finding
- Plumbing works end-to-end locally: model load → PM pass → auto-bbox → BA pass with
  NO_ID→PHOTOMAKER→BOTH switching; ~90 s/sample on the 4090 laptop; VRAM peak 15.8/16.4GB
  (with `enable_vae_tiling=true`).
- Output `outputs/ba_debug/smoke/Reading pa_jensen.png` **reproduces the failure locally**
  and reveals a NEW cause:
- **C6 — stale gen-bbox / composition divergence.** The auto-bbox from the PM pass
  ([267,67,499,398], upper-left) no longer matches the BA pass composition (subject ended
  bottom-center): the face branch painted ref content into the stale bbox → floating
  ghost-face blob at upper-left, while the real face grew outside the mask (weaker
  identity, pasted look). PM and BA passes share seed and are identical until step 15,
  but the trained branch deltas shift the trajectory after step 15 enough to move the
  composition — so C6 *scales with training drift* and can place "paint" either ON the
  face (when composition holds) or NEXT TO it (when it diverges). The user's screenshot
  (blotches on faces) and this ghost are two ends of the same failure.
- Implication for fixes: dynamic/tracked masks during the BA pass (or bbox re-detection at
  branched_attn_start_step from the x0 preview) instead of a frozen PM-pass bbox.

### Results (matrix complete — full writeup in `ba_debug_results_v1.md`)
- All 6 tests ran OK (~10 min/test incl. model reload; VRAM held with vae tiling).
- mean id_sim / no-face: T0 0.345 (1 no-face, score inflated by pasted-ref detection),
  **T0b 0.446**, T1_gs1 0.167, T7_uncondfix 0.286, **T4_noca 0.414**, T3_refcrop 0.259.
- Visual (contact sheets in `assets/`): T0 catastrophic (second ref-face + giant mouth pasted
  into mask); T0b clean; **T4_noca clean ≈ untrained**; **T7 one-flag fix removes most of the
  catastrophe**; T1_gs1 washed out but face still a ref-fragment collage (core drift is
  CFG-independent); T3 no paint but mangled features (not an inference-side rescue).
- Verdicts: (1) trained CA branches = primary destroyer; (2) uncond face prompt under CFG =
  main amplifier (F1 works); (3) core drift needs training-side fixes; (4) C6 stale-bbox
  secondary; (5) T2 skipped locally (SDXL evicted for disk; question now secondary).
- Recommended: rerun `start_ba_cosm_new1_vast_Rx.sh R1R3` (+ optional
  `branched_attn_weight_mode=ref_only` variant); keep `ba_uncond_face_fix=true` everywhere.

## Session 2 — 03 Jul (evening): "make BA train as intended" (N1)

User direction: BA is the research contribution — it must stay trainable and beat stock
PhotoMaker; disabling it (T4-style) is diagnosis, not a solution. Approved plan:
`ba_training_fix_plan_v2.md`.

Reasoning chain for the N1 design:
- The failure channel is trainable capacity on the GEN side (noise_to_* clones let the
  face-masked loss warp whole-image generation). The intended mechanism only needs the
  REFERENCE side trained: how the ref image is encoded (SA ref branch), what the face
  branch reads from it (ref_to_k/v), and the ref half's text conditioning (CA ref half).
  → `branched_attn_weight_mode=ref_only` keeps BA fully trainable where it matters and
  makes whole-image drift structurally impossible (gen pathway = frozen base weights).
- Restore the last known-good face-branch conditioning (full prompt with boosted ID tokens,
  pre-Feb-18) behind a new `ba_face_prompt_mode` switch; keeps `id_only` for reproducibility.
  Composes with F1 (uncond half = plain negative under CFG).
- Broaden the training ref distribution (crop-margin jitter 0.2–0.6 + sharpness jitter) and
  move val refs into the training domain (cropped set) — the branches should generalize
  across ref framing instead of memorizing one crop style.
- Optimization hygiene: blended masked loss (λ=0.2), lr 5e-5, clip 1.0, wd 1e-2.
- Validation on the training base (null override; safe post-F9).
- Live drift canary: `ba_norm/{sa_ref,sa_noise,ca_ref,ca_noise}` (lora_B L2) to Comet every
  log_step — the old run's doubling-per-2k pattern would have been visible immediately.

Implementation log (this session):
- B1 `ba_face_prompt_mode` (id_only default | full_boosted) in `two_branch_predict` +
  plumbing (lora2 param, build_pipeline copy, infer.py string attr). Verified via source
  assertions + py_compile.
- B2 ref-crop jitter in `CosmicLargeTrain` (`ref_crop_margin_min/max`, `ref_downscale_jitter`).
  Unit-tested on the local cosmic_large sample with a pinned source image: legacy defaults
  bit-identical (single crop size), jitter varies sizes, bboxes remain valid. First test
  attempt had a wrong premise (multiple face_paths per identity make even legacy sizes vary)
  — fixed the test, not the code.
- B3 `ba_norm/{sa_ref,sa_noise,ca_ref,ca_noise}` canary in `PhotomakerLoraTrainer`
  (lora_B L2 per group, every `log_step=50`, same metrics route as grad_norm/*).
- `serv_new_runs/start_ba_ref_only_vast_N1.sh` created; run_name `ba_refonly_N1` →
  checkpoints under `saved/ba_refonly_N1/` (base_trainer: `save_dir / writer.run_name`),
  distinct from the old `saved/cosm_new1_vast/`.
- Runbook v3 written (launch, success criteria, N1-checkpoint probing overrides, fallbacks).
- Micro-train smoke attempts: #1 failed in the id_sim METRIC (`KeyError: 'jensen'`) because
  the smoke omitted `metrics=all_metrics` and the default `all_metrics_oneid` uses one_id
  embeddings — NOT a code bug; generation itself completed through the new B1/F1 path.
  #2 failed on launch cwd (accelerate config path). #3 crashed on train batch 2 with a
  PyTorch allocator INTERNAL ASSERT — a known WSL2 + `expandable_segments:True` failure
  when training-sized allocations exceed the 16GB card; batch 1 (forward+backward+step)
  completed. Local-hardware quirk only; vast A100s unaffected.
- **Micro-train smoke #4: PASS (exit 0)** — epoch_len=1, no expandable segments:
  * pre-train val id_sim 0.560 → post-step val id_sim 0.606 (cropped refs; old run: 0.204)
  * canary logged: `ba_norm/sa_ref=0.011, ca_ref=0.008, sa_noise=0.0, ca_noise=0.0` —
    ref groups move, noise groups exactly zero, as ref_only requires
  * blended loss finite (0.0153); F9 pre/post-val re-patch exercised; checkpoints saved to
    `saved/ba_refonly_smoke/` (run-name dir); weights 165MB (vs 237MB noise_and_ref)
  * `inspect_ba_checkpoint.py` on the saved weights: 420 tensors, ONLY ref groups —
    ref_only verified end-to-end through save/probe.
- Committed and pushed to `origin/main_clean` ("big fixes 3 Jul"): all fixes (F1, F9, B1-B3),
  scripts, configs, N1/Rx launch scripts, planning docs, and the cropped val-ref sets
  (dataset_full/val_dataset/references{,_two}_cropped + jsons, <1MB) so N1 is self-contained
  after `git pull` on vast. Excluded transient local caches (pm96_bboxes_new_auto.json*).

### Post-commit: made N1 a true single-command run + matched the original's dataset
User asked to copy the env handling from `start_ba_cosm_new1_vast.sh` into
`start_ba_ref_only_vast_N1.sh` so it runs with one command (no exported vars), and to
double-check it uses the same dataset as the original.
- **Self-contained secrets.** The original hardcodes `COMET_API_KEY` inline but still needs
  `${PM_PATH}` exported; `serv_new_runs/.env` (which holds both) is **gitignored**, so it
  won't arrive via `git pull`. Baked both into N1 near the top as override-friendly defaults:
  `PM_PATH="${PM_PATH:-/mnt/.../photomaker-v2.bin}"`,
  `COMET_API_KEY="${COMET_API_KEY:-wSzl6h2PsRcopvISb2TJvtkzH}"` (values from `.env`), then
  `export`. Safe under `set -u`. Changed the env-prefix `COMET_API_KEY="${COMET_API_KEY:?...}"`
  → `"${COMET_API_KEY}"`. Now `bash serv_new_runs/start_ba_ref_only_vast_N1.sh` is the whole run.
- **Dataset now matches the original.** Removed the val override
  `datasets.val.manual_val_two.images_dir=...references_two_cropped` +
  `bbox_mask_ref=...ref_bboxes_two_cropped.json`. N1 now uses the default `manual_val_two`
  (`references_two` + `ref_bboxes.json`, limit 24) — identical to
  `start_ba_cosm_new1_vast.sh`. Verified by `grep` diff of dataset hydra args: training
  (`cosmic_large_vast`, num_refs=1) and val (`manual_val_two`, limit=24) match; the only
  training-side delta is the 3 ref-crop/downscale jitter knobs, which are runtime
  augmentation on the *same* `cosmic_large_vast` files.
- **Correction to a prior worklog claim.** The cropped val refs *are* tracked in git — at
  **repo-root** `dataset_full/val_dataset/references{,_two}_cropped` (the config's
  `../dataset_full/...` from `diffusion_template/` resolves there). My first `git ls-files`
  check ran from `diffusion_template/` with a repo-root-relative pathspec and wrongly reported
  0 tracked; corrected after checking from the repo root. So reverting to the default val set
  is a deliberate comparability choice (same val inputs as the failing run), not a
  missing-file workaround. Both `references_two` and `references_two_cropped` hold the same 2
  identities (jensen, keanu) — original vs face-tight framing.
- `bash -n` clean; no stale `cropped`/`crop_refs`/`:?export` strings remain in the script.
  Runbook v3 "Run it (vast)" + val-set notes updated to the one-command flow.

### 04 Jul: N1 first-validation crash — fixed (see ba_val_crash_fix_04Jul.md)
`ba_refonly_N1` crashed at the first val step 0 with
`RuntimeError: Invalid branched batch: total=8, generation=2, reference=6`
(`attn_processor_cleanest._branch_batch_sizes`). Root cause: `set_validation_unet_mode`
trusted the cached `_runtime_uses_branched_unet` flag, which desyncs from the actual UNet
processors when validating on the training base (shared UNet, reused pipe) + F9's
`ensure_branched_after_eval` re-attaching branched processors post-val. The stale flag skipped
the swap-to-original, so branched processors (stale batch-2 training mask) ran on the normal
batch-8 val input. `cosm_new1` avoided it by validating on a separate base (fresh pipe each
time). Fix: `set_validation_unet_mode` now decides from the actually-attached processor types
(self-healing) + `base_trainer` resets the flag to None at each validation start. Verified via
`scratchpad/test_val_unet_mode_fix.py` (4/4, incl. exact bug repro). The
`_branch_batch_sizes` assertion is kept (it caught the bug). Touches only
`src/pipelines/br_pipeline_helpers.py` and `src/trainer/base_trainer.py`.
