# Make Branched Attention Train As Intended (N1 setup) — plan v2

On approval, this plan is first copied to `debug_planning_03Jul/ba_training_fix_plan_v2.md` (user's logging requirement) and the worklog is updated.

## Context

Branched attention (BA) is the research contribution — it must stay trainable and *improve* over stock PhotoMaker, not be disabled. Today's A/B matrix on the 4k-step checkpoint localized the failure precisely:

- The **trained gen/noise-pathway weights** are the destroyer (T4: disabling branched CA restores near-untrained quality; probes show attn2 `noise_to_v` — the whole-image text pathway — has the largest, linearly-doubling deltas). The face-masked loss lets the optimizer warp global generation to cut face MSE.
- The **uncond face prompt under CFG** is the amplifier (T7: the F1 flag alone removes the double-face catastrophe).
- Core drift exists even without CFG (T1 gs=1: face = ref-fragment collage) → training-side changes required.
- The BA structure itself is sound: untrained clones (T0b) generate cleanly through the full branched path.

Design principle: **concentrate trainable capacity on the reference side** (how the ref image is encoded for the face branch), keep the generator pathway at base weights, restore the known-good face-branch conditioning, align ref domains, and watch drift live.

## Changes

### A. Training configuration (the structural fix)
New launch script `serv_new_runs/start_ba_ref_only_vast_N1.sh` (clone of `start_ba_cosm_new1_vast.sh`) with:

| Override | Why |
|---|---|
| `branched_attn_weight_mode=ref_only` | **Key change.** Only `ref_to_q/k/v` (SA ref branch + face-branch K/V + ref-half CA) train; the gen/noise pathway keeps base weights — kills the E1 drift channel *by construction* while BA stays fully trainable. `configure_branched_trainables` and `init_from_attention` already support this mode (verified). |
| `train_branched_ca_lora=true` (keep) | Ref-half CA is part of the mechanism; gen-half CA is base under ref_only. Fallback to false if ref-CA drift reappears in the canary. |
| `+model.ba_uncond_face_fix=true` | F1 (implemented+verified today): uncond face prompt = plain negative embeds under CFG. |
| `+model.ba_face_prompt_mode=full_boosted` | New switch (change B1): known-good conditioning, no zero-token sinks. |
| `loss_kind=blended_masked lambda_face=0.2` | Smooth face weighting instead of hard alternation (wired in train.py already). |
| `lr_for_lora=5e-5 trainer.max_grad_norm=1.0 optimizer.weight_decay=1e-2` | Drift hygiene (was 1e-4 / no clip / no wd). |
| `pretrained_model_for_validation_name_or_path=null` | Validate on the training base — one mismatch axis removed; safe after today's F9 fix. |
| `datasets.val.manual_val_two.images_dir=../dataset_full/val_dataset/references_two_cropped` + `bbox_mask_ref=.../ref_bboxes_two_cropped.json` | Val refs in the training ref domain (generate on vast once via `scripts/crop_refs_to_face.py`). |
| `+datasets.train.cosmic_large_vast.ref_crop_margin_min=0.2 ...margin_max=0.6 ...ref_downscale_jitter=0.5` | Change B2: ref-context/sharpness augmentation so the branches generalize beyond one crop style. |
| `writer.run_name="ba_refonly_N1"`, rest identical to cosm_new1_vast | Comparable 2k-step epochs, same panel/seeds. |

### B. Code changes (3 small, targeted)

1. **`ba_face_prompt_mode` switch** in `two_branch_predict` (src/model/photomaker_branched/branched_runtime.py:476-500): `full_boosted` = `emb*(1−m) + emb*m*id_scale` (the pre-Feb-18 known-good variant, currently in a comment at line 476); `id_only` = current behavior (default, preserves reproducibility of old configs). Composes with F1 (uncond half stays plain neg regardless of mode). Plumbing mirrors `ba_uncond_face_fix`: `lora2.py` __init__ param + attr, copy in `build_pipeline_from_pretrained` (br_pipeline_helpers.py), infer.py attr pass-through — **as a string, outside the bool-cast loop**.
2. **Ref-crop jitter** in `CosmicLargeTrain` (src/datasets/cosmic.py): new params `ref_crop_margin_min/max` (default 0.2/0.2 = legacy) and `ref_downscale_jitter` (default 0.0); `get_ref_image` samples margin ∼U(min,max) for `_get_bigger_crop_with_bbox(..., scale=...)` and, with prob given by the jitter param, downscales+re-upscales the crop (sharpness variation). Defaults keep all existing configs bit-identical.
3. **Drift canary**: in `PhotomakerLoraTrainer.process_batch` (src/trainer/sdxl_trainers.py), every `log_step` compute L2 norms of processor `lora_B` params grouped as `ba_norm/{sa_ref, sa_noise, ca_ref, ca_noise}` and push via `train_metrics.update` — same route as the existing `grad_norm/*` pattern (base_trainer.py:380-382), so Comet picks them up with zero writer-config changes. (`lora_B` starts at 0, so its norm is a clean monotone drift signal.)

Prerequisites already implemented and verified this session (no further work): F1 `ba_uncond_face_fix`, F9 trained-processor preservation in `ensure_branched_after_eval`, `enable_vae_tiling` in infer.py, cropped-ref generator + checkpoint probe + idsim report scripts.

### Explicitly NOT changing (deliberate)
- The branched SA mechanism, mask-gated merge, and doubled-batch design — that's the contribution; T0b/T4/T7 show the structure works.
- Multiplicative K/V zeroing → additive attention masks: Stage-3 hardening, only if needed after N1.
- C6 dynamic gen-bbox re-detection: expected to shrink with drift removal; revisit if N1 panels show offset ghosts.
- No CFG-dropout training yet.

## Files touched
- `src/model/photomaker_branched/branched_runtime.py` (B1)
- `src/model/photomaker_branched/lora2.py`, `src/pipelines/br_pipeline_helpers.py`, `infer.py` (B1 plumbing)
- `src/datasets/cosmic.py` (B2)
- `src/trainer/sdxl_trainers.py` (B3)
- `serv_new_runs/start_ba_ref_only_vast_N1.sh` (new)
- `debug_planning_03Jul/ba_training_fix_plan_v2.md` (plan copy) + worklog append + runbook v3 note (probing N1 ckpts with infer.py needs `model.branched_attn_weight_mode=ref_only`)

## Verification
1. Local: py_compile + config-resolution + dataset-jitter unit check (instantiate `CosmicLargeTrain` on the local sample with jitter on, confirm varying crop sizes and valid bboxes); `two_branch_predict` source assertions for the new switch.
2. Local micro-train smoke if VRAM allows (bs=1, `trainer.epoch_len=2`, `cosmic_large_local`, val limit 1) — otherwise config-compose check via hydra and skip to vast.
3. On vast: run N1 for 2 epochs (4k steps). Success criteria: (a) `ba_norm/*` curves grow sublinearly/flat vs the old run's per-2k doubling; (b) step-2000/4000 panels: no paint/ghosts, id_sim ≥ 0.45 (untrained T0b level) and rising; (c) BA id_sim ≥ PM-only baseline on the same panel (T5-style run from the N1 checkpoint); (d) `inspect_ba_checkpoint.py` on N1 weights shows only `ref_to_*` groups.
