# Current setup vs original run (`saved/03Jul_start_ba_cosm_new1_vast`) — 04 Jul 2026

"Original" = the run saved in `saved/03Jul_start_ba_cosm_new1_vast` (ground-truth from its
`config.yaml`): **noise_and_ref**, `masked_alternating`, lr 1e-4, validated on RealVisXL, no
uncond-face-fix, no ref-crop jitter, run_name `cosm_new1_vast`.

Three layers of "current" to keep distinct:
- **(C) current training script** `serv_new_runs/start_ba_ref_only_vast_N1.sh` — what we'd retrain
  with ("N2").
- **(B) already-saved weights** `saved/ba_refonly1` (epoch-10) that we just evaluated — trained
  earlier under ref_only but with `ba_face_prompt_mode=full_boosted` and no jitter.
- **(V) validation/inference path** used to render the panels (infer.py + `ba_n1_realvis.yaml`).

## 1. Training-config changes  (C vs original)

| Knob | Original (03Jul) | Current N1 script | Why |
|---|---|---|---|
| `branched_attn_weight_mode` | **noise_and_ref** (840 tensors) | **ref_only** (420) | remove the trainable gen/noise cross-attn — the diagnosed whole-image **drift channel**. BA stays fully trainable on the ref side. |
| `loss_kind` | `masked_alternating` | `blended_masked` | smooth face weighting instead of hard on/off alternation |
| `lambda_face` | 0.1 | 0.2 | a bit more face emphasis under the smooth loss |
| `lr_for_lora` | 1e-4 | **5e-5** | drift hygiene (slower, more stable) |
| `optimizer.weight_decay` | 0 | **1e-2** | drift hygiene |
| `trainer.max_grad_norm` | null (no clip) | **1.0** | drift hygiene (clip spikes) |
| `ba_uncond_face_fix` | absent → **false** | **true** | F1: uncond face branch uses a plain negative prompt under CFG (kills the double-face amplifier) |
| `ba_face_prompt_mode` | absent → code default `id_only` | **id_only** (explicit) | same effective behaviour, now an explicit switch (the alt `full_boosted` ×2.5 was shown to manufacture ghosts) |
| ref-crop jitter (`ref_crop_margin_min/max`, `ref_downscale_jitter`) | none | **0.2 / 0.6 / 0.5** | augment ref context+sharpness so the branches generalise beyond one crop style |
| `writer.run_name` | `cosm_new1_vast` | `ba_refonly_N2` | label |
| drift canary `ba_norm/*` (Comet) | none | added | live monitor of sa/ca × ref/noise LoRA growth |

**Unchanged** (deliberately identical for comparability): `rank=32`, `epoch_len=2000`,
`warmup_steps=2000`, `masked_loss_step=2`, `train_ba_only=true`, `train_ba_all_steps=true`,
`ba_train_top_k=ba_patch_top_k=1.0`, `non_ba_train=false`, `train_branched_ca_lora=true`,
`automatic_bboxes=true`, train set `cosmic_large_vast` `num_refs=1`, val `manual_val_two`
(`references_two`), `pretrained_model_for_validation=RealVisXL_V4.0`.

## 2. The weights we actually evaluated  (B vs C)
`saved/ba_refonly1/weights-epoch10.pth` is **ref_only** already (good), but it was trained with
`ba_face_prompt_mode=full_boosted` and **without** ref-crop jitter — i.e. it predates two items in
the current script. It still validates cleanly under id_only (the mode only affects how the face
prompt is injected at run time), which is why the epoch-10 panels look fine. A fresh **N2** train
with script (C) would also get id_only-consistent training + jitter.

## 3. Validation / inference path changes  (V)
These change how images are *rendered/compared*, not the weights:
- Validation base pinned to **RealVisXL_V4.0** (matches the original) instead of the N1 run's
  earlier `null → SDXL-base`, which caused the "animation / drift" look.
- `ba_face_prompt_mode=id_only` at inference (was my earlier `full_boosted` A/B → ghosts).
- **infer.py generator re-seed** before the branched pass, so the preview (bbox) and branched
  passes share initial latents (training path already did this).
- Validation-crash fix: `set_validation_unet_mode` now decides the processor swap from the
  *actual* attached processors (+ `base_trainer` resets the cache each eval).
- **NEW C6 gen-bbox re-tracking** (`gen_bbox_retrack`, default off): re-detect the gen face box on
  the branched trajectory so the mask follows the branched face (fixes the keanu long-hair smear).
  See [ba_gen_bbox_retrack_04Jul.md](ba_gen_bbox_retrack_04Jul.md).

## 4. Net
Structurally, the current setup swaps the **drift-prone noise_and_ref** approach for **ref_only +
drift hygiene + F1 uncond fix + id_only + ref jitter**, and fixes the validation base + two-pass
alignment + (new) gen-bbox tracking. The one remaining approach-level regression vs the original —
the long-hair motion smear — is what C6 targets; everything else is either safer by construction or
comparability-neutral.
