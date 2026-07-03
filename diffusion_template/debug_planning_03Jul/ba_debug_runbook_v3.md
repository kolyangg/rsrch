# BA Runbook — v3 (03 Jul 2026): the N1 "trained as intended" setup

Companion to `ba_training_fix_plan_v2.md` (approved plan + rationale) and results in
`ba_debug_results_v1.md`. v1/v2 runbooks stay valid for the diagnostic matrix.

## What N1 is

Branched attention stays fully trainable — but only where the mechanism needs capacity:

- `branched_attn_weight_mode=ref_only`: trains SA ref branch (`ref_to_q/k/v`, which also
  serve the face-branch K/V) + CA ref half. The gen/noise pathway runs frozen base weights,
  so the whole-image drift that destroyed cosm_new1 is structurally impossible.
- `+model.ba_face_prompt_mode=full_boosted` (new switch): face branch conditions on the full
  fused prompt with ID tokens ×2.5 (pre-Feb-18 known-good) instead of the ID-only variant
  whose 75/77 zero tokens acted as attention sinks.
- `+model.ba_uncond_face_fix=true` (F1): plain negative prompt for the uncond face half under CFG.
- Blended masked loss (λ=0.2), lr 5e-5, grad clip 1.0, wd 1e-2, warmup 2000.
- Validation on the training base (`pretrained_model_for_validation_name_or_path=null`) —
  safe after the F9 `ensure_branched_after_eval` fix.
- Train ref-crop jitter: margin ∼U(0.2, 0.6) + sharpness jitter (p=0.5). Val set = the
  default `manual_val_two` (references_two) — same as the failing run, so N1 vs cosm_new1 is
  a clean A/B on identical val inputs. (Domain-aligned cropped val refs — same 2 identities,
  face-tight, committed — were dropped for comparability; revisit if val-domain mismatch
  looks material.)
- Live drift canary in Comet: `ba_norm/{sa_ref, sa_noise, ca_ref, ca_noise}` every 50 steps
  (L2 of processor `lora_B`; starts at 0; noise groups stay 0 in ref_only mode by design).

## Run it (vast)

Single command — `COMET_API_KEY` and `PM_PATH` are baked into the script, and the val set
is the default `manual_val_two` (identical to `start_ba_cosm_new1_vast.sh`), so there is no
prep step and nothing to export:

```bash
cd /workspace/rsrch/diffusion_template && git pull        # or rsync
bash serv_new_runs/start_ba_ref_only_vast_N1.sh
```

Override the baked-in secrets by exporting `COMET_API_KEY` / `PM_PATH` first if desired.
Extra hydra overrides can be appended to the script invocation (they pass through `"$@"`).

> Update (03 Jul, post-plan): the earlier cropped-val-refs override was **removed** at the
> user's request ("use the same dataset as start_ba_cosm_new1_vast.sh"). N1 now validates on
> the exact same default `manual_val_two` = `references_two` set (jensen, keanu) as
> cosm_new1_vast — a clean A/B on identical val inputs. (The `references_two_cropped` set is
> the same two identities face-tight and *is* committed to git, so this was a deliberate
> comparability choice, not a missing-file workaround.) Training still uses the same
> `cosmic_large_vast` set; the ref-crop/sharpness jitter is runtime augmentation on that data.

## What to watch (success criteria)

1. **Comet `ba_norm/*`**: sa_ref and ca_ref should grow *sublinearly* and flatten;
   the failing run's signature was doubling every 2k steps. sa_noise/ca_noise must stay 0
   (if not, the trainables selection is wrong — stop and investigate).
2. **step-2000 / step-4000 panels** (same prompts/seeds as before): no paint, no ghost
   pastes; id_sim ≥ 0.45 (the untrained T0b level) and rising.
3. **BA must beat PM**: from the N1 checkpoint run a BA-off pass for the PM baseline and
   compare id_sim on the same panel (see probing below + `scripts/idsim_report.py`).
4. `python scripts/inspect_ba_checkpoint.py saved/<N1 run>/weights-epoch*.pth` — only
   `ref` groups present; norms should mirror the Comet curves.

## Probing N1 checkpoints with infer.py

The debug config defaults to the old run's shape — override the mode (and keep the new
conditioning flags) or the state dict won't line up with the processors:

```bash
python infer.py --config-name=inference/ba_2k_debug \
  saved_checkpoint=saved/<N1 run>/weights-epochN.pth \
  model.branched_attn_weight_mode=ref_only \
  +ba_face_prompt_mode=full_boosted ba_uncond_face_fix=true \
  model.pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  pipeline.pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  output_dir=outputs/ba_debug/N1_e<N>
# PM baseline for the "beats PhotoMaker" comparison:
#   + validation_args.use_branched_attention=false validation_args.use_bbox_mask_gen=false automatic_bboxes=false
```

(Local machine note: SDXL-base was evicted from the HF cache for disk space — re-fetch via
`bash scripts/download_ba_debug_weights.sh --with-sdxl` before local probing, or probe on vast.)

## Fallbacks if N1 underdelivers

- `ca_ref` norm runs away / panels degrade → append `train_branched_ca_lora=false`
  (SA-ref-only training) to the same script.
- Identity gain too weak (face branch under-powered) → raise `lambda_face` to 0.3, or
  lengthen training before touching lr; consider `ba_train_top_k<1` to focus early sites.
- Ghost pastes reappear → implement the C6 dynamic-bbox re-detection at
  `branched_attn_start_step` (plan v2 "not changing" list — revisit trigger).

## Code delta for N1 (all verified by unit checks + local micro-train smoke)

- `branched_runtime.two_branch_predict`: `ba_face_prompt_mode` switch (B1).
- `lora2.py` / `br_pipeline_helpers.py` / `infer.py`: flag plumbing (string attr, not bool-cast).
- `cosmic.py CosmicLargeTrain`: `ref_crop_margin_min/max`, `ref_downscale_jitter`
  (defaults = legacy fixed +20% crop; verified bit-identical when unset).
- `sdxl_trainers.PhotomakerLoraTrainer`: `_update_ba_weight_norms` → `ba_norm/*` metrics.
- `serv_new_runs/start_ba_ref_only_vast_N1.sh`: the launch script.
