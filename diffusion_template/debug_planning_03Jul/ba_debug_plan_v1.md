# Branched-Attention Debug Plan — v1 (03 Jul 2026)

Goal: find and fix why PhotoMaker branched-attention (BA) training run `cosm_new1[_vast]`
(`one_id_09Feb_testing`, `noise_and_ref` + `lora`, `train_ba_only=true`, `train_branched_ca_lora=true`,
`loss_kind=masked_alternating`, lr 1e-4) produces "face-paint" blotches / waxy mottled faces in
validation, getting worse with training steps.

---

## 0. What we know (evidence so far)

### Mechanism recap
- `two_branch_predict` (src/model/photomaker_branched/branched_runtime.py) runs one UNet pass on a
  doubled batch `[gen latents, ref latents noised to same t]` with doubled prompts `[main, face]`.
- attn1 (`BranchedAttnProcessor`, attn_processor_cleanest.py): bg branch (Q=noise·(1−mask), K/V=noise),
  face branch (Q=noise·mask, K/V=`ref_hidden·ref_mask` — masked-out ref tokens stay as zero K/V),
  ref branch (full self-attn). Merge: `bg·(1−mask) + face·mask`.
- attn2 (`BranchedCrossAttnProcessor`): gen half → main prompt; ref half → "face prompt" = PhotoMaker-fused
  prompt zeroed except ID tokens ×2.5, per-token std-matched (`face_embed_strategy=id`).
- `noise_and_ref` + `lora`: every patched attn1+attn2 site (140 sites, `ba_patch_top_k=1.0`) gets six
  `BranchLoRALinear` clones (noise/ref × q/k/v); only their `lora_A/B` train → 1680 tensors (matches info.log).

### Facts established today
| # | Fact | Implication |
|---|------|-------------|
| 1 | Epoch-1 (step-0) validation is clean; degradation onset already visible at **step 2000** (e.g. `val_images/manual_val_two/step_{0,2000}_batch_0/Drumming_m_jensen.png`: waxy red mottling, washed hairline). id_sim=0.204 at 2k. | Trained branch deltas are the trigger; failure is progressive. 2k ckpt is already usable for A/B. |
| 2 | Weight probe of `weights-epoch1.pth` (2k steps): all LoRA deltas small (abs-max ≈ 0.01, Fro ≈ 0.2–2.6 vs base-weight Fro ~30–80). No blowup. | Not simple divergence. Small systematic drift is enough to break the *val-only* pathways (CFG, RealVis trunk, full-photo refs). Note lr was still in warmup (warmup_steps=2000). |
| 3 | Biggest deltas by far: **attn2 `noise_to_v`** (down_blocks.2) and attn2 `ref_to_v/k`. attn1 deltas smaller. | Training is mostly rewriting the **global cross-attn value pathway** (whole image text conditioning), not face-specific weights. Freezing/CA-off is a prime experiment. |
| 4 | Training runs **without CFG**; validation uses guidance 5. The uncond half also passes through branched processors, with an uncond "face prompt" built by masking the *negative* prompt at the *positive* prompt's ID-token positions ×2.5 (branched_runtime.py:451-493) — garbage tokens, untrained pathway. | CFG extrapolates `uncond + 5·(cond−uncond)` in the face region through weights that only ever saw the cond path → prime suspect for saturated blotches that grow with training. |
| 5 | Training refs = tight face crops (+20% margin, random mirror) from ~256px sources (blurry when upscaled to 1024, face fills ~50–100% of frame). Validation refs = sharp full photos (400–2400px, face = 2–15% of frame). The dataset flags `upscale_to_1024 / const_ref / crop_ref / ref_similar / origtarget_genref` are **silently ignored** — `CosmicLargeTrain.__init__` `del`s them (cosmic.py:950-959). | Large ref-domain gap. Also: masked-out ref tokens act as zero-K/V attention sinks; sink fraction ~50% at train vs ~85–95% at val, so face-branch attention is diluted much harder at val. |
| 6 | Training base = SDXL-base; validation base = RealVisXL_V4.0 (`pretrained_model_for_validation_name_or_path`), with `update_proc_weights_val=true` copying processor state **incl. `base_weight` buffers** into the val model. | BA sites run SDXL-flavored projections inside a RealVis trunk; deltas trained on SDXL activations. Mismatch grows in effect as deltas grow. |
| 7 | `max_grad_norm: null` (no clipping), weight_decay 0, lr 1e-4 for all branch LoRAs incl. global CA pathway; face-only MSE every 2nd batch (`masked_alternating`, `masked_loss_step=2`). | Stability risk factors for longer runs (20k+), even though 2k shows no blowup. |
| 8 | Known-good Feb artifacts (hm_debug/00) predate both the ID-only face prompt (commit 5273edd, Feb 18) and `noise_and_ref` separate LoRA weights (Mar 17). | The regression window contains exactly these two mechanism changes. |
| 9 | Checkpoint format (`weights-epoch1.pth`): `{lora_weights, attn_processors{140 sites × 12 lora tensors}}` — loadable by `model.load_state_dict_` / infer.py. Local RTX 4090 Laptop 16GB + conda env `photomaker` can run inference. | All tests below can run locally or on vast. |

### Ranked causes
1. **CFG through trained branch layers with garbage uncond face prompt** (val-only pathway; ×5 amplification, mask-localized, grows with training).
2. **Ref-domain gap** (tight blurry crops vs sharp full photos + K/V zero-token dilution scaling with face fraction).
3. **Global CA drift** (`noise_to_v` largest deltas; face-masked loss every 2nd step pushes whole-image text pathway).
4. **Base-model mismatch** SDXL(train) → RealVis(val).
5. Stability (no clipping / wd / high lr) — matters at 20k+, not at 2k.

---

## 1. (Q1) Tests runnable NOW on the 2k checkpoint

Yes — meaningful. Degradation onset is already visible at 2k with fixed prompt+seed, and deltas vs step-0
are measurable via `id_sim` + visual face comparison. Effects will be subtle; use identical seeds and
compare side-by-side per prompt. Anything that clearly *changes* the face degradation at 2k is a
confirmed causal pathway (and will be dramatic at 20k).

### Prerequisites (small, to implement on approval)
- **P1. New inference config** `src/configs/inference/ba_2k_debug.yaml` mirroring the training run
  (existing `inference/*.yaml` are stale: old `_old2` pipeline, `id_embeds` strategy). Contents:
  model = `photomaker_branched_lora2`-style block with `rank: 32` (must match ckpt lora_A shape),
  `branched_attn_weight_mode: noise_and_ref`, `branched_attn_new_weight_kind: lora`, `use_attn_v2: false`,
  `face_embed_strategy: id` via pipeline block = `photomaker_branched_clean.PhotomakerBranchedPipeline.from_pretrained`,
  steps 10/10/15, 50 steps, guidance 5, `use_bbox_mask_ref/gen: true`, `automatic_bboxes: true`
  (`face_detector: yolo`, `face_model: bbox_utils/yolov8n-face.pt`), dataset = `manual_val` block
  (`../dataset_full/val_dataset/references`, `prompts_10.txt`, `ref_bboxes.json`, `classes_ref.json`,
  seeds [0], `limit: 12`), `batch_size: 1–2` (16GB local), `output_dir` per test.
- **P2. Checkpoint probe script** `scripts/inspect_ba_checkpoint.py` (the probe already run today):
  per-site/branch/projection `‖B@A‖`, group stats, top sites; CPU-only.
- **P3. Ref-crop prep script** `scripts/crop_refs_to_face.py`: crop each reference around
  `face_bbox_ref` with the training logic (square + 20% margin — reuse/extract
  `CosmicLargeTrain._get_bigger_crop_with_bbox`), write cropped refs to a new dir + remapped
  `ref_bboxes_cropped.json`. Lets us A/B the ref domain **without touching pipeline code**.

### Test matrix (each = one `infer.py` run; fixed seeds; compare faces + id_sim/text_sim)

Run template (local):
```bash
conda activate photomaker && cd ~/rsrch/diffusion_template
python infer.py --config-name=inference/ba_2k_debug \
  saved_checkpoint=saved/03Jul_start_ba_cosm_new1_vast/weights-epoch1.pth \
  output_dir=outputs/ba_debug/<TEST_ID>
```

| ID | Overrides (on top of template) | Tests | Read-out |
|----|-------------------------------|-------|----------|
| T0 | (none) | Baseline reproduction of val behavior at 2k | Reference images |
| T0b | `saved_checkpoint=null` | Untrained processors (identity clones) | Should match step-0 quality; anchor |
| T1 | `validation_args.guidance_scale=1` (also 2, 3) | Cause 1: CFG amplification | Blotch/waxiness collapses at low gs ⇒ confirmed |
| T2 | `model.pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0` `pipeline.pretrained_model_name_or_path=...` (same) | Cause 4: base mismatch (runs BA in its training trunk) | Cleaner on SDXL ⇒ confirmed |
| T3 | `dataset.images_dir=<cropped refs dir>` `dataset.bbox_mask_ref=<remapped json>` (from P3) | Cause 2: ref domain gap | Cleaner with train-like refs ⇒ confirmed |
| T4 | `disable_branched_ca=true` | Cause 3: CA branch drift (biggest deltas) | Cleaner without branched CA ⇒ confirmed |
| T4b | `disable_branched_sa=true` | Converse: SA branch contribution | |
| T5 | `validation_args.use_branched_attention=false` | Sanity: plain PhotoMaker with ckpt loaded (`lora_adapter` is frozen-at-init ⇒ should be clean) | If dirty ⇒ something else leaks |
| T6 | `ba_patch_top_k=0.5` (then 0.25) | Depth localization: patch only earliest SA sites | Which depth drives artifacts |

Notes:
- infer.py auto-selects bf16 and overrides `model.weight_dtype`/`pipeline.torch_dtype` — fine.
- Subtle: infer.py rebuilds processors on the chosen base and loads only `lora_A/B`, while in-training
  validation *copies* SDXL `base_weight` buffers into the RealVis model (`update_proc_weights_val`).
  So T0 on RealVis ≈ but ≠ training-val. Direction of comparisons is still valid; exact replication
  (buffer copy) can be added later if T0 fails to reproduce the artifact at all.
- Keep `val_debug=true` for one run to dump `prediction_stepXXX.png` / branch previews into `hm_debug/`.

### Expected outcomes → next action
- T1 fixes it → implement uncond-face-prompt fix (Section 2, F1) + retrain/keep-training.
- T3 fixes it → adopt ref-crop at val (cheap) and/or diversify training ref context (Section 4).
- T4 fixes it → freeze CA branches (`train_branched_ca_lora=false`) in re-run.
- T2 fixes it → validate on training base while iterating; align bases long-term.
- Nothing changes at 2k → wait for 20k ckpt (Section 3) where effects are gross.

---

## 2. (Q2) Additional codebase issues worth testing/fixing

- **F1 (fix): uncond face prompt under CFG.** branched_runtime.py:451-493 masks the *negative* prompt
  with the *positive* ID-token positions ×2.5 → garbage uncond face conditioning. Fix: apply ID
  masking only to the cond half; use plain negative-prompt embeds (or zeros) for the uncond half.
  Correct regardless of A/B outcomes; affects only CFG inference, not training.
- **F2: CA face-prompt is ~97% zero tokens** ("ID-only" prompt): zero K/V tokens act as attention
  sinks and dilute the ref half's text conditioning (SDXL attn has no k/v bias ⇒ exact zeros).
  Options: keep full prompt with boosted ID tokens (the pre-Feb-18 variant, which was working), or
  add an additive attention mask over zeroed tokens. Applies to train+val ⇒ retrain-affecting.
- **F3: face-branch K/V dilution in attn1** (`ref_hidden·ref_mask` keeps zero tokens): behavior scales
  with ref face fraction → interacts with the ref-domain gap (fact 5). Proper fix = additive attn
  mask (−inf on masked ref keys), behind a config flag, same path train+val ⇒ retrain-affecting.
- **F4: `update_proc_weights_val` franken-model** (SDXL base_weight buffers inside RealVis trunk).
  While debugging, set `pretrained_model_for_validation_name_or_path=null` (validate on training base).
- **F5: silently ignored dataset flags** (cosmic.py:950-959): either implement or remove from configs —
  right now the launch script implies ref-handling behavior that doesn't exist.
- **F6: stability floor**: `trainer.max_grad_norm=1.0`, small weight decay for branch LoRA, lower
  `lr_for_lora` (3e-5) — cheap insurance for 20k+ runs.
- **F7 (landmine): `_encode_reference_latent` tensor path** (lora2.py:652-658) resizes pixel tensors to
  the *latent* shape (128×128 px) then 8×-upscales latents. Unreachable today (refs are PIL) but fix.
- **F8 (dead code/config)**: `equalize_face_kv`/`equalize_clip` set on CA processors but not implemented
  in attn_processor_cleanest.py; README_ba_new.md points to deleted `attn_processor_clean.py`;
  model yaml default `weight_dtype: bf32` is invalid (works only because CLI overrides bf16).
- **Observation**: 4/840 delta tensors exactly zero (CA `ref_to_q` sites) — gradient flow is uneven
  through the ref half (its CA output only affects loss via later layers' face-branch K/V). Not a bug
  per se; worth knowing when interpreting probes.

---

## 3. (Q3) Tests once a ~20k checkpoint exists

- Re-run the same T0–T6 matrix: at 20k effects should be unambiguous (screenshot-level artifacts).
  The single most informative pair: **T1 (gs=1) vs T0 (gs=5)** at 20k.
- **Drift trajectory**: `weights_only_save_period=1` saves per epoch (2k steps) — run P2 probe over all
  epoch checkpoints → per-group delta-norm growth curves; correlate with per-epoch val id_sim and the
  visual degradation onset. Identifies *when* and *which group* (attn2 noise_to_v vs attn1 ref_to_v…) drifts.
- **Bisection**: find the first epoch where T0 shows artifacts; compare its probe profile vs previous
  epoch. If one group dominates, freeze that group in the next run (config already supports
  `train_branched_ca_lora=false`; SA-only or ref-only via `branched_attn_weight_mode=ref_only`).
- If gs=1 is clean even at 20k ⇒ ship F1 and consider guidance-aware training or lower guidance;
  if gs=1 is also broken ⇒ focus on F2/F3 + ref-domain (T3) and retrain variants (Section 4).

---

## 4. (Q4) Quick 2k re-run variants (config-level, ~1h each on vast)

Apply F1 + `pretrained_model_for_validation_name_or_path=null` first so validation is trustworthy, then:

| Run | Change vs cosm_new1_vast | Hypothesis tested |
|-----|--------------------------|-------------------|
| R1 | `train_branched_ca_lora=false` | Global CA drift is the main driver (fact 3) |
| R2 | `loss_kind` → blended: `loss_function._target_=src.loss.diffusion_loss.BlendedMaskedDiffusionLoss` + `loss_function.lambda_face=0.15` (drop `masked_alternating`) | Face-only gradient spikes drive drift |
| R3 | `trainer.max_grad_norm=1.0 optimizer.weight_decay=1e-2 lr_for_lora=3e-5` | Optimization hygiene slows degradation |
| R4 | `branched_attn_weight_mode=ref_only` | Leaving the noise branch at base weights (train only ref branch) prevents whole-image drift |
| R5 | train base = RealVis (`model.pretrained_model_name_or_path=SG161222/RealVisXL_V4.0`, keep RealVis val) | Base alignment matters end-to-end |

Suggested order: **R1+R3 together** (one run), then R2, then R4. Keep everything else identical
(same seeds/prompts/val set `manual_val_two`) so step-2000 panels are directly comparable across runs.
Compare: face quality at step 2000, id_sim/text_sim, and P2 probe of each run's epoch-1 weights.

---

## 5. (Q5) Other advice

- **Monitoring**: add a tiny Comet hook logging per-group branch delta norms every N steps (reuse P2
  code) + keep per-epoch weights. Catch drift numerically instead of by eyeballing validation.
- **Fixed panel**: keep one prompt×ref×seed panel rendered every epoch (already have; keep filenames
  stable) to build a degradation timeline.
- **id_sim context**: log a PM-only (BA-off) baseline id_sim on the same val set (T5 gives it) so BA's
  value-add/regression is quantified, not just absolute numbers.
- **Longer-term mechanism hardening** (after cause confirmed): additive attention masks instead of
  multiplicative zeroing (F2/F3), and revisit the ID-only face prompt vs boosted-full-prompt variant —
  the latter was in the last known-good state.
- **Bookkeeping**: fix F7/F8 cleanups; either implement or delete the ignored dataset flags (F5) so
  launch scripts say what they do.

---

## Implementation checklist (on approval)

1. `src/configs/inference/ba_2k_debug.yaml` (P1)
2. `scripts/inspect_ba_checkpoint.py` (P2 — code already validated today)
3. `scripts/crop_refs_to_face.py` (P3)
4. F1 uncond-face-prompt fix in `two_branch_predict` (branched_runtime.py)
5. Run matrix T0–T6 locally (RTX 4090, conda env `photomaker`), collect panel + metrics into
   `outputs/ba_debug/`; write results as `debug_planning_03Jul/ba_debug_results_v1.md`
6. Prepare R1–R3 launch-script variants (`serv_new_runs/start_ba_cosm_new1_vast_R{1,2,3}.sh`)
7. (later, per results) F2/F3 mechanism flags, monitoring hook, F5–F8 cleanups
