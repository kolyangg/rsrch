# Codex 9 Jul — Deep Investigation Plan Before Next BA Long Run

Request: do not fine-tune loss weights yet. First make a detailed plan for investigating larger possible improvements and correctness issues in the PhotoMaker branched-attention project. No training-code changes in this planning step.

## Goal

Find the next high-leverage research direction by auditing the full branched-attention training/inference contract, not by small tuning of `id_loss_weight` or `id_loss_max_timestep`.

The investigation should answer:

1. Is the current branched attention implementation faithful to the original BA plan?
2. Are train/inference masks, dimensions, prompts, ID embeddings, and reference/target streams aligned correctly?
3. Why did frozen CA work better than trainable CA, and can trainable CA be fixed safely?
4. Should the next long run test a different objective, fixed trainable CA, ID embedding conditioning, masking/dimension fixes, or an add-on such as `POSE_ADAPT_RATIO` / `CA_MIXING_FOR_FACE`?

## Inputs To Use After Approval

Local project:

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/model/photomaker_branched/branch_helpers.py`
- `src/pipelines/photomaker_branched_clean.py`
- `src/pipelines/br_pipeline_helpers.py`
- `src/loss/diffusion_loss.py`
- `src/loss/id_loss.py`
- `src/trainer/sdxl_trainers.py`
- `train.py`
- `infer.py`
- recent run scripts under `serv_new_runs/`
- recent logs and generated configs under `saved/*`
- validation outputs under `full_validation_results/*`
- prior notes in `debug_04Jul/`

External / reference sources:

- original PhotoMaker paper: `https://arxiv.org/abs/2312.04461`
- original PhotoMaker code: `https://github.com/TencentARC/PhotoMaker`
- local older PhotoMaker copy if useful: `/home/kolyangg/rsrch/PhotoMaker`
- BA design PDF: `/home/kolyangg/rsrch/_ba_scheme/ba_original_plan.pdf`

## Phase 0 — Reconstruct Current Ground Truth

Purpose: avoid making recommendations from stale assumptions.

Checks:

- Build a compact table of recent runs N3a/N6/N10/N11/N12/N13/N14/N15/N16/N17/N20 with:
  - `loss_kind`
  - `train_branched_ca_lora`
  - `disable_branched_ca`
  - `train_ba_all_steps`
  - `branched_attn_weight_mode`
  - `branched_attn_new_weight_kind`
  - `face_embed_strategy`
  - `use_id_embeds`
  - `id_loss_weight`
  - `id_loss_max_timestep`
  - `pose_adapt_ratio`
  - `ca_mixing_for_face`
  - `photomaker_start_step`
  - `branched_attn_start_step`
  - validation ID-sim and key visual failure modes
- Confirm which configs actually ran and which scripts are only proposals.
- Verify whether the current training/inference code path uses `attn_processor_cleanest.py` or another processor file.
- Verify whether the active N20 script being empty is just a local working-tree artifact or needs reconstruction.

Deliverable:

- A single "current facts" table in the investigation MD, with uncertainties explicitly marked.

## Phase 1 — Original PhotoMaker Loss And Objective Comparison

Purpose: decide whether `masked_alternating`, `blended_masked`, ID loss, or a hybrid objective is the right base for the next run.

Checks:

- Read the PhotoMaker paper's training objective and schedule, especially how identity-conditioned samples, class-word tokens, face crops, and alternating/regular losses are used.
- Compare original TencentARC implementation against local code:
  - original training scripts / loss implementation
  - local `src/loss/diffusion_loss.py`
  - local `src/loss/id_loss.py`
  - local `train.py` loss dispatch
  - local trainer output and weighting
- Trace exact behavior of:
  - `masked_alternating`
  - `blended_masked`
  - any regular full-image MSE term
  - any face-only / mask-only MSE term
  - ID loss gate by timestep
- Check whether alternating loss in the local code matches original PhotoMaker or has drifted due to BA-specific batch doubling, masks, or ref/target separation.
- Compare old run evidence:
  - original/alternating-style runs such as N3a/N4/N5
  - blended runs N6/N11/N14/N15/N17/N20
  - ID-loss runs N13/N14/N16/N17/N20
- Look for signs that `blended_masked` improved stability but weakened full-image prompt/pose learning, or that `masked_alternating` damaged face localization by over-isolating face steps.

Decision criteria:

- Prefer a loss if it improves face identity without creating fixed face position, long necks, or global prompt degradation.
- Avoid small weight tuning unless a code-level audit shows the objective is otherwise correct.

Likely candidate outcomes:

- Return to original-style alternating loss, but only after confirming masks/dimensions are correct.
- Use a scheduled hybrid: early full-image/blended, later face or ID emphasis.
- Keep blended loss but remove or reduce the failure source elsewhere, such as face-stream leakage or mask shape mismatch.
- Test training without explicit ID loss only if ID conditioning and BA transfer are made stronger through architecture/masking.

## Phase 2 — Branched Cross-Attention Audit And Fix Plan

Purpose: decide whether trainable CA is intrinsically bad or currently implemented/trained too broadly.

Known prior observation:

- Current CA runtime mostly follows the earlier screenshot:
  - generation half attends to general prompt
  - reference half attends to face prompt
  - no direct spatial face/background CA merge
- Previous N14 vs N16 evidence suggests trainable CA hurts strongly under current recipe.

Checks:

- Re-read `BranchedCrossAttnProcessor` end to end:
  - hidden-state split: `[noise_hidden, ref_hidden]`
  - encoder split: `[gen_prompt, face_prompt]`
  - query/key/value projection ownership
  - output concatenation
  - residual and output projection
  - whether `mask`, `mask_ref`, `attention_mask`, and `class_tokens_mask` are used
- Verify parameter cloning and trainability:
  - `ref_to_q/k/v`
  - `noise_to_q/k/v`
  - `to_out` handling
  - LoRA rank and scale
  - saved checkpoint contents when CA is frozen
- Verify optimizer grouping:
  - whether SA/CA ref/noise groups are separable
  - actual LR for CA ref vs CA noise
  - weight decay behavior
- Verify train/inference schedule mismatch:
  - `train_ba_all_steps=true`
  - inference `branched_attn_start_step=15`
  - PhotoMaker start step interaction
- Compare CA processor against BA plan PDF and screenshot:
  - whether generation branch should ever attend to face prompt directly
  - whether CA should spatially mix face/background outputs
  - whether ref branch should influence gen branch only through later SA or immediately

Fix ideas to evaluate, not implement until after audit:

- CA trainable only on reference branch first; keep generation CA frozen.
- Separate optimizer groups: `sa_ref`, `sa_noise`, `ca_ref`, `ca_noise`.
- Lower CA LR independently, especially `ca_ref`.
- Match training BA schedule to inference schedule instead of `train_ba_all_steps=true`.
- Add spatial mask use in CA output mixing if BA design expects it.
- Train CA only in selected UNet blocks / resolutions.
- Add a direct face-prompt CA path for generated face region only if the BA scheme supports it.

Decision criteria:

- Do not run trainable CA again until there is a narrow fix that prevents global prompt-attention drift.

## Phase 3 — Masking Logic Audit

Purpose: check whether failures are caused by incorrect or inconsistent face masks.

Checks:

- Trace mask origin for training:
  - dataset bbox JSON
  - image transforms
  - collate
  - model forward
  - mask resize to latent resolutions
  - mask passed to SA/CA processors
- Trace mask origin for inference/full validation:
  - validation bbox JSON
  - reference bbox JSON
  - generated-target bbox JSON
  - pipeline kwargs
  - `prepare_mask4`
  - runtime patching
- Verify mask coordinate systems:
  - original image pixel space
  - transformed target pixel space
  - transformed reference pixel space
  - latent pixel space
  - flattened attention token space
- Check whether bbox masks are correctly scaled for images where reference and target resolutions differ.
- Check whether mask and mask_ref stay attached to the correct half after batch doubling.
- Check CFG duplication:
  - unconditional/conditional latents
  - mask duplication
  - class token mask duplication
  - ID embedding duplication
- Check whether masks are binary or soft, and whether softness/expansion produce unexpected face-box sizes.
- Visually dump masks over target/reference for a small set:
  - Keanu "Rushing"
  - Jisoo "Night-ride"
  - Marion "Kickboxing"
  - Eddie low-ID examples
- Add temporary shape/assertion instrumentation only after approval, preferably behind a debug flag.

Decision criteria:

- Any mask-coordinate bug outranks new experiments. Fix first, then re-run a short validation.

## Phase 4 — Dimension And Resolution Audit

Purpose: explicitly verify all tensor dimensions because this dataset has reference and target images with different resolutions.

Checks:

- Training sample fields:
  - target image size before transform
  - target image tensor size after transform
  - reference image size before transform
  - reference tensor size after transform
  - bbox/mask sizes before/after transform
  - VAE latent shape for target
  - VAE latent shape for reference
- Runtime attention dimensions by UNet block:
  - batch size before/after branch doubling
  - hidden token count
  - inferred spatial H/W
  - mask token count
  - ref mask token count
  - prompt token count
  - face prompt token count
  - class-token mask shape
  - ID embedding shape
- Check non-square and mismatched aspect behavior even if current validation is 1024x1024.
- Verify that resize uses structured image geometry, not accidental assumptions like `sqrt(sequence_length)` when token grids may not be square.
- Verify that reference latents are resized/noised consistently with target latents at each timestep.

Concrete diagnostic to create after approval:

- A small debug script or mode that runs one train batch and one inference prompt, logging a compact shape table per stage and failing on mismatches.

Decision criteria:

- If shape/mask logic assumes square equal-size ref/target images anywhere, fix before another long run.

## Phase 5 — ID Embedding Conditioning And Face Embed Strategy

Purpose: decide whether identity should be injected through PhotoMaker tokens, BA face prompt, SA ID embeddings, explicit ID loss, or some combination.

Checks:

- Compare local strategies:
  - `face_embed_strategy=face`
  - `face_embed_strategy=id`
  - `face_embed_strategy=id_embeds`
  - `model.use_id_embeds=true/false`
  - `id_alpha`
  - `ba_face_prompt_mode`
- Trace where each strategy changes:
  - PhotoMaker `id_encoder`
  - `prompt_embeds`
  - `class_tokens_mask`
  - `face_prompt_embeds`
  - `id_embeds`
  - SA processor `id_to_hidden`
  - CA processor K/V
- Revisit N12 (`id_embeds`) and any old local/local PhotoMaker notes to understand whether it failed due to concept or due to CA trainability / other confounds.
- Check whether ID embeddings are trained/conditioned in the same place during training and inference.
- Check whether multiple reference images are averaged, concatenated, or encoded token-wise as expected.
- Compare with original PhotoMaker's identity token injection:
  - class-word replacement
  - number of ID tokens
  - whether local class-token mask matches original
- Evaluate whether ID loss may be compensating for weak/misaligned ID conditioning.

Possible directions:

- Re-test `id_embeds` only with frozen CA and verified masks.
- Use ID embeddings in SA face branch but not CA.
- Keep original PhotoMaker token conditioning and remove explicit ID loss for a clean architecture probe.
- Add normalization / scale checks for ID embeddings if the face branch overpowers pose.

Decision criteria:

- Prefer conditioning that improves identity without pinning face position or creating over-rigid reference geometry.

## Phase 6 — Compare Code To BA Original Plan PDF

Purpose: check whether the implementation still matches the intended BA design.

Checks:

- Extract text and diagrams from `/home/kolyangg/rsrch/_ba_scheme/ba_original_plan.pdf`.
- Make a scheme-vs-code table for:
  - latent/reference branch construction
  - self-attention Q/K/V routing
  - cross-attention Q/K/V routing
  - face/background spatial mixing
  - mask usage
  - ID conditioning
  - timestep schedule
  - trainable parameter selection
  - inference cleanup/restoration
- For every mismatch, classify:
  - intentional simplification
  - stale implementation
  - possible bug
  - design ambiguity needing user decision
- If the PDF expects something not implemented, propose the smallest testable implementation path.

Deliverable:

- A clear table: `BA plan`, `current code`, `risk`, `recommended action`.

## Phase 7 — Out-Of-Box Improvements To Consider

Purpose: add fresh ideas only after correctness checks, so they do not mask bugs.

Candidate ideas:

- Architecture / training:
  - freeze global generation text CA permanently, train only face/reference pathways
  - train only mid/up-block BA layers first
  - add low-rank branch adapters only to selected resolutions
  - use EMA or checkpoint averaging for long runs to avoid late over-strength
  - use regularization toward original attention outputs for CA if CA becomes trainable
  - KL/MSE distillation from frozen base on non-face regions
- Loss / schedule:
  - curriculum: early full-image prompt/pose preservation, later face identity emphasis
  - alternate by timestep bucket rather than by step parity
  - ID loss only on confidently detected generated faces, with skipped/flagged bad crops
  - non-face preservation loss to prevent face identity pressure leaking into props/body
- Data:
  - balance prompts by pose difficulty and identity
  - add stronger hard cases to validation subset: hats, profile, motion, occlusion, small face
  - verify bad examples are not annotation/mask outliers
- Inference:
  - checkpoint selection / early stopping from intermediate full-val, not final-step assumption
  - optional prompt-specific BA start step or strength schedule
  - face-aware reranking only for analysis, not as core training fix
- Reference processing:
  - face alignment/crop normalization check
  - multiple-reference selection or weighted reference aggregation
  - reference-pose diversity effects

Decision criteria:

- Out-of-box ideas should become experiments only if they isolate one mechanism and can be validated in a short run.

## Phase 8 — Last-Step Add-On: `POSE_ADAPT_RATIO` And `CA_MIXING_FOR_FACE`

Purpose: evaluate these knobs only after core implementation issues are understood.

Checks:

- Verify whether current code actually reads these knobs in the active processor path.
- Compare active `attn_processor_cleanest.py` behavior with older processors where these knobs were active.
- Determine whether `POSE_ADAPT_RATIO` should affect:
  - only self-attention face hidden mixing
  - CA face branch
  - early steps only
  - all BA steps
- Determine whether `CA_MIXING_FOR_FACE` should be used as:
  - face-region text/path blending
  - direct face prompt injection into generated face region
  - a runtime-only inference control
- If useful, design a small inference-only sweep first:
  - no training
  - use an existing checkpoint
  - test Keanu "Rushing", Jisoo "Night-ride", Marion "Kickboxing"
  - visually inspect face position and identity

Decision criteria:

- Treat these as add-ons, not primary fixes, unless the code audit shows they were part of the intended BA design and are currently accidentally disabled.

## Investigation Order

1. Reconstruct current facts and run table.
2. Read original PhotoMaker paper/code and compare loss/objective.
3. Extract BA plan PDF and compare scheme to code.
4. Audit current active BA code path: runtime, SA, CA, training helpers, pipeline helpers.
5. Audit masking and dimensions with targeted shape/mask diagnostics.
6. Audit ID conditioning and face embedding strategies.
7. Rank candidate fixes/experiments.
8. Only then consider `POSE_ADAPT_RATIO` / `CA_MIXING_FOR_FACE` sweeps.
9. Produce next-run recommendation and script only if the evidence is strong enough.

## Expected Outputs After Approval

- New detailed MD findings file in `debug_04Jul/`, likely named `Codex_9Jul_deep_ba_code_investigation.md`.
- Tables:
  - recent run config/evidence table
  - original PhotoMaker vs local loss table
  - BA plan PDF vs current code table
  - train vs inference tensor/mask shape table
  - ID conditioning strategy table
  - ranked issue list with severity and evidence
- Optional debug artifacts:
  - mask overlay images for selected problematic examples
  - shape logs for one train batch and one inference sample
  - small code snippets or scripts if needed for diagnosis
- A final recommendation:
  - next best fix
  - next short probe run
  - next long run only after the short probe passes
  - exact script if code changes are approved

## Proposed Go / No-Go Rules

Go to code changes only if:

- the issue is confirmed in the active code path, not just an old backup file;
- the change can be tested with a one-batch or small-inference diagnostic;
- expected effect is tied to a known failure mode such as face-position pinning, prompt drift, or identity loss.

Do not start a new long run if:

- masks/dimensions are not verified;
- trainable CA remains globally trainable without spatial/schedule/optimizer constraints;
- train/inference conditioning differs;
- the proposed change only tunes `id_loss_weight` without addressing a broader mechanism.

## Initial Hypotheses To Test

These are not conclusions yet; they are the main hypotheses the investigation should confirm or reject.

1. Current frozen-CA success means trainable CA is too broad/global, not that CA runtime itself is useless.
2. A mask or dimension mismatch could explain some fixed-position face failures and must be ruled out before more experiments.
3. `blended_masked` + ID loss may be compensating for weak ID conditioning, but can over-strengthen identity and freeze reference geometry late in training.
4. Original PhotoMaker alternating loss may still be valuable, but only if local BA batch/mask logic preserves its intended meaning.
5. `id_embeds` may deserve a clean re-test with frozen/fixed CA, because old results likely confounded it with trainable CA and other settings.
6. `POSE_ADAPT_RATIO` and `CA_MIXING_FOR_FACE` are likely secondary controls unless the audit shows they were accidentally disabled from the intended core BA path.

