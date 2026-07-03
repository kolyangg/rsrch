# BA Debug Results — v1 (03 Jul 2026, local matrix on epoch-2 / 4k-step checkpoint)

Setup: `serv_new_runs/run_ba_debug_matrix.sh` with `CKPT=saved/03Jul_start_ba_cosm_new1_vast/weights-epoch2.pth`,
8 images/test (jensen + keanu × 4 prompts, seed 0), RealVisXL trunk, local RTX 4090 laptop.
Images: `outputs/ba_debug/<ID>/`; contact sheets: `debug_planning_03Jul/assets/*_matrix.jpg`.
Scores: `scripts/idsim_report.py` (InsightFace cosine to the reference; NOTE — id_sim can be
*inflated* by pathology: in T0 the detector locks onto the pasted mini-ref-face).

## Result table

| Test | What it isolates | mean id_sim | no-face | Visual verdict (Kickboxing/jensen & panel) |
|------|------------------|------------:|--------:|--------------------------------------------|
| T0 (trained, gs 5) | baseline = val behavior | 0.345* | 1/8 | **Catastrophic**: shrunken second ref-face + giant mouth pasted into the mask region, paint smears. *score inflated by detector finding the pasted ref face |
| T0b (untrained clones) | is architecture-at-inference OK? | **0.446** | 0/8 | Clean, good identity — matches step-0 val. Architecture fine; **trained deltas are the trigger** |
| T1 (gs 1) | CFG amplification | 0.167 | 0/8 | No paint explosion, but face = collage of ref fragments; whole image washed out (gs1 not viable). **Core corruption exists without CFG; CFG multiplies severity** |
| T7 (F1 uncond-face fix, gs 5) | garbage uncond face prompt | 0.286 | 0/8 | **Most of the catastrophe gone** with a one-flag fix: coherent face, decent identity, mild red/blue tint remnants |
| T4 (branched CA disabled, gs 5) | trained CA branches | **0.414** | 0/8 | **Best trained-ckpt result** — clean, close to untrained quality; slight pasted-edge look |
| T3 (training-style cropped refs) | ref-domain gap at inference | 0.259 | 0/8 | No paint, but mangled features (eyes lost); blurry 177px crop also weakens the ID embedding. Not an inference-side rescue |

Not run locally: T2 (SDXL-trunk swap) — SDXL-base was evicted for disk (would leave <3GB);
its question (trunk mismatch) is now secondary to the findings below. Run on vast if still wanted.

## Conclusions (ranked, evidence-backed)

1. **The trained cross-attention branches are the primary destroyer.** Disabling branched CA
   at inference (T4) on the *same trained checkpoint* removes the catastrophe and nearly
   restores untrained quality (0.414 vs 0.446). Matches the weight probes: largest deltas =
   attn2 `noise_to_v` (global text-value pathway), doubling every 2k steps after warmup
   (epoch1→epoch2 mean fro 0.84→1.22).
2. **The uncond face prompt under CFG is the main amplifier.** The F1 flag alone (T7) turns
   the double-face horror into a coherent face. Legacy behavior feeds the negative prompt
   masked by cond ID positions ×2.5 into an untrained pathway that guidance-5 extrapolates.
3. **Core drift is real even without CFG** (T1: ref-fragment collage at gs 1) — training-side
   fixes are still required; F1 alone only removes the explosion.
4. **C6 — stale gen-bbox**: the PM-pass bbox goes stale when the BA trajectory shifts the
   composition (smoke run: ghost ref-face painted into the old bbox, real face grew outside).
   Scales with drift; secondary once 1–2 are fixed, but argues for re-detecting/tracking the
   face mask at `branched_attn_start_step` instead of freezing the PM-pass bbox.
5. **Ref-crop at inference is not a rescue** (T3) — the ref-domain gap matters for what the
   branches *learn* during training, not as an inference-time patch; the 4× blurry upscale
   also weakens the PM ID embedding.

## Recommended next actions

1. **Rerun training as R1R3 + F1** (already scripted):
   `bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R1R3` — freezes CA branches (kills the
   dominant drift channel), grad-clip 1.0 + wd 1e-2 + lr 3e-5 (slows the rest), F1 on,
   validation on the training base (safe with the F9 fix). Expect step-2000/4000 panels ≈
   T4-quality with growing identity gains from the SA face branch.
2. Strong secondary variant worth queuing: `branched_attn_weight_mode=ref_only`
   (train only ref-branch weights; leaves the entire gen/noise pathway at base weights —
   removes the whole-image drift channel by construction):
   `bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh R1R3 branched_attn_weight_mode=ref_only`
3. Keep `ba_uncond_face_fix=true` for ALL future validation/inference of BA runs.
4. Stage-2 mechanism items (plan v1): mask re-detection at BA start (C6), additive attention
   masks instead of multiplicative zeroing, training-ref context diversification.

## Raw metric detail

See `outputs/ba_debug/*.log` and the idsim printout in the worklog; per-image scores in this
doc's source run: T0 {Kick_j 0.53*, Ski_j 0.09, Ski_k NO FACE, ...}, T0b {0.23–0.59}, ...
(*pasted-face detection). 2k vs 4k delta-norm trajectory: `scripts/inspect_ba_checkpoint.py
saved/03Jul_start_ba_cosm_new1_vast/weights-epoch{1,2}.pth`.
