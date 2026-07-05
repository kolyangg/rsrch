# Overnight ablation matrix — N4 / N5 / N6 (design + why)

**Date:** 06 Jul 2026. **Driver:** `04Jul_findings.md` §9 — N3a proved that training DEGRADES face
identity below the untrained step-0 baseline (~0.40 id-sim), and more steps don't recover. This
matrix is a one-night screen to localize *why* and find any config that beats step-0.

**Master runner:** `serv_new_runs/run_overnight_N4_N5_N6.sh` — runs the three in order, each
self-stops after 3000 steps (`trainer.n_epochs × trainer.epoch_len`) and exits cleanly so the next
starts; a failure is logged and the master continues. Per-run logs in `serv_new_runs/logs/`.

## The three experiments (each changes ONE thing vs the N4 anchor)

| exp | script | loss | `ba_noise_lr_scale` | epoch_len / n_epochs | steps | isolates |
|---|---|---|---|---|---|---|
| **N4** | `start_ba_nr_alt_vast_N4.sh` | masked_alternating | **0.1** | 500 / 6 | 3000 | anchor: clean-ref + damped-noise + fast warmup; fine early val to find the peak |
| **N5** | `start_ba_nr_alt_vast_N5.sh` | masked_alternating | **0.0** (noise frozen) | 1000 / 3 | 3000 | the **noise pathway** — is it the melt/drift vector? |
| **N6** | `start_ba_nr_blend_vast_N6.sh` | **blended_masked** λ0.15 | 0.1 | 1000 / 3 | 3000 | the **loss** — does an every-step full-image anchor damp the melt? |

Shared recipe (all three), vs the initial `cosm_new1` run: `noise_and_ref`, RealVisXL validation,
clean reference crops (**no ref-crop jitter** — it hurt N3a, §9), `lr_for_lora=1e-4`,
`weight_decay=1e-3`, `max_grad_norm=1.0`, `warmup_steps=200`, `ba_uncond_face_fix=true`,
`ba_face_prompt_mode=id_only`, batch 2, rank 32, `CUDA_LAUNCH_BLOCKING=0`.

## Why these three (the reasoning)

The §9 diagnosis leaves two candidate causes for "training degrades identity":
1. **The noise cross-attn pathway** warps the face (the orange/melt cast; §4.2 shows noise-CA
   renders the whole gen image with no face/bg split, and `masked_alternating` trains it unanchored).
2. **The MSE objective itself** doesn't reward identity — it rewards denoising the training image,
   so it drags the already-good step-0 face toward a dataset average.

The matrix separates them:
- **N4 = anchor.** Best guess at "train without breaking": clean ref, ref pathway at full LR, noise
  pathway damped 10×, fast warmup so the real 0–2000 trajectory is visible at 500-step val. Answers
  "is there an early peak that beats 0.40?"
- **N4 → N5** ablates the **noise pathway** (0.1 → 0.0, everything else equal). Verified the noise
  group gets lr=0 and does not update.
  - N5 clean (no melt) **and** id-sim ≥ N4 ⇒ noise pathway is the damage vector ⇒ fix = make noise
    trainable-without-melting (identity loss), not the current MSE.
  - N5 clean **but** face↔body smear returns (the keanu motion failure) ⇒ confirms both pathways are
    needed (constraint #2) yet noise melts ⇒ problem well-posed.
  - N5 **also** crashes below 0.40 ⇒ the MSE objective (not the pathway) is the problem ⇒ identity
    loss needed regardless.
  - (N5 is numerically ref_only-equivalent in the forward, but a *clean, matched* diagnostic — the
    old refonly1 differed in loss/jitter/LR/boost. It is a probe, **not** a production candidate;
    constraint #2 stands.)
- **N4 → N6** ablates the **loss** (alternating → blended, same 0.1 damper). blended keeps a
  `(1-λ)` full-image anchor every step, which should reduce the noise-CA drift that alternating's
  face-only steps drive. Also the alt-vs-blend comparison the user asked for, on the better recipe.
  - N6 < melt / higher id-sim than N4 ⇒ adopt blended.
  - N6 ≈ N4 ⇒ loss shape isn't the lever; damper + objective dominate.

## Timing (45 GB card, ~2.5 s/step, ~15–20 min/val, async CUDA)

N4 ≈ 3.5–4 h (6 vals), N5 ≈ 3–3.5 h (3 vals), N6 ≈ 3–3.5 h → **total ~9.5–11 h**. Fits an
overnight. If the night is short: drop N6, or lower each `n_epochs`. If a specific card OOMs at
bs=2: `dataloaders.train.batch_size=1` + grad-accum 2, or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## What to read in the morning

Per run, `saved/<run_name>/` has `val_images/step_*`, `weights-epoch*.pth`, `config.yaml`,
`info.log` (with `manual_val_two/id_sim` per epoch and the `ba_norm/{sa,ca}×{ref,noise}` canary).
Score each with the same tool used in §9:

```
python scripts/idsim_report.py --refs-dir ../dataset_full/val_dataset/references_two \
    saved/ba_nr_alt_N4 saved/ba_nr_alt_N5 saved/ba_nr_blend_N6
```
(or the per-step sheet builder in scratchpad `make_n3a_sheets.py`, repointed).

**Benchmarks to beat:** step-0 baseline **0.40** (the real target — nothing has beaten it yet);
initial-run plateau **~0.32**; N3a plateau **~0.21**. Success = any checkpoint clearly > 0.40 with
no melt and integrated faces. If none clears 0.40, the matrix has still told us which lever matters,
and the next step is the identity-loss code change (needs approval).

## Results (fill in after the run)

| run | step 0 | 1000 | 2000 | 3000 | best | melt? | face↔body | notes |
|---|---|---|---|---|---|---|---|---|
| N4 | | | | | | | | |
| N5 | | | | | | | | |
| N6 | | | | | | | | |
