# 7 Jul — N12/N14/N15/N16 results, the long-run config, and the 96-image validation

Continues `8Jul_results_N10_N11_N13_and_next.md`. This is the 6k batch (N14 combo, N15 SA-only, N16
ID-loss) + N12 (id_embeds, 3k). id-sim = InsightFace cosine of the generated face vs the reference
image, mean over the 24-image subsample (references_two = jensen+keanu), **excluding
`Reading_pa_jensen`** for apples-to-apples with prior runs. Runs in
`saved/{ba_combo_N14, ba_saonly6k_N15, ba_idloss6k_N16, ba_idembeds_N12}`.

## 1. Per-step id-sim (my method, excl. Reading; 23 images)

| step | N14 combo (SA-only+ID) | N15 SA-only | N16 ID-loss (CA trained) | N12 id_embeds |
|---|---|---|---|---|
| lever | freeze-CA **+** ID loss | freeze-CA | ID loss only | id_embeds cond. |
| bs | 1 | 2 | 1 | 2 |
| 0 | 0.40 | 0.40 | 0.40 | 0.37 |
| 2000 | 0.34 | 0.36 | 0.25 | — |
| 4000 | 0.369 | 0.373 | 0.319 | — |
| 6000 | **0.391** ↑ | **0.382** → | 0.324 (unstable) | 0.31 @3k |

Training-time metric (IDSimBest, incl. Reading, its own scale) agrees on ranking and reads a bit
higher: N14 **0.413** / N15 **0.411** / N16 0.347 / N12 0.341 at their final step. Benchmarks:
untrained step-0 **0.40**, old N4–N6 ceiling **~0.30**.

## 2. What works / what doesn't

**✓ N14 (combo: freeze-CA + ID loss) — the winner.** Smooth rise to 0.391 (mine) / 0.413
(training), **still climbing at 6k** (increments +0.029, +0.022 over the last 2 epochs). Both
levers compound: freeze-CA gives stable training, the ID loss keeps pushing identity.

**✓ N15 (SA-only alone) — strong, but plateauing.** 0.36→0.37→0.382, crossed the old ceiling
decisively but decelerating (last increment +0.009). Runs at **bs=2 (2× faster)**. Essentially ties
N14 at 6k but with less remaining headroom.

**✗ N16 (ID loss alone, CA still trained) — UNSTABLE, fails.** 0.40→0.25→0.32→0.324: crashes at 2k
and never recovers past ~0.32. Two runs of "ID-loss + CA-trained" (N13 @3k=0.383, N16 @3k=0.338)
even diverge from each other. **Key ablation result: the ID loss only works when paired with
freeze-CA** — with the cross-attn (drift/melt) pathway still trainable, the ID gradient destabilizes
it. So **freeze-CA is the essential stabilizer; the ID loss is the additive booster on top of it.**

**✗ N12 (id_embeds conditioning) — fails.** 0.37→0.31 at 3k, below step-0. Injecting PhotoMaker ID
features into the face-branch CA does not help; drop this lever.

## 3. Visual (`debug_04Jul/n14_n15_{keanu,jensen}.png`)

N14 and N15 @6k are clean and recognizable, **fix the hard motion poses** step-0 smears (keanu
Dancing 0.16→0.29/0.35, Jumping no-face→0.23/0.30) and **hold the easy frontal poses** near step-0.
No drift-melt. The one persistent artifact is the **Skiing goggle-collision melt** (keanu Skiing:
step-0 0.369, N15 0.213, **N14 0.045** — worst): the ID loss pushes the face harder into the
goggle-occupied box, so the combo actually *worsens* the prop-collision. This remains a MASK problem
(face bbox includes the prop), unaffected by these training levers — the next workstream.

## 4. Best config for the long run (20–30k) — N17

**`serv_new_runs/start_ba_longrun_vast_N17.sh`** — the **combo** recipe (the winner, and the one
still rising at 6k), scaled up:
- SA-only (`train_branched_ca_lora=false`) + ID loss (`use_id_loss=true`, weight 0.1, gate t≤500),
  on the blended N6 recipe (blended λ0.15, noise_and_ref, `ba_noise_lr_scale=0.1`, lr 1e-4, clean
  ref, wd 1e-3, clip 1.0, warmup 200, uncond_face_fix, id_only, RealVis val), **bs=1**.
- **26000 steps** (`epoch_len=2000 × n_epochs=13`), val + checkpoint **every 2000**. ~24–30 h at
  bs=1. Verified (compose): steps=26000, train_ca=False, use_id_loss=True.
- **Pick the best checkpoint by the id-sim curve + the 96-image validation**, not by assuming the
  last epoch is best (the combo was still rising at 6k, but may plateau; keep every 2k checkpoint).

Why the combo over SA-only alone: at 6k they tie (~0.39/0.41) but N14 was **still rising** while N15
had **plateaued**, and the ID loss directly targets the objective we care about — so the combo has
the most headroom over a long run. **Faster alternative if time-constrained:** SA-only alone at
bs=2 (2× faster; raise `n_epochs` on `start_ba_saonly6k_vast_N15.sh`) — nearly as good at 6k but
plateauing, so likely a lower final ceiling.

Not changed for the long run (kept constant, as they're what worked): constant LR after warmup (no
decay), weight 0.1, freeze-CA. If N17 plateaus early, the levers to try next are a mild cosine LR
decay and/or a small ID-weight bump — but only after seeing the curve.

## 5. Will the 96-image full validation help? — YES, strongly. Do it.

The current verdict rests on the **2-identity subsample** (jensen + keanu, 23 images), which is
noisy and dominated by those two specific faces (jensen ~0.42, keanu ~0.35). Two concrete problems
it can't resolve:
1. **"Does the winner actually beat untrained (0.40)?"** is currently *metric- and sample-dependent*
   — N14 is 0.391 (my method, below 0.40) yet 0.413 (training metric, above 0.40). A 4×-larger,
   8-identity set gives a robust aggregate that settles this.
2. **Generalization** — everything so far is measured on 2 faces. The 96-image set (8 identities ×
   12 prompts) shows whether the gains hold across diverse identities or are a jensen/keanu artifact,
   and gives per-identity breakdowns (already emitted by `full_val_metrics.py`).

So the 96-image validation is not just "nice to have" — it is what promotes these subsample results
to a trustworthy conclusion, and it's how we should choose the winning checkpoint for N17. It's
already set up (`serv_new_runs/run_full_validation.sh`), and per your request the run list is now
**ordered most→least promising** (N14, N15, N11, N13, N6-baseline, then the rest) so the winners +
the baseline reference are scored first if time runs short.

## 6. Summary ranking (subsample, for prioritizing the 96-val)

| rank | run | id-sim (mine / training) | verdict |
|---|---|---|---|
| 1 | N14 combo 6k | 0.391 / 0.413 (rising) | best; long-run base |
| 2 | N15 SA-only 6k | 0.382 / 0.411 (plateau) | strong, faster (bs2) |
| 3 | N11 SA-only 3k | 0.384 / — | winner (earlier) |
| 4 | N13 ID-loss 3k | 0.383 / — | winner (earlier) |
| — | N6 blended | 0.297 | old-ceiling reference |
| 5 | N16 ID-loss 6k (CA on) | 0.324 / 0.347 | unstable — fails |
| 6 | N12 id_embeds | 0.313 / 0.341 | fails |
| 7 | N10 co-adapt | ~0.30 | fails |
| 8–10 | N4 / N5 / N3a | 0.31 / 0.27 / 0.21 | early runs |
