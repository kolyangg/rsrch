# 08 Jul — 96-image full validation, ALL 11 runs

Extends `7Jul_96val_analysis.md` with the runs that finished since (N12 id_embeds, N4 alt-loss, N10
co-adapt, N5 frozen-noise). All 11 runs now scored on the full set (8 identities × 12 prompts = 96
images, final checkpoint). Metrics: `full_validation_results/metrics.json`. Visual comparison PDF
(all 11 runs side by side, per identity, id-sim on every image): `full_validation_results/full_val_report.pdf`.

## 1. Full ranking (96-image mean id-sim)

| rank | run | step | mean | beats N6 baseline on |
|---|---|---|---|---|
| 1 | **ba_combo_N14** (SA-only + ID loss) | 6k | **0.332** | **8/8** identities |
| 2 | ba_idloss_N13 (ID loss) | 3k | 0.315 | 7/8 |
| 3 | ba_saonly6k_N15 (SA-only) | 6k | 0.312 | **8/8** |
| 4 | ba_idloss6k_N16 (ID loss, CA trained) | 6k | 0.281 | 5/8 |
| 5 | ba_saonly_N11 (SA-only) | 3k | 0.277 | 6/8 |
| 6 | ba_idembeds_N12 (id_embeds) | 3k | 0.272 | 6/8 |
| 7 | ba_nr_alt_N4 (masked_alternating) | 3k | 0.256 | 8/8 |
| — | **ba_nr_blend_N6 (blended baseline)** | 3k | **0.239** | — |
| 8 | ba_coadapt_N10 (co-adaptation) | 3k | 0.232 | 4/8 (below base) |
| 9 | ba_nr_alt_N5 (frozen-noise) | 3k | 0.224 | 3/8 (below base) |
| 10 | ba_nr_alt_N3a | 10k | 0.171 | — |

## 2. What the newly-added runs confirm

- **Co-adaptation (N10, 0.232) and frozen-noise (N5, 0.224) are BELOW the blended baseline (0.239)**
  and beat it on only 4/8 and 3/8 identities — definitively dead ends (training the base LoRA, or
  freezing the noise pathway, actively hurts).
- **id_embeds (N12, 0.272) and alt-loss (N4, 0.256) are mild improvements over baseline but well
  below the winners** — not worth pursuing.
- **Only three recipes clearly work** (0.31–0.33, beating the baseline on 7–8/8 identities):
  **N14 (combo), N15 (SA-only), N13 (ID loss)** — and the combo is best. This exactly matches the
  subsample verdict, now confirmed across 8 diverse identities.
- **N14 (combo) and N15 (SA-only) beat the baseline on ALL 8 identities** → the winning recipe
  generalizes; it is not a jensen/keanu artifact.

## 3. Unchanged conclusions (from `7Jul_96val_analysis.md`)

- **N14 combo is the best recipe**, generalizes, and was still rising at 6k.
- **ID loss requires freeze-CA:** N16 (ID loss @6k, CA trained) = 0.281 < N13 (ID loss @3k) = 0.315
  — with CA unfrozen the ID loss degrades over steps.
- **Mean is dragged by hard/occluded identities:** eddie 0.14 (best-any-run only 0.141 — the model
  simply can't reproduce Eddie Murphy), marion 0.26, jisoo 0.31 (earring/hair occlusion melt). Same
  MASK problem as the Skiing goggles; a separate (inference/data) workstream that would specifically
  lift these.
- **STILL MISSING — the untrained / stock-PhotoMaker baseline on 96 images.** We know the winners
  beat the *trained* blended baseline by ~+0.09, but not whether they beat *untrained* (on the
  subsample, untrained step-0 0.40 > trained ~0.39). Run these two cheap inferences to settle it:
  `infer.py --config-name inference/full_val saved_checkpoint=null …` (untrained BA), and the same
  with `validation_args.use_branched_attention=false` (stock PhotoMaker). See `7Jul_96val_analysis.md` §5.

## 4. Recommended next run — the combo at 10k (N18)

`serv_new_runs/start_ba_combo10k_vast_N18.sh` — the winning recipe (SA-only + ID loss, w0.1, blended
λ0.15, freeze-CA, bs=1) extended to **10000 steps** (val + checkpoint every 1000). It sits between
the done 6k (N14) and the long 26k (N17): confirms whether the still-rising combo keeps improving
past 6k and yields a stronger checkpoint, at a tractable ~10–14 h. Freeze-CA is mandatory; keep every
1k checkpoint and choose the winner by the 96-image validation, not the last epoch. If N18 is still
clearly rising at 10k, promote to N17 (26k); if it has plateaued, 10k is the sweet spot.

## 5. Artifacts
- Metrics: `full_validation_results/metrics.json` (11 runs; per-run mean + per-identity + per-image).
- PDF (all 11 runs, per-identity grids + summary table): `full_validation_results/full_val_report.pdf`
  — rebuild with `python infer_tools/pdf_full_val.py --config infer_tools/full_val_report.yaml`.
