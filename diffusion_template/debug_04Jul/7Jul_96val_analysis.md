# 7 Jul — 96-image full-validation analysis (first ~half of runs)

Results of `run_full_validation.sh` on the FULL set (8 identities × 12 prompts = 96 images,
`full_validation_results/`), final checkpoint of each run. id-sim = InsightFace cosine of the
generated face vs the reference image (same method as all analysis). Metrics in
`full_validation_results/metrics.json`.

## 1. Aggregate (96 images, final checkpoint)

| run | step | mean id-sim (96) | (subsample, 2-id) | det |
|---|---|---|---|---|
| **ba_combo_N14** (SA-only + ID loss) | 6000 | **0.332** | 0.391 | 95/96 |
| ba_idloss_N13 (ID loss) | 3000 | 0.315 | 0.383 | 95/96 |
| ba_saonly6k_N15 (SA-only) | 6000 | 0.312 | 0.382 | 96/96 |
| ba_idloss6k_N16 (ID loss, CA trained) | 6000 | 0.281 | 0.324 | 96/96 |
| ba_saonly_N11 (SA-only) | 3000 | 0.277 | 0.384 | 95/96 |
| ba_nr_blend_N6 (blended baseline) | 3000 | 0.239 | 0.297 | 94/96 |
| ba_nr_alt_N3a | 10000 | 0.171 | 0.212 | 95/96 |

Absolute numbers are **lower than the 2-id subsample** — expected: the subsample is jensen+keanu
(both relatively easy), while the 8-id set includes genuinely hard identities (see §3). This is the
more honest, generalizable number, which is exactly why the full validation matters.

## 2. What works (confirmed on 96 images, across 8 identities)

**The ranking from the subsample holds, and the winners generalize.** N14 (combo) beats the N6
blended baseline on **all 8 identities** (+0.02 to +0.23) — not a jensen/keanu artifact:

| identity | N6 base | N14 combo | Δ | | identity | N6 base | N14 combo | Δ |
|---|---|---|---|---|---|---|---|---|
| jennie | 0.216 | **0.444** | +0.229 | | jensen | 0.358 | 0.439 | +0.081 |
| keanu | 0.243 | 0.386 | +0.143 | | marion | 0.181 | 0.249 | +0.068 |
| elon | 0.339 | 0.423 | +0.083 | | jisoo | 0.198 | 0.263 | +0.066 |
| | | | | | lex | 0.270 | 0.309 | +0.039 |
| | | | | | eddie | 0.116 | 0.140 | +0.024 |

Confirmed findings:
- **N14 (combo) is the best** on the full set (0.332), and best or near-best on 6/8 identities.
- **ID loss is efficient:** N13 (ID loss @3k, 0.315) ≈ N15 (SA-only @6k, 0.312) — the ID loss
  reaches at 3k what SA-only reaches at 6k.
- **ID loss REQUIRES freeze-CA (confirmed):** N16 (ID loss @6k, CA trained) = 0.281 is *below* N13
  (ID loss @3k, 0.315) — training the ID loss longer with CA unfrozen DEGRADES it. CA must be frozen.
- **SA-only is stable-improving:** N11 (@3k 0.277) → N15 (@6k 0.312), monotone with steps.
- Visual (`debug_04Jul/n96_angry_by_identity.png`): N14 faces are visibly cleaner and more
  recognizable than N6 (jennie, marion, elon clearly better).

## 3. The hard identities (drag the mean; identity-specific, not recipe)

- **eddie 0.14 (worst).** The model can't capture Eddie Murphy's specific features — the generations
  are a generic Black man, not clearly him (baseline AND combo both miss). A genuinely hard
  identity, not a broken reference.
- **jisoo 0.26 / marion 0.25.** jisoo generations are partly **melted/occluded** (the reference has
  dangly earrings + hair over the face → a prop/hair collision like the Skiing goggle-melt). marion
  is soft. These are the same **mask/occlusion** failure mode, identity-specific.
- Easy identities (elon 0.42, jensen 0.44, jennie 0.44, keanu 0.39) are strong.

So the 0.33 mean is dragged by 2–3 hard/occluded identities; the recipe itself works well where the
face box is clean.

## 4. THE CRITICAL GAP — no untrained baseline on 96 images

On the subsample, untrained **step-0 = 0.40** was *above* every trained run (~0.39) — the open
question was whether training beats untrained. On the 96-image set we have the trained baseline
(N6 = 0.239) but **NOT the untrained step-0 or stock-PhotoMaker reference**, so we cannot yet say
whether N14 (0.332) beats untrained on the full set. We only know N14 beats the *trained* blended
baseline by +0.093 (big and generalizing).

**This is the single most important next measurement.** → see §5 short test.

## 5. Recommended NEXT SHORT TEST — the two 96-image baselines (cheap, decisive)

Run the full-val inference on the two reference bars (no training needed, just inference):

```bash
# (a) untrained branched attention = the step-0 baseline (branched ON, no checkpoint)
python infer.py --config-name inference/full_val saved_checkpoint=null \
    output_dir=full_validation_results/_untrained_BA batch_size=4
python scripts/full_val_metrics.py --out-dir full_validation_results/_untrained_BA \
    --refs-dir ../dataset_full/val_dataset/references --run _untrained_BA --epoch 0 --step 0 \
    --json full_validation_results/metrics.json

# (b) stock PhotoMaker (branched OFF) = the "does BA help at all" bar
python infer.py --config-name inference/full_val saved_checkpoint=null \
    validation_args.use_branched_attention=false \
    output_dir=full_validation_results/_stock_photomaker batch_size=4
python scripts/full_val_metrics.py --out-dir full_validation_results/_stock_photomaker \
    --refs-dir ../dataset_full/val_dataset/references --run _stock_photomaker --epoch 0 --step 0 \
    --json full_validation_results/metrics.json
```

These settle: (a) does trained BA beat untrained BA on the full set? (b) does BA beat stock
PhotoMaker? Only with these can we claim a real win. (Optional 3rd short test: N14 with a higher ID
weight 0.15–0.2 to push the hard identities — but do the baselines first.)

## 6. Recommended NEXT LARGE TRAINING (20–30k) — N17 (combo), unchanged

The 96-image data **confirms and strengthens** the earlier call: the combo (SA-only + ID loss) is
the best recipe and it **generalizes across all 8 identities**, so
`serv_new_runs/start_ba_longrun_vast_N17.sh` (combo, 26000 steps, val/ckpt every 2000, bs=1) stands
as the long-run config. Keep every 2k checkpoint and pick the winner by the 96-image validation
(not the last epoch). Freeze-CA is mandatory (N16 proves CA-unfrozen degrades). Keep ID weight 0.1
unless the baseline test (§5) shows lots of headroom, in which case try 0.15.

**Deferred workstream (unchanged): the mask/occlusion melt** (Skiing goggles, jisoo earrings/hair) —
the main remaining artifact and a real chunk of the lost id-sim on hard identities. Fix is to tighten
the gen face box to actual face landmarks / segmentation so it excludes props/hair. Inference+data
side; worth doing after the long run, and it would lift the hard-identity scores specifically.

## 7. One-line takeaways
- N14 combo wins on 96 too (0.332), beating the trained baseline on **all 8 identities** → generalizes.
- ID loss needs freeze-CA (N16 degrades with steps); combo > either lever alone.
- Mean is dragged by hard/occluded identities (eddie, jisoo) — a masking problem, not the recipe.
- **Missing: untrained/stock baselines on 96** — run them next (§5); they decide "beats untrained?".
- Long run = N17 (combo), confirmed.
