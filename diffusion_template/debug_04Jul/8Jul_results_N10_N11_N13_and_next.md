# 8 Jul — results of N10/N11/N13 + next experiments

Continues `7Jul_experiments_analysis.md` §7 (the substantial matrix). N12 (id_embeds) still running.
id-sim = InsightFace cosine vs reference, mean over the 24-prompt panel **excluding `Reading_pa_jensen`**
for apples-to-apples with N4–N6 (it was broken then; it is FIXED now via `force_manual`, but kept
excluded here so the fix doesn't inflate the new runs). Runs in `saved/{ba_coadapt_N10,
ba_saonly_N11, ba_idloss_N13}`. All ran clean (N13 confirmed at bs=1, "Training finished
successfully"; the 491 OOM_SKIP in its info.log are residual from the earlier bs=2 attempt).

## 1. Per-step id-sim (excl. Reading_pa_jensen; 23 panels)

| step | N10 co-adapt | N11 SA-only | N13 id-loss |
|---|---|---|---|
| lever | `non_ba_train=true` | `train_branched_ca_lora=false` | `use_id_loss=true` (w0.1) |
| 0 | 0.400 | 0.400 | 0.395 |
| 1000 | 0.258 | 0.317 | 0.280 |
| 2000 | 0.296 | 0.368 | 0.329 |
| 3000 | 0.275 | **0.384** | **0.383** |
| jensen / keanu @3000 | 0.322 / 0.232 | 0.419 / 0.352 | 0.404 / **0.362** |

Benchmarks: step-0 (untrained) **0.40**, old N4–N6 ceiling **~0.30** (N6 0.297).

## 2. What works / what doesn't

**✓ N11 (SA-only — freeze the branched cross-attention):** smooth monotonic rise
0.317→0.368→0.384, blew past the 0.30 ceiling, **still climbing at 3k**. Freezing the CA branch
removes the §4.2 drift/melt pathway (attn2 noise_to_v), so training is clean and stable. bs=2 OK.

**✓ N13 (ID loss — identity-supervised objective):** strong rise 0.280→0.329→0.383, **still
climbing** (fastest of the three: +0.054/1k in the last stretch), and the **best keanu / hard-pose
score of any trained run (0.362)**. The direct identity supervision is doing exactly what it should.
Ran at bs=1 (memory).

**✗ N10 (co-adaptation — train base LoRA too):** 0.258→0.296→0.275, back to the old ~0.30 plateau.
More trainable capacity is NOT the lever; if anything it re-introduces mild drift. Drop this line.

**Both winners are competitive with untrained step-0 (~0.38 vs 0.40) and STILL RISING at 3k** — the
first trained checkpoints to escape the 0.30 ceiling. Extrapolating the slopes, both would likely
cross 0.40 with more steps. This is the breakthrough.

## 3. Visual confirmation (`debug_04Jul/n11_n13_{keanu,jensen}.png`)

The id-sim gains are real, not artifacts. Faces are clean, recognizable, well-integrated; **no
orange-cast/melt drift** (that was the N3a/initial failure — gone here). Mechanism, per the panels:
- **Both fix the hard cases step-0 smears:** keanu Dancing 0.165→0.36–0.38, Jumping no-face→0.24,
  Chef 0.076→0.47(N11); jensen Dancing 0.347→0.40, Jumping 0.306→0.42.
- **Both hold the easy frontal cases near step-0** (Rushing/Kickboxing/Angry/Crying).
- **Persistent artifact — the Skiing goggle-melt** in ALL trained runs (keanu Skiing N11 0.011 /
  N13 0.085): the gen face bbox includes the goggles, so the strengthening face branch paints the
  ref face onto the prop. This is a MASK problem (inference/data-side), unaffected by these training
  levers — the next thing to fix after the training recipe is locked.

## 4. Key inference

N11 (freeze-CA) and N13 (ID loss) are **independent levers** — one is architectural pathway
selection, the other is the objective. Both work on their own. They should **compound**: freeze-CA
gives clean, stable training; the ID loss adds direct identity pressure (and helps hard poses most).
And both were still rising at 3k → **run longer**.

## 5. Next experiments — combine the winners + run longer

All on the N6 blended recipe (blended λ0.15, noise_and_ref, lr 1e-4, clean ref, wd 1e-3, clip 1.0,
warmup 200, uncond_face_fix, id_only, RealVis val), **6000 steps** (2× the current runs; both
winners were still climbing at 3k), val every 1000.

| run | levers | vs N6 | bs | tests |
|---|---|---|---|---|
| **N14** (primary) | SA-only **+** ID loss | `train_branched_ca_lora=false` + `use_id_loss=true` (w0.1) | 1 | **the compounding bet** — both winners together, longer. Best shot at clearly beating step-0 0.40 |
| **N15** | SA-only only | `train_branched_ca_lora=false` | 2 | SA-only alone at 6k — does the ID loss add on top of it? (N14 vs N15) |
| **N16** | ID loss only | `use_id_loss=true` (w0.1) | 1 | ID loss alone at 6k — does freeze-CA add on top of it? (N14 vs N16) |

Read-out: N14 vs N15/N16 isolates each lever's marginal value at scale; all three show whether the
winners cross **0.40** (step-0) with 2× steps. Success = N14 > 0.40 with clean faces.

**If time-constrained, run only N14** (the combination, longer) — it's the single highest-value run.
Scripts: `serv_new_runs/start_ba_{combo_N14, saonly6k_N15, idloss6k_N16}_vast.sh`; master
`serv_new_runs/run_batch_N14_N15_N16.sh` (self-stop 6000 steps each; continue-on-fail; N14 first).
NB N14/N16 use bs=1 (ID-loss VAE decode memory); N15 uses bs=2.

**Deferred (next workstream, not this batch): the prop-collision melt** (Skiing goggles, Kickboxing
sweatband). Fix = exclude props from the gen face box (tighten the bbox to face landmarks / face
segmentation instead of the YOLO box). Inference/data-side; tackle after the training recipe is
locked by this batch.

## 6. Results (fill after the run)

| run | 0 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | best | beats step-0 0.40? |
|---|---|---|---|---|---|---|---|---|---|
| N14 combo | | | | | | | | | |
| N15 SA-only | | | | | | | | | |
| N16 id-loss | | | | | | | | | |
