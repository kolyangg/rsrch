# 7 Jul — analysis of the overnight matrix (N3a/N4/N5/N6) + next 3 experiments

Continues `04Jul_findings.md` §9 and `overnight_N4_N6_plan.md`. All id-sim = InsightFace cosine of
the generated face vs the reference, mean over the 24-prompt panel **excluding `Reading_pa_jensen`**
(broken val mask — see §4). Runs live in `saved/{ba_nr_alt_N3a, ba_nr_alt_N4, ba_nr_alt_N5,
ba_nr_blend_N6}`. (N6 was saved as `ba_nr_blend_N6`, the script's own run_name — the same run the
prompt calls "N6".)

## 1. Per-step id-sim (excl. Reading_pa_jensen; 23 panels)

| step | N3a | N4 | N5 | N6 |
|---|---|---|---|---|
| config | alt, noise .25, lr 5e-5, +jitter, wd 1e-2 | **alt, noise .1, lr 1e-4, clean, wd 1e-3** | **alt, noise 0.0 (frozen)** | **blended λ.15, noise .1** |
| 0 | 0.400 | 0.395 | 0.399 | 0.398 |
| 500 | — | 0.172 | — | — |
| 1000 | — | 0.192 | 0.228 | 0.257 |
| 1500 | — | 0.268 | — | — |
| 2000 | 0.167 | **0.308** | 0.259 | 0.294 |
| 2500 | — | 0.260 | — | — |
| 3000 | — | 0.286 | 0.271 | **0.297** |
| 4000–10000 | 0.17–0.21 (flat) | — | — | — |

Per-identity at the best checkpoints (jensen / keanu): N4@2000 0.348 / 0.272 · N5@3000 0.296 / 0.249
· N6@3000 0.318 / 0.276.

## 2. What the numbers + montages say (`best_compare_{jensen,keanu}.png` in debug_04Jul)

1. **Step-0 (untrained clones) is still the best in aggregate (~0.40)** — unbeaten by every trained
   checkpoint. The §9 conclusion stands at the panel level.
2. **The N3a regression is fixed.** Removing ref-crop jitter + LR back to 1e-4 (N4/N5/N6) recovered
   to ~0.29–0.31 — the initial-run level — vs N3a's 0.17–0.21 plateau. So the two N3a suspects
   (jitter weakening the ref signal; over-low LR) are **confirmed**.
3. **Training and step-0 are COMPLEMENTARY, not simply worse** (the key new insight):
   - Training **helps the hard cases** where step-0's face branch smears/fails: keanu Dancing
     0.16→0.35, keanu Jumping no-face→0.27, keanu Chef 0.07→0.33, jensen Jumping 0.30→0.50.
   - Training **hurts the easy frontal cases** where step-0 is already excellent: keanu Rushing
     0.49→0.12–0.42, Angry 0.48→0.25, Crying 0.45→0.01–0.26; similar for jensen.
   - Net aggregate favours step-0 only because the panel has more easy cases. This reframes the
     problem: we don't need "train better everywhere", we need "keep step-0's easy-case quality
     while adding training's hard-case gains".
4. **blended (N6) > alternating (N4) on behaviour, not peak.** N4 peaks higher (0.308@2000) but is
   peaky (declines after) and **melts more**; N6 is smoother, sustains to 3000, and has **less
   melt** — the clearest single example: jensen Kickboxing N4@2000 **0.063** (dark band melted over
   the eyes) vs N6@3000 **0.325**. blended's every-step full-image anchor damps the noise-CA drift
   (§4.2) that alternating's face-only steps drive. **⇒ blended is the recipe to build on.**
5. **Freezing the noise pathway (N5) did NOT help** — N5 ≤ N4/N6 at matched steps and did **not**
   fix the melt (Skiing still −0.04). So at 0.1× the noise pathway is not the dominant damage
   vector, and freezing it just removes the hard-case integration it provides. Revises the §4.2
   "noise-CA is the melt vector" hypothesis: at a damped LR it's fine; the melt is elsewhere (§3.6).
6. **The prop-collision melt is the worst remaining artifact and is a MASK problem, not a weight
   problem.** Skiing (goggles) and Kickboxing (sweatband) melt in *all* trained runs because the
   face bbox includes the prop, so the strengthening face branch paints the ref face onto the
   goggles/band. Freezing noise (N5) doesn't fix it; blended reduces it. The real fix is masking
   (exclude props from the face box), which is inference/data-side.

## 3. Direction — where to go next

**Strategic read:** config sweeps are converging in the **0.29–0.33 band, below step-0's 0.40**.
The MSE denoising objective rewards reconstructing the training image, not identity, so it can
recover toward but not beat the untrained baseline. **The single highest-value direction is to
change the objective — add an identity loss** (ArcFace/InsightFace cosine between the generated
face crop and the reference, as an auxiliary term). That is the only lever likely to push a trained
checkpoint *above* 0.40. It is a contained code change (new loss in `src/loss/diffusion_loss.py` +
a face-embed hook in the training step) and needs your approval before I implement it.

**Tonight (config-only, no code change), the most useful thing is to lock the best recipe and
directly attack the two aggregate drags** (easy-case degradation + prop-melt), both of which point
the same way: **less aggressive face injection**. So the 3 experiments complete a clean **λ × noise
2×2 around N6** (blended), biased toward the "gentle face, integrated body" corner that is the best
config-only bet to close the gap to step-0.

Secondary (not tonight): the prop-melt mask fix (exclude props from the face bbox) and the
Reading_pa_jensen auto-bbox fix (§4) — both inference/data-side.

## 4. The Reading_pa_jensen mask bug — FIXED

Root cause: with `automatic_bboxes=true` the gen face box is YOLO-detected per val; for this one
prompt+seed it placed a **spurious second face region in the top-right**, so the ref face is painted
into the trees there (main face is fine, hence a deceptively high id-sim ~0.71). Only this val panel
is affected; **training is unaffected** (it uses the cosmic dataset, not these prompts).

**Fix applied (07 Jul):** set `force_manual: true` on `pm96_bboxes_new.json['Reading pa_jensen.png']`
(its `face_crop_new=[273,93,499,393]` is the correct person-face box). The trainer honors this per
`sdxl_trainers.py:584-586` (`if force_manual: entry = manual_entry`) — so future runs (N10–N12+) use
the manual box for this one image only and skip the bad auto-detection. The lookup key
`f"{prompt[:10]}_{id}.png"` = `"Reading pa_jensen.png"` (with a space) matches the json key exactly,
so no key-mismatch issue. Backup at `pm96_bboxes_new.json.bak`. All scoring in §1 still excludes the
image (it was generated before the fix).

## 5. Next 3 experiments — λ × noise 2×2 around N6 (all blended)

Shared with N6 (unchanged): `noise_and_ref`, blended_masked, `lr_for_lora=1e-4`, clean ref (no
jitter), `wd 1e-3`, grad-clip 1.0, `warmup 200`, `ba_uncond_face_fix`, `id_only`, RealVis val,
`epoch_len=1000`, `n_epochs=3` → 3000 steps, val at 1000/2000/3000.

| run | λ_face | ba_noise_lr_scale | hypothesis |
|---|---|---|---|
| N6 (done) | 0.15 | 0.1 | anchor (existing corner) |
| **N7** | **0.05** | 0.1 | ↓face pressure → less melt, easy cases stay near step-0 |
| **N8** | 0.15 | **0.25** | ↑noise → more body integration for hard cases (N5=0<N6=.1, extend up) |
| **N9** | **0.05** | **0.25** | **best config-only bet to beat step-0**: gentle face (preserve easy) + strong body (integrate hard) |

Read-out: main effect of λ (N6 vs N7, N8 vs N9), main effect of noise (N6 vs N8, N7 vs N9),
interaction (N9). Benchmarks: step-0 **0.40** (target), N6 **0.297** (best trained so far), initial
**0.32**. Success = any checkpoint clearly > 0.32 with less melt (check Skiing/Kickboxing) and easy
cases held near step-0. If none beats N6 → local optimum mapped → the identity-loss objective change
is clearly justified.

Scripts: `serv_new_runs/start_ba_nr_blend_vast_N{7,8,9}.sh`; master
`serv_new_runs/run_overnight_N7_N8_N9.sh` (same self-stop + continue-on-fail design as before).

## 6. SUPERSEDED — the λ×noise 2×2 (N7/N8/N9) was NOT run

The user asked for **substantial** experiments instead of more hyperparameter tuning. The N7/N8/N9
scripts remain in `serv_new_runs/` but are shelved. See §7 for the substantial matrix that replaced
them.

## 7. Substantial experiments (N10/N11/N12) — three distinct axes, not hyperparameters

Rationale for going substantial: config sweeps (N4–N6) converge in 0.29–0.33, below step-0's 0.40.
The remaining high-value moves are structural. The single highest-value one — an **identity loss**
(the objective doesn't reward identity) — needs a differentiable face embedder (facenet/kornia are
NOT installed; insightface is ONNX) and careful testing, so it is **not** safe to fire unattended;
it's the recommended *flagship next step* to implement with verification. Tonight's three are
substantial changes that use **existing, wired machinery** (robust to run overnight), each attacking
a different diagnosed problem. All built on the N6 blended anchor; each changes ONE thing.

| run | axis | change vs N6 | hypothesis / targets |
|---|---|---|---|
| **N13** id loss | training objective | `+model.use_id_loss=true` (weight 0.1, gate t≤500) — auxiliary FaceNet cosine identity loss on the decoded face | directly reward identity (the MSE ceiling); the only lever likely to beat step-0's 0.40. **Runs first.** §7.1 |
| **N10** co-adapt | trainable capacity | `non_ba_train=false→true` (train base PhotoMaker LoRA + BA together; `train_ba_only` stays true) | lets body/lighting/hair **co-adapt** with the injected face → fix the easy-case degradation (trained face on un-co-adapted body). Risk: more surface → possible global drift (guarded by blended+hygiene+damped noise). |
| **N11** SA-only | which pathway trains | `train_branched_ca_lora=true→false` (freeze the branched cross-attn; train only branched self-attn) | direct test of §4.2 "CA is the drift/melt pathway": does the SA branch alone give cleaner identity / less prop-melt? (N5 froze noise on attn1+attn2; N11 freezes CA on ref+noise — the complementary cut) |
| **N12** id_embeds | identity conditioning | `face_embed_strategy id→id_embeds` (+`use_id_embeds=true`) — face-branch CA attends to PhotoMaker **ID features** from the reference, not the generic prompt | inject a real identity signal into the face branch. Least-exercised path → **runs last**; fast-fails at startup if the batch lacks id features (master continues). |

Not tested (out): `num_refs>1` — the training forward uses `refs[0]` only (`lora2_helpers.py:196`),
so multi-ref is a no-op without code.

### 7.1 Identity loss — IMPLEMENTED & VERIFIED (now experiment N13, runs FIRST)

The flagship objective change is built and tested. `src/loss/id_loss.py` (`IdentityLoss`) decodes the
predicted x0 → crops the face at the gen bbox → embeds with a **frozen FaceNet** (facenet-pytorch
InceptionResnetV1 / VGGFace2, 512-d, differentiable) → cosine distance to the FaceNet embedding of
the **ground-truth face** (from `pixel_values`, same bbox). Fully differentiable, so it trains the
BA weights toward matching identity — the thing MSE never rewards.

- **Wiring:** `lora2.py` `__init__` gains `use_id_loss` (default **false** → zero overhead/behaviour
  change), `id_loss_weight`, `id_loss_max_timestep`, `id_loss_face_size`; `forward` computes it on
  gated steps and returns `id_loss`; `sdxl_trainers.py:process_batch` adds `weight * id_loss` to the
  total before backward and logs `train/id_loss`. Easy on/off via `+model.use_id_loss=true`.
- **Timestep gate:** the whole batch shares one timestep/step, so `id_loss_max_timestep=500` means
  high-noise steps skip the VAE decode entirely (only ~low-noise steps pay the cost).
- **Verified:** (a) FaceNet discriminates identity (cos(same)=1.00, cos(jensen,keanu)=0.50) and the
  loss is 0 for matching / 0.50 for mismatched identity, degenerate bbox → 0, differentiable;
  (b) a real local training smoke run (3 steps, cosmic data, bs=1, 16 GB) logged finite
  `train/id_loss` (0.38, 0.24), finite grads, no OOM/NaN.
- **Weight calibration:** the smoke showed weight 0.5 makes the ID term ~4× the diffusion MSE (too
  dominant). N13 uses **0.1** (a meaningful but non-dominant nudge). This is the key knob — raise if
  id-sim barely moves, lower if the base image degrades.

**Batch order (updated):** N13 (id loss) runs FIRST, then N10 → N11 → N12.
Scripts: `serv_new_runs/start_ba_{idloss_N13, coadapt_N10, saonly_N11, idembeds_N12}_vast.sh`;
master `serv_new_runs/run_overnight_N13_N10_N11_N12.sh` (self-stop 3000 steps each; continue-on-fail;
~12.5–14.5 h total — comment out the tail if the night is short). N13 needs `facenet-pytorch` on the
box (`pip install facenet-pytorch`; ~107 MB weights download once). Benchmarks: step-0 **0.40**
(target), N6 **0.297** (best trained). For N13 also watch `train/id_loss` trending DOWN.

## 8. Results (fill after the run)

| run | 0 | 1000 | 2000 | 3000 | best | melt (Ski/Kick)? | easy-case vs step0 | notes |
|---|---|---|---|---|---|---|---|---|
| N10 co-adapt | | | | | | | | |
| N11 SA-only | | | | | | | | |
| N12 id_embeds | | | | | | | | |
