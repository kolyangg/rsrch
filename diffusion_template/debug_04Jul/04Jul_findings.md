# 04 Jul findings — Branched Attention over PhotoMaker

> **⚠️ 06 Jul UPDATE — read §9 first.** The N3a run (masked_alternating + hygiene + noise damper,
> 8k steps) landed. Decisive result: **the untrained branched path (step 0) is the best face for
> BOTH runs (~0.40 id-sim); ANY training degrades it** and N3a is a regression vs even the initial
> run (0.21 vs 0.32 at 8k). This reframes the problem — the MSE objective doesn't reward identity —
> and changes the recommendation below. Details + next config in §9.

Scope constraints (from the user, binding on any proposal):
- **Keep the BA high-level logic identical** to the initial model (suspect *implementation*, not
  architecture).
- **Validation model stays RealVisXL_V4.0** (same as the original).
- **Train BOTH ref + noise** (`noise_and_ref`). `ref_only` is off the table — it broke face↔body
  consistency.

---

## 0. The two models (ground truth from saved `config.yaml` + checkpoint probes)

| | **Initial** (`saved/03Jul_start_ba_cosm_new1_vast`) | **New** (`saved/ba_refonly1`) |
|---|---|---|
| weight mode | **noise_and_ref** (840 tensors) | ref_only (420) |
| loss | `masked_alternating`, λ_face 0.1 | `blended_masked`, λ_face 0.2 |
| lr / clip / wd | 1e-4 / none / 0 | 5e-5 / 1.0 / 1e-2 |
| val base | RealVisXL_V4.0 | (trained-base SDXL — later corrected to RealVis for eval) |
| eval'd weights | epoch-14 (step 28k) | epoch-10 (step 20k) |
| checkpoint probe (top Frobenius) | **`attn2 noise_to_v` 1.72 / max 3.71**, `attn1 noise_v` 1.31, `ref_k` 1.39, `ref_v` 1.47 | `attn2 ref_to_k/v` ~0.85, ref groups only, **no noise groups** |

Probe takeaway: the initial model trains the **noise/gen cross-attention hardest** (`noise_to_v` on
attn2 is the single largest delta) — the whole-image-warping channel. `ref_only` removed it entirely.

---

## 1. Result analysis (both models on the same RealVisXL base)

Artifacts: `outputs/ba_debug/N1_realvis_final/{_contact_*,_compare_*}`, `compare_old20k_vs_new_e10.pdf`
(OLD 20k vs NEW e10), `N1_realvis_retrack/compare_old_vs_new_vs_retrack.pdf` (adds retrack column).

**1a. New (ref_only, corrected inference) — the adjusted panel.**
- Base drift is gone (it was the RealVis→SDXL val-base swap, not BA). Base now photographic & matches
  the original. ✅ constraint #1 satisfied at the base level.
- Severe artifacts gone: no duplicate people / melted goggles / ghosts (those were
  `ba_face_prompt_mode=full_boosted` ×2.5; `id_only` fixes them).
- **jensen (short hair): ~11–12/12 clean, recognizable.** Strong.
- **keanu (long hair): ~7/12; motion poses (Dancing/Jumping/Skiing/Crying) smear** — a curtain of
  ref hair over blurred features.

**1b. Old vs new, matched step 20000 (both RealVis, id_only), same prompts/refs/seed.**
- **jensen: tie** — both photographic, recognizable, clean.
- **keanu: OLD (noise_and_ref) wins on motion poses** — the face stays visible & integrated with the
  body; NEW (ref_only) smears it.

**1c. This directly confirms the user's claim.** `ref_only` injects identity into the face region
but leaves the **gen/noise pathway at base weights**, so the model never learns to *blend* the
injected face with the surrounding body/hair/lighting. On easy cases (static, short hair) it's fine;
on hard cases (motion, long hair) the face **does not integrate** → smear / face-body inconsistency.
`noise_and_ref` trains the gen pathway too, so it learns to blend — better face↔body consistency, at
the cost of the drift risk. **⇒ the fix must keep noise_and_ref and tame the drift, not delete the
noise pathway.**

**1d. C6 gen-bbox retrack does NOT fix the smear** (ran it, `N1_realvis_retrack`). Mechanism works
(re-detects the face on the branched x0, updates the mask) but the box was never mislocated — steps
0–15 are non-branched PhotoMaker so the face location is fixed before branching. The smear is
face-branch **content** (ref hair painted over the face inside a correctly-placed box), not a bbox
error. `mask_expansion_ratio=1.0` = no growth, so it's not an over-large mask either. Retrack stays
as an off-by-default toggle; it is not the lever here.

---

## 2. Initial model's issues — earlier findings, reframed to KEEP ref+noise

From the T0–T7 A/B matrix + probes (`ba_training_fix_plan_v2.md`, worklog):
- **E1 drift** (T4 + probes): the trained gen/noise CA (`noise_to_v`) warps global generation; the
  face-masked loss lets the optimizer cut face MSE by distorting the whole image. Largest, roughly
  linearly-growing deltas.
- **Double-face** (T7): the **uncond** face prompt under CFG amplifies a second face → `ba_uncond_face_fix`
  (F1, plain negative embeds for the uncond face branch) removes it.
- **Core drift without CFG** (T1, gs=1): present even at gs=1 → a **training-side** issue, not only CFG.
- **Structure is sound** (T0b): untrained clones generate cleanly through the full branched path →
  the BA mechanism itself is fine (consistent with constraint #3).

**Reframe:** the earlier plan "fixed" E1 by switching to `ref_only` — but that violates constraint #2
and breaks face↔body consistency (§1c). The correct target is to **keep both pathways trainable but
prevent the noise pathway from over-training into whole-image drift.**

---

## 3. Additional findings this session
- **Validation base swap** (`pretrained_model_for_validation=null → SDXL-base`) was the dominant cause
  of the "animation, not a photo" look — **not** BA. Pin RealVis (constraint #1). ✅
- **`full_boosted` (×2.5 id) manufactures ghosts/duplicates**; `id_only` is clean. Inference + training
  face-injection should be `id_only`.
- **infer.py two-pass generator reuse** made the preview (bbox) and branched passes start from
  different latents → misalignment; fixed with a re-seed (training path already did this).
- **Validation crash** `Invalid branched batch: total=8, gen=2, ref=6` — stale
  `_runtime_uses_branched_unet` + stale batch-2 mask during a non-branched val step; fixed by deciding
  the processor swap from the *actual* attached processors + resetting the flag each eval.

---

## 4. Mechanism (read from `attn_processor_cleanest.py` forward) + implementation issues

**Confirmed forward (self-attn branched processor), doubled batch `[noise, ref]`:**
- **Background/body branch:** `q_bg = q_noise·(1−mask)`, `K/V = noise_to_k/v(noise_hidden)` → `hidden_bg`.
  So **`noise_to_*` renders the non-face (body + background)** region.
- **Face branch:** `q_face = q_noise·mask`, `K/V = ref_to_k/v(face_hidden_mixed)` → `hidden_face`.
  So **`ref_to_*` renders the face** region, from the reference.
- **Merge:** `merged = hidden_bg·(1−mask) + hidden_face·mask·scale` (`scale=1.0`). Output `[merged, hidden_ref]`.

**⇒ This mechanistically proves the user's claim.**
- `ref_only`: the face branch (`ref_to_*`, trained) is merged onto a body/background rendered by the
  **base** `to_k/v` (untrained). The trained face sits on an **un-co-adapted body** → face↔body
  inconsistency (worst on hard poses = the keanu smear).
- `noise_and_ref`: body (`noise_to_*`) **and** face (`ref_to_*`) train together → co-adapted → consistent
  body. Cost: `noise_to_v` (the body branch) can over-train → the whole body/background warps = the
  **E1 drift**. Probe agrees: `attn2 noise_to_v` is the single largest delta in the initial model.
- So: **both pathways are needed for consistency; the fix is to stop the body branch (`noise_to_v`)
  from over-training, not to delete it.**

**Intentional design choices (NOT bugs — confirmed by the user, do not change):**
- `POSE_ADAPT_RATIO = 0.0` (`attn_processor_cleanest.py:308`) — the face branch injects the **pure
  reference face** (no pose blending from the current generation). This is deliberate: maximise
  identity fidelity. Leave as-is.
- `CA_MIXING_FOR_FACE = False` (`:309`) — deliberate. Leave as-is.

**Concrete levers that keep BA logic (constraint #3):**
1. **Face-merge `scale`** (`:390`, default 1.0) — face-injection strength at merge time; a clean knob
   if the face over-/under-injects.
2. **Gradient routing — VERIFIED (04 Jul session 2), see §4.2.**
3. **Per-group LR:** `ref_to_*` and `noise_to_*` are **separate cloned Linear params**, so a
   **differential LR / weight-decay** between them is straightforward — the most direct lever against
   `noise_to_v` runaway while keeping both pathways trainable (constraint #2).

### 4.2 Gradient-routing inspection (VERIFIED — this is the drift engine)

Question was: does the face-masked loss push `noise_to_*` to paint the face? **Answer: yes — via
the cross-attention layers, at full strength, by construction.**

- **attn2 (cross-attn) has NO mask and NO merge for the noise half**
  (`BranchedCrossAttnProcessor.__call__`, `attn_processor_cleanest.py:670–713`): the **entire**
  generated image — face region included — is rendered as
  `SDPA(noise_to_q(noise_hidden), noise_to_k/v(gen_prompt))`. The `ref_to_*` CA weights only serve
  the reference half (`q_ref` × `face_prompt`). "Branching" in CA = separate weights + separate
  prompts per batch half, not a spatial face/bg split. (This is the intended design —
  `CA_MIXING_FOR_FACE=False` is deliberate, constraint #4 — but its *gradient* consequence was
  unexamined.)
- **Therefore, on `masked_alternating` face-only steps** (initial run: `masked_loss_step=2`, i.e.
  **every 2nd step's loss is ONLY the face bbox crop**, `diffusion_loss.py:_masked_face_mse`), the
  gradients flow into `noise_to_q/k/v` (CA) **with zero background anchor** — nothing in that
  step's loss penalizes warping the rest of the image. At lr 1e-4, wd 0, no grad-clip, this is a
  textbook runaway. Empirical confirmation on both ends: the checkpoint probe (§0: `attn2
  noise_to_v` = single largest delta) and the canary docstring's note from the failure run
  ("doubling per 2k steps, worst in **ca_noise**", `sdxl_trainers.py:243`).
- **attn1 (self-attn) is clean per-layer:** the merge gate is strictly binary
  (`force_binary_masks=True`, `:138,444–445`), so SA `noise_to_k/v` get no face-region gradient
  through the merge. Two known, by-design exceptions: `noise_to_q` is shared by both branches
  (`q_face = noise_to_q(·)·mask`, `:250,354`) so the face loss trains it; and cross-layer coupling
  (later layers see merged states) mixes gradients anyway — unavoidable, not a bug.
- **Ref pathway:** the loss supervises only the noise-half prediction (`run_branched_forward_pass`
  returns the merged noise pred; ref reconstruction unsupervised), so `ref_to_*` train exclusively
  through the face-injection path. As intended.
- **Optimizer:** single param group — every trainable tensor at `lr_for_lora`
  (`lora2.py:get_trainable_params`, `:264–269`). A commented-out draft of custom grouping already
  sits at `lora2.py:237–260` — the natural, minimal insertion point for per-group ref/noise LR+WD.
- **Drift canary already exists**: `ba_norm/{sa,ca}×{ref,noise}` logged every `log_step`
  (`sdxl_trainers.py:_update_ba_weight_norms`). No code needed — just watch `ca_noise`.
- **Drift timing (probe of e1/e2/e14, this session):** mean Frobenius of `attn2 noise_to_v`:
  **0.84 @ e1 (2k) → 1.22 @ e2 (4k) → 1.72 @ e14 (28k)** — i.e. ~70% of the final delta is in place
  by epoch 2, then growth decelerates (Adam random-walk regime). Two consequences: (a) the drift
  engine acts **from step 0** — hygiene must be on from the start; early-stopping an unhygienic run
  does not help; (b) a **2-epoch probe run is sufficient** to tell whether the engine is off
  (canary flat vs. the historical near-doubling per 2k steps).

**Consequence for the fix:** the drift is not a maskable bug in the BA forward pass — it is the
**interaction of (a) unshielded noise-CA ownership of the face region (intentional architecture)
with (b) face-only loss steps (masked_alternating) and (c) an unconstrained optimizer.** Since (a)
is off-limits (constraints #3/#4), the fix targets (b)+(c): a blended loss that keeps a full-image
anchor in every step, plus per-group damping of the noise weights. Both preserve the BA forward
logic exactly.

---

## 5. Proposed next setup (DRAFT — FOR USER APPROVAL, no code changed yet)

**Keep:** BA architecture, `noise_and_ref`, RealVisXL validation, doubled-batch + mask-gated merge.

**Change (drift hygiene + known-good conditioning, all preserving both pathways):**
| Item | Initial | Proposed | Rationale |
|---|---|---|---|
| `ba_uncond_face_fix` | false | **true** | removes the CFG double-face (T7) |
| `ba_face_prompt_mode` | (id_only — predates the switch) | **id_only** (explicit, train AND infer) | ghost-free at inference; matches the initial model's effective training conditioning (constraint #3). Note: refonly1 trained with `full_boosted` but was *evaluated* with `id_only` — a train/infer mismatch we drop |
| ref-crop jitter (`ref_crop_margin 0.2–0.6`, `ref_downscale_jitter 0.5`) | off (legacy 0.2 fixed) | **on** (as in refonly1) | ref-domain generalization; dataset-side only, BA logic untouched |
| `lr_for_lora` | 1e-4 | **5e-5** | slow the runaway |
| grad clip / weight-decay | none / 0 | **1.0 / 1e-2** | drift hygiene |
| loss | masked_alternating | **blended_masked**, λ_face **0.1→0.15** | **primary fix** (§4.2): alternating face-only steps are the drift engine — they train the noise CA group on the face with zero background anchor. Blended keeps a `(1−λ)` full-image anchor in **every** step |
| **differential LR on noise groups** | — (same LR) | **noise groups at ~0.2–0.3× the ref-group LR** (or higher wd) | secondary damper: the blended face term still pushes noise-CA every step (§4.2 — CA face region is noise-owned by design); this bounds the push |
| drift canary `ba_norm/{sa,ca}×{ref,noise}` | — | **watch `ca_noise`** (already implemented, logs every `log_step` — §4.2) | catch noise-group runaway live; enables early stop |
| val base | RealVis | **RealVis** (unchanged) | constraint #1 |

Note: `POSE_ADAPT_RATIO` / `CA_MIXING_FOR_FACE` stay as-is (intentional, §4).

**One logic-preserving code change (the "implementation" lever), to do AFTER approval:**
- add per-group (ref vs noise) LR / weight-decay in the optimizer setup — keeps the BA architecture
  identical. Insertion point already exists: the commented-out grouping draft at `lora2.py:237–260`.
- (The gradient-routing sanity check is DONE — §4.2. It found no maskable bug: the face→noise-CA
  gradient path is a consequence of the intentional CA design, so the loss/optimizer levers above
  are the correct response. Everything else in the proposal is config-only.)

**Validation of the proposal:**
- Local A/B — DONE (§6). Verdict: the initial model's face damage is **trained-in** (corrected
  inference does not rescue e14), and the two weight modes fail in complementary pathway-specific
  ways → confirms both halves of this proposal (keep noise_and_ref for integration; add the
  loss/optimizer hygiene to kill the drift-driven color-warp/melt).
- Next: a short vast run (**2 epochs is decisive** — §4.2 drift-timing: the old run had ~70% of
  its drift by e2) with the proposed config; success = `ba_norm/ca_noise` flat/sublinear (vs
  historical doubling per 2k), no color-cast/seam on the face patch, keanu motion poses stay
  integrated (OLD's win preserved), id-sim ≥ ref_only e10 (jensen 0.416 / keanu 0.287 to beat).

---

## 6. Local A/B evidence (this session) — COMPLETE

**A/B-1 — initial model (noise_and_ref e14) under the exact corrected inference of the new model**
(`ba_old_noiseref_realvis.yaml` → `outputs/ba_debug/OLD_noiseref_realvis`, 24 panels; config diff
vs `ba_n1_realvis.yaml` confirms only weights/weight-mode differ — same base, prompts, seed 0,
shared gen-bbox cache, id_only, F1). This supersedes the earlier §1b head-to-head, whose OLD side
came from training-time live-val panels rather than the same inference path.

Artifacts: `OLD_noiseref_realvis/{compare_oldE14_vs_newE10.pdf, _contact_*.png, idsim_report.txt}`,
`N1_realvis_final/idsim_report.txt`.

**id-sim (InsightFace cosine vs ref, 12 prompts × 2 ids, excluding sheets):**

| | OLD noise_and_ref e14 | NEW ref_only e10 |
|---|---|---|
| jensen mean | 0.341 | **0.416** |
| keanu mean | 0.237 | **0.287** |
| pairwise wins | 8/24 | 16/24 |
| catastrophic (<0.10) | **4** (Rushing-k −0.01, Skiing-k −0.03, Skiing-j 0.06, Chef-j 0.14*) | 1 (Chef-j 0.09*) |

*Chef is bad for both (toque + second-person context).

**The two models fail in complementary, pathway-specific ways:**
- **OLD (noise_and_ref): perfect integration, warped content.** All 12 keanu poses are
  pose-integrated — zero hair-curtain smears (the ref_only failure mode is entirely absent). It
  wins exactly the motion-smear set: Crying-k 0.270 vs 0.145, Dancing-k 0.324 vs 0.146, Jumping-k
  0.237 vs 0.180, Drumming-k 0.392 vs 0.277. **But** the face patch carries a systematic
  orange/waxy color-cast with a visible merge seam (~8/12 keanu panels), and prompts that put
  props in the face box collide catastrophically: Skiing = goggles melted INTO the face,
  Kickboxing = sweatband melted into a translucent face-shield, Rushing = strong orange face vs
  pale neck (id-sim −0.01 despite a recognizable, integrated face). This is **trained-in** damage —
  corrected inference does not rescue it (consistent with T1 gs=1) — and it is precisely the §4.2
  drifted-noise-CA signature rendering the face region.
- **NEW (ref_only): clean content, broken integration.** No color casts, no melts (never below
  0.10 except Chef) — the untouched noise pathway renders base-faithful color/texture. But without
  co-adaptation the injected face doesn't follow the body on hard poses → the keanu motion smears
  (its 4 worst scores = exactly the smear set, §1a).

**Conclusion (drives §5):** neither pathway choice is acceptable alone. The next model must keep
**noise_and_ref for integration** (constraint #2, now backed by matched-inference numbers) while
**preventing the noise-CA drift that produces the color-warp/melt** — which §4.2 shows is driven
by the alternating face-only loss + unconstrained optimizer, both fixable without touching BA
logic. Expected outcome of §5: OLD's integration wins + NEW's content cleanliness on the same
panel; the drift canary (`ba_norm/ca_noise` flat vs doubling by epoch 2) gives the early verdict.

---

## 8. APPROVED execution plan (05 Jul) — user decision + what was implemented

**User decision:** plan approved, but the loss function stays an open A/B — original PhotoMaker
LoRA uses `masked_alternating`, so test that first with all other fixes, then `blended_masked` on
top as a separate run. Two vast scripts prepared (both `noise_and_ref`, differing ONLY in loss):

- `serv_new_runs/start_ba_nr_alt_vast_N3a.sh` — `masked_alternating` (original loss) + full
  hygiene package. **Run first.**
- `serv_new_runs/start_ba_nr_blend_vast_N3b.sh` — `blended_masked lambda_face=0.15` on top of the
  identical package. The pair isolates the loss's contribution (§4.2 predicts N3b drifts less).

Shared package (vs the initial cosm_new1_vast run): lr 5e-5, grad-clip 1.0, wd 1e-2,
`+ba_noise_lr_scale=0.25` (new per-group damper), `ba_uncond_face_fix=true`,
`ba_face_prompt_mode=id_only` (train+infer consistent), ref-crop jitter 0.2–0.6/0.5, val base
pinned to RealVisXL, canary `ba_norm/*` already logging. From scratch (not resumed).

**Code change implemented (the one approved):** `lora2.py:get_trainable_params` now supports
per-branch optimizer grouping — top-level config keys `ba_noise_lr_scale` (default 1.0) and
`ba_noise_weight_decay` (optional). `noise_to_*` processor clones go into a `ba_noise_params`
group at `lr_for_lora × scale`; defaults reproduce the old single-group behaviour bit-identically
(also when a ref_only checkpoint has no noise clones). Verified: 4-case unit test of the grouping
+ AdamW instantiation, `bash -n` on both scripts, and full hydra compose of both override lists.

**Schedule / gates** (cadence from the initial run's `info.log`: real start 12:38:14 → epoch-1
done 13:28:14 = **~50 min per 2k-step epoch incl. validation, on the vast A100-class GPU**. NB:
the checkpoint *file mtimes* are download timestamps — `Zone.Identifier` present — not train times.
On a ~half-throughput 45 GB card expect ~2× this, i.e. ~100 min/epoch — hardware, not config; see
the memory note below and the session Q&A.):
1. **Canary gate @ 4k steps (2 epochs, ~1.5–2 h):** Comet `ba_norm/ca_noise` flat/sublinear vs
   the initial run's near-doubling per 2k; step-4k panels free of orange cast/seams. Fail → stop,
   lower `ba_noise_lr_scale` (0.1) or raise `ba_noise_weight_decay`, restart.
2. **Main result @ 20k steps (10 epochs, ~8–9 h):** equal-step three-way comparison vs
   initial@20k and refonly1@e10(=20k). Targets: id-sim ≥ 0.416 (jensen) / 0.287 (keanu; the
   ref_only numbers), keanu motion poses integrated (no smears), no color-warp/melts.
3. Optional extension to 28k (=e14 evaluation point) only if the canary stays flat.

**GPU memory (45 GB card, `dataloaders.train.batch_size=2`):** fits with ~10 GB headroom.
Static ≈ 11.5 GB (UNet bf16 5.1 + branch clones ≈1.50 B params/3.0 + text encoders 1.6 + VAE 0.2 +
id-encoder ~0.5 + LoRA 0.15 + AdamW moments 0.6 + grads 0.2); training activations at effective
batch 4 through the UNet (bs 2 doubled [noise, ref]) at 128×128 latents with SDPA ≈ 16–20 GB;
CUDA context ≈ 2 GB → **peak ≈ 30–34 GB**. The same architecture/batch already trained on the
vast A100. Val-time transient (fresh RealVis pipe ~7 GB) lands while train activations are freed —
below the training peak. If a specific card still OOMs: `dataloaders.train.batch_size=1` +
grad-accum 2 (pattern in `start_ba_cosm_new1_vast.sh`), and/or
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 7. Open questions — ALL RESOLVED (05 Jul, see §8)
1. ~~Differential LR between ref and noise groups?~~ **Approved & implemented**
   (`ba_noise_lr_scale=0.25` in both N3 scripts; `ba_noise_weight_decay` available as reserve).
2. ~~§4 code-inspection first?~~ **Done (§4.2)** — no maskable bug; the drift path is structural
   (intentional CA design × alternating face-only loss × unconstrained optimizer), so the
   loss/optimizer levers are the right response.
3. ~~Scratch vs resume?~~ **From scratch** (both N3 scripts) — e14's noise groups are deep in
   drift territory; resuming would anchor on a warped solution and muddy the canary read-out.
4. Loss function stays a two-run A/B per the user: `masked_alternating` (original, N3a) first,
   then `blended_masked` (N3b) — §8.

---

## 9. N3a RESULT (06 Jul) — training degrades identity; step-0 is best. STOP N3a.

Ran `start_ba_nr_alt_vast_N3a.sh` (masked_alternating + hygiene + `ba_noise_lr_scale=0.25`) to
**8k steps / epoch 4** on a 45 GB card. Saved: `saved/ba_nr_alt_N3a/{config.yaml, info.log,
val_images/step_{0,2000,4000,6000,8000}, weights/checkpoint-epoch4}`. Optimizer split confirmed
live in `info.log`: `lora_params` 840 @ 5e-5 / `ba_noise_params` 840 @ 1.25e-5, wd 0.01 both.

**Per-step mean id-sim (InsightFace cosine vs ref, 24 panels; sheets:
`debug_04Jul/n3a_progression_{jensen,keanu}.png`):**

| step | N3a | initial cosm_new1 |
|---|---|---|
| **0** (untrained clones) | **0.412** | **0.402** |
| 2000 | 0.188 | 0.193 |
| 4000 | 0.197 | 0.296 |
| 6000 | 0.228 | 0.301 |
| 8000 | 0.211 | 0.321 |
| 20000 / 28000 | — | 0.310 / 0.317 |

**Three hard conclusions:**
1. **Step 0 is the best face — for BOTH runs (~0.40).** At step 0 the branch clones equal base
   weights (LoRA `lora_B`=0), so the face branch is just frozen-PhotoMaker identity through the
   branch plumbing. It works. This is the old **T0b** observation, now quantified at panel scale.
2. **Training degrades identity and does not recover.** Both runs crash to ~0.19 by 2k. The
   initial run claws back to ~0.30–0.32 and **plateaus there through 28k — never re-reaching the
   0.40 step-0 baseline.** N3a plateaus even lower (~0.21) and flat 2k→8k. So **more steps will
   not help**: the curve is crash-then-flat, and the ceiling (initial @28k = 0.32) is below the
   untrained floor (0.40). → **Stop N3a.**
3. **N3a is a regression vs the initial recipe** (0.21 vs 0.32 at 8k) — the user's eyeball was
   right. The hygiene package made identity *worse*, not better. Likely causes (see below).

**Why training hurts (root-cause reframe):** the objective is denoising **MSE** (masked to the
face on alternating steps) — it rewards reconstructing the *training image's* noise, **not**
similarity to the reference identity. The frozen PhotoMaker path already injects identity well at
step 0, so gradient descent mostly moves the branch weights *away* from that good solution toward
a dataset-average face, while the residual noise-CA drift (§4.2) adds the orange/melt cast
(visible progressively on Skiing/Kickboxing/Crying → several go to "no-face" = face corrupted
past detection). The damper (0.25×) slowed the drift but the net effect is still degradation.

**Why N3a < initial specifically** (two fixable regressions, both introduced in N3a):
- **Ref-crop jitter** (`ref_crop_margin 0.2–0.6`, `ref_downscale_jitter 0.5`) feeds the face branch
  *blurrier, variable* reference crops → a weaker identity signal than the initial run's clean
  fixed crop. Prime suspect. **Revert it.**
- **Over-low LR** (5e-5, noise 1.25e-5) stalled recovery in a worse basin: initial recovered by 4k
  at 1e-4; N3a hadn't by 8k. Lower LR ≠ better here.

### 9.1 Recommended next run — N4 (keep alternating loss, noise_and_ref, RealVis val)

Primary goal is now **diagnostic**: (a) find whether *any* checkpoint beats the step-0 baseline
(0.40), and (b) test whether hard-damping the noise pathway lets the ref pathway climb without the
orange/melt damage. Changes:

| knob | initial | N3a | **N4** | why |
|---|---|---|---|---|
| ref-crop jitter | off | on | **off** | restore clean reference identity signal (§9) |
| `lr_for_lora` | 1e-4 | 5e-5 | **1e-4** | un-stall; initial recovered at 1e-4 |
| `ba_noise_lr_scale` | 1.0 | 0.25 | **0.1** | hard-damp the drift/damage pathway (the orange/melt vector) while ref learns |
| `optimizer.weight_decay` | 0 | 1e-2 | **1e-3** | light pull toward base (=good step-0), not so strong it adds noise |
| `trainer.max_grad_norm` | none | 1.0 | **1.0** | keep (cheap) |
| `ba_uncond_face_fix` / `ba_face_prompt_mode` | off / id_only | on / id_only | **on / id_only** | keep known-good |
| **`trainer.epoch_len`** (val cadence) | 2000 | 2000 | **500** | **the key diagnostic**: the crash happens inside 0–2000 with zero visibility. Val at 500/1000/1500/2000 to locate the peak — the best trained checkpoint may be at a few hundred steps |
| run length | 28k | 8k | **~3000 steps** | crash+plateau is fully visible by 3k; no need for more |

Decision rule after N4:
- If N4's best (likely early) checkpoint **> 0.40** → real improvement; extend + lock config.
- If N4 still **< 0.40** at every step → training-with-MSE cannot beat untrained. Two escalations:
  (i) **near-freeze** (`ba_noise_lr_scale≈0`, `lr_for_lora`≤2e-5, ~500 steps) to confirm the
  minimal-training product, and (ii) the principled fix — **add an identity loss** (InsightFace/
  ArcFace cosine between the generated face crop and the reference) so the objective actually
  rewards identity; this is a real (but contained) code change and would need approval.

**Tension to flag for the user:** constraint #2 says train both pathways because `ref_only` broke
face↔body consistency — yet noise_and_ref training is what warps the face. N4's `ba_noise_lr_scale=0.1`
is the compromise: both pathways stay *trainable* (constraint respected) but the noise/drift
pathway barely moves from base. If N4 shows the noise pathway must move even less, we should
revisit constraint #2 explicitly with the user rather than silently violate it.
