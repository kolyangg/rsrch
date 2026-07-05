# Isolating N1's face improvement from base drift (04 Jul 2026)

Goal (user): the NEW run's validation images "drifted" — they no longer match the base
PhotoMaker generation (look like animations, messy), so it's impossible to tell whether the
face improved. Re-run the NEW setup so the **base is the same** as the old setup and only the
**face area** differs, then judge whether the face/weights improved.

## 0. Which folder is which (the message labels were swapped)

Confirmed from the saved `config.yaml` (ground truth), not folder names:

| Folder | weights | `branched_attn_weight_mode` | val base | = |
|---|---|---|---|---|
| `saved/ba_refonly1` | 165 MB | **ref_only**, blended_masked, lr 5e-5, `ba_face_prompt_mode=full_boosted`, `ba_uncond_face_fix=true` | **null → SDXL-base** | the **NEW** run (N1 I proposed) |
| `saved/03Jul_start_ba_cosm_new1_vast` | 237 MB | **noise_and_ref**, masked_alternating, lr 1e-4 | **RealVisXL_V4.0** | the **OLD** original run |

So "ba_refonly1" (which the user called the *initial* run) is actually the **N1 ref_only** run,
and "03Jul_start_ba_cosm_new1_vast" (called the *new* run) is the **old noise_and_ref** run.

## 1. Why the NEW run's base "drifted" — it's the validation base model, not the BA

The two runs validate on **different base models**:
- OLD run: `pretrained_model_for_validation_name_or_path = SG161222/RealVisXL_V4.0` (photorealism
  fine-tune of SDXL).
- NEW/N1 run: `pretrained_model_for_validation_name_or_path = null` → validates on the **training
  base = SDXL-base-1.0**. (I set this deliberately in N1 to remove a train/val base mismatch.)

With different base weights, even the same seed diverges into a completely different image —
SDXL-base is softer / more "illustrative", RealVis is crisp/photographic. Direct evidence, same
prompt/ref/seed (`step_20000_batch_2/Night-ride_jensen.png`):
- OLD (noise_and_ref, RealVis): crisp night-street photo, natural face.
- NEW (ref_only, SDXL-base): hazier, softer, slightly "AI" look, different composition.

They are entirely different images (different pose/framing), **not** the same base with a
different face. So the perceived "drift" is dominated by the **RealVis→SDXL-base validation
swap**, not by the branched attention. To compare fairly, run the N1 (ref_only) weights on
**RealVisXL** — then the base matches the old run and only the face branch differs.

## 2. Weights analysis (request #2) — did N1 learn, and is it structurally safer?

`scripts/inspect_ba_checkpoint.py` (Frobenius norm of each processor's branch LoRA delta):

**N1 `ba_refonly1/weights-epoch10.pth` — 420 tensors, ONLY ref groups:**
```
('attn1','ref','k') 0.677   ('attn1','ref','q') 0.616   ('attn1','ref','v') 0.814
('attn2','ref','k') 0.486   ('attn2','ref','q') 0.421   ('attn2','ref','v') 0.847
```
Top sites: `ref_to_k/v` on attn2 (cross-attention). → the reference/face branch genuinely
trained; **no noise/gen groups exist at all**.

**OLD `03Jul.../weights-epoch14.pth` — 840 tensors, noise + ref:**
```
('attn2','noise','v') 1.72 / max 3.71   <-- largest: the whole-image text (gen) pathway
('attn1','noise','v') 1.31   ('attn1','ref','k') 1.39   ('attn2','ref','v') 1.47
```
Top sites: `noise_to_v` on attn2/mid-block. → the OLD run trained the **gen/noise cross-attention
hardest** — exactly the channel that warps global generation (the drift mechanism diagnosed in
the T0–T7 matrix).

**Takeaway:** N1 is structurally safer *by construction* — it has no trainable gen/noise
pathway, so whole-image drift is impossible; and it still learned real ref/face features. Whether
that yields a visibly better face is what the RealVis isolation run (below) tests.

## 3. Isolation run (request #1) — N1 ref_only weights on RealVisXL

Config: `src/configs/inference/ba_n1_realvis.yaml` (RealVis trunk, ref_only, `full_boosted`,
`ba_uncond_face_fix`, guidance 5, branched from step 15, `references_two`, `enable_vae_tiling`
for the 16 GB laptop GPU). Checkpoint: `ba_refonly1/weights-epoch10.pth`.

Two passes on identical seeds/prompts/refs (4 prompts × jensen/keanu):
- **ON**  = branched attention on  → `outputs/ba_debug/N1_realvis_on`  (N1 face branch active)
- **OFF** = branched attention off → `outputs/ba_debug/N1_realvis_off` (PhotoMaker base only)

Because ref_only leaves the gen pathway at base weights and branching only starts at step 15,
ON and OFF share the same base; ON/OFF diff isolates the face-branch effect on a photographic
RealVis base.

### Results (8 ON + 8 OFF, jensen/keanu × Reading/Skiing/Night-ride/Chef, seed 0)

**(a) Base drift is fixed — it was the base model, confirmed.** The branched-OFF pass (pure
PhotoMaker on RealVis) is crisp/photographic and matches the OLD run's composition & quality.
So the "animation, not a photo" look was **entirely** the RealVis→SDXL-base validation swap
(`pretrained_model_for_validation_name_or_path=null`), not the branched attention or the weights.

**(b) ref_only preserves the base — confirmed.** ON vs OFF share the same base (outfit, setting,
pose); only local changes near the face. The gen pathway stays at base weights, as designed.

**(c) The face INJECTION is broken — this is the real remaining problem.** With branched ON the
face region is corrupted:
- `Reading_jensen`: melted-face blob **+ a duplicated standing person**.
- `Skiing_jensen`: stacked/melted goggles + orange face-paint blobs (the classic artifact).
- `Reading_keanu`: the main face is a **correct** Keanu, but a ghost/melted blob sits at a fixed
  offset + the frame is duplicated.
- `Night-ride_*` (helmet, small face): mostly fine.
Pattern: identity **is** injected (Keanu looks like Keanu), but a **phantom face** also appears
at a misaligned location → ghost/duplicate. Worst for large/frontal faces, mild for small ones.

**(d) Cause = gen-bbox mask misalignment (the C6 item).** The face branch injects identity at the
auto-YOLO gen-bbox, which is detected on the PhotoMaker **preview**. Once branched attention
(step 15+) shifts the face, the bbox no longer matches where the face actually is → identity is
painted at the wrong spot = ghost. Evidence: `outputs/ba_debug/bbox_gen_auto_n1_realvis.json`
has a single face box per image at the *preview* location; the ghost sits near it while the real
face drifts away. OFF images (same `enable_vae_tiling`) are clean, so VAE tiling is ruled out.
`ba_face_prompt_mode=full_boosted` (×2.5) makes the phantom more prominent.

**(e) Weights (from §2) are sound.** N1 learned real ref/face deltas and has no drift channel, so
the *training* is fine; the failure is at **inference-time mask alignment + injection strength**,
not in what was learned.

### Recommended next steps
1. **Fix gen-bbox alignment (C6):** re-detect the face bbox on the branched trajectory (or track
   the actual face), instead of freezing it from the pre-branch PhotoMaker preview. This is the
   primary fix for the ghost/duplicate face.
2. **A/B the injection strength:** rerun ON with `ba_face_prompt_mode=id_only` (no ×2.5 boost) —
   if the phantom shrinks, `full_boosted` is over-driving it. (One `infer.py` override, ~8 min.)
3. **For N1 training panels:** set `pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0`
   so validation is photographic and comparable to the old run (the base won't look "drifted").
   Training itself is unaffected (still SDXL-base).

## 4. RESOLVED — the mess is `full_boosted`, not the base or the mask (id_only A/B)

Ran the id_only A/B (`outputs/ba_debug/N1_realvis_on_idonly`, same seeds/prompts/refs, only
`ba_face_prompt_mode` changed). Result is decisive across both identities and both prompts that
were broken under full_boosted:

| prompt/ref | `full_boosted` (my B1) | `id_only` (default) |
|---|---|---|
| Reading_jensen | melted blob + duplicate person | **clean single Jensen** |
| Skiing_jensen | stacked/melted goggles | **clean single skier** |
| Reading_keanu | ghost blob + frame duplication | **clean single Keanu** |

So the duplicate/ghost/blotch is caused by **`ba_face_prompt_mode=full_boosted`** — the ×2.5 ID
boost over-drives the face branch so any small preview↔branched offset blooms into a *second*
face inside the (expanded) mask. `id_only` injects gently onto the existing face → single,
coherent, good identity. Note the id_only run still had the infer.py generator-reuse bug and was
clean, so **the generator bug was not the driver** (fixed anyway for hygiene — see below).

**This matches the user's point** ("keep preview & branched face location the same") on two
fronts: (a) `id_only` stops the injection from manufacturing a second face; (b) the infer.py
generator re-seed keeps both passes' initial latents — hence face locations — identical (the
training path already did this via separate same-seed generators).

### The corrected "new approach" (all config, no new code needed for the fix)
- `branched_attn_weight_mode=ref_only`  ✓ (no drift channel — keep)
- `ba_face_prompt_mode=id_only`          ← **change from full_boosted** (the artifact fix)
- `pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0`  ← photographic base
- (`ba_uncond_face_fix=true` kept; infer.py generator re-seed kept as hygiene)

Net: RealVis base (matches the old approach) + ref_only (base preserved) + id_only (clean face
with injected identity) = reliable validation images, same base, improved face. Residual: mild
skin-texture roughness / faint mask edge under id_only — cosmetic, tunable later via
`mask_softness` / `mask_expansion_ratio`.

### Code/config touched
- `infer.py`: re-seed generators before the branched pass (align with training path).
- `src/configs/inference/ba_n1_realvis.yaml`: default `ba_face_prompt_mode: id_only`.
- `serv_new_runs/start_ba_ref_only_vast_N1.sh`: `id_only` + RealVis validation base.

## 5. FINAL epoch-10 panel — the "correct validation" the user asked to wait for

Full 24-image panel (12 prompts × jensen/keanu), `saved/ba_refonly1/weights-epoch10.pth`
(unchanged N1 ref_only weights) on RealVisXL, `ba_face_prompt_mode=id_only`, branched ON,
seed 0 → `outputs/ba_debug/N1_realvis_final/` (+ `_contact_jensen.png`, `_contact_keanu.png`).

**Headline: the severe full_boosted catastrophe is gone.** Across all 24 there are **no
duplicate people, no melted goggle-stacks, no ghost blobs**. Every image is a single coherent
subject on a photographic RealVis base whose scene follows the prompt — i.e. the "same base as
the old approach, improved face area" goal is met at the structural level.

**Identity splits sharply by reference hair length:**

| | jensen (short hair) | keanu (long hair) |
|---|---|---|
| clean, recognizable, single face | ~11–12 / 12 | ~7 / 12 |
| residual issue | faint mask-edge seam (e.g. Kickboxing neck), one softer id (Reading) | **hair draped over & smearing the face on motion poses** |
| worst cases | none severe | Dancing / Jumping / Skiing / Crying — a curtain of hair covers the face → smeared/blank features |

- **jensen** is a strong result: Angry/Crying/Drumming/Jumping/Kickboxing/Laughing/Night-ride/
  Rushing/Skiing are all clean recognizable Jensen; Chef has a *natural* second chef (scene, not a
  same-identity duplicate); Reading is slightly softer. Skiing — which was "stacked/melted goggles"
  under full_boosted — is now clean. This is the on-base + injected-face result we wanted.
- **keanu** is mixed. Static/portrait poses (Angry, Chef, Kickboxing, Laughing, Reading, Rushing)
  are good and recognizable. But **high-motion poses (Dancing, Jumping, Skiing, Crying) drape the
  ref's long hair over the face and smear it** — the face region is filled with hair strands and
  the features blur out.

**Cause of the keanu hair-smear = the same C6 static-bbox misalignment, now without a second
face.** id_only stopped the injection from *manufacturing* a duplicate, but the face branch still
paints ref content (including long hair) into the **static rectangular gen-bbox** detected on the
pre-branch preview. On motion poses the real face tilts/shifts out of that box, so what lands in
the box is ref hair over a displaced face → smear. Short hair (jensen) has little to drape, so it
stays clean; long hair (keanu) fills the box. So C6 (track/re-detect the gen-bbox on the branched
trajectory, or soften+shrink the mask) is the **primary remaining fix**, and it matters most for
long-haired / high-motion subjects.

**Bottom line for "go from there":**
- ✅ Reliable, on-base, photographic validation images — confirmed (was full_boosted + base-swap).
- ✅ Identity is injected and recognizable; ref_only weights are sound (no drift channel, §2).
- ⚠️ Residual: (a) long-hair motion smear (C6 static-bbox), (b) faint mask-edge seams
  (`mask_softness`/`mask_expansion_ratio`). Both are inference-time mask issues, **not** the
  weights and **not** the base.
- ➡️ Candidate next steps: (1) ON-vs-OFF on this same RealVis panel to quantify face improvement
  over plain PhotoMaker; (2) C6 gen-bbox tracking for the long-hair smear; (3) mask-edge softening;
  (4) whether to retrain with `id_only` so train- and infer-time face conditioning match (current
  epoch-10 weights were trained under full_boosted but validate fine under id_only).

## 6. Old-vs-new head-to-head (same prompts/refs/seed/base/step) — is new BETTER?

Direct A/B at matched step 20000 (= epoch 10, both runs are 2000 steps/epoch):
- OLD = `saved/03Jul_start_ba_cosm_new1_vast` `step_20000` (noise+ref, RealVis, its own live val)
- NEW = `outputs/ba_debug/N1_realvis_final` (ref_only epoch-10 weights, RealVis, infer.py, id_only)
Same 12 prompts × jensen/keanu, seed 0, RealVis base both. Sheets:
`_compare_jensen.png`, `_compare_keanu.png`, PDF `compare_old20k_vs_new_e10.pdf`.

**Verdict: NEW is NOT better on raw image quality — tie for jensen, somewhat worse for keanu.**

- **jensen (short hair): ~tie.** Both photographic, both recognizable, both single clean subjects.
  NEW marginally cleaner on Crying/Laughing; OLD holds identity marginally better on Reading. No
  meaningful winner.
- **keanu (long hair): OLD wins on motion poses.** Dancing, Jumping, Skiing (and milder Night-ride,
  Rushing) — OLD keeps the face **visible and recognizable**, while NEW smears it with the ref's
  long hair (the §5 static-bbox issue). Static poses (Angry, Chef, Kickboxing, Reading) ~tie.
  Net ~5/12 keanu worse in NEW, rest tied.

**Important framing (so this isn't over-read):**
1. This measures *validation-image quality on this 12-prompt panel*, not the two approaches'
   safety. OLD's images look fine **here**, but OLD is exactly the run whose `noise_to_v` (whole-
   image) channel trained hardest (§2) — the diagnosed drift mechanism. ref_only removes that
   channel by construction; its benefit is structural and doesn't show as prettier faces on this
   panel.
2. NEW's regressions (keanu hair-smear, mask seams) are **inference-time mask** faults (static
   gen-bbox not tracking the moving face; hard rectangular mask), **not** the weights — the
   epoch-10 ref deltas are sound (§2). So they're fixable without retraining: C6 gen-bbox tracking
   + `mask_softness`/`mask_expansion_ratio`.

**So, answer to "are the new results better than 03Jul_start_ba_cosm_new1_vast?": not yet.** We
achieved "same photographic base" (the drift complaint is resolved) but not "better face" — it's a
tie for short hair and a regression for long-hair motion poses. Getting a genuine *win* needs the
inference-time mask fix (C6) and/or an N2 retrain (id_only + ref-crop jitter; script ready at
`serv_new_runs/start_ba_ref_only_vast_N1.sh`).

## Runtime notes
- Local (WSL2, RTX 4090 Laptop 16 GB): RealVis + SDXL-base + PhotoMaker-V2 all cached; inference
  ~50 s/image, peak ~15.7 GB VRAM (fits, no OOM) with `enable_vae_tiling`.
- `infer.py` takes dotlist overrides; base/off toggled with
  `validation_args.use_branched_attention=false`.
