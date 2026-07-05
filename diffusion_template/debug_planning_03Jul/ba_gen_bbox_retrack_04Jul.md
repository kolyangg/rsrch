# C6 — gen-bbox re-tracking on the branched trajectory (04 Jul 2026)

## Problem (from §5/§6 of ba_realvis_isolation_04Jul.md)
The generation face mask is a **static rectangle** detected once, on the pre-branch PhotoMaker
**preview** (branched OFF). When branched attention turns on (step 15+) the face the branched pass
actually produces can sit at a different place/scale than the preview. The static mask then:
- paints reference content (esp. **long hair**) into the frozen box where there's no real face →
  the keanu "curtain of hair over a smeared face" on motion poses (Dancing/Jumping/Skiing);
- leaves the real (branched) face partly outside the merge region.
Short-hair refs (jensen) barely show it; long hair fills the box → smear.

## Fix — re-detect the box on the branched trajectory (toggle)
During the branched denoising loop, every `gen_bbox_retrack_every` steps (once past
`gen_bbox_retrack_min_frac` of the schedule, so the decode is clean enough), decode the current
latents, run the same face detector, and **rebuild the gen mask in place** so the merge follows
the branched face. `prepare_mask4` already re-reads `pipeline._face_mask` every step, so updating
it mid-loop propagates with no other plumbing.

**Easy on/off:** single flag `gen_bbox_retrack` (default **false** = original frozen-box
behaviour, byte-identical). Cadence knobs `gen_bbox_retrack_every` (default 6) and
`gen_bbox_retrack_min_frac` (default 0.5). All live in `validation_args`, so infer.py forwards
them via `call_args`; flip on the CLI with `validation_args.gen_bbox_retrack=true`.

## Implementation (all in `src/pipelines/photomaker_branched_clean.py`)
- `__call__`: new params `gen_bbox_retrack{,_every,_min_frac,_detector,_model,_conf,_padding,_debug_dir}`.
- Denoising loop: after `scheduler.step`, when enabled + branched + in-window + on cadence, call
  `self._retrack_gen_bbox(...)`.
- `_decode_latents_to_pil(latents)`: clone-decode (mirrors the final VAE block; never touches the
  loop's `latents`).
- `_retrack_gen_bbox(...)`: lazy-init the detector on **CPU** (no extra VRAM), decode → detect per
  sample → keep the old box where detection fails → rebuild the mask via `prepare_gen_mask_helper`
  (per-sample list when batch>1). If nothing is detected, keep the existing mask (no-op).
- Config: `src/configs/inference/ba_n1_realvis.yaml` `validation_args.gen_bbox_retrack: false` (+ knobs).

Cost: ~1 extra VAE decode + 1 CPU-YOLO pass per retrack (a few per image). Detector on CPU to
protect the 16 GB laptop GPU during the branched pass.

## A/B plan
Same 24-panel (12 prompts × jensen/keanu, RealVis, id_only, epoch-10 weights, seed 0):
- NEW (no retrack)  = `outputs/ba_debug/N1_realvis_final`   (already have it)
- NEW + retrack     = `outputs/ba_debug/N1_realvis_retrack` (`gen_bbox_retrack=true`)
Add a 3rd column to the comparison PDF: OLD 20k | NEW e10 | NEW e10 + retrack. Expectation: keanu
motion poses lose the hair-curtain; jensen ~unchanged (little hair to drape). Results below once
the run finishes.

## Results — mechanism works, but it does NOT fix the smear (honest negative)

Full 24-panel rerun with `gen_bbox_retrack=true` → `outputs/ba_debug/N1_realvis_retrack/`; 3-col
sheets `_compare3_{jensen,keanu}.png` + `compare_old_vs_new_vs_retrack.pdf` (OLD 20k | NEW e10 |
NEW e10 + retrack).

**Mechanism verified.** Retrack fires and updates the box on the model's x0 estimate (e.g. subway-
Jensen step 30 → `[292,101,498,369]`, cleanly on the face); no OOM; 24/24 at ~44 s/img.

**But visually it barely moves the keanu smear.** Dancing / Jumping / Skiing keanu are still
hair-over-face smeared with retrack on (Jumping keanu: features still blurred out under a curtain
of hair). Jensen (short hair) is ~identical across NEW and NEW+retrack, as expected.

**Why — the box wasn't mislocated.** Steps 0–15 are non-branched PhotoMaker, so the composition
(face location) is fixed *before* branching starts; the preview box is already ~correct, and
re-detecting on the branched x0 returns essentially the same location. The smear is not a
mask-location error — it's the **ref_only face branch painting the ref's long hair *over* the
face inside a correctly-placed box**. Re-centering the same-size box on the same hairy face keeps
the same content. Also note `mask_expansion_ratio=1.0` = **no** growth (grow=ratio−1=0), so it's
not an over-large mask either.

**Takeaway.** C6 retrack is sound engineering and now a clean toggle (default off) — worth keeping
for cases where the branched face genuinely *moves* — but it is **not** the lever for the long-hair
motion smear. The smear is a face-branch **content/injection** issue specific to ref_only + long
hair, not a bbox-tracking issue.

### Better levers to try next (cheap A/Bs, all inference-side)
1. `mask_softness > 0` (e.g. 0.2–0.4): feather the merge edge (helps the seam; won't clear hair at
   the face center).
2. `mask_expansion_ratio < 1.0` (e.g. 0.7–0.85): **shrink** the mask to the inner face so less
   hair-heavy border is painted by the face branch (most promising for the smear; watch the jaw).
3. N2 retrain (ref_only + id_only + ref-crop jitter): the sharpness/context jitter may teach the
   branch to inject less literally — speculative, expensive.
The honest current state: ref_only + id_only is strong for short-hair identities and still loses to
the OLD noise_and_ref run on long-hair **motion** poses; retrack didn't close that gap.
