# 04 Jul worklog — session 2 (continuation after reboot)

Running log of actions/decisions this session. Findings land in `04Jul_findings.md`; this file is
the how/why trail.

## State found at session start
- `04Jul_findings.md` §0–§5 already drafted (session 1). §6 (local A/B evidence) marked *running*.
- `outputs/ba_debug/OLD_noiseref_realvis/` had only **6/24 images** (jensen prompts 1–6, no keanu,
  no logs) — the A/B-1 inference was killed by a machine reboot mid-run (images timestamped
  00:02–00:18, system up since 00:43).
- GPU idle, `photomaker` env present.

## Actions
1. **Relaunched A/B-1** (background):
   `python infer.py --config-name inference/ba_old_noiseref_realvis`
   → `outputs/ba_debug/OLD_noiseref_realvis` (+ `.log`). Deterministic (seed 0, same bbox cache
   `bbox_gen_auto_n1_realvis.json`), so overwriting the 6 existing images is safe. ~66 s/panel →
   ~25 min for 24 panels.
2. **§4.2 gradient-routing inspection** (the "to verify" item from findings §4) — done, see
   findings §4.2. Key discovery: **the branched CROSS-attn processor has no mask and no merge for
   the noise half** — the whole generated image (face included) goes through `noise_to_q/k/v`
   against the gen prompt; `ref_to_*` in attn2 only serves the reference half
   (`attn_processor_cleanest.py:670–713`). Combined with `masked_alternating` (`masked_loss_step=2`
   → every 2nd step's loss is ONLY the face bbox crop, `diffusion_loss.py:_masked_face_mse`), the
   face-only steps push the noise CA group at full strength with **zero background anchor** —
   at lr 1e-4 / wd 0 / no clip. This is the drift engine; matches the checkpoint probe
   (`attn2 noise_to_v` = largest delta) and the canary docstring's empirical note
   ("doubling per 2k steps, worst in ca_noise", `sdxl_trainers.py:243`).
3. Verified supporting facts:
   - SA merge gate is strictly binary (`force_binary_masks=True`,
     `attn_processor_cleanest.py:138,444`) → SA `noise_to_k/v` are shielded from face-loss
     gradients per-layer. Clean.
   - `noise_to_q` is shared by both SA branches (`q_face = noise_to_q(noise_hidden)·mask`) →
     face loss trains it **by design** (query must come from current generation).
   - Loss supervises only the noise-half prediction (`run_branched_forward_pass` returns merged
     noise pred; ref reconstruction unsupervised) → `ref_to_*` trains via the face-injection path
     only. As intended.
   - Optimizer: **single param group** (`lora2.py:get_trainable_params`), all params at
     `lr_for_lora`; AdamW wd=0 (initial run). A commented-out draft of custom grouping sits at
     `lora2.py:237–260` — the natural insertion point for per-group (ref vs noise) LR/WD.
   - Drift canary `ba_norm/{sa,ca}×{ref,noise}` **already implemented** in
     `sdxl_trainers.py:_update_ba_weight_norms`, logs every `log_step`. Proposal item downgraded
     from "add" to "watch".
4. **Implication accepted into the proposal:** `blended_masked` is not merely "smoother" — it is
   the structural fix for the drift engine (every step keeps a `(1−λ_face)` full-image anchor on
   the noise CA group). Differential LR/WD on `noise_*` becomes the secondary damper. Both are
   config/optimizer-level; **no BA forward-logic change** (constraint #3 intact).
5. Ran `scripts/idsim_report.py` (CPU) on the new-model panel + will score the old-model panel
   when inference completes; numbers go to findings §6.
6. Next (pending run completion): build comparison contact sheets old-e14 vs new-e10, fill §6,
   finalize §5, ask user for approval.

## Interim results while A/B-1 renders
- **idsim (NEW, ref_only e10, `N1_realvis_final`, 24 panels):** jensen mean **0.416**, keanu mean
  **0.287**. The 4 worst keanu scores are exactly the visually-identified smear set — Crying 0.146,
  Dancing 0.146, Jumping 0.180, Skiing 0.235 → the face↔body inconsistency is quantitatively
  visible in id-sim. (Full report: scratchpad `idsim_new_e10.txt`; means exclude contact sheets.)
- **Checkpoint-probe growth (initial model, `attn2 noise_to_v` mean Frobenius):** 0.84 @e1 → 1.22
  @e2 → 1.72 @e14. ~70% of the drift is in place by epoch 2 → hygiene must be on from step 0;
  2-epoch probe runs are decisive. Added to findings §4.2.
- **Early qualitative peek (Skiing jensen):** OLD e14 under corrected inference still melts the
  goggles across the face (trained-in face-branch damage; corrected inference does NOT rescue e14),
  while NEW keeps a coherent face. Consistent with T1 (core drift is training-side).
- **A/B validity check:** diffed `ba_n1_realvis.yaml` vs `ba_old_noiseref_realvis.yaml` — only
  checkpoint, `branched_attn_weight_mode`, and output/debug paths differ. Valid isolation.
- **Provenance note:** `setup_diff_vs_original_04Jul.md` §2 says refonly1 predates ref-crop jitter,
  but `saved/ba_refonly1/config.yaml` records jitter 0.2/0.6/0.5 — trust the saved config; either
  way the next run keeps jitter.
- Note on the earlier head-to-head (isolation doc §6): its OLD side used the *training-time live
  val* step-20k panels; A/B-1 is the cleaner version (both sides through the same infer.py +
  corrected inference + shared bbox cache).

## A/B-1 complete (24/24 panels, exit 0)
- Scored with `idsim_report.py`; reports copied into both output dirs (`idsim_report.txt`).
- Built `OLD_noiseref_realvis/compare_oldE14_vs_newE10.pdf` (6 pages, per-prompt OLD|NEW with
  id-sim in labels; generator script in scratchpad `make_ab_compare.py`) + `_contact_*.png`.
- **Result (full write-up in findings §6):** OLD jensen 0.341 / keanu 0.237; NEW jensen 0.416 /
  keanu 0.287. Complementary failures: OLD = all poses integrated (no smears) but systematic
  orange/waxy face-patch color-cast + seam, catastrophic prop-melts (Skiing goggles, Kickboxing
  sweatband, Rushing orange-face −0.01); NEW = color-clean but motion smears (worst-4 = smear set).
  Neither acceptable alone → keep noise_and_ref + kill the drift with loss/optimizer hygiene.
- Findings §5/§6/§7 finalized; proposal presented to user for approval. **No code changed.**

## User approval + implementation (05 Jul)
- User approved the plan with one amendment: **loss stays an A/B** — original `masked_alternating`
  (as in stock PhotoMaker LoRA) with all other fixes first, then `blended_masked` on top as a
  separate run. Asked for two vast scripts, a steps recommendation, and a 45 GB batch-size check.
- **Implemented** (first and only code change of this effort):
  - `lora2.py:get_trainable_params` — per-branch optimizer grouping via top-level
    `ba_noise_lr_scale` (default 1.0) / `ba_noise_weight_decay` (optional). noise_to_* processor
    clones → `ba_noise_params` group at scaled LR. Neutral defaults = bit-identical old behaviour;
    ref_only checkpoints (no noise clones) fall back to single group.
  - `serv_new_runs/start_ba_nr_alt_vast_N3a.sh` (masked_alternating) and
    `start_ba_nr_blend_vast_N3b.sh` (blended λ0.15) — clones of the N1 script pattern with
    noise_and_ref + hygiene + `+ba_noise_lr_scale=0.25`; diff between them = loss overrides +
    run_name only (verified).
- **Verified:** 4-case unit test of the grouping (single-group default, 0.25× split, wd override,
  ref_only fallback) + AdamW instantiation; `bash -n` both scripts; full hydra compose of both
  exact override lists (N3a + N3b) against `one_id_09Feb_testing`.
- **Timing basis:** saved-run checkpoint mtimes (e2→e14 = 24k steps in ~8.8 h) → ~45 min per
  2k-step epoch on the vast A100. Gates: canary @4k steps, main comparison @20k steps (matches
  initial@20k and refonly@e10). Memory estimate for 45 GB @ bs=2: peak ≈30–34 GB (details in
  findings §8).

## Notes / decisions
- Chose **full rerun** over a resume hack: infer.py has no skip-existing logic and adding one would
  be an unapproved code change; determinism makes the rerun exact. Cost ~25 min, acceptable.
- CA-processor docstring says "Output: [merged_result, …]" but no merge exists in the code — the
  merge was deliberately removed (`CA_MIXING_FOR_FACE=False` hardcoded, user-confirmed intent).
  Treated as ground truth (constraint #4); documented the gradient consequence instead of
  "fixing" the docstring or the routing.
