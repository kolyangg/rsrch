# Branched-Attention over PhotoMaker — brief for a senior researcher (04 Jul 2026)

## 1. Situation (concise)

**Goal.** Take PhotoMaker (https://github.com/TencentARC/PhotoMaker) and add a **branched-attention
(BA)** mechanism so identity is injected in the **face region** while the rest of the image stays
exactly the photographic PhotoMaker generation. Success = same image as stock PhotoMaker, only the
**face area** improved (better identity).

**Initial model** (commit `9b0dc27`, trained via `serv_new_runs/start_ba_cosm_new1.sh`; results in
`saved/03Jul_start_ba_cosm_new1_vast`):
- `branched_attn_weight_mode = noise_and_ref` — **both** the reference-side and the noise/gen-side
  attention weights are trained (840 processor tensors).
- `masked_alternating` loss, `lr 1e-4`, no grad-clip / no weight-decay.
- Validated on **`SG161222/RealVisXL_V4.0`** (photographic SDXL fine-tune).
- This model had generation issues (drift / double-face / face artifacts) — identified and logged
  across `debug_planning_03Jul/` (T0–T7 A/B matrix, checkpoint probes, worklog).

**New model** (proposed to fix those issues): `branched_attn_weight_mode = ref_only` — only the
reference-side weights train (420 tensors), the noise/gen pathway stays at base weights. Trained
→ `saved/ba_refonly1` (weights-epoch10). Inference was then **adjusted to match the original
validation setup** (RealVisXL base + `id_only` face injection + `ba_uncond_face_fix`), results in:
- `outputs/ba_debug/N1_realvis_final/`  — adjusted new-model panel (branched ON, no retrack)
- `outputs/ba_debug/N1_realvis_retrack/` — same + the C6 gen-bbox-retrack experiment

**User requirements / constraints (must respect):**
1. **Validation model stays the same** as the original (**RealVisXL_V4.0**). The new model's
   generations must look the **same** as the original except a **hopefully improved face area**;
   everything else identical.
2. **Both reference and noise must be trained** (`noise_and_ref`). `ref_only` previously caused the
   generated **face to be inconsistent with the rest of the body** — so the next model must go back
   to training both pathways.
3. The **BA mechanism's high-level logic must stay the same** as the initial model. The suspicion is
   **implementation issues**, not the architecture — do not re-design BA.
4. **Intentional design choices — do NOT treat as bugs:** `POSE_ADAPT_RATIO = 0.0` and
   `CA_MIXING_FOR_FACE = False` (both hardcoded in `attn_processor_cleanest.py`) are deliberate — the
   face branch injects the pure reference face for maximum identity fidelity. Leave as-is.

## 2. Further steps (what to do)

1. **Analyze results of both models.** For the adjusted (new) model, use the images generated with
   the **same validation model (RealVisXL)** as the original, so the two differ **only** in the face
   area — then judge face quality and, crucially, face↔body consistency.
2. **Re-read the earlier observations** on the initial model's issues in `debug_planning_03Jul/`
   and identify any additional ones — **under the constraints**: keep BA high-level logic the same,
   keep the validation model = RealVisXL, keep **ref+noise** both trainable. Reframe the earlier
   `ref_only` fix (which violated constraint #2) into fixes that keep both pathways.
3. **Run local A/B tests** as needed (inference from the saved weights; the 16 GB local GPU is fine
   for inference).
4. **Record all findings** in `debug_04Jul/04Jul_findings.md` (new folder).
5. **Propose the best next setup** in that same file. **Ask the user to approve it, and only then
   modify code.**

## 3. Pointers (ground truth)
- Initial weights: `saved/03Jul_start_ba_cosm_new1_vast/weights-epoch{1,2,14}.pth` (+ `config.yaml`,
  `val_images/…/step_*` panels through step 28000). noise_and_ref, 237 MB.
- New weights: `saved/ba_refonly1/weights-epoch10.pth`. ref_only, 165 MB.
- Adjusted-inference results: `outputs/ba_debug/N1_realvis_final` (+ `_contact_*`,
  `compare_old20k_vs_new_e10.pdf`, `_compare_*`), `outputs/ba_debug/N1_realvis_retrack`.
- Prior analysis: `debug_planning_03Jul/ba_realvis_isolation_04Jul.md` (§0 folder-label map, §1–§6),
  `ba_training_fix_plan_v2.md`, `ba_debug_worklog_03Jul.md`, `ba_val_crash_fix_04Jul.md`,
  `ba_gen_bbox_retrack_04Jul.md`, `setup_diff_vs_original_04Jul.md`.
- Checkpoint probe: `scripts/inspect_ba_checkpoint.py` (Frobenius norms per processor branch group).
- Findings destination: `debug_04Jul/04Jul_findings.md`.
