# Codex 8 Jul — N17 long-run full-validation analysis

Inputs:

- Images: `full_validation_results/ba_longrun_N17/`
- N17 metrics: `full_validation_results/metrics_ba_longrun_N17.json`
- Prior-run metrics: `full_validation_results/metrics.json`
- Combined metrics for the refreshed PDF: `full_validation_results/metrics_with_N17.json`
- Refreshed PDF: `full_validation_results/full_val_report.pdf`
- Visual contact sheets:
  - `debug_04Jul/Codex_8Jul_N17_issue_contact.png`
  - `debug_04Jul/Codex_8Jul_N17_gain_contact.png`

## PDF update

Created `full_validation_results/metrics_with_N17.json` by merging the original 11-run
`metrics.json` with `metrics_ba_longrun_N17.json`.

Updated `infer_tools/full_val_report.yaml`:

- `metrics_json: full_validation_results/metrics_with_N17.json`
- added `ba_longrun_N17` as the first run column;
- added label `N17 long 26k`.

Rebuilt:

```bash
cd /home/kolyangg/rsrch/diffusion_template
python3 infer_tools/pdf_full_val.py --config infer_tools/full_val_report.yaml
```

Output:

```text
[pdf] wrote full_validation_results/full_val_report.pdf  (10 pages: 1 summary + 1 config + 8 identities, 12 runs)
```

## Metric read

N17 is now the top final-checkpoint run on the 96-image full validation, but the margin over N14 is
modest:

| rank | run | step | mean id-sim |
|---:|---|---:|---:|
| 1 | **ba_longrun_N17** | 26000 | **0.3482** |
| 2 | ba_combo_N14 | 6000 | 0.3324 |
| 3 | ba_idloss_N13 | 3000 | 0.3150 |
| 4 | ba_saonly6k_N15 | 6000 | 0.3115 |
| 5 | ba_idloss6k_N16 | 6000 | 0.2811 |

N17 vs N14: **+0.0158 mean id-sim**. That is a real aggregate gain, but not large enough to ignore
visible regressions.

Per identity:

| identity | N17 | N14 | best run among all 12 | read |
|---|---:|---:|---|---|
| elon | **0.499** | 0.423 | N17 | clear N17 win |
| lex | **0.379** | 0.309 | N17 | clear N17 win |
| keanu | **0.395** | 0.386 | N17 | small N17 win |
| jensen | **0.445** | 0.439 | N17 | tiny N17 win |
| marion | **0.271** | 0.249 | N17 | small N17 win, still visually mixed |
| jennie | 0.436 | **0.444** | N14 | N17 slightly worse |
| jisoo | 0.242 | 0.263 | N13, 0.312 | N17 regresses vs N14/N13 |
| eddie | 0.119 | 0.140 | N12, 0.141 | N17 does not solve this identity |

Among the main candidates (`N17`, `N14`, `N15`, `N13`, `N6`), N17 is best on **39/96** images.
N14 is best on 22, N15 on 16, N13 on 13, and N6 on 6. So N17 is the strongest single checkpoint,
but it is not uniformly best.

The training-time 2-identity validation curve also argues against blindly trusting the final epoch:

| epoch | step | manual_val_two/id_sim |
|---:|---:|---:|
| 6 | 12000 | 0.4388 |
| 7 | 14000 | 0.4425 |
| **8** | **16000** | **0.4474** |
| 9 | 18000 | 0.4418 |
| 10 | 20000 | 0.4447 |
| 11 | 22000 | 0.4257 |
| 12 | 24000 | 0.4372 |
| 13 | 26000 | 0.4269 |

Locally, only `weights-epoch13.pth` / `checkpoint-epoch13.pth` are present in
`saved/ba_longrun_N17/`, even though the log shows earlier checkpoints were saved during training.
If epoch 8/10/12 checkpoints still exist on the training box, full-validating them is the cheapest
next step.

## Visual read

I relied most on `Codex_8Jul_N17_issue_contact.png` and `Codex_8Jul_N17_gain_contact.png`, not just
small metric differences.

### Where N17 clearly helps

N17 is visibly stronger on many clean or moderately hard cases:

- Elon: `Dancing ma_elon`, `Crying man_elon`, `Kickboxing_elon`, `Night-ride_elon` are much more
  identity-faithful than N14/N15/N13.
- Lex: `Night-ride_lex`, `Rushing ma_lex`, `Skiing man_lex` improve strongly.
- Keanu: `Kickboxing_keanu` is substantially better integrated than N14 and cleaner than N6.
- Some previously bad occlusion cases improve: `Crying wom_jisoo`, `Angry woma_jisoo`.
- Marion improves on several non-prop cases: `Reading pa_marion`, `Dancing wo_marion`,
  `Laughing w_marion`.

This confirms the long frozen-CA combo is still learning useful identity/pose behavior. N17 is not
just a metric artifact.

### Key issues in N17

1. **Mask/occlusion collision is still the main unsolved failure mode.**

   The worst N17 images are dominated by cases where face-box content includes props, hair, goggles,
   hands/gloves, or chef hats. N17 often pushes identity harder into the whole box and makes the
   visual result worse:

   - `Chef woman_jisoo`: N17 0.026 vs N14 0.172 / N13 0.269. Visually, the face/hat/hair region is
     confused; N13 is clearly cleaner.
   - `Skiing wom_jisoo`: all runs struggle; N17 still paints into the orange goggles/helmet area.
   - `Kickboxing_jisoo`: N17 0.064 vs N14 0.262 / N6 0.223. N17 over-injects into the glove/hair
     occlusion area.
   - `Reading pa_jisoo`: N17 is visibly less recognizable than N14/N13.
   - `Kickboxing_marion`: N17 0.076 vs N15 0.402 / N13 0.311. The shorter SA-only/ID-loss runs look
     much better here.

   This is not primarily a long-run config problem. It is the same mask/data-side issue already
   diagnosed: the generated face region must exclude occluders/props using landmarks or segmentation,
   not just a broad face bbox.

2. **There is some long-run over-strength / over-training behavior.**

   Several cases that were already good at N14 become less recognizable or less natural at N17:

   - `Crying man_jensen`: N17 0.385 vs N14 0.580 / N13 0.517. Visually N14/N13 are better.
   - `Angry man _keanu`: N17 0.396 vs N14 0.504 / N15 0.470.
   - `Kickboxing_jennie`: N17 0.457 vs N14 0.552 / N13 0.521.
   - `Kickboxing_jensen`: N17 0.489 vs N14 0.590.
   - `Night-ride_jensen`: N17 0.408 vs N14 0.494 / N13 0.464.

   These are not catastrophic, but they make the final 26k checkpoint a mixed visual choice. The
   training log peak around epoch 8 reinforces that checkpoint selection matters.

3. **Eddie remains a hard identity, not a solved training-recipe problem.**

   N17 is not better on Eddie overall: N17 0.119 vs N14 0.140, and every recipe is weak. Images such
   as `Laughing m_eddie`, `Jumping ma_eddie`, `Night-ride_eddie`, and `Chef man i_eddie` are mostly
   generic Black male generations rather than Eddie Murphy. More of the same training is unlikely to
   fix this without better identity signal, references, or a more robust identity loss/evaluator.

4. **The aggregate gain is real but comes from uneven distribution.**

   N17 wins big on Elon and Lex and enough hard-pose cases to take the mean. It does not uniformly
   dominate N14/N15/N13. For publication/claim purposes, N17 should be presented as the best current
   aggregate checkpoint, not as a universally better visual checkpoint.

## Next-run recommendation

Do **not** make N19 the next long run by default.

Reason: N19 trains branched cross-attention again. N16 already showed that ID loss with trainable CA
degrades over steps, and N17's main visible failures do not look like "missing CA capacity"; they
look like mask/occlusion collision plus cumulative over-strength. N19 is still a useful hypothesis,
but it should be a short probe first, not a 15k long run.

Do **not** run original N18 blindly as the main answer either.

Reason: after N17, the question is no longer "does the combo keep improving past 6k?" It does. The
question is "where is the visual/full-val sweet spot before over-strength starts to hurt?" Original
N18 stops at 10k, but the N17 manual-val curve peaked around 16k.

Best next action, if possible:

1. Recover N17 intermediate checkpoints, especially epoch 8/10/12.
2. Run the 96-image full validation on those checkpoints.
3. Pick the best checkpoint by full-val + visual sheets, not final epoch.

If intermediate N17 checkpoints are not recoverable, the best next long run is **N18b / N20**, not
N19:

- same core recipe as N17/N14: frozen BA cross-attn + ID loss;
- `train_branched_ca_lora=false`;
- `+model.use_id_loss=true`;
- `+model.id_loss_weight=0.1` for a clean checkpoint-selection rerun;
- `+model.id_loss_max_timestep=500`;
- `loss_kind=blended_masked`;
- `lambda_face=0.15`;
- `branched_attn_weight_mode=noise_and_ref`;
- `+ba_noise_lr_scale=0.1`;
- `lr_for_lora=1e-4`, `optimizer.weight_decay=1e-3`, `trainer.max_grad_norm=1.0`,
  `lr_scheduler.warmup_steps=200`;
- `+model.ba_uncond_face_fix=true`, `+model.ba_face_prompt_mode=id_only`;
- RealVis validation, clean ref, `bs=1`;
- **epoch_len=1000, n_epochs=16 or 18**, save/validate every 1000 steps.

Then full-validate checkpoints around **8k, 10k, 12k, 14k, 16k**. If the visual regressions are
already present by 10k, the next variant should lower ID pressure (`id_loss_weight=0.075`) rather
than train CA.

Separate high-value workstream before more long runs:

- replace broad gen face bbox with face-landmark or segmentation mask that excludes goggles, gloves,
  hands, earrings, hair occluders, and hats;
- re-run inference/full-val for N17 or the selected checkpoint with the tighter mask.

Expected payoff: this should specifically lift Jisoo/Marion/Skiing/Kickboxing/Chef failures, where
another plain long run is least likely to help.

## Short answer

- Current best aggregate checkpoint: **N17 final**, but only by a small margin over N14.
- Best immediate measurement: full-val N17 epoch 8/10/12 if those checkpoints can be recovered.
- Best next long run if rerun is required: **N18-style frozen-CA combo with 1k checkpoints extended
  to 16k/18k**, not N19.
- Best next research fix: tighter face/occlusion mask before spending another long run.
