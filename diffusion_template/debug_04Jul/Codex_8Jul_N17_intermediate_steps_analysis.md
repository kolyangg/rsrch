# Codex 8 Jul — N17 intermediate checkpoints vs key configs

Inputs:

- N17 step images/metrics: `full_validation_results/ba_longrun_N17_different_steps/`
- Step metrics: `full_validation_results/ba_longrun_N17_different_steps/metrics_ba_longrun_N17_steps.json`
- Dedicated PDF:
  `full_validation_results/ba_longrun_N17_different_steps/full_val_report_N17_steps_vs_key.pdf`
- Report config:
  `infer_tools/full_val_n17_steps_report.yaml`
- Visual sheets:
  - `debug_04Jul/Codex_8Jul_N17_steps_issue_progression.png`
  - `debug_04Jul/Codex_8Jul_N17_steps_gain_progression.png`
  - `debug_04Jul/Codex_8Jul_N17_steps_identity_heatmap.png`
- Next-run script:
  `serv_new_runs/start_ba_combo_id075_16k_vast_N20.sh`

## PDF generation

Built a report input directory using symlinks:

```text
full_validation_results/ba_longrun_N17_different_steps/report_inputs/
```

It contains the N17 step folders plus the three comparison configs:

- `ba_combo_N14` — best previous combo, 6k.
- `ba_saonly6k_N15` — best SA-only, 6k.
- `ba_idloss_N13` — best short ID-loss / CA-trained reference, 3k.

Combined metrics written to:

```text
full_validation_results/ba_longrun_N17_different_steps/metrics_n17_steps_vs_key.json
```

Rebuild command:

```bash
cd /home/kolyangg/rsrch/diffusion_template
python3 infer_tools/pdf_full_val.py --config infer_tools/full_val_n17_steps_report.yaml
```

Output:

```text
[pdf] wrote full_validation_results/ba_longrun_N17_different_steps/full_val_report_N17_steps_vs_key.pdf  (10 pages: 1 summary + 1 config + 8 identities, 9 runs)
```

## Aggregate metrics

| run | step | mean id-sim | det |
|---|---:|---:|---:|
| N17 step 8k | 8000 | 0.3423 | 1.0000 |
| N17 step 10k | 10000 | 0.3431 | 1.0000 |
| **N17 step 12k** | **12000** | **0.3500** | 0.9896 |
| N17 step 14k | 14000 | 0.3379 | 1.0000 |
| N17 step 16k | 16000 | 0.3357 | 1.0000 |
| N17 final | 26000 | 0.3482 | 1.0000 |
| N14 combo | 6000 | 0.3324 | 0.9896 |
| N15 SA-only | 6000 | 0.3115 | 1.0000 |
| N13 ID-loss | 3000 | 0.3150 | 0.9896 |

The best aggregate is **N17@12k = 0.3500**, slightly above final N17@26k = 0.3482 and above
N14 = 0.3324.

The one missing detection at 12k is `Skiing wom_jisoo.png`, the same no-face case that also affects
N14 and N13. So 12k's mean is directly comparable to those runs.

## Per-identity read

| identity | 8k | 10k | 12k | 14k | 16k | 26k | best among report columns |
|---|---:|---:|---:|---:|---:|---:|---|
| eddie | 0.139 | 0.133 | 0.115 | 0.114 | 0.110 | 0.119 | N14, 0.140 |
| elon | 0.480 | 0.493 | **0.503** | 0.495 | 0.481 | 0.499 | N17 12k |
| jennie | 0.439 | 0.421 | 0.424 | 0.430 | 0.398 | 0.436 | N14, 0.444 |
| jensen | 0.392 | 0.424 | 0.424 | 0.434 | **0.450** | 0.445 | N17 16k |
| jisoo | 0.255 | 0.253 | **0.312** | 0.214 | 0.238 | 0.242 | N17 12k / N13 tie |
| keanu | 0.407 | 0.399 | 0.383 | 0.378 | **0.428** | 0.395 | N17 16k |
| lex | 0.348 | 0.361 | **0.383** | 0.361 | 0.361 | 0.379 | N17 12k |
| marion | **0.279** | 0.262 | 0.253 | 0.278 | 0.220 | 0.271 | N17 8k |

Interpretation:

- 12k is best for the identities that drove N17's aggregate gain: **Elon, Jisoo, Lex**.
- 16k is best for **Jensen and Keanu**, but visual inspection still shows pose/placement artifacts
  on some Keanu cases.
- 8k is best for **Marion**, and N14 remains best for **Eddie and Jennie**.
- There is no monotonic "more steps is better" trend.

## Per-image / visual read

The visual sheets matter more than small mean deltas here.

### Strong 12k wins

N17@12k clearly improves several hard cases over N14/N15/N13:

| image | 12k | N14 | note |
|---|---:|---:|---|
| `Crying wom_jisoo.png` | 0.337 | -0.003 | real recovery from the face/hand failure |
| `Reading pa_marion.png` | 0.423 | 0.179 | better face; 14k/26k also strong |
| `Skiing man_lex.png` | 0.365 | 0.129 | much more recognizable |
| `Dancing ma_elon.png` | 0.472 | 0.300 | better identity in hard pose |
| `Rushing ma_elon.png` | 0.613 | 0.463 | very strong |
| `Kickboxing_elon.png` | 0.573 | 0.428 | very strong |
| `Kickboxing_keanu.png` | 0.485 | 0.341 | better than N14/N15/N13, though 16k/26k are also strong |
| `Laughing w_jisoo.png` | 0.418 | 0.261 | strong recovery |
| `Dancing ma_lex.png` | 0.361 | 0.209 | strong recovery |
| `Drumming w_jennie.png` | 0.526 | 0.444 | strong recovery |

This confirms that the long frozen-CA combo is useful and that 12k is not just a noisy metric blip.

### 12k weaknesses

12k still loses badly on some occlusion/prop and already-good cases:

| image | 12k | N14 | note |
|---|---:|---:|---|
| `Skiing wom_jennie.png` | 0.167 | 0.438 | goggles/helmet mask collision; N14 visibly better |
| `Skiing man_keanu.png` | 0.197 | 0.363 | 10k is better than 12k here |
| `Crying man_jensen.png` | 0.464 | 0.580 | N14 remains visibly best |
| `Chef woman_marion.png` | 0.090 | 0.206 | N13 best, N17 weak |
| `Chef woman_jisoo.png` | 0.108 | 0.172 | 8k/10k/16k are better than 12k; final collapses |
| `Kickboxing_jennie.png` | 0.487 | 0.552 | N14/N13 better |

This reinforces the earlier conclusion: a single scalar mean hides the fact that different prompts
peak at different steps.

### Late-training collapse examples

Final 26k is not visually safe, even though its aggregate mean is high:

- `Kickboxing_marion`: 14k = 0.462, final = 0.076. Final visibly collapses the face.
- `Chef woman_jisoo`: 8k = 0.281, final = 0.026. Final is much worse.
- `Kickboxing_jisoo`: 12k = 0.365, final = 0.064.
- `Crying man_jensen`: N14 = 0.580, 14k = 0.535, final = 0.385.
- `Angry man _keanu`: N14 = 0.504, 14k = 0.491, final = 0.396.

This is the key new information from the intermediate checkpoints: **the over-strength/overfit
failure appears after the useful training phase**, and it is prompt/identity-specific.

### Rushing Keanu

`Rushing ma_keanu.png`:

| run | id-sim |
|---|---:|
| N17 8k | 0.463 |
| N17 10k | 0.452 |
| N17 12k | 0.459 |
| N17 14k | 0.451 |
| N17 16k | 0.478 |
| N17 26k | 0.424 |
| N14 | 0.448 |
| **N15** | **0.522** |
| N13 | 0.498 |

The N17 intermediate checkpoints are better than final, but N15/N13 still have the more natural
face placement. This supports the earlier diagnosis: the frozen-CA + ID-loss recipe improves
identity, but it does not fully solve pose/body placement for this prompt.

## Step selection

If choosing one checkpoint from the current N17 run:

1. **Use N17@12k as the best aggregate checkpoint**.
2. Keep N17@10k and N17@14k as useful visual alternatives for case-by-case panels.
3. Avoid using final 26k as the representative checkpoint for qualitative examples; it has too many
   visible late collapses.

Why not choose 10k? 10k wins the most individual images among N17-only checkpoints
(`22/96` by count), but 12k has the best mean and stronger recoveries on Elon/Jisoo/Lex hard cases.
For a single checkpoint, 12k is the better compromise.

## Next-run recommendation

The next long run should **not** be N19 as the default. N19 re-enables trainable CA; that is a useful
short probe, but the N17-step data says the main issue is excessive cumulative identity pressure
inside the fixed face box, not lack of CA capacity.

Best next long run: **N20 = N17 recipe with reduced ID-loss weight**.

Script created:

```text
serv_new_runs/start_ba_combo_id075_16k_vast_N20.sh
```

Config:

- frozen CA: `train_branched_ca_lora=false`;
- ID loss on, but reduced: `+model.id_loss_weight=0.075`;
- `+model.id_loss_max_timestep=500`;
- `loss_kind=blended_masked`;
- `lambda_face=0.15`;
- `branched_attn_weight_mode=noise_and_ref`;
- `+ba_noise_lr_scale=0.1`;
- `lr_for_lora=1e-4`;
- `optimizer.weight_decay=1e-3`;
- `trainer.max_grad_norm=1.0`;
- `lr_scheduler.warmup_steps=200`;
- `+model.ba_uncond_face_fix=true`;
- `+model.ba_face_prompt_mode=id_only`;
- RealVis validation;
- `batch_size=1`;
- `trainer.epoch_len=1000`;
- `trainer.n_epochs=16`;
- writer name: `ba_combo_id075_16k_N20`.

Rationale:

- N17@12k proves the recipe works.
- N17@14k/16k/26k show over-strength and prompt-specific collapse.
- Lowering ID loss from `0.1` to `0.075` is the narrowest single change that should preserve the
  gains while reducing pasted/canonical face pressure.
- 16k with 1k checkpoints is long enough to see whether the lower weight shifts the sweet spot
  later or stabilizes the late cases.

Run command:

```bash
cd /home/kolyangg/rsrch/diffusion_template
bash serv_new_runs/start_ba_combo_id075_16k_vast_N20.sh
```

After N20 finishes, full-validate selected checkpoints with:

```bash
BATCH_SIZE=4 bash serv_new_runs/run_full_validation_steps.sh ba_combo_id075_16k_N20 8000 10000 12000 14000 16000
```

If N20 still has the Rushing/Keanu long-neck issue, the next code-level lever is to un-hardcode
`POSE_ADAPT_RATIO=0.0` and test small pose blending (`0.1` or `0.2`). That is a different workstream
because it changes the face-branch forward logic.
