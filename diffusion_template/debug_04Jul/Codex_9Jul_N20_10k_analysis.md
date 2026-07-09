# Codex 9 Jul — N20 10k full-validation analysis

Inputs:

- N20 saved run: `saved/ba_combo_id075_16k_N20`
- N20 full-val images: `full_validation_results/ba_combo_id075_16k_N20/step_10000`
- N20 metrics: `full_validation_results/ba_combo_id075_16k_N20/metrics.json`
- Comparison PDF:
  `full_validation_results/ba_combo_id075_16k_N20/full_val_report_N20_vs_key.pdf`
- PDF config:
  `infer_tools/full_val_n20_report.yaml`
- Combined comparison metrics:
  `full_validation_results/ba_combo_id075_16k_N20/metrics_n20_vs_key.json`
- Contact sheets:
  - `debug_04Jul/Codex_9Jul_N20_issue_contact.png`
  - `debug_04Jul/Codex_9Jul_N20_gain_contact.png`
- Next-run script:
  `serv_new_runs/start_ba_combo_idloss_t400_14k_vast_N21.sh`

## PDF generation

Built a dedicated report input folder:

```text
full_validation_results/ba_combo_id075_16k_N20/report_inputs/
```

It links:

- N20 10k from `full_validation_results/ba_combo_id075_16k_N20/step_10000`
- N17 10k
- N17 12k
- N17 final 26k
- N14 combo 6k
- N15 SA-only 6k
- N13 ID-loss 3k

Rebuild command:

```bash
cd /home/kolyangg/rsrch/diffusion_template
python3 infer_tools/pdf_full_val.py --config infer_tools/full_val_n20_report.yaml
```

Output:

```text
[pdf] wrote full_validation_results/ba_combo_id075_16k_N20/full_val_report_N20_vs_key.pdf  (10 pages: 1 summary + 1 config + 8 identities, 7 runs)
```

## Intermediate inference note

I tried local N20@5k full-val inference with the `photomaker` conda env:

```bash
/home/kolyangg/anaconda3/envs/photomaker/bin/python infer.py \
  --config-name inference/full_val \
  saved_checkpoint=saved/ba_combo_id075_16k_N20/weights-epoch5.pth \
  output_dir=full_validation_results/ba_combo_id075_16k_N20/step_5000 \
  batch_size=1
```

The env works and CUDA is available, but on the local 16 GB 4090 Laptop GPU it ran at about
65-72 seconds/image even with `batch_size=1`, implying roughly two hours for 96 images. Since the
N20@10k result is already diagnostic and the 5k point was optional, I stopped the partial run and
removed the incomplete `step_5000` folder to avoid mixing partial results into reports.

## Aggregate comparison

| run | step | mean id-sim | detection |
|---|---:|---:|---:|
| **N17 12k** | 12000 | **0.3500** | 95/96 |
| N17 final | 26000 | 0.3482 | 96/96 |
| N17 10k | 10000 | 0.3431 | 96/96 |
| N14 combo | 6000 | 0.3324 | 95/96 |
| **N20 id075** | 10000 | **0.3238** | 96/96 |
| N13 ID-loss | 3000 | 0.3150 | 95/96 |
| N15 SA-only | 6000 | 0.3115 | 96/96 |

N20 is **not an improvement** at 10k. It is:

- `-0.0193` vs N17 10k;
- `-0.0262` vs N17 12k;
- `-0.0244` vs N17 final;
- `-0.0086` vs N14 combo.

It is only slightly above N13/N15.

## Per-identity comparison

| identity | N20 10k | N17 10k | N17 12k | N17 final | N14 | read |
|---|---:|---:|---:|---:|---:|---|
| eddie | 0.099 | 0.133 | 0.115 | 0.119 | **0.140** | N20 weak |
| elon | 0.474 | 0.493 | **0.503** | 0.499 | 0.423 | N20 OK but below N17 |
| jennie | 0.418 | 0.421 | 0.424 | 0.436 | **0.444** | N20 below N14/N17 |
| jensen | 0.426 | 0.424 | 0.424 | **0.445** | 0.439 | N20 competitive |
| jisoo | 0.194 | 0.253 | **0.312** | 0.242 | 0.263 | N20 clear regression |
| keanu | 0.370 | 0.399 | 0.383 | **0.395** | 0.386 | N20 regression |
| lex | 0.339 | 0.361 | **0.383** | 0.379 | 0.309 | N20 below N17 |
| marion | 0.270 | 0.262 | 0.253 | **0.271** | 0.249 | N20 competitive |

N20 mostly preserves Jensen/Marion and remains decent on Elon, but it loses too much on Jisoo,
Keanu, Lex, and Eddie.

## Visual / per-image read

### Main N20 failures

See `debug_04Jul/Codex_9Jul_N20_issue_contact.png`.

Worst and most informative cases:

| image | N20 | better comparator | read |
|---|---:|---:|---|
| `Night-ride_jisoo.png` | 0.039 | N17 final 0.371 / N14 0.353 | near-collapse for Jisoo |
| `Crying wom_jisoo.png` | 0.048 | N17 12k 0.337 / N13 0.339 | loses the N17 12k recovery |
| `Kickboxing_keanu.png` | 0.275 | N17 final 0.511 / N17 12k 0.485 | weaker identity and face integration |
| `Rushing ma_keanu.png` | 0.339 | N15 0.522 / N13 0.498 / N17 12k 0.459 | does not solve the long-neck/placement issue |
| `Skiing man_lex.png` | 0.164 | N17 10k 0.375 / N17 12k 0.365 | large Lex regression |
| `Skiing wom_jennie.png` | 0.162 | N14 0.438 | N20 weaker on goggles/helmet case |

The important result is that lower ID weight did **not** solve the hard pose/placement cases. It
also weakened several identities that N17 handled better.

### Main N20 gains

See `debug_04Jul/Codex_9Jul_N20_gain_contact.png`.

N20 does improve some N17 final late-collapse cases:

| image | N20 | N17 final | read |
|---|---:|---:|---|
| `Kickboxing_marion.png` | 0.361 | 0.076 | N20 avoids the severe final N17 collapse |
| `Chef woman_jisoo.png` | 0.154 | 0.026 | better than final N17, still below N13 0.269 |
| `Crying man_jensen.png` | 0.502 | 0.385 | better than final N17, still below N14 0.580 |
| `Angry man _keanu.png` | 0.511 | 0.396 | strong metric recovery |
| `Dancing ma_elon.png` | 0.466 | 0.476 | close to N17, much better than N14 |
| `Reading pa_marion.png` | 0.359 | 0.453 | N20 strong vs old baselines, below final N17 |

This confirms the hypothesis that reducing ID pressure can reduce some late over-strength artifacts.
But at `0.075`, the cost is too high: the aggregate and key identities drop.

## Interpretation

N20 answers the question: **ID-loss weight 0.075 is too weak at 10k** for the current frozen-CA
recipe.

It partially fixes a few overfit/collapse cases, but it also under-trains identity on exactly the
identities where N17 made the largest gains. The failure is not just "needs more steps"; N17 at 8k
and 10k was already around `0.342-0.343`, while N20 at 10k is only `0.3238`.

If N20 is cheap to continue, it is still worth collecting 12k and 14k checkpoints because lower ID
weight may shift the sweet spot later. But with limited GPU time, I would not make continuing N20
the main next bet.

## Next best config: N21

Best next run: **N21 = N17/N14 core with ID weight restored to 0.1, but ID loss gated more tightly**.

Script:

```text
serv_new_runs/start_ba_combo_idloss_t400_14k_vast_N21.sh
```

Key change vs N17:

```text
+model.id_loss_weight=0.1
+model.id_loss_max_timestep=400   # was 500 in N17/N20
trainer.epoch_len=1000
trainer.n_epochs=14
```

Why this over another lower-weight run:

- N20 shows `0.075` loses too much identity.
- N17 shows `0.1` learns strong identity but over-forces some late cases.
- Tightening the timestep gate keeps the per-step ID-loss strength but applies it less often and only
  when predicted x0 should be cleaner.
- This is a more targeted reduction of ID pressure than lowering the weight everywhere.

Keep:

- frozen CA: `train_branched_ca_lora=false`;
- `loss_kind=blended_masked`;
- `lambda_face=0.15`;
- `branched_attn_weight_mode=noise_and_ref`;
- `branched_attn_new_weight_kind=lora`;
- `ba_noise_lr_scale=0.1`;
- `lr_for_lora=1e-4`;
- `optimizer.weight_decay=1e-3`;
- `trainer.max_grad_norm=1.0`;
- RealVis validation;
- batch size 1;
- checkpoints every 1k.

Run:

```bash
cd /home/kolyangg/rsrch/diffusion_template
bash serv_new_runs/start_ba_combo_idloss_t400_14k_vast_N21.sh
```

After N21, full-validate:

```bash
BATCH_SIZE=4 bash serv_new_runs/run_full_validation_steps.sh ba_combo_idloss_t400_14k_N21 8000 10000 12000 14000
```

If N21 still has the same Rushing/Keanu placement issue, the next change should be code-level:
un-hardcode `POSE_ADAPT_RATIO=0.0` in `BranchedAttnProcessor` and test small pose blending
(`0.1` or `0.2`). CA training should remain a short diagnostic probe only, not the next main run.
