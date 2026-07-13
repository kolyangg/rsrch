# N23/N24 Full-Validation Analysis and Next Runs

Date: 13 July 2026

## Scope and Artifacts

This analysis uses all 96 matched full-validation images at 1k, 5k, and 10k for both runs:

- N23: `full_validation_results/ba_camix_train_N23_steps/`
- N24: `full_validation_results/ba_dualgate_train_N24_steps/`
- PhotoMaker V2 baseline: `full_validation_results/photomaker_baseline/`
- Plain N17 26k: `full_validation_results/ba_longrun_N17/`
- N17 with inference-only CAMIX: `full_validation_results/pose0_camix_ablation/`

Generated comparison reports:

1. [N23/N24 steps vs PhotoMaker and N17](../full_validation_results/ba_n23_n24_13Jul/full_val_report_N23_N24_steps_vs_PM_N17.pdf)
2. [Focused N23 vs PhotoMaker and N17 CAMIX](../full_validation_results/ba_n23_n24_13Jul/full_val_report_N23_vs_PM_CAMIX.pdf)

Each PDF contains a summary page, a full config-difference table, and one complete image-grid page
per identity. The report generator matched 96/96 images for every column, including filenames with
spaces.

## Executive Conclusion

N23 is the better-looking result of the two, but mainly because it stays very close to PhotoMaker.
It fixes the severe N17 face-placement failures, yet does not establish a clear, repeatable BA gain
over PhotoMaker. Training from 1k to 10k barely changes either aggregate identity or composition.

N24 avoids the exact fixed-token CAMIX mechanism and changes faces more substantially, but its
dual-attention output blend is not yet a good fusion mechanism. It often damages identity or local
face coherence while retaining the PhotoMaker scene. N24 improves from 5k to 10k, but the recovery
is non-monotonic and concentrated in several large outliers rather than broad, stable progress.

Do not extend either recipe unchanged. The next pair should return to legacy BA, isolate the staged
training schedule, and then test only a tightly constrained reference-side CA update.

## Metric Summary

All runs detected a face in 96/96 outputs. Identity similarity is useful for trend detection, but
the visual conclusions below take priority because goggles, hands, chef hats, small faces, and
expression changes can move the recognizer score sharply.

| Run | Step | Mean ID sim | Eddie | Elon | Jennie | Jensen | Jisoo | Keanu | Lex | Marion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PhotoMaker V2 | 0 | **0.4886** | 0.205 | 0.546 | 0.590 | 0.557 | 0.577 | 0.500 | 0.473 | 0.461 |
| N17 plain BA | 26k | 0.3482 | 0.119 | 0.499 | 0.436 | 0.445 | 0.242 | 0.395 | 0.379 | 0.271 |
| N23 CAMIX | 1k | 0.4677 | 0.217 | 0.529 | 0.569 | 0.534 | 0.484 | 0.481 | 0.487 | 0.441 |
| N23 CAMIX | 5k | 0.4651 | 0.188 | 0.536 | 0.551 | 0.522 | 0.518 | 0.479 | 0.469 | 0.456 |
| N23 CAMIX | 10k | 0.4653 | 0.188 | 0.518 | 0.561 | 0.506 | 0.496 | 0.502 | 0.475 | 0.474 |
| N24 dual gate | 1k | 0.3693 | 0.134 | 0.484 | 0.487 | 0.446 | 0.354 | 0.380 | 0.386 | 0.283 |
| N24 dual gate | 5k | 0.3565 | 0.142 | 0.464 | 0.421 | 0.426 | 0.287 | 0.390 | 0.380 | 0.343 |
| N24 dual gate | 10k | 0.3899 | 0.151 | 0.485 | 0.456 | 0.455 | 0.373 | 0.436 | 0.389 | 0.376 |

Important trends:

- N23 changes only `-0.0024` from 1k to 10k. There is no aggregate learning trend after 1k.
- N23-10k is `-0.0233` below PhotoMaker overall. It only exceeds PhotoMaker for Keanu, Lex, and
  Marion, and those margins are small: `+0.003`, `+0.003`, and `+0.013`.
- N24 drops at 5k and then recovers to `0.3899` at 10k. That is `+0.0206` over N24-1k, but remains
  `-0.0754` below N23-10k and is lower for every identity.
- N24-10k does beat plain N17-26k in aggregate and notably for Jisoo, Keanu, and Marion. This means
  access to current-generation geometry/staged training has value, but the gate is not preserving
  identity cleanly.

## Is N23 Mostly PhotoMaker?

Yes. It is not byte-identical to PhotoMaker, but it is much closer to PhotoMaker than a useful new
BA regime should be.

Same-seed images were resized to 48x48 and compared directly. This deliberately measures broad
layout/color/pose similarity rather than face recognition.

| Pair | Mean absolute RGB difference | Mean pixel correlation |
|---|---:|---:|
| PhotoMaker vs N23-1k | 0.0262 | 0.9770 |
| PhotoMaker vs N23-5k | 0.0252 | **0.9790** |
| PhotoMaker vs N23-10k | 0.0258 | 0.9779 |
| N23-1k vs N23-5k | 0.0121 | 0.9938 |
| N23-5k vs N23-10k | 0.0108 | 0.9948 |
| N23-1k vs N23-10k | 0.0122 | 0.9940 |

The strongest near-copies include jumping scenes, night ride, and the rushing-woman scenes.
Differences are most visible in faces for laughing, dancing, kickboxing, and some skiing outputs.
This is expected from the code: CAMIX concatenates reference-face K/V with K/V from the current
generated face. The latter already carries PhotoMaker conditioning and the correct target pose.
Training can minimize denoising loss by relying heavily on that current-face source, so the BA
reference source becomes a local perturbation rather than the dominant identity mechanism.

What N23 improves relative to PhotoMaker:

- It retains PhotoMaker's correct Keanu subway face position instead of N17's long-neck/high-face
  failure.
- Marion improves modestly overall, including some laughing, skiing, and chef cases.
- Several difficult chef outputs improve numerically, especially Lex and Jisoo, although chef faces
  remain small/occluded enough that these scores should not drive the decision.
- Some expression-heavy Keanu/Elon images gain identity without visible scene damage.

What N23 does not improve reliably:

- The 1k, 5k, and 10k columns are almost the same composition and remain close to PhotoMaker.
- Jisoo is `-0.081` below PhotoMaker overall at 10k, with a large skiing regression.
- Jumping Elon/Lex, night-ride Jensen, and several laughing/dancing cases regress while the scene
  remains essentially the PhotoMaker scene.
- PhotoMaker weaknesses remain: tiny distant jumping faces, goggles obscuring skiing identity,
  poor chef identity, and occasional expression-driven identity drift.
- There is no evidence that another 5k-10k of the same training would unlock a stronger BA effect.

Conclusion: N23 verifies that access to the current generated face solves placement, but fixed
CAMIX over-corrects toward PhotoMaker and largely bypasses the intended reference-driven BA gain.

## N24 Detailed Assessment

N24 uses separate attention against reference-face K/V and current-generation-face K/V, then blends
the two attention outputs with a learned per-head gate. The configured current-face contribution
starts at 0.20 and is capped at 0.50.

What works:

- Global pose, clothing, and background usually remain coherent and PhotoMaker-like.
- The Keanu subway placement problem is avoided.
- N24-10k recovers strongly from 5k for Jisoo kickboxing, Marion crying/reading/jumping, and Keanu
  crying.
- N24-10k is a meaningful improvement over plain N17 for identities that suffered N17's worst
  placement/appearance failures.
- The 1k-to-10k images differ more than N23's, showing that the gate path is not completely ignored.

What does not work:

- Face identity is lower than N23 for all eight identities at 10k.
- Local faces often look averaged or inconsistent with the surrounding head/hair. This is visible
  in Jisoo and Marion occlusion/expression cases and several skiing/goggle cases.
- Training is non-monotonic: 5k is worse than 1k overall. A robust fusion mechanism should not
  require recovering from such a broad mid-run drop.
- Much of the 10k metric gain comes from a handful of large reversals. For example, Jisoo
  kickboxing moves from `-0.026` at 5k to `0.433` at 10k. This is useful recovery, but not evidence
  of uniformly better face fusion.
- Scene similarity to PhotoMaker remains high (`0.9718` mean low-resolution correlation at 10k),
  while the face itself is often worse. The mechanism is changing the local result without
  improving the global generation contract.

Gate saturation cannot be established from result images alone because the N24 checkpoint/gate
parameters were not copied into this local result folder. If the server checkpoint still exists,
retain it for a later per-layer/per-head gate histogram. The image evidence is sufficient to reject
an unchanged longer N24 run regardless of whether the gates saturated.

## What to Keep and What to Stop

| Mechanism | Decision | Reason |
|---|---|---|
| PhotoMaker-first inference stages | Keep | They provide correct target pose/layout and prevent N17-style face placement failure. |
| Staged BA training (`train_ba_all_steps=false`) | Isolate next | N24 used it but confounded it with the new gate; it remains the cleanest unresolved schedule fix. |
| N23 fixed CAMIX | Stop as primary path | It largely reproduces PhotoMaker and is flat after 1k. |
| N24 dual-attention gate | Stop unchanged | It changes faces but does not preserve identity/coherence reliably. |
| Legacy reference-driven BA | Restore as anchor | It is the least confounded way to test whether staging fixes placement without PM overshadowing. |
| Broad CA training | Do not use | Earlier runs showed drift and it roughly doubles trainable processor capacity. |
| Reference-only CA at low LR | Test in one arm | It may improve prompt/pose integration while excluding the known CA noise drift channel. |
| Alternating loss | Defer | Previous N4-style evidence showed higher early peaks but worse drift; changing loss now would confound the schedule/CA test. |
| Extra ID-embedding injection | Defer | N23 already shows the risk of becoming too PhotoMaker-like; test it only after a stable non-CAMIX anchor exists. |

## Recommended Next Runs

The pair is intentionally controlled. Both use the same stable N17 objective and legacy face
attention, but match training to the staged inference schedule. N26 changes only CA trainability.

| Setting | N25 staged legacy | N26 staged + CA-ref |
|---|---|---|
| Run name | `ba_staged_legacy_N25` | `ba_staged_caref_N26` |
| Config | `one_id_ba_staged_legacy_N25` | `one_id_ba_staged_caref_N26` |
| Face fusion | legacy reference BA | legacy reference BA |
| CAMIX / PAR | off / 0 | off / 0 |
| Train BA at all timesteps | **false** | **false** |
| Branched CA trainable | no | yes, **ref_only** |
| CA LR | n/a | `0.1 x 1e-4 = 1e-5` |
| CA noise path | frozen | frozen |
| SA ref/noise | train / train at 0.1x | same |
| Loss | blended masked, lambda 0.15 | same |
| Explicit ID loss | weight 0.1, t<=500 | same |
| Length | 10k | 10k |

### Why N25

N25 is the clean schedule experiment that N24 could not answer. It removes both PM-like CAMIX and
the unstable output gate while preserving the proven frozen-CA, damped-noise, blended-loss recipe.
If it keeps the N17 identity contribution but fixes the Keanu/Jisoo placement failures, the main
problem was the all-timestep training mismatch rather than source fusion.

### Why N26

N26 tests the user's trainable-CA hypothesis with the risky part removed. Only CA `ref_to_*`
parameters train, and the whole CA group runs at 0.1x LR. CA `noise_to_*` stays frozen, so it cannot
reopen the strongest previously observed whole-image drift channel. Comparing N26 directly with
N25 reveals whether constrained CA adds useful prompt/pose adaptation.

## Training and Validation Schedule

Run both for 10 epochs x 1,000 steps = 10k total. Do not plan an automatic extension.

- Built-in 24-image validation and checkpoint: every 1k steps.
- Full 96-image validation: 1k, 3k, 5k, 7k, and 10k.
- Inspect the fixed difficult cases at every full validation: Keanu rushing, Jisoo skiing and
  kickboxing, Marion crying/jumping, Jensen night ride, all chef prompts, and distant jumping faces.
- Stop N26 early if face/background drift worsens in two consecutive checkpoints; do not wait for
  a mean-ID recovery.
- Select the winning checkpoint visually first, then use mean/per-identity ID similarity as a
  secondary check.

## Parallel Server Launch

The launchers already detach themselves through `nohup`, so these commands survive SSH disconnects.
Distinct master ports are also configured to avoid an Accelerate rendezvous collision.

```bash
cd /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template
conda activate photomaker_NS
export COMET_API_KEY="..."

CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29525 \
  bash serv_new_runs/start_ba_staged_legacy_serv_N25.sh

CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29526 \
  bash serv_new_runs/start_ba_staged_caref_serv_N26.sh
```

The scripts print the detached PID and exact log path. Monitor with:

```bash
watch -n 2 nvidia-smi
tail -f "$(ls -t logs_new_runs/ba_staged_legacy_N25_*.log | head -1)"
tail -f "$(ls -t logs_new_runs/ba_staged_caref_N26_*.log | head -1)"
```

After checkpoints are available, run full validation on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=4 \
RESULTS_DIR=full_validation_results/ba_staged_legacy_N25_steps \
nohup bash serv_new_runs/run_full_validation_steps.sh \
  ba_staged_legacy_N25 1000 3000 5000 7000 10000 \
  > logs_new_runs/ba_staged_legacy_N25_fullval.log 2>&1 </dev/null &

CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=4 \
RESULTS_DIR=full_validation_results/ba_staged_caref_N26_steps \
nohup bash serv_new_runs/run_full_validation_steps.sh \
  ba_staged_caref_N26 1000 3000 5000 7000 10000 \
  > logs_new_runs/ba_staged_caref_N26_fullval.log 2>&1 </dev/null &
```

## Files Added

- `infer_tools/full_val_n23_n24_13jul_report.yaml`
- `infer_tools/full_val_n23_vs_pm_13jul_report.yaml`
- `src/configs/one_id_ba_staged_legacy_N25.yaml`
- `src/configs/one_id_ba_staged_caref_N26.yaml`
- `serv_new_runs/start_ba_staged_legacy_serv_N25.sh`
- `serv_new_runs/start_ba_staged_caref_serv_N26.sh`

No active model/trainer behavior was changed for these runs. All new behavior is selected through
new Hydra configs, and existing run reproduction remains unchanged.
