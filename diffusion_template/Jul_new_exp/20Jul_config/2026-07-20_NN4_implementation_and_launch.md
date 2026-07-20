# NN4 implementation and launch handoff

## Purpose

NN4 implements the minimal correctness-and-causality training run recommended in
`2026_07_20_photomaker_branched_attention_code_and_literature_review.md`. It
keeps the core PPR branched-attention hypothesis:

```text
ordinary target self-attention
+ bounded Connector(C_matched_reference - C_learned_null)
  inside the feathered target face core
```

The ordinary PhotoMaker epsilon prediction remains the output anchor outside
the face core. Pose adaptation and CA face mixing remain disabled.

## Configuration

Hydra config:

```text
src/configs/one_id_ba_NN4_causal_null_up0.yaml
```

NN4 inherits NN3b's:

- packed target-query/reference-KV PPR processor;
- learned null memory;
- zero-initialized bias-free connector;
- training-only target PhotoMaker-ID attenuation on half of a physical batch;
- low-timestep identity loss;
- exact PhotoMaker output anchor outside the feathered core.

NN4 changes:

1. **Correct CFG reference noise.** Reference noise is cached at output-image
   batch size and duplicated exactly into CFG's unconditional and conditional
   copies. The legacy behavior remains available with
   `model.ba_cfg_reference_noise_pairing=false`.
2. **Isolate reference text semantics.** Reference token embeddings and SDXL
   pooled text embeddings are zeroed, while target PhotoMaker conditioning is
   unchanged. Old behavior is restored with
   `ba_reference_token_text_mode=original` and
   `ba_reference_pooled_text_mode=target`.
3. **Disable split branched cross-attention.** `disable_branched_ca=true`
   restores the standard cross-attention processors. This flag is now also
   propagated correctly to alternate-base validation models/pipelines.
4. **Restrict spatial authority to `up_blocks.0`.**
   `ba_site_policy=up_blocks0_attn1` leaves all other self-attention sites
   ordinary.
5. **Use core-normalized diffusion MSE.** Every sample is normalized by its
   feathered face-core area.
6. **Add candidate-level matched/null supervision.** The learned null candidate
   passes through the same target-query and reference K/V projection route as
   the matched candidate. NN4 penalizes its connector response, adds a small
   matched/null separation margin, and penalizes pre-cap norm excess.
7. **Lower the residual cap.** `ba_delta_rms_cap` is reduced from `0.25` to
   `0.15`; the supervised pre-cap target is `0.12`.

## Post-audit corrections

The follow-up NN4 code audit identified and corrected three objective-safety
issues before training:

1. The matched/null margin now uses the main connector response
   `D(C_ref - C_null)` directly. The previous expression subtracted
   `D(C_null)` a second time and therefore measured
   `D(C_ref) - 2*D(C_null)`.
2. Auxiliary losses include a sample only when both its packed reference ROI
   and its target face core have support.
3. An empty feathered target core is rejected during sample preparation through
   the existing DDP-safe invalid-batch path. The core-normalized criterion also
   rejects empty rows as a defensive invariant.

The auxiliary objective is deliberately a low-memory candidate-level paired
screen inside each attention processor. It is not yet a second full U-Net pass
with a separately encoded null reference image. That larger experiment should
only follow if NN4 establishes measurable matched-reference identity direction.

## Local one-GPU server

Launcher:

```text
jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh
```

Defaults:

- GPU `0`;
- `cosmic_large_neb`;
- physical/effective batch `2`;
- 20,000 optimizer steps (`10 × 2,000`);
- 96-image RealVis validation at step 0 and every 2,000 steps;
- validation batch `12`;
- PhotoMaker conda environment selected by the shared launcher.

The launcher checks that `NUM_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH == 20000`.
Shorter training invocations are intentionally rejected; 2k/4k are evaluation
checkpoints within the 20k run rather than separate training budgets.

Run:

```bash
bash jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh
```

Foreground/debug run:

```bash
RUN_FOREGROUND=1 bash jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh
```

## NFS/MLS server

The shared server launcher uses the same NFS paths, `cosmic_large` dataset
definition, PhotoMaker conda environment, RealVis validation, and batch settings
as NN3b.

Submit one GPU:

```bash
mls job submit --config ./serv_new_runs/run_ba_NN4_causal_null_up0_1gpu.yaml
```

Submit two GPUs:

```bash
mls job submit --config ./serv_new_runs/run_ba_NN4_causal_null_up0_2gpu.yaml
```

Underlying launchers:

```text
serv_new_runs/_start_ba_NN4_server_common.sh
serv_new_runs/start_ba_NN4_causal_null_up0_1gpu.sh
serv_new_runs/start_ba_NN4_causal_null_up0_2gpu.sh
```

Per-rank train batch is `2`, so the default global batch is `2` on one GPU and
`4` on two GPUs. No gradient accumulation is enabled.

The NFS launcher enforces the same 20,000-step total. Validation defaults to
RealVis. For a same-SDXL-base checkpoint validation without editing scripts,
set:

```bash
NN4_VALIDATION_MODEL=null
```

This changes only the validation base; it does not change the training base or
the 20k training schedule.

## 2k/4k causal checkpoint gate

Training still runs with a 20k budget and saves/validates every 2k. Use the
validation-only helper on the epoch-1 (2k) and epoch-2 (4k) checkpoints to run
the fixed-target five-way `PM0/R1N1/R2N1/R1N2/R2N2` reference-versus-noise
matrix:

```bash
bash jul_serv_runs/start_ba_NN4_checkpoint_reference_vs_noise_1gpu.sh \
  /absolute/path/to/checkpoint-epoch2.pth
```

The checkpoint epoch is inferred from the filename. Defaults are 96 samples,
validation batch 12, and RealVis. For the recommended same-base companion:

```bash
NN4_VALIDATION_MODEL=null \
  bash jul_serv_runs/start_ba_NN4_checkpoint_reference_vs_noise_1gpu.sh \
  /absolute/path/to/checkpoint-epoch2.pth
```

This helper is inference-only and does not shorten or resume the training run.

## Metrics to watch

In addition to the existing diffusion, identity, and PPR diagnostics:

```text
ba_aux/null_residual
ba_aux/null_residual_weighted
ba_aux/match_null_margin
ba_aux/match_null_margin_weighted
ba_aux/cap_excess
ba_aux/cap_excess_weighted
```

The key visual/causal stop criterion remains matched-versus-wrong reference
identity direction at fixed target seed—not simply larger face differences.

## Validation completed

- Hydra NN4 config resolves successfully.
- Python compilation passes in the PhotoMaker conda environment.
- Shell syntax checks pass for all launchers.
- The packed-residual regression suite passes: 35 tests, one optional full
  parity matrix skipped.
- Added tests cover CFG reference-noise equality, reference token/pooled-text
  isolation, `up_blocks.0` selection, core-normalized loss, the corrected
  matched/null algebra, empty-core handling, and auxiliary-loss gradients.
