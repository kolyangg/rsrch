# Anchored interpolation BA-v3 implementation

**Date:** 2 August 2026  
**Status:** implemented and locally verified; not synchronized to or launched on Neb

## Outcome

`anchored_mix_sa_v3` is a new defaults-off self-attention architecture. It
retains target queries and native target self-attention, attends explicit
reference K/V with a true reference-face key mask, and interpolates only
inside the target face:

```text
reference = frozen_native_to_out(reference_attention)
          + zero_initialized_trainable_output_delta(reference_attention)

target = native
       + target_mask * alpha * (RMSMatch(reference, native) - native)
```

The first controlled settings are `alpha_init=.50`, `alpha_floor=.25`, and
`alpha_max=.90`. Thus the reference path is nonzero at initialization and
cannot collapse to plain PhotoMaker. `model.ba_mix_override=0` remains an exact
native diagnostic; `=1` is the forced-reference endpoint. Historical
`hard_replace_v1` and `residual_sa_v2` selectors and checkpoint tensor names
are unchanged.

## Critical integration changes

- `anchored_mix_sa_processor_v3.py` implements frozen target Q, explicit
  reference K/V LoRA, true key masking, frozen native reference output,
  zero-initialized output LoRA, clipped detached RMS matching, bounded
  interpolation, and face-local merge.
- Strict ownership declares three roles: `ref_kv`, `ref_output`, and `mix`.
  The same 46 mid/up sites own exactly **414 tensors / 10,567,818 FP32
  parameters**. Schema-v2 checkpoints record processor code version 3,
  routing, merge equation, output-base mode, mix bounds, RMS controls,
  telemetry controls, and reference-loss mode. Cross-version or changed-mix
  manifests fail closed.
- Interval-sampled processor telemetry stores detached scalars only and logs
  actual mix, reference/native RMS, contribution/native RMS, reference-key
  fraction, and denoising progress by `mid`, `up0`, and `up1`. The shuffled
  counterfactual explicitly suppresses collection, so it cannot overwrite the
  matched-forward telemetry.
- Reference-noise is passed explicitly from the correct forward to the
  counterfactual forward. Only reference content and its bbox mask are
  permuted.
- `detached_diagnostic` preserves the v2 no-grad wrong-reference behavior.
  The reversible `differentiable_rank` mode keeps both predictions in the
  graph and optimizes a relative face-error margin. Model and loss modes must
  match or startup fails.
- Training and alternate-base validation propagate the exact versioned
  runtime fields. `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and
  branched CA-off remain mandatory.

## Prepared E3 run

Run on an idle Neb GPU only:

```bash
cd /home/niko/rsrch/diffusion_template
bash launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_2k.sh
```

The launcher fails if `nvidia-smi` reports another compute PID, then delegates
the established sealed-dataset, pinned-schedule, ONNX CUDA, fixed-96, Comet,
and integrity preflights. It forces `TRAIN_EPOCH_LEN=2000` and
`TRAIN_EPOCHS=1`; this is a 2k E3 gate, not another unchecked 40k run.

- Run name:
  `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r1`
- Config:
  `src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_2k.yaml`
- Plan:
  `experiments/big_celebs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r1.json`
- Comet project: `jul-comet-large-testing-tr`
- Validation: fixed 96 at step 0 and step 2,000
- E3 objective: detached diagnostic, 25% cross-identity spatial shuffle,
  reference weight zero, so the architecture is isolated.

For the later E4 objective arm, keep the architecture/data fixed and override
both sides together:

```yaml
model:
  ba_reference_loss_mode: differentiable_rank
  ba_spatial_reference_shuffle_probability: 0.50
loss_function:
  reference_mode: differentiable_rank
  reference_weight: 0.10
  reference_relative_margin: 0.02
```

Use a new run name/spec for E4; do not reuse the E3 saved directory or Comet
identity.

## Local verification

The existing local `photomaker` environment passed:

- Python compilation, JSON parsing, shell syntax, and Hydra composition for
  hard-replace v1, residual v2, and anchored v3;
- exact alpha-zero native equality including residual connection and output
  rescaling;
- exact native behavior outside the target face at alpha one;
- zero target-lane influence from arbitrarily perturbed invalid reference
  keys, plus nonzero response to shuffled valid reference content;
- first-backward finite nonzero gradients for reference-K B, reference-V B,
  reference-output B, mix logit, timestep mix, and face-area mix;
- detached wrong-reference gradients remain absent, while differentiable mode
  gives nonzero gradients to both correct and wrong predictions;
- exact 414 / 10,567,818 optimizer membership, role counts 184 / 92 / 138,
  schema-v2 tensor round-trip, and rejection of a changed-mix manifest;
- residual-SA-v2 zero-initialization/native parity and unchanged gate tensor
  names.

These are local algebra and contract checks, not evidence of training quality.
The implementation has not yet produced a Neb Comet key, validation images,
or an observed 2k causal gap.
