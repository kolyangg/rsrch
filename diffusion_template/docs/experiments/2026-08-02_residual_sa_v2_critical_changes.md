# Residual SA-v2 critical changes

**Date:** 2 August 2026  
**Status:** implemented and locally verified; Neb launchers prepared, no run launched by this change  
**Control:** `rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1`  
**New candidate:** `rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r1`

## Outcome

The new opt-in architecture keeps branched attention as the model's identity
path but changes the target self-attention merge from hard face replacement to
a bounded residual:

```text
target output = frozen native target SA
              + target-face mask
              * bounded layer/timestep/area gate
              * low-rank output(reference-attention(target Q, reference K/V))
```

Reference keys outside the reference face mask are excluded from softmax by an
additive attention bias. Target Q/K/V and native output projection remain
frozen. Reference K/V LoRA, a branch-local output LoRA, and gates are the only
trainables. The selected `mid_block`, `up_blocks.0`, and `up_blocks.1` set has
an exact expected contract of **414 tensors / 10,567,818 parameters**, all
FP32. Branched CA stays off, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`.

## Reversibility

Historical behavior remains the default:

- `model.ba_architecture_version=hard_replace_v1`
- `model.branched_trainable_dtype=inherit`
- `model.ba_training_timestep_policy=uniform_all`
- `model.ba_spatial_reference_shuffle_probability=0`

The old control launcher remains
`launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh`. The new
architecture is selected only by
`src/configs/big_celebs_scheduled_rhca_residual_sa_v2_40k.yaml` and
`launchers/neb/start_rhca_big_celebs_scheduled_residual_sa_v2_40k.sh`.
Independent toggles cover reference-K/V rank, output rank, semantic layer
groups, gate initialization/maximum/timestep/face-area terms, trainable dtype,
training timestep policy, and spatial-reference shuffle probability.

## Training and checkpoint changes

- Inference-active training samples the actual 50-step DDIM timesteps at which
  BA is enabled, independently per sample. Historical configurations retain
  their repeated scalar timestep.
- Optimizer groups come from processor-declared roles: reference K/V,
  reference output, and gate. Exact inclusion and exclusion are checked before
  Accelerate wraps the model.
- Schema-v2 checkpoints save the exact trainable names, values, shapes, dtypes,
  processor set, routing semantics, ranks, gates, timetable policy, and
  architecture version. Loading fails on any manifest or tensor mismatch.
- The new `branched_reference` loss combines full-image, face, and boundary-ring
  denoising losses. Its wrong-spatial-reference forward is detached and has
  zero auxiliary weight in the prepared run, so it is a causal diagnostic and
  does not yet change the objective.

## Prepared Neb scripts

### New training candidate

```bash
bash launchers/neb/start_rhca_big_celebs_scheduled_residual_sa_v2_40k.sh
```

Default run name:
`rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r1`.
Experiment record:
`experiments/big_celebs/rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r1.json`.
The launcher delegates the existing sealed dataset, sampling-plan, full-96,
ONNX CUDA, and immutable-Comet preflights.

### D0 clean-32k validation matrix

```bash
bash launchers/neb/run_clean_ba32_32k_d0_validation_matrix.sh
```

It evaluates `weights-epoch16.pth` from the immutable clean run at step 32k in
four output arms:

1. `d0_clean_ba32_32k_legacy_matched`
2. `d0_clean_ba32_32k_native_matched`
3. `d0_clean_ba32_32k_native_zero_spatial`
4. `d0_clean_ba32_32k_native_shuffle_spatial`

All four use the fixed 96-image `manual_val` panel in eight identity-homogeneous
batches of 12. The spatial interventions leave matched PhotoMaker input images
and identity tokens unchanged. The zero arm removes the reference latent and
its reference noise at the BA input while retaining the matched spatial mask;
it should be interpreted as a zero-input causal diagnostic, not as removal of
the doubled reference lane. Each arm writes generated images, per-image hashes,
identity/text metrics, resolved config, processor/checkpoint audit, and the
four standard no-reference IQA families. Each arm also opens a separate Comet
experiment in `jul-comet-large-testing-tr`, writes its immutable key to
`saved/<arm>/comet_experiment.json`, and publishes the metrics, 96 images, and
audit assets. The matrix record is
`experiments/big_celebs/d0_clean_ba32_32k_validation_matrix.json`.

## Verification completed locally

Using `/home/kolyangg/anaconda3/envs/photomaker`:

- old and new Hydra configurations compose with the expected toggles;
- changed Python files compile, both launchers pass `bash -n`, and both JSON
  records parse;
- zero-initialized SA-v2 matches frozen target native SA exactly;
- gradients reach the output adapter on update one and reference K/V plus gates
  after the output path becomes nonzero;
- masked-out reference tokens have exactly zero effect, while a zero valid-key
  mask fails closed;
- all declared trainables are FP32;
- the 46-site ownership contract is exactly 414 tensors / 10,567,818
  parameters;
- the fixed-96 validation dataset is eight ordered, identity-homogeneous
  12-image batches with complete reference bboxes.

No code was synchronized to Neb and neither prepared experiment was launched.
Neb must be checked for an active GPU process immediately before using either
script; the D0 launcher fails closed on an occupied GPU by default.
