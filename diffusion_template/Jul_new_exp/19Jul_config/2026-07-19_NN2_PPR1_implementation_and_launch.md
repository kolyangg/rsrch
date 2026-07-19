# NN2-PPR1 implementation and launch

Date: 19 July 2026

## Outcome

NN2-PPR1 is implemented as an opt-in processor variant. Existing NN1/NN2
behavior remains the default through:

```yaml
model:
  ba_processor_variant: legacy
  ba_site_policy: all
```

The new run selects:

```yaml
model:
  ba_processor_variant: packed_residual_v1
  ba_site_policy: up_blocks_attn1
```

No existing run config or launcher was changed.

## Architecture implemented

At the selected SDXL up-block self-attention sites, the processor:

1. computes ordinary self-attention once on the full doubled
   `[target, reference]` batch;
2. keeps the target and reference base continuations unchanged;
3. packs only hard-valid reference-bbox tokens and masks padding with additive
   `-inf`;
4. reuses the ordinary target query to compute a reference retrieval candidate;
5. forms `reference_candidate - target_base`;
6. passes that difference through a rank-16 connector whose up projection is
   initialized to exactly zero;
7. applies a scalar gate bounded by `0.5` and a per-sample FP32 RMS cap of
   `0.25`;
8. adds the result only through a cosine-feathered inner target-face core.

Consequently, step-zero output is exactly the ordinary self-attention output.
Reference evidence is additive and bounded; it is never the target face’s only
self-attention candidate.

Reference K/V retrieval uses rank-32 LoRA. The reference continuation uses the
frozen ordinary U-Net Q/K/V path, so retrieval specialization cannot alter the
memory stream passed to later layers.

The cached SDXL registry contains 70 self-attention and 70 cross-attention
sites. The explicit PPR1 policy selects 36 up-block self-attention sites and
retains all 70 existing split cross-attention processors. Cross-attention
parameters are frozen.

## Configuration

The experiment config is:

```text
src/configs/one_id_ba_NN2_ppr1.yaml
```

Important fixed choices:

- one SDXL base for training and validation;
- no alternate RealVis validation model;
- active but frozen split cross-attention;
- pose adaptation off;
- CA face mixing off;
- decoded identity loss off;
- packed reference bbox ROI;
- target inner-core output mask with 10% cosine erosion;
- BA-active inference timestep sampling;
- blended diffusion loss with `lambda_face=0.20`;
- existing cross-image `CosmicLargeTrain` sampler;
- fixed 96-image validation at step 0 and every 2,000 optimizer steps.

The default screening budget is 6,000 optimizer steps: three epochs of 2,000
steps. Extend the same checkpoint only if the 6k panel is anatomically clean
and still improving.

## Trainable state and checkpoint guards

At every selected self-attention site, only these seven tensors are trainable:

```text
ref_to_k.lora_A
ref_to_k.lora_B
ref_to_v.lora_A
ref_to_v.lora_B
connector_down.weight
connector_up.weight
gate_logit
```

Strict startup guards verify exact processor names and trainable keys. Strict
checkpoint manifests now include the processor variant, site policy, PPR
connector/gate/cap settings, processor classes, and trainable keys. Loading a
checkpoint with a different topology fails rather than silently applying a
partial state.

The optimizer keeps one common LR/weight-decay recipe but exposes separate
logging groups for reference K, reference V, connector-down, connector-up, and
gate gradients.

Repeated mask updates preserve processor object identities; they do not rebuild
modules or detach the optimizer from live parameters.

## Launcher

The one-GPU launcher is:

```text
jul_serv_runs/start_ba_NN2_ppr1_1gpu.sh
```

It defaults to physical GPU 1 and port 29620. It prefers the PhotoMaker conda
environment and accepts `PHOTOMAKER_ENV_BIN` when the environment is in a
machine-specific location.

Launch from the repository:

```bash
bash jul_serv_runs/start_ba_NN2_ppr1_1gpu.sh
```

Select another GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash jul_serv_runs/start_ba_NN2_ppr1_1gpu.sh
```

The launcher runs detached by default and prints the PID and log path. Use
`RUN_FOREGROUND=1` for an interactive preflight.

## Checks completed locally

The implementation was checked with the repository’s `photomaker` conda
environment (`diffusers 0.29.1`, `torch 2.7.1+cu126`):

- Python compilation for every modified module;
- shell syntax and executable permission;
- Hydra composition and resolved PPR1 invariants;
- cached SDXL attention registry: 70 SA, 70 CA, 36 selected PPR SA;
- exact FP32 branch-off parity for 3D/4D inputs and batches 1/2;
- BF16 branch-off parity;
- valid-token packing and additive padding exclusion;
- mixed valid/empty ROI behavior;
- zero residual for an empty ROI;
- expected first-step and second-step gradient staging;
- per-sample RMS cap;
- retrieval/reference-continuation isolation;
- feathered inner-core mask semantics;
- repeated-patch processor identity preservation;
- exact trainability manifest with zero trainable CA parameters.

Run the quick suite with:

```bash
PYTHONPATH=. \
PYTHONWARNINGS='ignore::FutureWarning' \
/home/kolyangg/anaconda3/envs/photomaker/bin/python \
tests/test_packed_residual_attn_processor.py -v
```

Run the full 8/16/32/64 FP32/BF16 parity matrix on a GPU machine with:

```bash
PPR_RUN_FULL_PARITY=1 \
PYTHONPATH=. \
python tests/test_packed_residual_attn_processor.py -v
```

## Required server preflight

The local machine does not mount the server CosmicLarge JSON/images or the
server PhotoMaker checkpoint. Before committing a long run on the GPU machine:

1. perform the handoff’s read-only 2,000-pair data audit against the exact
   Hydra-resolved CosmicLarge paths;
2. run one real two-sample forward/backward/optimizer step;
3. verify finite losses and the expected staged gradients;
4. run the fixed 96-image step-zero validation;
5. verify the log reports `variant=packed_residual_v1`, `SA=36`, `CA=70`, and
   no alternate validation base;
6. inspect the step-zero panel before allowing training to continue.

The resolved configuration, selected site list, strict trainability manifest,
diagnostics, and first-prediction fingerprint are printed/saved by the existing
training and logging path.
