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

Important fixed choices for the primary, same-base run:

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

The primary one-GPU launcher is:

```text
jul_serv_runs/start_ba_NN2_ppr1_1gpu.sh
```

It validates with the same configured SDXL base used for training. A separate
launcher validates the same trained architecture with RealVisXL:

```text
jul_serv_runs/start_ba_NN2_ppr1_realvis_1gpu.sh
```

Both default to physical GPU 0. The primary launcher uses port 29620 and the
RealVis launcher uses port 29621. They prefer the PhotoMaker conda
environment and accept `PHOTOMAKER_ENV_BIN` when the environment is in a
machine-specific location. Both PPR launchers default to
`train_dataset_name=cosmic_large_neb`. This entry exactly mirrors
`cosmic_large_vast`, except its two `/workspace/datasets/...` roots are
`/home/niko/datasets/...`.

Launch from the repository:

```bash
bash jul_serv_runs/start_ba_NN2_ppr1_1gpu.sh
```

Launch with RealVis validation:

```bash
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_1gpu.sh
```

Select another physical GPU for either launcher:

```bash
CUDA_VISIBLE_DEVICES=1 \
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
   the expected validation base (`null` for the primary launcher or RealVisXL
   for the RealVis launcher);
6. inspect the step-zero panel before allowing training to continue.

The resolved configuration, selected site list, strict trainability manifest,
diagnostics, and first-prediction fingerprint are printed/saved by the existing
training and logging path.

## Startup compatibility fix

The first server launch exposed a mixed-processor compatibility bug in the
training installer. PPR1 deliberately patches only up-block self-attention
sites, so down/mid self-attention sites remain Diffusers
`AttnProcessor2_0` objects. Those objects are callable processors but are not
`torch.nn.Module` instances and do not expose `parameters()`. The installer
incorrectly called `parameters()` on every registered processor and failed
before optimizer creation.

The installer now enables parameters only for processor objects that are
PyTorch modules. Untouched Diffusers processors are skipped. This does not
change the PPR architecture, selected sites, or trainable-parameter manifest.
A regression test covers the intended mixed registry.

## Step-zero comparison with NN1a–f

The PPR step-zero panel is not expected to be pixel-identical to NN1a–f. This
is not a seed mismatch:

- the common launcher and `manual_val` both use seed `0`;
- sample ordering, prompts, reference images, bounding boxes, inference steps,
  guidance scale, PhotoMaker/BA switch steps, and RealVis validation base are
  shared;
- target diffusion noise and reference noise use per-sample seeded generators;
- reference VAE latents use the deterministic posterior mode.

The output-changing difference is the architecture being tested. NN1d uses
legacy reference-owned replacement self-attention at all 70 SA sites. PPR1
uses ordinary target self-attention plus an exactly zero-initialized residual
at 36 up-block SA sites. Once BA starts at denoising step 15, NN1d changes the
target face even before training, while PPR1 intentionally remains on its
ordinary target path. Making those panels identical would require restoring
the legacy replacement operator and would invalidate PPR1.

For reproducibility and easier log auditing, the common launcher now passes
`trainer.seed=0` and `datasets.val.manual_val.seeds=[0]` explicitly. The
RealVis launcher is also accepted by PPR preflight when strict processor
restore and processor-state transfer are enabled. Under that path, a temporary
RealVis model constructs its own base projections and receives only the
trainable PPR adapter/connector state; frozen SDXL base projections are not
transplanted.

The valid comparisons are therefore:

1. PPR1 step zero versus a same-base ordinary-PhotoMaker control, to verify
   branch-off parity;
2. PPR1 versus NN1d on the same fixed inputs, to measure the effect of replacing
   legacy absolute BA with the new residual topology;
3. PPR1 checkpoints versus PPR1 step zero, to measure learning within the new
   architecture.
