# CL39 clean extension

> **Source-layout update, 22 August 2026:** the null-key route is implemented
> in `hardcase_attn_processor.py`; the shared June attention file is unchanged.
> See the [implementation map](../../analysis/2026-08-22_june2_key_file_rebase_implementation.md).

**Date:** 21 August 2026

**Branch:** `clean`

**Parent:** `CL27_cosmic_frequency_surface_energy_24k`

**Test provenance:** sealed source revision
`cl38-cl45-v5-frequencycollector-scope-20260820`; CL39 r4 Comet key
`b1ca0b3da679401c85b991f1bbdf0b2a`

## Outcome

CL39 is a parameter-free child of CL27. In `up_blocks.0/1`, it measures the
normalized entropy of each target query's reference-key attention and scales
CL27's routed reference delta by a bounded detached confidence. High-entropy
queries retain more native target self-attention, which is intended for
target-owned details with no clear reference match. All other blocks remain
exact CL27.

No training or Serv job was launched by this clean port. The cited test run is
source/smoke provenance, not a clean-branch quality result.

## Exact delta

For per-head reference probabilities (p_{hij}), the mean normalized entropy,
virtual-null mass, and reference confidence are

$$
e_i=\frac1H\sum_h\frac{-\sum_j p_{hij}\log(p_{hij}+10^{-8})}{\log L},
\quad
n_i=\sigma\left(\frac{e_i-0.75}{0.08}\right),
\quad
c_i=\operatorname{clip}(1-0.75n_i,0.25,1).
$$

Selected blocks replace CL27's (Y=N+\Delta) with

$$
Y=N+c\odot\Delta.
$$

Confidence is computed under `no_grad`; CL39 adds no predictor, parameter, or
optimizer role. The trainable contract remains exactly 2,240 tensors and
219,217,920 parameters. The leaf keeps `pose_adapt_ratio=0`,
`ca_mixing_for_face=false`, SA-only BA, subject-v2 fixed-96 validation, and
CL27's deterministic 25% semantic-occluder supervision.

## What the native fallback is—and is not

`native` means target Q attending target K/V in the current self-attention
layer. It is always CL39's base message; confidence scales only the added
CL27 reference-minus-native correction. Outside the target mask the router is
zero and the block is exactly target-only. Inside `up_blocks.0/1`, confidence
has a declared lower bound of 0.25 and is approximately 0.282 even at maximum
normalized entropy, so CL39 does not completely turn off the reference
correction in the face interior. `[code]`

This lane is not a call to frozen, unmodified PhotoMaker. Its target-only
operator has the ordinary self-attention form, but it uses CL39's trained
rank-128 target Q/K/V LoRA and trained output adapters, and its input already
contains upstream BA and PhotoMaker cross-attention effects. The fixed
generation schedule separately has a true unbranched PhotoMaker interval:
steps 10–14 use the original attention processors, while steps 15–49 run
PhotoMaker and BA together. Training uses BA at every sampled timestep.

The test-source run identified above logged mean
`ba/null_key/reference_fraction/all` between `0.2998` and `0.3693` through
optimizer step `23,950`, with `0.3183` latest. Thus attenuation was active in
training. `[measured]` This number is the mean correction multiplier, not the
fraction of the final activation attributable to the reference, and it
includes background queries where the target-mask router is zero. Face-only
usage and quality improvement over CL27 are not established by this metric.
See Section 14 of
[`2026-08-13_e13_family_architecture_reference.md`](2026-08-13_e13_family_architecture_reference.md)
for the complete equations, schedule table, code quotation, and confidence
assessment.

## Minimal file map

| Area | Change |
|---|---|
| `attn_processor_cleanest.py` | Add detached entropy confidence and scale only CL27's low/high routed components. |
| `branched_runtime.py` | Install the router only in declared `up_blocks.0/1`. |
| `lora2.py`, `e13_contract.py` | Persist and fail closed on the six CL39 controls; checkpoint manifests record the output-affecting route. |
| `lora2_helpers.py` | Reuse CL27's single cached processor map to aggregate null-key metrics. |
| validation wrapper/trainer | Copy the same CL39 controls onto the RealVis validation pipeline. |
| config, validator, preflight, launcher | Add one leaf while retaining the existing CL27 data and runtime gates. |

## Fixed training pipeline

CL39 retains the 16 August optimized pipeline:

- Diffusers' recursive `unet.attn_processors` map is resolved once per active
  collector, never inside a per-layer loop;
- disabled full-activation BA telemetry remains off;
- no active-gradient metrics are requested, so no active-gradient scan runs;
- CL27's surface collector is installed only in its declared `up_blocks.0/1`;
- semantic masks are allocated only because the inherited CL27 objective
  consumes them.

The entropy logits are chunked by 256 target queries and detached. That work
is CL39's declared scientific computation, not telemetry overhead.

## Prepared Serv run

Config:
`src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml`

Prepared experiment record:
`experiments/cosmic_large/CL39_cosmic_null_key_confidence_router_24k_full96_clean_r1.json`

Prepared one-A100 run name:
`CL39_cosmic_null_key_confidence_router_24k_full96_clean_r1`

Submission command, only after explicit authorization and a live project-GPU
audit:

```bash
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_e13_family_clean/diffusion_template/serv_run_packages/CL39_cosmic_null_key_confidence_router_24k_full96_clean_r1/run_CL39_cosmic_null_key_confidence_router_24k_full96_clean_r1_1gpu.yaml
```

The YAML rejects a dirty/wrong branch and records the exact source commit.
The shared launcher then runs Hydra, architecture, dataset, CUDA ONNX Runtime,
trainable-ownership, fixed-validation, and immutable-Comet-startup gates.
