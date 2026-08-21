# CL23 and CL27 clean extension

**Date:** 18 August 2026

**Branch:** `clean`

**Clean parent before this extension:** `fe39dc0cc72696a0adabc13a19d16a14d9cd88d1`

**Authoritative source inspected:** committed `test` snapshot `6eb6613`

## Outcome

This port adds only the production CL23 route and CL27 training objective to
the existing clean E13 family. It deliberately excludes the surrounding
CL21/22/24–29 experimental framework. Both leaves preserve the fixed 96-image
validation protocol, RealVis/DDIM50/CFG5 inference contract,
`pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and the existing 2,240-tensor
/ 219,217,920-parameter ownership contract.

No training or Serv job was launched by this change.

## Scientific changes

### CL23: one inference-time routing delta from CL19

CL23 retains CL19's native full-query message \(N\), reference full-query
message \(R\), and two-cell cosine mask \(C\). It splits \(D=R-N\) with the
fixed separable Gaussian kernel `[1,4,6,4,1]/16`:

$$
D_L=G*D,\qquad D_H=D-D_L.
$$

For denoising progress \(p=1-t/(T-1)\), its only new equation is

$$
Y=N+C\odot\left[(0.50+0.35p)D_L+(0.75+0.50p)D_H\right].
$$

The implementation is
[`_call_temporal_frequency`](../../src/model/photomaker_branched/attn_processor_cleanest.py)
and the leaf is
[`CL23_cosmic_temporal_frequency_router_24k.yaml`](../../src/configs/CL23_cosmic_temporal_frequency_router_24k.yaml).
There are no new parameters or dataset changes versus CL19.

### CL27: one training-only objective and its supervision mask

CL27's inference path is exactly CL23. On 25% of Cosmic training samples, a
deterministic seed-150017 overlay supplies a top-object mask. Only
`up_blocks.0/1` consume it, and only while gradients are enabled. With routed
low/high components \(L,H\), native message \(N\), routed delta
\(\Delta=L+H\), top mask \(M_T\), and visible face \(M_V\):

$$
L_{CL27}=L_{face}
+0.02\left(\operatorname{mean}_{M_T}H^2
+0.25\operatorname{mean}_{M_T}L^2\right)
+0.005\max\left(0,0.35-
\frac{\operatorname{RMS}_{M_V}\Delta}
{\operatorname{stopgrad}(\operatorname{RMS}_{M_V}N)}\right)^2.
$$

The overlay/mask is in
[`cosmic_large_adapted.py`](../../src/datasets/cosmic_large_adapted.py), the
loss is in
[`attn_processor_cleanest.py`](../../src/model/photomaker_branched/attn_processor_cleanest.py),
and aggregation is in
[`lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py).
The exact leaf is
[`CL27_cosmic_frequency_surface_energy_24k.yaml`](../../src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml).

## Pipeline-only throughput fixes

These controls implement the fixed pipeline identified on 16 August without
changing scientific computation:

- every collector resolves Diffusers' recursive `unet.attn_processors`
  property once before its per-layer loop;
- the CL27 collector returns before processor lookup when its objective is
  disabled;
- CL27's eligible-sample reduction remains on-device and does not convert a
  CUDA tensor to a Python boolean per processor;
- the dataset allocates the 1024×1024 semantic mask only when CL27 enables
  semantic occlusion;
- CL23/CL27 disable unconsumed full-activation frequency telemetry. The routed
  activations, loss graph, gradients, ownership, and generated pixels do not
  depend on that detached telemetry.

The validator rejects the known hot-loop regression if
`model.unet.attn_processors.get(...)` reappears in the helper.

## File-by-file delta

| Area | Files | Reason |
|---|---|---|
| Attention | `attn_processor_cleanest.py` | Fixed CL23 Gaussian route and CL27 auxiliary loss only. |
| Runtime | `branched_runtime.py` | Pass real scheduler progress, selected groups, and optional supervision mask. |
| Model contract | `e13_contract.py`, `lora2.py`, `lora2_helpers.py` | Fail-closed schedules/objective, unchanged ownership, live loss collection. |
| Dataset | `cosmic_large_adapted.py`, `configs/datasets/all_datasets.yaml` | Defaults-off deterministic CL27 overlay/mask. |
| Validation | `photomaker_branched_cl18_cl20.py`, `base_trainer.py` | Install the trained CL23/CL27 processor flags in the existing subject-v2 validation path. |
| Config/gates | CL23/CL27 leaves, `validate_cl23_cl27_config.py`, Cosmic preflight | Pin exact values and reject contract drift before model startup. |
| Serv | shared launcher and two `serv_run_packages` YAMLs | One-A100, 24k-step, full-96 prepared jobs; no submission. |

## Source and result provenance

- CL23 immutable Comet key: `a9ec9c59d1624c68acb98737dcd65298`.
- CL27 r3 immutable Comet key: `dbfbf40c3bdd4f70bedc58bda3dfb9cd`.
- CL27 r3 was promoted at its 16k checkpoint (`id_sim=0.547260`); its 24k
  endpoint was `0.543081`. The clean YAML defines the full historical 24k
  recipe and does not silently substitute or fetch a checkpoint.
- Production formulas and the 16 August processor-lookup repair were taken
  from committed `test` snapshot `6eb6613`, not its dirty working tree.

## Verification and parity boundary

The local verification set comprises Python compilation, shell syntax,
strict YAML parsing, Hydra composition for both new leaves, all existing clean
family validators, whitespace checks, and an in-memory CPU fixture that loaded
the committed processor from `test@6eb6613`. On identical weights, tensors,
masks, and progress, CL23 output plus both CL27 loss scalars were bit-exact
(`rtol=0`, `atol=0`) against the concise clean processor. The branch keeps
CL14/CL18/CL19/CL20 defaults inert because every new argument defaults off;
the sealed CL14 source and fixed-96 gate also passes.

An exact A100 checkpoint/full-96 RGB replay was not run: the task explicitly
prepared training without launching it and this local machine has no A100.
Therefore parity here means exact scientific-source/config parity, not a newly
observed pixel-equality claim.
The shared launcher reruns Hydra composition, source parity, dataset preflight,
ONNX Runtime CUDA, trainable ownership, fixed validation, and immutable Comet
startup gates on Serv before training.

## Serv runbook

From the clean checkout on Serv:

1. pull `clean` and require an empty `git status`;
2. populate the gitignored `diffusion_template/.env` with Comet, dataset,
   subject-v2, PhotoMaker, and face-quality paths;
3. confirm the corrected-r2 Cosmic manifest hash and sealed subject-v2 file;
4. inspect Running and Pending MLS jobs and remain within the normal six-A100
   project ceiling;
5. submit exactly one prepared YAML only after explicit authorization:

```bash
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_e13_family_clean/diffusion_template/serv_run_packages/CL23_cosmic_temporal_frequency_router_24k_full96_clean_r1/run_CL23_cosmic_temporal_frequency_router_24k_full96_clean_r1_1gpu.yaml
```

or:

```bash
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_e13_family_clean/diffusion_template/serv_run_packages/CL27_cosmic_frequency_surface_energy_24k_full96_clean_r1/run_CL27_cosmic_frequency_surface_energy_24k_full96_clean_r1_1gpu.yaml
```

Each YAML rejects a dirty/wrong branch, records `source_commit.txt`, invokes
the shared fail-closed launcher, and requires creation of the immutable Comet
experiment-key record during startup.
