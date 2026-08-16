# CL21--CL26 slowdown is experiment-specific compute, not a common Serv regression

**Date:** 13 August 2026  
**Scope:** read-only diagnosis of CL21--CL26 training throughput. No running job
was stopped or restarted, and no training code/configuration was changed.  
**Evidence cutoff:** first-epoch training through at least `176--677` iterations
per arm; all six jobs still `running` on one `a100.1gpu.8C.243G` each.

## Executive conclusion

The apparent `~5 s/it` slowdown is not uniform. Against CL19's matched warmed
rate of `3.930 s/it`, CL24 and CL25 are close (`+5.1%` and `+6.7%`), while
CL21, CL22, CL23, and CL26 are `+41.7%` to `+55.7%` slower. [measured] Since
these arms run concurrently on the same Serv instance type and allocation,
the split strongly argues against a dominant shared GPU, data-loader, or NFS
regression. [measured][inference]

The added architecture explains the ordering. CL21 executes residual identity
cross-attention at every selected up0/up1 CA site; CL22 runs a dense per-token
three-state ownership router and its supervision at up0/up1; CL23 applies a
full-precision 5x5 depthwise spatial filter at every selected SA site; and CL26
contains a concrete redundant forward: it computes a complete legacy BA
baseline, discards it, then computes the CL19 baseline and ROI branch. [code]
CL24/CL25 add only sparse teacher/reward work, matching their much smaller
average overhead. [code][measured]

The current jobs cannot inherit a future source optimization without a restart.
Per instruction, they should remain untouched for now.

## 1. Matched throughput

Primary operational metric: median tqdm seconds per optimizer iteration over
first-epoch steps `21--120` (`100` warmed observations; startup excluded).
This is an end-to-end loop metric, not a scientific quality metric.

| Arm | Serv job | Immutable Comet key | `s/it` | vs CL19 | Hours / 2k steps |
|---|---|---|---:|---:|---:|
| CL19 control | `lm-mpi-job-f1b9d006-208c-4b35-8e4a-ab0ab2f030a9` | `cfeda7b55c174b3c83e8d40537ebb6dd` | **`3.930`** | -- | `2.18` |
| CL21 residual ID CA | `lm-mpi-job-fba7a7ca-ce8f-4b65-a7e5-f139cb3187af` | `6670db89c44a489388b8f09b91423b0d` | `6.070` | `+54.5%` | `3.37` |
| CL22 visibility router | `lm-mpi-job-84855e01-da1a-4066-b2b3-e71d4904f66e` | `b181feb6c54644e69fb7e8709a59f32e` | `5.790` | `+47.3%` | `3.22` |
| CL23 temporal-frequency | `lm-mpi-job-f9160c9d-2b18-401d-844c-1e1116f17c3e` | `a9ec9c59d1624c68acb98737dcd65298` | `5.570` | `+41.7%` | `3.09` |
| CL24 PM boundary teacher | `lm-mpi-job-caae3dad-99ab-40ac-80f2-6ebb106f813a` | `a18e22ae9f0e4a24b6252f6b392fab62` | `4.130` | `+5.1%` | `2.29` |
| CL25 low-noise ID reward | `lm-mpi-job-893096da-e633-40cc-9a28-cde68fd4e813` | `120b72df8134474ca094e6162d085eb0` | `4.195` | `+6.7%` | `2.33` |
| CL26 anchored ROI | `lm-mpi-job-e07a2b02-6f5b-4ad8-bf80-e1f36c24cd4b` | `e9c0a9b505f041a68a183ca3cb4ca0af` | `6.120` | `+55.7%` | `3.40` |

The rates remain stable after warmup: current tail medians are `6.05`, `5.75`,
`5.49`, `4.11`, `4.20`, and `6.10 s/it`, respectively. [measured] Historical
CL15--CL20 medians also track mechanism cost (`4.00`, `4.55`, `4.54`, `4.06`,
`3.94`, `3.66 s/it`), reinforcing that this is not a one-off launch anomaly.
[measured]

For context, historical CL14 was `2.190 s/it` over the same warmed window
(Comet `6fe0028be92242c38056b3d36665fdd6`). [report] Thus much of the perceived
gap from older runs already exists in the CL19 base (`3.930 s/it`); the current
arms then add the overhead shown above. Exact kernel-level attribution of the
CL14-to-CL19 gap is not established by this audit.

## 2. Root cause by arm

| Arm | Verified hot path | Diagnosis |
|---|---|---|
| CL21 | `residual_identity_ca_processor_v3.py:118-194` performs native CA plus a second target-Q/ID-KV attention, rank-64 output, normalization, masking, and full-tensor RMS diagnostics at each selected CA site. | Mostly intended architectural cost; the earlier native-row fusion optimization is already present. [code] |
| CL22 | `attn_processor_cleanest.py:561-605,800-838` forms float32 disagreement features, runs a hidden-128 MLP and 3-way softmax for every spatial token, computes class-weighted CE when ownership labels are supplied, then several float32 diagnostics at every selected up0/up1 SA site. | Intended router/supervision is expensive because it is repeated per layer and per token. [code] |
| CL23 | `attn_processor_cleanest.py:608-619,840-865` casts the complete routed delta to fp32 and applies a non-trainable 5x5 depthwise convolution over every channel at all seven U-Net groups, followed by fp32 RMS reductions. | Avoidably expensive implementation of an intended frequency split. [code] |
| CL24 | `lora2.py:1677-1726` conditionally runs one no-grad native PhotoMaker teacher U-Net on sampled synthetic-top batches. | Sparse forward-only teacher cost; observed average overhead is modest. [code][measured] |
| CL25 | `lora2.py:1546-1564,1605-1650,2548-2571` activates ArcFace/low-noise work and a frozen CL19 prediction every 16th step. It also changes from one batched reference to three references with batched conditioning disabled. | The periodic slow spikes in the log match the cadence; average overhead is modest. [code][measured] |
| CL26 | `attn_processor_cleanest.py:734-758` first calls `_call_legacy`, then overwrites `baseline` with `_full_target_lanes`/`_finish_full_router`, and only then adds ROI attention. | **Definite redundant forward computation** in anchored-ROI mode; the legacy result has no effect on output or loss. [code] |

Pure diagnostic telemetry is also calculated every training step for the new
hard-case routes, even though reporting only needs sparse samples. Its exact
share of latency is not measured, but the repeated fp32 reductions are an
execution-only optimization opportunity. [code][hypothesis]

## 3. What is not the cause, and what remains unestablished

- All six jobs are live on the same one-A100 SKU, with no OOM, data-loader,
  NCCL, ENOSPC, or training traceback in the inspected logs. The only network
  anomaly was two CL23 Comet timeout retries; its step rate is otherwise stable.
  [measured]
- Six simultaneous jobs/NFS traffic cannot be excluded as a small common cost,
  but it does not explain why contemporaneous CL24/25 stay near CL19 while four
  architecture-heavy arms are `42--56%` slower. [measured][inference]
- Serv NFS is currently `100%` used with only `18 GB` free (inode use `28%`).
  This is **not established as a throughput cause** and no write error exists,
  but it is a separate checkpoint-capacity risk. Nothing was deleted. [measured]
- No CUDA trace was captured inside the live jobs, so the percentage of time
  attributable to each kernel, telemetry, data loading, or optimizer work is
  not established. [measured]

## 4. Priority fix plan (not applied)

| Priority / proposed smoke | Single change | Prediction and risk | Decision gate |
|---|---|---|---|
| P0 `CL26_anchored_roi_no_discarded_legacy_smoke` | Skip `_call_legacy` only when mode is `anchored_roi`; retain the CL19 baseline and ROI branch exactly. | Largest low-risk fix; removes a provably unused forward. Risk is accidental route/telemetry drift. | Fixed-input output, loss, and gradient parity; unchanged trainable contract; then require lower whole-step latency. |
| P1 `CL23_separable_frequency_kernel_smoke` | Replace the fp32 5x5 Gaussian convolution with its mathematically equivalent horizontal+vertical 1D depthwise passes. | Reduces filter arithmetic from 25 to 10 taps; two launches may limit the gain. | Output/gradient allclose against current code and at least `10%` whole-step speedup before adoption. |
| P1 `CL21_face_query_sparse_idca_smoke` | Reuse native target Q and evaluate the extra ID attention/output only for nonzero face-mask queries, then scatter. | Query attention is position-separable, so masked-away queries need not be computed. Ragged gathering may offset savings. | Same masked output/loss/gradients and unchanged route/trainables; profile all up0/up1 sites. |
| P2 `hardcase_sparse_telemetry_smoke` | Compute detached RMS/ratio diagnostics only at the configured logging cadence and cache the last scalar; never gate CL22's live ownership loss. | Small-to-moderate execution-only gain, especially CL21/23/26. Risk is stale between-log dashboard values. | Identical training loss/gradients and matching telemetry on enabled steps. |
| P3 `CL22_face_local_router_profile` | First profile MLP, CE, and diagnostics separately; if MLP/CE dominates, train the ownership router on face/contact tokens plus sampled background rather than every background token. | Potentially large, but this changes CL22 supervision and is therefore a new scientific arm, not a transparent fix. | Attempt only if CL22 quality justifies continuation; compare fixed-96 quality and matched throughput against current CL22. |

CL24/CL25 do not warrant throughput-driven architecture changes before their
quality results are available. Their current `5--7%` overhead is consistent
with the intended sparse objectives. Before future checkpoint boundaries, run
a read-only NFS ownership/size audit and obtain explicit approval before any
cleanup.

## 5. Confidence

| Claim | Confidence | Basis |
|---|---|---|
| The slowdown is not a uniform Serv regression | High | Same concurrent A100 substrate, but a `4.13--6.12 s/it` split aligned with experiment mechanism. [measured] |
| CL26 performs an unused legacy forward | High | Direct control-flow inspection: the value is overwritten before use. [code] |
| CL21/22/23 overhead is primarily their new per-layer work | Medium-high | Strong timing/architecture correlation and verified hot paths; no in-job CUDA profile. [code][measured] |
| Sparse telemetry is a material fraction | Medium-low | Repeated fp32 reductions are present, but isolated latency is unmeasured. [code][hypothesis] |
| NFS fullness causes current step latency | Low | No I/O error and near-baseline arms run concurrently; only capacity risk is established. [measured] |

## 6. Reproducing

From `diffusion_template/`, extract tqdm rates below `20 s/it` (excluding
manual validation) and compare the median of observations `21--120`. Source
hot paths are in:

```text
src/model/photomaker_branched/attn_processor_cleanest.py
src/model/photomaker_branched/residual_identity_ca_processor_v3.py
src/model/photomaker_branched/lora2.py
src/model/photomaker_branched/lora2_helpers.py
src/trainer/sdxl_trainers.py
```

Live status was checked with `python ../local_scripts/serv_job.py status
<job-id> --branch test` inside the `photomaker` environment. Immutable Comet
keys in the table, not display names, should be used for any later audit.
