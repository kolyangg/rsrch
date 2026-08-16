# CL14 to CL29 training-throughput diagnosis and optimization plan

**Date:** 16 August 2026  
**Repository:** /home/kolyangg/rsrch_apr_test/diffusion_template  
**Evidence cutoff:** 16 August 2026, 12:25 Europe/London  
**Scope:** training throughput only; no training, validation, checkpoint, or
production-job change was made for this report

> **16 August correction after matched CL14 replay inspection:** the Git
> ancestor `c04970f...` lacks `_record_active_gradient_norms`, but the exact
> sealed CL14 overlay that produced the 2.19 s/iteration run contains and calls
> it every step. The scan is still removable dead work when its metrics are not
> requested, but it cannot explain the CL14-to-CL19 regression. The causal
> contribution previously assigned to that scan is withdrawn. **[code]**

## Executive conclusion

The slowdown is real, stable within runs, and cumulative. After discarding the
first 20 batches of every epoch, CL14 trains at a median **2.19 s/iteration**.
CL26 is **6.12 s/iteration**, while the currently running CL27, CL28, and CL29
are **7.23, 7.02, and 7.37 s/iteration** respectively. The latest runs are
therefore **2.79x to 3.37x slower** than CL14, adding roughly **26 to 35 hours**
to a 24k-step run before validation time. **[measured]**

This is not one unexplained Serv-wide regression. The evidence resolves into
three cumulative bands:

1. **CL14 to CL19: 2.19 -> 3.94 s/iteration.** The exact sealed CL14 overlay
   already performs the every-step 219M-parameter active-gradient scan. The
   scan is unused and removable, but it is common to both sides and therefore
   does not explain this band. The remaining cause is **not established**;
   matched source replay and component timing are required. **[code]**
2. **CL19 to CL23/CL26: 3.94 -> 5.50/6.12 s/iteration.** CL23 performs a full
   fp32 5x5 depthwise Gaussian convolution and multiple full-activation fp32
   diagnostic reductions at all 70 BA self-attention processors on every
   step. CL26 computes a legacy result at each selected up0/up1 processor,
   discards it, reconstructs the CL19 full-query baseline, and then adds its
   ROI residual. These are concrete additional compute paths. **[code]**
3. **CL23 to CL27-CL29: 5.50 -> 7.02-7.37 s/iteration.** CL27 performs a
   device-to-host bool(eligible.any()) decision at each of 36 selected up0/up1
   processors, serializing the GPU repeatedly. CL28 adds per-processor schedule
   math, anchor reductions, and additional telemetry across all 70 processors.
   CL29 loads a second reference for every sample, synchronizes on a GPU random
   scalar every step, and on sampled eligible batches executes a complete
   second branched U-Net pass plus additional reference-attention and Gaussian
   paths. **[code]**

The first implementation priority is therefore not a broad rewrite. It is to
retain the already implemented unused-gradient-scan bypass, eliminate CL26's discarded
legacy path, tensorize CL27's eligibility reduction without host
synchronization, and separate scientific losses from diagnostic telemetry.
These changes can be placed behind explicit compatibility toggles and tested
without changing Q/K/V routing, masks, losses, optimizer ownership, or the
fixed validation contract.

## 1. Audit method

### 1.1 Immutable runs and timing extraction

Runs were identified by immutable Comet key and Serv job ID, not display name.
The source for each run was read from its sealed source_manifest.json; live
machine state was rechecked on Serv. Neb was not accessed.

Training times were parsed from the complete available Serv stderr progress
lines. For each epoch, batches 0-20 were discarded, then batches 21-120 were
sampled. The reported statistic is the median across those warmed samples.
This avoids initialization and first-batch cache effects while sampling every
available epoch. CL14 and CL26 are complete 12-epoch runs. At the cutoff,
CL27 and CL28 contributed 1,000 warmed samples each; CL29 contributed 948.

This is a training-loop measure. It excludes step-zero and periodic 96-image
validation, deferred face-quality scoring, and post-run work.

### 1.2 Code comparison

The comparison used the exact per-file SHA-256 manifests rather than the
current dirty checkout. The current worktree already contained unrelated user
changes and was not normalized or reset. The principal comparisons were:

- CL14 sealed source versus its recorded Git ancestor;
- CL14 ancestor c04970f... versus CL19 ancestor ad194a0...;
- CL19 r2 versus CL23 r1 and CL26 r3 sealed sources;
- CL23 r1 versus CL27 r3 and CL28/CL29 corrected sealed sources;
- exact leaf configs and launchers for CL14, CL19, CL23, and CL26-CL29.

CL23 to CL28 changes seven core runtime files. The largest exact diffs are
attn_processor_cleanest.py (+319/-23), lora2.py (+395/-1),
branched_runtime.py (+133), lora2_helpers.py (+87),
base_trainer.py (+28), and train.py (+24). CL28 and CL29 use the same core
runtime source; their performance difference comes from configuration and
enabled execution paths, not distinct attention implementation files.

## 2. Exact run and source provenance

### 2.1 Authoritative ledger

- **CL14 r1, 9 August**
  - Serv: lm-mpi-job-2ff91c51-1eb2-4290-bd7f-0d4bfcd5f227
  - Comet: 6fe0028be92242c38056b3d36665fdd6
  - Source: c04970f342a186d1092f07f9a08d7d8a797383e8 plus
    cl12-cl14-snapshot-v1-20260809; 1,220-file manifest.
- **CL19 r2, 11 August**
  - Serv: lm-mpi-job-f1b9d006-208c-4b35-8e4a-ab0ab2f030a9
  - Comet: cfeda7b55c174b3c83e8d40537ebb6dd
  - Source: ad194a026ab701dd979712d415c487dd536a4645 plus
    cl15-cl20-r2-mountfix-valseal-metricseal-20260811; 614 files.
- **CL23 r1, 13 August**
  - Serv: lm-mpi-job-f9160c9d-2b18-401d-844c-1e1116f17c3e
  - Comet: a9ec9c59d1624c68acb98737dcd65298
  - Source: cl19-cfeda7b5+cl21-cl26-20260813-v1; 724 files.
- **CL26 r3, 13 August**
  - Serv: lm-mpi-job-e07a2b02-6f5b-4ad8-bf80-e1f36c24cd4b
  - Comet: e9c0a9b505f041a68a183ca3cb4ca0af
  - Source: cl19-cfeda7b5+cl21-cl26-20260813-r3-activation-dtype-fix;
    728 files.
- **CL27 r3, 14 August**
  - Serv: lm-mpi-job-6af73e51-e281-4356-adde-767f15cc7607
  - Comet: dbfbf40c3bdd4f70bedc58bda3dfb9cd
  - Source: cl27-no-grad-surface-guard-20260814-v4; 640 files.
- **CL28 r4, 14 August**
  - Serv: lm-mpi-job-6681ed16-1d71-4175-a5a0-fca7b3a1b632
  - Comet: 3d8aca3b4cbb4ddc9338f14952c5bd0e
  - Source: cl28-cl29-training-transition-fix-20260814-v5; 642 files.
- **CL29 r3, 14 August**
  - Serv: lm-mpi-job-4977ec3e-aab0-4793-9330-116e721801f5
  - Comet: 2981820837564d01b1cefbf52c4dabd0
  - Source: cl28-cl29-training-transition-fix-20260814-v5; 641 files.

### 2.2 What “exact commit” means here

Only CL14 and CL19 have a real Git SHA embedded in the sealed source revision:

- **CL14 Git ancestor:** c04970f342a186d1092f07f9a08d7d8a797383e8,
  authored 26 July 2026 13:08 +01:00, subject
  “Preserve single-checkpoint validation load”.
- **CL19 Git ancestor:** ad194a026ab701dd979712d415c487dd536a4645,
  authored 6 August 2026 19:37 +01:00, subject “06 Aug”.

Neither plain commit alone is the exact training source. Both runs included a
sealed, uncommitted experiment overlay whose exact contents are represented by
the manifest's per-file SHA-256 map. The CL14 source was later ported into Git
by the clean-port series 27e72bd9... through 704d4914... on 10 August, but
those commits postdate the CL14 job and must not be cited as its training
commit.

CL23 and later revision labels are **not Git commits**. In particular,
cfeda7b5 and a9ec9c59 are prefixes of immutable Comet experiment keys, not Git
SHAs. There is no honest single-commit answer for those jobs. Their exact
reproducible source identity is the sealed revision plus the full manifest.
This is a provenance gap in the launch process, not missing evidence for this
audit. Future production snapshots should record both git rev-parse HEAD and a
dirty-overlay hash in addition to the current file manifest.

## 3. Throughput measurements

### 3.1 Cross-epoch warmed medians

| Run | Mechanism | Median s/iter | Versus CL14 | Hours / 2k | Projected train hours / 24k |
|---|---|---:|---:|---:|---:|
| CL14 | hard spatial BA, feathered target mask | **2.19** | 1.00x | 1.22 | 14.6 |
| CL19 | full-query soft router | **3.94** | 1.80x | 2.19 | 26.3 |
| CL21 | CL19 + residual identity CA | 6.04 | 2.76x | 3.36 | 40.3 |
| CL22 | visibility-order router | 5.80 | 2.65x | 3.22 | 38.7 |
| CL23 | temporal-frequency router | **5.50** | 2.51x | 3.06 | 36.7 |
| CL24 | sparse PM boundary distillation | 4.08 | 1.86x | 2.27 | 27.2 |
| CL25 | cadence-16 identity reward | 4.24 | 1.94x | 2.36 | 28.3 |
| CL26 | CL19 + anchored high-resolution ROI | **6.12** | 2.79x | 3.40 | 40.8 |
| CL27 | CL23 + frequency-surface loss | **7.23** | 3.30x | 4.01 | 48.2 |
| CL28 | CL23 + learnable schedules | **7.02** | 3.21x | 3.90 | 46.8 |
| CL29 | CL23 + sampled low-band contrastive path | **7.37** | 3.37x | 4.09 | 49.1 |

CL14, CL26, and the available CL27-CL29 epochs are internally stable: their
last-100 medians are 2.26, 6.09, 7.22, 6.98, and 7.34 s/iteration. The gap is
therefore not a first-epoch warm-up artifact. **[measured]**

The earlier matched-first-epoch audit independently found CL19 3.93, CL21
6.07, CL22 5.79, CL23 5.57, CL24 4.13, CL25 4.20, and CL26 6.12
s/iteration. See
analysis/2026-08-13_cl21_cl26_training_throughput_diagnosis.md.
**[report]**

### 3.2 Why a common machine regression is unlikely

- All audited production jobs requested the same
  a100.1gpu.8C.243G class. **[report]**
- CL24 and CL25 ran in the newer source era at approximately 4.1-4.2
  s/iteration while CL21-CL23/CL26 on the same service were 5.5-6.1. The
  mechanism selected by the config predicts the band better than calendar
  time. **[measured]**
- The active launchers leave CUDA_LAUNCH_BLOCKING unset, use one Accelerate
  process, suppress Comet auto-logging, and fail closed around the intended
  GPU-based validation dependencies. **[code]**
- CL27-CL29 were running concurrently, so shared CPU/NFS contention cannot be
  completely excluded for their absolute 7-second values. A controlled replay
  remains necessary before assigning every additional millisecond to CL28's
  schedule operations. **[not established]**

## 4. Exact pipeline differences and likely cost

### 4.1 P0: unused all-parameter active-gradient scan, including sealed CL14

PhotomakerLoraTrainer.process_batch() calls
_record_active_gradient_norms(batch) after every backward and before every
optimizer step. That function walks all three optimizer groups and, for every
non-null gradient, executes grad.detach().float().square().sum(). The audited
contract has 2,240 parameter tensors and 219,217,920 trainable parameters
(2,310 tensors for CL28). **[code]**

CL19, CL23, and CL26-CL29 do not request active_grad_norm_ba,
active_grad_norm_generic_adapter, or active_grad_norm_photomaker_default in
writer.loss_names. The result is not used for clipping
(trainer.max_grad_norm=null), loss construction, or the optimizer. It is
discarded. **[code]**

The method and call are absent from Git ancestor c04970f..., but they are
present in the exact sealed `cl12-cl14-snapshot-v1-20260809` overlay used by
CL14 and remain present later. The ordinary _get_grad_norms() path is already
gated by trainer.grad_norm_log_only=true and log_step=50. The scan is definite
dead work when unrequested, but its presence in the 2.19 s/iteration CL14 run
rules it out as the cause of the CL14-to-CL19 step. Its exact recoverable time
remains **not established** without a matched A/B. **[code]**

**Fix:** add an explicit trainer.active_grad_norm_mode with backward-compatible
every_step, plus requested_log_steps and off. Resolve whether any active metric
is requested once during trainer initialization. For CL19+ configs that request
none, skip both scalar initialization and all gradient scans. For experiments
that need the diagnostic, evaluate it only on the declared cadence. Because
the current operation is detached and post-backward, the off path should
produce identical losses, gradients, optimizer states, and weights.

### 4.2 P0: CL26 computes and discards a legacy baseline

In BranchedAttnProcessor._call_hardcase(), modes highres_roi, anchored_roi, and
clean_memory first execute:

    baseline = self._call_legacy(attn, hidden_states, temb=temb)

For anchored_roi, CL26 then calls _full_target_lanes(), constructs the CL19
soft-router result, and overwrites baseline through _finish_full_router().
The first legacy result has no consumer. This occurs at the selected up0/up1
self-attention processors before the ROI addition is calculated. **[code]**

**Fix:** branch before _call_legacy(): build the CL19 soft baseline directly for
anchored_roi; retain legacy behavior for highres_roi and clean_memory. Add a
dated invariant comment that target Q and reference K/V are unchanged. Gate
with exact forward, loss, gradient, and two-step optimizer parity from the same
checkpoint and batch.

### 4.3 P0: CL27 serializes the GPU at every selected processor

CL27 enables the frequency-surface loss in up0/up1, which corresponds to 36 of
the 70 stateful SDXL BA processors. Each selected processor computes eligible
on the GPU and then evaluates:

    if not bool(eligible.any()):
        return metrics

Python must wait for the device result, creating up to 36 device-to-host
synchronization points in one U-Net forward. Eligible processors additionally
perform several fp32 masked reductions over full layer activations. **[code]**

**Fix:** replace the branch with a tensorized weighted reduction. Compute an
eligible float mask, divide by eligible.sum().clamp_min(1), and multiply the
per-example terms by that mask. A no-eligible batch then produces a
graph-connected zero without a host decision. Preserve the existing
applied-fraction definition and confirm exact zero behavior plus allclose
nonzero losses/gradients.

### 4.4 P0/P1: temporal-frequency telemetry ignores its configured cadence

At every one of the 70 temporal-frequency processors, CL23 and its successors
compute detached fp32 means/RMS values for low scale, high scale, low delta,
high delta, and merged/native ratio. collect_branched_telemetry() then loops
over all processors, converts every scalar to fp32, stacks them by semantic
group and again for all, and reduces them on every training step. CL28 adds
raw schedule mean/max telemetry. **[code]**

The model already has ba_telemetry_interval=50, and the older v3/v4 processors
honor it. The hard-case implementation in attn_processor_cleanest.py does not.
Its expensive activation diagnostics are produced every step even though
trainer.log_step=50. **[code]**

**Fix:** separate loss state from diagnostic state. Losses needed for backward
must remain every step; detached diagnostics should honor an explicit cadence.
Do not silently change existing curve semantics: keep a compatibility mode for
the historical 50-step mean and introduce separately named sampled telemetry
if diagnostics are evaluated only on log steps. First optimize aggregation and
skip metrics not listed by the writer; then benchmark cadence sampling as an
explicitly documented metric-definition change for new runs.

### 4.5 P1: full fp32 5x5 Gaussian filter at all 70 sites

_gaussian_split() casts each full [batch, tokens, channels] delta to fp32,
constructs the same binomial 5x5 kernel, expands it to every channel, and runs
a grouped 2-D convolution. CL23 does this at every temporal-frequency
processor, every step. **[code]**

**Fixes, in increasing risk order:**

1. cache the immutable 1-D/2-D kernel per device, dtype, and channel count;
2. use two cached separable depthwise convolutions (5x1 then 1x5), reducing
   filter multiplications by about 60%;
3. profile an exact face-ROI-plus-two-cell-halo implementation, because the
   routed output is zero outside the router but face pixels depend on the
   convolution halo.

The separable form is mathematically equivalent but not necessarily bitwise
identical because floating-point accumulation order changes. Require output,
loss, and gradient allclose gates before use. Do not switch the filter to bf16
inside an eligible production experiment without treating it as a numerical
ablation.

### 4.6 P1: CL29's sampled auxiliary repeats the U-Net

CL29 has three distinct avoidable costs. **[code]**

- torch.rand(..., device=latents.device).item() forces one host
  synchronization every training step; building the wrong-ID permutation also
  calls .item() on device values.
- same_identity_dual_reference=true makes the dataset select, read, and
  transform an alternate reference even on the 87.5% of steps where the
  auxiliary is not sampled.
- On a sampled eligible batch, the trainer VAE-encodes the alternate reference
  and executes a complete second run_branched_forward_pass(). At the 46
  selected mid/up0/up1 processors, the first pass also recomputes a detached-Q
  reference message and Gaussian split; the contrast pass calculates positive
  and wrong-reference messages and two more Gaussian splits. The second pass
  still computes native lanes and unrelated U-Net blocks.

**Fix:** create a deterministic CPU-side batch schedule, eliminating device
.item() decisions and allowing the data pipeline to load alternate references
only for scheduled batches. More importantly, compute primary, same-ID
positive, and wrong-ID reference K/V messages as explicit auxiliary lanes at
the selected processors during one U-Net pass, reusing the same detached target
Q. The existing two-pass code remains a compatibility oracle. Require matched
processor embeddings, InfoNCE loss, gradients, and sampled-step frequency.

### 4.7 P1: fuse independent attention lanes

CL19/CL23 evaluate native target attention, target-Q/reference-KV attention,
and reference self-attention as separate SDPA calls at compatible shapes. The
Q/K/V projections have different ownership and must remain explicit, but the
already projected tensors can be concatenated across batch for one SDPA launch
and split afterward. Compatible output projections can be fused only when
their modules are the same. **[code/hypothesis]**

This is analogous to the already measured CL14_CA optimization, where batching
independent work and removing scalar synchronization improved throughput from
about 3.48-3.59 to 3.23 s/iteration. See
analysis/2026-08-13_CL14_CA_training_throughput_optimization.md.
**[report]**

### 4.8 Lower-priority conditional improvements

These should be attempted only if the phase-0 profile assigns meaningful time
to the relevant region:

- **Conditioning/data:** preserve the existing conditioning cache. Profile
  data-wait and frozen reference VAE/ID encoding before designing an offline
  cache. Cosmic reference selection and transforms are stochastic, so cache
  keys must include the complete augmentation state. CL29's unsampled
  alternate-reference work is the immediate data fix.
- **Transfers:** the current loader uses two workers and pin_memory=false, and
  the trainer performs blocking .to(device) for pixel_values. A prior CL14_CA
  trial did not show a useful gain from generic pin/persistent/nonblock
  changes. Revisit only if measured data-wait exceeds 5%.
- **Optimizer:** the config uses torch.optim.AdamW without an explicit fused
  mode. Benchmark the exact installed PyTorch default against fused=True in a
  speed smoke. Treat any changed optimizer numerics/state as an explicit new
  training mode, not a transparent production substitution.
- **Compilation:** torch.compile may reduce Python/kernel-launch overhead but
  interacts with dynamic masks, processor state, alternate validation pipes,
  and sampled branches. It is a later isolated experiment, not a P0 fix.

## 5. What is not the cause

- The standard validation contract does not explain the reported seconds per
  training iteration; validation is outside the parsed loop. **[measured]**
- The newer one-GPU scalar writer already batches requested loss scalars and
  performs one .cpu().tolist() synchronization instead of one gather/item pair
  per metric. That is an improvement over older logging code, not the source
  of the latest regression. **[code]**
- trainer.grad_norm_log_only=true already limits the ordinary
  _get_grad_norms() logger to every 50 steps. The separate
  _record_active_gradient_norms() pass is the every-step problem. **[code]**
- Mask preparation caching is already enabled by the inherited replay config.
  It should be verified, but simply turning on the existing toggle is not a new
  fix for CL19-CL29. **[code]**
- max_grad_norm is null, so gradient clipping does not consume the active norm
  scan. **[code]**
- There is no evidence that CPU InsightFace/ONNX validation scoring is running
  inside every training iteration. **[code/log]**

## 6. Prioritized implementation and experiment plan

### Phase 0 - measure component time without changing production

Add a defaults-off speed-smoke profiler that records CUDA-event time for:
data wait/H2D, frozen conditioning, branched U-Net forward, loss, backward,
post-backward diagnostics, optimizer, and scalar logging. Use 20 warm-up plus
100 measured steps on one A100 with validation disabled only in the explicitly
labeled smoke. Capture one short torch.profiler trace with CPU and CUDA
activities after the lightweight timing has located the hot region.

**Gate:** replay exact CL19, CL23, CL26, and one of CL27-CL29 from their sealed
sources/configs. Same batch size, seed, dataset schedule, precision, and A100
class. Report median and p10/p90 per region; never compare profiler-on time to
production time.

### Phase 1 - pipeline-neutral P0 fixes

Implement one change at a time:

1. skip unused active-gradient norms;
2. remove CL26's discarded legacy result;
3. tensorize CL27 eligibility without host synchronization;
4. skip unrequested telemetry and honor explicit telemetry policy.

**Correctness gates:**

- old and new toggles compose successfully;
- expected trainable/optimizer ownership is unchanged exactly;
- same fixed batch and checkpoint produce identical outputs/loss/gradients for
  the detached-gradient-scan and CL26-dead-path removals;
- CL27 produces exact zero loss when ineligible and allclose nonzero
  loss/gradients when eligible;
- after two optimizer steps, weights and optimizer state are exact where the
  transformation is intended to be bit-preserving;
- all configured writer keys resolve, and historical telemetry mode retains
  its old definition;
- validation routing keeps pose_adapt_ratio=0 and ca_mixing_for_face=false.

**Performance gate:** require at least a 5% median step-time improvement for an
individual optimization or a profiler-confirmed reduction in its target
region. Revert complexity that does not pass. The first combined milestone is
CL19/CL24 at or below 3.2 s/iteration and CL23 at or below 4.5 s/iteration on
the matched A100 smoke; these are engineering targets, not measured promises.

### Phase 2 - frequency and attention kernels

1. cache Gaussian kernels;
2. test separable Gaussian convolution;
3. batch compatible SDPA lanes and output projections;
4. only then evaluate exact ROI-plus-halo filtering.

**Gate:** fixed-tensor forward and backward comparisons across all five
installed groups (down1, down2, mid, up0, up1), both old/new modes, square
token sizes used by SDXL, zero/nonzero masks, and bf16 production activations
with fp32 frequency math. No change may replace reference K/V with target
features.

### Phase 3 - CL29 single-pass auxiliary

Move sampling to a deterministic CPU schedule, lazy-load alternate references,
then reproduce the current two-pass embeddings and loss with auxiliary
reference lanes inside one U-Net pass.

**Gate:** the same scheduled batches, identities, alternate references,
negative permutations, target Q, low-band embeddings, InfoNCE loss, and
gradients must match the current oracle. Unscheduled steps must do no alternate
image decode, VAE encode, or auxiliary attention.

### Phase 4 - controlled production qualification

After the speed smokes pass, create a new explicitly named optimization
qualification run. Do not overwrite or relabel CL14-CL29. Keep the fixed
manual-val-96 panel, prompts, seeds, references, boxes, DDIM50 scheduler,
step-zero/every-2k cadence, one generated image per item, batch 2, model
precision, and all metric definitions unchanged. Record the new immutable
Comet key and sealed source manifest at startup.

Promotion requires:

- throughput improvement sustained over at least 2,000 optimizer steps;
- no NaN/OOM/graph break or trainable-ownership change;
- bitwise-equal validation images for strictly pipeline-neutral changes, or a
  separately approved numerical tolerance for separable/fused kernels;
- unchanged fixed-panel ID, prompt, face-quality, artifact, and face/body
  metrics within the declared parity rule;
- exact source provenance containing both Git HEAD and dirty-overlay hashes.

No training job should be launched merely to implement this plan. Inspect live
Running/Pending Serv allocations immediately before any authorized submission.

## 7. Confidence and unresolved questions

| Claim | Confidence | Basis |
|---|---|---|
| CL26-CL29 are 2.79x-3.37x slower than CL14 | High | hundreds of warmed stderr samples across many epochs |
| The active-gradient scan is dead work in CL19-CL29 | High | exact config/source; metrics unrequested; clipping disabled |
| The scan explains a large fraction of CL14 -> CL19 | Disproved | exact sealed CL14 overlay already contains and calls the scan |
| CL26 contains a discarded legacy attention result | High | direct control-flow inspection |
| CL27 creates repeated device-host synchronization | High | Python bool() on CUDA reductions at 36 processors |
| CL23 filter/telemetry materially contributes to +1.56 s over CL19 | Medium-high | unavoidable fp32 convolution/reductions at 70 sites; no trace yet |
| CL28 schedule alone explains its full +1.52 s over CL23 | Low-medium | code adds 70-site work; concurrent/resource effects not excluded |
| CL29's sampled second pass is avoidable | High | direct two-pass control flow and explicit 0.125 probability |
| A common Serv regression is the primary cause | Low | contemporary runs separate by mechanism; matched replay still needed |

Not established by this audit:

- exact milliseconds attributable to each operation without CUDA-event and
  profiler A/B runs;
- GPU clocks, SM occupancy, or NFS/data-wait percentages for historical CL14;
- whether CL27-CL29's concurrent execution contributes a smaller common tax;
- final CL27-CL29 scientific quality, because they were still running at the
  evidence cutoff;
- end-to-end speed after proposed fixes, because no code was changed or job
  launched.

## 8. Reproduction checklist

Run from diffusion_template/ in the existing photomaker_NS environment on Serv
where required.

    # Git ancestors and dates
    git show --no-patch --format='%H%n%aI%n%s' \
      c04970f342a186d1092f07f9a08d7d8a797383e8 \
      ad194a026ab701dd979712d415c487dd536a4645

    # Git ancestry alone is insufficient: verify the sealed CL14 overlay too
    git show c04970f342a186d1092f07f9a08d7d8a797383e8:\
    diffusion_template/src/trainer/sdxl_trainers.py | \
      rg 'record_active_gradient_norms|active_grad_norm'
    git show ad194a026ab701dd979712d415c487dd536a4645:\
    diffusion_template/src/trainer/sdxl_trainers.py | \
      rg 'record_active_gradient_norms|active_grad_norm'

    # Current relevant code paths
    rg -n '_record_active_gradient_norms|_gaussian_split|eligible.any|\
    collect_branched_telemetry|lowband_permutation|_call_legacy' src

    # Verify any sealed runtime before using it
    python tools/verify_serv_source_manifest.py \
      --root /absolute/runtime/source/diffusion_template \
      --manifest /absolute/runtime/source/source_manifest.json

For timing, save the complete training stderr, extract tqdm values matching
[0-9.]+s/it, group by epoch, discard batches 0-20, and take batches 21-120.
Store the parser and raw-log SHA-256 beside the speed-smoke result so later
reports do not rely on display-name or first-screen estimates.

## Decision

Proceed with the phase-0 timing harness and the pipeline-neutral P0 changes in
separate branches/toggles. Retain the unused-gradient-scan bypass as dead-work
removal, but do not use it as the explanation for CL14-to-CL19. The first
causal speed smoke should replay exact CL14 source on the current A100, then
profile the remaining source delta. Do not alter the fixed
validation contract or launch a production experiment until the parity and
performance gates above pass.
