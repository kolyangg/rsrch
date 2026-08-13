# CL14_CA reduces training latency 7--10% while preserving corrected residual CA

**Date:** 13 August 2026

**Scope:** CL14_CA training-throughput diagnosis, localized implementation,
Serv smoke measurement, corrected-Eddie validation integration, and production
relaunch. The original `CL14_CA_r7` job was not changed or stopped.

**Evidence cutoff:** 13 August 2026 01:57 Europe/London, after the production
startup gate described below.

## Executive conclusion

The optimized CL14_CA implementation is stable and reduces warmed one-A100
iteration latency from the current run's recent median `3.480 s/it` to
**`3.230 s/it`**, a `7.2%` latency reduction. Against the matched first-epoch
window (`3.590 s/it`), the reduction is `10.0%`. This is measured over 100 consecutive
optimizer steps after startup; the direct elapsed-time calculation and median
displayed rate agree. [measured] The change does not remove or approximate the
new residual identity cross-attention. It batches equivalent native target and
reference CA work, builds the fixed identity-token indices once per U-Net call,
and replaces 19 separate telemetry synchronization pairs with one. [code]

The result does not reach historical CL14's `~2.19 s/it`: optimized CL14_CA is
still `47.5%` slower per iteration because CL14 did not execute the added
identity-conditioned residual cross-attention at 36 `attn2` sites. [measured]
[code] Treating the whole difference as a pipeline regression would therefore
be incorrect. Further large speed gains would require changing the new
architecture, precision, or processor coverage and would no longer be a safe
execution-only optimization. [hypothesis]

The production validation path also now matches CL20's repaired subject-v2
contract. It uses `bbox_overlap_v2` subject selection and the sealed
foreground-Eddie identity embeddings, rather than selecting the background
face from Eddie's multi-face reference. [code][report] The production run is
`CL14_CA_optimized_r11`; its immutable identifiers and final startup state are
recorded below.

## Run ledger

| Arm | MLS job | Immutable Comet key | Step / status | Headline evidence |
|---|---|---|---|---|
| CL14 control | `lm-mpi-job-2ff91c51-1eb2-4290-bd7f-0d4bfcd5f227` | `6fe0028be92242c38056b3d36665fdd6` | completed 24k | Historical median `2.190 s/it`; no residual identity CA. [measured][report] |
| Original CL14_CA | `lm-mpi-job-244ef7b2-3943-4998-a82e-ae1be2208169` | `4d96dc8e776b4039b1116acc5cdcf706` | left running | Recent 100-sample median `3.480 s/it`; retained unchanged as requested. [measured] |
| Live-r7 cold-start diagnostic | `lm-mpi-job-60b5e876-20d8-44f9-b7f3-4baa7a1c1c20` | `cc096b408f184faf9f1618afc4eb9588` | first backward failed | Exact healthy-r7 training source still segfaulted when step-zero validation was skipped. [measured] |
| Fused-CA cold-start diagnostic | `lm-mpi-job-537542bc-9e3e-44d6-95c7-6b3ab2aada1f` | `1225223a2d624d38a2d54f0bc7976f53` | first backward failed | Fused CA plus explicit device re-home did not make the unsupported cold path safe. [measured] |
| Optimized one-batch smoke r12 | `lm-mpi-job-b3db33aa-5b8e-45fd-a48e-e7fe7d7ab9af` | `75ab71fc2d4c44a5b4b625ebf20b89ed` | stopped after step 126 | One unchanged 12-image validation batch, finite training, `3.230 s/it` over steps 21--120. [measured] |
| Optimized production r11 | `lm-mpi-job-26dc8f54-1b96-4129-9151-a4fb066a7ff7` | `fafd7a61b06c4114b9dec2c21d29ca38` | running | Full fixed-96 corrected-Eddie validation and training startup gate; left running. [measured] |

## 1. Measured throughput

The operational primary metric is one-A100 seconds per optimizer iteration.
The smoke measurement excludes the first 20 training iterations and uses the
next 100 consecutive iterations. The progress bar's median displayed rate is
`3.230 s/it`; elapsed time advances from `01:07` at step 20 to `06:30` at step
120, independently giving `(390 - 67) / 100 = 3.23 s/it`. [measured]

| Run | Measurement | Samples | Median seconds / iteration |
|---|---|---:|---:|
| Historical CL14 | first epoch, steps 21--120 | `100` | `2.210` |
| Historical CL14 | steps 21--120 across all 12 epochs | `1,200` | `2.190` |
| Original CL14_CA r7 | first epoch, steps 21--120 | `100` | `3.590` |
| Original CL14_CA r7 | most recent 100 samples | `100` | `3.480` |
| Optimized smoke r12 | first epoch, steps 21--120 | `100` | **`3.230`** |

The progress-bar rate from the first few iterations is misleading because it
contains graph/cache warmup. Validation duration is also not a training-speed
metric. The 12-image smoke is an operational exception and must not be used for
scientific quality comparisons against the fixed 96-image runs. [code]

## 2. Implemented fixes

### 2.1 One telemetry synchronization

CL14_CA logs 19 loss and branched-attention diagnostics. The earlier loop
performed an accelerator gather and `.item()` synchronization for every value.
`PhotomakerLoraTrainer` now stacks the detached scalars, performs one gather,
one CPU transfer, and updates the same metric names. On a one-GPU run the
collective is bypassed completely. [code]

```python
# 12 Aug 2026 - Training optimization: gather every scalar in one
# vector and synchronize the GPU once.
loss_names = tuple(self.config.writer.loss_names)
local_scalars = torch.stack(
    [batch[name].detach().reshape(()) for name in loss_names]
).float()
if self.accelerator.num_processes == 1:
    gathered_matrix = local_scalars.unsqueeze(0)
else:
    gathered_matrix = self.accelerator.gather(local_scalars).reshape(
        -1, len(loss_names)
    )
mean_values = gathered_matrix.mean(dim=0).cpu().tolist()
```

### 2.2 Identity-token indices built once

All corrected residual-CA processors use the same prompt identity-token mask
within one U-Net call. The runtime now validates the nonempty equal token count
once, constructs a `[batch, active_tokens]` index tensor, and passes it to the
processors. Each CA layer uses `torch.gather` with those indices instead of
recomputing a boolean selection and synchronizing token-count assertions. The
fallback mask path remains for compatibility. [code]

### 2.3 Equivalent native-CA fusion

The residual processor formerly invoked the identical projection/SDPA/output
path separately for the target row and reference row. These rows have no
cross-row attention dependency, so they are concatenated on the batch axis,
processed once, and split afterward. This preserves the Q/K/V routing and
mathematical attention rows while reducing kernel-launch overhead. [code]

```python
# 12 Aug 2026 - Training optimization: native target/reference CA are
# independent batch rows, so fuse their identical projection/SDPA path.
native_hidden = torch.cat([target_hidden, reference_hidden], dim=0)
native_prompt = torch.cat([generation_prompt, identity_prompt], dim=0)
native_output = self._project_attention(
    native_hidden,
    native_prompt,
    query_projection=attn.to_q,
    key_projection=attn.to_k,
    value_projection=attn.to_v,
    heads=int(attn.heads),
)
native_target, native_reference = native_output.split(batch_size, dim=0)
```

### 2.4 CL14 operational substrate retained

The production source is derived from the proven running `CL14_CA_r7` package,
which itself follows CL14's loader and validation lifecycle: batch size `2`,
two data-loader workers, `pin_memory=false`, the existing training/validation
base-model swap, and processor reinstall sequence. [code] Later unrelated
CL15--CL20 trainer changes were not copied into this training substrate.

Speculative changes to pinned/persistent workers, nonblocking batch transfer,
and manual CPU/GPU re-homing were tested during diagnosis and removed. They
were not required for the measured gain and did not establish safe cold-start
training. [measured][code]

## 3. Corrected Eddie validation

The concern about Eddie was valid for the original CL14_CA r7 validation path.
The new run composes `all_metrics_subject_v2`, configures both dataset and
pipeline subject selection as `bbox_overlap_v2`, and supplies the same sealed
embedding asset as CL20. [code]

The launch gate requires asset SHA-256
`e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465`.
The versioned selector audit identifies Eddie as the only multi-face reference,
selects detector index `1` with bbox IoU `0.896066`, and selects index `0` for
the other eleven identities. [report] Thus both PhotoMaker conditioning and
`id_sim_subject_v2` refer to foreground Eddie in the new production run.

As an end-to-end startup sanity check, the optimized smoke's 12 Eddie rows have
mean intended-subject similarity `0.243214` at step 0, versus `0.097842` in the
original r7 table. [measured] This difference is not a model-quality comparison:
the repair deliberately changes both Eddie conditioning and the identity target.
It confirms that the new validation chain is no longer reproducing the old
background-face contract.

This is a validation-contract repair, not a model-training change: training
data, loss, initialization, gradients, optimizer, and residual-CA routing are
unchanged. The old r7 job remains live with its historical validation namespace
and should not be used for Eddie comparisons. [code]

## 4. Root cause and what is not established

Two issues were distinct:

1. The throughput gap to CL14 is primarily architectural: CL14_CA adds bounded
   residual identity-token cross-attention at 36 sites, while CL14 does not.
   Repeated logging synchronization and duplicated native-CA execution added
   avoidable overhead on top of that. [code][measured]
2. Completely skipping the initial validation exposed a reproducible native
   autograd segfault on the first backward pass. The same failure occurred with
   an exact copy of the healthy live-r7 source, so it was not introduced by the
   optimizations. One unchanged 12-item validation batch exercises the normal
   CL14 model-swap/reinstall lifecycle and safely enters training. [measured]

The exact low-level cause of the cold-start segfault is **not established**.
Evidence localizes it to bypassing the established validation lifecycle, but
does not prove which CUDA/PyTorch state transition is necessary. [measured]
It is also not established that the execution-only changes improve image
quality; that requires matched-checkpoint subject-v2 validation. [hypothesis]

## 5. Production experiment and decision gate

**Config:** `CL14_CA.yaml`, run `CL14_CA_optimized_r11`.

**Single scientific change versus CL14:** enable bounded residual
identity-token CA v3 in `up_blocks.0/1`; the throughput transformations are
execution-equivalent. [code]

**Hypothesis:** explicit identity-token CA increases subject identity while
preserving CL14 prompt adherence and face quality. [hypothesis]

**Prediction:** at matched checkpoints, `id_sim_subject_v2` improves without a
material prompt-similarity, face-quality, or alignment regression. [hypothesis]

**Risk:** the additional CA remains `~48%` slower per step than CL14 even after
optimization; Eddie scores from legacy-validation runs are not comparable.

**Decision gates:** compare the fixed 96-image panels at the same optimizer
steps, with primary scientific metric `id_sim_subject_v2`; reject an overall ID
regression greater than `0.01`, and inspect prompt similarity, the seven compact
face-quality curves, artifacts, and per-identity/per-prompt tails before any
promotion. [report]

The exact trainable contract is `2,348 tensors / 224,624,676 parameters` in the
model and optimizer. `pipeline.pose_adapt_ratio=0` and
`pipeline.ca_mixing_for_face=false`; seeds, prompts, references, bboxes,
RealVisXL validation base, DDIM 50, CFG 5, and 24k-step schedule remain fixed.
[code][measured]

## 6. Reproducing and auditing

Run from `diffusion_template/` with the project environment and machine-local
secrets loaded:

```bash
python tools/validate_CL14_CA_config.py \
  --config-name CL14_CA \
  --run-name CL14_CA_optimized_r11 \
  --experiment-spec experiments/cosmic_large/CL14_CA_optimized_r11.json

bash -n launchers/active/run_CL14_CA_24k_1gpu.sh \
  serv_run_packages/CL14_CA_relaunch_common/start_CL14_CA_variant_1gpu.sh \
  serv_run_packages/CL14_CA_optimized_r11/start_CL14_CA_optimized_r11_1gpu.sh

python ../local_scripts/serv_job.py check --branch test
python ../local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v23/CL14_CA_optimized_r11/diffusion_template/serv_run_packages/CL14_CA_optimized_r11/run_CL14_CA_optimized_r11_1gpu.yaml \
  --branch test
```

The sealed runtime is `runtime_sources_cl14_ca_v23`, source revision
`live-r7-v8+a65ffcb2c95f+ca-optimization+cl20-validation`, with manifest
SHA-256 `8d03af06bcc32b306ce3cbf83d56180573cac709467b02512b3cf817ef7999d5`.
Retrieve future metrics with the immutable production Comet key
`fafd7a61b06c4114b9dec2c21d29ca38`, not the display name. [measured]

## 7. Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Optimized latency is `3.230 s/it` | High | 100 consecutive warmed iterations; median progress rate and direct elapsed time agree. [measured] |
| Optimization preserves residual-CA routing | High | Localized source diff; Q/K/V ownership and 2,348/224,624,676 contract unchanged. [code][measured] |
| New validation selects foreground Eddie | High | CL20-compatible selector/config plus sealed asset SHA and versioned selector audit. [code][report] |
| Cold skip-validation failure is not caused by these optimizations | High | Exact healthy-r7 source reproduced the first-backward segfault when validation was bypassed. [measured] |
| Exact CUDA cause of cold-start segfault | Low | Reproducible lifecycle boundary, but no lower-level causal isolation. [measured] |
| CL14_CA will outperform CL14 scientifically | Low until matched checkpoints | Startup and throughput do not measure quality. [hypothesis] |

## Final live state

The optimized smoke completed its 12-image startup validation, wrote a 12-row
identity table, staged face-quality inputs, and logged finite losses at steps
`0`, `50`, and `100`; it was intentionally stopped after step 126. [measured]

The full production job completed all eight fixed-validation batches (`96/96`
images) in `25:18`, wrote the 96-row subject-v2 table, staged all 96 face-quality
inputs, restored the CL14 training base, and logged finite step-0 loss
`0.064709`. It advanced through at least optimizer step 10 at `3.23 s/it` and
remained Running when monitoring stopped. Its first 12 rows are Eddie and have
mean subject-v2 similarity `0.243214`, exactly matching the corrected smoke
table. The original `CL14_CA_r7` also remained Running and was never modified or
stopped. [measured]

## Evidence sources

- `src/trainer/sdxl_trainers.py`, `src/model/photomaker_branched/branched_runtime.py`, `src/model/photomaker_branched/identity_ca_processor_v2.py`, and `src/model/photomaker_branched/residual_identity_ca_processor_v3.py`. [code]
- `src/configs/large_dataset_joint_r128_24k.yaml`, `src/configs/metrics/all_metrics_subject_v2.yaml`, `src/datasets/manual_val.py`, and `src/face_subject_selector.py`. [code]
- Sealed Serv stdout/stderr, source manifest, and immutable Comet records for the jobs in the run ledger. [measured]
- `analysis/2026-08-12_CL14_CA_startup_failure_fix_and_relaunch.md`, `analysis/2026-08-09_eddie_validation_pre_vs_post_reference_fix.md`, and `docs/handoffs/LATEST.md`. [report]
