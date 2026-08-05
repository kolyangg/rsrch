# E11/E12 Large Dataset branched-attention capacity plan

**Date:** 4 August 2026

**Status:** deployed to an isolated Serv runtime and running

**Baseline:** `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1`

**Baseline immutable Comet key:** `5b5cbd1584184ce1a9032dd6fafb91c5`

**Decision metric:** full-96 mean and per-image ID similarity. Text quality is
not a decision metric for this pair; face-quality curves remain logged only for
protocol continuity and gross-failure detection.

## Outcome and priority

Implement these as two separate one-element experiments:

1. **E11** increases the rank of the existing hard branched-self-attention
   Q/K/V projections from 32 to 128. It tests whether the clean E0 plateau is
   primarily a self-attention BA capacity limit.
2. **E12** keeps the E0 rank-32 spatial BA unchanged and adds a new corrected,
   hard face-local target-query/PhotoMaker-identity-token cross-attention branch
   in `up_blocks.0` and `up_blocks.1`, at rank 256. It tests whether a
   comparable amount of additional capacity is more useful when allocated to
   explicit identity-token CA rather than spatial-reference SA.

The two arms have deliberately similar total trainable BA capacity:

| Arm | Existing spatial-SA BA | New capacity | Expected total BA |
|---|---:|---:|---:|
| Fixed E0 | 31,948,800 | 0 | 31,948,800 |
| E11 SA rank 128 | 127,795,200 | 0 | 127,795,200 |
| E12 SA rank 32 + corrected ID-CA rank 256 | 31,948,800 | 102,629,376 | 134,578,176 |

The E12 count assumes the current SDXL topology: 36 selected CA sites and four
rank-256 projections per site. Startup must derive and assert the count from
the installed model instead of trusting this document.

Do **not** combine E11 and E12. That would remove the useful answer about where
extra BA capacity belongs.

## Implementation status

Both experiments are implemented behind defaults-off toggles. Existing E0-E10
configuration gates still pass.

- E11 uses `model.ba_hard_v1_lora_rank=128`; the generic U-Net adapter remains
  rank 32 and frozen.
- E12 installs `HardIdentityCrossAttnProcessorV2` only at the 36 selected
  `up_blocks.0/1.attn2` sites. Legacy `BranchedCrossAttnProcessor` remains off.
- Strict ownership expects E11 `840 / 127,795,200` and E12
  `1,128 / 134,578,176` trainable tensors/parameters.
- Both configs resolve to 20k steps, full-96 validation every 2k, per-image ID
  tables, and Comet project `aug-large-ds`.
- The one-A100 start scripts and MLS YAMLs exist at the paths listed below.
- Both jobs run from the isolated runtime
  `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804`,
  so the six pre-existing jobs' runtime trees were not changed.
- E11 is MLS job `lm-mpi-job-f0ba530e-5398-4e45-982b-3e130ae0fca3` and Comet
  experiment `e748a5e136b3441688aaf968294612a1`.
- E12 is MLS job `lm-mpi-job-6590eff3-244c-4df7-bba5-1c1a5aaa9be4` and Comet
  experiment `d06ab51afbff4cacac1877632e26cf24`.
- The user authorized these two named one-GPU submissions to raise the
  project's temporary ceiling from six to eight requested A100s. The ceiling
  returns to six after E11/E12 finish or are removed.

## Shared immutable contract

Both experiments inherit
`src/configs/large_dataset_rhca_hard_v1_audited_20k.yaml` and preserve:

- adjusted Large Dataset training data and the same sampling policy;
- one A100 and one training process;
- batch size 2, LR `1e-4`, warmup, loss, seeds, prompts, reference images,
  target/reference bboxes, inference base, scheduler, 50 inference steps, and
  CFG exactly as fixed E0;
- 20,000 optimizer steps: `trainer.epoch_len=2000`, `trainer.n_epochs=10`;
- validation at step 0 and every 2,000 steps on all 96 `manual_val` items;
- Comet project `aug-large-ds`;
- per-image ID-similarity table/CSV logging at every validation event;
- schema-v2 exact trainable checkpointing and strict validation processor copy;
- hard reference-conditioned branched SA as the core face mechanism;
- `pipeline.pose_adapt_ratio=0.0` and
  `pipeline.ca_mixing_for_face=false` in training and validation;
- no generic `lora_adapter`, PhotoMaker `default`, text encoder, VAE, or other
  ordinary U-Net trainables;
- no change to face masks, reference masking, PhotoMaker token scale, or loss.

The user has explicitly deprioritized text and face-quality differences for
this direction. Still fail a run on NaNs, missing faces, empty ID-token masks,
broken doubled-batch routing, or a train/validation architecture mismatch.

## E11 — larger existing spatial-SA BA

### Proposed identity

- Run name: `E11_large_ds_ba_sa_r128_20k_full96_r1`
- Hydra config:
  `src/configs/E11_large_ds_ba_sa_r128_20k.yaml`
- Experiment record:
  `experiments/large_dataset/E11_large_ds_ba_sa_r128_20k_full96_r1.json`
- Serv package:
  `serv_run_packages/E11_large_ds_ba_sa_r128_20k_full96_r1/`
- Start script:
  `start_E11_large_ds_ba_sa_r128_20k_full96_r1_1gpu.sh`
- MLS YAML:
  `run_E11_large_ds_ba_sa_r128_20k_full96_r1_1gpu.yaml`

Suggested Comet comment:

> E11 vs fixed E0: raise only the six hard spatial-BA noise/reference Q/K/V
> LoRA projections from rank 32 to rank 128 at all 70 self-attention sites;
> test whether substantially larger core BA capacity raises full-96 ID
> similarity without training any non-BA adapter.

### Single scientific delta

Add an explicit hard-v1 BA rank selector and set it to 128:

```yaml
model:
  ba_hard_v1_lora_rank: 128
```

The selector should default to `null`, meaning “use the historical
`model.rank` value.” Do not set `model.rank=128`: that also changes the shape of
frozen generic adapters and makes the experiment artifact less clean even
though strict ownership should keep them frozen.

The rank applies to the existing processor-local:

- `noise_to_q`, `noise_to_k`, `noise_to_v`;
- `ref_to_q`, `ref_to_k`, `ref_to_v`.

It must not add `face_to_out`, branched CA, generic CA, or PhotoMaker adapter
ownership. E2 already answers the separate branch-output question.

### Expected ownership

The current rank-32 state contains 840 trainable tensors and 31,948,800
parameters. LoRA parameter count is linear in rank, so rank 128 should contain:

```text
branched SA tensors       840
branched SA parameters    127,795,200
optimizer tensors         840
optimizer parameters      127,795,200
all other trainables      0
```

Keep `lora_alpha=rank`, as the current `BranchLoRALinear` does, so the scaling
remains 1.0 and rank is the only scientific change.

### Implementation map

1. In `src/configs/model/photomaker_branched_lora2.yaml`, add
   `ba_hard_v1_lora_rank: null`.
2. In `src/model/photomaker_branched/lora2.py`, accept the optional selector,
   validate it as positive, store its resolved value, and include the explicit
   E11 value in the schema-v2 architecture manifest. Preserve the old manifest
   for `null` if required for historical checkpoint compatibility.
3. In `src/model/photomaker_branched/branched_runtime.py`, pass the resolved
   value as `branched_attn_lora_rank` only when constructing
   `hard_replace_v1` self-attention processors.
4. Do not change the newer `ba_ref_kv_rank`, `ba_output_rank`, or
   `ba_branch_q_rank` controls; they belong to other architecture versions.
5. Add the config and its exact expected ownership contract to
   `tools/validate_aug_large_ds_config.py`. Count only
   `model.ba_hard_v1_lora_rank=128` as the scientific delta.
6. Extend the shared E-series launcher, Serv source template, package builder
   mapping, audited runtime hashes, and experiment JSON in the established
   E7-E10 pattern.

### Binary interpretation

- A material ID-similarity improvement over fixed E0 means the existing
  spatial BA route was capacity-limited.
- No meaningful improvement by 20k, despite verified gradients and parameter
  updates, is strong evidence that simply making the same spatial route wider
  is not the main solution.
- Compare E11 with historical E0 as a performance target, but use fixed E0 as
  the causal baseline because historical E0 trained additional adapters.

## E12 — corrected identity-token branched CA capacity

### Proposed identity

- Run name: `E12_large_ds_ba_idca_up_r256_20k_full96_r1`
- Hydra config:
  `src/configs/E12_large_ds_ba_idca_up_r256_20k.yaml`
- Experiment record:
  `experiments/large_dataset/E12_large_ds_ba_idca_up_r256_20k_full96_r1.json`
- Serv package:
  `serv_run_packages/E12_large_ds_ba_idca_up_r256_20k_full96_r1/`
- Start script:
  `start_E12_large_ds_ba_idca_up_r256_20k_full96_r1_1gpu.sh`
- MLS YAML:
  `run_E12_large_ds_ba_idca_up_r256_20k_full96_r1_1gpu.yaml`

Suggested Comet comment:

> E12 vs fixed E0: keep rank-32 spatial BA unchanged and add a corrected hard
> face-local target-query/active-PhotoMaker-ID-token CA branch, rank 256 in
> up_blocks.0/1 only; test whether approximately 103M additional BA parameters
> improve full-96 ID similarity when allocated to explicit identity-token CA.

### Do not reactivate the legacy CA processor

Do not implement E12 by setting `disable_branched_ca=false` and
`train_branched_ca_lora=true` on the existing
`BranchedCrossAttnProcessor`.

Observed and inspected reasons:

- the matched historical Cosmic CA-on run scored ID `0.0351`, versus `0.1418`
  with CA disabled, and caused broader scene/body corruption;
- the current processor computes its face query from `ref_hidden`, not the
  target/noise hidden state;
- it concatenates `[hidden_bg, hidden_ref]`, so the target half receives the
  generation-prompt result while the identity-prompt result is returned to the
  reference half;
- it attends the 77-token face-prompt tensor even when most positions were
  zeroed, leaving irrelevant zero tokens in the softmax denominator;
- its ordinary output projection and whole-lane replacement are not face-local.

Legacy controls should remain:

```yaml
disable_branched_ca: true
train_branched_ca_lora: false
model:
  train_branched_ca_lora: false
```

Introduce a separately versioned corrected processor instead.

### Single scientific delta

The one E12 element is a corrected hard identity-token CA branch:

```yaml
model:
  ba_identity_ca_v2_enabled: true
  ba_identity_ca_v2_groups: [up_blocks.0, up_blocks.1]
  ba_identity_ca_v2_rank: 256
```

All four projections are parts of this one new branch, not separate experiment
toggles.

For the target half at an eligible `attn2` site:

```text
native_target = ordinary frozen PhotoMaker/text cross-attention

Q_id = id_to_q(target_hidden)
K_id = id_to_k(gathered active PhotoMaker identity tokens)
V_id = id_to_v(gathered active PhotoMaker identity tokens)
id_message = id_to_out(attention(Q_id, K_id, V_id))

target_output = native_target * (1 - target_face_mask)
              + id_message * target_face_mask
```

For the reference half, keep the ordinary frozen native cross-attention result.
This preserves the doubled reference stream used by spatial BA without letting
the new CA processor overwrite the reference lane.

The hard spatial merge is deliberate. It preserves the project's requirement
that `ca_mixing_for_face=false`: inside the target face the ID branch owns CA;
there is no learned or fixed interpolation with the native PhotoMaker/text CA
output. PhotoMaker identity tokens are evidence consumed by explicit branched
attention, not a PhotoMaker-generated face output mixed after attention.

Required invariants:

- target queries come from `target_hidden`, never `ref_hidden`;
- K/V contain only positions selected by `class_tokens_mask` from the
  PhotoMaker identity-token prompt lane;
- gather the active tokens before attention; do not multiply 75 positions by
  zero and leave them in the softmax;
- fail closed if a conditional training/validation sample has no active ID
  tokens or inconsistent token counts;
- the ID-CA message owns the target face while native CA owns the exterior;
- the native target CA remains frozen and unchanged;
- there is no gate, alpha, residual addition, or native/ID face interpolation;
- no text-quality objective, generic adapter, pose adaptation, or PhotoMaker
  face-output mixing is introduced.

### Why rank 256 and only two up-block groups

The previous architecture audit recommended corrected identity CA first in
`up_blocks.0` and `up_blocks.1`, because the historical all-layer CA path
caused global corruption. Restricting the branch spatially and by block protects
the scene while keeping the identity intervention near face synthesis.

Rank 256 over four projections at those 36 sites adds approximately 102.63M
parameters, making E12's total capacity close to E11 without changing spatial
SA rank. A rank-128 all-layer CA would have a similar raw count but would repeat
the risky all-layer intervention. Parameter matching is useful; blindly
maximizing rank is not.

Expected corrected-CA ownership under the current SDXL topology:

```text
selected CA sites                     36
Q/K/V/output LoRA tensors per site     8
corrected CA tensors                  288
corrected CA parameters       102,629,376

existing rank-32 SA tensors           840
existing rank-32 SA parameters 31,948,800

total trainable tensors             1,128
total trainable parameters    134,578,176
```

The projection count uses `cross_attention_dim=2048`, 30 rank-256 sites at
hidden size 1280, and 6 at hidden size 640. Treat these as an exact startup
assertion only after confirming the resolved model topology.

### Implementation map

1. Add a versioned processor, suggested path:
   `src/model/photomaker_branched/identity_ca_processor_v2.py`.
   It should own `id_to_q`, `id_to_k`, `id_to_v`, and `id_to_out`, expose
   `named_ba_trainables()`, and provide setters for the target mask and current
   `class_tokens_mask`.
2. Reuse or extract `BranchLoRALinear` without changing historical processor
   behavior. Q and output are square hidden projections; K/V are
   2048-to-hidden projections. Keep trainable parameters in the audited BA
   dtype and frozen effective base weights as buffers.
3. In `lora2.py`, add defaults-off constructor/config controls, validate the
   group names/rank, and serialize the complete identity-CA contract in the
   schema-v2 manifest.
4. In `branched_runtime.py`, install the corrected processor only at the
   selected `attn2` names while leaving the legacy branched-CA path disabled.
   On every patch call—not only first installation—refresh both masks and
   `class_tokens_mask`; the current reuse path does not refresh the latter.
5. Ensure training and validation pass the same PhotoMaker prompt lane and
   exact class-token mask. Correctly expand masks for batch, CFG, and
   `num_images_per_prompt`; fail closed if any routed row has no active token
   rather than inventing identity evidence or silently reverting to native CA.
6. Extend strict ownership in `lora2_helpers.py` through processor-declared
   roles. Do not reopen the historical substring-based attn2
   `ref_to_*/noise_to_*` allowlist.
7. Save and load every corrected-CA tensor through schema-v2 with exact
   processor names, shapes, selected groups, and rank. Strict
   validation processor copying must install the same class and load the same
   state rather than silently restoring native or legacy CA.
8. Add telemetry for ID-token count, ID-message RMS, and the native-versus-ID
   face-output RMS ratio by selected block group. These are diagnostics, not
   decision metrics.
9. Add the config and exact expected contract to
   `tools/validate_aug_large_ds_config.py`. The corrected identity-CA enablement
   is the sole scientific delta; its rank/groups are required members of that
   element.
10. Extend the shared E-series launcher, Serv source template, package builder
    mapping, audited hashes, and experiment JSON exactly as for E11.

### Binary interpretation

- E12 above fixed E0 means a face-local target-query identity-token CA route is
  useful even when spatial-SA BA rank remains 32.
- E12 above E11 means extra parameters are more useful in an explicit
  identity-token CA route than in widening spatial reference Q/K/V.
- E11 above E12 means spatial reference attention benefits more from capacity,
  or the corrected CA token route is still too compressed/poorly routed.
- Neither above fixed E0 means parameter count is not the main bottleneck;
  prioritize correspondence, sampling/loss causality, or better reference
  evidence rather than rank 256/full-rank escalation.

## Minimal verification before packaging

Do focused checks only:

1. Hydra composition proves 20k steps, full-96/2k validation, one scientific
   delta, project/name/comment, and all immutable baseline fields.
2. Processor installation lists exactly 70 hard SA sites for both arms and
   exactly 36 corrected identity-CA sites for E12; legacy branched CA count is
   zero.
3. Exact requires-grad and optimizer membership matches the contracts above;
   generic/default adapters have zero trainables.
4. A single synthetic E11 forward/backward proves rank-128 A/B gradients and
   unchanged routing.
5. A single synthetic E12 processor check proves:
   target query use, active-token gathering, native CA outside the target mask,
   ID-only CA inside it, ordinary reference-lane CA, finite gradients for
   Q/K/V/output, and no legacy processor.
6. Schema-v2 save/load round-trip exactly reproduces all new tensors and fails
   on rank/group/processor mismatch.
7. Training-to-validation processor transfer retains the new E11 rank or E12
   CA class/state and produces identical deterministic output before/after the
   transfer.
8. Shell syntax, source hashes, experiment JSON, package tokens, and MLS YAML
   contract pass. Do not run broad or vanity suites.

The real-model startup gate on Serv must independently print and assert the
resolved trainable tensor/parameter counts before the run registers as ready.

## Packaging and launch state

Each package requests one A100. The launch procedure was:

1. confirm exactly six project one-GPU jobs were Running/Pending;
2. apply the user's named eight-GPU exception for E11/E12;
3. create and sync an isolated `test` worktree without touching live jobs;
4. verify both configs/specs, one-GPU YAMLs, shell syntax, and all 19 audited
   runtime SHA-256 hashes;
5. submit each exact YAML through `local_scripts/serv_job.py submit` with its
   hypothesis comment;
6. verify the online Comet record and real-model trainable contract.

The first E11/E12 MLS attempts (`lm-mpi-job-4c25106f-...` and
`lm-mpi-job-7a8dfc1d-...`) failed before Python/Comet because the copied
full-96 metadata landed one directory too deep. Their empty logs and failed
audit records are preserved. After correcting only that isolated deployment
path, the successful jobs above passed E11's exact `840 / 127,795,200` and
E12's exact `1,128 / 134,578,176` trainable/optimizer gates. No scientific
configuration changed between the failed attempts and the running jobs.
