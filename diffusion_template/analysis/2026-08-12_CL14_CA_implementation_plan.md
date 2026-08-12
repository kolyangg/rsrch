# CL14_CA: exact residual branched cross-attention implementation plan

**Date:** 12 August 2026  
**Status:** implementation-ready design; no code has been changed and no run has
been launched by this report  
**Control:** `CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1`  
**Control Comet key:** `6fe0028be92242c38056b3d36665fdd6`  
**New run name:** exactly `CL14_CA`

## 1. Decision

Implement `CL14_CA` as a fresh, single-delta retraining of CL14 with the
corrected residual identity cross-attention v3 processor enabled at
`up_blocks.0` and `up_blocks.1`. Do **not** resume the trained CL14 24k
checkpoint, do not revive the historical `BranchedCrossAttnProcessor`, and do
not enable `ca_mixing_for_face`. `[code] [report]`

Here, “CL14 as a base” means:

- inherit the complete CL14 configuration and initialization;
- retain the same training data, masks, optimizer, schedule, 24k budget,
  validation model, prompts, seeds, references, boxes, and metrics;
- add only a bounded, face-local ID-token CA residual plus its non-causal
  observability logging;
- compare `CL14_CA` against the immutable CL14 run above at the same optimizer
  steps, with 24k as the pre-registered primary endpoint.

Starting from CL14's **trained** 24k checkpoint would instead be a 24k+X
fine-tune and would not isolate CA. If checkpoint continuation is desired, it
should be a separately named experiment and must not be reported as
`CL14_CA versus CL14`.

This is worth running as a clean architectural ablation. E17 showed that this
residual formulation is visually safe, but it was `-0.00599` against its E15
base and its branch telemetry was not logged. E17 therefore lowers the prior
probability of a gain; it does not answer the CL14 question because it used a
different dataset/substrate and a persist-trained PhotoMaker-default path that
was itself weak. `[measured] [report]` The expected outcome should consequently
be treated as uncertain rather than assumed positive.

## 2. What “cross-attention” means in this run

CL14 already retains ordinary SDXL/PhotoMaker cross-attention. The disabled
component is the additional branched `attn2` route. `CL14_CA` should install the
corrected residual route below, not the legacy replacement route:

```text
batch layout     = [target B, reference B]

native_target    = CA(target hidden Q, full generation prompt K/V)
native_reference = CA(reference hidden Q, full identity prompt K/V)

active_id_tokens = identity_prompt[class_tokens_mask]
id_message       = CA(target hidden Q, active PhotoMaker ID-token K/V)
id_delta         = rank64_zero_init_output(id_message)
gate             = 0.20 * sigmoid(gate_logit)        # starts at 0.02

target_output    = native_target
                 + target_face_mask * gate * rms_normalize(id_delta)
reference_output = native_reference
```

The important invariants are:

1. The target lane supplies the identity branch queries.
2. Only active PhotoMaker identity tokens supply its K/V.
3. Native PhotoMaker/text CA remains the complete base path inside and outside
   the face.
4. The added message is target-face-local, bounded, and zero at initialization.
5. The reference half is unchanged native CA.
6. `pose_adapt_ratio=0` and `ca_mixing_for_face=false` remain fixed.

The current working tree already implements this equation in
[`residual_identity_ca_processor_v3.py`](../src/model/photomaker_branched/residual_identity_ca_processor_v3.py),
installs it through
[`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py),
declares its trainables in
[`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py), and
stores the route in the checkpoint architecture manifest in
[`lora2.py`](../src/model/photomaker_branched/lora2.py). `[code]`

## 3. Source baseline and minimal port

> **Launch correction, 12 August 2026.** The historical CL14 directory's
> original manifest no longer verifies because its completed run rewrote 96
> `hm_debug` images and added Hydra output files. The CL14_CA v2 package excludes
> those generated artifacts, preserves every common source hash, and adds only
> the six config/validation/launch files described here. The sealed CL14 source
> also predates live subject-v2 metric composition, so CL14_CA intentionally
> retains CL14's live legacy metrics and applies the same sealed subject-v2
> backfill to immutable validation images after generation. This avoids porting
> later trainer/face-ownership code into the causal architecture comparison.

Use the immutable CL14 runtime snapshot, not an arbitrary copy of the current
dirty worktree:

```text
runtime_sources_cl1_cl3_v1/
  CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1
```

The CL14 experiment record identifies this as a sealed 1,220-file snapshot
based on `c04970f3+cl12-cl14-snapshot-v1-20260809`. The plain Git commit
`c04970f3` does not contain the residual identity-CA files, while the vetted
E17 v3 implementation and its zero-RMS-gradient fix came from the separate
`8b8b9ab` and `1a88f6a` lineage. `[code] [report]`

Therefore the packaging procedure must be:

1. Copy the sealed CL14 snapshot to a new immutable `CL14_CA` runtime source.
2. Verify the copied source manifest before any edit.
3. Check whether the snapshot's uncommitted sealed additions already include
   residual identity CA v3. Do not infer this from the base Git hash.
4. If absent, port only the following v3-related code from the vetted current
   implementation:

   - the residual identity-CA processor and its small projection helpers;
   - the five model configuration fields;
   - corrected-CA selection/installation and class-token-mask refresh;
   - explicit trainable enumeration and checkpoint-manifest fields;
   - telemetry collection and forwarding into the loss result.

5. Do not wholesale copy the current `lora2.py`, runtime, trainer, or dataset
   files into the CL14 source; those files contain unrelated work after CL14.
6. Generate a new source manifest after the minimal port and make the start
   script verify it before credentials are loaded.

The architecture port is complete only when a source diff against the sealed
CL14 snapshot contains the CA implementation, the `CL14_CA` configuration and
launch plumbing, and no unrelated training/inference behavior.

## 4. Exact processor behavior

The core merge in the vetted implementation is already the desired code:

```python
native_target = self._project_attention(
    target_hidden,
    generation_prompt,
    query_projection=attn.to_q,
    key_projection=attn.to_k,
    value_projection=attn.to_v,
    heads=int(attn.heads),
)
native_reference = self._project_attention(
    reference_hidden,
    identity_prompt,
    query_projection=attn.to_q,
    key_projection=attn.to_k,
    value_projection=attn.to_v,
    heads=int(attn.heads),
)
native_target = attn.to_out[1](attn.to_out[0](native_target))
native_reference = attn.to_out[1](attn.to_out[0](native_reference))

token_mask = self._expanded_token_mask(
    batch_size=batch_size,
    token_count=identity_prompt.shape[1],
    device=identity_prompt.device,
)
token_counts = token_mask.sum(dim=1)
if bool((token_counts <= 0).any()) or int(torch.unique(token_counts).numel()) != 1:
    raise RuntimeError(
        "Residual identity CA requires equal, nonzero active ID-token counts"
    )
active_count = int(token_counts[0].item())
gathered_identity = identity_prompt[token_mask].reshape(
    batch_size, active_count, identity_prompt.shape[-1]
)

identity_hidden = self._project_attention(
    target_hidden,
    gathered_identity,
    query_projection=attn.to_q,
    key_projection=attn.to_k,
    value_projection=attn.to_v,
    heads=int(attn.heads),
)
identity_delta = self.id_delta_out(identity_hidden)

# The clamp is required: sqrt'(0) caused NaN gradients in the first E17
# attempt even though the zero-init forward value was finite.
delta_rms = (
    identity_delta.float()
    .square()
    .mean(dim=-1, keepdim=True)
    .clamp_min(self.rms_epsilon**2)
    .sqrt()
)
normalized_delta = identity_delta / delta_rms.to(identity_delta.dtype)
gate = torch.sigmoid(self.gate_logit) * self.gate_max

target_mask = self._prepare_spatial_mask(
    target_len=target_hidden.shape[1],
    batch_size=batch_size,
    device=native_target.device,
    dtype=native_target.dtype,
)
residual_message = (
    target_mask
    * gate.to(native_target.dtype)
    * normalized_delta.to(native_target.dtype)
)
target_output = native_target + residual_message
hidden_states = torch.cat([target_output, native_reference], dim=0)
```

Do not substitute E12's hard equation:

```text
native * (1 - mask) + id_message * mask
```

That equation lets an ID-only message replace all native face CA and caused the
large E12 failure and face-plate artifacts. `[measured] [code]`

## 5. Exact Hydra configuration

Create `src/configs/CL14_CA.yaml` with the following complete content:

```yaml
defaults:
  - CL14_cosmic_joint_shadow_sa128_softmask_24k
  - _self_

# CL14_CA = CL14 + one corrected residual identity-token cross-attention path.
# Native PhotoMaker/text CA and CL14's spatial BA remain intact.
model:
  ba_identity_ca_v2_enabled: false
  ba_residual_identity_ca_v3_enabled: true
  ba_residual_identity_ca_v3_groups: [up_blocks.0, up_blocks.1]
  ba_residual_identity_ca_v3_rank: 64
  ba_residual_identity_ca_v3_gate_init: 0.02
  ba_residual_identity_ca_v3_gate_max: 0.20
  train_branched_ca_lora: false

# These refer only to the unsafe legacy BranchedCrossAttnProcessor.
# They must remain off while the separately versioned residual processor is on.
disable_branched_ca: true
train_branched_ca_lora: false

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false

expected_trainable_contract:
  enabled: true
  total_tensors: 2348
  total_parameters: 224624676
  optimizer_tensors: 2348
  optimizer_parameters: 224624676
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    residual_identity_ca_r64:
      name_substring: ".attn2.processor."
      tensors: 108
      parameters: 5406756
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

# E17 computed these values but failed to include them in writer.loss_names.
# List every emitted up0/up1/all aggregate so branch use is observable.
writer:
  loss_names:
    - loss
    - ba/identity_ca_token_count/up0
    - ba/identity_ca_token_count/up1
    - ba/identity_ca_token_count/all
    - ba/identity_ca_delta_rms/up0
    - ba/identity_ca_delta_rms/up1
    - ba/identity_ca_delta_rms/all
    - ba/identity_ca_gate/up0
    - ba/identity_ca_gate/up1
    - ba/identity_ca_gate/all
    - ba/identity_ca_native_face_rms/up0
    - ba/identity_ca_native_face_rms/up1
    - ba/identity_ca_native_face_rms/all
    - ba/identity_ca_residual_face_rms/up0
    - ba/identity_ca_residual_face_rms/up1
    - ba/identity_ca_residual_face_rms/all
    - ba/identity_ca_residual_native_ratio/up0
    - ba/identity_ca_residual_native_ratio/up1
    - ba/identity_ca_residual_native_ratio/all
  experiment_comment: >-
    CL14_CA vs immutable CL14 adds only the corrected rank64 residual
    identity-token CA v3 at up_blocks.0/1. Target queries attend active
    PhotoMaker ID-token K/V through a zero-initialized output delta, a gate
    initialized at 0.02 and bounded by 0.20, and the target face mask. Native
    PhotoMaker/text CA remains intact. Legacy branched CA, pose adaptation,
    and ca_mixing_for_face remain disabled. Primary endpoint is subject-v2
    ID similarity at matched step 24k on the fixed manual_val 96.
```

The ownership values are the known 36-site E17 v3 values added to CL14's
`2,240 / 219,217,920` contract:

```text
CL14                         2,240 tensors   219,217,920 parameters
residual CA v3 addition        108 tensors     5,406,756 parameters
CL14_CA expected total       2,348 tensors   224,624,676 parameters
```

These values are a fail-closed expectation, not a substitute for measurement.
The actual composed model and optimizer must re-derive and match them before
training.

## 6. Runtime installation requirements

In `patch_unet_attention_processors`, select corrected CA names independently
of the legacy `disable_branched_ca` flag:

```python
identity_ca_names = [
    name
    for name in pipeline.unet.attn_processors
    if name.endswith("attn2.processor")
    and any(
        name.startswith(f"{group}.")
        for group in pipeline.ba_residual_identity_ca_v3_groups
    )
]

if name in set(identity_ca_names):
    proc = ResidualIdentityCrossAttnProcessorV3(
        hidden_size=hidden_size,
        cross_attention_dim=int(cross_attention_dim),
        rank=pipeline.ba_residual_identity_ca_v3_rank,
        gate_init=pipeline.ba_residual_identity_ca_v3_gate_init,
        gate_max=pipeline.ba_residual_identity_ca_v3_gate_max,
        trainable_dtype=torch.float32,
    )
    proc.init_from_attention(_resolve_attn_module(pipeline.unet, name))
    proc = proc.to(pipeline.device)
    proc.set_masks(target_mask, reference_mask)
    proc.set_class_tokens_mask(class_tokens_mask)
elif disable_branched_ca:
    new_procs[name] = pipeline._original_attn_processors[name]
```

On every subsequent forward, refresh both masks and the current
`class_tokens_mask`; prompt token membership can change with the batch and CFG
layout. Processor reuse must fail if the installed corrected-CA name set does
not exactly equal the configured set.

Before accepting startup, assert:

```python
from src.model.photomaker_branched.residual_identity_ca_processor_v3 import (
    ResidualIdentityCrossAttnProcessorV3,
)
from src.model.photomaker_branched.attn_processor_cleanest import (
    BranchedCrossAttnProcessor,
)

processors = model.unet.attn_processors
residual_ca = {
    name: proc
    for name, proc in processors.items()
    if isinstance(proc, ResidualIdentityCrossAttnProcessorV3)
}
legacy_ca = {
    name: proc
    for name, proc in processors.items()
    if isinstance(proc, BranchedCrossAttnProcessor)
}

assert len(residual_ca) == 36, sorted(residual_ca)
assert not legacy_ca, sorted(legacy_ca)
assert all(
    name.startswith(("up_blocks.0.", "up_blocks.1."))
    for name in residual_ca
)
```

Also assert that hard identity CA v2 has zero installed instances. The
checkpoint manifest must record processor code version 4, all 36 names, rank,
gate bounds, target-Q/active-ID-KV routing, residual merge, and zero-init output.

## 7. Config/spec validator

Add `tools/validate_CL14_CA_config.py`. Its important behavior should be a
fail-closed recursive comparison against CL14: every difference must live under
one of these paths and must then match the exact values above.

```python
ALLOWED_DIFFS = {
    "model.ba_identity_ca_v2_enabled",
    "model.ba_residual_identity_ca_v3_enabled",
    "model.ba_residual_identity_ca_v3_groups",
    "model.ba_residual_identity_ca_v3_rank",
    "model.ba_residual_identity_ca_v3_gate_init",
    "model.ba_residual_identity_ca_v3_gate_max",
    "expected_trainable_contract",
    "writer.loss_names",
    "writer.experiment_comment",
}

def allowed(path: str) -> bool:
    return any(path == root or path.startswith(root + ".") for root in ALLOWED_DIFFS)

with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
    base = compose(
        config_name="CL14_cosmic_joint_shadow_sa128_softmask_24k",
        overrides=["writer=cometml"],
    )
    candidate = compose(config_name="CL14_CA", overrides=["writer=cometml"])

drift = {
    path: (base_flat.get(path, "<missing>"), candidate_flat.get(path, "<missing>"))
    for path in sorted(set(base_flat) | set(candidate_flat))
    if base_flat.get(path, "<missing>") != candidate_flat.get(path, "<missing>")
    and not allowed(path)
}
if drift:
    raise RuntimeError(f"CL14_CA has non-CA drift from CL14: {drift}")
```

The validator must additionally require:

```python
require(args.run_name, "CL14_CA")
require(candidate, "disable_branched_ca", True)
require(candidate, "train_branched_ca_lora", False)
require(candidate, "pipeline.pose_adapt_ratio", 0.0)
require(candidate, "pipeline.ca_mixing_for_face", False)
require(candidate, "model.ba_architecture_version", "hard_replace_v1")
require(candidate, "model.ba_training_mask_feather", 2)
require(candidate, "model.ba_residual_identity_ca_v3_groups",
        ["up_blocks.0", "up_blocks.1"])
require(candidate, "trainer.epoch_len", 2000)
require(candidate, "trainer.n_epochs", 12)
require(candidate, "trainer.validation_interval_steps", 2000)
require(candidate, "datasets.val.manual_val.limit", 96)
require(candidate, "validation_args.num_images_per_prompt", 1)
require(candidate, "validation_args.num_inference_steps", 50)
require(candidate, "expected_trainable_contract.total_tensors", 2348)
require(candidate, "expected_trainable_contract.total_parameters", 224624676)
```

It must also parse the experiment JSON and require exact run name, config,
launcher, one Serv A100, `aug-large-ds`, and the immutable CL14 baseline key.

## 8. Experiment record and launcher

Create `experiments/cosmic_large/CL14_CA.json`:

```json
{
  "schema_version": 1,
  "run_name": "CL14_CA",
  "plan": {
    "status": "planned",
    "machine": "serv",
    "gpus": 1,
    "objective": "Measure the causal value of corrected residual identity cross-attention on CL14.",
    "baseline": "CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1",
    "baseline_comet_experiment_key": "6fe0028be92242c38056b3d36665fdd6",
    "single_scientific_change": "Enable zero-initialized bounded residual identity CA v3 at up_blocks.0/1, rank64, gate 0.02..0.20; telemetry only otherwise.",
    "config": "src/configs/CL14_CA.yaml",
    "launcher": "launchers/active/run_CL14_CA_24k_1gpu.sh",
    "comet_project": "aug-large-ds",
    "fixed_contract": "Fresh CL14 initialization; 24k steps; batch 2; fixed manual_val 96 at step 0 and every 2k; identical validation model, DDIM50, CFG5, seed0, prompts, references, boxes and subject-v2 metrics; pose_adapt_ratio=0; ca_mixing_for_face=false.",
    "primary_metric": "Mean subject-v2 ID similarity at matched step 24000; paired by the fixed 96 output keys.",
    "risk": "The residual may remain unused, duplicate identity already carried by native PhotoMaker CA, or reintroduce face-local artifacts. E17 was safe but did not improve its own base.",
    "expected_trainable_contract": "2348 tensors / 224624676 parameters"
  }
}
```

Create a dedicated `launchers/active/run_CL14_CA_24k_1gpu.sh` by retaining the
sealed checks and startup Comet-key loop from
[`run_CL15_CL20_hardcases_24k_1gpu.sh`](../launchers/active/run_CL15_CL20_hardcases_24k_1gpu.sh),
with these exact substitutions:

```diff
-: "${RUN_NAME:?Set the unique CL15-CL20 run name}"
-: "${CONFIG_NAME:?Set the matching CL15-CL20 config name}"
+: "${RUN_NAME:?Set RUN_NAME=CL14_CA}"
+: "${CONFIG_NAME:?Set CONFIG_NAME=CL14_CA}"

-case "${CONFIG_NAME}" in
-  CL15_...|CL16_...|CL17_...|CL18_...|CL19_...|CL20_...) ;;
-  *) ... ;;
-esac
+[[ "${RUN_NAME}" == "CL14_CA" ]] || exit 2
+[[ "${CONFIG_NAME}" == "CL14_CA" ]] || exit 2

-python tools/validate_CL15_CL20_config.py \
+python tools/validate_CL14_CA_config.py \
   --config-name "${CONFIG_NAME}" \
   --run-name "${RUN_NAME}" \
   --experiment-spec "${EXPERIMENT_SPEC_PATH}"

-if [[ "${CONFIG_NAME}" == CL20_* ]]; then
-  ... CL20 curriculum preflight ...
-else
-  python tools/datasets/preflight_cosmic_cl.py ...
-fi
+python tools/datasets/preflight_cosmic_cl.py \
+  --config-name "${CONFIG_NAME}" \
+  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
+  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"
```

Keep the existing fixed validation-file hashes, subject-v2 embedding hash,
one-process Accelerate launch, immutable Comet registration check, and deferred
face-quality finalization. The training command is:

```bash
accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  --config-name=CL14_CA \
  writer=cometml \
  writer.run_name=CL14_CA \
  writer.project_name=aug-large-ds
```

The launcher must reject ad-hoc Hydra arguments.

## 9. Verification before the 24k run

### 9.1 Static gates

Run from `diffusion_template/` in `photomaker_NS`:

```bash
python tools/validate_CL14_CA_config.py \
  --config-name CL14_CA \
  --run-name CL14_CA \
  --experiment-spec experiments/cosmic_large/CL14_CA.json

python -m py_compile \
  src/model/photomaker_branched/residual_identity_ca_processor_v3.py \
  src/model/photomaker_branched/branched_runtime.py \
  src/model/photomaker_branched/lora2.py \
  src/model/photomaker_branched/lora2_helpers.py \
  tools/validate_CL14_CA_config.py

bash -n launchers/active/run_CL14_CA_24k_1gpu.sh
```

### 9.2 Focused processor smoke

Before scheduling the long run, perform one deterministic doubled-batch
forward/backward and require:

- exactly 36 residual v3 processors, all in `up_blocks.0/1`;
- zero legacy `BranchedCrossAttnProcessor` and zero hard identity-CA v2;
- the target query changes the ID message while perturbing the reference query
  does not;
- non-ID prompt tokens cannot enter the gathered identity K/V set;
- zero-initialized `id_delta_out.lora_B` makes the residual message exactly
  zero;
- outside-face output equals native output;
- reference-lane output equals native output;
- first backward has finite gradients; the zero-initialized output-B gradient
  must be live, while a zero first-step gate/A gradient is not by itself a
  failure;
- `2348 / 224624676` trainables equal optimizer membership exactly;
- save/load reproduces all 36 processors, their gates, and delta weights.

The RMS telemetry at initialization is expected to show the clamped epsilon
(`~1e-6`) rather than mathematical zero; the decisive zero-init signals are
`identity_ca_residual_face_rms=0` and
`identity_ca_residual_native_ratio=0`.

### 9.3 Step-zero comparability gate

Generate the fixed 96 at step zero before the first optimizer update. Because
v3's residual is zero, `CL14_CA` should reproduce CL14's output. However, the
processor recomputes the native CA path, and the CA code is being ported across
source lineages, so equality must be measured rather than assumed.

Pass only if:

- all 96 filenames and input protocol hashes match CL14;
- pixels are byte-identical, or a numerical tolerance was pre-registered and
  the measured difference is within it;
- subject-v2 ID, text, alignment, and face-quality values match within only
  the consequences of that pre-registered numerical tolerance;
- no unexplained configuration/source diff remains.

Stop before training on unexplained step-zero drift. Do not rationalize the
drift after seeing later metrics.

### 9.4 Live startup gate

After submission, require within the launcher timeout:

```text
saved/CL14_CA/comet_experiment.json
```

It must contain a new 32-character immutable experiment key, run name
`CL14_CA`, and project `aug-large-ds`. Record the key and URL back into
`experiments/cosmic_large/CL14_CA.json`.

## 10. Evaluation contract

The control result is immutable CL14 key
`6fe0028be92242c38056b3d36665fdd6`. Its corrected subject-v2 identity is
`0.456116` at 24k and peaks at `0.457096` at 22k. `[measured]`

### Primary endpoint

Compare mean subject-v2 ID at **matched step 24k** on the fixed 96, paired by
output key:

```text
delta_ID_24k = ID(CL14_CA, 24k) - 0.456116
```

Promotion requires `delta_ID_24k >= +0.010` and a paired 96-image analysis that
does not show the gain coming only from selector changes, detection failures,
or a few outliers. The step-grid peak is secondary and must not replace the
pre-registered 24k endpoint.

### Guardrails

At the same matched step require:

- text similarity no worse than CL14 by more than `0.20`;
- mask IoU no worse by more than `0.005`;
- `96/96` expected validation rows and no increase in no-face outputs;
- no material regression in the seven face-quality curves;
- blind visual review showing no increase in face plates, seams, doubled
  glasses/goggles, incorrect expression, hand-eye fusion, pose drift, or
  person/layout changes.

### Early gates

- **2k/4k:** no NaNs, complete 96 panels, live bounded residual telemetry, and
  no E12-like face replacement artifacts.
- **8k:** continue only if ID is neutral within `0.005` of CL14 at the matched
  step, or a predeclared hard identity/occlusion slice has a credible visual
  gain; the residual/native ratio must be nonzero but bounded.
- **24k:** apply the primary promotion rule above.

Log these telemetry curves by `up0`, `up1`, and `all`:

```text
identity_ca_token_count
identity_ca_delta_rms
identity_ca_gate
identity_ca_native_face_rms
identity_ca_residual_face_rms
identity_ca_residual_native_ratio
```

A safe-looking result with a residual/native ratio effectively at zero is a
**no-op result**, not evidence that active CA is safe or useful.

## 11. What must remain unchanged

| Contract item | CL14_CA value |
|---|---|
| Initialization | Fresh CL14 initialization; no CL14 checkpoint resume |
| Training dataset | Exact CL14 Cosmic Large manifest/root and reference policy |
| Batch | 2, one A100 |
| Budget | 24,000 optimizer steps |
| Validation | Step 0 and every 2,000; fixed `manual_val` 96; one image/item |
| Validation model/sampler | Exact CL14 model, DDIM 50, CFG 5, seed 0 |
| Inputs | Same prompts, references, boxes/masks, and subject-v2 embeddings |
| Spatial BA | CL14 hard-v1 rank 128, unchanged |
| Generic adapter | Effective rank 32, unchanged |
| PhotoMaker default adapter | Effective rank 64 shadow path, unchanged |
| Training mask | CL14 feather 2, unchanged |
| Legacy branched CA | Disabled |
| Hard identity CA v2 | Disabled |
| `pose_adapt_ratio` | `0.0` |
| `ca_mixing_for_face` | `false` |
| New residual CA | up0/up1, rank 64, gate 0.02..0.20 |

## 12. Risks and not-established claims

- E17 is prior evidence against expecting a large average gain, though its
  substrate and missing telemetry prevent a clean transfer to CL14.
- Native PhotoMaker CA already consumes identity-conditioned prompt tokens; the
  residual may duplicate rather than complement that signal.
- Zero initialization protects the starting output but can also produce an
  effectively dead branch.
- A face-local identity gain can still harm expression, accessories, anatomy,
  or boundary integration.
- This experiment answers whether residual identity CA adds value to **CL14**.
  It does not answer whether CA adds value to CL19, the stronger soft-router
  architecture, or to CL20's different training curriculum.
- It is not established in advance that `CL14_CA` will beat CL14. That is the
  experimental hypothesis.

## 13. Confidence

| Claim | Confidence | Basis |
|---|---:|---|
| Residual v3 is safer than legacy/hard replacement CA | High | Routing equation, zero initialization, bounded gate, and E17 visuals `[code] [measured]` |
| The proposed config isolates CA from CL14 | High, conditional | Exact inheritance plus fail-closed diff gate; conditional on using the sealed CL14 source `[code]` |
| Expected ownership is `2348 / 224624676` | High | CL14 and E17 exact startup contracts; must be re-derived `[code]` |
| Step-zero should match CL14 | Medium-high | Zero residual, but native CA reimplementation/source port requires empirical parity `[code]` |
| CL14_CA will improve identity by at least 0.01 | Low-medium | Architecturally plausible; E17 did not show a mean gain `[hypothesis] [measured]` |

## 14. Implementation checklist

1. Clone and verify the immutable CL14 source snapshot.
2. Port only residual identity-CA v3 and its required install/ownership/
   checkpoint/telemetry hooks.
3. Add `src/configs/CL14_CA.yaml` exactly as above.
4. Add the fail-closed config/spec validator.
5. Add `experiments/cosmic_large/CL14_CA.json`.
6. Add the dedicated active launcher and immutable source manifest.
7. Pass composition, compile, shell, processor, ownership, backward, and
   checkpoint round-trip checks.
8. Pass the 96-image step-zero CL14 parity gate.
9. Check current Serv Running/Pending allocations against the project GPU
   ceiling, then submit one A100 only after all gates pass.
10. Verify and record the new immutable Comet key during startup.
11. Evaluate matched 2k increments and apply the pre-registered 24k decision.

No production code, experiment package, Comet run, or Serv job was created by
this document.
