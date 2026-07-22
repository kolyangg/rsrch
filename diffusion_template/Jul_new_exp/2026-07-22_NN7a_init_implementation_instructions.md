# NN7a_init — warm-started branched-attention initialization

**Purpose:** implementation handoff for another coding agent  
**Base branch:** current `main_clean`  
**Base experiment:** `one_id_ba_NN7a_clean_patch_takeover_up1`  
**New experiment name:** `NN7a_init`  
**Repository mutation by this review:** none  
**Do not overwrite or silently change the existing NN7a run. Create this as an additive, strictly checkpoint-separated variant. Do not push unless explicitly requested.**

---

## 1. Objective

Implement a controlled NN7a variant that begins with a **meaningful, nonzero reference-attention path**, rather than:

```text
current NN7a:
    random Xavier clean-patch K/V
    × effective gate ≈ 0.02
```

or:

```text
NN5/NN6:
    zero connector-up
    → exact PhotoMaker at initialization
```

The intended initialization is:

```text
clean CLIP face patches
→ frozen pretrained PMv2 Perceiver patch projection
→ PhotoMaker-aware sibling attn2 K/V initialization
→ target-query/local-reference attention
→ direct candidate takeover with effective alpha = 0.05
```

The result must retain all current NN7a safety mechanisms:

- ordinary target self-attention remains the explicit baseline;
- only `up_blocks.1.attn1` is modified;
- the branch is restricted to the eroded/feathered face core;
- ordinary PhotoMaker epsilon remains exact outside the core;
- branched cross-attention stays disabled;
- pose adaptation stays off;
- CA face mixing stays off;
- no noised reference-U-Net memory is used by the active spatial lane;
- NN5 counterfactual A/B supervision remains unchanged.

This is an **initialization ablation**, not a new correspondence architecture. Do not add landmarks, UV warping, parsers, new losses, PM attenuation, or additional attention sites in this run.

---

## 2. Ground truth motivating the change

The historical N3a/NN1 route did not begin with previously trained custom BA deltas. Its cloned branch projections began from the effective pretrained attention projections, while the forward equation immediately gave the face region to the reference candidate:

```text
Q_face = Q_target
K_face = K_reference
V_face = V_reference
A_face = Attention(Q_face, K_face, V_face)
```

That made the branch strongly active from the first BA step, even though the LoRA deltas themselves began at zero.

The later protected-residual experiments reversed the ownership hierarchy. NN6 initializes `identity_connector_up` to zero, and the inference path explicitly returns ordinary PhotoMaker when the connector remains zero. Current NN7a removes the zero connector, but its clean-patch K/V matrices are Xavier-random and its effective gate starts near `0.02`.

The intended middle ground is therefore:

```text
not N3a:
    100% unaligned reference ownership

not NN6:
    0% branch ownership

NN7a_init:
    meaningful pretrained reference K/V
    + 5% initial direct ownership
    + target fallback
    + core mask
    + cap
    + PhotoMaker epsilon anchor
```

---

## 3. Exact NN7a_init definition

Relative to current NN7a, make only these initialization changes:

| Component | Current NN7a | NN7a_init |
|---|---|---|
| Clean patch representation | raw CLIP patches, 1024-D | frozen PMv2 Perceiver-context patches, 2048-D |
| Spatial K/V base | Xavier-random full `Linear(1024 → hidden)` | effective sibling `attn2.to_k/to_v` copied as frozen base |
| Trainable K/V delta | full matrix | rank-32 LoRA delta with `lora_B=0` |
| Effective initial spatial gate | approximately `0.02` | exactly approximately `0.05` |
| Direct takeover | enabled | unchanged |
| Spatial and total cap | `0.45` | unchanged |
| Sites | `up_blocks.1.attn1` | unchanged |
| Losses | NN5a causal objective | unchanged |

Do **not** change the cap in this experiment. Keeping the existing `0.45` cap makes the comparison attributable to warm initialization and initial authority.

### Step-zero equation

Let:

```text
P_clip     = clean face-crop CLIP patch grid, 1024-D
P_context  = PMv2 first-stage patch projection and normalization, 2048-D

K_ref      = Wk_attn2,PM(P_context)
V_ref      = Wv_attn2,PM(P_context)

A_target   = Attention(Q_target, K_target, V_target)
A_ref      = LocalAttention5x5(Q_target, K_ref, V_ref)

Delta      = RMSCap(A_ref - A_target, ratio=0.45)
alpha_init = 0.05

A_out      = A_target + M_core * alpha_init * Delta
```

The trainable LoRA delta on `Wk_attn2,PM` and `Wv_attn2,PM` begins at exactly zero, but the **base reference candidate is already meaningful and active**.

---

## 4. New configuration fields

Add these fields with backward-compatible defaults:

```yaml
model:
  # Existing NN7a behavior must remain the default.
  ba_spatial_patch_projection: raw_clip
  # raw_clip | pmv2_perceiver_context

  ba_spatial_kv_init: xavier
  # xavier | sibling_attn2

  ba_spatial_kv_kind: full
  # full | lora
```

Validation rules:

```text
raw_clip                     → patch dimension must be 1024
pmv2_perceiver_context       → patch dimension must be 2048

sibling_attn2                → patch dimension must equal sibling attn2 K/V input dim
sibling_attn2 + lora         → recommended NN7a_init path
direct_candidate_takeover    → still requires clean_clip_patches memory
```

Do not infer the mode from tensor shape alone. Persist the explicit fields in the checkpoint architecture manifest.

---

## 5. New Hydra config

Create:

```text
diffusion_template/src/configs/one_id_ba_NN7a_init.yaml
```

Content:

```yaml
defaults:
  - one_id_ba_NN7a_clean_patch_takeover_up1
  - _self_

# NN7a_init changes only the spatial-memory/KV initialization and initial
# ownership. All topology, causal losses, masks, caps, sites, and anchors remain
# identical to NN7a.
model:
  ba_spatial_patch_projection: pmv2_perceiver_context
  ba_spatial_patch_dim: 2048

  ba_spatial_kv_init: sibling_attn2
  ba_spatial_kv_kind: lora

  # gate = gate_max * sigmoid(logit)
  # 0.80 * sigmoid(-2.70805020110221) = 0.05
  ba_spatial_gate_max: 0.80
  ba_gate_init_logit: -2.70805020110221

  # Keep NN7a's authority envelope unchanged for a clean init ablation.
  ba_spatial_delta_rms_cap: 0.45
  ba_total_delta_rms_cap: 0.45

  # Explicitly restate the protected topology.
  ba_site_policy: up_blocks1_attn1
  ba_spatial_site_policy: up_blocks1_attn1
  ba_identity_token_lane: false
  ba_spatial_lane_enabled: true
  ba_spatial_mix_mode: direct_candidate_takeover
  ba_target_core_erode_frac: 0.15

  train_branched_ca_lora: false
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
  ba_output_anchor_mode: base_outside_core
  ba_pm_id_attenuation_probability: 0.0
  ba_pm_id_attenuation_scale: 1.0
```

Hydra composition must show an effective gate of approximately `0.05`, not `0.02` and not `0.80`.

---

## 6. Code changes

## 6.1 `model_v2_NS.py`: expose PMv2-context patch tokens

Current `extract_spatial_patch_tokens()` returns raw 1024-D CLIP patches.

Extend it without changing the current default:

```diff
diff --git a/diffusion_template/src/model/photomaker_branched/model_v2_NS.py b/diffusion_template/src/model/photomaker_branched/model_v2_NS.py
--- a/diffusion_template/src/model/photomaker_branched/model_v2_NS.py
+++ b/diffusion_template/src/model/photomaker_branched/model_v2_NS.py
@@
-    def extract_spatial_patch_tokens(self, id_pixel_values):
+    def extract_spatial_patch_tokens(
+        self,
+        id_pixel_values,
+        projection: str = "raw_clip",
+    ):
@@
         hidden = self.vision_model(pixels)[0]
         patches = hidden[:, 1:]
+        projection = str(projection or "raw_clip").lower()
+        if projection == "pmv2_perceiver_context":
+            resampler = self.qformer_perceiver.perceiver_resampler
+            # Reuse the pretrained path that maps these exact CLIP patches into
+            # the 2048-D PMv2 Perceiver context space.
+            patches = resampler.proj_in(patches)
+            # The first Perceiver attention layer consumes the projected patch
+            # bank after this trained normalization.
+            patches = resampler.layers[0][0].norm1(patches)
+        elif projection != "raw_clip":
+            raise ValueError(
+                f"Unknown spatial patch projection: {projection}"
+            )
@@
         return patches.view(
             b,
             n,
             patches.shape[1],
             patches.shape[2],
         ).mean(dim=1)
```

Requirements:

- keep this method under `torch.no_grad()`;
- do not make the ID encoder trainable;
- verify raw mode remains `[B, P, 1024]`;
- verify PMv2-context mode is `[B, P, 2048]`;
- average multiple references only after applying the selected projection;
- fail on non-square patch count or non-finite values.

The `proj_in` and first `norm1` parameters come from the loaded PhotoMaker V2 ID encoder and remain frozen.

---

## 6.2 Training and inference reference preparation

Update both matched and counterfactual extraction paths.

### `lora2_helpers.py`

In:

```text
prepare_branched_training_inputs()
prepare_spatial_reference_batch()
```

replace:

```python
model.id_encoder.extract_spatial_patch_tokens(crop_pixels)
```

with:

```python
model.id_encoder.extract_spatial_patch_tokens(
    crop_pixels,
    projection=model.ba_spatial_patch_projection,
)
```

Add a strict dimension check:

```python
if patch_tokens.shape[-1] != model.ba_spatial_patch_dim:
    raise RuntimeError(
        "Spatial patch dimension mismatch: "
        f"tokens={patch_tokens.shape[-1]}, "
        f"config={model.ba_spatial_patch_dim}"
    )
```

This must be applied identically to:

- matched reference A;
- counterfactual reference B.

### `br_pipeline_helpers.py`

In `prepare_spatial_identity_tokens()`, call:

```python
patch_tokens = pipeline.id_encoder.extract_spatial_patch_tokens(
    crop_pixels,
    projection=pipeline.ba_spatial_patch_projection,
)
```

Perform the same dimension and finite checks.

The diagnostic R1/R2 token hashes must be based on the final 2048-D patch tensor actually supplied to the processor.

---

## 6.3 `packed_residual_attn_processor.py`: warm-start from sibling attn2

Add constructor fields:

```python
spatial_patch_projection: str = "raw_clip"
spatial_kv_init: str = "xavier"
spatial_kv_kind: str = "full"
```

Validate:

```python
spatial_patch_projection in {
    "raw_clip",
    "pmv2_perceiver_context",
}

spatial_kv_init in {
    "xavier",
    "sibling_attn2",
}

spatial_kv_kind in {
    "full",
    "lora",
}
```

Change the initialization signature:

```python
def init_from_attention(
    self,
    attn,
    *,
    sibling_attn2=None,
) -> None:
```

Implement the clean-patch K/V initialization as:

```python
if self.spatial_memory_mode == "clean_clip_patches":
    if self.spatial_kv_init == "xavier":
        # Exact old NN7a path.
        self.ref_to_k = nn.Linear(
            self.spatial_patch_dim,
            projection_dim,
            bias=False,
            device=base_q.weight.device,
            dtype=base_q.weight.dtype,
        )
        self.ref_to_v = nn.Linear(
            self.spatial_patch_dim,
            projection_dim,
            bias=False,
            device=base_q.weight.device,
            dtype=base_q.weight.dtype,
        )
        nn.init.xavier_uniform_(self.ref_to_k.weight)
        nn.init.xavier_uniform_(self.ref_to_v.weight)

    elif self.spatial_kv_init == "sibling_attn2":
        if sibling_attn2 is None:
            raise RuntimeError(
                f"{self.processor_name}: sibling attn2 is required "
                "for spatial_kv_init=sibling_attn2"
            )

        k_base = (
            sibling_attn2.to_k.get_base_layer()
            if hasattr(sibling_attn2.to_k, "get_base_layer")
            else sibling_attn2.to_k
        )
        v_base = (
            sibling_attn2.to_v.get_base_layer()
            if hasattr(sibling_attn2.to_v, "get_base_layer")
            else sibling_attn2.to_v
        )

        if int(k_base.in_features) != self.spatial_patch_dim:
            raise RuntimeError(
                f"{self.processor_name}: sibling attn2 K input "
                f"{k_base.in_features} != spatial patch dim "
                f"{self.spatial_patch_dim}"
            )
        if int(v_base.in_features) != self.spatial_patch_dim:
            raise RuntimeError(
                f"{self.processor_name}: sibling attn2 V input "
                f"{v_base.in_features} != spatial patch dim "
                f"{self.spatial_patch_dim}"
            )
        if (
            int(k_base.out_features) != projection_dim
            or int(v_base.out_features) != projection_dim
        ):
            raise RuntimeError(
                f"{self.processor_name}: sibling attn2 output does not "
                "match attn1 hidden size"
            )

        self.ref_to_k = _clone_effective_linear(
            sibling_attn2.to_k,
            kind=self.spatial_kv_kind,
            rank=self.ref_kv_rank,
            adapter_name="default",
        )
        self.ref_to_v = _clone_effective_linear(
            sibling_attn2.to_v,
            kind=self.spatial_kv_kind,
            rank=self.ref_kv_rank,
            adapter_name="default",
        )
```

For `NN7a_init`, `spatial_kv_kind` is `lora`, so:

```text
base_weight = effective sibling attn2 PhotoMaker K/V
lora_A      = Kaiming initialized
lora_B      = exactly zero
```

Do not clone sibling attn2 Q. Target queries must remain target self-attention queries.

Do not add a connector. The current direct full-dimensional difference remains:

```python
spatial_raw_delta = spatial_reference_candidate - target_base
```

---

## 6.4 `branched_runtime.py`: resolve and pass sibling attn2

When creating a packed processor for an `attn1` site:

```python
attn1_module = _resolve_attn_module(pipeline.unet, name)
sibling_attn2_name = name.replace(
    ".attn1.processor",
    ".attn2.processor",
)
sibling_attn2_module = _resolve_attn_module(
    pipeline.unet,
    sibling_attn2_name,
)
proc.init_from_attention(
    attn1_module,
    sibling_attn2=sibling_attn2_module,
)
```

Use the sibling only when:

```text
spatial_memory_mode == clean_clip_patches
and spatial_kv_init == sibling_attn2
```

For all old modes, preserve the previous call and behavior.

Pass the three new config fields into the processor constructor:

```python
spatial_patch_projection=str(
    getattr(pipeline, "ba_spatial_patch_projection", "raw_clip")
),
spatial_kv_init=str(
    getattr(pipeline, "ba_spatial_kv_init", "xavier")
),
spatial_kv_kind=str(
    getattr(pipeline, "ba_spatial_kv_kind", "full")
),
```

Strictly assert that the `.attn2` sibling exists at every selected `up_blocks.1.attn1` site.

---

## 6.5 `lora2.py`: config, validation, checkpoint manifest and pipeline propagation

Add constructor arguments and stored attributes:

```python
ba_spatial_patch_projection: str = "raw_clip"
ba_spatial_kv_init: str = "xavier"
ba_spatial_kv_kind: str = "full"
```

Validate the enums and dimensional combinations.

For `NN7a_init`, enforce:

```text
ba_spatial_memory_mode          = clean_clip_patches
ba_spatial_patch_projection     = pmv2_perceiver_context
ba_spatial_patch_dim            = 2048
ba_spatial_kv_init              = sibling_attn2
ba_spatial_kv_kind              = lora
ba_spatial_mix_mode             = direct_candidate_takeover
```

Add all fields to `_ba_architecture_state()` so strict checkpoint restore rejects NN7a/NN7a_init interchange.

Propagate them into the validation pipeline in `build_pipeline_from_pretrained()` and assert equality with the training model.

Do not load an NN7a checkpoint into NN7a_init or vice versa under strict restore.

---

## 6.6 Trainability and optimizer manifests

Current direct-spatial NN7a expects:

```text
ref_to_k.weight
ref_to_v.weight
gate_logit
```

NN7a_init must instead train exactly:

```text
ref_to_k.lora_A
ref_to_k.lora_B
ref_to_v.lora_A
ref_to_v.lora_B
gate_logit
```

Update:

```text
lora2_helpers._assert_branched_installation()
lora2_helpers.configure_branched_trainables()
lora2.get_trainable_params()
```

Branch on:

```python
direct_spatial = (
    ba_spatial_mix_mode == "direct_candidate_takeover"
)
warm_lora_spatial = (
    direct_spatial
    and ba_spatial_kv_init == "sibling_attn2"
    and ba_spatial_kv_kind == "lora"
)
```

Expected local keys for NN7a_init:

```python
{
    "ref_to_k.lora_A",
    "ref_to_k.lora_B",
    "ref_to_v.lora_A",
    "ref_to_v.lora_B",
    "gate_logit",
}
```

The sibling-attn2 base weights are buffers inside `BranchLoRALinear`; they must not be trainable and do not belong in optimizer groups.

Optimizer groups may remain:

```text
ba_ppr_ref_k
ba_ppr_ref_v
ba_ppr_gate
```

but must contain only the five expected trainable tensors per processor.

---

## 6.7 Checkpoint behavior across SDXL training and RealVis validation

This is important.

The warm base K/V buffers should **not** be serialized as learned branch state. On restore:

1. instantiate the selected processor on the current backbone;
2. recreate its frozen base K/V from that backbone's effective sibling attn2;
3. load only the learned LoRA A/B deltas and gate from the checkpoint.

This preserves the project's established cross-backbone behavior:

```text
train on SDXL:
    PM-aware SDXL attn2 base + learned branch LoRA

validate on RealVis:
    PM-aware RealVis attn2 base + the same learned branch LoRA
```

The strict architecture manifest must record the initialization mode, even though it does not store the base buffers.

---

## 7. Initialization correctness guards

At processor installation, add a one-time guard for every selected site.

### LoRA zero-delta check

```python
assert torch.count_nonzero(proc.ref_to_k.lora_B) == 0
assert torch.count_nonzero(proc.ref_to_v.lora_B) == 0
```

### Effective base parity

Use a deterministic finite `[1, P, 2048]` test tensor in the processor dtype.

Before any training:

```python
actual_k = proc.ref_to_k(test_tokens)
actual_v = proc.ref_to_v(test_tokens)

expected_k = effective_sibling_attn2_to_k(test_tokens)
expected_v = effective_sibling_attn2_to_v(test_tokens)
```

Require:

```text
FP32: atol/rtol 1e-5
BF16: atol/rtol 2e-2
```

Log:

```text
[NN7a_init warm start]
site=...
patch_projection=pmv2_perceiver_context
patch_dim=2048
kv_init=sibling_attn2
kv_kind=lora
k_base_parity=true
v_base_parity=true
effective_gate_init=0.050000
```

### Branch must not be exactly off

For a real valid batch at step zero:

```text
RMS(A_ref - A_target) > 0
applied_delta_rms_ratio > 0
```

Outside the target core, the final epsilon must remain exactly ordinary PhotoMaker.

---

## 8. Tests

Add tests before launching.

## 8.1 PMv2 patch projection

```text
raw_clip output:                [B, P, 1024]
pmv2_perceiver_context output: [B, P, 2048]
both finite
same patch count
ID encoder parameters remain frozen
```

## 8.2 Sibling-attn2 K/V parity

Build a small attention pair where:

```text
attn1 query/output width = H
attn2 cross_attention_dim = 2048
attn2 output width = H
```

Initialize `NN7a_init` and assert:

```text
ref_to_k(x) == effective attn2.to_k(x)
ref_to_v(x) == effective attn2.to_v(x)
LoRA B tensors are zero
```

## 8.3 Nonzero reference sensitivity at initialization

With identical target hidden states and two different patch banks:

```text
output(ref_A) != output(ref_B)
```

inside the core at initialization.

The same test must confirm:

```text
output outside core is identical
```

## 8.4 Gate value

Assert:

```python
0.80 * sigmoid(-2.70805020110221) ≈ 0.05
```

to within `1e-6`.

## 8.5 Gradients

At step zero, backpropagate a target-face loss and verify nonzero gradients on:

```text
ref_to_k.lora_B
ref_to_v.lora_B
gate_logit
```

It is acceptable for `lora_A` gradients to be zero on the very first backward while `lora_B=0`; they must become nonzero after a simulated optimizer update that opens `lora_B`.

## 8.6 Backward compatibility

The existing NN7a config must still instantiate:

```text
raw_clip
xavier
full Linear K/V
effective gate ≈ 0.02
```

and pass its old trainability manifest unchanged.

## 8.7 Strict checkpoint separation

- NN7a_init checkpoint restores into NN7a_init.
- NN7a checkpoint is rejected by NN7a_init strict restore.
- NN7a_init checkpoint is rejected by NN7a strict restore.

---

## 9. Launch scripts

Create:

```text
diffusion_template/jul_serv_runs/start_ba_NN7a_init_1gpu.sh
diffusion_template/jul_serv_runs/start_ba_NN7a_init_train_then_diagnose_1gpu.sh
diffusion_template/jul_serv_runs/start_ba_NN7a_init_checkpoint_reference_vs_noise_24_1gpu.sh
```

### Training launcher

Clone the current NN7a launcher and change only:

```bash
export NN1_CONFIG_NAME="one_id_ba_NN7a_init"
export NN1_RUN_NAME_DEFAULT="ba_NN7a_init_1gpu"
export NN1_DESCRIPTION="NN7a_init: PMv2-context clean patches with sibling-attn2 warm-started local takeover"
```

Keep:

```text
one GPU
physical batch 1
effective batch 2
2 × 2,000 optimizer steps
96-image RealVis in-training validation
same dataset and counterfactual sampling
```

### Combined launcher

Use:

```bash
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-ba_NN7a_init_1gpu}"
```

and call the new training and diagnostic launchers.

### Diagnostic launcher

Clone the current NN7a 24-case wrapper, but make it instantiate `one_id_ba_NN7a_init` through `start_ba_NN7a_init_1gpu.sh`.

Keep:

```text
RealVisXL V4.0
scale 1
same deterministic 24/96 subset
same subset seed 20260722
same two reference-noise seeds
strict checkpoint model config
```

Launch:

```bash
cd /home/niko/rsrch/diffusion_template

CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_init_train_then_diagnose_1gpu.sh
```

---

## 10. Recommended preflight before the 4k run

The main point of this experiment is initialization, so inspect it before spending 4k steps.

### Required one-batch smoke test

Confirm the logs show:

```text
patch_dim=2048
kv_init=sibling_attn2
kv_kind=lora
effective_gate_init=0.05
K/V base parity passed
nonzero applied residual
no trainable connector
no trainable branched CA
```

### Recommended checkpoint-zero image comparison

On the deterministic validation subset, compare:

```text
PM0
current NN7a init
NN7a_init
```

The desired NN7a_init step-zero behavior is:

- visibly more reference-sensitive than current NN7a;
- still much closer to PhotoMaker geometry than N3a;
- no duplicated face, boundary seam, or pose rotation;
- 100% face detection;
- zero outside-core change after the anchor.

If the `0.05` initialization is visually indistinguishable from current NN7a, run a diagnostic-only override:

```bash
model.ba_gate_init_logit=-1.9459101490553132
```

which gives:

```text
0.80 × sigmoid(-1.94591) = 0.10
```

Do **not** make `0.10` the training default until its checkpoint-zero geometry is visually safe.

---

## 11. Metrics and decision rules

Compare NN7a_init directly against current NN7a at the same seeds and checkpoints.

### Initialization metrics

At step zero:

```text
candidate reference-content difference
applied_delta_rms_ratio
face-core MAE versus PM0
face-core LPIPS versus PM0
landmark displacement versus PM0
face detection
cap fraction
outside-core epsilon difference
```

Expected:

```text
reference-content effect > current NN7a
applied residual > current NN7a
outside-core effect = 0
geometry remains near PM0
```

### Training metrics

Watch:

```text
ba_cf/directional_gain
ba_cf/sim_to_wrong
ba_cf/sim_to_matched
ba_cf/applied_fraction
ba_cf/reference_noise_equal
ba_norm/sa_ref
grad_norm/ba_ppr_ref_k
grad_norm/ba_ppr_ref_v
grad_norm/ba_ppr_gate
cap_fraction
applied_delta_rms_ratio
```

### Causal approval at 2k/4k

Proceed only if:

- mean target-averaged directional gain is positive;
- B similarity actually increases;
- N1 and N2 agree;
- face detection remains complete;
- target pose and landmarks remain stable;
- no boundary, hair, hand, glasses, or jaw/neck artifacts appear;
- the branch remains visibly active at scale 1.

A larger face change without positive B direction is not success.

---

## 12. Stop conditions

Stop or revise the initialization if any of these occur:

```text
step-zero duplicated landmarks or pasted-face boundaries
large target-pose displacement before training
cap saturation on most samples at initialization
gate rapidly collapses toward zero while K/V gradients remain healthy
reference change affects only expression/lighting and not identity
outside-core epsilon is not exact PhotoMaker
strict checkpoint restore cannot distinguish NN7a from NN7a_init
```

If the warm-start is safe but the gate collapses during the first 500 steps, diagnose that separately before adding a gate-floor or coverage loss. Do not silently add such an objective to this initialization ablation.

---

## 13. Files expected in the implementation

```text
diffusion_template/src/configs/one_id_ba_NN7a_init.yaml

diffusion_template/src/model/photomaker_branched/model_v2_NS.py
diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py
diffusion_template/src/model/photomaker_branched/branched_runtime.py
diffusion_template/src/model/photomaker_branched/lora2.py
diffusion_template/src/model/photomaker_branched/lora2_helpers.py
diffusion_template/src/pipelines/br_pipeline_helpers.py

diffusion_template/tests/test_packed_residual_attn_processor.py
diffusion_template/tests/test_nn5_components.py

diffusion_template/jul_serv_runs/start_ba_NN7a_init_1gpu.sh
diffusion_template/jul_serv_runs/start_ba_NN7a_init_train_then_diagnose_1gpu.sh
diffusion_template/jul_serv_runs/start_ba_NN7a_init_checkpoint_reference_vs_noise_24_1gpu.sh
```

---

## 14. Agent completion checklist

The implementing agent should report all of the following:

- [ ] existing NN7a remains backward-compatible;
- [ ] new config composes successfully;
- [ ] PMv2-context patch output is exactly 2048-D;
- [ ] sibling attn2 K/V dimensions are asserted at every selected site;
- [ ] copied warm bases match effective sibling attn2 outputs at initialization;
- [ ] LoRA B begins at zero;
- [ ] effective spatial gate begins at approximately `0.05`;
- [ ] direct reference candidate is nonzero and reference-sensitive at step zero;
- [ ] outside-core final epsilon remains exact PhotoMaker;
- [ ] optimizer contains only spatial K/V LoRA and gate parameters;
- [ ] strict checkpoint manifest records all new initialization fields;
- [ ] NN7a and NN7a_init checkpoints cannot be interchanged;
- [ ] unit tests pass;
- [ ] shell syntax checks pass;
- [ ] no Git push was performed without explicit authorization.

---

## 15. Concise scientific interpretation

`NN7a_init` is meant to test a precise hypothesis:

> The recent branches may remain too PhotoMaker-like partly because their active reference path begins as zero or random. A PhotoMaker-aware reference K/V warm start with modest nonzero ownership may let the branch learn identity direction earlier without restoring N3a's unsafe absolute reference-face replacement.

A positive result is not merely more visual change. It is:

```text
earlier and stronger reference causality
+ actual movement toward reference B
+ unchanged target pose
+ unchanged boundaries and occluders
+ exact PhotoMaker outside the core
```
