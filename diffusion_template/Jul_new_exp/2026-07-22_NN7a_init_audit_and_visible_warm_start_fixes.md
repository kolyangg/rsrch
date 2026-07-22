# NN7a_init implementation audit and visible warm-start fixes

**Date:** 22 July 2026  
**Branch audited:** latest `main_clean` visible through GitHub  
**Launch audited:**

```bash
bash jul_serv_runs/start_ba_NN7a_init_train_then_diagnose_1gpu.sh
```

**Repository mutation:** none.

---

## 1. Verdict

The current `NN7a_init` implementation is **mechanically correct relative to the previous instruction**, and the branch is not initialized to exact zero. It correctly implements:

```text
face-cropped CLIP patches
→ frozen PMv2 Perceiver projection to 2048-D
→ sibling attn2 K/V copied as the frozen LoRA base
→ zero LoRA-B delta
→ local 5×5 reference attention
→ direct target/reference candidate difference
→ effective scalar gate ≈ 0.05
→ target-core mask
→ PhotoMaker epsilon outside the core
```

The configuration, extraction path, optimizer/trainability manifest, cross-backbone reconstruction of frozen bases, and strict checkpoint architecture fields are present.

However, the current implementation does **not** yet provide the meaningful N3a-like warm start that was intended. Two design details make its step-zero image effect far weaker and less semantically coherent than the config comments suggest:

1. **Only sibling `attn2` K/V are warm-started.**  
   The reference candidate still uses the `attn1` self-attention query and is mapped through the `attn1` output projection. This is a hybrid attention space:

   ```text
   attn1 Q
   + attn2 K/V
   + attn1 output projection
   ```

   Sibling `attn2.to_k/to_v` are pretrained to work with sibling `attn2.to_q` and sibling `attn2.to_out`, not with the `attn1` pair. K/V parity alone therefore does not establish that the complete reference candidate is pretrained or meaningful.

2. **The code caps first and gates second.**  
   With:

   ```text
   spatial cap = 0.45
   effective gate = 0.05
   ```

   the final local attention perturbation is bounded by:

   ```text
   0.45 × 0.05 = 0.0225
   ```

   or only **2.25% of target-attention RMS**, before the ordinary transformer residual and the final epsilon anchor dilute it further.

So the observation that full validation images look exactly like PhotoMaker is not surprising. The branch is numerically nonzero, but the current warm start is still extremely conservative and only partially pretrained.

### Recommendation

Do not interpret the current step-zero visual parity as proof that the warm path is disconnected.

Implement **NN7a_init-v2** with:

```text
complete sibling-attn2 Q/K/V/out warm space
+ post-output target/reference arbitration
+ gate before cap
+ effective initial alpha 0.10
+ final local cap 0.20
```

This is the smallest principled change that should make step-zero reference control visible while preserving target fallback, the inner-core mask, and the PhotoMaker epsilon anchor.

Also fix the checkpoint preflight: the current post-training wrapper requests a nonzero PPR checkpoint, but the preflight still requires `connector_up` tensors, which do not exist in direct-takeover NN7a/NN7a_init.

---

## 2. What is already correct

### Configuration

`one_id_ba_NN7a_init.yaml` correctly composes from NN7a and sets:

```yaml
ba_spatial_patch_projection: pmv2_perceiver_context
ba_spatial_patch_dim: 2048
ba_spatial_kv_init: sibling_attn2
ba_spatial_kv_kind: lora

ba_spatial_gate_max: 0.80
ba_gate_init_logit: -2.70805020110221   # effective 0.05

ba_spatial_mix_mode: direct_candidate_takeover
ba_site_policy: up_blocks1_attn1
ba_spatial_delta_rms_cap: 0.45
ba_total_delta_rms_cap: 0.45
ba_output_anchor_mode: base_outside_core
```

The launcher selects this config and RealVis validation correctly.

### Clean PMv2-context patches

`model_v2_NS.py` correctly maps the 1024-D CLIP patch grid through:

```text
qformer_perceiver.perceiver_resampler.proj_in
→ first Perceiver-attention norm1
```

to produce a frozen, finite 2048-D patch bank.

Both training and inference pass `ba_spatial_patch_projection` into this method and assert the configured patch dimension.

### Warm K/V initialization

For every selected `up_blocks.1.attn1` site, the runtime resolves the sibling `attn2` module. The processor:

- clones the effective PhotoMaker sibling `to_k` and `to_v`;
- uses them as nonpersistent LoRA base buffers;
- initializes LoRA B to zero;
- verifies output parity with the active sibling K/V;
- rebuilds the base from the current backbone on restore.

This is correct for SDXL training followed by RealVis validation.

### Branch activity guard

On the first real forward, the processor fails if either:

```text
reference candidate == target candidate
```

or:

```text
applied spatial residual == 0
```

inside a supported core. Therefore, if step-zero validation completed and the log contains:

```text
[NN7a_init first batch]
```

the processor-local branch was active.

### Safety envelope

The implementation correctly retains:

- target self-attention;
- local 5×5 clean-reference windows;
- `up_blocks.1.attn1` scope;
- eroded/feathered target core;
- frozen branched CA;
- no pose adaptation;
- no CA face mixing;
- exact PhotoMaker epsilon outside the core.

---

## 3. Why the faces still look like PhotoMaker

## 3.1 The stated `0.05` is not a 5% final attention perturbation

Current direct-spatial code effectively does:

```python
raw_delta = reference_candidate - target_candidate

bounded_delta = rms_cap(
    raw_delta,
    max_ratio=0.45,
)

applied_delta = 0.05 * bounded_delta
```

Therefore:

```text
RMS(applied_delta) / RMS(target_candidate) ≤ 0.0225
```

The total cap does not increase it.

The attention module then applies `attn1.to_out`, adds the full transformer residual hidden state, and divides by the module rescale factor. Thus the branch change is smaller relative to the complete transformer output than 2.25%.

The final pipeline then preserves ordinary PhotoMaker epsilon outside the eroded core:

```text
epsilon_out =
    epsilon_PM
  + M_core × (epsilon_branch - epsilon_PM)
```

A small change at one late attention family can easily become visually imperceptible in the final 1024×1024 image.

## 3.2 The warm start is only partial

Current reference attention is:

```text
Q = attn1.to_q(target hidden)
K = copied attn2.to_k(PMv2 patch context)
V = copied attn2.to_v(PMv2 patch context)
output mapping = attn1.to_out
```

This is not a complete pretrained attention operator.

A meaningful sibling-attn2 warm candidate should be:

```text
Q_ref = sibling_attn2.to_q(target hidden)
K_ref = sibling_attn2.to_k(PMv2 patch context)
V_ref = sibling_attn2.to_v(PMv2 patch context)

A_ref_pre  = Attention(Q_ref, K_ref, V_ref)
A_ref_post = sibling_attn2.to_out(A_ref_pre)
```

The ordinary target self-attention candidate should separately be:

```text
A_target_pre  = Attention(attn1 Q, attn1 K, attn1 V)
A_target_post = attn1.to_out(A_target_pre)
```

Only then are two candidates combined in the common transformer hidden space.

## 3.3 Matched validation is a weak visual test

Normal validation supplies identity A through both:

```text
PhotoMaker target conditioning A
reference image A
```

A successful branch may leave the matched image close to PhotoMaker while becoming causally responsive to an A→B reference swap.

The decisive comparison is:

```text
R1 versus R2 at the same checkpoint
```

not only:

```text
step 0 versus step 2k with the same matched reference
```

Nevertheless, with the intended warm initialization, the R1/R2 difference should already be measurable at step zero. The current branch is too weak and hybridized for that expectation to be reliable.

---

## 4. Required change 1 — use a complete sibling-attn2 attention space

### New fields

Add backward-compatible fields:

```yaml
model:
  ba_spatial_attention_space: attn1_hybrid
  # attn1_hybrid | sibling_attn2_full

  ba_spatial_gate_position: post_cap
  # post_cap | pre_cap
```

Old NN7a and current NN7a_init remain reproducible under the defaults.

NN7a_init-v2 uses:

```yaml
ba_spatial_attention_space: sibling_attn2_full
ba_spatial_gate_position: pre_cap
```

### Processor members

In `packed_residual_attn_processor.py`, add:

```python
self.spatial_to_q = None
self.spatial_to_out = None
self.spatial_q_norm = None
self.spatial_k_norm = None
```

When:

```text
spatial_attention_space == sibling_attn2_full
```

initialize:

```python
self.spatial_to_q = _clone_effective_linear(
    sibling_attn2.to_q,
    kind="full",
    rank=self.ref_kv_rank,
    adapter_name="default",
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
self.spatial_to_out = _clone_effective_linear(
    sibling_attn2.to_out[0],
    kind="full",
    rank=self.ref_kv_rank,
    adapter_name="default",
)

for module in (self.spatial_to_q, self.spatial_to_out):
    module.requires_grad_(False)
```

If sibling q/k norms exist, deep-copy and freeze them:

```python
self.spatial_q_norm = copy.deepcopy(
    getattr(sibling_attn2, "norm_q", None)
)
self.spatial_k_norm = copy.deepcopy(
    getattr(sibling_attn2, "norm_k", None)
)
```

Do not make Q or output trainable in this experiment. Train only:

```text
ref_to_k.lora_A/B
ref_to_v.lora_A/B
gate_logit
```

### Candidate calculation

For `sibling_attn2_full`, calculate the spatial candidate from target hidden states, not from the attn1 query tensor:

```python
reference_query = self._to_heads(
    self.spatial_to_q(target_hidden),
    attn.heads,
)
if self.spatial_q_norm is not None:
    reference_query = self.spatial_q_norm(reference_query)

reference_key = self._to_heads(
    self.ref_to_k(patches),
    attn.heads,
)
reference_value = self._to_heads(
    self.ref_to_v(patches),
    attn.heads,
)
if self.spatial_k_norm is not None:
    reference_key = self.spatial_k_norm(reference_key)
```

Retain the local 5×5 correspondence indexing, but use `reference_query` for the local dot products.

After local attention:

```python
reference_pre = self._from_heads(local_output)
reference_post = self.spatial_to_out(reference_pre)
```

### Mix after each candidate's own output projection

Refactor the direct warm path so the target and reference candidates are compared after their own output projections.

Conceptually:

```python
target_pre = ordinary_attn1_pre_out(...)
target_post = attn.to_out[1](
    attn.to_out[0](target_pre)
)

reference_post = sibling_attn2_reference_candidate_post_out(...)

raw_delta_post = reference_post - target_post
```

Then apply the target core and bounded takeover.

Do not run `attn1.to_out` over `reference_post` a second time.

For the reference continuation half, continue using the ordinary attn1 candidate and its own attn1 output projection.

### Why this is required

The current unit test proves:

```text
ref_to_k(tokens) == sibling_attn2.to_k(tokens)
ref_to_v(tokens) == sibling_attn2.to_v(tokens)
```

but it does not prove that:

```text
attn1_Q × sibling_attn2_K/V × attn1_out
```

is a pretrained or semantically meaningful operator.

Add a full-path parity test instead.

---

## 5. Required change 2 — gate before cap

The intended takeover equation is:

```text
A_out =
    A_target
  + cap(
      alpha × (A_reference - A_target)
    )
```

not:

```text
A_out =
    A_target
  + alpha × cap(A_reference - A_target)
```

### Patch logic

For direct takeover with `ba_spatial_gate_position=pre_cap`:

```diff
- spatial_bounded_delta, spatial_cap_scale, ... = self._masked_rms_cap(
-     spatial_raw_delta,
-     base=target_base,
-     mask=target_core,
-     max_ratio=self.spatial_delta_rms_cap,
- )
  spatial_gate = self.spatial_gate_max * torch.sigmoid(
      self.gate_logit
  )
+ spatial_scaled_delta = (
+     spatial_gate
+     * float(self.runtime_scale)
+     * spatial_raw_delta
+ )
+ spatial_bounded_delta, spatial_cap_scale, ... = self._masked_rms_cap(
+     spatial_scaled_delta,
+     base=target_candidate_post,
+     mask=target_core,
+     max_ratio=self.spatial_delta_rms_cap,
+ )
  spatial_applied_delta = (
      target_core
      * spatial_has_roi[:, None, None].to(target_base.dtype)
-     * spatial_gate
-     * float(self.runtime_scale)
      * spatial_bounded_delta
  )
```

Keep the old order for other configurations.

### NN7a_init-v2 authority

Use:

```yaml
ba_spatial_gate_max: 0.80

# 0.80 × sigmoid(-1.9459101490553132) = 0.10
ba_gate_init_logit: -1.9459101490553132

ba_spatial_delta_rms_cap: 0.20
ba_total_delta_rms_cap: 0.20
```

This means:

```text
initial candidate interpolation coefficient = 0.10
maximum final local attention residual = 20% of target candidate RMS
```

rather than the current effective maximum of 2.25%.

This is still far below N3a's absolute reference replacement.

### Safer preflight sweep

Before training, evaluate:

```text
alpha 0.05: logit -2.70805020110221
alpha 0.10: logit -1.9459101490553132
alpha 0.20: logit -1.0986122886681098
```

with the same final cap `0.20`.

Choose the strongest setting that preserves:

- head pose;
- eye/mouth count;
- jaw/neck attachment;
- face detection;
- clean boundary;
- target occluders.

Use `0.10` as the recommended default, but do not launch 4k before inspecting the no-training sweep.

---

## 6. Required change 3 — fix direct-takeover checkpoint preflight

The combined launcher runs:

```bash
ppr_checkpoint_require_nonzero=true
```

The current `BaseTrainer._check_ppr_checkpoint_preflight()` still requires one or more nonzero `connector_up` tensors.

Direct NN7a/NN7a_init has no connector, so the 4k diagnostic can fail even when K/V LoRA and the gate were trained correctly.

### Track direct-spatial learned state

In `lora2.load_state_dict_()`, add:

```python
direct_spatial_tensors = 0
direct_spatial_nonzero = 0
direct_spatial_l2_sq = 0.0
```

For every processor state:

```python
for key in (
    "ref_to_k.lora_B",
    "ref_to_v.lora_B",
):
    value = sd.get(key)
    if value is None:
        continue
    value = value.detach().float()
    direct_spatial_tensors += 1
    direct_spatial_nonzero += int(
        torch.count_nonzero(value).item()
    )
    direct_spatial_l2_sq += float(
        value.square().sum().item()
    )
```

Store:

```python
self._last_ppr_checkpoint_diagnostics.update(
    {
        "direct_spatial_tensors": direct_spatial_tensors,
        "direct_spatial_nonzero": direct_spatial_nonzero,
        "direct_spatial_l2": direct_spatial_l2_sq ** 0.5,
    }
)
```

### Generalize preflight

In `BaseTrainer._check_ppr_checkpoint_preflight()`:

```python
connector_count = int(
    diagnostics.get("connector_up_tensors", 0)
)
direct_count = int(
    diagnostics.get("direct_spatial_tensors", 0)
)

if connector_count > 0:
    if int(diagnostics["connector_up_nonzero"]) == 0:
        raise RuntimeError(
            "PPR checkpoint has only zero connector_up tensors"
        )
elif direct_count > 0:
    if int(diagnostics["direct_spatial_nonzero"]) == 0:
        raise RuntimeError(
            "Direct-spatial checkpoint has only zero K/V LoRA-B tensors"
        )
else:
    raise RuntimeError(
        "PPR checkpoint contains neither connector nor direct-spatial learned state"
    )
```

Log the direct-spatial norm and gate range.

---

## 7. Recommended change 4 — make step-zero activity measurable

Current first-batch logging reports absolute candidate and applied RMS. Add relative ratios:

```python
base_rms = torch.sqrt(
    (
        core
        * target_candidate_post.float().square()
    ).sum(dim=(1, 2))
    / count
    + 1e-12
)

candidate_ratio = candidate_rms / (base_rms + 1e-12)
applied_ratio = applied_rms / (base_rms + 1e-12)
```

Log:

```text
candidate_ratio_min/median
applied_ratio_min/median
cap_fraction
effective_gate
```

This distinguishes:

```text
branch disconnected
```

from:

```text
branch active but too small to see
```

For NN7a_init-v2 at step zero, require:

```text
median applied ratio ≥ 0.03
outside-core exact zero
```

as a practical preflight target. This is not an identity-success criterion; it only proves meaningful authority.

---

## 8. New config

Create:

```text
src/configs/one_id_ba_NN7a_init_v2.yaml
```

```yaml
defaults:
  - one_id_ba_NN7a_init
  - _self_

model:
  # Complete pretrained sibling-attention space.
  ba_spatial_attention_space: sibling_attn2_full

  # Make alpha the actual candidate interpolation coefficient.
  ba_spatial_gate_position: pre_cap

  # Initial alpha = 0.10.
  ba_spatial_gate_max: 0.80
  ba_gate_init_logit: -1.9459101490553132

  # Bound final local attention authority.
  ba_spatial_delta_rms_cap: 0.20
  ba_total_delta_rms_cap: 0.20

  # Unchanged experiment topology.
  ba_spatial_patch_projection: pmv2_perceiver_context
  ba_spatial_patch_dim: 2048
  ba_spatial_kv_init: sibling_attn2
  ba_spatial_kv_kind: lora
  ba_spatial_memory_mode: clean_clip_patches
  ba_spatial_local_window: 5
  ba_spatial_mix_mode: direct_candidate_takeover

  ba_site_policy: up_blocks1_attn1
  ba_spatial_site_policy: up_blocks1_attn1
  ba_identity_token_lane: false
  ba_spatial_lane_enabled: true
  ba_target_core_erode_frac: 0.15

  train_branched_ca_lora: false
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
  ba_output_anchor_mode: base_outside_core
  ba_pm_id_attenuation_probability: 0.0
  ba_pm_id_attenuation_scale: 1.0
```

Persist both new fields in:

```text
_ba_architecture_state()
validation-pipeline propagation
strict restore comparisons
processor diagnostics
```

Do not allow v1 and v2 checkpoints to restore into each other.

---

## 9. Tests to add

## 9.1 Complete sibling path

Create separate `attn1` and `attn2` modules with intentionally different Q and output matrices.

Assert the full reference candidate equals:

```python
attn2.to_out[0](
    SDPA(
        attn2.to_q(target_hidden),
        attn2.to_k(reference_tokens),
        attn2.to_v(reference_tokens),
    )
)
```

inside the selected local window.

This test must fail under the current hybrid implementation.

## 9.2 Correct output-space arbitration

Assert that direct takeover mixes:

```text
attn1 post-out target candidate
with
attn2 post-out reference candidate
```

and does not apply `attn1.to_out` twice.

## 9.3 Gate-before-cap semantics

Use a synthetic raw delta with known RMS.

Verify:

```text
small raw delta:
    output delta ≈ alpha × raw delta

large raw delta:
    output delta ratio == configured final cap
```

## 9.4 Step-zero reference sensitivity

At initialization:

```text
output(reference A) != output(reference B)
```

inside the core, with a minimum relative RMS threshold.

Outside the core:

```text
output(reference A) == output(reference B)
```

exactly.

## 9.5 Direct checkpoint preflight

Create a state with:

```text
no connector_up
nonzero ref_to_k/ref_to_v LoRA-B
nonzero gate
```

and verify `ppr_checkpoint_require_nonzero=true` passes.

A state with zero LoRA-B must fail.

## 9.6 Backward compatibility

Current NN7a and current NN7a_init must reproduce their old:

```text
attn1_hybrid
post_cap
```

behavior when the new fields are absent.

---

## 10. What to check in the existing run now

Search the log:

```bash
grep -E \
'\[NN7a_init warm start\]|\[NN7a_init first batch\]|\[BA output anchor\]' \
logs_new_runs/ba_NN7a_init_1gpu_*.log
```

Expected:

```text
kv_init=sibling_attn2
kv_kind=lora
k_base_parity=true
v_base_parity=true
effective_gate_init=0.050000

candidate_rms_min > 0
applied_rms_min > 0
outside_core_exact_zero=true

state=base-outside-core
```

If these appear, the current branch is active and the PhotoMaker-like appearance is explained by weak/hybrid initialization.

If the output anchor says:

```text
state=exact-zero-bypass
```

that is a bug for direct takeover.

### Check whether images are literally identical

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np
import sys

a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.int16)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.int16)
d = np.abs(a - b)

print("exact:", bool(np.array_equal(a, b)))
print("max_abs:", int(d.max()))
print("mean_abs:", float(d.mean()))
print("changed_pixels:", int(np.any(d != 0, axis=2).sum()))
PY \
  /path/to/photomaker.png \
  /path/to/nn7a_init.png
```

Across all 96 images, byte-identical PM/BA results would be inconsistent with the current first-batch nonzero guard and should trigger a validation-routing investigation.

Visually identical but numerically different results are expected under the current ≤2.25% local attention bound.

---

## 11. Run decision

### Current run

The current run is not invalid. It is a useful control for:

```text
partial sibling-K/V warm start
+ post-cap 0.05 gate
```

If it is already near 4k, allow it to finish and retain its metrics.

If it has only completed the initial validation and the scientific goal is visibly non-PhotoMaker reference ownership, stop it and implement v2 before spending the full budget.

### NN7a_init-v2

Before the 4k run:

1. run a no-training 24-image alpha sweep;
2. inspect `PM0`, `R1`, and `R2` face crops;
3. select the strongest safe alpha;
4. require nontrivial R1/R2 differences;
5. then run 500-step, 2k, and 4k checkpoints.

The success criterion remains:

```text
R2 becomes more similar to identity B
while target pose, expression, boundaries, occluders and body remain stable
```

A larger generic face difference is not sufficient.

---

## 12. Concise implementation handoff

An implementation agent should:

- [ ] retain the current NN7a_init as a reproducible control;
- [ ] add `ba_spatial_attention_space`;
- [ ] add `ba_spatial_gate_position`;
- [ ] clone frozen sibling `attn2.to_q` and `attn2.to_out[0]`;
- [ ] retain LoRA sibling `attn2.to_k/to_v`;
- [ ] compute reference attention entirely in sibling-attn2 space;
- [ ] combine target/reference candidates after their own output projections;
- [ ] move the direct gate before the cap;
- [ ] initialize effective alpha to `0.10`;
- [ ] cap the final local residual at `0.20`;
- [ ] fix direct-spatial checkpoint preflight;
- [ ] add relative step-zero activity logging;
- [ ] add the listed tests;
- [ ] create a new strict config/checkpoint namespace;
- [ ] do not enable branched CA, pose adaptation, CA face mixing, PM attenuation, new sites, or geometry dependencies in the same run;
- [ ] do not push Git changes unless explicitly authorized.

---

## 13. Bottom line

The existing implementation is not silently returning PhotoMaker. It is numerically active, but:

```text
partial warm start
× very small post-cap authority
× one late site family
× transformer residual
× final core epsilon anchor
```

makes the full images look unchanged.

The most important correction is not merely increasing the scalar gate. It is making the reference candidate a **complete pretrained attention operator** and then applying a clearly interpretable, bounded ownership coefficient in the common post-output hidden space.
