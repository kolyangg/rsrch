# INIT-R-NATIVE — Quick Implementation Plan

**Experiment ID:** `CL39-INIT-R-NATIVE`  
**Suggested config name:** `CL39I_cosmic_reference_native_init_24k.yaml`  
**Parent:** `CL39_cosmic_null_key_confidence_router_24k.yaml`  
**Purpose:** Test whether CL39's explicit reference branch can learn effectively without inheriting PhotoMaker's self-attention Q/K/V delta at initialization.

---

## 1. Exact scientific change

Current branch projection initialization is effectively:

\[
W_{\text{noise},0}=W_{\text{SDXL}}+\Delta W_{\text{PM}}
\]

\[
W_{\text{ref},0}=W_{\text{SDXL}}+\Delta W_{\text{PM}}
\]

with each branch's trainable LoRA contribution initially zero because `lora_B` is zero-initialized.

For INIT-R-NATIVE, change only the reference branch:

\[
W_{\text{noise},0}=W_{\text{SDXL}}+\Delta W_{\text{PM}}
\]

\[
W_{\text{ref},0}=W_{\text{SDXL}}
\]

\[
\Delta W_{\text{branch},0}=B A=0
\]

This means:

- `noise_to_q`, `noise_to_k`, `noise_to_v` retain the current PhotoMaker-effective initialization.
- `ref_to_q`, `ref_to_k`, `ref_to_v` start from native SDXL weights.
- The trainable branch LoRA matrices retain their current initialization: random `A`, zero `B`.
- **Do not initialize complete Q/K/V matrices to literal zeros.**
- No new trainable parameters are added.

The target-to-reference message still uses target/noise Q with reference K/V:

\[
R = W_o\,\mathrm{Attn}(Q_n(T),K_r(H_r),V_r(H_r))
\]

`ref_to_q` is nevertheless made native because it controls evolution of the doubled reference row itself.

---

## 2. Keep everything else exactly equal to CL39

This experiment must not change:

- CL39 temporal-frequency routing.
- CL27 frequency-surface objective.
- CL39 entropy confidence.
- Native `N` anchor.
- Target/noise Q/K/V initialization.
- PhotoMaker ID encoder and inserted ID tokens.
- Native PhotoMaker/SDXL cross-attention.
- PhotoMaker-default outer adapter.
- Generic outer adapter.
- `attn.to_out[0]` and output routing.
- Dataset, reference policy, masks, loss, optimizer, LR, scheduler, batch size or 24k budget.
- Validation prompts, references, boxes, seeds or inference timing.
- `photomaker_start_step=10`, `merge_start_step=10`, `branched_attn_start_step=15`.
- `pose_adapt_ratio=0` and `ca_mixing_for_face=false`.

This is an **initialization-only training experiment**. It must start from a fresh training run, not resume a CL39 checkpoint.

---

## 3. Configuration field

Add one defaults-preserving setting:

```yaml
ba_reference_qkv_pm_delta_scale: 1.0
```

Semantics:

- `1.0`: historical behavior, `W_ref = W_SDXL + ΔW_PM`.
- `0.0`: INIT-R-NATIVE, `W_ref = W_SDXL`.

For this first experiment, reject values other than exactly `0.0` or `1.0`. Fractional scales can be a later experiment only if the binary ablation fails for optimization reasons.

---

## 4. File-by-file implementation

### 4.1 `src/model/photomaker_branched/attn_processor_cleanest.py`

#### A. Extend `_clone_effective_linear`

Replace the current helper with a defaults-preserving scale argument:

```python
def _clone_effective_linear(
    attn_linear,
    *,
    rank: int = 128,
    pm_delta_scale: float = 1.0,
):
    if float(pm_delta_scale) not in (0.0, 1.0):
        raise ValueError(
            "pm_delta_scale must be exactly 0.0 or 1.0 for the sealed experiment"
        )

    base = (
        attn_linear.get_base_layer()
        if hasattr(attn_linear, "get_base_layer")
        else attn_linear
    )

    cloned = BranchLoRALinear(
        base.in_features,
        base.out_features,
        rank=rank,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )

    with torch.no_grad():
        weight = base.weight.detach().clone()

        if (
            pm_delta_scale != 0.0
            and hasattr(attn_linear, "lora_A")
            and "default" in attn_linear.lora_A
        ):
            pm_delta = attn_linear.get_delta_weight("default").detach()
            weight.add_(
                pm_delta.to(device=weight.device, dtype=weight.dtype),
                alpha=float(pm_delta_scale),
            )

        cloned.base_weight.copy_(weight)

        if base.bias is not None:
            cloned.base_bias.copy_(base.bias.detach())

    return cloned
```

The historical path with `pm_delta_scale=1.0` must remain numerically identical.

#### B. Extend `BranchedAttnProcessor.__init__`

```python
def __init__(
    self,
    hidden_size: int,
    cross_attention_dim: Optional[int] = None,
    scale: float = 1.0,
    reference_qkv_pm_delta_scale: float = 1.0,
):
    super().__init__()
    ...
    self.reference_qkv_pm_delta_scale = float(
        reference_qkv_pm_delta_scale
    )
    if self.reference_qkv_pm_delta_scale not in (0.0, 1.0):
        raise ValueError(
            "reference_qkv_pm_delta_scale must be 0.0 or 1.0"
        )
```

#### C. Change only reference initialization

```python
def init_from_attention(self, attn) -> None:
    ref_scale = self.reference_qkv_pm_delta_scale

    self.ref_to_q = _clone_effective_linear(
        attn.to_q, pm_delta_scale=ref_scale
    )
    self.ref_to_k = _clone_effective_linear(
        attn.to_k, pm_delta_scale=ref_scale
    )
    self.ref_to_v = _clone_effective_linear(
        attn.to_v, pm_delta_scale=ref_scale
    )

    # Historical PM-effective initialization stays unchanged for N.
    self.noise_to_q = _clone_effective_linear(
        attn.to_q, pm_delta_scale=1.0
    )
    self.noise_to_k = _clone_effective_linear(
        attn.to_k, pm_delta_scale=1.0
    )
    self.noise_to_v = _clone_effective_linear(
        attn.to_v, pm_delta_scale=1.0
    )
```

Do not change `BranchLoRALinear.forward`, LoRA rank, bias handling or zero initialization.

---

### 4.2 `src/model/photomaker_branched/hardcase_attn_processor.py`

Add the constructor argument and pass it to the parent:

```python
def __init__(
    self,
    *,
    hidden_size: int,
    cross_attention_dim: int,
    scale: float,
    hardcase_mode: str,
    reference_qkv_pm_delta_scale: float = 1.0,
    ...
):
    super().__init__(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        scale=scale,
        reference_qkv_pm_delta_scale=reference_qkv_pm_delta_scale,
    )
```

In `create_hardcase_processor(...)`:

```python
return HardcaseBranchedAttnProcessor(
    ...
    reference_qkv_pm_delta_scale=float(
        getattr(
            pipeline,
            "ba_reference_qkv_pm_delta_scale",
            1.0,
        )
    ),
    ...
)
```

No routing equation should change.

---

### 4.3 `src/model/photomaker_branched/branched_runtime.py`

When creating the ordinary base `BranchedAttnProcessor`, pass the same setting:

```python
proc = BranchedAttnProcessor(
    hidden_size=hidden_size,
    cross_attention_dim=hidden_size,
    scale=scale,
    reference_qkv_pm_delta_scale=float(
        getattr(
            pipeline,
            "ba_reference_qkv_pm_delta_scale",
            1.0,
        )
    ),
)
```

The hardcase factory already receives it through the preceding change.

Do not rebuild processors per denoising step. Existing processors should only receive updated masks/runtime flags.

---

### 4.4 `src/model/photomaker_branched/e13_contract.py`

#### A. Add default

```python
DEFAULT_E13_SETTINGS = {
    ...
    "ba_reference_qkv_pm_delta_scale": 1.0,
}
```

The existing normalization code should convert it to `float`.

#### B. Copy it to validation runtime

Add to `PIPELINE_RUNTIME_SETTINGS`:

```python
"ba_reference_qkv_pm_delta_scale",
```

#### C. Fail closed

In `initialise_e13_contract(...)`:

```python
reference_scale = values["ba_reference_qkv_pm_delta_scale"]

if reference_scale not in (0.0, 1.0):
    raise ValueError(
        "ba_reference_qkv_pm_delta_scale must be exactly 0.0 or 1.0"
    )

if reference_scale == 0.0 and not values["ba_null_key_router_enabled"]:
    raise ValueError(
        "INIT-R-NATIVE is defined only as a CL39 leaf"
    )
```

Optionally also require the exact CL39 parent contract:

```python
if reference_scale == 0.0 and (
    hardcase_mode != "temporal_frequency"
    or not values["ba_frequency_surface_loss_enabled"]
    or not values["ba_null_key_router_enabled"]
):
    raise ValueError(
        "INIT-R-NATIVE requires the complete CL39 parent"
    )
```

#### D. Record the non-default initialization in the manifest

The branch base weights are frozen buffers and are not part of `trainable_unet`. The checkpoint manifest must therefore prevent loading a native-reference checkpoint under PM-effective reconstruction.

Add this **only when the setting is non-default**, so historical CL39 manifests remain exactly unchanged:

```python
reference_scale = float(
    getattr(model, "ba_reference_qkv_pm_delta_scale", 1.0)
)

if reference_scale != 1.0:
    hard_v1_extensions["reference_qkv_initialization"] = {
        "noise_qkv_base": "sdxl_plus_photomaker_default",
        "reference_qkv_base": "sdxl_native",
        "reference_pm_delta_scale": reference_scale,
        "branch_lora_B_zero_init": True,
    }
```

Set a new processor code version only for this arm:

```python
if reference_scale != 1.0:
    processor_code_version = 5
elif model.ba_null_key_router_enabled:
    processor_code_version = 4
...
```

Do **not** change schema version or expected trainable counts.

This conditional-manifest design preserves loading of existing CL39 checkpoints because their default manifest remains byte-for-byte unchanged.

---

### 4.5 `tools/validate_e13_family_config.py`

Add checks that:

- INIT-R-NATIVE inherits CL39.
- `ba_reference_qkv_pm_delta_scale == 0.0`.
- All other CL39 settings are unchanged.
- Trainable tensor count remains `2,240`.
- Trainable parameter count remains `219,217,920`.
- No new optimizer role exists.
- Pose adaptation and face CA mixing remain disabled.

---

## 5. New Hydra leaf

Create:

`src/configs/CL39I_cosmic_reference_native_init_24k.yaml`

```yaml
defaults:
  - CL39_cosmic_null_key_confidence_router_24k
  - _self_

# INIT-R-NATIVE:
# Remove the PhotoMaker default LoRA delta only from the explicit
# reference branch Q/K/V base initialization. N remains PM-effective.
model:
  e13_settings:
    ba_reference_qkv_pm_delta_scale: 0.0

writer:
  experiment_comment: >-
    CL39 INIT-R-NATIVE: ref_to_q/k/v start from native SDXL weights,
    while noise_to_q/k/v retain SDXL plus PhotoMaker-default weights.
    Branch LoRA B remains zero-initialized; all CL39 routing, objectives,
    trainables, data and validation settings are unchanged.
```

Also create the usual:

- `experiments/cosmic_large/CL39I_...json`
- `serv_run_packages/CL39I_.../run_CL39I_..._1gpu.yaml`

Copy the CL39 package and change only the config/experiment identifiers.

---

## 6. Required initialization assertions

Add a focused test/helper that resolves one real attention module before training and verifies:

```python
native_q = attn.to_q.get_base_layer().weight.detach()
pm_q = attn.to_q.get_delta_weight("default").detach()

assert torch.allclose(
    proc.noise_to_q.base_weight,
    native_q + pm_q.to(native_q),
)

assert torch.allclose(
    proc.ref_to_q.base_weight,
    native_q,
)

assert torch.count_nonzero(proc.noise_to_q.lora_B) == 0
assert torch.count_nonzero(proc.ref_to_q.lora_B) == 0
```

Repeat for K and V.

Across all installed self-attention processors assert:

```python
assert proc.reference_qkv_pm_delta_scale == 0.0
```

Also verify that reference base weights are **not** literal zero matrices.

Recommended startup telemetry:

- mean `||W_noise_base - W_native|| / ||W_native||`;
- mean `||W_ref_base - W_native|| / ||W_native||`;
- `lora_B` nonzero count at initialization;
- number of affected processors.

Expected at step 0:

- noise difference from native: nonzero;
- reference difference from native: exactly zero;
- branch `lora_B` nonzero count: zero;
- affected self-attention processors: 70.

---

## 7. Verification gates before the 24k run

### 7.1 Historical parity

With the default setting omitted or set to `1.0`:

- Existing CL39 checkpoint loads with no manifest mismatch.
- Existing CL39 manifest is unchanged.
- A 12-image fixed validation smoke is byte-identical to sealed CL39.
- Trainable names, shapes and optimizer groups are identical.

### 7.2 INIT-R-NATIVE construction

With scale `0.0`:

- Reference Q/K/V bases equal native SDXL bases.
- Noise Q/K/V bases equal native plus PM-default delta.
- No new parameters or checkpoint tensors.
- New manifest contains `reference_qkv_initialization`.
- Loading the checkpoint under scale `1.0` fails closed.
- Loading it under scale `0.0` succeeds.

### 7.3 Training smoke

Run at least two optimizer steps and check:

- finite diffusion and auxiliary losses;
- nonzero gradients on reference `lora_B` at the first step;
- reference `lora_B` changes after the first optimizer step;
- reference `lora_A` may have zero gradient on the very first step because `B=0`, but must receive finite/nonzero gradient after `B` has moved;
- no NaN/Inf in Q/K/V, attention output or CL39 telemetry;
- all ranks construct the same initialization.

Do not alter LR or add a warmup specific to this arm in the first run.

---

## 8. Training protocol

Train from scratch for the standard 24k budget with the same seed and all CL39 settings.

Log by branch and projection:

- `ref_to_q/k/v` LoRA update RMS;
- `noise_to_q/k/v` LoRA update RMS;
- gradient RMS for reference and noise LoRAs;
- ratio of learned reference LoRA delta to the omitted PM delta;
- CL39 null-mass/reference-fraction telemetry;
- routed correction/native RMS;
- ordinary loss and CL27 auxiliary loss.

Run fixed-96 validation at the same cadence as CL39. Preserve the same checkpoint-selection rule and report both:

- selected checkpoint;
- exact 24k endpoint.

---

## 9. Evaluation

Compare INIT-R-NATIVE directly against matched CL39.

At the selected checkpoint run:

1. normal actual route;
2. N-only / correction-zero;
3. raw R-on-face stress route;
4. confidence forced to one;
5. low-only;
6. high-only;
7. spatial-reference shuffle while retaining correct PM identity tokens.

Primary metrics:

- subject-v2 ID similarity;
- text similarity;
- mask IoU;
- TOPIQ-Face;
- MUSIQ;
- MANIQA;
- fixed Skiing/Crying topology rubric;
- severe raw-R artifact count on the predeclared 16 diagnostic cells.

---

## 10. Interpretation and promotion gate

### Strong success

- Actual ID is at least non-inferior to CL39 (`ΔID >= -0.005`) and preferably improves.
- Face quality and prompt adherence are non-inferior.
- Raw-R severe artifacts decline materially.
- Actual remains better than N-only.
- Correct spatial-reference shuffle degrades the result.
- Reference LoRAs learn a substantial update without reconstructing the omitted PM delta exactly.

This would show that BA can learn a useful explicit reference pathway without inheriting PM self-attention Q/K/V.

### Scientific success but no promotion

- Final quality is close to CL39 but early convergence is slower.
- Learned reference LoRA approximately rebuilds the omitted PM delta.

Conclusion: PM initialization is mainly an optimization prior, not a necessary final representation.

### Failure

- Persistent large ID/quality regression.
- Reference gradients vanish or training becomes unstable.
- Raw R worsens and the actual correction loses causal benefit.

Conclusion: PM-effective initialization is currently important. A later, separate experiment may test a partial scale such as `0.25` or `0.5`, but do not bundle that into INIT-R-NATIVE.

---

## 11. Minimal agent completion checklist

- [ ] Add `ba_reference_qkv_pm_delta_scale` with default `1.0`.
- [ ] Scale only the PM-default delta folded into `ref_to_q/k/v`.
- [ ] Keep `noise_to_q/k/v` at scale `1.0`.
- [ ] Pass the setting through hardcase/base processor construction.
- [ ] Add conditional non-default manifest entry and code version 5.
- [ ] Keep historical CL39 manifest/checkpoint parity.
- [ ] Add the CL39I Hydra leaf and Serv package.
- [ ] Prove Q/K/V initialization mathematically on real modules.
- [ ] Prove unchanged trainable/optimizer counts.
- [ ] Pass two-step training smoke.
- [ ] Launch fresh matched 24k run.
- [ ] Run fixed-96 and branch-causal evaluations.

---

## Source basis

This plan targets the current clean CL39 implementation in:

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/hardcase_attn_processor.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/e13_contract.py`
- `src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml`

It preserves the established CL39 equation in which native `N` remains the anchor and only the reference-derived correction is frequency- and confidence-routed.
