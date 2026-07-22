# NN6a quick implementation audit and concise fixes

**Date:** 22 July 2026  
**Repository/branch:** `kolyangg/rsrch`, current `main_clean`  
**Launch audited:**

```bash
mls job submit --config \
  ./serv_new_runs/run_ba_NN6a_factorized_identity_only_up0_combined_2gpu.yaml
```

**Repository mutation:** none.  
**Audit type:** static code and launch-path review; the GPU job and test suite were not executed here.

## Verdict

The **NN6a architecture itself is implemented correctly and follows the recommendation closely**. I found no reason to redesign or cancel the experiment on architectural grounds.

Correctly implemented:

- ordinary target self-attention remains the protected base;
- clean PMv2 two-token identity memory is the only target-conditioning lane;
- the noised spatial-reference K/V lane is disabled and its projections, null, connector, and gate are not instantiated;
- real and learned-null identity tokens use the same identity K/V projections;
- the identity lane has a dedicated rank-16, bias-free connector;
- connector-up is zero initialized;
- the identity lane has its own gate and RMS cap;
- injection is limited to `up_blocks.0.attn1`;
- residual application is restricted to the feathered target-face core;
- `base_outside_core` preserves ordinary PhotoMaker epsilon outside the core;
- NN5a counterfactual supervision and weights remain unchanged;
- full target PhotoMaker A conditioning remains enabled;
- branched CA, pose adaptation, and CA face mixing remain disabled;
- strict optimizer/checkpoint manifests include only the identity-lane parameters;
- RealVis validation matches the existing RealVis-derived target bbox file;
- the post-training diagnostic checks per-lane tensors and final-image reference-noise invariance.

Three concise changes are advisable before calling the implementation final.

---

## 1. Required for robust 2-GPU training: synchronize the sampled timestep

### Problem

`PhotomakerBranchedLora.forward()` samples one local `t_scalar` independently on each rank:

```python
t_scalar = torch.randint(...)
counterfactual_active = t_scalar <= 300
```

The counterfactual loss keys are emitted only when `counterfactual_active` is true. The trainer correspondingly enters `accelerator.gather()` only for keys that exist.

If ranks ever obtain different timesteps, one rank can enter a gather while the other skips it, causing a distributed hang/desynchronization.

Both ranks currently start with the same seed, so they will often remain synchronized. That is not a sufficient guarantee:

- ranks process different data;
- an invalid/OOM sample can make one rank consume a different number of random values;
- resume and future data-path changes can also separate RNG states.

### Fix

Broadcast the batch-level timestep from rank 0 before deriving `timesteps` and `counterfactual_active`.

```diff
diff --git a/diffusion_template/src/model/photomaker_branched/lora2.py b/diffusion_template/src/model/photomaker_branched/lora2.py
--- a/diffusion_template/src/model/photomaker_branched/lora2.py
+++ b/diffusion_template/src/model/photomaker_branched/lora2.py
@@
         t_scalar = torch.randint(
             0,
             max_timestep_exclusive,
             (1,),
             device=latents.device,
         ).long()
+        # Counterfactual losses are conditionally emitted and gathered by the
+        # trainer. Keep that control-flow decision identical on every DDP rank
+        # instead of relying on rank-local RNG streams remaining in lockstep.
+        if (
+            torch.distributed.is_available()
+            and torch.distributed.is_initialized()
+        ):
+            torch.distributed.broadcast(t_scalar, src=0)
         timesteps = t_scalar.repeat(batch_size)
```

Add a trainer-side assertion and count actual optimizer-step decisions rather than rank events:

```diff
diff --git a/diffusion_template/src/trainer/sdxl_trainers.py b/diffusion_template/src/trainer/sdxl_trainers.py
--- a/diffusion_template/src/trainer/sdxl_trainers.py
+++ b/diffusion_template/src/trainer/sdxl_trainers.py
@@
             gathered_values = self.accelerator.gather(
                 value.detach().reshape(1)
             )
             gathered = gathered_values.mean()
             train_metrics.update(metric_name, gathered.item())
             if output_name == "ba_cf_applied_fraction":
-                active = int((gathered_values > 0.5).sum().item())
-                inactive = int(gathered_values.numel()) - active
+                active_flags = gathered_values > 0.5
+                if not bool(
+                    (active_flags == active_flags[:1]).all().item()
+                ):
+                    raise RuntimeError(
+                        "Counterfactual activation differs across DDP ranks"
+                    )
+                active = int(active_flags[0].item())
+                inactive = 1 - active
                 self._ba_cf_active_updates = int(
                     getattr(self, "_ba_cf_active_updates", 0)
                 ) + active
```

This also makes `ba_cf/active_updates` mean optimizer-step decisions rather than twice that number on two GPUs.

### Current job

If the job has already started:

```bash
grep -nE '\[(INVALID_SAMPLE_SKIP|OOM_SKIP)\]' \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/logs/logs_NN6a_combined_2gpu.txt \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/logs/err_NN6a_combined_2gpu.txt \
  2>/dev/null || true
```

- No skip/OOM records and normal progress: the ranks are likely still following the same timestep sequence, although the broadcast remains the correct final implementation.
- Any skip/OOM record: restart from the preceding clean checkpoint after applying the broadcast, because rank RNG states may have diverged.
- If the run is only just starting, apply the patch and restart now.

---

## 2. Required diagnostic fix: exact tolerance must require exact hashes

### Problem

The identity-only diagnostic is configured with:

```yaml
ppr_identity_noise_tolerance: 0.0
```

The current check calculates:

```python
exact = left_hash == right_hash
relative = difference_between_512_value_sketches

if not exact and relative > tolerance:
    raise
```

With tolerance zero, a full tensor can have a different SHA-256 hash but pass if all changed elements happen to fall outside the deterministic 512-value sketch, making `relative == 0`.

The final RGB comparison is exact, but the requested invariant also covers every identity-lane intermediate tensor.

### Fix

```diff
diff --git a/diffusion_template/src/trainer/ppr_reference_noise.py b/diffusion_template/src/trainer/ppr_reference_noise.py
--- a/diffusion_template/src/trainer/ppr_reference_noise.py
+++ b/diffusion_template/src/trainer/ppr_reference_noise.py
@@
                     exact = (
                         left[field]["sha256"]
                         == right[field]["sha256"]
                     )
                     relative = _relative_signature(
                         left[field],
                         right[field],
                     )
-                    if (
-                        not exact
-                        and relative > float(identity_noise_tolerance)
-                    ):
+                    tolerance = float(identity_noise_tolerance)
+                    violates_invariant = (
+                        (not exact)
+                        if tolerance <= 0.0
+                        else ((not exact) and relative > tolerance)
+                    )
+                    if violates_invariant:
                         raise RuntimeError(
                             f"{sample}: identity-only reference-noise leak at "
                             f"{field} {left_name}/{right_name}: "
-                            f"relative={relative}"
+                            f"exact={exact}, relative={relative}, "
+                            f"tolerance={tolerance}"
                         )
```

Apply this before the automatic 4k diagnostic starts. It does not change training weights.

### Complete the token-hash symmetry checks

The code verifies R1 token stability across noise, but the direct hash guard should be symmetric:

```diff
@@
     if identity_token_lane:
         token_field = ("spatial_identity_tokens_sha256",)
         if equal("R1N1", "R2N1", token_field):
             raise RuntimeError(
                 f"{sample}: R1/R2 swap did not change identity tokens"
             )
         if not equal("R1N1", "R1N2", token_field):
             raise RuntimeError(
                 f"{sample}: reference-noise swap changed identity tokens"
             )
+        if not equal("R2N1", "R2N2", token_field):
+            raise RuntimeError(
+                f"{sample}: R2 reference-noise swap changed identity tokens"
+            )
+        if equal("R1N2", "R2N2", token_field):
+            raise RuntimeError(
+                f"{sample}: N2 R1/R2 swap did not change identity tokens"
+            )
```

### Regression test

Add a test in `tests/test_nn5_components.py` where two tensor signatures have:

- different SHA-256 values;
- identical 512-value sketches;
- `identity_noise_tolerance=0.0`.

`_assert_integrity()` must reject the pair.

---

## 3. Protocol gap: the combined job diagnoses only 4k, not 2k

The launcher saves `checkpoint-epoch1.pth`, but the combined script runs the causal matrix only for `checkpoint-epoch2.pth`.

That does not invalidate NN6a, but it omits the requested 2k/4k causal curve and removes the early indication of whether the clean lane opens correctly.

### Minimal fix: evaluate both saved checkpoints after training

```diff
diff --git a/diffusion_template/serv_new_runs/start_ba_NN6a_train_then_diagnose_2gpu.sh b/diffusion_template/serv_new_runs/start_ba_NN6a_train_then_diagnose_2gpu.sh
--- a/diffusion_template/serv_new_runs/start_ba_NN6a_train_then_diagnose_2gpu.sh
+++ b/diffusion_template/serv_new_runs/start_ba_NN6a_train_then_diagnose_2gpu.sh
@@
-CHECKPOINT="${PROJECT_DIR}/saved/${TRAIN_RUN_NAME}/checkpoint-epoch2.pth"
-[[ -f "${CHECKPOINT}" ]] || {
-    echo "Missing NN6a 4k checkpoint: ${CHECKPOINT}" >&2
-    exit 3
-}
-# The training processes have exited, so use GPU 0 for the deterministic
-# 96-case test.
-CHECKPOINT_EPOCH=2 \
-RUN_NAME="${TRAIN_RUN_NAME}_4k_diagnostic" \
-OUTPUT_DIR="${PROJECT_DIR}/ppr_${TRAIN_RUN_NAME}_4000step_realvis_scale1_reference_vs_noise" \
-CUDA_VISIBLE_DEVICES=0 \
-BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-12}" \
-bash "${SCRIPT_DIR}/start_ba_NN6a_checkpoint_reference_vs_noise_1gpu.sh" \
-    "${CHECKPOINT}"
+# The DDP workers have exited. Evaluate both approval checkpoints on GPU 0.
+for CHECKPOINT_EPOCH in 1 2; do
+    CHECKPOINT_STEP=$((CHECKPOINT_EPOCH * 2000))
+    CHECKPOINT="${PROJECT_DIR}/saved/${TRAIN_RUN_NAME}/checkpoint-epoch${CHECKPOINT_EPOCH}.pth"
+    [[ -f "${CHECKPOINT}" ]] || {
+        echo "Missing NN6a ${CHECKPOINT_STEP}-step checkpoint: ${CHECKPOINT}" >&2
+        exit 3
+    }
+    CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH}" \
+    RUN_NAME="${TRAIN_RUN_NAME}_${CHECKPOINT_STEP}step_diagnostic" \
+    OUTPUT_DIR="${PROJECT_DIR}/ppr_${TRAIN_RUN_NAME}_${CHECKPOINT_STEP}step_realvis_scale1_reference_vs_noise" \
+    CUDA_VISIBLE_DEVICES=0 \
+    BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-12}" \
+    bash "${SCRIPT_DIR}/start_ba_NN6a_checkpoint_reference_vs_noise_1gpu.sh" \
+        "${CHECKPOINT}"
+done
```

This is post-hoc rather than an automatic early stop, but preserves the scientifically useful 2k/4k comparison without complicating the cluster job.

---

## Final assessment

### Architecture

**Pass.** NN6a is a faithful implementation of the factorized identity-only `up_blocks.0` experiment. No architecture or hyperparameter change is recommended before seeing its causal results.

### Training launch

**Pass after DDP timestep synchronization.** The two-GPU global effective batch and 4k budget are correct.

### Diagnostic

**Pass after the exact-hash correction.** The current final-image equality guard is strong, but the intermediate-tensor exact invariant has a small sketch-based loophole.

### Evaluation schedule

**Minor deviation.** Add the 2k checkpoint diagnostic; retain the full 96-case RealVis scale-1 matrix at 4k.

Do not add spatial K/V, new sites, PM attenuation, branched CA, pose adaptation, CA mixing, or larger caps to this run. The purpose of NN6a is to obtain a clean answer about the dedicated PMv2 identity-token lane.
