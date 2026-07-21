# NN5a / NN5b implementation audit and finalization fixes

**Audit date:** 21 July 2026  
**Branch:** `main_clean`  
**Audited head:** `4116a35a1146b037500e2d2988d696c5d4ed2dc0`  
**Scope:** static audit of the launchers, resolved configuration inheritance, counterfactual dataset path, paired forward, identity losses, packed-residual processor, identity-token lane, checkpoint/validation plumbing, and causal reference/noise diagnostic.  
**Repository mutation:** none.

## Verdict

NN5a and NN5b implement the intended experiments correctly at the architecture and loss-routing level.

### NN5a is correctly implemented

The current code does all of the important things correctly:

- keeps the NN4 protected PPR operator;
- retains full target PhotoMaker identity A;
- duplicates the target latent, target diffusion noise, timestep, prompt/PhotoMaker embeddings, pooled embeddings, target mask, and time IDs exactly;
- uses matched reference A in row 0 and wrong-identity reference B in row 1;
- uses one explicit reference-noise sample duplicated exactly across A/B;
- applies core diffusion reconstruction only to the matched row;
- applies absolute identity-to-B, directional B-over-A, and boundary-ring losses to the swapped row;
- keeps null suppression, cap protection, `up_blocks.0.attn1` scope, neutral reference text, disabled branched CA, disabled pose adaptation, and disabled CA face mixing;
- runs the initial approval budget as 2 × 2k optimizer steps with physical batch 1 and two-microbatch accumulation.

### NN5b is correctly wired as the one-variable representation extension

The identity-token lane:

- extracts the two unpooled PMv2 2048-D tokens from the **spatial reference**, not from target PhotoMaker A;
- extracts separate A and B tokens in counterfactual training;
- recomputes the tokens from `ppr_reference_image` during an R1→R2 diagnostic swap;
- uses separate trainable identity K/V projections;
- computes a separate target-query identity candidate;
- blends it 50/50 with the packed spatial candidate before learned-null subtraction;
- remains exact-output-safe at step zero because `connector_up` is still zero initialized;
- is included in trainability checks, optimizer groups, strict checkpoint manifests, restore checks, and validation-pipeline propagation.

No change is recommended to the NN5a pair construction, counterfactual equations, NN5b candidate equation, cap, gate, site policy, output anchor, pose adaptation, or CA-mixing settings.

## Required fixes

## 1. Preserve accumulation-window integrity after a skipped/OOM microbatch

**Files:**

- `src/trainer/sdxl_trainers.py`
- `src/trainer/base_trainer.py`

Both active launchers use physical batch 1 and effective batch 2, so one optimizer update consists of two valid microbatches.

The current skip path can produce a half update:

1. logical microbatch 0 succeeds and stores gradients;
2. logical microbatch 1 is rejected or OOMs, so `process_batch()` clears gradients and returns `skip_batch`;
3. `completed_batches` remains 1;
4. the replacement sample is again assigned `batch_idx=1`;
5. it is treated as `is_accum_end=True` but `is_accum_start=False`;
6. one loss divided by two is backpropagated, then `optimizer.step()` and `scheduler.step()` run.

This matters more for NN5 because reference recognition and wrong-reference validation add additional guarded rejection opportunities.

### Patch

```diff
diff --git a/diffusion_template/src/trainer/base_trainer.py b/diffusion_template/src/trainer/base_trainer.py
--- a/diffusion_template/src/trainer/base_trainer.py
+++ b/diffusion_template/src/trainer/base_trainer.py
@@
 class BaseTrainer:
+    def _optimizer_step_from_microbatches(self, microbatches: int) -> int:
+        """Map accepted logical microbatches to completed optimizer updates."""
+        accumulation = max(1, int(getattr(self, "grad_accum_steps", 1)))
+        return int(microbatches) // accumulation
+
@@
-        if self.accelerator.is_main_process:
-            self.writer.set_step((epoch - 1) * self.epoch_len)
+        if self.accelerator.is_main_process:
+            self.writer.set_step(
+                self._optimizer_step_from_microbatches(
+                    (epoch - 1) * self.epoch_len
+                )
+            )
             self.writer.add_scalar("general/epoch", epoch)
@@
             batch = self.process_batch(
                 batch,
                 train_metrics=self.train_metrics,
             )
             if batch.get("skip_batch", False):
+                # If a later microbatch in an accumulation window fails,
+                # process_batch() has already cleared the partial gradients.
+                # Rewind the logical counter to the start of that window so
+                # the replacement data form a complete optimizer update.
+                accumulation = max(
+                    1, int(getattr(self, "grad_accum_steps", 1))
+                )
+                partial = completed_batches % accumulation
+                if partial:
+                    completed_batches -= partial
+                    self.optimizer.zero_grad(set_to_none=True)
+                    progress_bar.update(-partial)
+                    self.train_metrics.update(
+                        "accumulation/rewound_microbatches",
+                        float(partial),
+                    )
                 continue
@@
-                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
+                    self.writer.set_step(
+                        self._optimizer_step_from_microbatches(
+                            (epoch - 1) * self.epoch_len
+                            + batch_idx
+                            + 1
+                        )
+                    )
@@
-        if self.writer is not None:
-            self.writer.set_step(epoch * self.epoch_len, part)
+        if self.writer is not None:
+            self.writer.set_step(
+                self._optimizer_step_from_microbatches(
+                    epoch * self.epoch_len
+                ),
+                part,
+            )
```

This diff also fixes the current Comet step labels. With accumulation 2, `epoch_len=4000` is 4,000 microbatches but only 2,000 optimizer steps. Until this patch is applied:

- `checkpoint-epoch1.pth` is the true **2k optimizer-step** checkpoint;
- `checkpoint-epoch2.pth` is the true **4k optimizer-step** checkpoint;
- Comet validation points can appear at 4k and 8k because they are labelled in microbatches.

### Current-run decision

Check both logs now:

```bash
grep -R -nE '\[(INVALID_SAMPLE_SKIP|OOM_SKIP)\]' \
  /home/niko/rsrch/diffusion_template/logs_new_runs 2>/dev/null || true

grep -nE '\[(INVALID_SAMPLE_SKIP|OOM_SKIP)\]' \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/logs/logs_NN5b_clean_identity_tokens_1gpu.txt \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/logs/err_NN5b_clean_identity_tokens_1gpu.txt \
  2>/dev/null || true
```

- **No matches:** the running weights are not affected; apply the patch before a resume or subsequent run.
- **A match with odd `batch_idx` under accumulation 2:** a partial window may have been replaced by a half update. Apply the patch and resume from the preceding clean checkpoint.
- **Only even `batch_idx` matches:** the rejected sample occurred at an accumulation boundary; weights are not affected by this specific bug.

## 2. Make NN5b causal validation prove that the identity-token lane followed R1→R2

**Files:**

- `src/pipelines/br_pipeline_helpers.py`
- `src/trainer/ppr_reference_noise.py`

The runtime routing is correct: `spatial_reference_images` is passed into `run_branched_setup()`, and NN5b extracts `_ppr_identity_tokens` from those images. Therefore an R2 diagnostic does recompute tokens from R2.

Two proof gaps remain:

1. inference-side `ensure_id_embeds()` silently substitutes a zero 512-D embedding when face recognition fails;
2. the five-way diagnostic verifies the swapped image and latent, but does not hash or assert the swapped identity-token tensor.

These do not require restarting training. Apply them before the 2k/4k NN5b approval matrix.

### 2a. Reject invalid spatial identity tokens and clear their generation cache

```diff
diff --git a/diffusion_template/src/pipelines/br_pipeline_helpers.py b/diffusion_template/src/pipelines/br_pipeline_helpers.py
--- a/diffusion_template/src/pipelines/br_pipeline_helpers.py
+++ b/diffusion_template/src/pipelines/br_pipeline_helpers.py
@@
 def prepare_spatial_identity_tokens(
     pipeline,
     *,
     input_id_images: Sequence[Any],
     device: torch.device,
 ) -> None:
@@
     spatial_id_embeds = ensure_id_embeds(
         pipeline,
         id_embeds=None,
         input_id_images=[[ref] for ref in refs],
         device=device,
         dtype=dtype,
     )
+    flat_identity = spatial_id_embeds.detach().float().flatten(1)
+    valid_identity = (
+        torch.isfinite(flat_identity).all(dim=1)
+        & (flat_identity.norm(dim=1) > 0)
+    )
+    if not bool(valid_identity.all()):
+        bad = (
+            (~valid_identity)
+            .nonzero(as_tuple=False)
+            .flatten()
+            .tolist()
+        )
+        raise RuntimeError(
+            "Identity-token PPR could not extract a valid spatial-reference "
+            f"recognition embedding at rows {bad}"
+        )
     with torch.no_grad():
         tokens = pipeline.id_encoder.extract_id_tokens(
             pixels,
             spatial_id_embeds,
         )
+    flat_tokens = tokens.detach().float().flatten(1)
+    valid_tokens = (
+        torch.isfinite(flat_tokens).all(dim=1)
+        & (flat_tokens.norm(dim=1) > 0)
+    )
+    if not bool(valid_tokens.all()):
+        bad = (
+            (~valid_tokens)
+            .nonzero(as_tuple=False)
+            .flatten()
+            .tolist()
+        )
+        raise RuntimeError(
+            f"Identity-token PPR produced invalid tokens at rows {bad}"
+        )
     pipeline._ppr_identity_tokens = tokens.to(
         device=device,
         dtype=pipeline.unet.dtype,
     )
@@
-    prepare_spatial_identity_tokens(
-        pipeline,
-        input_id_images=input_id_images,
-        device=device,
-    )
+    if use_branched_attention:
+        prepare_spatial_identity_tokens(
+            pipeline,
+            input_id_images=input_id_images,
+            device=device,
+        )
+    else:
+        pipeline._ppr_identity_tokens = None
@@
     for attr in (
         "_ba_packed_branch_exactly_off",
         "_ba_output_anchor_logged",
         "_ref_noise_base",
+        "_ppr_identity_tokens",
     ):
@@
     for attr in [
         "_reference_latents",
         "_face_prompt_embeds",
         "_ref_latents_all",
         "_ref_noise",
         "_ref_noise_base",
+        "_ppr_identity_tokens",
         "_ba_packed_branch_exactly_off",
         "_ba_output_anchor_logged",
     ]:
```

### 2b. Fingerprint and assert identity-token swaps

```diff
diff --git a/diffusion_template/src/pipelines/br_pipeline_helpers.py b/diffusion_template/src/pipelines/br_pipeline_helpers.py
--- a/diffusion_template/src/pipelines/br_pipeline_helpers.py
+++ b/diffusion_template/src/pipelines/br_pipeline_helpers.py
@@
         mask_ref = getattr(pipeline, "_face_mask_t_ref", None)
         pm_id = getattr(pipeline, "_pm_id_embeds_2048", None)
+        spatial_identity_tokens = getattr(
+            pipeline,
+            "_ppr_identity_tokens",
+            None,
+        )
         fingerprints = {
@@
         if pm_id is not None:
             fingerprints["target_photomaker_id_embeds_sha256"] = _sample_hashes(
                 _match_samples(pm_id)
             )
+        if spatial_identity_tokens is not None:
+            fingerprints["spatial_identity_tokens_sha256"] = _sample_hashes(
+                _match_samples(spatial_identity_tokens)
+            )
         pipeline._ba_ppr_randomness_fingerprints = fingerprints
diff --git a/diffusion_template/src/trainer/ppr_reference_noise.py b/diffusion_template/src/trainer/ppr_reference_noise.py
--- a/diffusion_template/src/trainer/ppr_reference_noise.py
+++ b/diffusion_template/src/trainer/ppr_reference_noise.py
@@
     state = {
@@
         "observed_batch_sizes": [],
+        "identity_token_lane": bool(
+            getattr(
+                getattr(trainer.config, "model", None),
+                "ba_identity_token_lane",
+                False,
+            )
+        ),
     }
@@
 def _assert_integrity(
     sample: str,
     fingerprints: dict[str, dict[str, Any]],
     diagnostics: dict[str, list[dict[str, Any]]],
     reference_ca_mode: str,
+    identity_token_lane: bool = False,
 ) -> None:
@@
-        missing = [field for field in HASH_FIELDS if field not in fingerprint]
+        required_fields = list(HASH_FIELDS)
+        if identity_token_lane:
+            required_fields.append("spatial_identity_tokens_sha256")
+        missing = [
+            field
+            for field in required_fields
+            if field not in fingerprint
+        ]
@@
     for field in (
         "spatial_reference_image_sha256",
         "reference_latents_sha256",
     ):
@@
             )
+    if identity_token_lane:
+        token_field = ("spatial_identity_tokens_sha256",)
+        if equal("R1N1", "R2N1", token_field):
+            raise RuntimeError(
+                f"{sample}: R1/R2 swap did not change identity tokens"
+            )
+        if not equal("R1N1", "R1N2", token_field):
+            raise RuntimeError(
+                f"{sample}: reference-noise swap changed identity tokens"
+            )
@@
         _assert_integrity(
             filename,
             fingerprints[filename],
             sample_diagnostics,
             state["reference_ca_mode"],
+            identity_token_lane=state["identity_token_lane"],
         )
@@
     manifest = {
@@
         "integrity_assertions_passed": True,
+        "identity_token_lane": state["identity_token_lane"],
+        "identity_token_swap_integrity_checked": state[
+            "identity_token_lane"
+        ],
```

## 3. Add an NN5b checkpoint approval entrypoint

The repository currently has a same-SDXL scale-1 checkpoint wrapper for NN5a only. It cannot safely be reused unchanged for NN5b because strict checkpoint restore must compose the NN5b config and install the identity-token projections.

Add:

```diff
diff --git a/diffusion_template/serv_new_runs/start_ba_NN5b_checkpoint_reference_vs_noise_1gpu.sh b/diffusion_template/serv_new_runs/start_ba_NN5b_checkpoint_reference_vs_noise_1gpu.sh
new file mode 100755
--- /dev/null
+++ b/diffusion_template/serv_new_runs/start_ba_NN5b_checkpoint_reference_vs_noise_1gpu.sh
@@
+#!/usr/bin/env bash
+set -euo pipefail
+
+SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
+PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
+
+CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
+if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
+    echo "Set CHECKPOINT_PATH or pass checkpoint-epoch1/2.pth as argument 1." >&2
+    exit 2
+fi
+if [[ $# -gt 0 && "$1" == "${CHECKPOINT_PATH}" ]]; then
+    shift
+fi
+CHECKPOINT_PATH="$(
+    cd -- "$(dirname -- "${CHECKPOINT_PATH}")"
+    pwd
+)/$(basename -- "${CHECKPOINT_PATH}")"
+
+CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
+if [[ -z "${CHECKPOINT_EPOCH}" \
+      && "$(basename -- "${CHECKPOINT_PATH}")" =~ checkpoint-epoch([0-9]+)\.pth$ ]]; then
+    CHECKPOINT_EPOCH="${BASH_REMATCH[1]}"
+fi
+[[ "${CHECKPOINT_EPOCH}" =~ ^[12]$ ]] || {
+    echo "NN5b approval expects epoch 1 (2k) or epoch 2 (4k)." >&2
+    exit 2
+}
+
+CHECKPOINT_STEP=$((CHECKPOINT_EPOCH * 2000))
+export NN5_NUM_PROCESSES=1
+export NN5_RUN_NAME="${RUN_NAME:-ba_NN5b_${CHECKPOINT_STEP}step_same_sdxl_scale1_reference_vs_noise}"
+export NN5_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
+export NN5_MASTER_PORT="${MASTER_PORT:-29664}"
+export GLOBAL_EFFECTIVE_BATCH=2
+export TRAIN_BATCH_SIZE=1
+export STEPS_PER_EPOCH=2000
+export NUM_EPOCHS=2
+
+OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_NN5b_${CHECKPOINT_STEP}step_same_sdxl_scale1_reference_vs_noise}"
+
+exec bash "${SCRIPT_DIR}/_start_ba_NN5b_server_common.sh" \
+    validation_only=true \
+    continue_run=false \
+    saved_checkpoint="${CHECKPOINT_PATH}" \
+    ppr_checkpoint_require_nonzero=true \
+    strict_checkpoint_model_config=true \
+    ppr_expected_checkpoint_epoch="${CHECKPOINT_EPOCH}" \
+    ppr_reference_noise_test=true \
+    ppr_reference_noise_scale=1.0 \
+    ppr_reference_noise_output_dir="${OUTPUT_DIR}" \
+    ppr_reference_noise_overwrite="${OVERWRITE_OUTPUT:-false}" \
+    ppr_reference_noise_seeds="${NOISE_SEEDS:-[918273,271828]}" \
+    datasets.val.manual_val.limit="${LIMIT:-96}" \
+    dataloaders.manual_val.batch_size="${BATCH_SIZE:-12}" \
+    "$@"
```

Example on the NFS server:

```bash
cd /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template

bash serv_new_runs/start_ba_NN5b_checkpoint_reference_vs_noise_1gpu.sh \
  saved/ba_NN5b_clean_identity_tokens_1gpu/checkpoint-epoch1.pth
```

Repeat with `checkpoint-epoch2.pth`.

## Recommended instrumentation before interpreting NN5b

This is not required for model correctness, but it makes an NN5b failure attributable.

The current processor tensor diagnostic records only the already blended `reference_candidate`. Preserve the two component candidates:

```diff
diff --git a/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py b/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py
--- a/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py
+++ b/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py
@@
         reference_candidate = self._from_heads(reference_candidate).to(
             target_base.dtype
         )
+        spatial_reference_candidate = reference_candidate
+        identity_candidate = None
@@
                 tensors = {
                     "reference_hidden": _sample_rows(reference_hidden),
                     "reference_candidate": _sample_rows(reference_candidate),
@@
                 }
+                if identity_candidate is not None:
+                    tensors["spatial_reference_candidate"] = _sample_rows(
+                        spatial_reference_candidate
+                    )
+                    tensors["identity_candidate"] = _sample_rows(
+                        identity_candidate
+                    )
```

Then make the comparison stage list conditional:

```diff
diff --git a/diffusion_template/src/trainer/ppr_reference_noise.py b/diffusion_template/src/trainer/ppr_reference_noise.py
--- a/diffusion_template/src/trainer/ppr_reference_noise.py
+++ b/diffusion_template/src/trainer/ppr_reference_noise.py
@@
-            stages = (
-                (
+            if key[0] == "processor_tensor_signature":
+                stages = [
                     "reference_hidden",
                     "reference_candidate",
                     "connector_down",
                     "raw_delta",
                     "bounded_delta",
                     "applied_delta",
-                )
-                if key[0] == "processor_tensor_signature"
-                else (
+                ]
+                for optional_stage in (
+                    "spatial_reference_candidate",
+                    "identity_candidate",
+                ):
+                    if optional_stage in left and optional_stage in right:
+                        stages.append(optional_stage)
+            else:
+                stages = [
                     "target_epsilon_pre_anchor",
                     "target_epsilon_post_anchor",
-                )
-            )
+                ]
```

This answers whether NN5b failed because:

- identity tokens did not change;
- identity attention changed but was too small;
- identity attention dominated the spatial candidate;
- the connector removed the identity-specific direction;
- the final trajectory washed it out.

## Tests to add with the fixes

The existing tests cover basic collation, directional-loss behavior, connector parity, and token sensitivity. Add these regression cases:

1. **Accumulation rollback**
   - valid microbatch 0;
   - rejected microbatch 1;
   - verify no optimizer or scheduler step;
   - verify replacement sample is assigned accumulation-start semantics.

2. **NN5b swapped-reference token integrity**
   - R1 and R2 images produce different token hashes;
   - R1N1 and R1N2 produce identical token hashes;
   - zero/invalid recognition embedding raises.

3. **Identity-projection gradient**
   - open `connector_up`;
   - backpropagate an identity-token-sensitive loss;
   - verify nonzero gradients on all four `identity_to_{k,v}.{0,2}.weight` tensors.

4. **Loss routing**
   - changing only `eps_swap` cannot alter core diffusion loss;
   - ring loss is exactly zero when `eps_match == eps_swap`;
   - no counterfactual loss is emitted above the configured timestep.

## Run interpretation

### NN5a

Use the existing wrapper at epoch 1 and epoch 2:

```bash
bash jul_serv_runs/start_ba_NN5a_checkpoint_reference_vs_noise_1gpu.sh \
  /absolute/path/to/checkpoint-epoch1.pth
```

Approve continuation only from same-SDXL, scale-1 directional results—not from normal validation or a larger face delta.

### NN5b

Use the new NN5b-specific wrapper after applying the token-integrity patch. Compare NN5b directly with NN5a at the same checkpoint epoch, seeds, batch size, and residual scale.

Watch:

```text
ba_cf/applied_fraction
ba_cf/directional_gain
ba_cf/sim_to_matched
ba_cf/sim_to_wrong
ba_cf/reference_identity_cosine_A_B
ba_cf/reference_noise_equal
ba_aux/null_residual
ba_aux/cap_excess
```

A/B reference cosine is already logged. If near-duplicate identities are common—for example, a large upper tail close to one—add a configurable rejection threshold before another run. Do not introduce that threshold mid-run without first checking the observed distribution.

## Final recommendation

The NN5 model code does not need an architectural rewrite.

Apply:

1. accumulation-window rewind and optimizer-step labelling;
2. strict NN5b token extraction and token-swap fingerprints;
3. an NN5b-specific same-SDXL scale-1 checkpoint diagnostic entrypoint.

The currently running training can remain valid if there are no skip/OOM records. The NN5b validation hardening and diagnostic wrapper can be added before checkpoint evaluation without restarting training.
