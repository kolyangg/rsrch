# BA bug-fix and N27/N28 proposal

Date: 14 July 2026

Status: **approved and implemented on 14 July 2026**

## Fixed constraints

1. PhotoMaker remains the full-image generator. BA may change only the face bbox.
2. Keep hard rectangular target and reference bboxes. No soft masks, parsing masks, or dynamic re-tracking in these runs.
3. Do not use `POSE_ADAPT_RATIO` or `CA_MIXING_FOR_FACE`. They blend PhotoMaker and BA face representations without fixing correspondence.
4. Preserve all legacy behavior behind explicit flags; defaults remain compatible with existing checkpoints.
5. Keep N25/N26 loss weights and training length initially, so the experiment isolates architecture rather than retuning losses.

## Proposed fixes

### F1. Make PhotoMaker the invariant target path

**Problem:** [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py), approximately lines 330-505, replaces normal target self-attention with separate background/face outputs. The background is therefore not guaranteed to be the ordinary PhotoMaker result.

**Change:** add opt-in `ba_sa_mode=pm_face_residual`:

```python
pm_out = standard_photomaker_self_attention(noise_hidden)
face_delta = zero_init_delta_projection(face_attention(...))
target_out = pm_out + hard_target_mask * active_face_gate * face_delta
```

- Compute `pm_out` with the existing effective PhotoMaker `to_q/to_k/to_v/to_out` weights.
- Add the BA correction **after** its own zero-initialized output projection.
- Initialize only `face_delta_out` to zero and initialize the gate to an active value such as `1`. Do not initialize both factors to zero, which would block gradients to the new branch. At initialization the new mode must still be numerically equal to PhotoMaker.
- Do not use the current absolute merge at line 505 in this mode.
- Keep `legacy` and `dual_attention_gate` untouched for old checkpoints.

**Files:**

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py` for flag propagation and checkpoint loading
- `src/model/photomaker_branched/lora2_helpers.py` for trainable parameter selection

### F2. Hard-merge the final noise prediction with PhotoMaker

**Problem:** even a masked residual inside an early UNet layer can propagate through later convolutions/attention and alter pixels outside the bbox.

**Change:** add opt-in `ba_pm_preservation_mode=hard_epsilon_merge` in `two_branch_predict()`:

```python
with torch.no_grad():
    eps_pm = run_original_photomaker_target(...)
eps_ba = run_branched_face_path(...)
eps = eps_pm * (1 - hard_latent_bbox) + eps_ba * hard_latent_bbox
```

- Use the same latent, timestep, prompt, PhotoMaker ID conditioning, and seed for both predictions.
- Swap between the already-owned original and branched processor instances; never rebuild processors during this pass, so optimizer/checkpoint references remain valid.
- No Gaussian blur at this merge.
- Outside-bbox BA gradient is exactly zero.
- This guarantees the per-step epsilon outside the bbox comes from PhotoMaker. It does not promise pixel-identical output at the VAE boundary, but it is much stronger than the current branch merge.
- Cost is one extra no-grad target UNet prediction, roughly three branch-equivalents instead of N25's two.

**Files:**

- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/pipelines/br_pipeline_helpers.py`

### F3. Replace zero-masked full-grid K/V with compact hard-bbox tokens

**Problem:** lines 399-470 of `attn_processor_cleanest.py` zero tokens outside the reference bbox but still include them in softmax. These tokens are attention sinks, and the full grid retains reference crop location.

**Change:** add `ba_face_kv_mode=compact_hard_bbox`:

- Select only tokens whose hard reference bbox mask is true.
- Pack variable-length token sets and pass a real boolean/additive K/V attention mask for padding.
- For the spatial-reference run, ROI-normalize the selected rectangle to a small fixed grid before K/V projection. This removes absolute crop location while retaining local reference appearance.
- Never multiply non-face tokens by zero and leave them in softmax in the new mode.
- Keep `zero_masked_full` as the legacy default.

**Files:**

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- optionally a small token-packing helper in the same module; no new abstraction unless reused by SA and CA

### F4. Implement genuine target-face cross-attention

**Problem:** current CA (processor lines 783-826) applies generation-prompt CA to the target and face-prompt CA to the reference. N26's `ref_only` mode therefore trains the reference branch, not generated-face CA. Its stored spatial masks and `class_tokens_mask` are unused.

**Change:** add `ba_ca_mode=target_face_residual`:

```python
pm_ca = standard_cross_attention(noise_hidden, generation_prompt)
q_face = target_face_queries(noise_hidden, hard_target_mask)
k_id, v_id = compact_identity_tokens(reference_identity)
face_delta = zero_init_delta_projection(attend(q_face, k_id, v_id))
target_out = pm_ca + hard_target_mask * face_gate * face_delta
```

- Target face queries, not reference queries, receive identity conditioning.
- Use compact nonzero ID tokens; do not construct 77 tokens with roughly 75 zeros.
- Consume `class_tokens_mask` or explicit projected ID tokens and pass a real attention mask.
- Add `ba_ca_train_mode=target_face` selecting only the new target-face projections/gate.
- Preserve `legacy_ref_branch` for checkpoint reproduction.

**Files:**

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/lora2_helpers.py`, currently lines 78-92
- `src/model/photomaker_branched/branched_runtime.py`, currently lines 185-214

### F5. Preserve hard bboxes correctly at every UNet resolution

**Problem:** `_prepare_mask()` bilinearly resizes and thresholds at `>0.5`. Small bboxes can shrink or disappear at coarse layers.

**Change:** add `ba_hard_mask_resize=area_preserving`:

- Keep masks binary.
- Downsample with area coverage/max-pooling semantics, or derive integer bounds directly using `floor(start)` and `ceil(end)`.
- Require at least one token for every valid bbox at every patched resolution.
- Keep the current bilinear-threshold path as `legacy_threshold`.

**Files:**

- `src/model/photomaker_branched/attn_processor_cleanest.py`, `_prepare_mask()` around lines 541-569
- `src/model/photomaker_branched/lora2.py`, bbox helpers around lines 694-764

### F6. Make ID loss reference-supervised

**Problem:** [`id_loss.py`](../src/loss/id_loss.py) compares generated face to the ground-truth target face. It does not explicitly supervise similarity to the selected reference.

**Change:** add `id_loss_identity_source=reference`:

- Precompute the frozen recognizer embedding of the actual reference face crop.
- Compare generated target-bbox face embedding to that reference embedding.
- Retain `ground_truth_target` as legacy behavior.
- Keep the existing `id_loss_weight=0.1` for N27/N28; do not tune it yet.

**Files:**

- `src/loss/id_loss.py`
- `src/model/photomaker_branched/lora2.py`, `_compute_id_loss()` around lines 538-592
- `src/model/photomaker_branched/lora2_helpers.py`, where reference images/bboxes are already available

### F7. Stop optimizer decay on inactive staged parameters

**Problem:** `attach_inactive_branched_params()` creates zero gradient tensors on non-BA stages. With N25/N26's AdamW `weight_decay=1e-3`, these nominally inactive parameters still decay.

**Change:** expose `model._ba_active_this_batch`. Before `optimizer.step()`, set `.grad=None` for BA-only parameter groups when BA was inactive. Handle gradient accumulation by tracking whether any microbatch activated each group.

**Files:**

- `src/model/photomaker_branched/lora2.py`, schedule branch around lines 471-530
- `src/model/photomaker_branched/lora2_helpers.py`, `attach_inactive_branched_params()`
- `src/trainer/sdxl_trainers.py`, immediately before `optimizer.step()` around lines 356-364

### F8. Small correctness cleanup

- Fix tensor-reference encoding in `lora2.py::_encode_reference_latent()` lines 766-803. Image tensors must be letterboxed at image resolution before VAE encoding, not resized to latent `target_shape` before encoding.
- Stop wiring inert CA fields (`equalize_face_kv`, `id_embeds`, `class_tokens_mask`) unless the selected CA mode consumes them.
- Add assertions for finite/nonempty compact token sets and matching target/reference batches.

These are switches and correctness fixes, not reasons for a separate long run.

## Required tests before training

1. With all new flags off, N17/N25 inference remains unchanged.
2. With residual projections at zero, output equals PhotoMaker within numerical tolerance.
3. Under `hard_epsilon_merge`, maximum outside-bbox epsilon difference from PhotoMaker is zero.
4. Hard masks are nonempty and binary at every patched attention resolution.
5. Compact attention assigns no probability to padded or outside-bbox tokens.
6. N28 target-face CA parameters receive nonzero gradients; legacy `ref_to_*` parameters do not.
7. Inactive staged groups have `grad=None` at optimizer step and are not decayed.
8. Save/load round-trip reproduces gate, delta projection, and compact-token weights.

## Proposed next runs

Both runs use the common fixes F1, F2, F5-F8 and differ in the source of extra face identity.

| Decision | N27: spatial ROI residual | N28: ID-token CA residual |
|---|---|---|
| Goal | retain useful local reference appearance without full-grid geometry | inject pose-invariant identity without any reference spatial transfer |
| PhotoMaker target path | unchanged full-image SA/CA | unchanged full-image SA/CA |
| BA module | target-face SA residual | target-face CA residual |
| Reference source | compact ROI-normalized hard-bbox UNet tokens | compact projected PhotoMaker/InsightFace ID tokens |
| Full reference UNet branch | yes | no, unless needed only to compute existing ID encoder input |
| Target query | frozen PhotoMaker target-face Q | frozen PhotoMaker target-face Q |
| Trainables | reference K/V LoRA, zero-init face delta output, active scalar/per-head gate | ID-token K/V projection, zero-init face delta output, active scalar/per-head gate |
| Branched CA | disabled; ordinary PhotoMaker CA | new direct target-face residual CA |
| Branched SA | new residual mode | disabled; ordinary PhotoMaker SA |
| Main risk | ROI features can still carry reference pose/hair | global ID tokens may lack fine local appearance |

### Common settings

```yaml
pipeline:
  pose_adapt_ratio: 0
  ca_mixing_for_face: false
  use_bbox_mask_gen: true
  use_bbox_mask_ref: true
  use_dynamic_mask: false

mask_softness: 0
train_ba_all_steps: false
loss_kind: blended_masked
lambda_face: 0.15

model:
  use_id_loss: true
  id_loss_weight: 0.1
  id_loss_identity_source: reference
  ba_pm_preservation_mode: hard_epsilon_merge
  ba_hard_mask_resize: area_preserving
```

Keep the same dataset, batch size, base model, `lr_for_lora`, start steps, and 10k maximum used by N25/N26. Validate full 96 images at 1k, 3k, 5k, and 10k. Stop at 5k if known hard cases show no architectural improvement.

### N27-specific shape

```yaml
model:
  ba_sa_mode: pm_face_residual
  ba_face_kv_mode: compact_hard_bbox
  ba_ca_mode: standard
  train_branched_ca_lora: false
  ba_reference_memory: roi_spatial
```

N27 is the closest repair of the original BA idea. It tests whether the useful part is local reference appearance once absolute grid position, zero-token sinks, and destructive replacement are removed.

### N28-specific shape

```yaml
model:
  ba_sa_mode: standard
  ba_ca_mode: target_face_residual
  ba_ca_train_mode: target_face
  ba_identity_memory: projected_id_tokens
  train_branched_ca_lora: true
  disable_reference_spatial_branch: true
```

N28 is the cleaner identity-only design. It tests whether PhotoMaker needs only a learned face-local boost from pose-invariant identity tokens, without reference pose or hair entering the target representation.

N28 should use a target-only adapter forward rather than forcing a dummy reference half through `two_branch_predict()`. Its two predictions are the frozen ordinary PhotoMaker target and the trainable target-with-ID-residual path.

## Recommendation

Implement the common preservation and correctness fixes once, then run N27 and N28 in parallel. Do not combine their memories in the first experiment. The comparison should answer one architectural question clearly:

> Does identity improve more from normalized local reference appearance, or from pose-invariant identity tokens, when PhotoMaker remains the target generator and BA is restricted to a zero-initialized hard-bbox residual?

## Implemented files

- `src/model/photomaker_branched/attn_processor_cleanest.py`: zero-init PhotoMaker-preserving SA residual, compact hard-bbox ROI memory, direct ID-token CA residual, and area-preserving hard masks.
- `src/model/photomaker_branched/branched_runtime.py`: architecture switches, target-only N28 forward, and exact hard epsilon merge.
- `src/model/photomaker_branched/lora2_helpers.py`: N27/N28 input preparation, forwards, and strict trainable selection.
- `src/model/photomaker_branched/lora2.py`: opt-in configuration, reference-supervised ID loss plumbing, bbox coverage, and tensor-reference resolution fix.
- `src/loss/id_loss.py`: optional identity comparison against the actual reference face crop.
- `src/pipelines/br_pipeline_helpers.py`: matching validation/inference behavior, including target-only N28.
- `src/trainer/sdxl_trainers.py`: no AdamW decay for BA parameters on fully inactive accumulation windows.
- `infer.py`: restores architecture switches from a checkpoint's adjacent `config.yaml`; explicit CLI overrides still win.

Legacy defaults remain `ba_sa_mode=legacy`, `ba_ca_mode=legacy_ref_branch`, `ba_pm_preservation_mode=none`, `ba_hard_mask_resize=legacy_threshold`, `ba_skip_inactive_optimizer_decay=false`, and `ba_fix_tensor_ref_resolution=false`.

## Run commands

The launchers detach themselves with `nohup`, use separate distributed ports, and create timestamped logs under `logs_new_runs/`.

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker

# GPU 0; defaults to MASTER_PORT=29527
COMET_API_KEY="$COMET_API_KEY" \
  bash serv_new_runs/start_ba_spatial_roi_residual_serv_N27.sh

# GPU 1; defaults to MASTER_PORT=29528
COMET_API_KEY="$COMET_API_KEY" \
  bash serv_new_runs/start_ba_idtoken_ca_residual_serv_N28.sh
```

Override GPU assignment without editing either script:

```bash
CUDA_VISIBLE_DEVICES=2 bash serv_new_runs/start_ba_spatial_roi_residual_serv_N27.sh
CUDA_VISIBLE_DEVICES=3 bash serv_new_runs/start_ba_idtoken_ca_residual_serv_N28.sh
```

Each run is 10 epochs of 1,000 steps. Keep checkpoints at 1k, 3k, 5k, and 10k for full 96-image validation; stop after the 5k comparison if neither architecture improves the known hard prompts.
