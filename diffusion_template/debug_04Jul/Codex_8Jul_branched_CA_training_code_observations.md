# Codex 8 Jul — Branched CA training code observations

Request: investigate whether the current branched-attention cross-attention (CA / `attn2`) training path has issues that could explain why freezing CA gives better results. No code changes made.

Files inspected:

- `src/model/photomaker_branched/lora2.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/trainer/sdxl_trainers.py`
- `src/trainer/base_trainer.py`
- `src/configs/one_id_09Feb_testing.yaml`
- `src/configs/inference/full_val.yaml`
- saved configs/logs for N13/N14/N15/N16/N17

## Bottom line

I do **not** see a simple wiring bug where `train_branched_ca_lora=false` accidentally disables branched CA at inference/training runtime. The flag does what the experiments imply: it freezes CA processor parameters while branched CA can still run with cloned/base weights.

But I do see several code-structure reasons why **training** branched CA can plausibly hurt:

1. `BranchedCrossAttnProcessor` does not use the face mask to spatially merge face/background CA outputs.
2. With `noise_and_ref`, trainable `attn2.processor.noise_to_*` changes the whole generation prompt-attention path, not just the face.
3. Trainable CA roughly doubles processor-trainable capacity in these runs.
4. There is no separate CA-specific LR/WD group; `ca_ref` is at full LR.
5. CA training is active at all sampled training timesteps when `train_ba_all_steps=true`, while inference only uses BA after `branched_attn_start_step`.
6. The current self-attention processor hardcodes pose adaptation off, so enabling CA is unlikely to fix the face-placement issue directly.

So my read is: **freezing CA is not just a lucky minor regularizer; it avoids a currently broad, weakly localized trainable pathway.**

## Experiment evidence from saved runs

All relevant recent configs use:

- `branched_attn_weight_mode=noise_and_ref`
- `branched_attn_new_weight_kind=lora`
- `loss_kind=blended_masked`
- `lambda_face=0.15`
- `ba_noise_lr_scale=0.1`
- `train_ba_only=true`
- `train_ba_all_steps=true`

Important comparisons:

| Run | CA trainable? | ID loss? | Steps | 96-val mean id-sim | Read |
|---|---:|---:|---:|---:|---|
| N13 `ba_idloss_N13` | yes | yes, `0.1` | 3k | 0.3150 | ID loss helps early, but CA is trainable. |
| N16 `ba_idloss6k_N16` | yes | yes, `0.1` | 6k | 0.2811 | Same direction as N13 but longer; degrades. |
| N15 `ba_saonly6k_N15` | no | no | 6k | 0.3115 | Freezing CA without ID loss is stable. |
| N14 `ba_combo_N14` | no | yes, `0.1` | 6k | 0.3324 | Best short run; same ID loss but CA frozen. |
| N17 `ba_longrun_N17` | no | yes, `0.1` | 26k | 0.3482 | Best aggregate, with late over-strength issues. |

Most direct comparison: **N14 vs N16**. Both have ID loss `0.1`, blended loss, same BA mode, same LR/WD knobs, same 6k length. N14 freezes CA and gets `0.3324`; N16 trains CA and gets `0.2811`. That strongly suggests the current CA train path is harmful under this recipe.

## Observation 1 — `train_branched_ca_lora=false` freezes CA, it does not disable branched CA runtime

In `configure_branched_trainables()`:

- `train_ca = model.train_branched_ca_lora`
- selected trainable processors include cross-attention only when `include_cross_attention=train_ca`
- when `train_ca=false`, `attn2.processor.*` parameters are not re-enabled after the global freeze

But `patch_unet_attention_processors()` still installs `BranchedCrossAttnProcessor` for `attn2` unless `disable_branched_ca=true`.

Implication:

- N14/N15/N17 are not "no CA" runs.
- They are "branched CA runtime with frozen cloned/base CA weights" runs.
- That is exactly the useful distinction: CA behavior remains present, but its trainable drift is removed.

This is a good sign: the ablation is meaningful.

## Observation 2 — Current branched CA is not spatially face-masked

`BranchedAttnProcessor` for self-attention uses `mask_gate` to build face/background behavior and merge:

```text
merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
```

`BranchedCrossAttnProcessor` does not do an equivalent spatial face/background merge. In its forward path:

- first half / generation branch:
  - Q from `noise_hidden`
  - K/V from `gen_prompt`
  - output is `hidden_bg`
- second half / reference branch:
  - Q from `ref_hidden`
  - K/V from `face_prompt`
  - output is `hidden_ref`
- final output:

```text
hidden_states = torch.cat([hidden_bg, hidden_ref], dim=0)
```

The CA mask is set on the processor, but it is not used in the CA forward path. So when CA weights are trainable, the training signal is not spatially constrained to the face box inside CA itself.

Implication:

- `attn2.processor.noise_to_*` can change the whole generation text-conditioning path.
- `attn2.processor.ref_to_*` can change the whole reference/face-prompt path.
- A face-weighted loss or ID loss can push these global text-attention transforms in ways that improve ID but harm pose/body/props/background.

This is probably the most important code-level reason CA training is risky.

## Observation 3 — In `noise_and_ref`, trainable CA includes a global generation path

With `branched_attn_weight_mode=noise_and_ref`, `BranchedCrossAttnProcessor.init_from_attention()` creates both:

- `ref_to_q/k/v`
- `noise_to_q/k/v`

When `train_branched_ca_lora=true`, `configure_branched_trainables()` enables both ref and noise CA LoRA weights:

- `.attn2.processor.ref_to_*`
- `.attn2.processor.noise_to_*`

In CA forward, `noise_to_*` is used for the generation branch:

```text
query_bg = self._q_noise(attn, noise_hidden)
key_bg   = self._k_noise(attn, gen_prompt)
value_bg = self._v_noise(attn, gen_prompt)
```

That is not a face-only branch. It is the generation prompt-attention branch for the full latent stream.

Implication:

- Even with `ba_noise_lr_scale=0.1`, the trainable CA noise path can modify global text following.
- This fits the visual failures where the face/identity force bleeds into pose, prop, hat, glove, or body placement rather than staying inside a clean face replacement.

## Observation 4 — CA training roughly doubles trainable processor capacity

From saved logs:

- N13 / CA trainable: `1680` tensors, `71.60M` trainable processor params.
- N11 / CA frozen: `840` tensors, `31.95M` trainable processor params.
- N14/N15/N17 info logs show `420 + 420` parameter tensors across `lora_params` and `ba_noise_params`.
- N13/N16 show `840 + 840`.

So enabling CA is not a small switch. It adds a large amount of trainable cross-modal capacity.

Implication:

- Degradation from CA training can be caused by over-capacity / under-regularization, not necessarily a single implementation bug.
- The current N19 idea of lowering CA-related pressure is directionally right, but a full long CA-training run is still risky.

## Observation 5 — No CA-specific optimizer group

`get_trainable_params()` only splits parameters into:

- `lora_params`: everything except `.processor.noise_to_*`
- `ba_noise_params`: `.processor.noise_to_*`, scaled by `ba_noise_lr_scale`

That means:

- `sa_ref` and `ca_ref` share the full `lr_for_lora`
- `sa_noise` and `ca_noise` share the scaled LR
- there is no separate `ca_ref` or `ca_noise` LR/WD control

In the current recipe, `ca_ref` trains at full `1e-4`, while only `ca_noise` is damped by `ba_noise_lr_scale=0.1`.

Implication:

- If CA ref drift is the damaging part, the current optimizer cannot damp it independently.
- A future CA probe should probably use separate CA groups or avoid `ca_ref` training first.

## Observation 6 — `train_ba_all_steps=true` trains CA outside the inference BA schedule

In `PhotomakerBranchedLora.forward()`:

- if `train_ba_all_steps=true`, it always calls `run_branched_forward_pass()`
- the normal schedule branches by `photomaker_start_ratio` and `branched_start_ratio` are bypassed

All recent N13/N14/N15/N16/N17 scripts set:

```text
train_ba_all_steps=true
```

Inference/full validation uses:

```text
photomaker_start_step=10
branched_attn_start_step=15
num_inference_steps=50
```

Implication:

- Trainable CA sees every sampled diffusion timestep in the branched path, including very noisy regimes.
- In inference, BA CA only starts after step 15/50.
- SA-only training also has this mismatch, but CA is the cross-modal text pathway and may be more sensitive to it.

This is another reason CA trainability can hurt even if the runtime CA structure is useful.

## Observation 7 — The face prompt path is sparse/id-only, which may be brittle for trainable CA

With `face_embed_strategy=id` and `ba_face_prompt_mode=id_only`, `two_branch_predict()` masks the face prompt to ID-token positions and zeros the other tokens, then normalizes token std.

Comments in code already note a side effect:

```text
~75/77 zero K/V tokens act as attention sinks in the ref branch cross-attention.
```

Frozen CA inherits a stable pretrained text-attention mapping. Trainable CA can adapt to this sparse prompt distribution.

Implication:

- CA training may overfit to the id-only prompt distribution and degrade general text alignment.
- This is especially relevant because the CA processor output is not face-spatially merged.

## Observation 8 — Some CA-related knobs are currently inactive or dead in the current processor

In `attn_processor_cleanest.py`, self-attention hardcodes:

```text
POSE_ADAPT_RATIO = 0.0
CA_MIXING_FOR_FACE = False
```

The runtime kwargs that used to pass pose/CA mixing knobs are commented out in `branched_runtime.py`.

Also, `patch_unet_attention_processors()` sets CA attributes:

```text
equalize_face_kv = True
equalize_clip = (1/3, 8.0)
```

but the current `BranchedCrossAttnProcessor` does not read those attributes.

Implication:

- Enabling trainable CA is unlikely to solve the current Rushing/Keanu face-placement issue by itself, because the main face self-attention path still uses pure reference face hidden (`POSE_ADAPT_RATIO=0.0`).
- Some apparent CA-stabilization hooks are not active in the current code path.

## Observation 9 — `ba_train_top_k` / `ba_patch_top_k` are not great tools for partial CA tests

`ba_patch_top_k` only affects self-attention patching in `patch_unet_attention_processors()`. Cross-attention processors are patched everywhere unless `disable_branched_ca=true`.

`ba_train_top_k` selects a prefix of candidate processor names. If CA training is enabled and `top_k<1`, the subset is based on current processor order, not a balanced per-block/per-attention-type choice.

Implication:

- A "partial CA training" test using only top-k may be hard to interpret.
- Better to create explicit controls such as CA-ref-only, CA-noise-only, CA-down/up/mid selection, or separate CA LR groups.

## Risk ranking

| Issue | Confidence | Why it matters |
|---|---:|---|
| CA trainability is global/not spatially face-masked | high | Directly visible in `BranchedCrossAttnProcessor`; explains prop/body/pose bleed. |
| CA adds large capacity | high | Logs show ~31.95M → ~71.60M trainable processor params. |
| `ca_ref` has full LR with no separate group | high | Current optimizer grouping only separates `noise_to_*`. |
| `train_ba_all_steps` schedule mismatch | medium | True in configs; likely worse for CA than SA, but not isolated. |
| id-only sparse face prompt is brittle for trainable CA | medium | Code comments and prompt masking support it, but needs ablation. |
| inactive pose/CA-mixing knobs | high | Directly visible; this affects what CA training can realistically fix. |
| outright CA-training bug | low/medium | I did not find a single obvious broken flag; the likely problem is design/optimization. |

## Practical recommendation

I would keep N20 as the main run: frozen CA + lower ID loss.

For CA, I would **not** do a long full-CA run next. If we want to test the suspicion, do a short, explicitly diagnostic probe:

1. `train_branched_ca_lora=true`
2. ID loss lower or off, not `0.1`
3. 3k-6k only first
4. log/compare `train/ba_norm/ca_ref` and `train/ba_norm/ca_noise`
5. inspect same hard cases: `Rushing ma_keanu`, `Kickboxing_marion`, `Chef woman_jisoo`, `Skiing wom_jennie`

Most informative future code experiments, in order:

1. Add separate optimizer groups for `sa_ref`, `sa_noise`, `ca_ref`, `ca_noise`; try CA LR much lower than SA.
2. Test CA-noise frozen but CA-ref trainable, and vice versa, instead of both.
3. Test `disable_branched_ca=true` at inference on N17/N20 checkpoints to separate "branched CA runtime helpful/harmful" from "CA training harmful".
4. Re-enable a small pose-adapt path (`POSE_ADAPT_RATIO=0.1` or `0.2`) before expecting CA training to fix face placement.
5. If retraining CA, consider matching the inference schedule instead of `train_ba_all_steps=true`.

Current conclusion: **the evidence and code both support freezing CA for the next main run. CA training is worth probing, but it should be treated as a risky pathway that needs tighter controls, not as the obvious next improvement.**
