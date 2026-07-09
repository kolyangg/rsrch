# Codex 8 Jul — Does current branched CA match the screenshot?

Request: check whether current `BranchedCrossAttnProcessor` follows the attached diagram, then rank and explain the key CA issues. No code changes made.

Relevant code:

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/model/photomaker_branched/lora2.py`

## Short Answer

Yes, the current CA processor **mostly follows the screenshot**:

```text
hidden_states = [noise_hidden, ref_hidden]
encoder_hidden_states = [gen_prompt, face_prompt]

hidden_bg  = Attn(Q(noise_hidden), K(gen_prompt),  V(gen_prompt))
hidden_ref = Attn(Q(ref_hidden),   K(face_prompt), V(face_prompt))

output = [hidden_bg, hidden_ref]
```

But there are important implementation details:

1. In our main configs (`branched_attn_weight_mode=noise_and_ref`), the arrows are not always raw `attn.to_q/k/v`. They are branch-specific cloned modules:
   - `noise_to_q/k/v`
   - `ref_to_q/k/v`
2. The CA mask is set on the processor but is **not used** to spatially merge face/background CA outputs.
3. The generation half (`noise_hidden`) does **not** directly attend to `face_prompt`.
4. The reference half (`ref_hidden`) does **not** attend to `gen_prompt`.
5. The final image uses the first half of the final UNet output. The second half is a reference stream that can affect generation through later branched self-attention, not by being directly decoded.

So if the screenshot means "two independent CA computations, one for generation prompt and one for face prompt", then yes. If the screenshot implies spatial face/background CA mixing or crossed prompt routing, then no.

## Current CA Dataflow

Current `BranchedCrossAttnProcessor.__call__()` does:

```text
noise_hidden = hidden_states[:batch_size]
ref_hidden   = hidden_states[batch_size:]

gen_prompt  = encoder_hidden_states[:batch_size]
face_prompt = encoder_hidden_states[batch_size:]
```

Then:

```text
query_bg = _q_noise(noise_hidden)
key_bg   = _k_noise(gen_prompt)
value_bg = _v_noise(gen_prompt)
hidden_bg = scaled_dot_product_attention(query_bg, key_bg, value_bg)
```

and:

```text
query_ref = _q_ref(ref_hidden)
key_ref   = _k_ref(face_prompt)
value_ref = _v_ref(face_prompt)
hidden_ref = scaled_dot_product_attention(query_ref, key_ref, value_ref)
```

Then:

```text
hidden_states = cat([hidden_bg, hidden_ref])
hidden_states = attn.to_out(hidden_states)
hidden_states = hidden_states + residual
```

This is very close to the screenshot.

## What `attn.to_*` Means In Current Configs

In `noise_and_ref` mode, `init_from_attention()` creates clones:

```text
ref_to_q, ref_to_k, ref_to_v
noise_to_q, noise_to_k, noise_to_v
```

Those clones are initialized from the effective original attention layer. With `branched_attn_new_weight_kind=lora`, each is a `BranchLoRALinear`:

```text
output = base_linear(x) + lora_delta(x)
```

This means the screenshot's `attn.to_q/k/v` labels are conceptually right, but technically in current N13/N14/N15/N16/N17/N20-style configs they are branch-specific cloned projections.

When `train_branched_ca_lora=false`:

- the `BranchedCrossAttnProcessor` can still be installed;
- its cloned CA projections still run;
- its CA clone LoRA params are not made trainable;
- saved processor state only includes trainable processor params, so frozen CA does not become a learned checkpoint component.

This is why N14/N15/N17 are better described as **branched CA runtime, frozen CA weights**, not "no CA".

## Ranked CA Issues

### 1. Highest: CA is not spatially face-masked

The self-attention processor spatially merges face/background streams:

```text
merged = hidden_bg * (1 - mask) + hidden_face * mask
```

The cross-attention processor does not do this. It stores `mask` and `mask_ref`, but the CA forward path does not use them for attention or output mixing.

Why this matters:

- Face/ID losses are spatially face-focused, but trainable CA is a global text-conditioning path.
- `attn2.processor.noise_to_*` can change full-image prompt attention, not just face attention.
- This can explain face identity pressure leaking into pose, props, hands, hats, goggles, and body layout.

This is the most likely code-level reason CA training hurts.

### 2. Very High: Generation CA never directly sees the face prompt

The generation half computes:

```text
hidden_bg = Attn(Q(noise_hidden), K(gen_prompt), V(gen_prompt))
```

It does not compute:

```text
Attn(Q(noise_hidden), K(face_prompt), V(face_prompt))
```

The face prompt updates the reference half:

```text
hidden_ref = Attn(Q(ref_hidden), K(face_prompt), V(face_prompt))
```

That second half can influence generation later through branched self-attention, because self-attention uses the reference hidden stream as the face source. But CA itself is not directly injecting face prompt into the generation stream.

Why this matters:

- The name "branched cross-attention" sounds like the generated face region gets special face-prompt CA. Current code does not do that directly.
- Training CA may optimize an indirect pathway: face prompt changes `ref_hidden`, then later self-attention transfers from `ref_hidden` into the generated face.
- This indirect path is harder to control and can create delayed, layer-dependent behavior.

### 3. Very High: Trainable CA includes the whole generation prompt path

In `noise_and_ref`, if CA training is enabled, these are trainable:

```text
attn2.processor.noise_to_q/k/v
attn2.processor.ref_to_q/k/v
```

The `noise_to_*` CA path is used for:

```text
Q(noise_hidden), K(gen_prompt), V(gen_prompt)
```

That is the full generation text-conditioning path, not a face-only path.

Why this matters:

- It can change prompt following and global image structure.
- `ba_noise_lr_scale=0.1` damps it, but does not make it face-local.
- The N16 result is consistent with this: same ID loss and blended loss as N14, but trainable CA drops full-val mean from `0.3324` to `0.2811`.

### 4. High: CA roughly doubles trainable processor capacity

Saved logs show:

| Setup | Trainable processor tensors | Trainable processor params |
|---|---:|---:|
| CA frozen / SA-only style | 840 | ~31.95M |
| CA trainable | 1680 | ~71.60M |

Why this matters:

- `train_branched_ca_lora=true` is not a small tweak.
- It adds a large trainable cross-modal capacity.
- With a small personalized dataset and strong face/ID losses, that capacity can overfit or drift.

### 5. High: No separate CA optimizer group

Current optimizer grouping only separates:

```text
lora_params       = everything except .processor.noise_to_*
ba_noise_params   = .processor.noise_to_* with scaled LR
```

So:

- `sa_ref` and `ca_ref` share full `lr_for_lora`;
- `sa_noise` and `ca_noise` share `lr_for_lora * ba_noise_lr_scale`;
- there is no separate `ca_ref_lr_scale`;
- there is no separate `ca_noise_lr_scale` independent from SA noise.

Why this matters:

- If the harmful part is `ca_ref`, current `ba_noise_lr_scale` does not damp it.
- N19-style "CA train with lower noise scale" may still leave CA ref too strong.
- A cleaner CA probe would split optimizer groups into `sa_ref`, `sa_noise`, `ca_ref`, `ca_noise`.

### 6. Medium/High: CA training sees all training timesteps under `train_ba_all_steps=true`

Recent configs set:

```text
train_ba_all_steps=true
```

In `PhotomakerBranchedLora.forward()`, that bypasses the schedule and always runs the branched forward pass.

Inference/full validation uses:

```text
photomaker_start_step=10
branched_attn_start_step=15
num_inference_steps=50
```

Why this matters:

- Trainable CA gets gradients from very noisy timesteps where the inference BA behavior would not yet be active.
- This schedule mismatch also applies to SA, but CA is the text-conditioning path and may be more sensitive.
- If CA is trained again, matching the inference schedule is worth testing.

### 7. Medium: `face_prompt` is sparse/id-only and may be brittle for trainable CA

Current full-val/main configs use:

```text
face_embed_strategy=id
ba_face_prompt_mode=id_only
```

In `two_branch_predict()`, the face prompt is created by zeroing all non-ID tokens and boosting ID tokens. The code comment already notes the side effect:

```text
~75/77 zero K/V tokens act as attention sinks in the ref branch cross-attention.
```

Why this matters:

- Frozen CA preserves the pretrained text-attention mapping.
- Trainable CA can adapt to this sparse prompt in a way that improves ID but damages general prompt structure.
- This is another reason CA trainability may degrade props and scene composition.

### 8. Medium: CA ignores `attention_mask`

`BranchedCrossAttnProcessor.__call__()` accepts `attention_mask`, but its `scaled_dot_product_attention()` calls do not pass an `attn_mask`.

Why this matters:

- In many SDXL runs the text attention mask may be `None`, so this may not be active.
- But if padding/text masks are provided, current branched CA ignores them.
- This is probably not the main N14/N16 gap, but it is a correctness risk.

### 9. Medium: CA stabilization attributes are set but unused

`patch_unet_attention_processors()` sets:

```text
equalize_face_kv = True
equalize_clip = (1/3, 8.0)
```

The current `BranchedCrossAttnProcessor` does not read those attributes.

Why this matters:

- There are apparent stabilizers in the patching code that do not affect current CA.
- If we thought those were protecting the face branch, they are not.

### 10. Medium/Low: `ba_train_top_k` is not a clean partial-CA control

`ba_patch_top_k` only controls which self-attention processors are patched. Cross-attention processors are patched everywhere unless `disable_branched_ca=true`.

`ba_train_top_k` selects a prefix of processor names after filtering. That is not necessarily a balanced blockwise subset.

Why this matters:

- A top-k CA run may be hard to interpret.
- Better CA ablations should explicitly choose CA ref/noise and block ranges.

## How This Relates To The Results

The cleanest evidence is still N14 vs N16:

| Run | CA trainable? | ID loss | Steps | Mean id-sim |
|---|---:|---:|---:|---:|
| N14 | no | yes, `0.1` | 6k | `0.3324` |
| N16 | yes | yes, `0.1` | 6k | `0.2811` |

Because the rest of the recipe is essentially the same, this strongly suggests CA trainability is harmful in the current design.

The code explains why:

- CA trainability is broad/global.
- CA is not face-spatially masked.
- CA ref trains at full LR.
- CA uses sparse id-only face prompt.
- CA training doubles capacity.

## Recommendation

For the next main run, keep **N20 frozen CA**.

If we want to revisit CA, do it as a short diagnostic probe, not a long run:

1. Add separate optimizer groups for `sa_ref`, `sa_noise`, `ca_ref`, `ca_noise`.
2. Try CA LR much lower than SA, especially for `ca_ref`.
3. Test CA-ref-only and CA-noise-only separately.
4. Consider using `disable_branched_ca=true` at inference on existing checkpoints to isolate whether CA runtime itself helps or hurts.
5. If training CA, consider matching inference schedule instead of `train_ba_all_steps=true`.
6. Do not expect CA training alone to fix face placement; the current self-attention path still hardcodes `POSE_ADAPT_RATIO=0.0`.

Current conclusion: the screenshot is a fair high-level diagram of the current CA path, but that path is broad and not face-localized. That makes CA training a plausible source of worse performance rather than an obvious next lever.
