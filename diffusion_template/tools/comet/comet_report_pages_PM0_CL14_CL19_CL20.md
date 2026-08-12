<!-- report-page: CL14 architecture -->
<!-- layout: architecture -->
<!-- column: left -->
# CL14 architecture: the reference design

## CL14 - actual hard spatial branched self-attention

The spatial self-attention input is a doubled batch: target/noisy features X and reference features R. M_t and M_r are the target-face and reference-face masks. CL14 uses separate trainable low-rank Q/K/V deltas for the target and reference streams (noise_and_ref mode, BA rank 128), while the frozen SDXL/PhotoMaker path remains the initialization anchor.

$$\mathrm{SDPA}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d})V$$

The background lane keeps target information. The identity lane keeps target queries, but reads masked reference-face keys and values. Because pose_adapt_ratio = 0, no target feature replaces reference K/V.

$$A_{bg}=\mathrm{SDPA}((1-M_t)Q_t(X),K_t(X),V_t(X))$$

$$A_{id}=\mathrm{SDPA}(M_tQ_t(X),K_r(M_rR),V_r(M_rR))$$

$$Y_t=O((1-M_t)A_{bg}+M_tA_{id})+X$$

The reference half is propagated independently so later U-Net blocks retain reference state: Y_r = O(SDPA(Q_r(R), K_r(R), V_r(R))) + R. Reference features outside M_r are zeroed, but true key masking is off, so zero-key softmax sinks remain part of the historical CL14 contract.

<!-- column: right -->
## Timing and optimization contract

- PhotoMaker token conditioning starts at denoising step 10; BA starts at step 15 of DDIM50.
- Spatial self-attention BA is enabled; branched cross-attention is disabled.
- Training owns 2,240 tensors / 219,217,920 parameters: BA plus generic and PhotoMaker-default LoRA lanes.

## CL14 boundary-mask behavior

CL14 writes two inward training rings with weights 1/3 and 2/3. The installed processors then bilinearly resize and threshold at > 0.5. Therefore the outer 1/3 ring is removed and the 2/3 ring remains: the actual route is a hard one-cell erosion, not a continuous boundary blend. Validation uses the unchanged binary bbox mask.

<!-- report-page: CL19 and CL20 differences -->
<!-- layout: architecture -->
<!-- column: left -->
# CL19 and CL20: differences from CL14

## CL19 - true soft full-query router vs CL14

CL19 keeps CL14's target-query/reference-KV identity path and zero sinks, but stops masking Q before attention. It computes two complete target messages, then blends exactly once with a two-cell cosine router S.

$$A_n=\mathrm{SDPA}(Q_t(X),K_t(X),V_t(X))$$

$$A_r=\mathrm{SDPA}(Q_t(X),K_r(M_rR),V_r(M_rR))$$

$$Y_t=O((1-S)A_n+SA_r)+X$$

For ring j of c = 2 transition cells, S_j = 0.5[1 - cos(pi j/(c+1))]; the inner face remains S = 1 and background S = 0. This removes CL14's query masking and its second output mask, making the handover genuinely continuous across all down, mid, and up spatial-attention groups.

<!-- column: right -->
## CL20 - data curriculum vs CL14

- Architecture, optimizer, loss, trainable parameter count, and hard spatial BA are CL14-exact; hardcase mode is off.
- Only training sampling changes: 80% Cosmic / 20% curated BigCelebs through 20k, followed by 4k Cosmic-only re-anchoring.
- BigCelebs rows require identity depth >= 6 and balance synthetic-small-face, occlusion-caption, and action-caption strata.

## Shared validation invariants

- RealVisXL V4.0 validation base, DDIM50, CFG 5, batch 12, seed 0.
- Same 96 prompts/reference pairs, bboxes, PhotoMaker V2 checkpoint, and subject-v2 identity metric.
- pose_adapt_ratio = 0 and ca_mixing_for_face = false.

<!-- report-page: Critical implementation excerpts -->
<!-- layout: code -->
<!-- column: left -->
# CL14 critical implementation excerpts

## CL14: hard target/reference attention lanes

The processor keeps target queries in target coordinates. It gates target queries by the target mask, obtains identity K/V only from the masked reference face, then merges background and identity messages spatially. Excerpt from `attn_processor_cleanest.py`:

```python
# Background: target Q/K/V.
q_bg = q * (1.0 - mask_gate)
hidden_bg = F.scaled_dot_product_attention(
    q_bg, key_bg, value_bg,
    dropout_p=0.0, is_causal=False,
)

# Identity: target Q, reference-face K/V.
ref_face_hidden = ref_hidden * ref_mask_flat
key_face = self._k_ref(attn, ref_face_hidden)
value_face = self._v_ref(attn, ref_face_hidden)
q_face = q * mask_gate
hidden_face = F.scaled_dot_product_attention(
    q_face, key_face, value_face,
    dropout_p=0.0, is_causal=False,
)

merged = (
    hidden_bg * (1.0 - mask_flat)
    + hidden_face * mask_flat * self.scale
)
```

<!-- column: right -->
## CL14: why the written feather is still hard

The training mask writes `1/3` and `2/3` boundary rings, but every spatial-attention resolution uses this preparation path:

```python
m2d = F.interpolate(
    m4, size=(H, W), mode="bilinear",
    align_corners=False,
)
if self.force_binary_masks:
    m2d = (m2d > 0.5).to(m2d.dtype)
```

The threshold removes the `1/3` ring. This code-level fact is why CL14 is described as hard routing with one-cell erosion.

<!-- report-page: CL19 and CL20 implementation excerpts -->
<!-- layout: code -->
<!-- column: left -->
# CL19 and CL20 implementation excerpts

## CL19: full messages, one cosine blend

Unlike CL14, `_full_target_lanes` evaluates every target query in both lanes. `_soft_router_mask` constructs two inward transition rings, and the output is blended once:

```python
native_out, reference_out, _ = (
    self._full_target_lanes(attn, target, reference)
)
router = self._soft_router_mask(
    self.mask, target.shape[1],
    target.shape[0], native_out.dtype,
)
target_out = (
    native_out * (1.0 - router)
    + reference_out * router
)
```

```python
for index in range(cells):
    eroded = 1.0 - F.max_pool2d(
        1.0 - remaining, 3, stride=1, padding=1
    )
    ring = (remaining - eroded).clamp(0.0, 1.0)
    phase = float(index + 1) / float(cells + 1)
    weight = 0.5 - 0.5 * math.cos(math.pi * phase)
    result = result * (1.0 - ring) + ring * weight
    remaining = eroded
```

For two cells the ring weights are `0.25` and `0.75`; the interior is `1` and background `0`. Query support is full in both lanes, so no zeroed-query softmax message is later masked a second time.

<!-- column: right -->
## CL20: no attention-code change

CL20 selects the CL14 model path and changes only deterministic data order:

```yaml
train_dataset_name: cl20_hardcase_curriculum
train_dataloader_shuffle: false

model:
  ba_hardcase_mode: off
```

The sealed 48k-row schedule contains 32k Cosmic + 8k curated BigCelebs rows for the first 20k optimizer steps, then 8k Cosmic rows for the final 4k re-anchor.

<!-- report-page: Fixed validation references and prompts -->
<!-- layout: references_prompts -->
# Fixed 96-image validation contract

## References

![Eddie](../../../dataset_full/val_dataset/references/eddie.webp)
![Elon](../../../dataset_full/val_dataset/references/elon.jpg)
![Jennie](../../../dataset_full/val_dataset/references/jennie.webp)
![Jensen](../../../dataset_full/val_dataset/references/jensen.png)
![Jisoo](../../../dataset_full/val_dataset/references/jisoo.webp)
![Keanu](../../../dataset_full/val_dataset/references/keanu.jpg)
![Lex](../../../dataset_full/val_dataset/references/lex.jpeg)
![Marion](../../../dataset_full/val_dataset/references/marion.jpg)

## Prompt templates

1. Reading paper <class>, park bench, calm face, grey overcoat
2. Rushing <class> portrait, subway platform, anxious face, swinging briefcase
3. Skiing <class>, snowy slope, fearless grin, orange goggles
4. Drumming <class>, rock concert in rain, passionate angry face, black t-shirt
5. Kickboxing <class>, gym ring, fierce roar face, sweatband
6. Dancing <class>, neon club, euphoric face, silver jumpsuit
7. Angry <class>, traffic jam, scowling face, dark suit
8. Crying <class>, stormy alley, tearful face, ripped jeans
9. Laughing <class>, carnival wheel, carefree laugh, polka-dot shirt
10. Jumping <class>, beach sunset, ecstatic face, blue tank top
11. Night-ride biker <class>, neon city, excited face, reflective vest
12. Chef <class>, bustling kitchen, proud grin, tall white toque
