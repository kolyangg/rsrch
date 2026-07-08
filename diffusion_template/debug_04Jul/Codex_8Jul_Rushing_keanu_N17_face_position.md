# Codex 8 Jul — Rushing/Keanu N17 face-position issue

Target image:

```text
Rushing ma_keanu.png
```

Prompt:

```text
Rushing <class> portrait, subway platform, anxious face, swinging briefcase
```

Focused visual artifact:

```text
debug_04Jul/Codex_8Jul_Rushing_keanu_N17_bbox_comparison.png
```

This sheet shows the same generated image across:

1. N17 long 26k
2. N14 combo 6k
3. N13 ID-loss 3k, CA trained
4. N15 SA-only 6k
5. N16 ID-loss 6k, CA trained
6. N11 SA-only 3k
7. N6 blended base

Red box = `face_crop_new`; blue box = old bbox. Bottom row is the crop from the exact red box.

## Key observation

The face bbox/mask is **not different for N17**. The full-val config uses the fixed pm96 bbox map:

```json
"Rushing ma_keanu.png": {
  "face_crop_old": [535, 164, 776, 552],
  "face_crop_new": [552, 191, 759, 526],
  "body_crop": [67, 97, 862, 1023],
  "force_manual": false
}
```

So this is not a per-run detector drift. N17, N14, N15, N11, etc. all receive the same face-region
mask during inference.

The difference is what each checkpoint paints **inside the same mask**.

## What the comparison shows

Metrics for this one image:

| run | config shorthand | id-sim |
|---|---|---:|
| N17 | frozen-CA + ID loss, 26k | 0.424 |
| N14 | frozen-CA + ID loss, 6k | 0.448 |
| N13 | ID loss + CA trained, 3k | 0.498 |
| N15 | SA-only, 6k | **0.522** |
| N16 | ID loss + CA trained, 6k | 0.381 |
| N11 | SA-only, 3k | 0.494 |
| N6 | blended baseline | 0.117 |

Visual read:

- N17 puts a more reference-like Keanu head/neck crop into the mask, but the head sits high/right and
  leaves an unnatural long-neck / pasted-face impression.
- N14 is similar but less severe.
- N16, another longer ID-loss run, shows a related head/neck placement problem.
- N15 and N11, the SA-only runs without ID loss, place a more pose-compatible face in the correct
  body location.
- N13 is also better than N17 here, likely because the shorter run and trainable CA still leave more
  body/prompt coupling than the long frozen-CA ID-loss checkpoint.

This image is a good example of why N17 can win aggregate id-sim while still being visually worse on
specific pose/alignment cases.

## Likely mechanism

The self-attn face branch is currently hard-coded to use pure reference-face hidden states:

```text
src/model/photomaker_branched/attn_processor_cleanest.py:308
POSE_ADAPT_RATIO = 0.0

src/model/photomaker_branched/attn_processor_cleanest.py:341
face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden
```

With `POSE_ADAPT_RATIO=0.0`, this simplifies to:

```text
face_hidden_mixed = ref_face_hidden
```

Then the face output is merged into the generated image only inside the fixed mask:

```text
src/model/photomaker_branched/attn_processor_cleanest.py:390
merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
```

N17 config confirms the relevant setup:

```text
trainer.n_epochs=13
trainer.epoch_len=2000
model.use_id_loss=true
model.id_loss_weight=0.1
pipeline.pose_adapt_ratio=0
pipeline.ca_mixing_for_face=false
loss_kind=blended_masked
lambda_face=0.15
branched_attn_weight_mode=noise_and_ref
train_branched_ca_lora=false
ba_noise_lr_scale=0.1
```

Interpretation:

1. The fixed mask says “replace this upper-right face rectangle.”
2. The face branch is trained to inject a pure reference identity signal, not a pose-adapted
   generated-face signal.
3. N17 adds many more steps of ID loss on top of frozen-CA SA-only training.
4. Over time, the branch appears to learn a stronger canonical Keanu head/neck crop inside that
   rectangle.
5. Because cross-attn is frozen, there is no CA pathway adapting the face content back to the
   prompt/body geometry.
6. The result is a face that is identity-recognizable but spatially/body-inconsistent.

That matches the visual: the face is not outside the red mask; it is badly composed inside it.

## Why N15/N11 are better here

N15 and N11 are SA-only without ID loss. They have less direct pressure to maximize identity inside
the face crop, so the face patch remains more coupled to the generated body/pose. That gives a lower
identity ceiling in some cases, but here it keeps the face in a natural position.

This also explains why the issue can get worse with more training: the objective is rewarding
identity in the crop, not explicitly rewarding head/neck geometry or alignment with the suit/body.

## What this implies for next experiments

This specific failure argues against “just train the same frozen-CA + ID-loss recipe longer.”

Best targeted tests:

1. **Full-val intermediate N17 checkpoints** for this image and the rest of Keanu:
   - 8000, 10000, 12000, 14000, 16000, 18000 if available.
   - Check whether the long-neck/high-right crop appears gradually.
   - If yes, choose an earlier checkpoint or lower ID-loss pressure.

2. **Lower ID-loss pressure rerun**:
   - same N17 recipe;
   - `id_loss_weight=0.05` or `0.075`;
   - keep frozen CA;
   - save every 1k.

3. **Pose-adaptation ablation**:
   - un-hardcode `POSE_ADAPT_RATIO`;
   - test small values like `0.1` or `0.2`;
   - expected tradeoff: less pasted/reference-canonical face, possibly slightly lower identity.

4. **Mask tuning for this prompt**:
   - current bbox is broad and includes neck/hair space: `[552,191,759,526]`;
   - a tighter landmark/segmentation mask would reduce the area where the reference-like head/neck
     crop can overwrite the body.

N19-style trainable CA might help this exact pose-alignment issue, but given N16 instability it
should be a short probe, not the next default long run.

## Limitation of this analysis

I could not rerun InsightFace/face detection locally in this shell because the active Python
environment does not have `torch`. The analysis above is based on:

- the actual full-val images;
- the fixed pm96 bbox JSON;
- the existing id-sim metrics;
- code/config inspection;
- the bbox-overlay comparison sheet.

The next stronger diagnostic is to run the intermediate-checkpoint full-val script on the remote
server and compare `Rushing ma_keanu.png` at each checkpoint.
