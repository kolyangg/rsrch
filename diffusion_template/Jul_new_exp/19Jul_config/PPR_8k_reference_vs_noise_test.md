# PPR 8k reference-content versus reference-noise test

## Goal

Using the existing 8k checkpoint, determine whether the face change at PPR `scale=4` is driven by:

1. spatial-reference image content;
2. noise in the noised reference stream;
3. a generic or target/PhotoMaker-conditioned PPR correction.

This is an inference-only diagnostic. Do not update weights or save a modified checkpoint.

## Fixed settings

Keep these identical in every paired run:

- the same 8k checkpoint, loaded strictly once;
- RealVis validation base;
- target/PhotoMaker identity image and embeddings;
- target latent seed;
- prompt, negative prompt and class-token positions;
- target face bbox and core mask;
- resolution, CFG, scheduler and 50 denoising steps;
- PhotoMaker and BA start steps;
- `ba_output_anchor_mode=base_outside_core`;
- `ba_ppr_runtime_scale=4`, except for the baseline control;
- active PPR processor objects and weights.

Pose adaptation and CA mixing remain off. The ordinary target PhotoMaker identity must not change when the spatial reference is swapped.

## Required variants

For every validation sample, use its normal spatial reference `R1`, a different-identity spatial reference `R2`, and independently controlled reference-noise seeds `N1` and `N2`.

| Variant | PPR scale | Spatial reference | Reference-noise seed | Purpose |
|---|---:|---|---|---|
| `PM0` | 0 | R1 | N1 | Exact ordinary PhotoMaker control |
| `R1N1` | 4 | R1 | N1 | Normal PPR condition |
| `R2N1` | 4 | R2 | N1 | Change reference content only |
| `R1N2` | 4 | R1 | N2 | Change reference noise only |
| `R2N2` | 4 | R2 | N2 | Replicate both effects |

Two reference-noise seeds are the minimum. Four independently chosen reference-noise seeds are recommended for the final 96-image run.

Use a deterministic cyclic identity permutation for `R2`, or select a clearly different identity. Swap both the reference image and its corresponding bbox. Record the original and swapped identity explicitly.

## Independent reference-noise control

Add a diagnostic-only argument such as:

```python
ppr_reference_noise_seed: int | None = None
```

When set, construct `_ref_noise` from a separate generator:

```python
ref_generator = torch.Generator(device=device)
ref_generator.manual_seed(ppr_reference_noise_seed)
pipeline._ref_noise = torch.randn(
    pipeline._ref_latents_all.shape,
    generator=ref_generator,
    device=device,
    dtype=pipeline._ref_latents_all.dtype,
)
```

Do not advance or replace the generator used for target latents. Clear generation-local caches before each pipeline call so the requested reference-noise seed is not silently reused from a previous variant.

The only difference between `R1N1` and `R2N1` must be reference image content and its bbox. The only difference between `R1N1` and `R1N2` must be `_ref_noise`.

## Integrity assertions

Save per-sample SHA-256 values for:

```text
target_initial_latents
target_prompt_embeds
target_photomaker_id_embeds
spatial_reference_image
reference_latents
reference_mask
reference_noise
ref_noised at steps 15, 25 and 35
```

Assert:

- target latents and all target conditioning are identical across all five variants;
- `R1N1` and `R2N1` have identical reference noise but different reference-image and reference-latent hashes;
- `R1N1` and `R1N2` have identical reference-image, reference-latent and mask hashes but different reference-noise hashes;
- all reference masks are nonempty;
- every tested PPR site has a positive packed reference ROI token count;
- the `PM0` image is the exact ordinary PhotoMaker control;
- all scale-4 variants report nonzero applied PPR residuals.

The existing randomness fingerprint is insufficient by itself because it hashes target latents and reference noise but not the encoded reference content.

## Tensor diagnostics

At denoising steps 15, 25 and 35, capture at least one early and one late selected up-block PPR site:

```text
reference_hidden
reference_candidate
connector_down(reference_candidate - target_base)
raw_delta
bounded_delta
applied_delta
target epsilon before output anchor
target epsilon after output anchor
```

For any tensor `X`, calculate:

```text
relative_difference(Xa, Xb) =
    RMS(Xa - Xb) / (RMS(Xa) + 1e-12)
```

Report reference-content differences from `R1N1` versus `R2N1`, and reference-noise differences from `R1N1` versus `R1N2`.

## Image and identity metrics

Calculate within the full image and feathered face core:

- pixel MAE;
- LPIPS, if available;
- identity similarity to the original identity;
- identity similarity to the swapped identity;
- face-detection success and confidence;
- text-image similarity;
- landmark displacement and artifact/seam scores.

Create paired face crops and absolute-difference heatmaps.

For metric `d`, summarize:

```text
reference_image_effect = mean(
    d(R1N1, R2N1),
    d(R1N2, R2N2),
)

reference_noise_effect = mean(
    d(R1N1, R1N2),
    d(R2N1, R2N2),
)
```

With four noise seeds, average over every matched reference/noise comparison and report a paired bootstrap 95% confidence interval across the 96 samples.

## Decision rules

| Result | Conclusion | Next action |
|---|---|---|
| Reference-image effect is clearly larger than reference-noise effect and identity moves toward `R2` after swapping | PPR uses spatial-reference identity content | Select the best runtime scale and evaluate identity/quality trade-offs |
| Reference-noise effect is clearly larger | PPR is overly sensitive to noised-reference stochastic content | Reduce reference-noise dependence; test a deterministic or lower-noise reference memory schedule before retraining |
| Both effects are small, while all scale-4 variants differ from `PM0` | PPR residual is generic or target/PhotoMaker-conditioned | Trace the connector and reference-half conditioning; do not increase runtime scale |
| `reference_hidden` changes but `reference_candidate` does not | Reference K/V attention has collapsed to similar features | Audit attention concentration, K/V parameter changes and reference-token diversity |
| `reference_candidate` changes but `raw_delta` does not | The connector projects away reference-varying dimensions | Revise connector supervision/input or add reference-dependence training |
| `raw_delta` changes but `applied_delta` does not | Gate, cap or core mask suppresses reference sensitivity | Inspect gate, cap fraction and mask coverage |
| `applied_delta` changes but final epsilon/image does not | Later processing or denoising cancels the effect | Trace per-step epsilon accumulation and output anchoring |
| Encoded reference hashes do not change after swapping | Swap plumbing is broken | Fix reference override before interpreting model behavior |

## Follow-up if reference content is ignored

Do not simply train wrong references against the ordinary target diffusion loss; that encourages the model to ignore references.

First run an inference-only follow-up that keeps target PhotoMaker identity fixed but neutralizes the reference-half ID-only cross-attention for both `R1` and `R2`. If this restores spatial-reference sensitivity, the original target identity prompt was overriding the swapped reference stream.

If the connector remains reference-insensitive, the next training experiment should introduce one controlled dependence mechanism, such as limited target-side PhotoMaker ID dropout with the matched PPR reference retained, or a matched-versus-null/wrong reference objective whose null/wrong target explicitly suppresses the branch residual.

## Output structure

```text
ppr_8k_reference_vs_noise/
  manifest.json
  metrics_per_image.csv
  metrics_summary.csv
  tensor_diagnostics.jsonl
  contact_sheets/
  difference_heatmaps/
  PM0/
  R1N1/
  R2N1/
  R1N2/
  R2N2/
  conclusion.md
```

`conclusion.md` must state whether the observed face change is primarily reference-content-driven, reference-noise-driven, or generic/target-conditioned, and identify the first tensor stage at which swap sensitivity disappears.

## Implemented launcher

Run from the repository root:

```bash
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_reference_vs_noise_1gpu.sh
```

The launcher finds the same `checkpoint-epoch4.pth` locations as the existing
8k diagnostic scripts. Override explicitly when needed:

```bash
CHECKPOINT_PATH=/absolute/path/checkpoint-epoch4.pth \
CUDA_VISIBLE_DEVICES=0 \
OVERWRITE_OUTPUT=true \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_reference_vs_noise_1gpu.sh
```

For a short plumbing check before the complete 96-image run:

```bash
LIMIT=2 OUTPUT_DIR=/tmp/ppr_reference_noise_smoke \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_reference_vs_noise_1gpu.sh
```

The implemented tensor records store an exact SHA-256 and RMS for each
captured tensor plus a deterministic 512-value sketch. Relative differences
are calculated from matching sketch positions. This bounds disk/RAM usage
while retaining exact equality detection through the full-tensor hash.
