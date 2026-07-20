# PPR 8k: R1/R2 × N1/N2 Result Analysis

## Observed pattern

```text
4× differs visibly from 1×
R1N1 ≈ R1N2 ≈ R2N1 ≈ R2N2 at 4×
```

Likely interpretation: **PPR is active, but its learned residual is largely generic or target/PhotoMaker-conditioned rather than controlled by spatial-reference identity or reference noise.**

The processor applies:

```python
connector_hidden = connector_down(reference_candidate - target_base)
raw_delta = connector_up(connector_hidden)
applied_delta = core * gate * runtime_scale * bounded_delta
```

See [`packed_residual_attn_processor.py`](https://github.com/kolyangg/rsrch/blob/main_clean/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py#L436-L474).

Thus, `runtime_scale=4` can amplify a nonzero residual dominated by `-target_base` or a generic reference-face component even when changing the reference image or its noise has almost no effect.

## 1. Confirm that the R/N test is valid

Check that the run completed all integrity assertions in [`ppr_reference_noise.py`](https://github.com/kolyangg/rsrch/blob/main_clean/diffusion_template/src/trainer/ppr_reference_noise.py#L452-L543):

- Target latent, prompt, PhotoMaker ID and target seed are identical across runs.
- R1 and R2 have different image and encoded-reference hashes.
- N1 and N2 have different reference-noise hashes.
- Noised reference tensors differ.
- Reference mask is nonempty and packed ROI length is greater than zero.
- Applied PPR residual is nonzero at scale 4.

If any assertion fails, fix the test routing before interpreting the images.

## 2. Quantify the effects

Do not rely only on contact sheets. Read:

- `paired_effects.csv`
- `metrics_summary.csv`
- `difference_heatmaps/`
- `tensor_diagnostics.jsonl`

For each sample, compute face-core LPIPS and MAE; aggregate across samples:

```text
S = mean difference(scale 4, scale 0 or 1)
I = mean[diff(R1N1,R2N1), diff(R1N2,R2N2)]
N = mean[diff(R1N1,R1N2), diff(R2N1,R2N2)]

reference_fraction = I / S
noise_fraction     = N / S
```

Use both perceptual and pixel metrics. Report median, mean and per-sample distribution.

| Result | Interpretation |
|---|---|
| `S` large; `I/S` and `N/S` near zero | Strong evidence of generic/target-conditioned PPR residual |
| `I/S` meaningful but visually subtle | Spatial reference affects output below easy visual detection |
| `N/S` meaningful | Branch is sensitive to reference-stream noise |
| `S`, `I`, and `N` all tiny | Final-image PPR effect remains weak despite 4× |

The paired image effects are implemented in [`ppr_reference_noise.py`](https://github.com/kolyangg/rsrch/blob/main_clean/diffusion_template/src/trainer/ppr_reference_noise.py#L697-L743).

## 3. Check identity direction

For every output, calculate identity similarity against both R1 and R2:

```text
sim_to_R1
sim_to_R2
delta_toward_R2 = sim_to_R2 - sim_to_R1
```

Compare the change from R1 to R2 at fixed noise. Pixel differences alone do not prove identity transfer.

## 4. Locate where reference sensitivity disappears

At identical timesteps and processor sites, compare R1 versus R2 for:

```text
reference_hidden
reference_candidate
connector_down
raw_delta
bounded_delta
applied_delta
target_epsilon_pre_anchor
target_epsilon_post_anchor
```

For each tensor `X`, compute:

```text
r_X = RMS(X_R1 - X_R2) / (RMS(X_R1) + eps)
```

| First stage where R1/R2 difference disappears | Likely cause |
|---|---|
| `reference_hidden` | Swap routing, packed batch half, mask, or early reference conditioning |
| `reference_candidate` | Reference K/V attention collapses identities into generic features |
| `connector_down` | Connector projects away reference-specific dimensions |
| `bounded_delta` | RMS cap removes reference-dependent variation; inspect `cap_fraction` |
| `applied_delta` | Gate or spatial core suppresses the variation |
| Target epsilon | Later U-Net processing or final anchor cancels it |
| Final image only | Denoising washes out a weak but real epsilon difference |

Tensor capture/comparison is in [`ppr_reference_noise.py`](https://github.com/kolyangg/rsrch/blob/main_clean/diffusion_template/src/trainer/ppr_reference_noise.py#L375-L449).

## 5. Do not trust the automatic classifier alone

The current classifier uses a relative `1.25×` comparison but no absolute-effect floor. Tiny values can therefore be labelled reference- or noise-driven. See [`ppr_reference_noise.py`](https://github.com/kolyangg/rsrch/blob/main_clean/diffusion_template/src/trainer/ppr_reference_noise.py#L871-L886).

Base the conclusion on `I/S`, `N/S`, absolute face-core metrics, and tensor-stage results. The tensor diagnostic uses a deterministic 512-value sketch, so use SHA-256 for exact equality and the sketch only for approximate magnitude.

## 6. Run the decisive inference-only ablation

The current R2 swap changes the spatial latent, but the reference half still receives the original PhotoMaker identity prompt. Frozen reference-half cross-attention may recondition both R1 and R2 toward the original identity.

Using the same checkpoint at scale 4:

1. Keep target PhotoMaker identity, prompt, seed, mask and scheduler unchanged.
2. Neutralize reference-half ID-only cross-attention identically for R1 and R2.
3. Generate R1N1, R1N2, R2N1 and R2N2 again.
4. Recompute `S`, `I`, `N`, identity-to-both-references metrics, and tensor-stage differences.

Interpretation:

- **R1/R2 sensitivity appears:** original-ID reference-half cross-attention was overriding or normalizing the spatial reference.
- **R1/R2 remains negligible while PPR stays nonzero:** the checkpoint learned a generic/target-conditioned correction.
- **Reference candidates differ but connector output does not:** redesign or retrain the connector/objective; further inference scaling will not restore reference dependence.

## Decision

Do not retrain until the neutral-reference-CA ablation and tensor-stage analysis are complete.

If the residual is confirmed generic, increasing scale, gate or cap will only amplify the same generic correction. The next training run must explicitly require matched-reference dependence—for example target-side ID dropout/attenuation, a matched-versus-null/wrong-reference objective, or a connector design that cannot exploit the `-target_base` shortcut.

Do not train wrong references against the ordinary target without an explicit branch-off target; that would encourage the model to ignore the reference.
