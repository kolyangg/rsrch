# PPR 8k reference-content versus reference-noise results

Date: 20 July 2026

## Executive conclusion

The diagnostic is valid and the PPR branch is active. Changing the spatial
reference produces a statistically clear signal that is larger than changing
reference noise. However, the dominant 4× PPR effect is **not controlled by
the spatial reference's identity**.

The best description of this checkpoint is:

> PPR is strongly active, weakly-to-moderately sensitive to spatial-reference
> content, and largely generic, target/PhotoMaker-conditioned, or
> prompt-expression-conditioned. Its reference-sensitive component does not
> move generated identity toward a swapped reference identity.

This differs from the automatic `conclusion.md` label, “primarily
reference-content-driven.” That label compares reference and noise effects
only to each other. It does not account for the much larger scale-4 versus
PhotoMaker effect and does not test the direction of identity transfer.

The immediate next experiment should remain inference-only: neutralize the
original-ID cross-attention conditioning of the reference half, while leaving
the target PhotoMaker half unchanged. Do not train the checkpoint longer and
do not merely increase gate, cap, or runtime scale.

## Data reviewed

Results:

```text
ppr_8k_reference_vs_noise2/
```

The bundle is complete:

- 96 samples for each of `PM0`, `R1N1`, `R2N1`, `R1N2`, and `R2N2`;
- 480 full images and 480 face crops;
- 384 paired-reference/noise comparisons;
- 384 difference heatmaps;
- 10,752 tensor comparison records;
- LPIPS successfully backfilled;
- all integrity assertions passed.

The visual review covered all 16 contact sheets, with closer attention to
large-effect samples such as `007_eddie`, `040_jensen`, `067_keanu`, and
`087_marion`.

## 1. Test validity

The routing and randomness controls worked as intended for all 96 samples:

| Integrity condition | Result |
|---|---:|
| Target initial latent invariant across five variants | 96/96 |
| Target prompt embedding invariant | 96/96 |
| Target PhotoMaker ID embedding invariant | 96/96 |
| R1/R2 image hash different | 96/96 |
| R1/R2 encoded reference latent different | 96/96 |
| R1N1/R2N1 reference noise identical | 96/96 |
| R1N1/R1N2 reference noise different | 96/96 |
| Every reference mask nonempty | 96/96 |
| Every selected packed ROI nonempty | Passed |
| Every scale-4 variant applies nonzero PPR residual | Passed |
| `PM0` routed through exact ordinary PhotoMaker output | Passed |

Therefore, the absence of R2 identity transfer is not explained by a broken
reference swap, stale reference latent, changed target seed, or empty mask.

## 2. Size of the PPR, reference-content, and noise effects

Define:

```text
S = average distance of the four scale-4 outputs from PM0
I = average matched-noise R1/R2 distance
N = average matched-reference N1/N2 distance
```

These are distances, not an additive variance decomposition; `I/S` and `N/S`
should be interpreted as effect-size ratios.

| Metric | PPR effect S | Reference effect I | Noise effect N | I/S | N/S | I/N |
|---|---:|---:|---:|---:|---:|---:|
| Full-image pixel MAE | 0.013431 | 0.007436 | 0.007155 | 55.4% | 53.3% | 1.04× |
| Face-core pixel MAE | 0.073010 | 0.014865 | 0.010838 | 20.4% | 14.8% | 1.37× |
| Face-crop LPIPS | 0.206571 | 0.030688 | 0.019048 | 14.9% | 9.2% | 1.61× |

The full-image ratio is not the most informative measure because the absolute
changes are localized around the face and VAE decoding spreads local changes
over nearby pixels. Face-core MAE and LPIPS are the relevant measures.

Reference content is genuinely stronger than noise:

- face-core MAE bootstrap 95% intervals do not overlap:
  - reference: `[0.01372, 0.01607]`;
  - noise: `[0.01011, 0.01168]`;
- face LPIPS intervals also do not overlap:
  - reference: `[0.02749, 0.03405]`;
  - noise: `[0.01705, 0.02145]`;
- `I > N` for 95.8% of samples by face-core MAE and 94.8% by LPIPS;
- the median per-sample `I/N` is 1.26× for core MAE and 1.45× for LPIPS.

But most of the scale-4 perceptual displacement remains shared by all four
PPR variants. Only about 15% of `S` by LPIPS is associated with changing R1
to R2, and about 9% with changing reference noise. The automatic
“reference-content-driven” classification is therefore directionally true
but substantively misleading.

## 3. Identity direction: the decisive failure

The cyclic R2 swaps include large identity changes:

```text
eddie -> elon -> jennie -> jensen -> jisoo -> keanu -> lex -> marion -> eddie
```

This includes multiple male/female swaps. Nevertheless, contact sheets remain
visually the original target person after R2 is substituted.

At fixed noise, replacing R1 with R2 produced:

| Comparison | Mean change in similarity to original | Mean change in similarity to R2 | Mean directional gain toward R2 |
|---|---:|---:|---:|
| R1N1 → R2N1 | -0.00219 | -0.00049 | +0.00170 |
| R1N2 → R2N2 | -0.00117 | -0.00017 | +0.00100 |

The medians are approximately zero or negative, and directional gain is
positive for only 50.0% and 45.8% of samples. This is random-like, not
consistent identity transfer.

The result is also inconsistent across identities. Some identity groups have
small positive means and others negative means; most medians remain near zero.
There is no repeatable R2-directed effect.

More importantly, scale-4 PPR harms the original identity:

- PM0 mean original-ID similarity: `0.5237`;
- scale-4 variants: approximately `0.3455–0.3483`;
- mean PPR-minus-PM0 change: `-0.1771`;
- median change: `-0.1801`;
- only 4.2% of samples improve original-ID similarity.

Thus the branch is not trading some original identity for the spatial
reference's identity. It is moving away from the original identity without
moving toward R2.

## 4. Visual result

### What works

- PPR clearly changes the face at 4×; it is not flat-lined or bypassed.
- Body, pose, clothing, hands, occluders, and background remain very stable.
- Changes are spatially well aligned with the existing head and face.
- Face detection succeeds for 100% of all five variants.
- No systematic duplicated face, displaced facial patch, or catastrophic
  face/body mismatch is visible.
- Obstructed poses, such as hands beside or over the face, remain spatially
  coherent.

The localization is also reflected quantitatively: mean face-core MAE from
PM0 is `0.0730`, versus full-image MAE `0.0134`.

### What does not work

The visually dominant PPR behavior is a generic change in:

- mouth opening and smile strength;
- eye closure/opening;
- emotional intensity;
- apparent age and skin texture;
- local face shape and sharpness.

Examples include strongly amplified surprise, anxiety, anger, or laughter.
These changes often track the expression words in the prompt more than the
spatial reference identity. Mean text similarity rises by `+1.085`, with
68.8% of samples improving, while identity similarity falls sharply. This
supports the hypothesis that the shortcut is partly
prompt/expression-conditioned.

R1/R2 and N1/N2 differences are generally difficult to see without crops or
heatmaps. Even the largest R1/R2 cases mostly show expression/texture changes,
not acquisition of R2's identity.

### Artifacts and alignment

There are no pervasive hard seams in the contact sheets. The face remains
properly attached to the head and body. The main quality problem is semantic,
not geometric: overdriven expression and loss of identity.

The seam-gradient proxy averages `0.0215` relative to PM0, but this metric has
no calibrated perceptual threshold. Visual inspection does not show a
corresponding systematic boundary seam. Landmark movement is modest on
average (`0.00548` of the image diagonal), although a few expression-heavy
samples move more.

## 5. Tensor-stage diagnosis

Reference-content sensitivity does **not** disappear at one broken tensor
stage.

| Stage | Mean R1/R2 relative difference | Mean N1/N2 relative difference | Content/noise ratio |
|---|---:|---:|---:|
| `reference_hidden` | 1.1527 | 0.4109 | 2.81× |
| `reference_candidate` | 0.8692 | 0.2340 | 3.72× |
| `connector_down` | 0.3334 | 0.0723 | 4.61× |
| `raw_delta` | 0.3125 | 0.0527 | 5.93× |
| `bounded_delta` | 0.2524 | 0.0470 | 5.37× |
| `applied_delta` | 0.0944 | 0.0265 | 3.56× |
| Target epsilon before anchor | 0.0909 | 0.0904 | 1.01× |
| Target epsilon after anchor | 0.0879 | 0.0875 | 1.00× |

Key interpretation:

1. R2 is encoded differently from R1.
2. Reference K/V retrieval retains a large content signal.
3. The connector does not collapse R1/R2 to equality. In fact, it suppresses
   noise more strongly than content, increasing the content/noise ratio.
4. Gate, cap, and core application attenuate the signal but do not erase it.
5. By the trajectory-level epsilon trace, content and noise perturbations
   cause similarly sized divergence, despite the cleaner content dominance
   inside the processor.

The failure is therefore not “reference_candidate never changes” or
“connector completely projects away reference variation.” The branch is
reference-sensitive, but the learned variation is not semantically aligned
with reference identity.

The higher-resolution captured site is much more sensitive:

| Site | R1/R2 `applied_delta` | N1/N2 `applied_delta` |
|---|---:|---:|
| `up_blocks.0` | 0.0324 | 0.0056 |
| `up_blocks.1` | 0.1564 | 0.0474 |

This is consistent with later sites carrying texture/expression details more
than stable global identity. It is a useful architectural clue, though only
two representative sites were captured.

Approximately one third of processor applications hit the RMS cap
(`cap_fraction ≈ 0.33`), and the mean applied residual/base RMS ratio at 4× is
about `0.134`. The cap is active, but increasing it would amplify the same
non-directional behavior and is not a solution.

One diagnostic caveat: step-15 post-anchor sketch differences are reported as
zero while exact SHA-256 values differ. The deterministic 512-value sketch
sampled unchanged positions outside the local changed region. Exact equality
must be judged by the hash; sketch magnitudes are approximate.

## 6. Architectural diagnosis

The current connector input is:

```python
reference_candidate - target_base
```

This admits a shortcut. A residual can be learned from the stable
`-target_base` term and from target queries/prompt conditioning even if the
reference-varying part is weak or semantically irrelevant. The ordinary
diffusion objective rewards any target reconstruction improvement; it does
not require that the improvement depend on matched spatial-reference
identity.

At the same time, the reference half receives the original target's ID-only
cross-attention conditioning. In the counterfactual R2 test, spatial R2
therefore competes with an R1/target identity prompt inside the reference
stream. The resulting memory can preserve spatial variation without exposing
R2 identity in a usable form.

Together, these explain the observed result:

- a large generic/prompt-conditioned correction;
- measurable R1/R2 sensitivity;
- no consistent identity direction;
- severe degradation of original-ID similarity at 4×.

Training longer is unlikely to change this incentive. More steps may reinforce
the shortcut.

## 7. Recommended next sequence

### Priority 1 — decisive inference-only reference-CA ablation

Use the same checkpoint, scale 4, target seeds, prompts, masks, and scheduler.
Keep target-half PhotoMaker conditioning unchanged. For the reference half
only, compare:

1. current original-ID cross-attention;
2. neutral/null ID-only cross-attention;
3. optionally, identity conditioning extracted from the actual spatial
   reference rather than the target PhotoMaker identity.

Run a 12-sample smoke test first, deliberately including cross-sex R2 swaps
and the high-content-effect samples `040`, `067`, and `087`. If sensitivity
appears, run all 96.

Decision:

- R2 identity begins to transfer: reference-half target-ID CA is overriding
  spatial identity; redesign that conditioning before retraining.
- Tensor and pixel sensitivity remains but identity direction remains zero:
  the learned connector/objective is semantically wrong; proceed to Priority
  2.
- PPR effect collapses entirely: the connector depended mostly on
  target-ID-conditioned reference features, confirming the shortcut.

### Priority 2 — dependence-guaranteed residual architecture

Replace the shortcut-prone difference with a matched-versus-null reference
contrast:

```text
C_ref  = attention(Q_target, K_ref,  V_ref)
C_null = attention(Q_target, K_null, V_null)
delta  = connector(C_ref - C_null)
```

The null path should use the same target query and timestep but no person
reference evidence. This removes the direct `-target_base` shortcut and makes
zero reference evidence map naturally to zero branch residual.

Keep the core branched-attention mechanism:

- doubled target/reference U-Net streams;
- target Q retrieving reference K/V;
- packed reference-face ROI;
- additive bounded residual;
- PhotoMaker anchor outside the face core.

This is the highest-priority training architecture if the CA ablation does not
restore identity direction.

### Priority 3 — train an explicit matched-reference dependence objective

The ordinary diffusion target alone cannot distinguish a genuinely
reference-dependent residual from a useful generic correction.

For each target, create matched and null/wrong-reference branch evaluations
with explicit semantics:

- matched reference: normal diffusion/face objective;
- null reference: branch residual target is zero;
- wrong reference: either branch-off target or a contrastive identity target,
  not the ordinary matched diffusion target;
- dependence loss: require matched residual/features to differ from null while
  suppressing null residual magnitude.

Do not train wrong references against the ordinary target diffusion loss
without a branch-off target; that rewards reference invariance.

### Priority 4 — limited target PhotoMaker-ID attenuation

On a controlled fraction of training examples, attenuate or drop target-side
PhotoMaker ID conditioning while retaining the matched spatial reference.
Mix these with full-PhotoMaker examples so inference at full target
conditioning remains supported.

This can force the PPR route to carry identity information, but it should be
combined with a null-reference/dependence constraint; dropout alone may create
another generic restoration shortcut.

### Priority 5 — site specialization

The captured `up_blocks.1` site carries much more content and noise sensitivity
than `up_blocks.0`, while visual changes are dominated by expression and
texture. A promising architecture is:

- lower/mid-resolution PPR sites: identity-shape residual with identity
  supervision;
- later/higher-resolution sites: smaller gate/cap or detail-only residual;
- independent learned gates per resolution group with diagnostics.

Test this only after dependence is enforced. Simply reducing late-site scale
now may improve stability but will not create identity direction.

## 8. What not to do

- Do not train this checkpoint longer as the main next action.
- Do not increase runtime scale, gate maximum, or RMS cap.
- Do not interpret `I > N` alone as successful identity conditioning.
- Do not remove the PhotoMaker outside-core anchor; pose/body preservation is
  one of the parts that works.
- Do not train wrong references toward the ordinary target.
- Do not discard packed ROI branched attention: the tensor trace proves that
  content reaches the connector. The missing element is dependence semantics,
  not basic reference plumbing.

## Final recommendation

Run the neutral-reference-CA inference ablation next. In parallel, prepare one
fresh architecture—not a continuation—that uses
`C_ref - C_null` and an explicit matched/null dependence objective. Preserve
the current doubled-stream branched-attention topology, packed face ROI, and
outside-core PhotoMaker anchor.

The result is not a dead branch. It is a live branch trained to do the wrong
semantic job.
