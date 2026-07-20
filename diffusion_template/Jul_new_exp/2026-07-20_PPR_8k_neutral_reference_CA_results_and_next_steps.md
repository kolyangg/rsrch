# PPR 8k neutral-reference-CA results and next steps

Date: 20 July 2026

## Executive conclusion

Neutralizing cross-attention on the reference half does **not** restore
spatial-reference identity control in the NN2-PPR epoch-4 checkpoint.

The PPR branch is active, affects the face strongly, and is more sensitive to
reference content than to reference noise. However, changing R1 to a different
person R2 does not move the generated face toward R2. The mean directional
identity gain is slightly negative and statistically consistent with zero.

Visually, the branch behaves mainly as a target/prompt-conditioned facial
expression and rendering modifier:

- mouths open or smiles become stronger;
- eyes close or open and emotion becomes more intense;
- apparent age, skin texture, and local face shape change;
- body, pose, clothing, occluders, and background remain stable.

Therefore, original-ID reference-half cross-attention was not the primary
cause of the failed identity transfer. It may modulate the branch, but removing
it is insufficient. The checkpoint has learned a live but semantically
misdirected residual.

The next training architecture should explicitly make the residual depend on
reference evidence. The strongest candidate is a contrastive residual
`C_ref - C_null`, evaluated with the same target query, plus an explicit
matched/null dependence objective.

## Data reviewed

Primary result:

```text
ppr_8k_neutral_reference_ca/
```

Comparison result with ordinary reference-half cross-attention:

```text
ppr_8k_reference_vs_noise2/
```

The neutral-CA result bundle is complete:

- 96 samples;
- five variants per sample: `PM0`, `R1N1`, `R2N1`, `R1N2`, and `R2N2`;
- 480 generated images and 480 face crops;
- 384 difference heatmaps;
- 16 contact sheets;
- 384 paired reference/noise comparisons;
- 10,752 tensor diagnostic records;
- MAE, LPIPS, identity, text, face-detection, landmark, and seam-proxy metrics;
- all integrity assertions passed.

The run used:

- checkpoint `checkpoint-epoch4.pth` from `ba_NN2_ppr1_realvis_1gpu`;
- RealVisXL V4.0 validation base;
- 50 denoising steps and guidance scale 5;
- PhotoMaker from step 10;
- merge from step 10;
- branched attention from step 15;
- runtime PPR scale 4 for all R/N variants;
- reference-half CA mode `zero`;
- batch size 12.

## Important comparison caveat

The original-CA diagnostic was generated at batch size 1, while the neutral-CA
diagnostic was generated at batch size 12.

The test inputs are logically matched:

- all 480 initial-latent hashes match positionally;
- all 480 spatial-reference-image hashes match positionally;
- sample order, prompts, seeds, checkpoint, scheduler settings, and variants
  are the same.

But the executions are not numerically identical:

- target prompt and PhotoMaker embedding hashes differ between the two runs;
- reference RNG tensors are grouped differently by batch;
- none of the 96 ordinary PhotoMaker `PM0` image hashes are byte-identical;
- the old-versus-neutral `PM0` full-image MAE is approximately `0.0182`.

This is consistent with batch-shape-dependent floating-point behavior and RNG
grouping. It means that small old-versus-neutral metric changes cannot be
causally attributed to neutral reference CA.

All conclusions drawn **within** the neutral run are valid because its five
variants are controlled internally. A matched ordinary-CA run at batch size 12
is still needed to estimate the isolated effect of the CA ablation.

## 1. Effect decomposition

Define:

```text
S = average distance of the four scale-4 PPR outputs from PM0
I = average R1/R2 distance while holding reference noise fixed
N = average N1/N2 distance while holding reference image fixed
```

These are paired distances, not an additive variance decomposition.

| Metric | PPR effect S | Reference effect I | Noise effect N | I/S | N/S | I/N |
|---|---:|---:|---:|---:|---:|---:|
| Full-image pixel MAE | 0.015447 | 0.007623 | 0.007283 | 49.3% | 47.1% | 1.05× |
| Face-core pixel MAE | 0.072389 | 0.016308 | 0.011521 | 22.5% | 15.9% | 1.42× |
| Face-crop LPIPS | 0.202987 | 0.035162 | 0.020705 | 17.3% | 10.2% | 1.70× |

Reference content has a real and statistically measurable pixel/perceptual
effect:

- reference face-core MAE 95% bootstrap interval:
  `[0.01495, 0.01788]`;
- noise face-core MAE interval:
  `[0.01073, 0.01248]`;
- reference face LPIPS interval:
  `[0.03101, 0.03990]`;
- noise face LPIPS interval:
  `[0.01841, 0.02344]`.

The branch therefore does not ignore the reference tensor. Nevertheless, only
about 17% of the total scale-4 face LPIPS displacement is associated with
changing R1 to R2. Most of the visible PPR effect is shared by all four
reference/noise variants.

The automatic label “primarily reference-content-driven” is incomplete. It
only observes that `I > N`; it does not test whether the reference effect is
identity-directed.

## 2. Identity direction

The decisive test is whether replacing R1 with R2 increases similarity to R2
relative to similarity to the original identity:

```text
directional gain =
    change in similarity to R2
  - change in similarity to the original identity
```

| Fixed noise | Mean directional gain toward R2 | Median | Positive fraction | Bootstrap 95% interval |
|---|---:|---:|---:|---:|
| N1 | -0.00444 | -0.00413 | 41.7% | [-0.01002, +0.00146] |
| N2 | -0.00167 | -0.00279 | 46.9% | [-0.00709, +0.00385] |
| Combined | **-0.00306** | **-0.00323** | **44.3%** | **[-0.00721, +0.00095]** |

This is random-like and, if anything, weakly points in the wrong direction.
The cyclic swaps include conspicuous identity and sex changes, so a successful
identity route should have produced a visible and metric-detectable shift.

The result is also unstable across target identities:

| Original target | Mean directional gain | Positive fraction |
|---|---:|---:|
| Eddie | +0.01307 | 70.8% |
| Elon | -0.00541 | 41.7% |
| Jennie | -0.02945 | 16.7% |
| Jensen | -0.01137 | 29.2% |
| Jisoo | -0.00237 | 33.3% |
| Keanu | +0.01025 | 62.5% |
| Lex | +0.00514 | 58.3% |
| Marion | -0.00433 | 41.7% |

This pattern is incompatible with a general identity-transfer mechanism.
Large reference-sensitive image differences also fail this test: they mostly
change expression, eyes, mouth, or texture rather than acquiring R2 identity.

## 3. Visual assessment

### What works

- The PPR branch visibly changes the output; this is not a bypass or failed
  checkpoint load.
- Changes are concentrated on the face and remain aligned with the head.
- Body pose, camera, clothing, hands, occluders, and background are preserved.
- Face detection succeeds for 100% of PM0 and PPR outputs.
- There are no systematic duplicated faces, pasted facial patches, or severe
  face/body detachment.
- Even difficult hand-near-face examples remain geometrically coherent.

This confirms that the face mask, packed ROI, target/reference routing, and
outside-core PhotoMaker preservation are useful parts of the architecture.

### What does not work

- R1 and R2 variants generally look like the same target person.
- Reference identity is not visibly transferred, including for large
  male/female swaps.
- The dominant change is expression strength rather than identity.
- Several outputs exaggerate open mouths, laughter, distress, anger, or eye
  closure beyond the PM0 rendering.
- Some faces become smoother, sharper, older, or locally reshaped without a
  consistent reference-directed semantic effect.

Representative strong but misdirected changes include samples `007`, `031`,
`040`, `055`, `079`, `087`, and `088`. They demonstrate branch strength, not
identity control.

### Artifacts and face/body alignment

The principal failure is semantic rather than geometric.

- Mean landmark displacement from PM0 is `0.00379` of the image diagonal.
- The seam-gradient proxy is `0.02178`, but visual review does not show a
  pervasive hard face boundary.
- Mean face confidence changes only from `0.8340` at PM0 to `0.8327`.

There are occasional unnatural expressions and local facial distortions, but
no systematic pose mismatch or detached-face artifact.

## 4. PhotoMaker identity and prompt behavior

At scale 4:

| Quantity | PM0 | Mean PPR | PPR minus PM0 |
|---|---:|---:|---:|
| Original-identity similarity | 0.52313 | 0.35598 | **-0.16715** |
| Text similarity | 26.3659 | 27.4749 | **+1.1090** |
| Face-detection rate | 100% | 100% | 0 pp |

PPR strongly reduces the original identity while increasing text similarity.
Because it also fails to move toward R2, it is not performing a useful
identity trade. The result supports a prompt/expression-conditioned shortcut:
the branch improves or exaggerates prompt-level facial semantics while
damaging stable identity.

## 5. Tensor-stage diagnosis

Reference-content sensitivity survives every captured internal stage:

| Stage | Mean R1/R2 relative difference | Mean N1/N2 relative difference |
|---|---:|---:|
| `reference_hidden` | 1.1656 | 0.4253 |
| `reference_candidate` | 0.9542 | 0.2597 |
| `connector_down` | 0.2963 | 0.0734 |
| `raw_delta` | 0.2941 | 0.0591 |
| `bounded_delta` | 0.2388 | 0.0507 |
| `applied_delta` | 0.0879 | 0.0272 |
| Target epsilon before anchor | 0.0884 | 0.0893 |
| Target epsilon after anchor | 0.0857 | 0.0866 |

Interpretation:

1. R1 and R2 enter the branch as substantially different representations.
2. Target-Q/reference-KV attention retains a strong content difference.
3. The connector attenuates but does not erase reference sensitivity.
4. The bounded residual and face core attenuate it further.
5. At the whole target-epsilon trajectory, reference-content and noise
   perturbations become similarly sized.
6. No single broken stage explains the failure. The retained content
   dimensions are not semantically aligned with identity.

The later captured site dominates:

| Site | R1/R2 `applied_delta` | N1/N2 `applied_delta` |
|---|---:|---:|
| `up_blocks.0` | 0.03446 | 0.00689 |
| `up_blocks.1` | 0.14125 | 0.04755 |

This agrees with the visual result: the high-resolution route is effective at
changing expression and texture, but not at transferring stable identity
shape.

The mean applied residual/base RMS ratio is `0.1323` and approximately 30.6%
of processor applications hit the cap. The branch has ample magnitude.
Increasing scale, gate, or cap would amplify the wrong behavior.

## 6. What the neutral-CA ablation establishes

The ordinary-CA run and neutral-CA run both show:

- strong scale-4 facial modification;
- measurable R1/R2 content sensitivity;
- no reliable movement toward R2 identity;
- substantial loss of original identity;
- stable body and scene geometry.

Neutral CA does not reveal a hidden identity route. Its combined directional
gain is `-0.00306`, compared with approximately `+0.00135` in the old
ordinary-CA run; both are statistically and visually consistent with no
transfer.

Because batch sizes differ, small changes in MAE or LPIPS cannot be assigned
to the CA mode. But the main conclusion is robust: removing reference-half
ID cross-attention is not sufficient, and original-ID CA is not the primary
blocker.

## 7. Architectural diagnosis

The current PPR connector is driven by:

```text
reference_candidate - target_base
```

where `reference_candidate` uses target queries with reference K/V.

This gives the training objective several shortcuts:

1. The stable `-target_base` term is available even when reference-specific
   evidence is weak.
2. Target queries already contain target/prompt information.
3. Ordinary matched diffusion reconstruction rewards any useful face
   correction; it does not require the correction to change when the spatial
   reference changes.
4. The strongest route is at a late/high-resolution site, where expression
   and texture are easier shortcuts than global identity structure.

Neutralizing reference-half CA removes one competing identity signal, but it
does not remove these shortcuts or add a reference-dependence incentive.

The architecture is therefore reference-sensitive but not
reference-identity-controlled.

## 8. Immediate matched control

Before comparing CA modes numerically, rerun ordinary reference CA at the same
batch size 12:

```bash
cd /home/niko/rsrch/diffusion_template

CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=ba_NN2_ppr1_realvis_8k_original_ca_bs12 \
OUTPUT_DIR=/home/niko/rsrch/diffusion_template/ppr_8k_original_reference_ca_bs12 \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_reference_vs_noise_1gpu.sh 12
```

This is useful for measuring the isolated CA-mode effect, but it is not a
reason to delay architectural work: the neutral run itself already shows that
zero CA does not solve identity transfer.

## 9. Recommended next training runs

Keep the core branched-attention design in every run:

- doubled target/reference U-Net streams;
- target Q retrieving reference K/V;
- packed face ROI and face mask;
- additive bounded target residual;
- ordinary PhotoMaker preservation outside the face core;
- new behavior behind reversible configuration toggles.

### NN3a — reference-minus-null contrastive residual

Use the same target query and timestep for two reference memories:

```text
C_ref  = attention(Q_target, K_ref,  V_ref)
C_null = attention(Q_target, K_null, V_null)
delta  = connector(C_ref - C_null)
```

The null memory must contain no person-reference evidence. This cancels the
target-query/common component by construction and gives the branch a natural
zero point.

Training:

- matched reference: normal face/diffusion objective;
- null reference: explicitly penalize branch residual magnitude toward zero;
- retain the PM anchor outside the face.

This is the cleanest test of whether the connector can learn useful identity
information once the target-base shortcut is removed.

### NN3b — NN3a plus controlled target PhotoMaker-ID attenuation

Use NN3a and, on a controlled fraction of matched-reference training examples,
attenuate or drop target-side PhotoMaker ID conditioning during BA-active
timesteps.

Retain full-PhotoMaker examples in the mixture. The purpose is to prevent the
model from solving identity entirely through target PhotoMaker while still
supporting the intended inference regime.

Pair this with the null-reference zero-residual objective. ID attenuation
alone could otherwise produce another generic restoration shortcut.

### NN3c — NN3a plus multiscale identity specialization

Route identity-changing residuals through a coarser/mid-resolution BA site and
reduce the role of the late `up_blocks.1` route.

Suggested structure:

- coarse/mid site: identity shape and proportions;
- early upsampling site: face structure refinement;
- late/high-resolution site: separately gated, lower-cap detail residual;
- a low-timestep decoded face-identity objective tied to the matched spatial
  reference;
- unchanged outside-core PhotoMaker anchor.

This directly addresses the observation that the current late site dominates
expression and texture.

### GPU allocation

Use three two-GPU runs:

- four-GPU machine: NN3a on GPUs 0–1 and NN3b on GPUs 2–3;
- two-GPU machine: NN3c on GPUs 0–1.

Screen at 2k, 4k, 6k, and 10k steps. Stop any run that has a strong PPR effect
but still fails the directional R2 test. Do not wait for 20k if identity
direction remains random while original identity degrades.

## 10. Evaluation changes for the next runs

Use identical batch size for every compared configuration and retain the
current five-way test.

Add:

1. same-person, different-reference-image pairs to separate identity from
   pose/crop/lighting changes;
2. cross-person swaps as the decisive identity-control test;
3. null-reference outputs to verify that the new residual approaches zero;
4. identity similarity to both original and swapped identities;
5. outside-face LPIPS/MAE to protect body, pose, hands, and background;
6. per-site residual magnitude and reference/null separation;
7. matched versus mismatched reference tests with explicit semantics.

Do not interpret `I > N` as success. A promising run should show:

- a positive mean directional gain toward R2 with a bootstrap interval above
  zero;
- a clear majority of samples moving toward R2;
- preservation of body/scene geometry;
- null-reference residual close to zero;
- reduced dominance of generic expression amplification.

## 11. What not to do

- Do not continue training the current checkpoint as the main next action.
- Do not increase PPR runtime scale, gate, or RMS cap.
- Do not remove the outside-core PhotoMaker anchor.
- Do not discard packed ROI branched attention; the diagnostics show that
  reference content reaches the connector.
- Do not train wrong references against the ordinary target diffusion target.
  That directly rewards ignoring the wrong reference.
- Do not treat neutral reference CA as a fix by itself.

## Final recommendation

Run the matched batch-12 ordinary-CA control for clean attribution, then start
fresh NN3a/NN3b/NN3c training rather than continuing NN2-PPR.

The highest-priority change is not more residual strength. It is an
architecture and objective in which the branch cannot produce its useful
correction without reference evidence.
