# CL19 preserves reference-conditioned BA while fixing CL14's thresholded boundary handover

**Date:** 11 August 2026  
**Scope:** Read-only architecture analysis of CL19 versus CL14, pose adaptation, and historical `ca_mixing_for_face`. No code, configuration, checkpoint, or running job was changed.  
**Evidence cutoff:** Source and live startup records available on 11 August 2026. CL19 training outcomes are not yet evaluated here.

Run records:

- **CL14 architecture baseline:** immutable Comet key
  `6fe0028be92242c38056b3d36665fdd6`; final `24k` checkpoint.
- **CL19 architecture under analysis:** immutable Comet key
  `cfeda7b55c174b3c83e8d40537ebb6dd`; training in progress, with no result
  claim in this report.

The primary outcome metric for a later matched-step comparison is
`id_sim_subject_v2`. Hard-case identity slices (Marion, Crying, Skiing, and
small-face prompts) and visual face/body blending are required secondary
outcomes because an overall mean can hide the boundary failures that motivate
CL19.

## Executive conclusion

**CL19 keeps the core branched-attention idea.** `[code]` Its identity lane
still uses queries from the target/noisy image and keys/values from the
spatial reference image. It does not replace BA with PhotoMaker tokens, target
K/V, pose adaptation, or CA mixing.

CL19 changes the ownership rule at the target face boundary. CL14 masks the
query before attention and masks the resulting message again. Its intended
two-cell training feather is thresholded by `force_binary_masks=true`, so the
weights `1/3` and `2/3` become a hard `0/1` erosion. `[code]` CL19 instead
computes a complete native target message and a complete target-Q/reference-KV
message, then performs one convex spatial blend with router weights
`0`, `0.25`, `0.75`, and `1`. `[code]`

Pose adaptation and historical `ca_mixing_for_face` intervene at a different
place. Pose adaptation contaminates reference K/V with target-face features.
CA mixing expands the face lane's K/V memory with a second copy of native
target-face features and lets softmax choose between sources. CL19 leaves the
reference K/V source pure and decides only how much of each completed message
owns each target location. `[code]`

## 1. Shared CL14 and CL19 substrate

Both runs inherit the same PhotoMaker plus hard-v1 spatial BA substrate:

- `[code]` The U-Net input is doubled as `[target/noise, spatial reference]`.
- `[code]` The spatial identity lane uses target queries and reference-image
  face K/V.
- `[code]` `pose_adapt_ratio=0` and `ca_mixing_for_face=false`.
- `[code]` The legacy branched `attn2` cross-attention processor is disabled.
  Native PhotoMaker cross-attention remains active and consumes PhotoMaker/text
  conditioning.
- `[code]` Rank-128 hard-v1 BA, the effective generic rank-32 adapter, and the
  effective PhotoMaker-default rank-64 adapter remain the trainable substrate.
- `[measured]` The live CL19 startup contract is exactly `2,240` trainable
  tensors and `219,217,920` parameters, identical to CL14. CL19 adds no learned
  router or gate.
- `[code]` `ba_hard_v1_true_reference_key_mask=false`. Reference features
  outside the declared reference face are zeroed, but their token positions
  remain zero sinks in the reference softmax denominator. This behavior is
  deliberately unchanged from CL14.

At inference, both retain the same staged conditioning schedule: ordinary SDXL
before step `10`, PhotoMaker from step `10`, and PhotoMaker plus spatial BA from
step `15` of the fixed `50` DDIM steps. `[measured]`

\newpage

## 2. CL14's actual hard route

Let:

- $X_t$ be target/noisy-image features;
- $X_r$ be spatial-reference features;
- $M_t$ be the target face mask;
- $M_r$ be the reference face mask;
- $Q_t=Q(X_t)$.

Ignoring the unchanged reference-half self-attention, CL14 forms two target
messages. Its native lane is approximately:

$$
A_n = \operatorname{Attn}\left(Q_t(1-M_t), K_t(X_t), V_t(X_t)\right).
$$

Its identity lane is:

$$
A_r = \operatorname{Attn}\left(Q_tM_t, K_r(X_rM_r), V_r(X_rM_r)\right).
$$

It then merges the messages as:

$$
Y_t = O\left((1-M_t)A_n + M_tA_r\right).
$$

This is valid as a hard route when $M_t$ is binary. It is not a clean soft
router. A fractional mask first scales $Q_t$, changing the logits and entropy
inside attention, and then scales the output a second time. Therefore a mask
value of `0.5` does not mean a `50/50` convex mixture of complete native and
reference messages. `[code]`

CL14 constructs inward training rings with values `1/3` and `2/3`, but the
installed processor resizes and thresholds the mask at `>0.5`. The result is a
hard outer `0` ring and inner `1` ring, not a continuous feather. `[code]` This
explains why CL14 can still improve aggregate results while retaining abrupt
handover behavior around hair, goggles, and face-box edges. `[hypothesis]`

## 3. CL19's full-query soft router

CL19 computes both target lanes with the same full target query:

$$
A_n = O\left(\operatorname{Attn}(Q_t, K_t(X_t), V_t(X_t))\right),
$$

$$
A_r = O\left(\operatorname{Attn}(Q_t, K_r(X_rM_r), V_r(X_rM_r))\right).
$$

The BA invariant is explicit in the second equation:

> target Q + spatial-reference K/V.

Only after both complete messages are available does CL19 blend them:

$$
Y_t = (1-R)A_n + RA_r.
$$

The deterministic two-cell cosine router $R$ is:

| Target location | Native weight $1-R$ | Reference-BA weight $R$ |
|---|---:|---:|
| outside target face | `1.00` | `0.00` |
| outer boundary cell | `0.75` | `0.25` |
| inner boundary cell | `0.25` | `0.75` |
| face interior | `0.00` | `1.00` |

Dropout, the U-Net residual, and output rescaling are applied once after the
blend. `[code]` The reference half of the doubled batch still performs full
reference self-attention with reference Q/K/V, exactly as in CL14. `[code]`

CL19 installs `soft_router` in the configured down blocks `0-2`, mid block,
and up blocks `0-2`, covering the configured U-Net self-attention stack.
`[code]` The same processor map is copied into validation, so the trained and
inference equations are aligned. `[code]`

## 4. Difference from `pose_adapt_ratio`

Pose adaptation changes the source used to construct face K/V. With ratio
$\alpha$, the historical hard-v1 implementation first creates:

$$
Z = (1-\alpha)(X_rM_r) + \alpha(X_tM_t),
$$

and then uses $K(Z),V(Z)$ for the face lane. `[code]`

| Setting | Face K/V content | Consequence |
|---|---|---|
| `pose_adapt_ratio=0` | pure spatial reference | eligible project BA route; strongest explicit reference evidence |
| `0 < pose_adapt_ratio < 1` | reference and target features mixed before K/V projection | more target geometry, but weaker causal attribution to reference identity |
| `pose_adapt_ratio=1` | pure target-face features | no spatial-reference K/V remains; this is native target attention, not the project's eligible BA design |

The distinction from CL19 is structural:

- Pose adaptation changes **what evidence is stored in K/V**.
- CL19 changes **which completed message owns each target position**.
- Pose adaptation acts throughout the face lane, including the face interior.
- CL19 keeps the interior reference-conditioned and softens only the geometric
  handover rings.
- Pose adaptation can trade identity for pose because target features enter the
  identity memory. CL19 leaves reference identity causality intact.

For this reason, CL19 fail-closes unless `pose_adapt_ratio=0`. `[code]` A
nonzero value would make it impossible to attribute a result to the soft router
alone and would violate the current project experiment contract.

## 5. Difference from historical `ca_mixing_for_face`

The name is potentially misleading. Historical `ca_mixing_for_face` is not
the native U-Net `attn2` text/PhotoMaker cross-attention processor. It is a K/V
memory expansion inside the spatial branched self-attention face lane. `[code]`

After constructing pose-mixed features $Z$, that route concatenates:

$$
C = [Z; X_tM_t]
$$

along the token dimension and computes face K/V from $C$. Each target-face
query then attends over both the reference/pose-mixed sequence and an explicit
native target-face sequence. `[code]`

This differs from CL19 in three ways:

1. **Decision mechanism.** CA mixing delegates source selection to attention
   softmax independently for every face query. CL19 uses an explicit auditable
   spatial ownership router.
2. **Decision location.** CA mixing combines source memories before attention.
   CL19 combines completed native and reference attention messages after
   attention.
3. **Identity causality.** CA mixing lets the face lane ignore reference tokens
   in favor of native target tokens. CL19's reference lane remains pure
   target-Q/reference-KV, and native information enters only through the
   separately named native lane.

The current clean processor hardcodes the historical CA-mixing behavior off,
and both CL14 and CL19 configure `ca_mixing_for_face=false`. `[code]` Both also
keep the legacy branched `attn2` processor disabled. Their PhotoMaker identity
tokens still operate through native PhotoMaker cross-attention; CL19 does not
remove or replace that conditioning. `[code]`

## 6. One-table architecture comparison

| Property | CL14 | CL19 | Pose adaptation | Historical CA mixing |
|---|---|---|---|---|
| Target query source | target | target, full in both lanes | target | target |
| Reference-lane K/V | pure reference face | pure reference face | reference/target mixture | reference/target memory plus extra target sequence |
| Native target lane | yes | yes, computed in full | partly folded into reference lane | added inside face K/V memory |
| Where sources combine | masked Q plus hard output merge | one post-attention convex blend | before K/V projection | before attention by sequence concatenation |
| Spatial ownership | binary | `0/.25/.75/1` | still uses outer face mask; no boundary solution | still uses outer face mask; no boundary solution |
| Pure reference causality | yes | yes | no when ratio is nonzero | no |
| New learned parameters versus CL14 | none | none | none | none |
| Eligible current project BA design | yes | yes | only at ratio `0` | no; configured off |

## 7. Expected behavior and limits

`[hypothesis]` CL19 should primarily improve boundary continuity: hair crossing
the box edge, skin/background transitions, and seams caused by abruptly
switching between native and reference messages. It should retain identity
strength in the face core because $R=1$ there.

`[hypothesis]` CL19 is not a semantic occlusion solution. A hand, tear, pair of
glasses, or goggle located well inside the face mask remains in the
reference-dominated core. CL17, not CL19, is the arm that predicts native
ownership for those structures. CL19 also does not increase local face
resolution; CL15 targets small faces.

`[hypothesis]` The main risk is identity dilution in the two transition rings,
especially for very small faces where those rings occupy a large fraction of
the face tokens. This must be assessed on the small-face subset rather than
only through overall `id_sim_subject_v2`.

## 8. Confidence

| Claim | Confidence | Basis |
|---|---|---|
| CL19 keeps target-Q/reference-KV BA | high | direct inspection of `_full_target_lanes`; live mode is `soft_router` |
| CL14's intended feather is thresholded | high | direct mask-construction and `_prepare_mask` inspection |
| CL19 blends complete messages once | high | direct `_call_hardcase` and `_finish_full_router` inspection |
| CL19 and CL14 have the same trainable count | high | live CL19 startup summary and sealed CL14 contract |
| Pose adaptation contaminates reference K/V with target features | high | direct hard-v1 face-source equation in code |
| Historical CA mixing is K/V sequence concatenation, not `attn2` cross-attention | high | direct archived processor inspection plus current runtime installation rules |
| CL19 will improve boundary visuals without reducing identity | medium-low | architectural hypothesis; matched-step results are not yet available |

## 9. What is not established

- No CL19 identity, prompt-adherence, or face-quality gain is claimed here.
- CL19 has not yet been compared with CL14 at a matched checkpoint in this
  report.
- It is not established that boundary routing is Marion's dominant remaining
  failure.
- It is not established that CL19 improves glasses, tears, hands, or goggles
  inside the face core.
- Runtime speed is not evidence of better or worse architecture quality.

## 10. Existing experiment and decision gates

**Config:** `src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml`  
**Single scientific change:** Replace CL14's thresholded/double-applied target
mask with two full attention lanes and one deterministic two-cell cosine blend.  
**Hypothesis:** A continuous post-attention handover improves edge integration
without sacrificing the reference-conditioned face core.  
**Prediction:** Better boundary visuals and no material overall or hard-case
identity regression at matched checkpoints.  
**Primary risk:** The two rings consume too much of a small face and dilute
reference identity.  

Decision gates:

1. At matched `2k` and `4k`, require `96/96` face detection and no obvious new
   seam or face/body alignment failure.
2. Treat overall `id_sim_subject_v2` regression greater than `0.01` versus CL14
   as a stop signal unless the predefined hard-case slices show a compelling,
   visually verified tradeoff.
3. Inspect Marion, Crying, Skiing, Jumping, and Dancing separately. Aggregate
   identity alone is insufficient.
4. At `24k`, promotion requires a hard-case or boundary-quality gain with no
   material prompt-adherence or small-face regression.

## 11. Reproducing the architecture audit

Run from `diffusion_template/` in the `photomaker` environment:

```bash
python tools/validate_CL15_CL20_config.py

rg -n "soft_router|_soft_router_mask|_full_target_lanes|pose_adapt_ratio|CA_MIXING_FOR_FACE" \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/model/photomaker_branched/_old3/attn_processor_clean.py \
  src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml
```

The configuration validator also checks that CL19 preserves:

- `pose_adapt_ratio` equal to `0`;
- `ca_mixing_for_face` equal to `false`;
- the CL14 optimizer, loss, and validation contract;
- the fixed manual-val-96 protocol.

## 12. References

1. `src/model/photomaker_branched/attn_processor_cleanest.py` - current hard-v1
   processor, CL14 legacy route, CL19 router, and pose-adaptation equation.
2. `src/model/photomaker_branched/branched_runtime.py` - processor installation,
   group routing, and fail-closed project invariants.
3. `src/model/photomaker_branched/_old3/attn_processor_clean.py` - historical
   `ca_mixing_for_face` implementation.
4. `src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml` - CL14 delta.
5. `src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml` - CL19 delta.
6. `analysis/2026-08-11_cl14_hard_cases_architecture_research_and_experiment_plan.md`
   - prior hard-case architecture review and experiment rationale.
