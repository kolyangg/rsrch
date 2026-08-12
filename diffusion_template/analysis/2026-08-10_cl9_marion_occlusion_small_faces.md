# CL9 edge cases: Marion reference pose, occluder ownership, and the small-face resolution floor

**Date:** 10 August 2026  
**Run:** `CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1`  
**Immutable Comet experiment:** [`81bb311ed70545eda3281c64bc48be47`](https://www.comet.com/nikolay-2104/aug-large-ds/81bb311ed70545eda3281c64bc48be47)  
**Checkpoint and panel:** exact step `24,000`, fixed 96-image `manual_val`, one image per item  
**Scope:** Marion's off-axis reference; objects crossing the face, with Crying and
Skiing examined separately; and Jumping/Dancing small faces. This report refreshes
CL9 from its immutable Comet key, joins the current subject-v2 ID table to the
current face-quality table, reruns face-to-mask alignment with CL9's run-specific
boxes, inspects the active attention route, and converts the findings into gated
validation and training experiments. No training, validation, checkpoint, prompt,
seed, reference, scheduler, or metric definition was changed for the measurements
reported here.

Evidence labels used below are **[measured]**, **[visual]**, **[code]**,
**[prior audit]**, and **[hypothesis]**. A claim without an intervention is not
presented as causal.

| Fixed CL9 contract | Value |
|---|---|
| Validation model | `SG161222/RealVisXL_V4.0` |
| Validation loading | `legacy_full_copy`, shadow pretrained PhotoMaker default |
| Batch / CFG / sampler length | `12` / `5` / DDIM, `50` steps |
| BA route | hard spatial self-attention BA rank `128`, generic rank `32`, PhotoMaker default rank `64` |
| Trainable inventory | `2,240` tensors / `219,217,920` parameters |
| Required reference route | `use_branched_attention=true`, `pose_adapt_ratio=0`, `ca_mixing_for_face=false` |
| CL9 scientific change | training reference face area sampled over `[0.06, 0.30]`, edge fill, position jitter `0.15` |

---

## Executive conclusion

The three reported problems are real, but they are not one failure and should
not be attacked with one loss or one metric.

1. **Marion is a reference-robustness edge case, but reference pose is not yet a
   proven cause.** Her reference has a `-7.65` degree eye-line roll and a yaw
   proxy of `0.368`, versus at most `0.093` for the other seven panel identities.
   It is visually a clear 3/4 portrait. Marion's current CL9 mean is `0.3112`
   across 12 prompts and `0.3653` on six clean prompts, so her weakness is not
   confined to occlusion or small faces. However, six scoring embeddings rebuilt
   from the same file do not repair her: the best is roll correction at only
   `+0.0091`. **[measured]** This rules out a metric-side shortcut. It does not
   test what a roll-corrected image does when used as generation conditioning.
   That paired 12-image generation sidecar is the correct next step. A 2D roll
   correction is easy; frontalizing a 3/4 view without another photograph would
   hallucinate missing facial evidence and is not recommended.

2. **Crying and Skiing are different occlusion regimes.** Crying remains close
   to clean CL9 on current mask-owned ID (`0.473` versus `0.495`) and its prior
   within-identity retention loss is only `-0.124`: the right person is mostly
   present, but hands, hair, and closed eyes hide the features ArcFace reads.
   Skiing has the largest, sharpest faces in the panel (`316 px`, TOPIQ-Face
   `0.774`) yet much lower identity (`0.362`). Marion and Jisoo collapse to
   within-identity retention `0.243` and `0.191` when goggles cover their eyes,
   while the other six usually place goggles on the forehead and retain identity.
   **[measured, visual]** Skiing is a routing/composition target; Crying is mainly
   an evaluation and topology target.

3. **Small faces are an absolute resolution floor, not face-box underfill.** The
   current Jumping/Dancing faces average `120.9 px` in 1024 outputs, or `15.1`
   VAE-latent cells before deeper U-Net downsampling. They score `0.542` on
   TOPIQ-Face and `0.339` on identity, versus `0.696` and `0.495` on clean prompts.
   Yet all `96/96` detected faces fill the requested CL9 boxes: median rendered-to-
   requested size ratio `1.012`, minimum `0.868`, zero below the `0.8` undersized
   threshold, and median IoU `0.909`. **[measured]** Increasing the face box would
   change composition rather than solve the requested edge case. The likely fix
   needs extra local compute or scale-matched identity conditioning while keeping
   the final face small.

The first work should therefore be three fixed-checkpoint validation sidecars:
Marion same-file conditioning roll, occluder-aware branch ownership, and a
small-face local refinement probe. Training starts only when those interventions
show the expected mechanism.

| Issue | What is established | What is not established | First decision |
|---|---|---|---|
| Marion | Off-axis outlier; scoring normalization is ineffective | Off-axis conditioning causes the generation deficit | Run exact 12-image conditioning-side roll sidecar |
| Crying | Identity mostly survives; metric visibility falls | A model change would improve the intended image | Report separately; inspect hand/face topology |
| Skiing | Large, high-quality faces lose identity when goggles occupy the eyes | Native ownership of occluder pixels fixes it | Fixed-checkpoint ownership intervention |
| Small faces | Output fills the requested mask but has only 9-15 face cells at important resolutions | ROI refinement preserves identity and seams | Fixed-checkpoint local-resolution intervention |

---

## 1. Evidence, current metrics, and comparability

### 1.1 Current CL9 endpoint

The Comet export resolved an exact image step of `24,000`, with `96` images,
zero download warnings, and zero errors. The current subject-v2 repair changes
the identity reference and regenerated outputs for Eddie; it does not alter the
pixels for Marion or any other identity. All family summaries in this report use
the current 96-row subject-v2 ID table and current face-quality table unless a
row is explicitly marked as a prior embedding audit.

| Endpoint metric | CL9 at 24k |
|---|---:|
| `manual_val/id_sim` - current mask-owned subject-v2 | **`0.447997`** |
| `manual_val/id_sim_legacy_best` | `0.399063` |
| `manual_val/id_sim_mask_iou` | `0.895067` |
| `manual_val/id_sim_face_count` | `1.0833` |
| no-face / ambiguous / unowned | `0 / 0 / 0` |
| face detection rate | **`1.000`** |
| TOPIQ-Face mean / p10 | `0.679632 / 0.580499` |

The current mask-owned score is the promotion metric. The legacy best-face score
is retained only for historical comparison because it can select a bystander.
For an object covering the eyes, neither number answers whether the model drew
the same underlying person. For that question this report also uses
within-identity retention from the prior embedding audit:

```text
retention(generation) = cosine(generation embedding,
                               same identity's clean-prompt centroid)
```

The subject-v2 backfill did not alter generated pixels, so those generation-to-
generation comparisons remain valid. Retention is an issue metric, not a
replacement for the fixed panel metric.

### 1.2 Data joins and geometry refresh

The 96-row tables were joined by `output_key == file_name`, never by row order.
The refreshed face geometry used CL9's active generation box map, SHA256
`b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`.
The measurement manifest records hashes for the downloaded tables, Marion's
reference, and the alignment result:

- [measurement manifest](assets/cl9_edge_cases_20260810/data/measurement_manifest.json)
- [current family summary](assets/cl9_edge_cases_20260810/data/cl9_family_summary_current.csv)
- [current Marion prompt rows](assets/cl9_edge_cases_20260810/data/cl9_marion_prompt_rows_current.csv)

### 1.3 Execution limitation

The cheap Marion **scoring-side** retry was completed locally. The generation-
side retry was not launched during this analysis. At the execution check, the
project already had an eight-A100 Serv recovery job running under a separately
scoped ten-GPU exception; this report does not extend that authorization. The
local 16 GB GPU cannot safely reproduce CL9's exact batch-12 validation contract.
Running Marion at batch 1 would be a confound because this repository has already
observed validation differences from changed batch semantics. The exact sidecar
is fully specified in Section 5.1 and should run unchanged when one A100 is
available.

---

## 2. Marion: pose outlier, weak identity, no metric-side shortcut

### 2.1 What is unusual about the same reference image

Local InsightFace `antelopev2` detects one face in the unchanged 500 x 600 file,
with box `[239.0, 99.7, 442.8, 415.9]`. Its five landmarks give:

| Reference property | Marion | Other seven panel identities | Interpretation |
|---|---:|---:|---|
| eye-line angle in image coordinates | `-7.65 deg` | not the main outlier | easy 2D roll correction |
| yaw proxy `abs(nose_x - eye_mid_x) / eye_distance` | **`0.368`** | `<=0.093` | strong 3/4-view outlier |
| local training-reference percentile | about `97th` | - | underpowered: only `n=38` references |
| detected faces | `1` | - | not a multi-face selection bug |

The yaw percentile is only low-to-medium confidence because the local sample is
small. The pose measurement itself is high confidence. The causal statement
"Marion is weak because the reference is off-axis" remains **not established**.

![Marion reference, deterministic roll preview, and representative CL9 outputs](assets/cl9_edge_cases_20260810/fig_marion_reference_and_outputs.png)

The right reference panel is a deterministic preview, not a generated validation
result. It levels the eyes but visibly retains the 3/4 view. It neither estimates
a missing cheek and eye nor changes the subject to a synthetic frontal portrait.

### 2.2 Marion is weak even when nothing covers the face

Marion's current mean mask-owned ID is `0.3112` over all 12 prompts, compared
with the full-panel mean `0.4480`. Her six clean prompts average `0.3653` versus
`0.4955` over all identities on the same clean family. That clean deficit is the
main reason reference conditioning deserves a direct test.

| Prompt | ID | TOPIQ-Face | face short side |
|---|---:|---:|---:|
| Reading | `0.382` | `0.623` | `189 px` |
| Rushing | `0.313` | `0.678` | `188 px` |
| Skiing | **`0.221`** | `0.767` | `358 px` |
| Drumming | `0.365` | `0.653` | `184 px` |
| Kickboxing | `0.344` | `0.691` | `211 px` |
| Dancing | `0.391` | `0.616` | `153 px` |
| Angry | `0.340` | `0.685` | `194 px` |
| Crying | `0.370` | `0.641` | `217 px` |
| Laughing | **`0.105`** | `0.649` | `181 px` |
| Jumping | **`0.111`** | `0.587` | `127 px` |
| Night-ride | `0.419` | `0.641` | `209 px` |
| Chef | `0.372` | `0.728` | `218 px` |

The prompts stack with the reference weakness: Skiing adds goggles, Jumping adds
the resolution floor, and Laughing removes stable periocular geometry. Dancing
at `0.391` also shows that small size alone does not determine every sample.

The broader cross-arm rank audit found Marion's own reference was rank 1 for
`83/84` detected generations. **[prior audit]** This is weak fidelity, not a
systematic confusion with another panel identity.

### 2.3 Six same-file scoring normalizations

The 12 existing Marion generations were rescored against six embeddings rebuilt
from the same file. This changes only the measurement vector; it does not
regenerate an image.

| Scoring reference variant | mean cosine | delta from raw |
|---|---:|---:|
| raw rebuilt vector | `0.3013` | - |
| horizontal flip | `0.2975` | `-0.0038` |
| **roll fixed** | **`0.3103`** | **`+0.0091`** |
| multi-crop | `0.3084` | `+0.0071` |
| flip average | `0.3025` | `+0.0013` |
| full TTA | `0.3068` | `+0.0055` |

This paired audit has its own rebuilt raw baseline (`0.3013`); it should not be
numerically substituted for the current subject-v2 panel mean (`0.3112`). The
within-audit deltas are the result. Keanu gained more from ordinary multi-crop
TTA (`+0.0165`), so Marion's `+0.0091` is not a pose-specific repair.

**Conclusion:** scoring-side normalization is ruled out as a useful fix. A
generation-side test is still justified because the BA reference latents and
PhotoMaker conditioning see image pixels, not only the final scoring embedding.

### 2.4 What an acceptable same-image retry must preserve

The first intervention should correct only roll. It must keep the original 500 x
600 canvas, content scale, subject, and declared face occupancy. A tight crop in
the same arm would confound pose with reference scale, which CL9 was explicitly
trained to control. The affine-transformed box must be propagated and then
checked by re-detection. The original subject-v2 vector remains the scoring
target, otherwise generation and scoring effects are mixed.

3D frontalization, face swapping, or a generative restoration model is not an
"easy normalization" of the same evidence. Those methods invent the hidden side
of a 3/4 face. They can be a labeled augmentation experiment later, but they
cannot establish that CL9 handles the supplied edge-case photograph.

---

## 3. Objects over the face: Crying is not Skiing

### 3.1 Current family profile

![Current CL9 identity, face quality, and absolute face size by family](assets/cl9_edge_cases_20260810/fig_cl9_family_profile.png)

| Family | n | current mask-owned ID | TOPIQ-Face | face short side |
|---|---:|---:|---:|---:|
| Clean: six prompts | `48` | **`0.495`** | `0.696` | `208 px` |
| Crying | `8` | `0.473` | `0.666` | `203 px` |
| Skiing | `8` | **`0.362`** | **`0.774`** | **`316 px`** |
| Small-face: Jumping/Dancing | `16` | **`0.339`** | **`0.542`** | **`121 px`** |

Skiing falsifies a simple "the face is too small or blurry" explanation. It has
the largest and highest-quality detected faces but loses identity. The within-
identity audit makes the split sharper:

| Prompt regime | mean retention change from own clean centroid | Reading of the failure |
|---|---:|---|
| Crying | `-0.124` | underlying person largely survives |
| Skiing | **`-0.322`** | model changes the person on hard cases |
| Small-face | **`-0.292`** | identity and quality both decay |

For Skiing, Marion falls from a clean retention baseline `0.783` to `0.243` and
Jisoo from `0.899` to `0.191`. The other six retain `0.514-0.823`. **[prior
audit]** The visual split is exact in this eight-image sample: the six stronger
cases keep goggles on the forehead or keep the eyes readable; Marion and Jisoo
put the lenses over the eyes and acquire duplicated or generic periocular
geometry.

![CL9 Skiing and Crying for all eight identities](assets/hardcase_20260809/fig9_cl9_occlusion.png)

### 3.2 Occluder type matters

**Ski goggles.** The eye region is an opaque, prompt-required, high-contrast
object. The measured eye-patch saturation excess is `0.194` and contrast ratio
`1.385`. The model must decide whether the goggles go over the eyes, on the
forehead, or become fused with facial structure. On the two weak identities it
uses the branch-owned face region to synthesize both object and identity and
loses the latter. **[measured, visual]**

**Hands and hair in Crying.** The eye-patch saturation excess is only `0.023`
and contrast ratio `0.964`. Skin-toned hands, hair strands, compressed cheeks,
tears, and closed eyelids are not separable by a simple color threshold. ArcFace
also loses its strongest landmarks even when the person beneath is correct.
The useful checks are hand/face topology, duplicate eyes or mouths, object-edge
continuity, and retention. Chasing Crying `id_sim` alone risks removing the hands
or opening the eyes, which would violate the prompt.

**Ordinary glasses and hair.** These sit between the two regimes. Transparent
glasses should preserve identity evidence through the lenses; opaque sunglasses
should be judged on graceful identity retention and object geometry, not an
unattainable unobstructed ArcFace score. Hair is thin and spatially irregular, so
a learned or parsed visibility mask is more plausible than a rectangular eye
band. The validation diagnostic below therefore tests branch ownership first,
before choosing a production parser or gate.

### 3.3 Why the current hard branch is vulnerable

The active processor makes ownership explicit:

1. Target/native background queries are `q * (1 - mask_gate)` and target/native
   attention produces `hidden_bg`.
2. Face queries are `q * mask_gate`.
3. Face K/V come from the reference face mask. With `pose_adapt_ratio=0`, target
   face K/V are not substituted.
4. The result is a hard spatial merge:
   `hidden_bg * (1 - mask) + hidden_face * mask`.

See [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py),
especially lines 311-315, 365-390, 408-429, and 460-475. The pipeline constructs
and passes the generation and reference masks in
[`br_pipeline_helpers.py`](../src/pipelines/br_pipeline_helpers.py), lines
714-717. **[code]**

The target face mask is the full face rectangle, not a visible-surface mask.
Consequently a goggle lens, hand, or hair strand inside that rectangle is owned
by the reference branch even though the object belongs to the target prompt and
native scene. This code fact plus the visual split supports the ownership
hypothesis, but the causal fix remains unproven until the fixed-checkpoint mask
intervention in Section 5.2.

The eligible fix must preserve the BA idea: target queries continue to receive
explicit reference K/V on visible facial surface. Setting
`pose_adapt_ratio > 0`, turning on `ca_mixing_for_face`, or replacing BA with a
generic adapter would not test the project's reference-conditioned mechanism.

---

## 4. Small faces: requested composition, insufficient local resolution

### 4.1 CL9 already fills the requested boxes

The current 96-image alignment refresh gives:

| Geometry check | CL9 at 24k |
|---|---:|
| detected faces | `96/96` |
| rendered/requested short-side ratio, median | **`1.012`** |
| ratio, minimum | `0.868` |
| ratio below `0.8` | **`0`** |
| mask IoU, median | **`0.909`** |
| mask IoU below `0.3` | `0` |

CL9's original scale-calibration objective succeeded. The remaining small-face
problem is that the requested Jumping/Dancing boxes themselves are small. The
detected faces average `120.9 px`, range from `88.7` to `153.4 px`, and are
correctly placed inside those boxes.

### 4.2 Where the information disappears

At 1024 output resolution, a `120.9 px` face is about:

| Representation | approximate face short side |
|---|---:|
| output pixels | `120.9` |
| VAE latent, 8x downsample | `15.1` cells |
| next 2x U-Net scale | `7.6` cells |
| next 2x U-Net scale | `3.8` cells |

At 4-8 cells, eyes, nostrils, mouth corners, and hair boundaries cannot all be
represented independently. This matches the two measured losses: TOPIQ-Face
drops by `-0.154` from clean and current identity drops by `-0.157`. The prior
own-clean retention loss of `-0.292` shows that this is not only a no-reference
quality artifact.

The pooled five-arm analysis found `corr(face_px, TOPIQ-Face) = +0.824` at 24k.
The raw identity correlation was much smaller because very large Skiing faces
sit at the opposite extreme and lose identity for an unrelated reason. Prompt-
family separation is therefore required.

### 4.3 What is unlikely to solve it

- **Inverse-area weighting:** the active face loss already uses mean MSE over
  each face crop before averaging samples. The number of face pixels does not
  dilute the per-sample loss; see
  [`diffusion_loss.py`](../src/loss/diffusion_loss.py), lines 7-32. **[code]**
- **Generic small-face oversampling:** the audited large training dataset had
  `48.6%` of targets below 3% face area versus `17.7%` in the panel. Small targets
  are not scarce. **[prior audit]** More of the same samples does not add spatial
  bandwidth.
- **Making the requested box larger:** that changes pose and full-body
  composition, so it no longer answers whether the model can render an important
  small-face edge case.
- **A second identity loss with no new resolution path:** E16's predicted-x0 ID
  auxiliary loss underperformed its own base by about `0.06`. A new loss must
  expose additional usable face detail or a different route, not just add weight.

Two mechanisms remain plausible. First, match the spatial reference scale to
the target so the reference branch presents identity at the granularity needed
for that sample. Second, give the face a local high-resolution refinement path
while compositing back into the unchanged full-body output. The validation probe
should establish the second mechanism before training it.

---

## 5. Validation experiments: fixed checkpoint before new training

Every sidecar begins with an exact historical replay and preserves CL9's batch
of `12`, RealVis validation base, CFG `5`, DDIM `50`, seeds, prompts, scheduler,
generation boxes, reference boxes, and step-24k checkpoint. A replay is valid
only when all expected baseline RGB hashes match. Each sidecar writes a command
manifest, checkpoint hash, transformed-input hashes, box hashes, per-image table,
and environment snapshot. If it is Comet-tracked, startup must also produce a
new immutable `comet_experiment.json`.

### 5.1 V1 - Marion same-file conditioning roll, priority 1

**Name:** `CL9V_marion_samefile_roll_24k_r1`  
**Single change:** the conditioning image for Marion is rotated by the measured
`-7.6476` degree eye-line angle about the eye midpoint, with edge-replicate fill.
The original dimensions and scale are retained. The scoring reference and all
other inputs remain original.

**Implementation.** Build a one-image sidecar reference directory from
`references/marion.jpg`. Apply one affine matrix to image pixels and the four
corners of `face_crop_new`, clip the propagated box, re-detect the face, and
record both boxes and hashes. Select manual-val indices `84-95` as one unchanged
batch of 12. First replay the original reference and require 12/12 RGB matches;
then substitute only the rotated conditioning file. Do not use a tight crop in
this arm. Do not rebuild the scoring vector.

**Hypothesis:** the off-axis pixel geometry degrades PhotoMaker/BA conditioning
even though the scoring embedding is stable. **[hypothesis]**

**Prediction:** Marion current mean ID rises from `0.3112` by more than `0.03`,
with at least `9/12` paired prompt wins. The gain should appear on clean prompts,
not only Skiing or Jumping.

**Risk:** roll is not the issue; yaw and missing evidence remain. A positive
result could also be limited to easier face detection. Re-detection telemetry and
unchanged scoring prevent that from being mistaken for an identity gain.

**Decision gate:** promote a defaults-off conditioning roll-normalizer only if
mean ID gain is `>0.03`, at least `9/12` prompts improve, mean TOPIQ-Face falls by
less than `0.02`, and generation-mask IoU does not regress. Otherwise retain this
photograph as the unresolved edge case and move to the planned second real
reference comparison. A separate five-point similarity crop is allowed only as
a second labeled arm because it changes scale as well as roll.

### 5.2 V2 - occluder-aware branch ownership, priority 2

**Name:** `CL9V_occluder_ownership_24k_r1`  
**Single change:** inside the target face box, a frozen visibility mask replaces
the rectangular target branch mask. Visible facial surface remains reference-
branch owned; high-confidence occluder pixels are assigned to the existing native
target lane. Reference K/V, target Q, reference mask, and all weights are
unchanged.

**Implementation.** Replay the full 96-image panel, preserving each identity's
12-item batch. Build a frozen first-pass mask for the 16 Skiing/Crying items from
the exact baseline output: a face parser plus manually reviewed binary alpha for
goggles, hands, and hair. Rerun the same seeds with this mask fixed in image
coordinates. The other 80 prompts receive the original rectangle and act as
pixel-replay sentinels. Extend `prepare_gen_mask` to accept an optional
`ba_target_visibility_mask`; downsample it identically at every attention
resolution. In the current hard processor, excluded pixels naturally move from
`q_face` to `q_bg`, which is the intended fixed-checkpoint causal test.

This is an oracle diagnostic, not the production masking solution. Its job is to
answer whether ownership is the right mechanism before training a parser or
gate.

**Hypothesis:** opaque target objects fail because the reference lane exclusively
owns every pixel in the rectangular face box. **[hypothesis from code and visual
evidence]**

**Prediction:** Marion and Jisoo Skiing retention rise from `0.243` and `0.191`
to at least `0.40`; mean Skiing retention loss improves from `-0.322` to better
than `-0.20`. The other six Skiing identities do not lose more than `0.05`.
Crying ID need not rise; its hand/face topology and object continuity must not
worsen.

**Risk:** the second pass can move the object away from the first-pass mask;
native ownership can also erase identity beneath transparent lenses. Measure
final-mask overlap, reject masks below `0.7` IoU with the final parsed occluder,
and stratify transparent versus opaque eyewear.

**Decision gate:** advance to T1 only if both weak Skiing identities exceed
`0.40` retention, the six stronger cases stay within `-0.05`, clean replay
sentinels are exact, face detection remains `1.0`, and no new duplicated-eye or
mask-boundary artifact appears in blind review.

### 5.3 V3 - small-face local-resolution probe, priority 3

**Name:** `CL9V_smallface_roi_refine_24k_r1`  
**Single change:** after the exact CL9 first pass, Jumping/Dancing receive one
deterministic 2x-resolution face-ROI denoising/refinement pass using the same
prompt, same reference K/V route, and the same checkpoint. The result is
downsampled and feathered back only inside an expanded face ROI. Final face size
and full-body composition remain unchanged.

**Implementation.** Run the full 96 panel so batch semantics and 80 sentinels are
preserved. For the 16 small-face items, expand the declared face box by `1.5x`,
map the corresponding latent/noise crop, process it at a minimum `256 x 256`
image-space working size for a fixed late-step window, then composite it back
with a logged cosine feather. Use the original CL9 reference mask and explicit
reference K/V in the ROI branch. Outside the expanded ROI, the output must be
pixel-identical to baseline.

**Hypothesis:** the dominant deficit is local spatial bandwidth rather than
identity supervision. **[hypothesis]**

**Prediction:** small-face TOPIQ-Face rises from `0.542` to above `0.60`, current
small-face ID rises from `0.339` by at least `0.04`, and the detected face remains
within `0.9-1.1` of the same requested size.

**Risk:** paste seams, pose changes, over-sharpening, and a face that looks like a
different exposure from the body. Measure boundary color difference, outside-ROI
pixel equality, prompt CLIP, face/body alignment, and blind full-frame review.

**Decision gate:** train an ROI mechanism only if TOPIQ-Face is `>0.60`, ID gain
is `>=0.04`, outside-ROI pixels are exact, median mask IoU stays within `0.02` of
baseline, and no boundary metric or visual review regresses.

---

## 6. Training experiments, gated by the validation results

All new training arms retain CL9's dataset, 24k optimizer steps, batch 2, one
A100, step 0 plus every-2k full-96 validation, prompts, seeds, references,
scheduler, and metric definitions. They keep `pose_adapt_ratio=0` and
`ca_mixing_for_face=false` in training and validation. Defaults remain backward
compatible. An arm that adds a gate or ROI module will intentionally change the
trainable inventory; its exact tensor and parameter counts must be recorded at
preflight rather than falsely reported as CL9's `2,240 / 219,217,920`.

### 6.1 T1 - visibility-supervised native/reference face gate

**Config:** `CL9_occlusion_gate_visible_surface_24k_r1.yaml`  
**Gate:** run only if V2 passes.  
**Single scientific change:** add a bounded per-pixel selector between a full
target-native self-attention lane and the existing reference-only face lane.
Synthetic occluder alpha supplies visibility supervision; the reference lane
still uses explicit reference K/V.

For target query `q`, compute:

```text
A_face = g_visible * A_reference + (1 - g_visible) * A_native
```

The gate target is near 1 on visible face surface and near 0 on the known alpha
of pasted glasses, hair, and hand cutouts. On unaugmented samples the target is
the existing face rectangle, so reference ownership remains the default. Log
gate mean separately for visible skin, opaque occluder, transparent lens, and
background. A visibility BCE term prevents the prior failure mode where a
mixture gate simply retreats to the native model everywhere.

**Implementation map.** Add a defaults-off processor/version in
`src/model/photomaker_branched/`; install and inventory it through
`lora2_helpers.py`; serialize its architecture flags and gate weights through
`lora2.py`; and return `face_visibility_mask` plus synthetic alpha from the CL9
dataset path. Target Q is unchanged. Reference K/V and reference mask are
unchanged. Only the target merge becomes visibility-conditioned. Add one concise
dated invariant comment beside the merge when implemented.

**Prediction:** by 14k, Marion/Jisoo Skiing retention exceeds `0.40`; by 24k,
mean Skiing retention loss is better than `-0.20`; clean-family ID remains within
`0.01` of CL9; Crying topology does not regress.

**Risk:** synthetic-to-real domain gap, transparent-lens ambiguity, a gate that
learns object color rather than visibility, or collapse to native attention.

**Gates:** 4k - visible/occluder gate telemetry separates by at least `0.5` and
the reference contribution on unoccluded face remains `>=0.85`; 8k - clean ID
within `0.01` of matched CL9 and mask IoU within `0.02`; 14k - both weak Skiing
retentions `>0.40`; 24k - current full-panel ID no worse than CL9 by `0.01`,
Skiing improvement retained, and no Crying topology regression.

### 6.2 T2 - target-matched reference scale with position jitter retained

**Config:** `CL9_refscale_targetmatch_24k_r1.yaml`  
**Single change:** replace CL9's uniform reference face-area draw `[0.06, 0.30]`
with the target box's own scale. Keep edge fill and position jitter `0.15`
unchanged.

This path already exists. In
[`cosmic_large_adapted.py`](../src/datasets/cosmic_large_adapted.py), lines
469-517, `reference_scale_jitter=None` makes
`compose_target_frame_reference` match the target short side; CL9 currently
sets a random requested area fraction. The arm can therefore be a localized
config change:

```yaml
datasets:
  train:
    cosmic_large_adapted:
      reference_scale_jitter: null       # target scale, not a uniform draw
      reference_position_jitter: 0.15    # unchanged from CL9
```

This revisits CL2 without repeating its full degeneracy: CL2 locked both scale
and position, while this arm retains CL9's position jitter.

**Hypothesis:** small target faces need reference features presented at matched
granularity; a uniform reference-scale distribution weakens that per-sample
correspondence. **[hypothesis]**

**Prediction:** small-face ID reaches at least CL8's historical `0.3804`, from
current CL9 `0.3389`; TOPIQ-Face improves modestly above `0.56`; the full-panel
current ID stays within `0.01` of CL9.

**Risk:** the CL2 identity collapse recurs, or scale matching helps identity but
cannot overcome base-model resolution. The latter is informative and would
advance T3 rather than justify more sampling changes.

**Gates:** step 0 - processor install, trainable inventory, and panel replay
match CL9; 4k - undersized count stays zero; 8k - full ID is not more than `0.02`
below matched CL9; 14k - small-face TOPIQ-Face is `>0.56`; 24k - small-face ID
`>=0.380`, full ID within `0.01`, and clean/Skiing families do not regress.

### 6.3 T3 - high-resolution ROI BA branch

**Config:** `CL9_smallface_roi_branch_24k_r1.yaml`  
**Gate:** run only if V3 passes and T2 leaves TOPIQ-Face below `0.60`.  
**Single scientific change:** add a local high-resolution face feature path for
targets whose declared face short side is below `160 px`; route the same target Q
against explicit reference K/V in a zoomed ROI and scatter a bounded residual
back into the high-resolution U-Net feature map.

Training uses the same source image twice: the normal 1024 full frame for global
diffusion and a high-resolution crop around its declared face for the ROI target.
No new identity reference is introduced. The ROI branch must be disabled above
the threshold and defaults off for all old configs. Its checkpoint state and
site allowlist must be installed identically in training and validation.

**Prediction:** small-face TOPIQ-Face `>0.60`, small-face ID gain `>=0.04`,
rendered/requested size remains in `0.9-1.1`, and full-frame text similarity is
unchanged.

**Risk:** local/global seams, double features, excessive face sharpening, and a
new trainable path too large for clean causal comparison.

**Gates:** preflight - exact site list and parameter inventory; 4k - outside-ROI
feature residual is numerically zero and clean replay is unchanged; 8k - no
boundary or mask-IoU regression; 14k - small-face TOPIQ-Face `>0.58`; 24k - V3's
quality and identity gates are met on the canonical panel.

### 6.4 Conditional Marion training arm

Do not launch a Marion-specific training run before V1. If V1 passes, use
`CL9_reference_roll_canonicalized_24k_r1.yaml`: the single change is the same
deterministic five-point roll canonicalization in both training and validation
conditioning, with affine box propagation, fixed canvas scale, and defaults off.
First audit the full training reference roll distribution so a rare Marion case
does not silently become a broad data transform. If V1 fails, wait for the
planned second real Marion reference; do not train on synthetic frontalizations
of the same photograph.

---

## 7. Implementation and execution plan

### 7.1 Code map

- **Marion input transform:** implement in a sidecar tool, touching
  `src/datasets/manual_val.py` only if V1 is promoted. The image and bbox must
  share one affine, while the original scoring vector remains unchanged.
- **Validation visibility mask:** add the defaults-off input in
  `src/pipelines/br_pipeline_helpers.py`. Only target ownership may change;
  reference mask and reference K/V remain unchanged.
- **Occlusion gate:** add a new processor under
  `src/model/photomaker_branched/`, install it through `lora2_helpers.py`, and
  serialize its state/config in `lora2.py`. Target Q stays unchanged, explicit
  reference K/V remains, and native/reference contribution is logged.
- **Target-scale match:** use a CL9 child config and the existing
  `cosmic_large_adapted.py` plus `reference_frame.py` path. Position jitter stays
  `0.15`, and the cache descriptor remains target-dependent.
- **ROI refinement:** implement the validation sidecar first, then a defaults-off
  processor/module and checkpoint installer only if V3 passes. The residual is
  zero outside the ROI and the train/validation site list is identical.
- **Metrics:** join the current subject-v2 ownership, retention, TOPIQ-Face, IoU,
  and face-pixel rows by image key in the validation table and face-quality
  sidecar.

### 7.2 Safe order of work

1. **Freeze evidence.** Preserve the CL9 checkpoint hash, current Comet tables,
   run-specific box map hash, reference hash, batch-12 order, and exact replay
   images in each sidecar manifest.
2. **Run V1 first.** It is only 12 images and decides whether an input-only
   Marion solution is credible.
3. **Implement one defaults-off arbitrary target visibility mask and run V2.**
   Do not build a learned parser or gate until the oracle ownership intervention
   succeeds.
4. **Run V3 using the same fixed checkpoint.** It establishes whether local
   resolution can solve the small-face artifact without changing composition.
5. **Promote only passed mechanisms to training.** T2 is the lowest-risk config
   arm. T1 and T3 require exact processor-install, checkpoint-load, and trainable-
   inventory preflights in both old and new modes.
6. **Use the canonical training cadence.** Step 0 plus every 2,000 optimizer
   steps on all 96 items; `epoch_len=2000`; one image per item. Stop only on the
   stated matched-step kill rules.
7. **Respect live Serv capacity.** Inspect this project's Running and Pending
   one- and two-GPU requests before every submission. No retry should exceed the
   normal six-A100 ceiling without a new, experiment-specific authorization.

### 7.3 Focused verification, not a new test suite

- Compose every new Hydra config and its parent CL9 config.
- Compile changed Python and run one processor-install smoke pass in old and new
  modes.
- Load the unchanged 24k checkpoint and verify the expected missing/unexpected
  key sets are empty for fixed-checkpoint sidecars.
- Verify old-mode RGB replay before interpreting an intervention.
- Log target mask, reference mask, native contribution, reference contribution,
  and merge output at one fixed timestep and attention site.
- For trainable gates/modules, print the exact named trainable inventory and
  optimizer role groups before submission.
- Verify `pose_adapt_ratio=0` and `ca_mixing_for_face=false` from the resolved
  training and validation configs, not only the YAML source.

---

## 8. Metrics and decision rules

No single metric can promote all three fixes.

| Scope | Primary metric | Guard metrics |
|---|---|---|
| Full panel | current subject-v2 `manual_val/id_sim` | TOPIQ-Face mean/p10, detection, mask IoU, text similarity |
| Marion V1 | paired Marion mean ID and prompt win count | clean-prompt split, TOPIQ-Face, original scoring vector, box IoU |
| Skiing V2/T1 | own-clean identity retention | current ID, goggle placement, transparent/opaque stratum, topology |
| Crying | topology and own-clean retention | prompt adherence, current ID reported but not optimized alone |
| Small-face V3/T2/T3 | TOPIQ-Face on Jumping/Dancing | retention, current ID, face pixels, size ratio, boundary continuity |

Promotion guard for a training arm: full-panel current ID must remain within
`0.01` of CL9's `0.447997`, face detection must remain `1.0`, median mask IoU
must remain within `0.02` of `0.9088`, and no controlled visual regression may
be hidden by an average. For architecture arms, comparisons are matched at every
2k checkpoint, not only at 24k.

---

## 9. Confidence and open questions

- **High:** Marion's reference is a strong pose outlier within this panel. It has
  one detected face, stable landmarks, and yaw proxy `0.368` versus `<=0.093`.
- **Low-medium:** Marion is an extreme training-reference pose outlier. The
  estimate is about the 97th percentile, but the audit has only `n=38` references.
- **Ruled out:** scoring-side same-file normalization repairs Marion. Six paired
  embeddings give a best gain of only `+0.0091`.
- **Not established:** conditioning-side roll correction repairs Marion. The
  exact generation sidecar has not yet run.
- **High:** Crying usually preserves the underlying identity. Current ID is close
  to clean, the retention loss is only `-0.124`, and the visuals agree.
- **High:** Skiing is a genuine identity/composition failure for Marion and
  Jisoo. Retention is `0.243/0.191` despite sharp, large faces and an exact visual
  split.
- **Medium-high:** rectangular branch ownership contributes to Skiing. The
  hard-merge code and visual mechanism agree, but the intervention is pending.
- **Ruled out for CL9:** small-face underfill is the current problem. No size
  ratio is below `0.8`, and the median ratio is `1.012`.
- **High:** absolute small-face resolution limits quality and identity. The
  family averages `121 px`, TOPIQ-Face `0.542`, ID `0.339`, and retention change
  `-0.292`.
- **Medium-low:** ROI refinement will solve small faces without seams. The
  mechanism is plausible, but the fixed-checkpoint probe is pending.

The important unknowns are intentionally narrow: whether the same Marion pixels
condition better after roll correction; whether assigning occluders to the native
lane improves Skiing; and whether local extra resolution improves small faces
without a pasted-face look. The proposed sidecars answer those before another
24k run consumes GPU time.

---

## 10. Reproduction record

Run from `diffusion_template/`.

```bash
CL9_RUN=CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1
CL9_RECORD="comet_records/$CL9_RUN.json"
CL9_STEP_ROOT=comet_data/cl9_edge_cases_20260810/step_024000
CL9_PANEL_DIR="$CL9_STEP_ROOT/$CL9_RUN"

python tools/comet/comet_experiment.py fetch \
  --record "$CL9_RECORD" \
  --step-number 24000 \
  --output-dir "$CL9_STEP_ROOT"

python tools/datasets/measure_face_body_alignment.py \
  --images-dir "$CL9_PANEL_DIR" \
  --mask-boxes comet_data/cl9_edge_cases_20260810/pm96_bboxes_new_auto_cl9.json \
  --label 'CL9 24k current subject-v2' \
  --output comet_data/cl9_edge_cases_20260810/alignment_24k_current.json

python analysis/assets/cl9_edge_cases_20260810/build_assets.py
```

Downloaded asset hashes:

- Current subject-v2 per-image ID table:
  `af1980904509af4f3174c0d3ed83ed1febc4396dfaf875d18b60c2c0dd26dfcd`
- Current per-image face-quality table:
  `ee7fd56a89a5b048bf0cbaec23422849419899a14bde3499badf1aa2df291c2e`
- Unchanged Marion reference:
  `3884de5c8ca4c97840512c4976daa3cc79bb9e33eef4369c9b6ec93aed3f5a22`
- Refreshed alignment result:
  `a741b9e8b979dc5788d3916cd79e817a5c0b9a84fded975df886f26c829a3e38`

Related evidence:

- [Problematic prompts and identities root-cause report](2026-08-09_problem_prompts_identities_root_cause_and_e25_e27.md)
- [CL8-CL11 results and hard cases](2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md)
- [CL8/CL9 face-scale results](2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md)
- [Validation protocol](../docs/validation_protocol.md)
- [CL9 config](../src/configs/CL9_cosmic_joint_shadow_sa128_refscale_24k.yaml)

---

## Recommended next action

When one A100 is available, run `CL9V_marion_samefile_roll_24k_r1` exactly as a
single batch of indices `84-95`, with an original-reference replay gate first.
In parallel only when capacity permits, implement the defaults-off arbitrary
target visibility mask needed for V2. Do not start a new 24k training arm until
at least one fixed-checkpoint intervention passes its mechanism-specific gate.
