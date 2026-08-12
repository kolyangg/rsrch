---
title: "Foreground-Eddie conditioning recovers identity while preserving composition"
subtitle: "Contract-v2 image-by-image validation before and after the subject-selection fix"
date: "9 August 2026"
status: "FINAL: exact historical replay and corrected Serv validation completed 9 August 2026"
---

> **Contract audit passed.** The earlier invalid sidecar has been replaced.
> RealVis, strict `legacy_full_copy`, the pretrained PhotoMaker-default shadow
> restore, full-96 dataset context and batch size 12 now match in-training
> validation. Before correction, E13, BC_E13 and CL11 each reproduced all 12
> historical Eddie PNGs exactly: 36/36 RGB pixel-identical, zero failed pairs,
> zero contract mismatches. The corrected arms below ran only after those
> gates passed. **[measured] [code]**

# Executive conclusion

Correcting Eddie's subject selection produces a large, consistent identity
gain without the composition collapse seen in the invalid sidecar. Intended
Eddie improves in **36/36** pairs. Mean gains are `+0.360`, `+0.291`, and
`+0.289` for E13 24k, BC_E13 24k, and CL11 20k. Median face/mask IoU remains
close to historical (`0.896 -> 0.891`, `0.904 -> 0.875`, `0.885 -> 0.880`),
and no corrected output falls below `0.30`; the minima are `0.733`, `0.500`,
and `0.684`. **[measured]**

The old screenshots were right to flag a pipeline inconsistency. In the valid
run, Kickboxing and Jumping keep the same body pose and face ownership while
identity improves. E13 Kickboxing is `0.115 -> 0.485` with IoU
`0.891 -> 0.922`; E13 Jumping is `0.193 -> 0.341` with IoU
`0.860 -> 0.875`. Residual issues are facial detail and scene semantics:
goggles/eyes in Skiing, mouths in Laughing, small Dancing faces, and extra
people in Chef. **[measured] [visual]**

The selector and versioned mask-owned metric should ship first. The next E13
model diagnostic should target face anatomy/occlusion with a bounded native
target/reference message mixture, followed by target-scale matching for small
faces. A PhotoMaker-onset sweep is no longer prioritized. **[inference]**

# Scope and controlled comparison

"Historical" means the saved validation image generated with the erroneous
background-bystander ArcFace embedding. "Corrected" means a new generation
from the same checkpoint with the intended foreground-Eddie embedding. These
are **generated before/after images**, not retouched images. Both sides are
scored against the same intended foreground Eddie vector.

| Arm | Dataset | Immutable Comet key | Saved step | Intended-ID pre -> post |
|---|---|---|---:|---:|
| E13 | large_dataset | `1cc0a02371094b24a6a02a4cc649f10c` | 24k | `0.0653 -> 0.4254` |
| BC_E13 | BigCelebs | `c138db7c41ae435c8a7560f40cf5f58d` | 24k | `0.0626 -> 0.3540` |
| CL11 | cosmic_large | `32f4ba2a3b3a493f96a3a2345147e84c` | 20k | `0.0741 -> 0.3633` |

Checkpoint SHA-256 values:

- E13: `4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab`;
- BC_E13: `99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e`;
- CL11: `e65972c8c14b5031f879e1ee8b1e11a707823e0cfccdb80553219fc8069dbb83`.

Generation-bbox protocol SHA-256 is
`4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099`
for E13/BC_E13 and
`b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`
for CL11. Exact replay proves each matches its source runtime.

The intervention held reference pixels and bbox, prompts, seeds, fixed
generation masks, scheduler, inference steps, CFG and checkpoint fixed. Only
the selected ArcFace identity embedding changed. The BA spatial crop was
already foreground-Eddie-correct. All arms preserve
`use_branched_attention=true`, `pose_adapt_ratio=0`,
`ca_mixing_for_face=false`, and the checkpoint contract of `2,240` trainable
tensors / `219,217,920` parameters. **[code] [measured]**

The endpoint steps differ, so cross-model ranks are descriptive. The valid
causal comparison is **within each row**, historical versus corrected at the
same saved checkpoint. This is a 12-image Eddie diagnostic exception, not a
replacement for the standard fixed-96 validation event.

The primary identity measurement is intended-Eddie ArcFace cosine for the
detected face selected by greatest overlap with the fixed target mask. It
equals `IDSimBest` for all 72 images in this panel, but identity is always
reported with the selected face's mask IoU and face count. Historical logged
Eddie `id_sim` is invalid because its target is the bystander and is not used
for the paired comparison.

# 1. What the fix changes

![](assets/problematic_validation_20260809/eddie_reference_metric_error.png)

*Figure 1. Historical reference selection in red and intended foreground Eddie
in green. The historical stored vector matches the small right-edge bystander,
not Eddie.*

The reference contains two detected faces. Historical `faces[0]` selection
uses `[336, 136, 400, 257]`, while the intended/largest foreground face is
`[104, 0, 303, 291]`. The saved historical embedding has cosine `1.0000` to the
bystander and `-0.0078` to foreground Eddie. The corrected sidecar replaces
that vector only; it does not change the source image or the reference crop.
**[measured] [code]**

![](assets/problematic_validation_20260809/eddie_pre_post_metric_summary.png)

*Figure 2. Aggregate and prompt-level paired changes. Historical identity bars
use the correct foreground target, not the invalid logged target.*

| Arm | ID delta | ID wins | Median mask IoU pre -> post | IoU wins | Multi-face pre -> post | Post IoU <0.30 |
|---|---:|---:|---:|---:|---:|---:|
| E13 24k | **`+0.3600`** | **12/12** | `0.8959 -> 0.8911` | **5/12** | `1 -> 2` | **0** |
| BC_E13 24k | `+0.2913` | **12/12** | `0.9044 -> 0.8747` | 2/12 | **`1 -> 1`** | **0** |
| CL11 20k | `+0.2893` | **12/12** | `0.8849 -> 0.8798` | 4/12 | **`1 -> 1`** | **0** |

The fix is causally effective for identity and preserves mask ownership. The
ArcFace vector is fused into global PhotoMaker tokens, not a face-local BA-only
edit, so pixels outside the face box are not identical; nevertheless body pose,
scene layout and detected-face position remain stable. Mean absolute RGB change
is about twice as large inside the face box as outside. **[measured] [code]
[visual]**

The table below gives `intended-ID delta / fixed-mask-IoU delta` for every
paired prompt:

| Prompt | E13 24k | BC_E13 24k | CL11 20k | Visual reading |
|---|---:|---:|---:|---|
| Reading | `+.467 / +.005` | `+.298 / -.014` | `+.413 / -.018` | clean identity gain |
| Rushing | `+.418 / -.055` | `+.310 / -.055` | `+.363 / -.072` | E13 adds one face |
| Skiing | `+.222 / -.092` | `+.261 / -.095` | `+.299 / -.057` | ID improves; goggles remain hard |
| Drumming | `+.541 / +.010` | `+.296 / -.012` | `+.358 / +.009` | strongest clean win |
| Kickboxing | `+.370 / +.031` | `+.143 / -.047` | `+.239 / -.009` | same body layout; face improves |
| Dancing | `+.381 / -.065` | `+.270 / -.173` | `+.150 / +.026` | small-face detail remains weak |
| Angry | `+.461 / -.087` | `+.347 / -.072` | `+.411 / -.092` | clean identity gain |
| Crying | `+.432 / +.059` | `+.425 / -.008` | `+.271 / -.063` | expression still imperfect |
| Laughing | `+.217 / -.008` | `+.276 / -.035` | `+.122 / +.001` | low ID and mouth artifacts remain |
| Jumping | `+.148 / +.015` | `+.329 / +.011` | `+.098 / +.041` | placement preserved in all arms |
| Night ride | `+.306 / -.077` | `+.251 / -.130` | `+.384 / -.004` | useful gain; masks remain owned |
| Chef | `+.356 / -.023` | `+.291 / +.024` | `+.363 / -.024` | strong ID; extra people remain |

# 2. E13 24k: strongest identity recovery with stable action layout

E13 has the largest mean identity improvement and wins all 12 prompts. Reading,
Drumming, Angry, Night ride and Chef are clear practical improvements. Skiing
still contains weak goggle/eye structures, and Rushing increases from one
detected face to two. **[measured] [visual]**

![](assets/problematic_validation_20260809/eddie_pre_post_e13_24k_part1.png)

*Figure 3. E13 prompts 1-6. Red is the immutable target mask; cyan is the
mask-selected detected face. Each panel includes the full image and enlarged
face region.*

\newpage

![](assets/problematic_validation_20260809/eddie_pre_post_e13_24k_part2.png)

*Figure 4. E13 prompts 7-12.*

Kickboxing is the clearest audit of the repaired pipeline: intended ID improves
from `0.115` to `0.485`, with one detected face and IoU improving from `0.891`
to `0.922`. Jumping improves `0.193 -> 0.341` while IoU improves
`0.860 -> 0.875`. The body, gloves, horizon and pose stay visually aligned;
the face changes without the wholesale scene change in the invalid sidecar.
**[measured] [visual]**

# 3. BC_E13 24k: smaller identity gain and one Dancing alignment tail

BC_E13 has the lowest corrected Eddie mean (`0.354`) and no increase in the
number of multi-face prompts. Crying reaches `0.459`; Reading, Drumming and
Jumping all improve without changing ownership. Dancing is the alignment tail:
ID improves to `0.356`, but IoU falls to `0.500`. **[measured] [visual]**

![](assets/problematic_validation_20260809/eddie_pre_post_bc_e13_24k_part1.png)

*Figure 5. BC_E13 prompts 1-6.*

\newpage

![](assets/problematic_validation_20260809/eddie_pre_post_bc_e13_24k_part2.png)

*Figure 6. BC_E13 prompts 7-12.*

Kickboxing gains `+0.143` identity and remains seated at IoU `0.899`. Jumping
gains `+0.329`, the largest Jumping gain, with IoU `0.741`. Crying has the
highest BC_E13 identity but its open-mouth expression still needs visual
quality review. Dataset choice changes facial texture and expression severity,
not the ownership result. **[measured] [visual] [inference]**

# 4. CL11 20k: strong Skiing recovery, unresolved small-face detail

CL11 keeps corrected Jumping on the intended body (`0.703 -> 0.744`), as do
E13 and BC_E13. Its strongest distinctive result is Skiing: intended ID rises
to `0.443`, above E13 `0.260` and BC_E13 `0.307`, with IoU `0.804`. This is
useful evidence for its multi-reference/refscale ingredients, not a reason to
replace E13 as the base; its endpoint and training data differ. **[measured]
[inference]**

![](assets/problematic_validation_20260809/eddie_pre_post_cl11_20k_part1.png)

*Figure 7. CL11 prompts 1-6.*

\newpage

![](assets/problematic_validation_20260809/eddie_pre_post_cl11_20k_part2.png)

*Figure 8. CL11 prompts 7-12.*

Skiing identity improves strongly, although the goggle/eye boundary remains
visually imperfect. Kickboxing stays owned at IoU `0.890`. Dancing remains the
weakest CL11 identity (`0.146`) despite IoU `0.684`; Laughing is also low
(`0.136`) with an implausible mouth/eye region. CL11 therefore helps one
occluded prompt without solving the general small-face/anatomy problem.
**[measured] [visual]**

# 5. Root cause and what is not the cause

The reference-selection defect is proven: changing the identity embedding
causes the large paired identity gain. The contract-v2 replay also proves that
the gross old body/head movement was not a model property. The corrected vector
enters global PhotoMaker tokens, so non-face pixels can change, but body layout,
face ownership and fixed-mask geometry remain stable in the valid run.
**[measured] [code] [visual]**

What the images rule out:

- **Not one bad checkpoint or dataset:** the same residual hard prompts appear
  in all three saved models.
- **Not the Eddie BA reference crop:** it already uses the foreground face.
- **Not face absence:** all 36 corrected images have at least one detection.
- **Not simple face scale:** post-fix median size ratios remain near `1.0`.
- **Not an excuse to omit ownership:** the earlier invalid run demonstrates how
  a high ID score can accompany the wrong layout; mask IoU must remain logged
  even though every contract-v2 corrected image is owned.

| Conclusion | Confidence | Basis |
|---|---|---|
| Correct foreground conditioning recovers Eddie identity | **Very high** | exact-replay paired intervention; 36/36 wins **[measured]** |
| Contract-correct correction preserves face/body placement | **Very high** | median IoU `0.875-0.891`; no corrected IoU below `0.30` **[measured]** |
| E13 gives the largest Eddie identity gain | **High** | 12-prompt paired mean at the saved checkpoints **[measured]** |
| CL11 gives the strongest corrected Skiing identity | **High** | `0.443` versus `0.260/0.307` **[measured]** |
| Native target evidence can repair occlusion/anatomy | **Medium** | plausible from routing and visuals; dual-message sweep not yet run **[code] [hypothesis]** |

Not established:

- Whether the same selector correction causes equal gains for other multi-face
  reference images; this intervention covers Eddie only.
- Whether CL11's Skiing advantage comes from multi-reference training,
  reference scaling, checkpoint noise, or their interaction.
- CL10's corresponding final-checkpoint response; it was not part of the
  requested corrected sidecar.

# 6. Priority experiments and implementation plan

## P0 - productionize subject ownership and metric versioning

**Config/namespace:** `manual_val_subject_v2` and
`E13_val_subject_v2_full96`. **Single scientific change:** use the declared
subject for reference selection and generated-face ownership. **Hypothesis:**
the corrected namespace removes false targets and false wins while preserving
all unaffected cells. **Prediction:** Eddie selects the foreground vector;
non-Eddie embeddings reproduce legacy values; ownership failures are explicit.
**Risk:** silently joining V1 and V2 curves would create a false training trend.

Implementation:

1. Add a shared detector selector ranked by overlap with a declared subject box,
   with largest confident face only as a documented fallback.
2. Replace direct `faces[0]` reads in validation embedding creation, pipeline
   conditioning, and active training extraction paths.
3. Materialize a new versioned embedding file; never overwrite legacy vectors.
4. Log both legacy `IDSimBest` and `IDSimMaskMatched`, plus selected IoU, center,
   size, face count and an `owned` flag. A selected IoU below the registered
   ownership threshold must not silently count as a valid identity win.
5. Gate on exact non-Eddie reproduction and a full fixed-96 dry run with zero
   ambiguous or missing declared-subject selections.

This is validation/data plumbing only and does not change the trainable model.

## P1 - E13 dual native/reference face-message diagnostic

**Config:** `E13_subject_v2_dual_face_message_eval`. **Single scientific
change:** bounded mixture of separately normalized native target-face and
reference-face attention messages. **Hypothesis:** native target evidence
repairs goggles, mouths and expression while the explicit reference route
retains identity. **Prediction:** an intermediate mixture improves blinded hard
faces with at most `0.01` full-panel mask-owned ID loss. **Risk:** native
evidence can dilute identity.

Implementation:

1. Add a defaults-off `hard_replace_v1_dual_eval` path. For target-face queries,
   compute independent `C_native` and `C_ref` softmax messages; merge only their
   outputs as `(1-alpha) C_native + alpha C_ref`. Keep background routing and
   reference K/V explicit.
2. Load untouched E13 24k and sweep
   `alpha={1.00,0.85,0.70,0.55,0.35}` on fixed-96. `alpha=1.00` must reproduce
   E13 exactly.
3. Pre-register Skiing, Laughing, Crying and small Dancing/Jumping faces; log
   legacy ID, mask-owned ID, TOPIQ-Face, ownership geometry and face count.
4. Promote only a blinded anatomy improvement with at most `0.01` mask-owned
   ID loss and no alignment regression. Stop without training if no Pareto
   point exists.

## P2 - E13 target-scale-matched spatial reference diagnostic

**Config:** `E13_subject_v2_target_scale_eval`. Keep global PhotoMaker inputs
unchanged and sweep only spatial BA crop scale: exact baseline plus face short
side ratios `{0.75,1.00,1.25}` relative to the target mask. Run fixed-96 with
Marion/Jisoo/Lex Jumping and Dancing as primary cells. Promote only if blinded
small-face anatomy and TOPIQ-Face improve with at most `0.01` mask-owned ID loss
and no center/ownership regression. Preserve `pose_adapt_ratio=0` and
`ca_mixing_for_face=false`.

# 7. Reproducing the report

The frozen per-image audit is
`analysis/assets/problematic_validation_20260809/data/`\
`corrected_eddie_final_checkpoint_rows.csv`. Exact generation argv and hashes
are in each sidecar model folder's `command_manifest.json` and
`run_manifest.json`.

The guarded generation job was
`lm-mpi-job-baea4903-7f8d-4785-a67d-f153df3299da`; its package is
`serv_run_packages/eddie_revalidation_contract_v2_serv_20260809_r1/`.

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh
conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template

python analysis/assets/problematic_validation_20260809/analyze_final_corrected_sidecar.py
python analysis/assets/problematic_validation_20260809/build_eddie_pre_post_report_assets.py
python tools/reports/publish_report.py \
  analysis/2026-08-09_eddie_validation_pre_vs_post_reference_fix.md --upload
```

Derived paired values are in `data/eddie_pre_post_paired.csv`. The image
builder reads frozen audit outputs and does not rerun inference or detection.

# References

- [`2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.md`](2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.md) - full problematic-validation audit and architecture/dataset recommendations.
- [`docs/validation_protocol.md`](../docs/validation_protocol.md) - standard fixed-96 validation contract and diagnostic exceptions.
- `analysis/assets/problematic_validation_20260809/final_checkpoint_sidecar_contract_v2/` - exact replays, corrected images, replay gates, resolved configs, and immutable run manifests.
