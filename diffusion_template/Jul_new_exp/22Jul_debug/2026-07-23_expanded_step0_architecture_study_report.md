# Expanded PhotoMaker branched-attention step-zero architecture study

## Executive result

The most promising training initialization is now:

`n3a_fullgrid_up_dual35`

It keeps N3a's full spatial reference memory, disables branched cross-attention,
uses reference K/V only in up blocks, restores ordinary PhotoMaker epsilon
outside the target face core, and replaces hard core ownership with a trainable
per-head target/reference attention mix initialized to 65% target / 35%
reference.

On the canonical 24 cases it retains nearly canonical N3a face-change strength
while approximately halving landmark movement:

| architecture | faces | face MAE vs PM | landmark shift | bbox IoU | outside MAE |
|---|---:|---:|---:|---:|---:|
| canonical N3a core-ring | 24/24 | 0.08588 | 0.02411 | 0.94407 | 0.01192 |
| full-grid dual 0.25 | 24/24 | 0.05830 | 0.00902 | 0.96873 | 0.01093 |
| **full-grid dual 0.35** | **24/24** | **0.08121** | **0.01223** | **0.96120** | **0.01199** |
| full-grid dual 0.50 | 24/24 | 0.10353 | 0.01929 | 0.94207 | 0.01246 |

Visual inspection agrees with the geometry metrics. Dual-0.35 changes facial
appearance broadly across all eight identities while keeping expression,
pose, head placement, occluders, and face/body attachment substantially cleaner
than canonical core-ring N3a. Dual-0.25 is the recommended safety-biased
fallback if training amplifies reference authority.

The all-96 validation confirms the selection:

| architecture | faces | face MAE | landmark shift | bbox IoU | outside MAE |
|---|---:|---:|---:|---:|---:|
| canonical core-ring | 96/96 | 0.09124 | 0.02872 | 0.93418 | 0.01422 |
| dual 0.25 | 96/96 | 0.05787 | 0.00732 | 0.96643 | 0.01232 |
| **dual 0.35** | **96/96** | **0.07723** | **0.01134** | **0.95526** | **0.01358** |

Dual-0.35 cuts median landmark movement by about 61% on all 96 and improves
bbox alignment while preserving a strong face-local change.

The dual-0.35 improvement is broad but not uniform. Seven identities improve strongly in
landmark alignment; Marion remains the main tail case (dual-0.35 landmark
`0.03357`, bbox `0.90068` versus canonical `0.03176`, `0.93000`). This does not
invalidate the aggregate winner, but Marion-like pose/appearance cases should
be a specific training regression gate. Dual-0.25 repairs Marion landmark shift
to `0.01472` and improves its bbox over dual-0.35 to `0.90783`, although the bbox
still trails canonical. This supports 0.25 as the robust fallback.

Matched-reference identity metrics are recorded but are not used to reject
step-zero initializations. The selection criterion requested for this stage is:
active architecture, visibly different faces from PhotoMaker, coherent faces,
broad alignment, and contained scene change.

## Evaluation protocol

- repository: `main_clean` at `1e88825dc4a325ea1e146be2fa519801f048a73e`
- environment: `photomaker_NS`
- accelerator: NVIDIA H100 80 GB
- discovery schedule: 20 DDIM steps, PhotoMaker start 4, BA start 6
- guidance scale: 5
- base model: RealVisXL V4.0
- identity encoder: PhotoMaker V2
- matched deterministic seeds and prompts
- diverse-eight indices: `5,6,8,10,14,17,18,22`
- canonical-24 indices:
  `5,6,8,10,14,17,18,22,31,35,36,47,51,52,53,64,70,72,74,77,81,84,89,94`
- broad validation: all indices `0..95`

Every completed run stores its resolved experiment spec, compact config,
architecture signature, per-sample metrics, images, contact sheet, and summary
in an immutable run directory below `experiments/`.

For packed/NN7 architectures, activity is measured causally against a branched
zero (BA0) execution. Legacy N3a variants are measured against paired ordinary
PhotoMaker. The protected production files were hashed before and after runs;
the notebook reported them unchanged.

## Most important architecture findings

### 1. Canonical repaired N3a is a valid active baseline

`n3a_fullgrid_up_core_ring_anchor` completed 96/96 valid faces. It is visibly
active and sufficiently aligned for a training baseline, consistent with the
user's visual assessment. Its all-96 medians are face MAE `0.09124`, landmark
shift `0.02872`, bbox IoU `0.93418`, and outside MAE `0.01422`.

The hard inner-core reference ownership is also the main remaining source of
eye, mouth, lighting, and local geometry distortion. Reducing the core, delaying
BA, or increasing final-core erosion helps alignment mainly by reducing useful
activity.

### 2. Fixed target/reference dual attention is the effective repair

Dual attention computes target and reference face attention from the same
target-coordinate queries, then blends the outputs per attention head. This
retains target geometry everywhere instead of assigning an inner ellipse to
100% reference attention.

The authority sweep is smooth and exposes a clear knee:

| reference mix | set | face MAE | landmark | bbox IoU | outside |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 8 | 0.05876 | 0.00682 | 0.97817 | 0.00995 |
| 0.30 | 8 | 0.06836 | 0.00952 | 0.97317 | 0.01028 |
| 0.35 | 8 | 0.08051 | 0.01219 | 0.97379 | 0.01076 |
| 0.40 | 8 | 0.08417 | 0.01730 | 0.96004 | 0.01160 |
| 0.45 | 8 | 0.09716 | 0.01751 | 0.95509 | 0.01223 |
| 0.50 | 8 | 0.10386 | 0.01572 | 0.95603 | 0.01213 |

Moving from 0.35 to 0.40 adds little face-change magnitude but sharply worsens
geometry. The 24-case dual-0.50 validation also loses bbox alignment. This
supports 0.35 as the balanced default rather than selecting the strongest
possible branch.

### 3. Confidence residual is safe but too gentle

After correcting the runtime gain routing, full-grid confidence residual gives:

| gain | faces | face MAE | landmark | bbox IoU | outside |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 8/8 | 0.02566 | 0.00266 | 0.98463 | 0.00878 |
| 0.50 | 8/8 | 0.03524 | 0.00380 | 0.97845 | 0.00996 |
| 0.75 | 8/8 | 0.04386 | 0.00440 | 0.98003 | 0.01022 |

It is a useful ultra-safe family, but visual authority remains below fixed
dual-0.25. It is not the primary training candidate.

### 4. Landmark-local NN7 provides a modern safe alternative

Landmark-IDW 3x3 reference neighborhoods with target-coordinate queries are
causally active and extremely aligned. Gate `0.065` on 24 cases gives BA0-
relative face MAE `0.01145`, landmark shift `0.00111`, bbox IoU `0.99314`, and
24/24 faces. A staged up0/up1 gate `0.03/0.075` raises diverse-eight activity
to `0.01597` while retaining landmark `0.00148` and bbox `0.99235`.

These routes are coherent but much weaker than the N3a dual family at step
zero. They remain secondary architectures for a lower-risk training study.

### 5. Native NN4/NN5/NN6 are inactive at step zero

NN4, NN5a, NN5b, and NN6a are exactly equal to their branched-zero baselines:
face/outside MAE `0`, identity gain `0`, landmark shift `0`, bbox IoU `1`.
Their zero-initialized connectors make them training hypotheses, not rankable
step-zero architectures. Apparent differences versus ordinary PhotoMaker came
from the doubled execution path and disappear against BA0.

## Negative and non-promoted results

- Global landmark affine warping produced duplicated/boundary artifacts and
  large geometry movement; rejected.
- Semantic landmark-radius masks and delayed modern-local injection were safe
  but suppressed useful activity.
- N3a core `0.35/0.50/0.60`, delayed starts, and output erosion improved safety
  mainly by weakening the branch; none beat fixed dual mixing.
- Zero reference pooled text looked useful on eight but did not generalize on
  24: face `0.08389`, landmark `0.02147`, bbox `0.94362`.
- Pairing CFG reference noise provides only small mixed changes.
- Combining individually modest N3a refinements did not compound their gains.
- Dual-0.35 plus zero pooled text, paired noise, or a step-7 delay did not beat
  the plain dual-0.35 balance.
- Dual-0.50 is active and face-valid but crosses the useful alignment boundary
  on the 24-case set.
- Branched cross-attention remains disabled because recent and historical runs
  consistently associate it with drift and melting.

## Recommended training decision

### Primary: full-grid dual-0.35

Train the exact 24-case winner first:

```yaml
disable_branched_ca: true
strict_face_routing: false
model:
  ba_processor_variant: legacy
  ba_sa_ref_token_mode: full_grid
  ba_sa_face_mode: dual
  ba_sa_mix_init: 0.35
  ba_sa_ref_layer_scope: up
  ba_target_core_erode_frac: 0.10
  ba_output_anchor_mode: base_outside_core
```

The dual mix is trainable per head/layer. Log its distribution throughout
training and stop if it rapidly saturates toward full reference ownership.

### Safety fallback: full-grid dual-0.25

Use the identical architecture with `ba_sa_mix_init: 0.25`. This retains
clearly visible activity with materially tighter geometry. It is a good second
training arm or a rollback if 0.35 amplifies drift.

### Required controls

Retain canonical core-ring N3a as the active historical control and ordinary
PhotoMaker as the global baseline. Optionally train the staged landmark-local
route as the modern low-authority control.

Select checkpoints by 96-case visual and metric validation, not training MSE
or final epoch. Track face validity, landmarks, bbox IoU, outside exactness,
occluder preservation, face/body attachment, identity direction, and per-head
mix movement.

## Current broad-validation status

- canonical core-ring N3a: complete, 96/96
- dual-0.35: complete, 96/96; face `0.07723`, landmark `0.01134`, bbox
  `0.95526`, outside `0.01358`
- dual-0.25: complete, 96/96; face `0.05787`, landmark `0.00732`, bbox
  `0.96643`, outside `0.01232`

The final recommendation remains dual-0.35 for maximum balanced step-zero
activity and dual-0.25 for the safety-biased training arm.

## Main artifacts

- canonical all96 PDF:
  `visual_reports/20260722_n3a_fullgrid_up_core_ring_anchor_all96.pdf`
- canonical vs dual-0.25 vs dual-0.35 24-case PDF:
  `visual_reports/20260723_n3a_canonical_vs_dual25_dual35_24.pdf`
- dual-0.35 all96 PDF:
  `visual_reports/20260723_n3a_fullgrid_dual35_all96.pdf`
- dual-0.25 all96 PDF:
  `visual_reports/20260723_n3a_fullgrid_dual25_all96.pdf`
- consolidated latest results: `expanded_results_latest.md` and
  `expanded_results_latest.csv`
- per-identity table: `expanded_results_by_identity.csv`
- chronological log: `expanded_study_progress.md`
- historical-idea audit: `2026-07-23_recent_run_idea_audit.md`
- canonical training handoff:
  `promising_configs_MD/n3a_fullgrid_up_core_ring_anchor_training_handoff.md`
- dual-0.35 training handoff:
  `promising_configs_MD/n3a_fullgrid_up_dual35_training_handoff.md`

All new code, specs, logs, metrics, images, and PDFs were written only inside
`Jul_new_exp/22Jul_debug`. No production repository code was changed.
