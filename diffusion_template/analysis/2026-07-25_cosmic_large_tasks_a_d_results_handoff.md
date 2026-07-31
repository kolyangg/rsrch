# Cosmic Large one-ID Tasks A–D results handoff

**Date:** 25 July 2026

**Branch:** `test`

**Deployed Task D commit:** `5e55450b931bf737e20e4ccceab60d8683c092a3`

**Execution status:** Tasks A, B, C, and all three Task D arms are complete.
Neb is idle. No result is promoted.

## Executive conclusion

The experiments narrow the Cosmic failure to a stage-dependent spatial
reference problem rather than a single bad toggle:

1. Branched cross-attention contributes to broad scene/body corruption, but
   disabling it does not repair the dominant malformed-face failure.
2. Freezing the target/noise self-attention copies and training only the
   reference path does not repair it either.
3. At inference, a tight 256px Cosmic reference causally recreates malformed
   face fragments on an otherwise healthy checkpoint, while a full-scene
   reference largely restores attached facial anatomy.
4. During training, however, deterministic 256px references are the best of
   the controlled Task D arms when all arms are evaluated with the same
   full-scene reference. They improve both identity score and visual anatomy
   over full-scene training references.
5. Target-view diversity matters: repeating one target view with otherwise
   full-scene references is worse than using eight target views.

The practical direction is therefore **diverse full-scene targets plus
Cosmic-style crops during training, with full-scene references at
validation/inference and branched CA disabled**. This is a research direction,
not a promoted recipe: even the best Task D checkpoint retains clear Drumming
and Jumping facial failures and has been tested on only one identity.

## Source reports

- [Task A: Cosmic face-only CA-off result](../docs/experiments/2026-07-25_task_a_cosmic_faceonly_noca_4k_results.md)
- [Task B: fixed-checkpoint diagnostic matrix](../docs/experiments/2026-07-25_task_b_checkpoint_diagnostic_matrix_results.md)
- [Task C: CA-off reference-only result](../docs/experiments/2026-07-25_task_c_cosmic_faceonly_noca_refonly_4k_results.md)
- [Tasks A–D implementation handoff](2026-07-24_cosmic_large_next_steps_implementation_handoff.md)

This file is the consolidated decision handoff. The linked reports retain the
full commands, reproduction details, and earlier representative images.

## Evidence integrity

### Controlled artifact

Task D used the sealed woman-class artifact:

```text
/home/niko/rsrch/dataset_full/controlled_identity_factorial/nm0004960_v1
```

- Identity: `nm0004960`
- Training image IDs: `0, 19, 20, 27, 31, 49, 56, 59`
- Repeated target for `single_full`: image `0`
- Recurring validation identity image: `69`
- Final holdout image: `92`
- Full-scene and deterministic `cosmic_256` reference modes were derived from
  the same eight source images.
- The 256px crops use a 20% margin per side, bicubic interpolation, JPEG
  quality 95, and subsampling 0.
- The same 12 validation prompts, seeds, full-scene reference `69.jpg`,
  inspected generation bboxes, scheduler, inference steps, and RealVis
  validation base were reused for every arm.
- `manifest.sha256` passes locally and on Neb.
- `manifest.json` SHA-256:
  `dcc270c1a5893d8a30734a7b6d724636bd210161d49f65554d33581dd0015787`

### Run comparability

All Task D arms used:

```text
trainer.n_epochs: 8
trainer.epoch_len: 500
trainer.masked_loss_step: 1
trainer.seed: 0
train_dataset_name: controlled_identity_factorial
disable_branched_ca: true
train_branched_ca_lora: false
train_ba_only: true
branched_attn_weight_mode: noise_and_ref
model.rank: 32
lr_for_lora: 0.0001
```

The three step-0 validation sets are byte-identical as sets; their combined
hash is:

```text
5b2ba00c83dc0b40bda662320ea8f3716435d537fc646b2734921120bc7f2af3
```

Each run produced eight full checkpoints, eight weights-only checkpoints, and
12 images at step 4,000. There was no OOM, traceback, or fatal run error.
The recurring ONNX Runtime CUDA-provider messages were the known InsightFace
CPU fallback and did not prevent metrics or image generation.

Comet exports were resolved by immutable experiment ID. For every arm, the
step-4,000 export contained 12 images, no export warnings/errors, and matched
the corresponding Neb files in 12/12 SHA-256 hashes.

## Task A summary — branched CA disabled

Task A trained the original Cosmic one-ID face-only setup with branched CA
disabled while preserving branched self-attention.

Observed:

- Corrected genuine CA-off validation improved scene/body coherence over the
  historical CA-on checkpoint.
- Text similarity improved from `23.7565` to `24.7982`.
- Identity similarity improved from `0.0351` to `0.1418`.
- Approximately 9/12 prompts still had oversized, missing, or displaced
  facial features.

Conclusion:

Branched CA amplifies global/exterior corruption but is not the primary cause
of the face-local failure. Task A was not promoted.

## Task B summary — fixed-checkpoint causal diagnostics

Task B reproduced the saved Cosmic CA-on and leak-free one-ID endpoints at
exact 12/12 filename, file-hash, and decoded-pixel equality before applying
interventions.

Observed:

- Validation-native processor buffers did not repair the failure.
- CFG 1 produced global haze/collapse.
- CA-off cleaned scenes but left pasted or misregistered facial fragments.
- A tight 256px Cosmic reference recreated the pathology on the healthy
  one-ID checkpoint.
- A deliberately wrong 1024px Larry King full-scene reference changed
  identity but restored coherent attached faces on the Cosmic checkpoints.
- Null identity/reference input removed or destroyed the face region.

Conclusion:

The identity/reference path is causally active. Tight reference formatting at
inference is the leading trigger; validation-base buffers and CFG are not.
Task B is the strongest causal evidence in the sequence.

## Task C summary — reference-only self-attention training

Task C kept branched CA disabled but changed self-attention ownership from
`noise_and_ref` to `ref_only`, training 420 reference-path processor tensors
and no target/noise or cross-attention processor tensors.

Observed:

- Corrected endpoint text similarity: `24.4779`
- Corrected endpoint identity similarity: `0.1484`
- Approximately 9/12 prompts still failed the hard anatomy gate.
- Chef improved relative to Task A, while Jumping regressed; the overall
  failure rate did not change.

Conclusion:

Target/noise projection drift is not the primary cause. The remaining failure
lies further downstream in reference spatial representation, routing, or
target-face registration. Task C was not promoted.

## Task D design

Task D separated target diversity from training-reference format while using
the Task A CA-off architecture.

| Arm | Training targets | Training references | Isolated question |
|---|---|---|---|
| `multi_full` | Eight distinct full scenes | Another full scene | Clean control |
| `multi_cosref` | Same eight full scenes | Deterministic tight 256px crops | Cost or benefit of Cosmic-style training references |
| `single_full` | One repeated full scene | Seven distinct full scenes | Cost of losing target-view diversity |

All arms used the same full-scene validation reference. Task D therefore
changes **training** reference format; it does not repeat Task B's
inference-reference intervention.

## Task D completed results

### Endpoint results

| Arm | Text sim | ID sim | Manual endpoint anatomy review |
|---|---:|---:|---|
| `multi_full` | 25.7448 | 0.2357 | Approximately 6–7/12 clearly coherent; 5–6 malformed or borderline |
| `multi_cosref` | **26.9297** | **0.3375** | Best arm; two clear hard failures and several milder/borderline faces |
| `single_full` | 25.0182 | 0.1853 | Approximately five clearly coherent; repeated large-eye/missing-feature failures |

Relative to `multi_full`, `multi_cosref` gained `+1.1849` text similarity
(`+4.6%`) and `+0.1018` identity similarity (`+43.2%`). `single_full` lost
`-0.7266` text similarity (`-2.8%`) and `-0.0504` identity similarity
(`-21.4%`).

The identity metric is not an anatomy metric, but here the `multi_cosref`
increase agrees with a substantial visual improvement rather than merely
rewarding isolated identity fragments.

### Best logged identity checkpoints

| Arm | Peak step | Text sim | ID sim | Checkpoint SHA-256 | Visual decision |
|---|---:|---:|---:|---|---|
| `multi_full` | 3,500 | 26.2409 | 0.2466 | `d91b72f0fa1f6fd184b8be0550767ee52f98dd1cb2d2b20ae6975ebca31153cf` | Still contains multiple pasted-eye/blank-face failures |
| `multi_cosref` | 2,500 | 26.6471 | **0.3591** | `c6cbfffe0b7d0dde14970bfbdb9c51739c972e6285bd17e7ebddca0421da2d3c` | Best candidate; roughly 10/12 plausible, but Drumming and Jumping still fail |
| `single_full` | 3,000 | 24.0143 | 0.2033 | `cdab1dab81f61b4239cc59cf9b956ae635b3802bcabdab08727afad9b0b531f8` | Metric peak does not remove the dominant anatomy failures |

The `multi_cosref` epoch-5 checkpoint is the most useful diagnostic artifact,
but it does not pass the promotion gate.

### Endpoint artifacts and Comet records

| Arm | Runtime | Endpoint checkpoint SHA-256 | Weights SHA-256 | Comet |
|---|---|---|---|---|
| `multi_full` | 07:14–09:03 UTC, ~1h49m | `ea4fe47957fa64528e3324d9476e049280d35889f621b2afda823f5284a7d257` | `1e63acad2bcbe2993100be4ec03f4fa65cd9f0e2a57494d7598cf76f02d1f8e8` | [d6363cba…](https://www.comet.com/nikolay-2104/rsrch-jul/d6363cba32e444469cde81b1d6e291af) |
| `multi_cosref` | 10:14–12:09 UTC, ~1h54m | `74f4d12127af1f5e696d11ae34d367ab2fffb309dfe3b4bb55a25968351d9af9` | `0c06b07665beb7d126450e540f7ef9882aaf777f4379794849da2f06ad11efdb` | [3738f676…](https://www.comet.com/nikolay-2104/rsrch-jul/3738f67625894b1ba583d3c7eff06c51) |
| `single_full` | 12:10–14:00 UTC, ~1h50m | `2132534aae50a28b4d2df1b6cba4c952fb3827dd5518049078e9f52d1191bd8f` | `cb5dba29fc7f13fcb60335af77d2f0513486dc8824c78b7b68162b3516e97842` | [ce325660…](https://www.comet.com/nikolay-2104/rsrch-jul/ce3256602a7b4f09a82a30db616c3c3e) |

The backfilled non-secret Comet records are stored beside each Neb run as
`saved/<run_name>/comet_experiment.json` and cached locally under the ignored
`comet_records/` directory.

## Representative Task D images

These are exact step-4,000 PNGs downloaded from Comet and verified byte for
byte against the Neb validation folders.

### Reading — crop-trained arm repairs most of the face

| `multi_full` | `multi_cosref` | `single_full` |
|---|---|---|
| ![Reading, multi full](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/reading_multi_full.png) | ![Reading, multi cosref](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/reading_multi_cosref.png) | ![Reading, single full](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/reading_single_full.png) |

`multi_full` and `single_full` paste an oversized eye into an incomplete face.
`multi_cosref` produces a substantially more complete, attached face, although
the mouth remains imperfect.

### Jumping — persistent hard failure in every arm

| `multi_full` | `multi_cosref` | `single_full` |
|---|---|---|
| ![Jumping, multi full](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/jumping_multi_full.png) | ![Jumping, multi cosref](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/jumping_multi_cosref.png) | ![Jumping, single full](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/jumping_single_full.png) |

All three arms retain displaced, enlarged, or incomplete facial geometry in
this small, high-motion target pose. Training-reference format alone is not a
complete architecture fix.

### Night-ride — stable success across the factorial

| `multi_full` | `multi_cosref` | `single_full` |
|---|---|---|
| ![Night ride, multi full](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/night_ride_multi_full.png) | ![Night ride, multi cosref](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/night_ride_multi_cosref.png) | ![Night ride, single full](assets/2026-07-25_cosmic_large_tasks_a_d_results_handoff/night_ride_single_full.png) |

The mechanism can produce a coherent attached face in some prompt/pose
conditions; the failure is systematic but not universal.

## Factorial interpretation

### Observed evidence

1. `multi_full` is degraded even with diverse targets, full-scene training
   references, and CA disabled. The remaining architecture is unstable
   independently of the original Cosmic crop format.
2. `multi_cosref` is materially better than `multi_full`. Tight crops are not
   intrinsically harmful as **training** references; in this controlled setup
   they appear beneficial.
3. `single_full` is worse than `multi_full`. Target-view diversity contributes
   positively to identity learning and anatomy.
4. No arm reaches 12/12 valid facial anatomy, so none can be promoted.

### Stage-dependent reconciliation of Tasks B and D

Task B changed the reference at inference and showed that a tight crop can be
inserted too literally into target geometry. Task D changed the references
used during optimization but kept a shared full-scene validation reference.

The combined evidence supports this distinction:

```text
tight crops during training
    -> can focus identity learning and reduce nuisance scene context

tight crop injected through the current spatial reference path at inference
    -> can be pasted or misregistered as literal facial geometry
```

This is an inference from the controlled interventions, not yet a direct
mechanistic proof.

### Remaining hypotheses

- Cropped training references may act as an identity-focused regularizer,
  whereas full scenes add background/style nuisance variation.
- The remaining Drumming/Jumping failures are likely sensitive to face scale,
  pose, or reference-to-target spatial correspondence.
- Full-layer branched self-attention may still be too strong or lack a
  target-native fallback when the target face is small or rotated.
- Mask placement and layer/site selection may amplify the spatial
  misregistration.

## Decision and next work

Do not promote any completed checkpoint or start a long full-Cosmic run yet.

Recommended order:

1. Preserve `multi_cosref` epoch 5 / step 2,500 as the best diagnostic
   checkpoint.
2. On that fixed checkpoint, run a compact full-scene versus tight-crop
   inference-reference comparison plus wrong/null controls. This directly
   tests the training/inference stage distinction on the best Task D model.
3. Add spatial-path stabilization behind toggles: narrower semantic
   self-attention site selection, target-native fallback, and/or
   scale/pose-aware reference routing. Keep CA off for these diagnostics.
4. Repeat the winning controlled setup on at least two additional identities
   before considering promotion.
5. If adapting Cosmic Large now, use diverse target scenes and deterministic
   cropped training references, but keep full-scene validation/inference
   references and enforce the hard 12-prompt anatomy gate.

Avoid spending more runs on CFG 1, validation-base buffer replacement,
CA-on training, or repeated single-target training; Tasks A–D already provide
negative evidence for those directions.

## Repository state

The A–C reports, this handoff, representative assets, and the new
Comet-record/retrieval tooling are local and uncommitted. The deployed Task D
runs remain attributable to commit `5e55450b...`; do not rewrite their
manifests to claim the later documentation/tooling changes.
