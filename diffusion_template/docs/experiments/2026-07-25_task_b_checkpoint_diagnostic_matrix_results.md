# Task B result: fixed-checkpoint inference diagnostics

**Date:** 25 July 2026

**Branch:** `test`

**Git commit:** `437dd043f0b07cb535ee3761a43212a90fa51548`

## Experiment contract

Task B evaluated saved RHCA checkpoints without taking optimizer steps. The
standalone entry point was:

```text
tools/inference/evaluate_rhca_checkpoint.py
```

Every accepted row used the endpoint's fixed 12 prompts, per-prompt CUDA seed,
PhotoMaker reference, inspected generation bboxes, RealVisXL validation base,
DDIM scheduler, 50 inference steps, PhotoMaker start step 10, and branched
attention start step 15. Each output contains:

```text
images/*.png
command_manifest.json
resolved_config.yaml
run_manifest.json
per_image.json
```

`validation_native` installed processors from the validation base and then
loaded only the saved trainable deltas. Its audit found 70 branched
self-attention processors in every row. CA-on rows had 70 branched
cross-attention processors; CA-off rows had zero.

## Endpoints

| Endpoint | Checkpoint SHA-256 |
|---|---|
| Cosmic face-only CA-on, 4k | `197aff6f82f898c4f671852c3e780fb3046678e0f420843ab467fa082b9fbf4e` |
| Leak-free one-ID holdout-51, 4k | `bff146619407d6c3302b2cdeda5b6123eb2a39388e31804c5c0f8a5a0c7f14df` |
| Task A Cosmic face-only CA-off, 4k | `440e5404cdb5e6a554c286c52de9d22e822180adaaac8a90228635dcf7866dd3` |

All accepted rows produced 12 images and all four manifests. No OOM,
traceback, or fatal process error occurred. Repeated ONNX Runtime CUDA-provider
messages were the existing InsightFace CPU fallback caused by the unavailable
cuDNN 9 provider library; image generation and metric evaluation completed.

## Reproduction gates

### Cosmic CA-on

The first attempt used the manual index-keyed generation-bbox map and failed
the gate at 0/12 exact images. Inspection showed that trainer validation had
used its automatic filename-keyed cache instead.

The corrected input reindexed that exact automatic cache for the fixed
dataset. The accepted reproduction row is:

```text
diagnostics/cosmic_faceonly_4k/row01b_legacy_cfg5_matched_auto_bbox
```

It matched the trainer's saved step-4,000 output in 12/12 filenames, SHA-256
file hashes, and decoded pixel arrays.

### Leak-free one-ID

Trainer validation used automatic boxes for 11 samples and its configured
forced-manual Chef box for sample 11. The sealed reproduction input combines
those exact records. The accepted row is:

```text
diagnostics/one_id_holdout51_4k/row01_legacy_cfg5_matched_auto_plus_manual_bbox
```

It also matched 12/12 filenames, file hashes, and decoded pixels.

### Task A CA-off adjustment and validation-integrity finding

Task A's checkpoint contains self-attention processor state and no trained CA
processor state. Its matrix therefore kept CA disabled in every fair row;
enabling CA would invent untrained weights.

The saved Task A trainer images could not be used as the reproduction gate.
Code inspection found that `BaseTrainer._evaluation_epoch` instantiated the
temporary alternate-base validation model and called
`prepare_for_training()` before propagating `disable_branched_ca`. It also did
not propagate the architecture flags to the temporary pipeline. The saved
images therefore exercised randomly initialized branched CA even though
training itself was CA-off.

A bug-replay row deliberately re-enabled untrained CA and produced all 12
images, but was 0/12 pixel-identical because the unsaved random CA
initialization cannot be reconstructed from the checkpoint. It is retained
only as diagnostic evidence:

```text
diagnostics/cosmic_faceonly_noca_4k/row01c_historical_trainer_bug_untrained_ca_replay
```

The fair Task A rows used explicit CA-off construction and processor audits.

## Observed evidence

### Cosmic face-only CA-on matrix

| Row | Processor base / intervention | CFG | Text sim | ID sim |
|---:|---|---:|---:|---:|
| 1 | legacy, CA on, matched | 5 | 23.7565 | 0.0351 |
| 2 | validation-native, CA on, matched | 5 | 23.4232 | 0.0517 |
| 3 | validation-native, CA on, matched | 1 | 15.0762 | -0.0158 |
| 4 | validation-native, CA off, matched | 5 | 24.1875 | 0.1066 |
| 5 | validation-native, CA off, wrong full-scene Larry King reference | 5 | 26.2839 | 0.0350 |
| 6 | validation-native, CA off, null reference/identity | 5 | 27.9870 | -0.0060 |

Visual review of all 12 images per row showed:

- Changing from the legacy hybrid-base processor copy to validation-native
  processor bases did not repair malformed facial anatomy.
- CFG 1 caused a strong global gray haze and loss of useful image quality.
- Disabling CA cleaned scenes and bodies but retained oversized, pasted, or
  displaced face fragments under the matched Cosmic reference.
- Replacing the matched 256px Cosmic reference with the 1024px Larry King
  full-scene reference changed every generated identity and produced mostly
  coherent facial anatomy.
- The null intervention destroyed or removed the face region. The reference
  path is therefore causally active rather than ignored.

### Leak-free one-ID matrix

| Row | Processor base / intervention | CFG | Text sim | ID sim |
|---:|---|---:|---:|---:|
| 1 | legacy, CA on, matched | 5 | 23.2760 | 0.4068 |
| 2 | validation-native, CA on, matched | 5 | 23.2930 | 0.3956 |
| 3 | validation-native, CA on, matched | 1 | 20.0046 | 0.2375 |
| 4 | validation-native, CA off, matched | 5 | 25.3581 | 0.2637 |
| 5 | validation-native, CA off, wrong 256px Cosmic reference | 5 | 26.1224 | 0.0446 |
| 6 | validation-native, CA off, null reference/identity | 5 | 27.4909 | 0.0519 |

The matched legacy and validation-native CFG-5 outputs both retained coherent
anatomy. CFG 1 again degraded contrast and quality. CA-off matched remained
mostly coherent.

The wrong-reference intervention is the strongest causal result in the
matrix: substituting Cosmic `holdout_A.jpg` and its verified bbox
`[59, 42, 203, 236]` into this otherwise healthy one-ID checkpoint recreated
the oversized and misplaced face-fragment pathology. The null intervention
again destroyed the face region.

### Representative causal comparisons

The following images are copied from the accepted Task B diagnostic rows.
Columns within each table use the same checkpoint, prompt, seed, validation
base, scheduler, inference steps, CFG, and CA-off construction. The intended
intervention is the reference image and bbox.

#### Cosmic checkpoint

| Prompt | Matched tight 256px Cosmic reference | Wrong full-scene Larry King reference |
|---|---|---|
| Reading | ![Cosmic checkpoint with matched tight reference, Reading](assets/2026-07-25_task_b/cosmic_matched_reading.png) | ![Cosmic checkpoint with full-scene reference, Reading](assets/2026-07-25_task_b/cosmic_fullscene_reading.png) |
| Angry | ![Cosmic checkpoint with matched tight reference, Angry](assets/2026-07-25_task_b/cosmic_matched_angry.png) | ![Cosmic checkpoint with full-scene reference, Angry](assets/2026-07-25_task_b/cosmic_fullscene_angry.png) |

The matched tight reference produces an oversized isolated eye and loses
normal nose/mouth geometry. The deliberately wrong full-scene reference
changes the identity toward Larry King but restores a substantially coherent,
attached face while preserving the requested scene.

#### Leak-free one-ID checkpoint

| Prompt | Matched full-scene Larry King reference | Wrong tight 256px Cosmic reference |
|---|---|---|
| Reading | ![One-ID checkpoint with matched full-scene reference, Reading](assets/2026-07-25_task_b/oneid_matched_reading.png) | ![One-ID checkpoint with tight Cosmic reference, Reading](assets/2026-07-25_task_b/oneid_cosmic_reading.png) |
| Angry | ![One-ID checkpoint with matched full-scene reference, Angry](assets/2026-07-25_task_b/oneid_matched_angry.png) | ![One-ID checkpoint with tight Cosmic reference, Angry](assets/2026-07-25_task_b/oneid_cosmic_angry.png) |

The matched full-scene reference yields recognizable attached faces, with
minor eyewear or mouth artifacts. Changing only to the tight Cosmic reference
turns the face into a stretched or blank skin patch. Reproducing the pathology
on an otherwise healthy checkpoint is the strongest visual evidence that
reference formatting is causal.

### Task A genuine CA-off matrix

| Row | Processor base / intervention | CFG | Text sim | ID sim |
|---:|---|---:|---:|---:|
| 1 | legacy, CA off, matched | 5 | 24.4076 | 0.1407 |
| 2 | validation-native, CA off, matched | 5 | 24.7982 | 0.1418 |
| 3 | validation-native, CA off, matched | 1 | 16.6341 | 0.0169 |
| 4 | validation-native, CA off, wrong full-scene Larry King reference | 5 | 25.3698 | 0.0476 |
| 5 | validation-native, CA off, null reference/identity | 5 | 28.0469 | -0.0019 |

The genuine validation-native CA-off matched row made Dancing, Jumping, and
Skiing coherent, but approximately 9/12 prompts retained missing, displaced,
or oversized facial features. CFG 1 again collapsed into haze. The Larry King
intervention produced coherent facial anatomy in nearly every prompt, while
the null intervention removed or corrupted the face.

## Supported conclusions

1. The fixed evaluator is trustworthy for the two CA-on endpoints because both
   independent reproduction gates passed at exact decoded-pixel equality.
2. Validation-base processor buffers and CFG are not the primary cause of the
   Cosmic failure.
3. Branched CA contributes to broad scene/exterior corruption, but disabling
   it does not remove the dominant face-local geometry failure.
4. The conditioning reference is strongly causal: wrong and null interventions
   change or destroy the face.
5. Reference format/preprocessing is the leading causal variable. A tight
   256px Cosmic reference recreates the pathology on the healthy one-ID
   checkpoint, while a 1024px full-scene reference largely restores anatomy on
   two Cosmic checkpoints.

The metric rise in null rows is not evidence of better images. Text similarity
can improve while the primary face is absent, and identity similarity can
reward identity-bearing fragments. Visual anatomy remains the hard gate.

## Hypotheses not yet established

- The 20%-margin 256px crop may preserve a spatial face grid that branched
  self-attention inserts too literally into target geometry.
- Resolution, crop tightness, surrounding scene context, or their interaction
  may be responsible; Task B does not separate these subvariables.
- Layer coverage or face-mask correspondence may amplify the reference-format
  failure.

## Decision

Task B is complete. Do not pursue CFG 1 or validation-base buffer replacement
as fixes. Keep CA-off for the controlled factorial and prioritize Task D's
`multi_full` versus `multi_cosref` comparison, with `single_full` last. That
factorial is the next controlled test of full-scene versus deterministic
Cosmic-style reference format while holding identity and target images fixed.
