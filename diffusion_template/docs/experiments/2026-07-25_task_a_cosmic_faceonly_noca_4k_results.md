# Task A result: Cosmic one-ID face-only training with branched CA disabled

**Date:** 25 July 2026

**Branch:** `test`

**Git commit:** `437dd043f0b07cb535ee3761a43212a90fa51548`

## Validation-integrity correction

The trainer's saved step-0 through step-4,000 images are **not valid CA-off
evidence**. After training completed, Task B code inspection showed that
`BaseTrainer._evaluation_epoch` instantiated the alternate RealVis validation
model and called `prepare_for_training()` before propagating
`disable_branched_ca`. The temporary pipeline also lacked the disable flag.
Training itself was CA-off, but its temporary validation model installed
randomly initialized branched CA.

A corrected local change now propagates the architecture toggles before
processor installation and onto the temporary pipeline. It is not included in
the historical commit above and must be deployed before future training.
Task A's result below uses the standalone evaluator's explicitly audited
CA-off output, not the trainer's saved images.

## Experiment contract

Task A tested whether branched cross-attention was responsible for the
catastrophic CosmicLarge one-ID result. It changed only the branched-CA
installation/training settings relative to the completed face-only control.

Launcher:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh
```

Run:

```text
rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k
```

Key resolved settings:

```text
train_dataset_name: cosmic_large_one_id
trainer.epoch_len: 500
trainer.n_epochs: 8
trainer.masked_loss_step: 1
disable_branched_ca: true
train_branched_ca_lora: false
model.train_branched_ca_lora: false
branched_attn_weight_mode: noise_and_ref
model.rank: 32
lr_for_lora: 0.0001
lr_scheduler.warmup_steps: 20
continue_run: false
saved_checkpoint: null
trainer.resume_from: null
```

The startup optimizer audit reported 840/840 processor tensors under
`attn1.processor`; no branched `attn2.processor` trainables were selected.

## Runtime and artifacts

- Neb GPU: NVIDIA H100 80 GB.
- Start: 24 July 2026 22:55:03 UTC.
- Epoch-8 checkpoint written: 25 July 2026 00:39:03 UTC.
- Wall time: approximately 1 hour 44 minutes.
- Training endpoint: epoch 8 / step 4,000.
- Validation and checkpoint interval: 500 optimizer steps.
- Validation image count at every endpoint: 12.
- Training GPU allocation was approximately 36–45 GB.
- Fixed validation peaked at approximately 79.3 GB, so concurrent Neb GPU
  jobs are unsafe.

Endpoint:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k/checkpoint-epoch8.pth
```

SHA-256:

```text
440e5404cdb5e6a554c286c52de9d22e822180adaaac8a90228635dcf7866dd3
```

Historical trainer images affected by the validation bug:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k/val_images/cosmic_large_one_id_val/step_4000_batch_0
```

Corrected fixed evaluation:

```text
diagnostics/cosmic_faceonly_noca_4k/row02_validation_native_cfg5_caoff_matched
```

Its manifest reports 70 branched self-attention processors, zero branched
cross-attention processors, validation-native processor bases, CFG 5, the
matched reference, and 12 output images.

No OOM, Python traceback, or fatal training error occurred. Repeated
ONNX Runtime CUDA-provider messages were the existing InsightFace CPU fallback
caused by unavailable cuDNN 9 provider libraries; validation and metrics
completed.

## Observed evidence

### Visual anatomy

The historical step-500 and step-1,000 gates were both severely malformed, and
the run was continued because the strict early-stop condition was not met.
Those images are retained as execution history but cannot establish CA-off
behavior because of the validation bug.

At step 4,000, the corrected genuine CA-off evaluation still did **not**
restore reliable facial anatomy:

- Angry, Chef, Crying, Drumming, Kickboxing, Laughing, Night-ride, Reading,
  and Rushing retained an oversized, missing, or displaced eye/face fragment.
- Dancing, Jumping, and Skiing had coherent primary facial anatomy, although
  identity/expression quality remained imperfect.

The endpoint therefore fails the hard anatomy promotion gate in approximately
9 of 12 prompts.

#### Representative endpoint images

These files are copied from the accepted diagnostic rows so the report remains
visually auditable without access to Neb.

| Prompt | Matched face-only baseline, CA on | Task A, genuine CA off |
|---|---|---|
| Jumping | ![CA-on Jumping output](assets/2026-07-25_task_a/ca_on_jumping.png) | ![CA-off Jumping output](assets/2026-07-25_task_a/ca_off_jumping.png) |
| Reading | ![CA-on Reading output](assets/2026-07-25_task_a/ca_on_reading.png) | ![CA-off Reading output](assets/2026-07-25_task_a/ca_off_reading.png) |

Jumping illustrates the real but limited benefit: removing branched CA repairs
the previously featureless face. Reading illustrates the promotion failure:
the scene is cleaner with CA off, but the target still has missing and
misregistered facial features.

### Matched CA-on comparison

The exact matched baseline was:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_8k/val_images/cosmic_large_one_id_val/step_4000_batch_0
```

Its epoch-8 / 4k checkpoint SHA-256 is:

```text
197aff6f82f898c4f671852c3e780fb3046678e0f420843ab467fa082b9fbf4e
```

Compared prompt-by-prompt, the corrected CA-off run generally retained cleaner
backgrounds, bodies, clothing, and scene geometry. It also repaired Dancing
and Jumping relative to the CA-on baseline. The CA-on baseline more often
introduced broad smoky, melted, or displaced scene content around the subject.

The improvement is primarily global/exterior. The CA-off run still pasted or
misregistered tight facial fragments, so the face-local failure remains.

### Logged validation metrics

| Endpoint | Text similarity | Identity similarity |
|---|---:|---:|
| Task A, genuine CA off, validation-native | 24.7982 | 0.1418 |
| Matched face-only baseline, CA on, legacy reproduced | 23.7565 | 0.0351 |

Task A improved both logged aggregates, but these metrics reward some
identity-bearing facial fragments and do not override the obvious anatomy
failures.

The trainer-reported epoch metrics, including its earlier 0.1102 endpoint
identity score, came from the invalid temporary validation path and must not
be used as CA-off measurements. There is no visual evidence that extending
beyond 4k would repair the dominant anatomy failure.

## Interpretation

### Supported conclusion

Branched cross-attention was a contributor to broad exterior/scene corruption
on the Cosmic one-ID setup. Disabling it improved global coherence and the
logged text/identity aggregates.

Branched cross-attention was not the sole cause. The remaining failure is
concentrated in the branched self-attention spatial-reference path: tight
reference fragments are still inserted without coherent target-face geometry.

### Hypotheses, not yet established

- Trainable target/noise Q/K/V copies may be drifting away from the frozen
  target model and misregistering the reference fragment.
- Tight 256×256 Cosmic-style references may make the spatial grid behave more
  like a pasted face crop than identity evidence.
- Full-layer branched self-attention ownership may lack a usable target-face
  fallback.

Task A alone does not distinguish these explanations.

## Decision

Task A is complete and is **not promoted**.

Its result meets the documented Task C trigger: exterior/global corruption
fell, while pasted, missing, and misregistered face fragments remained.
The isolated CA-off `ref_only` arm was therefore run as Task C.

Task B subsequently passed exact 12/12 reproduction gates for both CA-on
endpoints. Its wrong-reference interventions show that a full-scene Larry King
reference produces mostly coherent anatomy on this checkpoint, while the
matched tight Cosmic reference remains malformed. Task D's controlled
full-scene versus Cosmic-style reference factorial is therefore the next
decision experiment.
