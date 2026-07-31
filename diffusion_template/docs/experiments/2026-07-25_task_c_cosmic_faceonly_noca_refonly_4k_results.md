# Task C result: Cosmic one-ID CA-off reference-only self-attention

**Date:** 25 July 2026

**Branch:** `test`

**Git commit:** `437dd043f0b07cb535ee3761a43212a90fa51548`

## Validation-integrity correction

The trainer's saved validation images and epoch metrics were produced through
the same temporary-model flag-ordering bug found after Task A: branched CA was
installed before the top-level disable flag was propagated. Training itself
used the intended CA-off reference-only processor set, but the saved
step-0 through step-4,000 images are not genuine CA-off evidence.

The corrected result below comes from the standalone evaluator with explicit
processor auditing. A local trainer fix propagates architecture flags before
processor installation and onto the temporary pipeline, but that fix is not
part of the historical commit above.

## Experiment contract

Task C was the conditional follow-up triggered by Task A. Task A showed that
removing branched cross-attention reduced broad scene corruption but left
pasted, missing, and misregistered facial fragments. Task C therefore changed
one architecture variable relative to Task A: the branched self-attention
weight mode was changed from `noise_and_ref` to `ref_only`.

Launcher:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k_1gpu.sh
```

Effective command:

```text
CUDA_VISIBLE_DEVICES=0 \
PM_PATH=/home/niko/models/PhotoMaker-V2/photomaker-v2.bin \
bash launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k_1gpu.sh
```

Run:

```text
rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k
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
branched_attn_weight_mode: ref_only
model.rank: 32
lr_for_lora: 0.0001
lr_scheduler.warmup_steps: 20
continue_run: false
saved_checkpoint: null
trainer.resume_from: null
```

The startup optimizer audit found 420/420 selected processor tensors under
`attn1.processor.ref_to_*`. It found no `noise_to_*` or `attn2.processor`
trainables. This confirms that the intended reference-only ownership was
active rather than merely present in the composed configuration.

## Runtime and artifacts

- Neb GPU: NVIDIA H100 80 GB.
- Run artifacts initialized: 25 July 2026 00:50:50 UTC.
- Epoch-8 checkpoint written: 25 July 2026 02:41:39 UTC.
- Wall time: approximately 1 hour 51 minutes.
- Training endpoint: epoch 8 / step 4,000.
- Validation and checkpoint interval: 500 optimizer steps.
- Validation image count at the endpoint: 12.

Endpoint:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k/checkpoint-epoch8.pth
```

Checkpoint SHA-256:

```text
e4cec1aa1933735314530c2fb66b9e1ef66ad4459485f56078885f0f78050b98
```

Weights-only endpoint:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k/weights-epoch8.pth
```

Weights-only SHA-256:

```text
ab3c4a8d6398b2b4d0c8ea093d128dbff8d9983bea8c554f9b813291c17d23d9
```

Historical trainer images affected by the validation bug:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k/val_images/cosmic_large_one_id_val/step_4000_batch_0
```

Corrected genuine CA-off evaluation:

```text
diagnostics/cosmic_faceonly_noca_refonly_4k/row01_validation_native_cfg5_caoff_matched
```

Its manifest records the expected checkpoint hash, 70 branched
self-attention processors, zero branched cross-attention processors,
validation-native processor bases, CFG 5, the matched reference, and 12
outputs.

No OOM, Python traceback, or fatal training error occurred. Startup emitted
the existing `AttnProcessor2_0` compatibility exception during an attempted
processor install, but execution continued and the subsequent optimizer audit
verified the expected 420 reference-path processor tensors. Repeated ONNX
Runtime CUDA-provider errors were the known InsightFace CPU fallback caused by
the unavailable cuDNN 9 provider library; validation and metrics completed.

## Observed evidence

### Visual gates and endpoint anatomy

The historical step-500 and step-1,000 images remained mostly malformed and
did not meet the strict all-12 early-stop rule. They remain useful as execution
history but cannot establish genuine CA-off behavior.

At step 4,000, the corrected genuine CA-off reference-only evaluation produced
clearly coherent primary faces for Chef, Dancing, and Skiing. The other nine
prompts still failed the hard anatomy gate:

- Angry, Crying, Drumming, Kickboxing, Laughing, Night-ride, Reading, and
  Rushing retained oversized, displaced, duplicated, or incomplete facial
  features.
- Jumping had an effectively blank face rather than a plausible attached face.

The endpoint therefore remains unusable in approximately 9 of 12 prompts.

### Matched visual comparisons

The matched Task A endpoint was:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k/val_images/cosmic_large_one_id_val/step_4000_batch_0
```

The matched historical CA-on endpoint was:

```text
saved/rhca_apr2026_cosmic_large_one_id_faceonly_8k/val_images/cosmic_large_one_id_val/step_4000_batch_0
```

All corrected comparisons used the same 12 prompts, seeds, PhotoMaker
references, reindexed exact automatic face-box cache, scheduler, inference
steps, and CFG.

Relative to the corrected Task A result, `ref_only` repaired Chef but lost
Task A's coherent Jumping result. Both arms retained coherent Dancing and
Skiing, and both failed approximately 9/12 prompts. It did not eliminate the
characteristic oversized-eye and displaced-feature failure.

Relative to the CA-on baseline, Task C retained the cleaner scenes and bodies
established by disabling branched CA. The improvement is therefore real but
partial: reference-only self-attention helps some prompt/pose combinations,
while spatial face registration remains unreliable.

#### Representative Task A versus Task C images

These images are copied from the accepted genuine CA-off diagnostic rows. Task
A uses `noise_and_ref`; Task C changes only the branched self-attention weight
mode to `ref_only`.

| Prompt | Task A: `noise_and_ref` | Task C: `ref_only` |
|---|---|---|
| Chef | ![Task A Chef output](assets/2026-07-25_task_c/task_a_chef.png) | ![Task C Chef output](assets/2026-07-25_task_c/task_c_chef.png) |
| Jumping | ![Task A Jumping output](assets/2026-07-25_task_c/task_a_jumping.png) | ![Task C Jumping output](assets/2026-07-25_task_c/task_c_jumping.png) |

Chef improves from a missing mouth and oversized isolated eye to a complete,
though still exaggerated, face. Jumping moves in the opposite direction:
Task A has an attached face while Task C produces a blank face. This
prompt-level swap explains why `ref_only` did not improve the overall 9/12
failure rate.

### Logged validation metrics

| Endpoint | Text similarity | Identity similarity |
|---|---:|---:|
| Task C, genuine CA off + reference-only | 24.4779 | 0.1484 |
| Task A, genuine CA off + noise/reference trainable | 24.7982 | 0.1418 |
| Matched face-only baseline, CA on, legacy reproduced | 23.7565 | 0.0351 |

Task C's corrected identity score is only slightly above Task A and its text
score is slightly lower. The larger trainer-reported identity values,
including the historical 0.1896 endpoint, came from the invalid temporary
validation path and are not genuine CA-off measurements.

These aggregate metrics do not encode whether eyes, mouths, and face crops are
geometrically attached to the head. They therefore do not override the
approximately 9/12 visual anatomy failure rate.

## Interpretation

### Supported conclusion

Freezing the target/noise Q/K/V copies while training only the explicit
reference-path processor tensors changes which prompts succeed, but does not
materially improve the overall hard anatomy gate over Task A. Both corrected
arms fail approximately 9/12 prompts, and the identity-score difference is
small.

The same face-local pasted or misregistered feature pattern remains dominant.
The experiment does not support target/noise projection drift as the primary
cause.

### Hypotheses, not yet established

- The tight reference crop and its spatial grid may still be treated as a
  literal face patch rather than identity evidence aligned to target geometry.
- Full-layer reference-path application may be too strong even when target
  Q/K/V are frozen.
- Face-mask placement or reference-to-target spatial correspondence may be
  insufficient for pose changes.

Task C does not distinguish among these explanations.

## Decision

Task C is complete and is **not promoted** because it fails the hard facial
anatomy gate in approximately 9 of 12 fixed prompts.

Preserve `ref_only` as an isolated control, but do not treat its small identity
metric difference as evidence of a successful model. Task B's completed
wrong-reference interventions point more strongly to reference
format/preprocessing. Continue with Task D's controlled identity/source
factorial; do not promote this checkpoint to a long full-Cosmic run.
