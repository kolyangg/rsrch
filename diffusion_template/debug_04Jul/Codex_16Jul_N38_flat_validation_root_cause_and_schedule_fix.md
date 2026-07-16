# N38 flat validation: root cause and concise fix

Date: 16 July 2026

## Result

The flat N38 validation was caused by a validation/inference scheduling bug, not by a frozen
optimizer or an empty checkpoint.

N36 has the same bug. N37 also has it because both inherit N36's equal start-step setting.
N34 and N35 do not hit this particular bug because their PhotoMaker and BA start steps are
different.

## Primary failure

N36/N37/N38 configure:

```yaml
photomaker_start_step: 10
branched_attn_start_step: 10
branched_start_mode: both
```

In `select_mode_and_prompts`, the no-end-step schedule handled:

- `photomaker_start_step < branched_attn_start_step`;
- `branched_attn_start_step < photomaker_start_step`;
- but not equality.

Equality fell into the second case. Because `min(start) == max(start) == 10`, the condition
that would activate BA was already false at step 10, and the function selected
`PHOTOMAKER` forever.

The observed N38 validation sequence was therefore:

```text
step 0  -> NO_ID
step 10 -> PHOTOMAKER
step 11 ... 49 -> PHOTOMAKER
```

It never entered `BOTH` or `BRANCHED`.

This exactly explains why:

- 1k, 2k, and 3k images looked identical to PhotoMaker;
- `manual_val/id_sim` was exactly `0.525783022477602` at every epoch;
- `manual_val/text_sim` was exactly `26.312337239583332` at every epoch;
- the validation log contained no `[Switch] ... BOTH` or `BRANCHED` event.

The temporary validation-model setup also printed `selected_processors=0`. That diagnostic
describes its training-selection bookkeeping, not the denoising schedule, so it is not used
as the primary proof. The repeated `NO_ID -> PHOTOMAKER` switches prove that the BA runtime
was never entered.

## Training checkpoint audit

The epoch-3 N38 checkpoint is not empty and training was not frozen:

- 16 BA processor states are saved.
- `face_delta_out.up.weight`, which is initialized to zero, has aggregate norm `15.42`.
- `target_id_to_k.lora_B` has aggregate norm `6.72`.
- `target_id_to_v.lora_B` has aggregate norm `9.23`.
- gates changed from their configured initial values.
- all 112 intended BA tensors / 4.23M parameters were present in the optimizer.

The checkpoint can therefore be reused. The previous validation simply never executed those
trained processors.

The near-zero `correct_gain` and `wrong_gain` training curves remain an architectural warning:
the learned correction may still have weak reference-specific separation. They are not the
cause of the pixel-identical validation images, because the complete BA path was disabled
during every validation. Corrected validation must be inspected before deciding whether the
identity objective also needs retuning.

## Fix

An explicit equality branch was added:

```python
elif photomaker_start_step == branched_attn_start_step:
    mode = "BOTH" if branched_start_mode == "both" else "BRANCHED"
```

The equality case is now corrected in the runtime, but the new N36/N37/N38 configs restore
the established N34/N35 staged schedule:

```text
step 0 ... 9  -> NO_ID
step 10 ... 14 -> PHOTOMAKER
step 15 ... 49 -> BOTH
```

N36 and N37 declare this schedule explicitly in both their `pipeline` and `model` sections;
N38 inherits it from N36. This retains five denoising steps for PhotoMaker to establish pose
and global rendering before the BA identity correction becomes active.

## Regression coverage

Added `tests/test_branched_identity_schedule.py`, covering:

- equal start steps activate `BOTH`;
- equal start steps activate `BRANCHED` in branched-only mode;
- the existing staggered PhotoMaker-then-BA schedule remains unchanged.
- composed N36, N37, and N38 configs resolve PM start to 10, BA start to 15, and
  `branched_start_mode=both` in model, pipeline, and validation arguments.

The focused test passes under the `photomaker` conda environment.

## Impact by run

| Run | Start steps | Affected? | Interpretation of previous validation |
|---|---:|---:|---|
| N34 | PM 10, BA 15 | No | BA was active from step 15; its PM dominance was an architecture/authority issue |
| N35 | PM 10, BA 15 | No | Same as N34 |
| N36 previous run | PM 10, BA 10 | Yes | Every validation was PhotoMaker-only |
| N37 previous config/run | PM 10, BA 10 | Yes | Validation was PhotoMaker-only with the old runtime |
| N38 previous run | PM 10, BA 10 | Yes | 0k–3k validation did not evaluate the trained BA checkpoint |
| N36/N37/N38 corrected configs | PM 10, BA 15 | No | PM runs at 10–14; BA is active in `BOTH` from step 15 |

## Recommended recovery

Do not discard the N38 epoch-3 checkpoint. Stop any still-running N36/N37/N38 process because
an already-running Python process retains the old scheduling code.

To continue N38 from epoch 3 using the corrected code and validate the loaded
checkpoint before any new optimizer step:

```bash
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29538 \
bash serv_new_runs/start_ba_identity_owner_cropped_qformer_2gpu_N38.sh \
  full_step0_val \
  continue_run=true \
  saved_checkpoint=checkpoint-epoch3.pth
```

This resumes the optimizer and scheduler, runs the corrected 96-image validation at
step 3000, and only then starts epoch 4.

The original initial-validation condition was `epoch == 1`, which silently skipped
`full_step0_val` on resumed runs because an epoch-3 checkpoint starts at epoch 4.
It now checks `epoch == self.start_epoch`, preserving fresh-run behavior while also
validating a resumed checkpoint at the start of the new invocation.

N36 can run concurrently on the other two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29536 \
bash serv_new_runs/start_ba_identity_owner_qformer_2gpu_N36.sh \
  full_step0_val \
  continue_run=true \
  saved_checkpoint=checkpoint-epoch3.pth
```

Both commands require the checkpoint under the matching default run directory in
`saved/`. Do not override `RUN_NAME` when resuming these checkpoints. For N37, apply
the same procedure with its own matching run directory and checkpoint.
