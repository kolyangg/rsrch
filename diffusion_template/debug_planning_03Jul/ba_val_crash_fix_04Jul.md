# N1 validation crash — root cause & fix (04 Jul 2026)

## Symptom
First validation of the `ba_refonly_N1` run (after epoch 1 / 2000 steps) crashed on the
very first val image, step 0:

```
File ".../attn_processor_cleanest.py", line 92, in _branch_batch_sizes
RuntimeError: Invalid branched batch: total=8, generation=2, reference=6;
              expected one reference per sample
```
(The trailing `c10::Error: invalid device pointer` / SIGABRT is just CUDA/NCCL teardown
after the Python exception aborts the process — not a second bug.)

## What the numbers mean
- `total=8` = 4 val samples × 2 (CFG uncond+cond) — a **normal, undoubled** batch. There are
  no reference latents at step 0.
- `mask.shape[0]=2` → `generation=2` — the mask is **stale from the last training step**
  (train `batch_size=2`), so `_branch_batch_sizes` infers `reference = 8 − 2 = 6` and rejects
  it (6 ≠ 2).
- Crash is at `br_pipeline_helpers.py:1024`, the **non-branched** path (`pipeline.unet(...)`),
  i.e. `branched_attn_start_step=15` had *not* fired yet. So the branched processors were
  running when they should have been swapped out.

## Root cause
Validation gates branched vs. normal attention by swapping the UNet's attention processors
(`set_validation_unet_mode`, `br_pipeline_helpers.py:447`). That function trusted a cached
flag `pipeline._runtime_uses_branched_unet` and early-returned when it already equalled the
requested state.

When validating on the **training base** (`pretrained_model_for_validation_name_or_path=null`,
an N1 change), the validation pipeline **shares the training UNet** and is **reused** across
epochs (built once; `base_trainer.py:562` gates on `self.pipe is None`). The lifecycle:

1. End of each val image → `cleanup_branched_runtime` sets the UNet to *original* processors
   and the flag to `False`.
2. After validation, `ensure_branched_after_eval` (the F9 fix) **re-attaches the trained
   branched processors** to the shared UNet for continued training — but leaves the flag `False`.
3. Next validation, step 0: `set_validation_unet_mode(branched_active=False)` sees
   `False == False`, **skips the swap-to-original**, so the branched processors (carrying the
   stale batch-2 training mask) run on the normal batch-8 input → the split assertion fires.

`cosm_new1_vast` never hit this because it validated on a *separate* base (RealVisXL): a fresh
`_val_model` + fresh pipeline are built for **every** validation (`base_trainer.py:543`), the
training model is offloaded, and the flag/processors never diverge. So this is specifically an
interaction of two N1 changes: **validate-on-training-base** + **F9 re-attach**.

## Fix (two layers)
1. **`set_validation_unet_mode` decides from the actually-attached processors, not the cached
   flag** (`br_pipeline_helpers.py`). It now checks `isinstance(p, Branched*Processor)` over
   `unet.attn_processors` and swaps whenever that disagrees with the requested mode. Self-healing:
   any external processor swap (like `ensure_branched_after_eval`) can no longer desync it.
   Falls back to the old flag only if the type check raises.
2. **Belt-and-suspenders**: `base_trainer._evaluation_epoch` resets
   `self.pipe._runtime_uses_branched_unet = None` at the start of every validation, so even the
   fallback path recomputes the swap on the first denoising step.

Neither touches the branched mechanism, training, or the doubled-batch design. The
`_branch_batch_sizes` assertion is deliberately **kept** — it correctly caught this bug and
still guards against genuine future batch-shape errors; the fix stops the branched processors
from ever being invoked on a non-branched step.

## Verification
- `py_compile` clean on both files.
- `scratchpad/test_val_unet_mode_fix.py` (real `BranchedAttnProcessor` + fake UNet) — 4/4 PASS:
  - **A (bug repro)**: stale `flag=False` + branched attached + request non-branched → now
    swaps to original (before the fix this early-returned and left branched attached → crash).
  - **B**: original attached + request non-branched → no redundant swap.
  - **C**: branched attached + request branched → no redundant swap, flag synced.
  - **D**: original attached + request branched → swaps branched in.

## Files
- `src/pipelines/br_pipeline_helpers.py` — `set_validation_unet_mode` actual-state check.
- `src/trainer/base_trainer.py` — reset `_runtime_uses_branched_unet` at validation start.
