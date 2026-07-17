# N3a critical code audit

Date: 17 July 2026

Baseline: `e42c96604ee73b8b073b3def268beead8c8af684`

Scope: obvious correctness or reproducibility risks only. No model code was
changed as part of this audit.

## What is correct and must be preserved

- Both `BranchedAttnProcessor` and `BranchedCrossAttnProcessor` are installed.
- All 70 self-attention and all 70 cross-attention sites participate when
  `ba_patch_top_k=1`, `disable_branched_sa=false`, and
  `disable_branched_ca=false`.
- The active forward is one doubled `[target, reference]` U-Net call.
- Target face queries use reference spatial K/V in branched self-attention.
- Cross-attention separately conditions the target and reference halves.
- The N3a commit retains the same trained processor objects after validation;
  this prevents the optimizer from updating detached/orphaned modules.
- N3a’s target/noise LR group is a training-control change, not a replacement
  of the BA forward path.

## Critical issue 1: processor installation can fail silently

`install_branched_processors_for_training()` wraps installation and trainable
selection in `except Exception` and only prints the error. Training can
therefore proceed without the intended architecture.

Evidence:

```text
src/model/photomaker_branched/lora2_helpers.py:87-127
```

Required before NN1 implementation:

- make installation failure fatal behind a correctness toggle;
- assert exactly 70 branched `attn1` and 70 branched `attn2` processors;
- assert expected trainable parameter counts before optimizer construction.

This is a correctness guard and does not change branched-attention math.

## Critical issue 2: invalid boxes fail open to the whole image

`_bbox_to_mask()` and `_bbox_to_ref_mask()` fill the mask with ones when a bbox
is missing, invalid, or collapses after conversion. For the target mask, that
can give the reference face branch authority over the entire target image.

Evidence:

```text
src/model/photomaker_branched/lora2.py:586-661
```

Required before NN1:

- validate target and reference boxes before forward;
- fail the synchronized batch/window or skip it consistently across ranks;
- never silently replace a missing target face with a full-image face mask.

## Critical issue 3: failed reference-face recognition becomes a zero identity

When InsightFace finds no face, training substitutes a zero 512-D embedding and
continues through the PhotoMaker ID encoder. This creates invalid conditioning
without marking or synchronizing the sample.

Evidence:

```text
src/model/photomaker_branched/lora2_helpers.py:175-194
```

For two-GPU NN1, invalid-reference handling must be DDP-consistent. A rank-local
skip can desynchronize collectives; a silent zero embedding corrupts the
identity signal.

## Critical issue 4: processor checkpoint restore is non-strict

Processor state is loaded with `strict=False` and no missing/unexpected-key
report. A mismatched or incomplete checkpoint can validate with base/partial
branch weights while appearing to load successfully.

Evidence:

```text
src/model/photomaker_branched/lora2.py:305-317
```

Required before resume-capable NN1:

- record the expected processor-name set;
- verify all selected processors are present;
- report and reject missing/unexpected trainable keys;
- verify object identity after validation swaps.

## Critical issue 5: N3a trains BA at timesteps where inference does not use BA

N3a passes `train_ba_all_steps=true`, while inference uses text-only steps
0-9, PhotoMaker steps 10-14, and spatial BA from step 15. This is an intentional
configuration choice rather than a processor bug, but it is a serious
train/inference mismatch.

Evidence:

```text
serv_new_runs/start_ba_nr_alt_vast_N3a.sh
src/model/photomaker_branched/lora2.py:431-480
```

Do not change this in the exact NN1a control. Test schedule-matched training only
as a later isolated experiment after NN1a reproduces the baseline.

## Critical issue 6: ID-only face prompts retain zero-token attention sinks

The ID-only mode zeros roughly 75 of 77 prompt-token embeddings but leaves them
inside cross-attention. Zero K/V entries can still receive softmax probability,
weakening the useful reference-half prompt signal.

Evidence:

```text
src/model/photomaker_branched/branched_runtime.py:476-492
```

This should be measured, not silently changed. A future option can compare
ID-only tokens with an explicit token mask or a train/inference-consistent
full-boosted prompt while preserving split cross-attention.

## Critical operational issue: credential embedded in launcher

The historical N3a launcher contains a default Comet API credential. This does
not affect model architecture, but it should be rotated and replaced with an
environment-only requirement before distributing or publishing the branch.
The credential is intentionally not reproduced in this document.

## Lower-priority constraints

- `src/model/attn_procs/attn_processor.py` begins with the invalid token
  `phaimport`. It is already broken in `e42c966` and is not imported by the N3a
  training/inference path, which uses
  `photomaker_branched/attn_processor_cleanest.py`; therefore it does not block
  N3a but does make whole-tree `compileall` fail. Leave it untouched for exact
  baseline parity or remove/fix it only as a separately documented cleanup.
- Mask reshaping assumes square attention grids. Current 1024×1024 runs satisfy
  this; non-square NN experiments would require an aspect-aware correctness fix.
- `POSE_ADAPT_RATIO=0` and `CA_MIXING_FOR_FACE=false` are hardcoded. They are
  part of the N3a behavior and should remain unchanged in NN1a.
- Broad face boxes include hair and nearby props. This is a data/routing
  limitation, not evidence that the full BA topology is inactive.

## Audit decision

Do not alter the active N3a model before the NN1 plan is approved. When
implementation is approved, backport only fail-fast/assertion and DDP-validity
guards first, each behind a default-compatible toggle. Prove with a one-batch
test that disabling the guard reproduces the N3a forward exactly.
