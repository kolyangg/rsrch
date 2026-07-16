# N32 versus N38: second root cause and CFG-strength fix

## Finding

The working N32 implementation was taken from commit `957b28d` (15 July).
The post-CFG composition was introduced later in commit `2126591` (16 July).

The first N38 failure was a schedule bug: the original run resolved
`model.photomaker_start_step == model.branched_attn_start_step == 10`, and the
old equality branch selected PhotoMaker forever. That explains the exactly flat
0k–3k validation from the downloaded log.

After correcting the schedule to PhotoMaker at step 10 and BA at step 15, the
new identity-owner path still remained much weaker than N32. The important
architectural difference is where the BA identity correction enters CFG.

N32 used the legacy pre-CFG composition. Ignoring its unconditional BA term for
clarity, its conditional identity correction was effectively:

```text
output_N32 = PM_CFG + guidance_scale * BA_delta
```

N34–N38 introduced `post_cfg_delta` to prevent BA identity from contaminating
the unconditional branch. The implementation instead computed:

```text
output_old_post_cfg = PM_CFG + BA_delta
```

At the configured guidance scale of 5 this made the same conditional correction
five times weaker. N34–N38 also restrict BA from all SDXL cross-attention sites
used by N32 to 16 up-block sites. Together, those choices made the visible BA
effect far smaller even though the N38 checkpoint contains trained, nonzero BA
weights.

## Checkpoint and validation evidence

- N32 changes every one of the 96 validation images versus the PhotoMaker
  baseline at step 2k; mean absolute pixel difference is about 5.70.
- The downloaded original N38 log never entered `BOTH` or `BRANCHED`, so its
  exactly flat 0k–3k metrics came from the separate schedule bug.
- The N38 epoch-3 checkpoint contains 16 processor states and nonzero learned
  `face_delta_out`, target-ID K/V, and gate tensors. It is reusable.
- Alternate-base validation instantiated its model without the top-level BA
  runtime constructor arguments. This produced the misleading
  `selected_processors=0` diagnostic and made training/validation construction
  unnecessarily different, although the old state-copy path could still install
  the processor modules.

## Fix

1. Added `ba_post_cfg_guidance_scale`.
   - Default: `false`, reproducing the previous post-CFG behavior.
   - Enabled in N34 and therefore inherited by N35, N36, N37, and N38.
   - Corrected CFG composition:

```text
output_fixed = PM_CFG
             + ba_residual_scale * guidance_scale * BA_delta
```

   BA remains absent from the unconditional branch, but its conditional strength
   now matches normal CFG semantics and the scale available to N32.

2. Alternate RealVis validation now receives the same runtime constructor
   arguments as the training model:

```text
train_ba_only
ba_train_top_k
ba_patch_top_k
non_ba_train
train_ba_all_steps
ba_weights_split
use_attn_v2
```

3. Strict validation now fails if any expected trained BA processor cannot be
   copied into the alternate-base model. The log reports:

```text
[BA Validation] copied trained processors 16/16
```

4. The first active BA denoising step logs the actual raw conditional correction
   and applied gain:

```text
[Switch] step 15 -> BOTH
[BA Runtime] conditional_delta abs_mean=... abs_max=... applied_gain=5
```

These diagnostics make another silent PhotoMaker-only validation impossible to
misinterpret.

## Compatibility and affected runs

- N32 and all `legacy_guided` runs are unchanged.
- N34, N35, N36, N37, and N38 use the corrected guidance-scaled post-CFG path.
- Set `model.ba_post_cfg_guidance_scale=false` to reproduce the old weak
  post-CFG behavior.
- Existing N36/N38 checkpoints can be resumed. The new flag changes inference
  composition, not checkpoint tensor shapes, and is intentionally not part of
  the strict architecture tensor manifest.
- `infer.py` treats older saved `post_cfg_delta` checkpoints that predate this
  flag as corrected (`true`) by default. A command-line
  `model.ba_post_cfg_guidance_scale=false` override still reproduces the old
  behavior.

The corrected epoch-3 validation should be inspected before discarding N36 or
N38. If its log shows `BOTH`, `16/16`, a nonzero conditional delta, and gain 5,
then any remaining weakness is an architecture-quality issue rather than a
disabled validation path.
