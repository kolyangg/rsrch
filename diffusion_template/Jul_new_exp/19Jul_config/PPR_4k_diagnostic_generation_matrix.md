# PPR 4k diagnostic generation matrix

## Goal

Determine whether the trained PPR correction is:

1. not reaching inference;
2. removed by `base_outside_core`;
3. active but too weak; or
4. active but insensitive to the spatial reference.

Use the **same 4k checkpoint, RealVis base, 96 validation samples, seeds, prompts, masks, scheduler, CFG and reference noise** for every option. Load the checkpoint once with its original strict architecture, then apply diagnostic overrides only at runtime.

## Images to generate

| ID | Option | Runtime behaviour | Purpose |
|---|---|---|---|
| A | Exact PhotoMaker control | At BA steps return the ordinary single-target `base_noise_pred`; do not use a doubled-call result | True branch-off baseline without BF16 doubled-batch drift |
| B | Current PPR | Unchanged checkpoint: scale `1`, `base_outside_core` enabled | Reproduce the 4k result |
| C | PPR without final anchor | After strict loading, set pipeline `ba_output_anchor_mode="none"` | Test whether the final output mask removes the learned effect |
| D | PPR ×4 | Keep the current anchor but multiply the **applied** processor residual, after gate, by `4` | Test whether PPR is active but too weak |
| E | Spatial-reference swap | Keep PhotoMaker ID embeddings/prompt conditioning from the original reference, but replace only PPR reference latents and reference mask with another identity | Test whether the PPR memory lane affects output |

Generate A–D for all 96 samples. Generate E for at least 12 representative samples with small, medium and large faces.

## Important implementation details

- Do not implement A by setting the gate to zero: a zero-gate doubled U-Net call can still differ numerically from ordinary PhotoMaker. Force the ordinary `base_noise_pred` result.
- Do not change checkpoint architecture fields before strict loading. Apply C only to the constructed validation pipeline after the checkpoint has loaded successfully.
- Add a runtime-only `ba_ppr_runtime_scale` defaulting to `1.0`. Apply it to `target_core * gate * bounded_delta`; use `4.0` for D and restore it after generation.
- For E, decouple `ppr_reference_image` from `input_id_images`: the former supplies only reference VAE latents/mask, while the latter continues to supply PhotoMaker ID conditioning.
- Use a fresh pipeline call and reset generation caches for every option.

## Required measurements

For every sample and option, save:

- final PNG;
- SHA-256;
- whole-image and face-core pixel MAE versus A;
- ID similarity and text similarity;
- at BA steps 15, 25, 35 and 49:
  - `RMS(epsilon_variant - epsilon_base) / RMS(epsilon_base)` inside the core;
  - the same ratio outside the bbox;
  - ratio before and after the final anchor.

Also log per selected PPR processor:

```text
applied_ratio = RMS(target_core * gate * runtime_scale * bounded_delta)
                / RMS(target_base)
```

Current `post_cap_ratio` is insufficient because it excludes the gate and runtime/output masks.

## Output layout

```text
ppr_4k_diagnostic/
  A_exact_pm/
  B_current_ppr/
  C_no_anchor/
  D_ppr_x4/
  E_reference_swap/
  metrics.csv
  epsilon_diagnostics.jsonl
  contact_sheets/
```

Use identical filenames across A–D. Create contact sheets with columns A/B/C/D and optional E.

## Interpretation

- **B ≈ A; C differs:** final anchor/core suppresses PPR.
- **B and C ≈ A; D differs:** learned residual is real but too weak.
- **D ≈ A and applied ratios are zero:** routing, masks or processor activation remain broken.
- **D differs but E does not:** PPR produces a generic correction rather than using reference evidence.
- **E differs and B has measurable deltas, but ID similarity does not improve:** inference works; the learned objective/direction is the problem.

Do not start another training run until this matrix is complete.
