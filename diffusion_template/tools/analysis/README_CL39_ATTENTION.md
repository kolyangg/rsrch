# CL39 attention/confidence audit

Publication results use the exact Serv trainer validation path. The three
validation-only Hydra configs are:

```bash
src/configs/CL39_attention_audit_serv_actual.yaml
src/configs/CL39_attention_audit_serv_c1.yaml
src/configs/CL39_attention_audit_serv_ba_off.yaml
```

Submit them through the hash-checked one-A100 YAML launchers in
`serv_run_packages/CL39_attention_audit_serv_final/`. Each arm runs all 96
validation items with the original batch size 12, 1024px, DDIM50, CFG 5,
prompts, references, boxes, seeds, RealVisXL base, and legacy processor copy.
The actual arm fails closed unless all 96 replay images match the sealed CL39
panel and telemetry exists for the deterministic 16-sample subset.

The three output arms are:

- `actual`: normal CL39, with compact attention telemetry;
- `c1`: the same checkpoint with only applied confidence fixed to `1`;
- `ba_off`: the same checkpoint with the final explicit BA correction fixed to
  `0`, retaining all trained adapters and the native target self-attention path.

The analysis hooks never retain an attention `L×L` matrix. They reduce heads
and query/key axes inside the processor, resize per-query/per-key maps to
64×64, and aggregate by denoising progress and `up0`/`up1`. Capture keeps only
the selected 16 samples while preserving the full batch-12 execution.

After the jobs finish, `assemble_cl39_serv_audit.py` joins their selected PNGs
and telemetry. `analyze_cl39_attention.py render` writes the joined records to
`artifacts/cl39_attention_24k_serv_a100/` and visual panels to
`analysis/assets/cl39_attention_24k_serv_a100/`.

The `generate` stage in `analyze_cl39_attention.py` remains useful for a local
one-item development probe, but it is not a sealed replay because changing
validation batch size changes the diffusion trajectory. Do not use it for
publication comparisons. The notebook
`notebooks/CL39_attention_analysis.ipynb` imports the Serv artifact helpers.

## Direct N / routed-R face views

`CL39_attention_audit_serv_reference_face.yaml` adds an evaluation-only
`reference_face` intervention. At each shipped CL39 processor it uses
`N + router * (R - N)`: raw reference attention replaces native attention
inside the existing soft face router, while the outside region stays on `N`.
It does not represent a new trained model or the ordinary CL39 operating point.

Run it through the hash-checked YAML package under
`serv_run_packages/CL39_attention_audit_serv_branch_faces_r1/`. After the
fixed-96 batch-12 validation completes, `render_cl39_branch_faces.py` combines
the new routed-R arm with the existing exact N-only/BA-off and actual arms. It
writes full-image and fixed-face-crop panels, signed `R-on-face - N` RGB
differences, magnitude overlays, and per-cell image-distance measurements.

## CL19 -> CL23 -> CL27 -> CL39 branch lineage audit

`serv_run_packages/BA_lineage_branch_audit_serv_r1/` contains one Serv MLS YAML
that runs 18 validation-only arms sequentially on one A100. Every arm uses its
immutable 24k checkpoint, fixed-96 `manual_val`, batch size 12, and the existing
trainer validation path. CL19 runs actual and global N-only; CL23 and CL27 add
global raw-R-on-face, low-only, and high-only; CL39 adds the same arms plus
`C=1`. The actual arm for each lineage is checked against its sealed historical
96-image panel before later counterfactuals run. The branch controls attach to
all 70 soft-router/temporal-frequency BA processors, avoiding the earlier
36-processor CL39 confidence-group limitation.

`render_ba_lineage_branch_audit.py` produces matched fixed-face panels and
image-space distances for the deterministic 16-cell view. Low-only/high-only
figures are whole-denoising stress tests and are not linearly additive in RGB
space.
