# CL23 successor blueprints

These files are **design-only and non-runnable**. They specify the three
controlled successors proposed in the 14 August CL21--CL26 current-results
report. An implementation agent must add the named model/config fields,
trainer logic, fail-closed validators, experiment JSONs, immutable runtime
source packages, and launcher allowlist entries before submission.

Every arm must inherit CL23 and preserve the fixed manual_val-96 contract,
Cosmic manifest, seeds, scheduler, DDIM50, CFG 5, subject-v2 identity metric,
`pipeline.pose_adapt_ratio=0`, `pipeline.ca_mixing_for_face=false`, and
branched cross-attention disabled. All three are cold-start 24k runs; no
experiment checkpoint may be loaded.

Priority order:

1. `CL27_cosmic_frequency_surface_energy_24k`: training-only top-object versus
   visible-surface shaping of CL23's routed frequency message.
2. `CL28_cosmic_learnable_frequency_schedule_24k`: zero-initialized bounded
   per-processor endpoint corrections around CL23's fixed schedule.
3. `CL29_cosmic_lowband_causal_contrastive_24k`: same-ID positive versus
   wrong-ID negative contrastive supervision on low-band reference messages.

The MLS YAMLs under `serv/` point at planned package paths and must not be
submitted until those paths exist and the source manifest/config validators
pass.

