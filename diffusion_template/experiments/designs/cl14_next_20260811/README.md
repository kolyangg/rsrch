# CL14 next-architecture experiment blueprints

These six files are implementation specifications, not runnable Hydra configs.
They intentionally live outside `src/configs/` and set `launchable: false` so a
design cannot be submitted to Serv before its code path, checkpoint semantics,
and training/validation parity have been verified.

Priority order is the filename order. Each arm inherits the CL14 architecture
and changes one mechanism or data policy. The architecture arms must retain
explicit target-query/reference-key-value branched self-attention, PhotoMaker
conditioning, `pipeline.pose_adapt_ratio=0`, and
`pipeline.ca_mixing_for_face=false`.

## Fixed comparison contract

- Base config: `src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`.
- 24,000 optimizer steps, batch size 2, one A100, 2,000-step epochs.
- Validate at step 0 and every 2,000 steps on the unchanged 96-image
  `manual_val` panel, one image per item.
- Preserve prompts, seeds, reference images, face boxes/masks, scheduler, 50
  inference steps, CFG, metric definitions, and corrected subject-v2 identity
  embeddings.
- Preserve CL14's hard BA rank 128, generic LoRA rank 32, PhotoMaker default
  rank 64, shadow validation, uniform-all training timesteps, reference
  zero-sink behavior, and Cosmic reference-scale policy unless the blueprint
  explicitly changes that single axis.
- Every implementation must report the exact trainable tensor/parameter
  inventory and must install/load the same processors and gates in training and
  validation.
- Startup is incomplete until `saved/<run_name>/comet_experiment.json` contains
  the immutable Comet experiment key.
- Before any submission, inspect Serv Running/Pending MLS jobs and keep this
  project's total at or below six requested A100s.

## Promotion sequence

1. Implement the proposed keys behind backward-compatible defaults.
2. Run config composition, import/compile, processor-installation, forward,
   backward, checkpoint-save/load, and deterministic train/validation parity
   checks.
3. Run a 100-step local/Serv smoke with branch telemetry. Do not infer quality
   from it; it is only a correctness and memory check.
4. Set `launchable: true` only in a copied active config after review. Do not
   rename the blueprint or silently change its comparison contract.
5. Apply the blueprint's intermediate kill rule and final decision gate.

The accompanying report is
`analysis/2026-08-11_cl14_hard_cases_architecture_research_and_experiment_plan.md`.

