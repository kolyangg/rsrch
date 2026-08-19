# CL27 successor suite - design handoff

These files are **design-only and non-runnable**. They describe eight controlled
successors selected after the matched CL23/CL27/CL28/CL29 review on 17 August
2026. An implementation agent must add the named defaults-off fields, loss
code, fail-closed validation, immutable experiment JSONs, a new launcher
allowlist, sealed runtime packages, and startup Comet-key verification before
submitting any Serv YAML.

The principal cold-start base is
`CL27_cosmic_frequency_surface_energy_24k`. CL36 is the only continuation: it
must load the immutable CL27 r3 **16k** checkpoint and train for 4k local steps.
CL35 adds one reward to CL31 and therefore depends on the CL31 implementation,
but it remains a separate cold-start experiment.

## Priority order

1. CL30 positive-only low-band same-ID attraction.
2. CL31 synthetic-mask-supervised attention ownership alignment.
3. CL32 contact-ring-partitioned frequency-surface loss.
4. CL33 visibility-normalized weighted reconstruction loss.
5. CL34 one shared, narrowly bounded frequency schedule correction.
6. CL35 attention-gated masked-patch DINO identity reward.
7. CL36 BA-only low-noise ArcFace hinge continuation from CL27-16k.
8. CL37 small-face ROI-teacher distillation into the ordinary CL27 path.

## Non-negotiable common contract

- Keep PhotoMaker plus explicit target-Q/reference-KV branched self-attention.
- Keep `pipeline.pose_adapt_ratio=0` and
  `pipeline.ca_mixing_for_face=false` in training and validation.
- Keep the same Cosmic manifest, optimizer/data seed, fixed 96-image
  `manual_val` panel, prompts, references, face boxes, DDIM50, CFG 5, and
  subject-v2 metric. Validate step 0 and every 2,000 optimizer steps.
- Keep `trainer.epoch_len=2000`; cold-start runs use `n_epochs=12`, while CL36
  uses `n_epochs=2` after loading the exact CL27-16k checkpoint.
- Apply the optimized pipeline from
  `analysis/2026-08-16_training_pipeline_processor_lookup_fix.md`: cache
  `unet.attn_processors` once per collector; skip disabled collectors; use
  `trainer.active_grad_norm_mode=requested_only`; disable full-activation BA
  telemetry; sample every optional low-band gate on CPU; and never allocate a
  disabled semantic mask. These are required execution controls, not
  scientific deltas.
- Preserve the CL27 trainable inventory of `2,240 tensors / 219,217,920
  parameters` unless the blueprint says otherwise. Frozen reward encoders are
  not optimizer parameters.
- A defaults-off composition must reproduce CL27. For loss-only cold starts,
  step zero must be byte-identical to the implemented CL27 source package. The
  CL23-to-successor source-revision pixel drift found in this review makes this
  an explicit release gate.
- Each loss must be added exactly once. In particular, any new criterion must
  retain CL27's live `ba_aux_loss`; switching to a criterion that silently
  drops the frequency-surface loss is a hard failure.

The Serv YAMLs in `serv/` are placeholders that point to planned package paths.
Do not submit them until the package manifest, Hydra composition, source hash,
trainable contract, one-batch forward/backward, checkpoint reload, validation
processor installation, and Comet record checks all pass.
