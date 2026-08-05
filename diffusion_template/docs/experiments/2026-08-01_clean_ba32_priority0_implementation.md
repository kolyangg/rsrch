# Clean BA32 Priority-0 implementation

**Date:** 1 August 2026  
**Status:** implemented, verified, and running on Neb

## Purpose

This change establishes a trustworthy rank-32, SA-only branched-attention
baseline before changing BA capacity or attention math. It fixes the
fail-open trainable-state and incomplete-checkpoint problems found in the
Large Dataset and BigCelebs audit.

The attention computation, rank, loss, optimizer, LR schedule, BigCelebs
sampling plan, and fixed-96 validation inputs remain unchanged.

## Changes

- BA processor installation can now fail closed instead of swallowing errors.
- Strict runs derive an exact trainable allowlist and audit both
  `requires_grad` and optimizer membership before Accelerate wraps the model.
- The expected clean configuration trains exactly **840 tensors / 31,948,800
  parameters**. The pretrained PhotoMaker adapter and generic U-Net adapter
  remain frozen.
- Schema-v2 checkpoints save every trainable U-Net tensor with an architecture
  manifest and exact name/shape checks. Historical schema-v1 checkpoints retain
  their original loader.
- Alternate-base validation now has explicit processor-base modes:
  `legacy_full_copy`, `validation_native`, and `no_processor_update`.
- The first clean control pins `legacy_full_copy` explicitly for historical
  comparability. A native-base validation run must use a separately recorded
  experiment/protocol.

## Reversible toggles

The historical behavior remains the default when these settings are absent:

```yaml
model:
  strict_branched_install: true
  strict_trainable_contract: true
  branched_state_dict_mode: trainable_v2

validation_processor_base_mode: legacy_full_copy
strict_validation_processor_copy: true
```

Use the existing historical configs/launchers, or set the model switches to
`false` and `branched_state_dict_mode: legacy`, to restore the old behavior.

## Main files

- [`src/model/photomaker_branched/lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py): fail-closed installation and exact trainable/optimizer contract.
- [`src/model/photomaker_branched/lora2.py`](../../src/model/photomaker_branched/lora2.py): schema-v2 complete trainable-state save/load and schema-v1 compatibility.
- [`train.py`](../../train.py): pre-Accelerate ownership audit and corrected processor reporting.
- [`src/trainer/base_trainer.py`](../../src/trainer/base_trainer.py): explicit validation processor-base modes and strict legacy copying.
- [`src/configs/big_celebs_scheduled_rhca_clean_ba32_40k.yaml`](../../src/configs/big_celebs_scheduled_rhca_clean_ba32_40k.yaml): isolated clean BA32 configuration.
- [`launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh`](../../launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh): prepared Neb launcher.
- [`experiments/big_celebs/rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1.json`](../../experiments/big_celebs/rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1.json): prepared immutable experiment plan.

## Verification

Passed checks:

- clean and historical Hydra composition;
- Python compilation and shell syntax;
- experiment JSON validation and launcher runtime hash locks;
- reproduction of the historical plain-processor warning path;
- strict BA allowlist and exact optimizer membership;
- schema-v2 tensor round-trip;
- rejection of an extra optimizer parameter;
- rejection of an incomplete schema-v2 checkpoint;
- explicit validation-mode selection and strict processor copying.

No commit or push was created.

## Neb launch

The verified Neb launch command is:

```bash
cd /home/niko/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0 \
  bash launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh
```

Startup must report the exact 840-tensor / 31,948,800-parameter contract and
must create `saved/<run_name>/comet_experiment.json` containing the immutable
Comet key before the run is treated as valid.

The run `rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1` passed those
gates on 1 August 2026. Its immutable Comet key is
[`700240d8f90b48cfa2cc16f8ff2886b6`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/700240d8f90b48cfa2cc16f8ff2886b6).
Step-0 produced 96 validation images and face-quality scoring detected 94
faces; more than three training batches then completed successfully, so the
detached process was left running.
