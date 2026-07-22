# NN6a implementation and secondary-server launch

**Date:** 22 July 2026  
**Run:** NN6a factorized clean identity residual, identity-only at `up_blocks.0`

## Implemented architecture

NN6a preserves ordinary PhotoMaker target self-attention as the protected base
and adds only this face-core residual at selected `up_blocks.0.attn1` sites:

```text
PMv2 clean identity tokens (2 x 2048)
  -> shared identity K/V projections
  -> target-query identity candidate

learned identity-null tokens (2 x 2048)
  -> the same identity K/V projections
  -> target-query identity-null candidate

candidate - null
  -> dedicated bias-free rank-16 connector
  -> identity RMS cap 0.15
  -> identity gate max 0.50
  -> target inner-face core only
  -> ordinary target self-attention + identity residual
```

The noised spatial-reference lane is disabled and its K/V, null, connector and
gate are not instantiated or optimized. Branched cross-attention, pose
adaptation and CA face mixing remain disabled. `base_outside_core` keeps final
epsilon exactly equal to ordinary PhotoMaker outside the protected core.

The NN5a counterfactual objective and weights are unchanged. This makes NN5a /
NN5b versus NN6a an architecture comparison rather than a simultaneous loss
change.

## Reversible controls

The processor supports:

- `ba_identity_fusion_mode: blend` — unchanged NN5b behavior;
- `ba_identity_fusion_mode: identity_only` — NN6a;
- `ba_identity_fusion_mode: factorized_dual` — separately bounded identity and
  spatial lanes for a later run;
- independent identity/spatial site policies, including `up_blocks1_attn1`;
- independent identity/spatial gates and caps plus a total cap.

NN6a itself is configured in
`src/configs/one_id_ba_NN6a_factorized_identity_only_up0.yaml`. The old NN5
configs remain unchanged.

## Strict checks added

- NN6a optimizer groups contain only identity K, identity V, identity null,
  connector-down, connector-up and gate parameters.
- Strict checkpoint manifests record every factorized-lane architecture field.
- Per-lane tensor signatures are written by the causal diagnostic.
- In identity-only mode the diagnostic fails if changing only reference noise
  changes identity candidate/null/input/raw/bounded/applied tensors, target
  epsilon, or final RGB pixels. The default tolerance is exact zero.
- Directional summaries now treat the 96 target means as the primary `all`
  bootstrap sample and report both-noise-positive fraction, noise-sign-flip
  fraction, mean B/A changes and per-identity means.

## Combined secondary-server jobs

Each job trains exactly 4,000 optimizer steps (`2 x 2,000`), checks that
`checkpoint-epoch2.pth` exists, and then runs the 96-image RealVisXL V4.0,
scale-1, five-condition reference/noise diagnostic with validation batch 12.

One GPU:

```bash
mls job submit --config ./serv_new_runs/run_ba_NN6a_factorized_identity_only_up0_combined_1gpu.yaml
```

Two GPUs:

```bash
mls job submit --config ./serv_new_runs/run_ba_NN6a_factorized_identity_only_up0_combined_2gpu.yaml
```

Both use global effective training batch 2. The one-GPU job uses physical batch
1 with two-way accumulation; the two-GPU job uses physical batch 1 per rank
without accumulation. The two-GPU job releases DDP and uses GPU 0 for the final
diagnostic.

Expected outputs are:

```text
saved/ba_NN6a_factorized_identity_only_up0_nfs_{1gpu|2gpu}/checkpoint-epoch2.pth
ppr_ba_NN6a_factorized_identity_only_up0_nfs_{1gpu|2gpu}_4000step_realvis_scale1_reference_vs_noise/
```

The job stops after diagnostics; there is no automatic continuation beyond 4k.

## Verification performed locally

- Python compilation for every modified model/trainer module;
- Hydra composition of the NN6a config, including fixed RealVis validation;
- shell syntax checks for all NN6 launch scripts;
- YAML parsing for both combined jobs;
- existing packed-residual/NN5 parity suite: 37 tests passed, 1 optional matrix
  skipped;
- NN5 component suite: 10 tests passed;
- new identity-only tests cover exact spatial/reference-noise independence,
  shared real/null K/V routing, token sensitivity after connector opening,
  gradients, exact trainability and optimizer grouping.
