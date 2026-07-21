# NN5a / NN5b implementation and launch handoff

**Date:** 21 July 2026  
**Source specification:** `2026-07-21_NN4_independent_results_review_and_NN5a_implementation_spec.md`

**Validation-base decision:** NN5 validation uses `SG161222/RealVisXL_V4.0`
for normal validation and checkpoint diagnostics. The fixed `face_bbox_gen`
annotations were measured on RealVisXL generations, so validating on the base
SDXL checkpoint would misalign the target branched-attention masks.

## Implemented experiments

### NN5a — counterfactual directional PPR

NN5a preserves NN4's protected operator exactly: ordinary target self-attention,
target queries over packed spatial-reference K/V, learned-null subtraction,
zero-initialized connector-up, gate/RMS cap, inner-core routing, independent
PhotoMaker output outside the core, `up_blocks.0.attn1` only, neutral reference
text/pooled conditioning, no branched CA, and paired CFG reference noise.

The changed variable is supervision. At `t <= 300`, one physical target is
duplicated exactly and run once with matched identity A and once with a sampled
different identity B. Target latent, target diffusion noise, timestep, prompt,
PhotoMaker-A conditioning, target mask, and reference noise are exact pairs.
Only the matched row enters core diffusion reconstruction. The B row receives:

- absolute identity-to-B loss, weight `0.05`;
- directional `sim(output,B)-sim(output,A)` margin loss, weight `0.10`, margin `0.03`;
- boundary-ring consistency against the matched prediction, weight `0.05`.

Matched ID supervision remains at `0.025`; learned-null and cap losses remain at
`0.10` and `0.01`. PM attenuation and matched/null magnitude margin are off.

### NN5b — clean PhotoMaker-V2 identity-token lane

NN5b is a separate config inheriting NN5a's causal supervision. It adds the two
unpooled 2048-D PhotoMaker-V2 QFormer/Perceiver tokens from the *spatial*
reference. Every selected PPR site computes a separate target-Q identity
candidate through low-rank identity K/V projections and fuses it 50/50 with the
packed spatial candidate before the existing learned-null connector. The
connector-up remains zero-initialized, so step-zero output parity and the NN4
output anchor are preserved.

The `ba_identity_token_lane` toggle is false in NN5a and true only in NN5b.
During reference-swap diagnostics, identity tokens are recomputed from the
swapped spatial reference while target PhotoMaker conditioning remains A.

## Data and routing changes

`CosmicLargeTrain` can now return a wrong-identity reference. Identity keys use
explicit identity metadata when present and otherwise the parent directory of
the first face path. B is sampled from a different identity, preferably from the
same coarse prompt class. Both references use the same crop, flip, and sharpness
augmentation path.

All additions are default-off. Existing configs retain their old data fields,
paired-forward behavior, and processor topology.

## Configs and launchers

- NN5a config: `src/configs/one_id_ba_NN5a_counterfactual_directional_ppr.yaml`
- NN5b config: `src/configs/one_id_ba_NN5b_clean_identity_tokens.yaml`
- Main-server NN5a: `jul_serv_runs/start_ba_NN5a_counterfactual_directional_ppr_1gpu.sh`
- NN5a 2k/4k approval test: `jul_serv_runs/start_ba_NN5a_checkpoint_reference_vs_noise_1gpu.sh`
- Secondary-server NN5b wrappers: `serv_new_runs/start_ba_NN5b_clean_identity_tokens_{1gpu,2gpu}.sh`
- Secondary-server job files: `serv_new_runs/run_ba_NN5b_clean_identity_tokens_{1gpu,2gpu}.yaml`

Both experiments use global effective batch 2. NN5a uses physical batch 1 and
two-step accumulation. NN5b uses the same on one GPU; on two GPUs it uses one
sample per rank without accumulation. The initial approval budget is 4k optimizer
steps (2 × 2k), with validation at step 0, 2k, and 4k. Increasing `NUM_EPOCHS`
requires an explicit decision after the causal gate.

## Launch commands

Main server, NN5a:

```bash
bash jul_serv_runs/start_ba_NN5a_counterfactual_directional_ppr_1gpu.sh
```

NN5a RealVisXL scale-1 causal gate at 2k or 4k:

```bash
bash jul_serv_runs/start_ba_NN5a_checkpoint_reference_vs_noise_1gpu.sh \
  /absolute/path/to/checkpoint-epoch1.pth
```

Secondary server, choose one job only:

```bash
mls job submit --config ./serv_new_runs/run_ba_NN5b_clean_identity_tokens_1gpu.yaml
mls job submit --config ./serv_new_runs/run_ba_NN5b_clean_identity_tokens_2gpu.yaml
```

## Verification completed locally

- both Hydra configurations compose with their real launcher overrides;
- shell syntax checks pass;
- Python compilation passes in the PhotoMaker environment;
- 36 existing packed-residual/runtime tests pass (one pre-existing skip);
- 5 NN5 component tests pass, covering counterfactual dataset collation,
  distinct identity keys, directional-loss behavior and gradient flow,
  zero-connector parity, identity-token sensitivity, and diagnostic scale.

No training was started and no commit or push was made.
