# NN7a_init implementation and launch

## Outcome

`NN7a_init` is implemented as an additive, checkpoint-separated variant of
NN7a. NN7a itself retains its previous raw-CLIP, Xavier-initialized full K/V
behavior and initial effective gate of approximately `0.02`.

NN7a_init changes only the initialization of the clean spatial branch:

- frozen PhotoMaker V2 Perceiver-context patch tokens, 2048-D;
- each selected `up_blocks.1.attn1` processor clones the effective sibling
  `attn2.to_k/to_v` projections as a frozen, backbone-local base;
- rank-32 branch LoRA deltas are applied to those bases, with LoRA B initialized
  to exact zero;
- direct candidate takeover starts at
  `0.80 * sigmoid(-2.70805020110221) = 0.05`;
- the NN7a face-core mask, local correspondence window, residual caps,
  counterfactual supervision, disabled branched CA, and exact PhotoMaker output
  outside the core are unchanged.

The frozen K/V bases are non-persistent buffers. A checkpoint therefore stores
only branch LoRA A/B and the gate. During RealVis validation, the bases are
rebuilt from the effective PhotoMaker-aware RealVis sibling `attn2`; during
training they are built from the training SDXL backbone.

## Safety and compatibility

The new fields have backward-compatible defaults:

```yaml
ba_spatial_patch_projection: raw_clip
ba_spatial_kv_init: xavier
ba_spatial_kv_kind: full
```

All three are recorded in the strict architecture manifest. Strict restore
accepts NN7a into NN7a and NN7a_init into NN7a_init, but rejects interchange in
both directions.

Initialization guards verify:

- sibling K/V dimensions at every selected site;
- effective K/V parity on deterministic finite tokens;
- exact-zero LoRA B;
- nonzero reference candidate and applied residual on the first valid batch;
- exact-zero spatial residual outside the target core.

## Training and diagnostic launch

On the primary server:

```bash
cd /home/niko/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_init_train_then_diagnose_1gpu.sh
```

This performs:

1. one-GPU training with physical batch 1 and effective batch 2;
2. 2 epochs × 2,000 optimizer steps = 4,000 steps;
3. all 96 manual-validation images on RealVisXL V4.0 during training;
4. a post-training five-condition reference/noise diagnostic on the stable
   24/96 subset (seed `20260722`), using two fixed noise seeds and validation
   batch 12.

To run only the checkpoint diagnostic:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_init_checkpoint_reference_vs_noise_24_1gpu.sh \
  saved/ba_NN7a_init_1gpu/checkpoint-epoch2.pth
```

Before committing GPU time, confirm the logs contain
`patch_dim=2048`, `kv_init=sibling_attn2`, `kv_kind=lora`, K/V parity success,
`effective_gate_init=0.050000`, a nonzero first-batch residual, and
`outside_core_exact_zero=true`.

## Main implementation files

- `src/configs/one_id_ba_NN7a_init.yaml`
- `src/model/photomaker_branched/model_v2_NS.py`
- `src/model/photomaker_branched/packed_residual_attn_processor.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/pipelines/br_pipeline_helpers.py`
- `jul_serv_runs/start_ba_NN7a_init_1gpu.sh`
- `jul_serv_runs/start_ba_NN7a_init_train_then_diagnose_1gpu.sh`
- `jul_serv_runs/start_ba_NN7a_init_checkpoint_reference_vs_noise_24_1gpu.sh`

No commit or push was performed.
