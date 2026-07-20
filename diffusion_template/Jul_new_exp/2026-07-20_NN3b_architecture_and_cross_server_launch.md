# NN3b architecture and cross-server launch

Date: 20 July 2026

## What NN3b tests

NN3b keeps the NN2-PPR/NN3a branched-attention core:

- a doubled target/reference U-Net call;
- target spatial queries attending to packed reference-face K/V;
- selected `up_blocks.*.attn1` sites only;
- frozen split target/reference cross-attention;
- a bounded, gated additive residual inside the feathered target-face core;
- the ordinary PhotoMaker prediction outside that core;
- a small decoded reference-ID loss at low timesteps.

It changes two things that were missing from the weak NN3a run.

### 1. Real learned null candidate

NN3a subtracted a zero tensor after attention:

```text
C_ref - 0 = C_ref
```

NN3b gives every selected self-attention site eight trainable null-memory
tokens. Both candidates use the same target query and the same reference K/V
projections:

```text
C_ref  = Attention(Q_target, K(packed reference face), V(packed reference face))
C_null = Attention(Q_target, K(learned null memory), V(learned null memory))
delta  = Connector(C_ref - C_null)
```

The null memory starts at zero and learns a query-dependent generic baseline.
It cannot contain per-sample reference identity because it is shared across
the dataset.

### 2. Controlled target PhotoMaker-ID attenuation

With physical batch two, exactly one sample keeps full target PhotoMaker
identity and one has only the PhotoMaker identity delta removed:

```text
50%: target prompt = full PhotoMaker prompt
50%: target prompt = pre-ID-fusion base prompt
```

The reference stream keeps the full same-identity prompt in both cases.
Validation and deployment also keep full target PhotoMaker conditioning.
This makes spatial BA useful during training without changing the inference
distribution for the comparison panel.

This is a concise dependence-enforcing screen. It does not add a second
full-U-Net null-reference forward, which would double the already doubled
training call and is likely to exceed the one-GPU memory budget. The null
candidate is paired with the matched candidate inside every selected
attention site instead.

## Reversible controls

NN3b is opt-in:

```yaml
model:
  ba_connector_input_mode: reference_minus_learned_null
  ba_null_memory_tokens: 8
  ba_pm_id_attenuation_probability: 0.50
  ba_pm_id_attenuation_scale: 0.0
  ba_reference_ca_preserve_full_pm: true
```

Existing behavior is restored with:

```yaml
model:
  ba_connector_input_mode: reference_minus_null  # NN3a
  ba_pm_id_attenuation_probability: 0.0
  ba_pm_id_attenuation_scale: 1.0
  ba_reference_ca_preserve_full_pm: false
```

NN2 behavior uses `ba_connector_input_mode=reference_minus_target`.

## Training protocol

All launchers use:

- 20k optimizer steps: 10 epochs × 2k steps;
- fixed 96-image RealVis validation at step 0 and every 2k;
- train batch 2 per GPU;
- validation batch 12 per process;
- BF16, rank 32, LR `5e-5`, 2k warmup;
- training seed 0 and validation seed 0;
- PhotoMaker at denoising steps 10–14 and BA from step 15.

The one-GPU runs have global batch 2. The two-GPU server run has global batch
4, matching the old two-GPU server convention of batch two per rank.

## Current `/home/niko` server: one GPU

Dataset:

```text
cosmic_large_neb
/home/niko/datasets/gathered_data_cosmic_large_filtered.json
/home/niko/datasets/LAION-5B-Filtered-Large-Faces/laion1B-nolang
```

Launch:

```bash
cd /home/niko/rsrch/diffusion_template
bash jul_serv_runs/start_ba_NN3b_learned_null_pm_attenuation_realvis_1gpu.sh
```

The run starts detached by default. Use `RUN_FOREGROUND=1` for an attached
process.

## NFS/MLSpace server

These launchers use the same paths as `run_cosm_new3_2gpu.yaml`:

```text
project:
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template

conda:
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS

dataset config:
cosmic_large

dataset JSON:
/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json

dataset images:
/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/LAION-5B-Filtered-Large-Faces/laion1B-nolang
```

The PhotoMaker path and Comet key are loaded from `serv_new_runs/.env`, as in
the older server launchers.

Submit one GPU:

```bash
mls job submit \
  --config ./serv_new_runs/run_ba_NN3b_learned_null_pm_attenuation_1gpu.yaml
```

Submit two GPUs:

```bash
mls job submit \
  --config ./serv_new_runs/run_ba_NN3b_learned_null_pm_attenuation_2gpu.yaml
```

The job YAML deliberately requests one scheduler process. The shell script
then starts one or two distributed workers through `accelerate`, matching the
pattern in `run_cosm_new3_2gpu.yaml`.

Direct shell launch on that server is also supported:

```bash
bash serv_new_runs/start_ba_NN3b_learned_null_pm_attenuation_1gpu.sh
bash serv_new_runs/start_ba_NN3b_learned_null_pm_attenuation_2gpu.sh
```

## Early checks

At startup, verify:

```text
connector_input=reference_minus_learned_null
sa_null_memory: tensors=36
ba_ppr_null_memory optimizer group is non-empty
ba_conditioning/pm_id_attenuated_fraction ~= 0.5
```

Step-zero validation should still be exact PhotoMaker because connector-up is
zero initialized and the outside-core base anchor is unchanged.

At 2k and 4k, continue only if reference swaps begin to move identity in the
reference direction. Stop if the run again changes expression/texture without
directional reference-identity gain.
