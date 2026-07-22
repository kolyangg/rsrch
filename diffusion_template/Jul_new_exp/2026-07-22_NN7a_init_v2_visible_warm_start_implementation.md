# NN7a_init-v2 visible warm-start implementation

## Implemented correction

The original `NN7a_init` remains available as the reproducible
`attn1_hybrid + post_cap` control. The new `NN7a_init-v2` changes only the
spatial warm-start operator and its bounded arbitration:

```text
target hidden
  -> frozen effective sibling attn2 Q

PMv2-context reference patches
  -> frozen effective sibling attn2 K/V base
  -> trainable rank-32 K/V LoRA

sibling Q/K/V local 5x5 attention
  -> frozen effective sibling attn2 output projection
  -> reference candidate in post-output hidden space

ordinary attn1 Q/K/V + attn1 output
  -> target candidate in post-output hidden space

target + cap(alpha * (reference - target))
```

The default v2 authority is:

```text
alpha = 0.80 * sigmoid(-1.9459101490553132) = 0.10
final local RMS cap = 0.20 of target-candidate RMS
```

Only spatial K/V LoRA A/B and `gate_logit` are trainable. Sibling Q, output,
Q/K norms, the PhotoMaker backbone, and branched cross-attention remain frozen.
The existing `up_blocks.1.attn1` site restriction, 5x5 correspondence,
feathered manual-bbox core, counterfactual supervision, RealVis validation and
exact PhotoMaker epsilon outside the core are unchanged.

## Checkpoint behavior

The new architecture manifest records:

```yaml
ba_spatial_attention_space: sibling_attn2_full
ba_spatial_gate_position: pre_cap
```

NN7a_init v1 and v2 checkpoints are rejected when restored into each other.
Frozen sibling Q/K/V/output state is reconstructed from the active backbone;
only learned K/V LoRA and gate tensors are restored.

Direct-takeover checkpoint preflight now accepts nonzero K/V LoRA-B state when
there is no connector. It still rejects an all-zero direct branch.

## Required step-zero sweep

Run this before the 4k training job:

```bash
cd /home/niko/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_init_v2_step0_alpha_sweep_1gpu.sh
```

It evaluates alpha `0.05`, `0.10`, and `0.20` on the same deterministic 24/96
RealVis subset, with final cap `0.20`, without loading or training a checkpoint.
Compare PM0/R1/R2 face crops and choose the strongest setting that keeps pose,
facial topology, neck attachment, boundary quality and occluders stable.

The first-batch log now reports candidate/applied RMS ratios, cap fraction,
effective gate, whether median authority reaches `0.03`, and exact outside-core
isolation.

## 4k launch

For the recommended default alpha `0.10`:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_init_v2_train_then_diagnose_1gpu.sh
```

If the sweep selects alpha `0.05` or `0.20`, pass the matching logit; the same
override is forwarded to post-training strict checkpoint validation:

```bash
# alpha 0.05
bash jul_serv_runs/start_ba_NN7a_init_v2_train_then_diagnose_1gpu.sh \
  model.ba_gate_init_logit=-2.70805020110221

# alpha 0.20
bash jul_serv_runs/start_ba_NN7a_init_v2_train_then_diagnose_1gpu.sh \
  model.ba_gate_init_logit=-1.0986122886681098
```

The job trains for 4,000 optimizer steps with all 96 in-run RealVis validation
images, then runs the deterministic 24-image five-condition diagnostic.

No Git commit or push was performed.
