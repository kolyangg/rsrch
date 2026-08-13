# Clean E13-family architectures and Serv jobs

This branch keeps one shared E13-family implementation and exposes differences
as small, audited config leaves. All recipes retain hard target-query/reference-
K/V branched self-attention, disabled legacy branched cross-attention, rank-128
BA, effective generic/default adapters, `pose_adapt_ratio=0`, and the fixed-96
DDIM50/CFG5/RealVis validation contract. CL14_CA alone adds corrected residual
identity-token cross-attention over the intact native CA path.

For equations, trainable ownership, data policies, and symbol-level code
references for every recipe, see
[`E13-family architecture reference`](../docs/architecture/2026-08-13_e13_family_architecture_reference.md).

| Recipe | Concise architecture or data delta | Exact config | Serv YAML |
|---|---|---|---|
| E13 | Shared architecture trained on Large Dataset. | `E13_large_ds_joint_shadow_sa128_24k` | [`E13...clean_r1 YAML`](E13_large_ds_joint_shadow_sa128_24k_full96_clean_r1/run_E13_large_ds_joint_shadow_sa128_24k_full96_clean_r1_1gpu.yaml) |
| BC_E13 | Exact E13 architecture; only the training dataset changes to sealed BigCelebs. | `BC_E13_big_celebs_joint_shadow_sa128_24k` | [`BC_E13...clean_r1 YAML`](BC_E13_big_celebs_joint_shadow_sa128_24k_full96_clean_r1/run_BC_E13_big_celebs_joint_shadow_sa128_24k_full96_clean_r1_1gpu.yaml) |
| CL14 | Exact E13 architecture on corrected Cosmic Large, plus a two-latent-cell feather on the training target mask. | `CL14_cosmic_joint_shadow_sa128_softmask_24k` | [`CL14...clean_r1 YAML`](CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_clean_r1/run_CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_clean_r1_1gpu.yaml) |
| CL14_CA | CL14 plus rank-64, zero-init, bounded target-Q/PhotoMaker-ID-KV residual CA only in `up_blocks.0/1`; native CA remains intact. | `CL14_CA_cosmic_residual_identity_ca_24k` | [`CL14_CA...clean_r1 YAML`](CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1/run_CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1_1gpu.yaml) |
| CL18 | CL14 plus a training-only, stop-gradient same-identity cross-view spatial-consistency loss; inference remains single-reference CL14. | `CL18_cosmic_crossview_spatial_consistency_24k` | [`CL18...clean_r1 YAML`](CL18_cosmic_crossview_spatial_consistency_24k_full96_clean_r1/run_CL18_cosmic_crossview_spatial_consistency_24k_full96_clean_r1_1gpu.yaml) |
| CL19 | CL14 with two complete full-query BA messages blended once by a two-cell cosine target router; reference support remains binary. | `CL19_cosmic_true_soft_fullquery_router_24k` | [`CL19...clean_r1 YAML`](CL19_cosmic_true_soft_fullquery_router_24k_full96_clean_r1/run_CL19_cosmic_true_soft_fullquery_router_24k_full96_clean_r1_1gpu.yaml) |
| CL20 | Exact CL14 model with a deterministic 20k Cosmic/BigCelebs hard-case curriculum followed by 4k Cosmic-only re-anchoring. | `CL20_cosmic_bigcelebs_hardcase_curriculum_24k` | [`CL20...clean_r1 YAML`](CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_clean_r1/run_CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_clean_r1_1gpu.yaml) |

The seven concrete YAMLs are adapted from the matching `test`-branch packages.
They retain the proven one-A100 image/resource request, but use the clean
checkout, shared fail-closed launcher, unique `*_clean_r1` names, and a
pre-launch clean-branch check. Each records the exact source commit in its log
directory before starting.

## Serv prerequisites

The concrete YAMLs assume these existing Serv paths:

- checkout: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_e13_family_clean`;
- Conda environment: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS`;
- logs: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/logs/<run_name>`.

Before submission, pull `kit/e13-family-clean`, require an empty `git status`,
and populate the gitignored `diffusion_template/.env` from `.env.example`.
E13 needs the sealed Large Dataset paths/hash; BC_E13 needs sealed BigCelebs.
CL14 needs the exact Cosmic inputs/hash. CL14_CA, CL18 and CL19 additionally
require the sealed subject-v2 embeddings. CL20 also requires sealed BigCelebs and builds
the hash-checked curriculum before model startup.

Inspect Running and Pending MLS jobs first and stay within the normal six-A100
project ceiling. Then submit exactly one linked YAML with its absolute path:

```bash
mls job submit --config /absolute/path/to/run_<run_name>_1gpu.yaml
```

Preparation here does not authorize submission. The shared launcher reruns
config, source-parity, dataset, ONNX Runtime CUDA, and immutable-Comet-startup
gates before training.
