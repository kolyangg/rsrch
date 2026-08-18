# E13-family clean branch

`kit/e13-family-clean` is the concise, auditable E13-family implementation.
It keeps the shared explicit target-query/reference-K/V branched self-attention
architecture and expresses experiment differences in small Hydra config leaves.

| Recipe | Difference from the shared E13 architecture |
|---|---|
| E13 | Large Dataset baseline. |
| BC_E13 | E13 architecture trained on sealed BigCelebs. |
| CL14 | Corrected Cosmic Large data policy plus a two-cell training-mask feather. |
| CL14_CA | CL14 plus bounded residual target-Q/PhotoMaker-ID-KV cross-attention in `up_blocks.0/1`. |
| CL18 | CL14 plus training-only same-ID cross-view spatial consistency; inference stays single-reference. |
| CL19 | CL14 with full-query BA messages blended once by a two-cell cosine router. |
| CL20 | CL14 model with a deterministic Cosmic/BigCelebs hard-case curriculum and final Cosmic re-anchoring. |
| CL23 | CL19 with fixed denoising-progress gains over low/high bands of the routed reference-minus-native message. |
| CL27 | Exact CL23 inference plus deterministic, training-only frequency-surface supervision. |

Exact one-A100 Serv YAMLs are ready for all nine branch recipes:

- [E13 YAML](diffusion_template/serv_run_packages/E13_large_ds_joint_shadow_sa128_24k_full96_clean_r1/run_E13_large_ds_joint_shadow_sa128_24k_full96_clean_r1_1gpu.yaml)
- [BC_E13 YAML](diffusion_template/serv_run_packages/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_clean_r1/run_BC_E13_big_celebs_joint_shadow_sa128_24k_full96_clean_r1_1gpu.yaml)
- [CL14 YAML](diffusion_template/serv_run_packages/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_clean_r1/run_CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_clean_r1_1gpu.yaml)
- [CL14_CA YAML](diffusion_template/serv_run_packages/CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1/run_CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1_1gpu.yaml)
- [CL18 YAML](diffusion_template/serv_run_packages/CL18_cosmic_crossview_spatial_consistency_24k_full96_clean_r1/run_CL18_cosmic_crossview_spatial_consistency_24k_full96_clean_r1_1gpu.yaml)
- [CL19 YAML](diffusion_template/serv_run_packages/CL19_cosmic_true_soft_fullquery_router_24k_full96_clean_r1/run_CL19_cosmic_true_soft_fullquery_router_24k_full96_clean_r1_1gpu.yaml)
- [CL20 YAML](diffusion_template/serv_run_packages/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_clean_r1/run_CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_clean_r1_1gpu.yaml)
- [CL23 YAML](diffusion_template/serv_run_packages/CL23_cosmic_temporal_frequency_router_24k_full96_clean_r1/run_CL23_cosmic_temporal_frequency_router_24k_full96_clean_r1_1gpu.yaml)
- [CL27 YAML](diffusion_template/serv_run_packages/CL27_cosmic_frequency_surface_energy_24k_full96_clean_r1/run_CL27_cosmic_frequency_surface_energy_24k_full96_clean_r1_1gpu.yaml)

See [the Serv package README](diffusion_template/serv_run_packages/README.md)
for exact config names, path assumptions, `.env` requirements, and submission
gates. See [the implementation ledger](diffusion_template/docs/architecture/2026-08-10_e13_family_clean_implementation.md)
and [CL18–CL20 extension ledger](diffusion_template/docs/architecture/2026-08-12_cl18_cl19_cl20_clean_extension.md)
and [CL23/CL27 extension ledger](diffusion_template/docs/architecture/2026-08-18_cl23_cl27_clean_extension.md)
for the detailed architecture and parity evidence. The
[formula reference](diffusion_template/docs/architecture/2026-08-13_e13_family_architecture_reference.md)
defines every supported architecture and objective.

No Serv job is submitted by these files. Check Running and Pending MLS jobs and
the normal six-A100 project ceiling before an explicitly approved submission.
