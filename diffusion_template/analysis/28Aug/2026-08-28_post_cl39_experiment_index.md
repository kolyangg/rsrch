# Post-CL39 experiment and validation index

**Snapshot:** 28 August 2026, 09:45 BST  
**Scope:** completed or active scientific runs and validation-only tests made after the canonical CL39 result. Superseded attempts are included only where they prevent accidental reuse.  
**Canonical CL39 baseline:** [`b1ca0b3da679401c85b991f1bbdf0b2a`](https://www.comet.com/nikolay-2104/aug-large-ds/b1ca0b3da679401c85b991f1bbdf0b2a), peak subject-v2 ID `0.570124 @16k`, endpoint `0.566342 @24k`. See the [Serv package](../serv_run_packages/CL39_cosmic_null_key_confidence_router_24k_full96_r4/), [experiment record](../experiments/cosmic_large/CL39_cosmic_null_key_confidence_router_24k_full96_r4.json), and [fixed-96 comparison report](2026-08-21_cl38_cl45_completed_results_and_photomaker_shortcut_audit.md).

## 1. New training runs

All endpoint values below are Comet `manual_val/id_sim` on the fixed 96-image panel. A numerical endpoint difference is not a paired promotion result; R2 image-level paired intervals and visual review have not yet been assembled.

| Run | Status at snapshot | Immutable identifiers | Local folder / record | Result so far |
|---|---|---|---|---|
| **BC39 r2** — CL39 transferred to BigCelebs with the CL27 ownership-mask objective | Completed, 24k | MLS `lm-mpi-job-b94c7816-f219-42d7-a189-b66b573ece7e`; Comet [`96cfa64b72934afc870432a243cd4a55`](https://www.comet.com/nikolay-2104/aug-large-ds/96cfa64b72934afc870432a243cd4a55) | [Serv package](../serv_run_packages/BC39_big_celebs_null_key_confidence_router_24k_full96_r2/) | Peak ID `0.550550 @22k`; endpoint `0.548610` (`-0.017732` vs CL39-24k). Text similarity `26.3283` (`+0.3415`). Dataset transfer lowers identity on this validation panel. |
| **CL39-R2-A / blueprint E1** — training-time coherent reference-face ownership | Completed, 24k | MLS `lm-mpi-job-a1bf6e81-92ad-4ea1-8974-321d0c0ed495`; Comet [`a110727f06994872bbae4d173ffbb3cc`](https://www.comet.com/nikolay-2104/aug-large-ds/a110727f06994872bbae4d173ffbb3cc) | [Serv package](../serv_run_packages/CL39R2_three_experiments_20260826_r1/) | Peak `0.564087 @16k`; endpoint `0.562272` (`-0.004071` vs CL39-24k). No numerical promotion. |
| **CL39-R2-D / blueprint E2** — target PhotoMaker-condition dropout | Completed, 24k | MLS `lm-mpi-job-8465ba02-25ef-480e-a5df-90cad9d9a25c`; Comet [`effe322abe4e4a56adc63c7b7e516464`](https://www.comet.com/nikolay-2104/aug-large-ds/effe322abe4e4a56adc63c7b7e516464) | [Serv package and submission record](../serv_run_packages/CL39R2_DE_blueprint_20260826_r1/) | Peak `0.557658 @22k`; endpoint `0.557103` (`-0.009239`). Negative aggregate identity result for this setting. |
| **CL39-R2-E / blueprint E3** — valid-key-only reference attention | Completed, 24k | MLS `lm-mpi-job-d4a31f03-36a3-40c3-a8e9-e4e44acaf765`; Comet [`858267db7062457cbe7ed476484e4a27`](https://www.comet.com/nikolay-2104/aug-large-ds/858267db7062457cbe7ed476484e4a27) | [Serv package and submission record](../serv_run_packages/CL39R2_DE_blueprint_20260826_r1/) | Peak `0.561537 @20k`; endpoint `0.560468` (`-0.005875`). Text similarity is `26.0853`, but identity does not improve. |
| **CL39-R2-B / blueprint E4** — learned bounded low/high reliability gate | Completed, 24k | MLS `lm-mpi-job-3ae354c8-00a4-4d9e-80fd-7926416e3532`; Comet [`40f98e137c8d45f09e841f03d84bbffe`](https://www.comet.com/nikolay-2104/aug-large-ds/40f98e137c8d45f09e841f03d84bbffe) | [Serv package](../serv_run_packages/CL39R2_three_experiments_20260826_r1/) | Peak and endpoint `0.569306 @24k` (`+0.002964` vs CL39-24k), the best R2 endpoint numerically. It remains `0.000819` below CL39's selected 16k peak; paired/visual promotion analysis is still required. |
| **CL39-R2-C / blueprint E5** — fixed low/high face-RMS caps | Completed, 24k | MLS `lm-mpi-job-febbb459-88d7-4fff-a7c9-f631a5317365`; Comet [`381793a24dcf4ce3ac1ffcd826f9e59a`](https://www.comet.com/nikolay-2104/aug-large-ds/381793a24dcf4ce3ac1ffcd826f9e59a) | [Serv package](../serv_run_packages/CL39R2_three_experiments_20260826_r1/) | Peak `0.569010 @16k`; endpoint `0.564836` (`-0.001506`). Near CL39 numerically, but not a promotion on current evidence. |
| **CL39I / INIT-R-NATIVE** — initialize explicit reference Q/K/V from native SDXL rather than the PhotoMaker delta | **Running**; about global step `9.2k` (epoch 5 batch 1,150) | MLS `lm-mpi-job-621c875f-378c-46e5-af0a-fc51598b4f5d`; Comet [`110a9dab669a4bffb5e403fe446fd618`](https://www.comet.com/nikolay-2104/aug-large-ds/110a9dab669a4bffb5e403fe446fd618) | [Serv package](../serv_run_packages/CL39I_reference_native_init_20260827_r1/); [experiment record](../experiments/cosmic_large/CL39I_cosmic_reference_native_init_24k_full96_r1.json) | Five validations through 8k. Current ID `0.545600 @8k`; current peak `0.547531 @6k`. Job is healthy, but no endpoint conclusion yet. |

The R2-A/B/C remote runtime root is `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl39r2_v1`; R2-D/E is under `runtime_sources_cl39r2_de_v1`; CL39I is under `runtime_sources_cl39i_v1`. The implementation plan for R2-A/B/C is [here](2026-08-26_cl39_r2_three_experiment_implementation_plan.md), and the first-five blueprint is [here](CL39_branched_attention_investigation_2026-08-26.md).

## 2. Completed validation-only and diagnostic tests

Except for the all-checkpoint BA@10 trajectory, these jobs intentionally created no child Comet experiment. Their immutable model provenance is the parent CL39 key [`b1ca0b3da679401c85b991f1bbdf0b2a`](https://www.comet.com/nikolay-2104/aug-large-ds/b1ca0b3da679401c85b991f1bbdf0b2a).

| Test | Serv jobs | Local folders / report | Main result |
|---|---|---|---|
| **CL39 24k attention/confidence audit** — actual, forced confidence `C=1`, up0/up1 correction-off, and raw-R-on-face | `lm-mpi-job-79af3dd2-f662-48bc-ac83-c18adc33d490`; `lm-mpi-job-ba15e767-18cb-4000-aa8a-8b1613c683a1`; `lm-mpi-job-01f46a9a-d2cb-4f4b-b279-0842bf1a2718`; `lm-mpi-job-1d94aa46-9197-466a-8f22-1abcce4e4312` | [Package](../serv_run_packages/CL39_attention_audit_serv_final/); [results](../artifacts/cl39_attention_24k_serv_a100/); [report](2026-08-25_cl39_entropy_confidence_attention_audit.md) | On the selected 16 cells, normal CL39 beats group-scoped correction-off by `+0.03829` ID and forced `C=1` by `+0.05770`; raw R is visibly fragile and often duplicates/warps facial parts. |
| **CL19/23/27/39 branch-lineage audit** — 18 arms, 1,728 images | `lm-mpi-job-a64df24d-350d-4cae-bdf5-8b31d2a5af29`; `lm-mpi-job-46454c8f-2967-4d12-9a28-a75d7232cf86`; `lm-mpi-job-c7456c95-415d-4ce4-9d3e-5026b5196440` | [Package](../serv_run_packages/BA_lineage_branch_audit_serv_r1/); [results](../artifacts/ba_lineage_branch_audit_20260826/); [report](2026-08-26_ba_lineage_r_frequency_confidence_audit.md) | CL39 actual beats raw R by `+0.13513`, low-only by `+0.02961`, and forced `C=1` by `+0.05770` on the 16-cell panel. Both frequency bands and confidence are active; raw R is not a viable standalone route. |
| **16k global all-70 BA-off + 24k A/B/C/D identity-source crossing** | `lm-mpi-job-8a1e80fe-4ae5-4ae1-b3a9-f7ab2c8e945f`; `lm-mpi-job-b890b097-0eab-40c0-854e-999e54617119` | [Package](../serv_run_packages/CL39_attribution_controls_20260826_r1/); [results](../artifacts/cl39_attribution_controls_20260827/); [report](2026-08-27_cl39_spatial_ba_attribution_controls.md) | At 16k, actual beats all-70 BA-off by `+0.03330` ID, 95% interval `[+0.02424,+0.04258]`. At 24k, correct spatial reference adds `+0.02951` ID with correct PM tokens, while PM-token identity contributes about `+0.54`: BA is causally useful but PM dominates identity. |
| **Multiseed A/B/C/D crossing with shared static seed-0 masks** | `lm-mpi-job-f1f0c20e-4941-4499-bd0b-14d823bbc7eb`; `lm-mpi-job-91332acd-1a05-4fd2-8f8c-345b79f2e50a`; `lm-mpi-job-7086f60e-f2b4-49fb-b367-7cd379379470` | [Package](../serv_run_packages/CL39_identity_crossing_multiseed_20260827_r1/); [preserved failure evidence](../artifacts/cl39_identity_crossing_multiseed_20260827/) | **Scientifically invalid and superseded.** Seeds 1–3 accidentally reused the seed-0 automatic face-mask cache during generation. Do not use its estimates. |
| **Corrected multiseed crossing with seed-specific PhotoMaker-only automatic masks** | Seed 1 `lm-mpi-job-ee43b350-de5c-44e3-9cab-d694e9f5806e`; seed 2 `lm-mpi-job-9c599a15-1d97-49a9-8609-81f38d03ca85`; seed 3 `lm-mpi-job-f04b6ebf-aded-4da2-ad5d-206a65534f15` | [Package](../serv_run_packages/CL39_identity_crossing_dynamic_masks_20260827_r1/); [accepted results](../artifacts/cl39_identity_crossing_dynamic_masks_20260827/); [report](2026-08-27_cl39_identity_crossing_multiseed.md) | Authoritative four-seed result: pooled A-B spatial effect `+0.03055`, 95% interval `[+0.02393,+0.03740]`; C-D `+0.01422`, `[+0.00896,+0.01917]`. Every per-seed interval is positive. PM identity remains dominant (`A-C=+0.53712`). |
| **BA starts at denoising step 10 — first 12 images** | Successful retry `lm-mpi-job-3f4f48e7-97f7-4340-811f-148bd8d2be24` | [Package](../serv_run_packages/CL39_ba_start10_batch12_20260827_r1/); [results](../artifacts/cl39_ba_start10_batch12_20260827/); [report](2026-08-27_cl39_ba_start10_batch12.md) | Intended ID `-0.00584` versus BA@15; interval crosses zero. This quick one-identity estimate is superseded by the full-96 test. |
| **BA starts at step 10 — full fixed-96 panel** | `lm-mpi-job-8ca02579-79d3-4c94-9515-a50bcde5b9c1` | [Same package](../serv_run_packages/CL39_ba_start10_batch12_20260827_r1/); [results](../artifacts/cl39_ba_start10_full96_20260827/); [metrics report](2026-08-27_cl39_ba_start10_full96.md); [all 96 visual pairs](2026-08-27_cl39_ba_start10_all96_visual_comparison.md) | BA@10 changes intended ID by `-0.00779`, 95% interval `[-0.01632,+0.00044]`, with 35/96 wins. Retain the original PM@10 / BA@15 schedule. |
| **BA@10 trajectory over CL39 steps 0, 2k, …, 24k** — 13 fixed-96 validations in one Comet run | `lm-mpi-job-f356f502-4f6d-49b4-8661-fd4009ff6595` | Comet [`b854b0d17dd64f82ae5b5b969d70be4c`](https://www.comet.com/nikolay-2104/aug-large-ds/b854b0d17dd64f82ae5b5b969d70be4c); [package](../serv_run_packages/CL39_ba_start10_all_checkpoints_20260827_r1/); [record](../experiments/cosmic_large/CL39_ba_start10_all_checkpoints_full96_r1.json) | Completed 28 August. Peak ID `0.565369 @16k`; endpoint `0.558638 @24k`. Both are below the original BA@15 values (`0.570124` and `0.566342`), so the checkpoint trajectory reinforces the full-96 rejection of BA@10. |

The principal sealed validation task roots are:

- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/BA_lineage_branch_audit_serv_r1`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_attribution_controls_20260826_r1`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_identity_crossing_dynamic_masks_20260827_r1`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_ba_start10_batch12_20260827_r1`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_ba_start10_all_checkpoints_20260827_r1`

## 3. Numbered CL39 descendants CL40–CL45

These six 24k runs completed before the later CL39 audits, but they are included because they are the directly numbered post-CL39 mechanisms. The controlled paired results and visual review are in the [CL38–CL45 report](2026-08-21_cl38_cl45_completed_results_and_photomaker_shortcut_audit.md).

| Run | Immutable Comet key | Selected / final subject-v2 ID | Controlled decision |
|---|---|---:|---|
| CL40 identity-motion projector | [`1c2e0ac2fcae433db18f55de663b59ef`](https://www.comet.com/nikolay-2104/aug-large-ds/1c2e0ac2fcae433db18f55de663b59ef) | `0.541975 @20k` / `0.540369` | Neutral versus matched CL27; no promotion. |
| CL41 landmark-canonical K/V | [`b40179ef6a9d4dd6954f6d06d148069c`](https://www.comet.com/nikolay-2104/aug-large-ds/b40179ef6a9d4dd6954f6d06d148069c) | `0.534795 @16k` / `0.529705` | Negative; reject tested configuration. |
| CL42 component-token memory | [`9613ca23f49f469b9bc0fda89055483d`](https://www.comet.com/nikolay-2104/aug-large-ds/9613ca23f49f469b9bc0fda89055483d) | `0.544544 @16k` / `0.544262` | Neutral; no promotion. |
| CL43 identity-adaptive modulation | [`d29cbfa7927547c9ac71a8da0b583e33`](https://www.comet.com/nikolay-2104/aug-large-ds/d29cbfa7927547c9ac71a8da0b583e33) | `0.541810 @22k` / `0.540837` | Neutral; no promotion. |
| CL44 semantic/time high-frequency window | [`42928f13f7ee41448d3d715231f8bb32`](https://www.comet.com/nikolay-2104/aug-large-ds/42928f13f7ee41448d3d715231f8bb32) | `0.550846 @22k` / `0.550284` | Secondary positive versus CL27 (`+0.008952`, paired interval `[+0.002135,+0.015999]`), but below CL39 and PhotoMaker. |
| CL45 asymmetric BA-only PCGrad | [`bfb129031773494f881ea629ced3fe60`](https://www.comet.com/nikolay-2104/aug-large-ds/bfb129031773494f881ea629ced3fe60) | `0.537525 @18k` / `0.535199` | Negative; reject tested configuration. |

Canonical config-to-key mappings are in [`src/configs/clean_full_runs.json`](../src/configs/clean_full_runs.json), and run records are in [`experiments/cosmic_large/`](../experiments/cosmic_large/).

## 4. Superseded and failed records that matter

- **BC39 r1:** Comet [`7f28fd59e7f8432ab94f7cb2b447d9e6`](https://www.comet.com/nikolay-2104/aug-large-ds/7f28fd59e7f8432ab94f7cb2b447d9e6), MLS UID `40f75a2c-913a-42d7-800f-e53fe172c6ff`, [Serv package](../serv_run_packages/BC39_big_celebs_null_key_confidence_router_24k_full96_r1/). It failed before the first optimizer step because BigCelebs did not emit the ownership mask required by the inherited frequency-surface loss. BC39 r2 is the accepted recovery.
- **Static-mask multiseed crossing:** all three jobs completed technically, but the scientific result is invalid because the generation mask belonged to seed 0. The corrected dynamic-mask report at the same Markdown path supersedes it.
- **BA@10 first attempt:** `lm-mpi-job-8e924484-43bb-4e31-8123-dda64974c924` failed the checkpoint architecture guard before generation. The successful retry changed only the validation schedule and added the isolated equal-start selector case.
- Other preflight failures archived in the package submission records failed before useful training or accepted generation and have no scientific result.

## 5. Current decision summary

1. CL39 remains the supported baseline. Spatial BA is now causally validated and repeatable across inference seeds, but PhotoMaker tokens still dominate identity selection.
2. Of the five completed R2 children, R2-B is the only one numerically above CL39 at the 24k endpoint (`+0.002964`), but it has not yet passed paired image-level and visual promotion gates. The other four do not improve aggregate endpoint ID.
3. Starting BA at denoising step 10 is rejected by both the full-96 endpoint comparison and the completed all-checkpoint trajectory; retain BA@15.
4. INIT-R-NATIVE is the only still-running post-CL39 experiment in this index. Its current measurements through 8k are preliminary.

## 6. Confidence and limitations

- **High confidence:** immutable job status, Comet keys, endpoint metric values, accepted validation gate counts, and conclusions already backed by the linked paired reports.
- **Moderate confidence:** the practical R2 ordering from aggregate Comet endpoints. It is an unpaired screening view until per-image tables, confidence intervals, hard-case slices, and visual face review are assembled.
- **Not established:** whether R2-B is a real improvement over CL39, and the final result of INIT-R-NATIVE.
