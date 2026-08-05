# `aug-large-ds` Comet export — 5 August 2026

This folder contains the two E0 controls and E1–E12 (14 immutable Comet experiments).

- [`ANALYSIS.md`](ANALYSIS.md): full aggregate/per-identity/per-prompt
  `ID_sim` analysis, visual comparison, training-setup audit, and the prepared
  E13-E18 parallel implementation plan. Reproducible charts and contact sheets
  are under `analysis_assets/`.
- [`ANALYSIS_EMBEDDED.md`](ANALYSIS_EMBEDDED.md): portable 41 MB single-file
  edition with all 13 report charts and selected comparison images embedded as
  PNG data URLs; the complete 28-sheet contact archive is omitted.
- `metrics/comet_metrics_full_history.json`: complete scalar metric histories for every run (38–39 metric names per run).
- `logs/<run_name>.log`: complete Comet console output for every run.
- `images_step_002000/<run_name>/`: 96 validation PNGs per run at step 2,000.
- `images_step_008000/<run_name>/`: 96 validation PNGs per run at step 8,000.
- `images_latest/<run_name>/`: 96 validation PNGs at each run's latest available step.
- `images_step_018000/E11_large_ds_ba_sa_r128_20k_full96_r1/`: the previously downloaded E11 18k set, retained after the latest set was refreshed to 20k.
- `per_image_id/<run_name>/`: `id_sim` CSVs for every available validation step, each containing exactly 96 image rows.
- `inventory.json`, `per_image_id_sources.json`, and `audit.json`: immutable run inventory, CSV provenance, and completed integrity audit.
- `serv_recovery/`: raw E0 CSV recovery source. Missing E0 Comet tables were copied read-only from Serv; E1–E12 tables came from Comet.

## Experiment documentation

- [E0 recovery and matched controls](../../docs/experiments/2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md): historical-r4 recovery, fixed-versus-historical E0 design, and the E1–E6 base audit.
- [E1–E6 hard-BA six-arm design](../../docs/experiments/2026-08-03_large_dataset_hard_ba_six_arm_design.md): scientific deltas, immutable Comet/job records, ownership, and validation contract.
- [E7–E10 adapter experiment rationale](../../analysis/2026-08-04_e0_historical_global_adapter_id_gain_and_next_experiments.md): evidence behind the four effective-adapter diagnostic arms.
- [E10 position/static-mask analysis](../../analysis/2026-08-04_e10_face_position_and_static_mask_drift.md): interpretation caveat for E10 fixed-mask validation metrics.
- [E11/E12 BA-capacity plan](../../docs/experiments/2026-08-04_e11_e12_large_ds_ba_capacity_plan.md): rank-128 spatial-SA BA versus corrected identity-token CA design and launch records.
- [Current project handoff](../../docs/handoffs/LATEST.md): broader experiment history and current-state context. Status statements in the dated design documents reflect their authoring time; the inventory below records the downloaded run endpoints.

| Run | Comet experiment key | Latest images / metrics |
|---|---|---:|
| E0 fixed — `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1` | `5b5cbd1584184ce1a9032dd6fafb91c5` | 20k |
| E0 historical — `E0_large_ds_base_historical_r4_20k_full96_r1` | `a5599bd06c9346978c1fca8b8087f634` | 20k |
| E1 — `E1_large_ds_truekey_r32_20k_full96_r1` | `ce0c9b918d79449b92fa83ef970285c3` | 20k |
| E2 — `E2_large_ds_branchout_r32_20k_full96_r1` | `4c8af4e867b14377b69fa250fae5cde9` | 20k |
| E3 — `E3_large_ds_roiwarp_r32_20k_full96_r1` | `9c5cbe4e49254134b4763ff7a4554c9b` | 20k |
| E4 — `E4_large_ds_midup_r32_20k_full96_r1` | `2d77f35256844e0399c1834859a45dc7` | 20k |
| E5 — `E5_large_ds_infersteps_r32_20k_full96_r1` | `4a107cbc30a04a858de3e3b5c411cdca` | 20k |
| E6 — `E6_large_ds_fp32_r32_20k_full96_r1` | `9f3e20a75a0a4304b12d724693e13fbf` | 20k |
| E7 — `E7_large_ds_generic_effective_r32_20k_full96_r1` | `e3d540a8f5c84e9db960214a1342ca04` | 20k |
| E8 — `E8_large_ds_generic_ca_r32_20k_full96_r1` | `db1326c7591e484597f3009db63af42f` | 20k |
| E9 — `E9_large_ds_shared_saout_r32_20k_full96_r1` | `deb40502cfc849a0aecc8e48b4eec005` | 20k |
| E10 — `E10_large_ds_pmdefault_effective_r64_20k_full96_r1` | `0375f172f75c482f840317ec5ae41c05` | 20k |
| E11 — `E11_large_ds_ba_sa_r128_20k_full96_r1` | `e748a5e136b3441688aaf968294612a1` | 20k |
| E12 — `E12_large_ds_ba_idca_up_r256_20k_full96_r1` | `d06ab51afbff4cacac1877632e26cf24` | 12k |

Audit result: 4,032 current requested PNGs, 150 per-image CSVs / 14,400 rows, 14 non-empty logs, and exact requested-step resolution. The separate archived E11 18k set contains another 96 PNGs. Two transient E12 12k image download failures were recovered and recorded in `images_latest/retry_downloads.json`.
