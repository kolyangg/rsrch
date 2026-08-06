# E13-E18 successful-run index

**Status checked:** 6 August 2026; Comet assets refreshed through 16:42 UTC
against immutable experiment records. This list excludes all earlier failed,
stopped, rejected, or superseded revisions.

## Final non-crashed runs

| Experiment | Successful run revision | Status at check | Immutable Comet ID | Experiment record |
|---|---|---|---|---|
| E13 | `E13_large_ds_joint_shadow_sa128_24k_full96_r4` | Completed 24k training and deferred face-quality finalization | [`1cc0a02371094b24a6a02a4cc649f10c`](https://www.comet.com/nikolay-2104/aug-large-ds/1cc0a02371094b24a6a02a4cc649f10c) | [JSON](../experiments/large_dataset/E13_large_ds_joint_shadow_sa128_24k_full96_r4.json) |
| E14 | `E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6` | Completed 24k training; deferred face-quality finalization still running | [`f53c2a2f130247a1b817c820ba7615ae`](https://www.comet.com/nikolay-2104/aug-large-ds/f53c2a2f130247a1b817c820ba7615ae) | [JSON](../experiments/large_dataset/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6.json) |
| E15 | `E15_large_ds_joint_persist_sa128_protected_24k_full96_r2` | Completed 24k | [`f320234a54624aa6a1a100307691b627`](https://www.comet.com/nikolay-2104/aug-large-ds/f320234a54624aa6a1a100307691b627) | [JSON](../experiments/large_dataset/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2.json) |
| E16 | `E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2` | Completed 24k | [`4561fb0de8c64b3da8663e3f4c37589c`](https://www.comet.com/nikolay-2104/aug-large-ds/4561fb0de8c64b3da8663e3f4c37589c) | [JSON](../experiments/large_dataset/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2.json) |
| E17 | `E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5` | Running after complete 16k validation; latest training metric step 16,550 | [`08ecedf8e058461abe952077f9623ab8`](https://www.comet.com/nikolay-2104/aug-large-ds/08ecedf8e058461abe952077f9623ab8) | [JSON](../experiments/large_dataset/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5.json) |
| E18 | `E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4` | Running after complete 22k validation; latest training metric step 22,150 | [`b9e118da6dc94cd9b3849566e18c67ff`](https://www.comet.com/nikolay-2104/aug-large-ds/b9e118da6dc94cd9b3849566e18c67ff) | [JSON](../experiments/large_dataset/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4.json) |

“Successful” here means the selected final revision did not crash. It does not
claim that the still-running E14, E17, or E18 job has completed every
post-training task.

## Experiment descriptions

The scientific hypotheses and implementation plans for E13-E18 are in
[the E0-E12 analysis, section “Recommended next six one-GPU experiments”](../comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS.md#recommended-next-six-one-gpu-experiments).
For a portable version with embedded figures, use
[ANALYSIS_EMBEDDED.md](../comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS_EMBEDDED.md#recommended-next-six-one-gpu-experiments).

The completed E13-E18 metric/visual review and proposed E19-E24 suite are in
[`2026-08-06_e13_e18_results_and_next_experiments.md`](2026-08-06_e13_e18_results_and_next_experiments.md).

## Run scripts

All six use the shared controlled launcher:
[run_E13_E18_large_ds_24k_1gpu.sh](../launchers/active/run_E13_E18_large_ds_24k_1gpu.sh).

- E13 r4: [startup script](../serv_run_packages/E13_large_ds_joint_shadow_sa128_24k_full96_r4/start_E13_large_ds_joint_shadow_sa128_24k_full96_r4_1gpu.sh) · [MLS YAML](../serv_run_packages/E13_large_ds_joint_shadow_sa128_24k_full96_r4/run_E13_large_ds_joint_shadow_sa128_24k_full96_r4_1gpu.yaml)
- E14 r6: [startup script](../serv_run_packages/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/start_E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6_1gpu.sh) · [MLS YAML](../serv_run_packages/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/run_E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6_1gpu.yaml)
- E15 r2: [startup script](../serv_run_packages/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/start_E15_large_ds_joint_persist_sa128_protected_24k_full96_r2_1gpu.sh) · [MLS YAML](../serv_run_packages/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/run_E15_large_ds_joint_persist_sa128_protected_24k_full96_r2_1gpu.yaml)
- E16 r2: [startup script](../serv_run_packages/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/start_E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2_1gpu.sh) · [MLS YAML](../serv_run_packages/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/run_E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2_1gpu.yaml)
- E17 r5: [startup script](../serv_run_packages/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/start_E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5_1gpu.sh) · [MLS YAML](../serv_run_packages/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/run_E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5_1gpu.yaml)
- E18 r4: [startup script](../serv_run_packages/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/start_E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4_1gpu.sh) · [MLS YAML](../serv_run_packages/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/run_E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4_1gpu.yaml)

The package wrappers dispatch through the shared Serv startup script:
[start_E13_E18_large_ds_24k_1gpu.sh](../serv_run_packages/_sources/start_E13_E18_large_ds_24k_1gpu.sh).

## Downloaded Comet assets

Downloaded 6 August 2026 by immutable Comet ID with
`tools/comet/comet_experiment.py`. Each step directory contains the generated
images and `comet_runs_export.json`; the latter preserves the complete Comet
metric histories, summaries, parameters, image asset records, requested and
resolved steps, and export warnings/errors. Each `logs/comet_output.json` is
the full response from Comet's experiment-output endpoint.

The shared local root is
[`comet_data/e13_e18_20260806/`](../comet_data/e13_e18_20260806/).
These downloaded data are gitignored.

| Experiment | Comet output log | Metrics/export record | Step 0 images | Step 8k images | Latest available images |
|---|---|---|---|---|---|
| E13 | [`logs/comet_output.json`](../comet_data/e13_e18_20260806/E13_large_ds_joint_shadow_sa128_24k_full96_r4/logs/comet_output.json) | [`latest/comet_runs_export.json`](../comet_data/e13_e18_20260806/E13_large_ds_joint_shadow_sa128_24k_full96_r4/latest/comet_runs_export.json) | [`step_000000/`](../comet_data/e13_e18_20260806/E13_large_ds_joint_shadow_sa128_24k_full96_r4/step_000000/) | [`step_008000/`](../comet_data/e13_e18_20260806/E13_large_ds_joint_shadow_sa128_24k_full96_r4/step_008000/) | [`latest/` (24k)](../comet_data/e13_e18_20260806/E13_large_ds_joint_shadow_sa128_24k_full96_r4/latest/) |
| E14 | [`logs/comet_output.json`](../comet_data/e13_e18_20260806/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/logs/comet_output.json) | [`latest/comet_runs_export.json`](../comet_data/e13_e18_20260806/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/latest/comet_runs_export.json) | [`step_000000/`](../comet_data/e13_e18_20260806/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/step_000000/) | [`step_008000/`](../comet_data/e13_e18_20260806/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/step_008000/) | [`latest/` (24k)](../comet_data/e13_e18_20260806/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/latest/) |
| E15 | [`logs/comet_output.json`](../comet_data/e13_e18_20260806/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/logs/comet_output.json) | [`latest/comet_runs_export.json`](../comet_data/e13_e18_20260806/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/latest/comet_runs_export.json) | [`step_000000/`](../comet_data/e13_e18_20260806/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/step_000000/) | [`step_008000/`](../comet_data/e13_e18_20260806/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/step_008000/) | [`latest/` (24k)](../comet_data/e13_e18_20260806/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/latest/) |
| E16 | [`logs/comet_output.json`](../comet_data/e13_e18_20260806/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/logs/comet_output.json) | [`latest/comet_runs_export.json`](../comet_data/e13_e18_20260806/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/latest/comet_runs_export.json) | [`step_000000/`](../comet_data/e13_e18_20260806/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/step_000000/) | [`step_008000/`](../comet_data/e13_e18_20260806/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/step_008000/) | [`latest/` (24k)](../comet_data/e13_e18_20260806/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/latest/) |
| E17 | [`logs/comet_output.json`](../comet_data/e13_e18_20260806/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/logs/comet_output.json) | [`latest/comet_runs_export.json`](../comet_data/e13_e18_20260806/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/latest/comet_runs_export.json) | [`step_000000/`](../comet_data/e13_e18_20260806/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/step_000000/) | [`step_008000/`](../comet_data/e13_e18_20260806/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/step_008000/) | [`latest/` (16k, 96/96)](../comet_data/e13_e18_20260806/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/latest/) |
| E18 | [`logs/comet_output.json`](../comet_data/e13_e18_20260806/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/logs/comet_output.json) | [`latest/comet_runs_export.json`](../comet_data/e13_e18_20260806/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/latest/comet_runs_export.json) | [`step_000000/`](../comet_data/e13_e18_20260806/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/step_000000/) | [`step_008000/`](../comet_data/e13_e18_20260806/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/step_008000/) | [`latest/` (22k, 96/96)](../comet_data/e13_e18_20260806/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/latest/) |

All step-0, step-8k, and latest downloads contain the complete 96
generated-image panel. The per-image `ID_sim` CSVs are under each run's
[`per_image_id/`](../comet_data/e13_e18_20260806/) subtree: E13-E16 have every
table through 24k, E17 through 16k, and E18 through 22k. Step zero additionally
contains mask image assets (96 for E13-E15, E17, and E18; 60 for E16); these
are retained because they are part of the Comet image record, but they are not
counted as generated panel images.
