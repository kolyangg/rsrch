# Large Dataset and Big Celebs Comet downloads

Downloaded on 1–2 August 2026 with `tools/comet/comet_experiment.py`. Every
listed step resolved exactly and contains 96 verified PNG images, the complete
Comet metric history, and an export manifest. Each run directory also contains
its Comet console output as `comet_output.log`.

| Entry script | Canonical run | Immutable Comet ID | Downloaded steps | Highest metric step | Local download |
|---|---|---|---|---:|---|
| `launchers/neb/start_rhca_large_dataset_sameid_40k.sh` | `rhca_large_dataset_sameid_40k_full96_r4` | `a99db1fb953d4511827672380e6c1645` | 2k, 20k | 34,550 | `comet_data/rhca_large_dataset_sameid_40k_full96_r4/` |
| `launchers/neb/start_rhca_big_celebs_sameid_40k.sh` | `rhca_big_celebs_sameid_40k_full96_r1` | `569cc685ff9144f5a9b42bf70e14e040` | 2k, 20k, 32k | 32,950 | `comet_data/rhca_big_celebs_sameid_40k_full96_r1/` |
| `launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh` | `rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1` | `700240d8f90b48cfa2cc16f8ff2886b6` | 20k, 32k | 32,750 | `comet_data/rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1/` |
| `serv_run_packages/rhca_large_dataset_sameid_40k_full96_serv_r1/start_rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu.sh` | `rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu` | `db32f157e75a4798b2dfa530477c66d6` | 20k, 32k | 40,000 | `comet_data/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/` |

The Neb wrappers delegate respectively to
`launchers/active/run_rhca_large_dataset_40k_1gpu.sh`,
`launchers/active/run_rhca_big_celebs_40k_1gpu.sh`, and
`launchers/active/run_rhca_big_celebs_scheduled_40k_1gpu.sh`.
