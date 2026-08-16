# Exact Serv checkpoint deletion manifest: keep newest full and newest weights-only file

**Generated:** 13 August 2026  
**Serv scope:** `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/` only  
**Policy:** For each inactive checkpoint-bearing run directory, retain the newest `checkpoint*.pth` and newest `weights*.pth` when that class exists; list every other matching file for deletion.  
**Serv actions:** Read-only inventory. No file was modified or deleted.

## Approval summary

| Item | Exact value |
|---|---:|
| Inactive run directories | `120` |
| Files retained | `239` |
| Files proposed for deletion | **`2149`** |
| Bytes retained | `113,501,316,764` = `113.50 GB` (`105.71 GiB`) |
| Bytes proposed for deletion | **`1,145,599,523,788` = `1145.60 GB` (`1066.92 GiB`)** |
| Canonical deletion-list SHA-256 | `a6bc07d632d0a75a16194746616381479ff505ba42566525ae7411485c8f9cfa` |

The SHA-256 seals newline-delimited canonical entries in the form `<bytes><TAB><absolute-path>`, in the exact order shown below. Generated validation images and non-checkpoint files are **not** included.

One directory has no weights-only file, so this inventory retains 239 rather than 240 files:

- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_par100_4k_r2/`

Of the 120 inactive directories, 110 contain older files proposed for deletion and appear in the detailed sections. The other 10 already contain only retained endpoint files and require no action.

## Excluded live scientific roots

The eight corresponding MLS jobs were rechecked and all reported `running`. Nothing below these roots is included:

- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v8/CL14_CA_r7/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v23/CL14_CA_optimized_r11/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl21_cl26_v1/CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r2/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl21_cl26_v1/CL22_cosmic_visibility_order_router_24k_full96_r2/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl21_cl26_v1/CL23_cosmic_temporal_frequency_router_24k_full96_r1/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl21_cl26_v1/CL24_cosmic_pm_boundary_distill_24k_full96_r1/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl21_cl26_v1/CL25_cosmic_low_noise_id_reward_4k_full96_r2/`
- `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl21_cl26_v1/CL26_cosmic_anchored_highres_roi_ba_24k_full96_r3/`

## Proposed deletion by top-level location

| Location below `nasilaev/` | Files | Bytes | GB |
|---|---:|---:|---:|
| `runtime_sources_cl1_cl3_v1/` | `308` | `270,785,056,656` | `270.79` |
| `rsrch/` | `834` | `255,680,321,658` | `255.68` |
| `runtime_worktrees/` | `328` | `231,379,652,220` | `231.38` |
| `rsrch_test/` | `393` | `133,017,828,062` | `133.02` |
| `runtime_sources_cl15_cl20_v1/` | `132` | `118,401,393,152` | `118.40` |
| `runtime_sources_e19_e24_v3/` | `132` | `116,993,319,856` | `116.99` |
| `runtime_sources_cl14_ca_v1/` | `22` | `19,341,952,184` | `19.34` |
| **Total** | **`2149`** | **`1,145,599,523,788`** | **`1145.60`** |

## Exact per-run manifest

Each section names the run directory relative to `nasilaev/`, shows the retained checkpoint files, then lists every proposed deletion as `bytes<TAB>absolute-path`.

### `rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/checkpoint-epoch5.pth` — `286,226,804` bytes (`286.23 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/weights-epoch5.pth` — `157,535,418` bytes (`157.54 MB`)

Delete `8` files / `1,775,048,888` bytes (`1.78 GB`):

```text
286226804	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/checkpoint-epoch1.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/weights-epoch1.pth
286226804	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/checkpoint-epoch2.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/weights-epoch2.pth
286226804	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/checkpoint-epoch3.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/weights-epoch3.pth
286226804	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/checkpoint-epoch4.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/checkpoint-epoch5.pth` — `286,226,996` bytes (`286.23 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/weights-epoch5.pth` — `157,535,418` bytes (`157.54 MB`)

Delete `8` files / `1,775,049,656` bytes (`1.78 GB`):

```text
286226996	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/checkpoint-epoch1.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/weights-epoch1.pth
286226996	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/checkpoint-epoch2.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/weights-epoch2.pth
286226996	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/checkpoint-epoch3.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/weights-epoch3.pth
286226996	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/checkpoint-epoch4.pth
157535418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_N3a_new2_nfs_2gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/checkpoint-epoch5.pth` — `525,106,066` bytes (`525.11 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/weights-epoch5.pth` — `237,114,630` bytes (`237.11 MB`)

Delete `8` files / `3,048,882,784` bytes (`3.05 GB`):

```text
525106066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/checkpoint-epoch1.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/weights-epoch1.pth
525106066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/checkpoint-epoch2.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/weights-epoch2.pth
525106066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/checkpoint-epoch3.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/weights-epoch3.pth
525106066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/checkpoint-epoch4.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1a_n3a_replay_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/checkpoint-epoch5.pth` — `525,106,130` bytes (`525.11 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/weights-epoch5.pth` — `237,114,630` bytes (`237.11 MB`)

Delete `8` files / `3,048,883,040` bytes (`3.05 GB`):

```text
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/checkpoint-epoch1.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/weights-epoch1.pth
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/checkpoint-epoch2.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/weights-epoch2.pth
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/checkpoint-epoch3.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/weights-epoch3.pth
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/checkpoint-epoch4.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1b_schedule_matched_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/checkpoint-epoch5.pth` — `525,106,130` bytes (`525.11 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/weights-epoch5.pth` — `237,114,630` bytes (`237.11 MB`)

Delete `8` files / `3,048,883,040` bytes (`3.05 GB`):

```text
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/checkpoint-epoch1.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/weights-epoch1.pth
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/checkpoint-epoch2.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/weights-epoch2.pth
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/checkpoint-epoch3.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/weights-epoch3.pth
525106130	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/checkpoint-epoch4.pth
237114630	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1c_masked_id_prompt_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/checkpoint-epoch5.pth` — `286,168,306` bytes (`286.17 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/weights-epoch5.pth` — `157,520,150` bytes (`157.52 MB`)

Delete `8` files / `1,774,753,824` bytes (`1.77 GB`):

```text
286168306	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/checkpoint-epoch1.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/weights-epoch1.pth
286168306	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/checkpoint-epoch2.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/weights-epoch2.pth
286168306	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/checkpoint-epoch3.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/weights-epoch3.pth
286168306	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/checkpoint-epoch4.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1d_frozen_ca_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/checkpoint-epoch5.pth` — `286,169,010` bytes (`286.17 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/weights-epoch5.pth` — `157,520,150` bytes (`157.52 MB`)

Delete `8` files / `1,774,756,640` bytes (`1.77 GB`):

```text
286169010	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/checkpoint-epoch1.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/weights-epoch1.pth
286169010	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/checkpoint-epoch2.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/weights-epoch2.pth
286169010	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/checkpoint-epoch3.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/weights-epoch3.pth
286169010	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/checkpoint-epoch4.pth
157520150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1e_frozen_ca_id_loss_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/checkpoint-epoch5.pth` — `157,684,018` bytes (`157.68 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/weights-epoch5.pth` — `114,728,822` bytes (`114.73 MB`)

Delete `8` files / `1,089,651,360` bytes (`1.09 GB`):

```text
157684018	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/checkpoint-epoch1.pth
114728822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/weights-epoch1.pth
157684018	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/checkpoint-epoch2.pth
114728822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/weights-epoch2.pth
157684018	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/checkpoint-epoch3.pth
114728822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/weights-epoch3.pth
157684018	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/checkpoint-epoch4.pth
114728822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN1f_ref_kv_id_loss_1gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/checkpoint-epoch6.pth` — `132,448,658` bytes (`132.45 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/weights-epoch6.pth` — `106,304,390` bytes (`106.30 MB`)

Delete `10` files / `1,193,765,240` bytes (`1.19 GB`):

```text
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/checkpoint-epoch1.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/weights-epoch1.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/checkpoint-epoch2.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/weights-epoch2.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/checkpoint-epoch3.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/weights-epoch3.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/checkpoint-epoch4.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/weights-epoch4.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/checkpoint-epoch5.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_1gpu/weights-epoch5.pth
```

### `rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/checkpoint-epoch6.pth` — `132,448,658` bytes (`132.45 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/weights-epoch6.pth` — `106,304,390` bytes (`106.30 MB`)

Delete `10` files / `1,193,765,240` bytes (`1.19 GB`):

```text
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/checkpoint-epoch1.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/weights-epoch1.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/checkpoint-epoch2.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/weights-epoch2.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/checkpoint-epoch3.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/weights-epoch3.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/checkpoint-epoch4.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/weights-epoch4.pth
132448658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/checkpoint-epoch5.pth
106304390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN4_causal_null_up0_nfs_2gpu/weights-epoch5.pth
```

### `rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_1gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_1gpu/checkpoint-epoch2.pth` — `170,940,018` bytes (`170.94 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_1gpu/weights-epoch2.pth` — `119,127,126` bytes (`119.13 MB`)

Delete `2` files / `290,067,144` bytes (`0.29 GB`):

```text
170940018	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_1gpu/checkpoint-epoch1.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_1gpu/weights-epoch1.pth
```

### `rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch13.pth` — `170,946,742` bytes (`170.95 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch13.pth` — `119,128,610` bytes (`119.13 MB`)

Delete `24` files / `3,480,863,632` bytes (`3.48 GB`):

```text
170940018	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch1.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch1.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch2.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch2.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch3.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch3.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch4.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch4.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch5.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch5.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch6.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch6.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch7.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch7.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch8.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch8.pth
170944178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch9.pth
119127126	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch9.pth
170946742	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch10.pth
119128610	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch10.pth
170946742	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch11.pth
119128610	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch11.pth
170946742	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/checkpoint-epoch12.pth
119128610	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN5b_clean_identity_tokens_nfs_2gpu/weights-epoch12.pth
```

### `rsrch/diffusion_template/saved/ba_NN6a_factorized_identity_only_up0_nfs_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN6a_factorized_identity_only_up0_nfs_2gpu/checkpoint-epoch2.pth` — `140,199,122` bytes (`140.20 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN6a_factorized_identity_only_up0_nfs_2gpu/weights-epoch2.pth` — `108,888,710` bytes (`108.89 MB`)

Delete `2` files / `249,087,832` bytes (`0.25 GB`):

```text
140199122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN6a_factorized_identity_only_up0_nfs_2gpu/checkpoint-epoch1.pth
108888710	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_NN6a_factorized_identity_only_up0_nfs_2gpu/weights-epoch1.pth
```

### `rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/checkpoint-epoch5.pth` — `212,971,066` bytes (`212.97 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/weights-epoch5.pth` — `133,126,946` bytes (`133.13 MB`)

Delete `8` files / `1,384,392,048` bytes (`1.38 GB`):

```text
212971066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/checkpoint-epoch1.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/weights-epoch1.pth
212971066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/checkpoint-epoch2.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/weights-epoch2.pth
212971066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/checkpoint-epoch3.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/weights-epoch3.pth
212971066	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/checkpoint-epoch4.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_bboxnorm_idtokens_N30/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_camix_train_N23`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch10.pth` — `286,147,510` bytes (`286.15 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch10.pth` — `157,487,362` bytes (`157.49 MB`)

Delete `18` files / `3,992,609,416` bytes (`3.99 GB`):

```text
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch1.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch2.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch3.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch4.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch5.pth
286143026	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch6.pth
286143026	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch7.pth
286143026	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch8.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch8.pth
286143026	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/checkpoint-epoch9.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_camix_train_N23/weights-epoch9.pth
```

### `rsrch/diffusion_template/saved/ba_causal_highres_qformer_4gpu_N34`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_causal_highres_qformer_4gpu_N34/checkpoint-epoch2.pth` — `108,820,154` bytes (`108.82 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_causal_highres_qformer_4gpu_N34/weights-epoch2.pth` — `98,442,722` bytes (`98.44 MB`)

Delete `2` files / `207,262,876` bytes (`0.21 GB`):

```text
108820154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_causal_highres_qformer_4gpu_N34/checkpoint-epoch1.pth
98442722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_causal_highres_qformer_4gpu_N34/weights-epoch1.pth
```

### `rsrch/diffusion_template/saved/ba_dualgate_train_N24`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch10.pth` — `286,220,134` bytes (`286.22 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch10.pth` — `157,507,756` bytes (`157.51 MB`)

Delete `18` files / `3,993,489,828` bytes (`3.99 GB`):

```text
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch1.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch1.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch2.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch2.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch3.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch3.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch4.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch4.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch5.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch5.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch6.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch6.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch7.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch7.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch8.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch8.pth
286215370	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/checkpoint-epoch9.pth
157505722	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_dualgate_train_N24/weights-epoch9.pth
```

### `rsrch/diffusion_template/saved/ba_facepatch_resampler_N32`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch9.pth` — `227,250,806` bytes (`227.25 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch9.pth` — `137,885,100` bytes (`137.89 MB`)

Delete `16` files / `2,921,087,248` bytes (`2.92 GB`):

```text
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch1.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch1.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch2.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch2.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch3.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch3.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch4.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch4.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch5.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch5.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch6.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch6.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch7.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch7.pth
227250806	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/checkpoint-epoch8.pth
137885100	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_facepatch_resampler_N32/weights-epoch8.pth
```

### `rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/checkpoint-epoch6.pth` — `212,971,834` bytes (`212.97 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/weights-epoch6.pth` — `133,126,946` bytes (`133.13 MB`)

Delete `10` files / `1,730,493,900` bytes (`1.73 GB`):

```text
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/checkpoint-epoch1.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/weights-epoch1.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/checkpoint-epoch2.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/weights-epoch2.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/checkpoint-epoch3.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/weights-epoch3.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/checkpoint-epoch4.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/weights-epoch4.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/checkpoint-epoch5.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_2gpu_N31/weights-epoch5.pth
```

### `rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch7.pth` — `212,971,834` bytes (`212.97 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch7.pth` — `133,126,946` bytes (`133.13 MB`)

Delete `12` files / `2,076,592,680` bytes (`2.08 GB`):

```text
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch1.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch1.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch2.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch2.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch3.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch3.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch4.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch4.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch5.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch5.pth
212971834	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/checkpoint-epoch6.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_dependence_4gpu_N31/weights-epoch6.pth
```

### `rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch8.pth` — `144,298,578` bytes (`144.30 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch8.pth` — `110,262,790` bytes (`110.26 MB`)

Delete `14` files / `1,781,924,968` bytes (`1.78 GB`):

```text
144297042	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch1.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch1.pth
144297042	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch2.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch2.pth
144297042	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch3.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch3.pth
144298578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch4.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch4.pth
144298578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch5.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch5.pth
144298578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch6.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch6.pth
144298578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/checkpoint-epoch7.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_cropped_qformer_2gpu_N38/weights-epoch7.pth
```

### `rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch8.pth` — `164,982,606` bytes (`164.98 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch8.pth` — `117,156,048` bytes (`117.16 MB`)

Delete `14` files / `1,974,970,578` bytes (`1.97 GB`):

```text
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch1.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch1.pth
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch2.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch2.pth
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch3.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch3.pth
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch4.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch4.pth
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch5.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch5.pth
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch6.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch6.pth
164982606	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/checkpoint-epoch7.pth
117156048	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_hybrid_2gpu_N37/weights-epoch7.pth
```

### `rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch8.pth` — `144,298,450` bytes (`144.30 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch8.pth` — `110,262,790` bytes (`110.26 MB`)

Delete `14` files / `1,781,924,072` bytes (`1.78 GB`):

```text
144296914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch1.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch1.pth
144296914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch2.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch2.pth
144296914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch3.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch3.pth
144298450	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch4.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch4.pth
144298450	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch5.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch5.pth
144298450	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch6.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch6.pth
144298450	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/checkpoint-epoch7.pth
110262790	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_identity_owner_qformer_2gpu_N36/weights-epoch7.pth
```

### `rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch10.pth` — `216,496,566` bytes (`216.50 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch10.pth` — `134,296,346` bytes (`134.30 MB`)

Delete `18` files / `3,157,090,776` bytes (`3.16 GB`):

```text
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch1.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch1.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch2.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch2.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch3.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch3.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch4.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch4.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch5.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch5.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch6.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch6.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch7.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch7.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch8.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch8.pth
216493202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/checkpoint-epoch9.pth
134294662	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_idtoken_ca_residual_N28/weights-epoch9.pth
```

### `rsrch/diffusion_template/saved/ba_qformer_continue40k_N33`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch14.pth` — `212,979,654` bytes (`212.98 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch14.pth` — `133,128,560` bytes (`133.13 MB`)

Delete `16` files / `2,768,846,920` bytes (`2.77 GB`):

```text
212976570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch6.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch6.pth
212976570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch7.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch7.pth
212976570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch8.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch8.pth
212976570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch9.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch9.pth
212979654	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch10.pth
133128560	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch10.pth
212979654	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch11.pth
133128560	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch11.pth
212979654	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch12.pth
133128560	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch12.pth
212979654	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/checkpoint-epoch13.pth
133128560	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_continue40k_N33/weights-epoch13.pth
```

### `rsrch/diffusion_template/saved/ba_qformer_idtokens_N29`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/checkpoint-epoch5.pth` — `212,971,002` bytes (`212.97 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/weights-epoch5.pth` — `133,126,946` bytes (`133.13 MB`)

Delete `8` files / `1,384,391,792` bytes (`1.38 GB`):

```text
212971002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/checkpoint-epoch1.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/weights-epoch1.pth
212971002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/checkpoint-epoch2.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/weights-epoch2.pth
212971002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/checkpoint-epoch3.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/weights-epoch3.pth
212971002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/checkpoint-epoch4.pth
133126946	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch10.pth` — `189,868,678` bytes (`189.87 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch10.pth` — `125,426,160` bytes (`125.43 MB`)

Delete `18` files / `2,837,611,260` bytes (`2.84 GB`):

```text
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch1.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch1.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch2.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch2.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch3.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch3.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch4.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch4.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch5.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch5.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch6.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch6.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch7.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch7.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch8.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch8.pth
189865594	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/checkpoint-epoch9.pth
125424546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_spatial_roi_residual_N27/weights-epoch9.pth
```

### `rsrch/diffusion_template/saved/ba_staged_caref_N26`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch10.pth` — `405,600,150` bytes (`405.60 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch10.pth` — `197,275,774` bytes (`197.28 MB`)

Delete `18` files / `5,425,806,384` bytes (`5.43 GB`):

```text
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch1.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch1.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch2.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch2.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch3.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch3.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch4.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch4.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch5.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch5.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch6.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch6.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch7.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch7.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch8.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch8.pth
405593986	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/checkpoint-epoch9.pth
197273390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_caref_N26/weights-epoch9.pth
```

### `rsrch/diffusion_template/saved/ba_staged_legacy_N25`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch10.pth` — `286,138,230` bytes (`286.14 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch10.pth` — `157,487,362` bytes (`157.49 MB`)

Delete `18` files / `3,992,572,296` bytes (`3.99 GB`):

```text
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch1.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch2.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch3.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch4.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch5.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch6.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch7.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch8.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch8.pth
286133746	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/checkpoint-epoch9.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ba_staged_legacy_N25/weights-epoch9.pth
```

### `rsrch/diffusion_template/saved/cometL_256_crop`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_crop/checkpoint-epoch2.pth` — `525,018,706` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_crop/weights-epoch2.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `2` files / `762,075,480` bytes (`0.76 GB`):

```text
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_crop/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_crop/weights-epoch1.pth
```

### `rsrch/diffusion_template/saved/cometL_256_new`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/checkpoint-epoch4.pth` — `525,034,578` bytes (`525.03 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/weights-epoch4.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `6` files / `2,286,274,056` bytes (`2.29 GB`):

```text
525034578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/weights-epoch1.pth
525034578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/weights-epoch2.pth
525034578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_256_new/weights-epoch3.pth
```

### `rsrch/diffusion_template/saved/cometL_const_ref_1024`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/checkpoint-epoch5.pth` — `525,017,682` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/weights-epoch5.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `8` files / `3,048,297,824` bytes (`3.05 GB`):

```text
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/weights-epoch1.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/weights-epoch2.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/weights-epoch3.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch7.pth` — `525,017,682` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch7.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `12` files / `4,572,446,736` bytes (`4.57 GB`):

```text
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch1.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch2.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch3.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch4.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch5.pth
525017682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_1024_2gpu/weights-epoch6.pth
```

### `rsrch/diffusion_template/saved/cometL_const_ref_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch7.pth` — `525,018,706` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch7.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `12` files / `4,572,452,880` bytes (`4.57 GB`):

```text
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch1.pth
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch2.pth
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch3.pth
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch4.pth
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch5.pth
525018706	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cometL_const_ref_2gpu/weights-epoch6.pth
```

### `rsrch/diffusion_template/saved/comet_large_10Apr_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/checkpoint-epoch5.pth` — `525,016,786` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/weights-epoch5.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `8` files / `3,048,294,240` bytes (`3.05 GB`):

```text
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/weights-epoch1.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/weights-epoch2.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/weights-epoch3.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_10Apr_2gpu/weights-epoch4.pth
```

### `rsrch/diffusion_template/saved/comet_large_256`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256/checkpoint-epoch2.pth` — `525,017,298` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256/weights-epoch2.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `4` files / `1,524,148,144` bytes (`1.52 GB`):

```text
525017298	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256/weights-epoch1.pth
525017298	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256/weights-epoch3.pth
```

### `rsrch/diffusion_template/saved/comet_large_256_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/checkpoint-epoch4.pth` — `525,017,362` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/weights-epoch4.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `6` files / `2,286,222,408` bytes (`2.29 GB`):

```text
525017362	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/weights-epoch1.pth
525017362	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/weights-epoch2.pth
525017362	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_256_2gpu/weights-epoch3.pth
```

### `rsrch/diffusion_template/saved/comet_large_9Apr`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/checkpoint-epoch6.pth` — `525,016,786` bytes (`525.02 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/weights-epoch6.pth` — `237,056,774` bytes (`237.06 MB`)

Delete `10` files / `3,810,367,800` bytes (`3.81 GB`):

```text
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/weights-epoch1.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/weights-epoch2.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/weights-epoch3.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/weights-epoch4.pth
525016786	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/comet_large_9Apr/weights-epoch5.pth
```

### `rsrch/diffusion_template/saved/cosm_new3`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch12.pth` — `525,053,366` bytes (`525.05 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch12.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `22` files / `8,383,146,552` bytes (`8.38 GB`):

```text
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch1.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch2.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch3.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch4.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch5.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch6.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch7.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch8.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch9.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch10.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3/weights-epoch11.pth
```

### `rsrch/diffusion_template/saved/cosm_new3_256`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch60.pth` — `525,053,366` bytes (`525.05 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch60.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `118` files / `44,964,567,864` bytes (`44.96 GB`):

```text
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch1.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch2.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch3.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch4.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch5.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch6.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch7.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch8.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch9.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch10.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch11.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch12.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch12.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch13.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch13.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch14.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch14.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch15.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch15.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch16.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch16.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch17.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch17.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch18.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch18.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch19.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch19.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch20.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch20.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch21.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch21.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch22.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch22.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch23.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch23.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch24.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch24.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch25.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch25.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch26.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch26.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch27.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch27.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch28.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch28.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch29.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch29.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch30.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch30.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch31.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch31.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch32.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch32.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch33.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch33.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch34.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch34.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch35.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch35.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch36.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch36.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch37.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch37.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch38.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch38.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch39.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch39.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch40.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch40.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch41.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch41.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch42.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch42.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch43.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch43.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch44.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch44.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch45.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch45.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch46.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch46.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch47.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch47.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch48.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch48.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch49.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch49.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch50.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch50.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch51.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch51.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch52.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch52.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch53.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch53.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch54.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch54.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch55.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch55.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch56.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch56.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch57.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch57.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch58.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch58.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/checkpoint-epoch59.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256/weights-epoch59.pth
```

### `rsrch/diffusion_template/saved/cosm_new3_256_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch50.pth` — `525,053,366` bytes (`525.05 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch50.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `98` files / `37,343,438,424` bytes (`37.34 GB`):

```text
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch1.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch2.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch3.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch4.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch5.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch6.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch7.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch8.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch9.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch10.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch11.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch12.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch12.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch13.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch13.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch14.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch14.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch15.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch15.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch16.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch16.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch17.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch17.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch18.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch18.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch19.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch19.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch20.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch20.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch21.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch21.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch22.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch22.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch23.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch23.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch24.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch24.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch25.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch25.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch26.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch26.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch27.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch27.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch28.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch28.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch29.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch29.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch30.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch30.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch31.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch31.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch32.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch32.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch33.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch33.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch34.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch34.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch35.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch35.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch36.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch36.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch37.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch37.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch38.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch38.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch39.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch39.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch40.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch40.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch41.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch41.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch42.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch42.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch43.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch43.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch44.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch44.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch45.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch45.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch46.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch46.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch47.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch47.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch48.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch48.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/checkpoint-epoch49.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_256_2gpu/weights-epoch49.pth
```

### `rsrch/diffusion_template/saved/cosm_new3_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch38.pth` — `525,053,366` bytes (`525.05 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch38.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `74` files / `28,198,083,096` bytes (`28.20 GB`):

```text
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch1.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch2.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch3.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch4.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch5.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch6.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch7.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch8.pth
525045522	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch9.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch10.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch11.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch12.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch12.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch13.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch13.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch14.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch14.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch15.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch15.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch16.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch16.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch17.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch17.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch18.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch18.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch19.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch19.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch20.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch20.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch21.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch21.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch22.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch22.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch23.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch23.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch24.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch24.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch25.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch25.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch26.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch26.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch27.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch27.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch28.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch28.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch29.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch29.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch30.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch30.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch31.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch31.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch32.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch32.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch33.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch33.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch34.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch34.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch35.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch35.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch36.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch36.pth
525053366	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/checkpoint-epoch37.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/cosm_new3_2gpu/weights-epoch37.pth
```

### `rsrch/diffusion_template/saved/large_ds`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch14.pth` — `525,030,838` bytes (`525.03 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch14.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `26` files / `9,907,079,576` bytes (`9.91 GB`):

```text
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch1.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch2.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch3.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch4.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch5.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch6.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch7.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch8.pth
525022994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch9.pth
525030838	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch10.pth
525030838	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch11.pth
525030838	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch12.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch12.pth
525030838	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/checkpoint-epoch13.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/large_ds/weights-epoch13.pth
```

### `rsrch/diffusion_template/saved/ot_gr`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch18.pth` — `525,043,062` bytes (`525.04 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch18.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `34` files / `12,955,649,048` bytes (`12.96 GB`):

```text
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch1.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch2.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch3.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch4.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch5.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch6.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch7.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch8.pth
525035218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch9.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch10.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch11.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch12.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch12.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch13.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch13.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch14.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch14.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch15.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch15.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch16.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch16.pth
525043062	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/checkpoint-epoch17.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr/weights-epoch17.pth
```

### `rsrch/diffusion_template/saved/ot_gr_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch31.pth` — `525,052,022` bytes (`525.05 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch31.pth` — `237,059,578` bytes (`237.06 MB`)

Delete `60` files / `22,863,252,168` bytes (`22.86 GB`):

```text
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch1.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch1.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch2.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch2.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch3.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch3.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch4.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch4.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch5.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch5.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch6.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch6.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch7.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch7.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch8.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch8.pth
525044178	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch9.pth
237056774	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch9.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch10.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch10.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch11.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch11.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch12.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch12.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch13.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch13.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch14.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch14.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch15.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch15.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch16.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch16.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch17.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch17.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch18.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch18.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch19.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch19.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch20.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch20.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch21.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch21.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch22.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch22.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch23.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch23.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch24.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch24.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch25.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch25.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch26.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch26.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch27.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch27.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch28.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch28.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch29.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch29.pth
525052022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/checkpoint-epoch30.pth
237059578	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template/saved/ot_gr_2gpu/weights-epoch30.pth
```

### `rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch10.pth` — `193,035,526` bytes (`193.04 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch10.pth` — `64,340,466` bytes (`64.34 MB`)

Delete `18` files / `2,316,345,480` bytes (`2.32 GB`):

```text
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch1.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch1.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch2.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch2.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch3.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch3.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch4.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch4.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch5.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch5.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch6.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch6.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch7.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch7.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch8.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch8.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/checkpoint-epoch9.pth
64339622	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch10.pth` — `653,124,406` bytes (`653.12 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch10.pth` — `157,495,218` bytes (`157.50 MB`)

Delete `18` files / `7,295,480,208` bytes (`7.30 GB`):

```text
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch1.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch1.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch2.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch2.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch3.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch3.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch4.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch4.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch5.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch5.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch6.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch6.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch7.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch7.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch8.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch8.pth
653115658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/checkpoint-epoch9.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E0_large_ds_base_historical_r4_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch10.pth` — `193,035,654` bytes (`193.04 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch10.pth` — `64,340,658` bytes (`64.34 MB`)

Delete `18` files / `2,316,348,360` bytes (`2.32 GB`):

```text
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch1.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch1.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch2.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch2.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch3.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch3.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch4.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch4.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch5.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch5.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch6.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch6.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch7.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch7.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch8.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch8.pth
193032226	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/checkpoint-epoch9.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E1_large_ds_truekey_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch10.pth` — `225,188,134` bytes (`225.19 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch10.pth` — `75,062,698` bytes (`75.06 MB`)

Delete `18` files / `2,702,212,740` bytes (`2.70 GB`):

```text
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch1.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch1.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch2.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch2.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch3.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch3.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch4.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch4.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch5.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch5.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch6.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch6.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch7.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch7.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch8.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch8.pth
225184146	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/checkpoint-epoch9.pth
75061714	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E2_large_ds_branchout_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch10.pth` — `193,035,718` bytes (`193.04 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch10.pth` — `64,340,658` bytes (`64.34 MB`)

Delete `18` files / `2,316,348,936` bytes (`2.32 GB`):

```text
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch1.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch1.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch2.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch2.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch3.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch3.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch4.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch4.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch5.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch5.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch6.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch6.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch7.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch7.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch8.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch8.pth
193032290	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/checkpoint-epoch9.pth
64339814	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E3_large_ds_roiwarp_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch10.pth` — `127,732,934` bytes (`127.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch10.pth` — `42,561,138` bytes (`42.56 MB`)

Delete `18` files / `1,532,621,160` bytes (`1.53 GB`):

```text
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch1.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch1.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch2.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch2.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch3.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch3.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch4.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch4.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch5.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch5.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch6.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch6.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch7.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch7.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch8.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch8.pth
127730658	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/checkpoint-epoch9.pth
42560582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E4_large_ds_midup_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch10.pth` — `193,035,526` bytes (`193.04 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch10.pth` — `64,340,530` bytes (`64.34 MB`)

Delete `18` files / `2,316,346,056` bytes (`2.32 GB`):

```text
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch1.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch1.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch2.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch2.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch3.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch3.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch4.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch4.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch5.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch5.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch6.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch6.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch7.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch7.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch8.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch8.pth
193032098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/checkpoint-epoch9.pth
64339686	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E5_large_ds_infersteps_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch10.pth` — `384,727,430` bytes (`384.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch10.pth` — `128,237,234` bytes (`128.24 MB`)

Delete `18` files / `4,616,643,528` bytes (`4.62 GB`):

```text
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch1.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch1.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch2.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch2.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch3.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch3.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch4.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch4.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch5.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch5.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch6.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch6.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch7.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch7.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch8.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch8.pth
384724002	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/checkpoint-epoch9.pth
128236390	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/E6_large_ds_fp32_r32_20k_full96_r1/weights-epoch9.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch8.pth` — `653,056,666` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,794,448` bytes (`5.67 GB`):

```text
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch1.pth
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch2.pth
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch3.pth
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch4.pth
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch5.pth
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch6.pth
653056666	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch8.pth` — `653,056,858` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,795,792` bytes (`5.67 GB`):

```text
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch1.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch2.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch3.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch4.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch5.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch6.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_legacy_4k_fast_r2/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch8.pth` — `653,056,858` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,795,792` bytes (`5.67 GB`):

```text
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch1.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch2.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch3.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch4.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch5.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch6.pth
653056858	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_fast_r3/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch8.pth` — `653,056,730` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,794,896` bytes (`5.67 GB`):

```text
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch1.pth
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch2.pth
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch3.pth
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch4.pth
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch5.pth
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch6.pth
653056730	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_4k_r1/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch9.pth` — `653,057,114` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch9.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `16` files / `6,484,340,096` bytes (`6.48 GB`):

```text
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch1.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch2.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch3.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch4.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch5.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch6.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch7.pth
653057114	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/checkpoint-epoch8.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/weights-epoch8.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch8.pth` — `653,056,922` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,796,240` bytes (`5.67 GB`):

```text
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch1.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch2.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch3.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch4.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch5.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch6.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch8.pth` — `653,056,922` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,796,240` bytes (`5.67 GB`):

```text
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch1.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch2.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch3.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch4.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch5.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch6.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_legacy_4k_fast_r1/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch8.pth` — `653,056,922` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,796,240` bytes (`5.67 GB`):

```text
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch1.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch2.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch3.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch4.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch5.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch6.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch8.pth` — `653,056,922` bytes (`653.06 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `14` files / `5,673,796,240` bytes (`5.67 GB`):

```text
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch1.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch2.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch3.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch4.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch5.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch6.pth
653056922	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/weights-epoch7.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch40.pth` — `653,095,430` bytes (`653.10 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `22` files / `10,898,563,668` bytes (`10.90 GB`):

```text
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch1.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch2.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch3.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch4.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch5.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch6.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/weights-epoch7.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch8.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch12.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch16.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch20.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch24.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch28.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch32.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_highest_4k/checkpoint-epoch36.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch40.pth` — `653,095,430` bytes (`653.10 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `22` files / `10,898,563,668` bytes (`10.90 GB`):

```text
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch1.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch2.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch3.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch4.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch5.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch6.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/weights-epoch7.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch8.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch12.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch16.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch20.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch24.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch28.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch32.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_top3softmax_4k_r2/checkpoint-epoch36.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch20.pth` — `653,095,430` bytes (`653.10 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `17` files / `7,633,086,518` bytes (`7.63 GB`):

```text
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch1.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch2.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch3.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch4.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch5.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch6.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/weights-epoch7.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch8.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch12.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_distinct_uniform_4k/checkpoint-epoch16.pth
```

### `rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch40.pth` — `653,095,430` bytes (`653.10 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch8.pth` — `157,485,398` bytes (`157.49 MB`)

Delete `22` files / `10,898,563,668` bytes (`10.90 GB`):

```text
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch1.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch1.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch2.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch2.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch3.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch3.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch4.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch4.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch5.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch5.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch6.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch6.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch7.pth
157485398	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/weights-epoch7.pth
653062234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch8.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch12.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch16.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch20.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch24.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch28.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch32.pth
653095430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_cosmic_initial_selfref_minface256_4k/checkpoint-epoch36.pth
```

### `rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch80.pth` — `653,138,678` bytes (`653.14 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch80.pth` — `157,495,218` bytes (`157.50 MB`)

Delete `38` files / `15,401,998,088` bytes (`15.40 GB`):

```text
653105418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch4.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch4.pth
653129930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch8.pth
157493254	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch8.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch12.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch12.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch16.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch16.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch20.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch20.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch24.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch24.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch28.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch28.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch32.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch32.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch36.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch36.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch40.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch40.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch44.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch44.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch48.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch48.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch52.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch52.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch56.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch56.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch60.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch60.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch64.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch64.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch68.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch68.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch72.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch72.pth
653138678	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/checkpoint-epoch76.pth
157495218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/saved/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/weights-epoch76.pth
```

### `runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,753,670` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,952,184` bytes (`19.34 GB`):

```text
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch1.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch2.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch3.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch4.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch5.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch6.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch7.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch8.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch9.pth
1318753670	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch10.pth
1318753670	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl14_ca_v1/CL14_CA/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,815,270` bytes (`1318.82 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch12.pth` — `439,632,454` bytes (`439.63 MB`)

Delete `22` files / `19,342,821,896` bytes (`19.34 GB`):

```text
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch1.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch1.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch2.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch2.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch3.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch3.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch4.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch4.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch5.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch5.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch6.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch6.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch7.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch7.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch8.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch8.pth
1318806098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch9.pth
439630174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch9.pth
1318815270	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch10.pth
439632454	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch10.pth
1318815270	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/checkpoint-epoch11.pth
439632454	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/diffusion_template/saved/CL15_cosmic_shared_highres_roi_ba_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch12.pth` — `1,446,044,782` bytes (`1446.04 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch12.pth` — `482,051,920` bytes (`482.05 MB`)

Delete `22` files / `21,208,947,784` bytes (`21.21 GB`):

```text
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch1.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch1.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch2.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch2.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch3.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch3.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch4.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch4.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch5.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch5.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch6.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch6.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch7.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch7.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch8.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch8.pth
1446034466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch9.pth
482049354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch9.pth
1446044782	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch10.pth
482051920	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch10.pth
1446044782	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/checkpoint-epoch11.pth
482051920	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/diffusion_template/saved/CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch12.pth` — `1,351,575,334` bytes (`1351.58 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch12.pth` — `450,558,006` bytes (`450.56 MB`)

Delete `22` files / `19,823,357,192` bytes (`19.82 GB`):

```text
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch1.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch1.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch2.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch2.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch3.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch3.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch4.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch4.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch5.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch5.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch6.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch6.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch7.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch7.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch8.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch8.pth
1351565586	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch9.pth
450555582	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch9.pth
1351575334	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch10.pth
450558006	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch10.pth
1351575334	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/checkpoint-epoch11.pth
450558006	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/diffusion_template/saved/CL17_cosmic_semantic_visibility_ownership_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,765,446` bytes (`1318.77 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch12.pth` — `439,614,970` bytes (`439.61 MB`)

Delete `22` files / `19,342,083,128` bytes (`19.34 GB`):

```text
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch1.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch1.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch2.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch2.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch3.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch3.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch4.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch4.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch5.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch5.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch6.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch6.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch7.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch7.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch8.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch8.pth
1318756418	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch9.pth
439612726	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch9.pth
1318765446	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch10.pth
439614970	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch10.pth
1318765446	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/checkpoint-epoch11.pth
439614970	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/diffusion_template/saved/CL18_cosmic_crossview_spatial_consistency_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,766,598` bytes (`1318.77 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch12.pth` — `439,615,354` bytes (`439.62 MB`)

Delete `22` files / `19,342,100,024` bytes (`19.34 GB`):

```text
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch1.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch1.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch2.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch2.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch3.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch3.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch4.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch4.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch5.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch5.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch6.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch6.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch7.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch7.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch8.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch8.pth
1318757570	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch9.pth
439613110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch9.pth
1318766598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch10.pth
439615354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch10.pth
1318766598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/checkpoint-epoch11.pth
439615354	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,765,574` bytes (`1318.77 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,342,083,128` bytes (`19.34 GB`):

```text
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch1.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch2.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch3.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch4.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch5.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch6.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch7.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch8.pth
1318756546	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch9.pth
1318765574	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch10.pth
1318765574	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl15_cl20_v1/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/diffusion_template/saved/CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,734,214` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,738,168` bytes (`19.34 GB`):

```text
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch1.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch2.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch3.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch4.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch5.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch6.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch7.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch8.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch9.pth
1318734214	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch10.pth
1318734214	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/diffusion_template/saved/CL0_cosmic_joint_shadow_sa128_asis_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,735,942` bytes (`1318.74 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,757,176` bytes (`19.34 GB`):

```text
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch1.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch2.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch3.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch4.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch5.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch6.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch7.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch8.pth
1318726914	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch9.pth
1318735942	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch10.pth
1318735942	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/diffusion_template/saved/CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,735,430` bytes (`1318.74 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,751,544` bytes (`19.34 GB`):

```text
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch1.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch2.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch3.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch4.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch5.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch6.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch7.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch8.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch9.pth
1318735430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch10.pth
1318735430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,754,118` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,957,112` bytes (`19.34 GB`):

```text
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch1.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch2.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch3.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch4.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch5.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch6.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch7.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch8.pth
1318745090	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch9.pth
1318754118	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch10.pth
1318754118	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/diffusion_template/saved/CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,753,670` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,952,184` bytes (`19.34 GB`):

```text
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch1.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch2.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch3.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch4.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch5.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch6.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch7.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch8.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch9.pth
1318753670	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch10.pth
1318753670	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/diffusion_template/saved/CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,753,670` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,952,184` bytes (`19.34 GB`):

```text
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch1.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch2.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch3.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch4.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch5.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch6.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch7.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch8.pth
1318744642	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch9.pth
1318753670	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch10.pth
1318753670	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/diffusion_template/saved/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,734,022` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,736,056` bytes (`19.34 GB`):

```text
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch1.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch2.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch3.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch4.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch5.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch6.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch7.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch8.pth
1318724994	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch9.pth
1318734022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch10.pth
1318734022	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/diffusion_template/saved/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,733,958` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,735,352` bytes (`19.34 GB`):

```text
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch1.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch2.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch3.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch4.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch5.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch6.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch7.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch8.pth
1318724930	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch9.pth
1318733958	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch10.pth
1318733958	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/diffusion_template/saved/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,734,150` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,737,464` bytes (`19.34 GB`):

```text
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch1.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch2.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch3.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch4.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch5.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch6.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch7.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch8.pth
1318725122	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch9.pth
1318734150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch10.pth
1318734150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/diffusion_template/saved/CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,734,214` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,738,168` bytes (`19.34 GB`):

```text
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch1.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch2.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch3.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch4.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch5.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch6.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch7.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch8.pth
1318725186	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch9.pth
1318734214	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch10.pth
1318734214	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/diffusion_template/saved/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,735,494` bytes (`1318.74 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,752,248` bytes (`19.34 GB`):

```text
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch1.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch2.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch3.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch4.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch5.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch6.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch7.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch8.pth
1318726466	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch9.pth
1318735494	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch10.pth
1318735494	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/diffusion_template/saved/CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,735,110` bytes (`1318.74 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,748,024` bytes (`19.34 GB`):

```text
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch1.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch2.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch3.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch4.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch5.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch6.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch7.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch8.pth
1318726082	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch9.pth
1318735110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch10.pth
1318735110	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/diffusion_template/saved/CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,735,430` bytes (`1318.74 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,751,544` bytes (`19.34 GB`):

```text
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch1.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch2.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch3.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch4.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch5.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch6.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch7.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch8.pth
1318726402	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch9.pth
1318735430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch10.pth
1318735430	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/diffusion_template/saved/CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,735,238` bytes (`1318.74 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,749,432` bytes (`19.34 GB`):

```text
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch1.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch2.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch3.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch4.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch5.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch6.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch7.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch8.pth
1318726210	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch9.pth
1318735238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch10.pth
1318735238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/weights-epoch11.pth
```

### `runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,728,326` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,672,824` bytes (`19.34 GB`):

```text
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch1.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch2.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch3.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch4.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch5.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch6.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch7.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch8.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch9.pth
1318728326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch10.pth
1318728326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/diffusion_template/saved/E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch12.pth` — `1,350,880,486` bytes (`1350.88 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch12.pth` — `450,337,150` bytes (`450.34 MB`)

Delete `22` files / `19,813,286,248` bytes (`19.81 GB`):

```text
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch1.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch1.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch2.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch2.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch3.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch3.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch4.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch4.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch5.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch5.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch6.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch6.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch7.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch7.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch8.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch8.pth
1350870898	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch9.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch9.pth
1350880486	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch10.pth
450337150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch10.pth
1350880486	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/checkpoint-epoch11.pth
450337150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/diffusion_template/saved/E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch12.pth` — `1,350,880,742` bytes (`1350.88 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch12.pth` — `450,337,150` bytes (`450.34 MB`)

Delete `22` files / `19,813,289,064` bytes (`19.81 GB`):

```text
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch1.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch1.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch2.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch2.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch3.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch3.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch4.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch4.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch5.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch5.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch6.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch6.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch7.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch7.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch8.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch8.pth
1350871154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch9.pth
450334766	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch9.pth
1350880742	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch10.pth
450337150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch10.pth
1350880742	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/checkpoint-epoch11.pth
450337150	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/diffusion_template/saved/E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,732,230` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch12.pth` — `439,615,482` bytes (`439.62 MB`)

Delete `22` files / `19,341,723,384` bytes (`19.34 GB`):

```text
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch1.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch1.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch2.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch2.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch3.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch3.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch4.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch4.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch5.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch5.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch6.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch6.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch7.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch7.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch8.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch8.pth
1318723202	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch9.pth
439613238	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch9.pth
1318732230	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch10.pth
439615482	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch10.pth
1318732230	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/checkpoint-epoch11.pth
439615482	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/diffusion_template/saved/E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,727,942` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,668,600` bytes (`19.34 GB`):

```text
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch1.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch2.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch3.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch4.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch5.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch6.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch7.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch8.pth
1318718850	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch9.pth
1318727942	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch10.pth
1318727942	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/diffusion_template/saved/E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2/weights-epoch11.pth
```

### `runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,728,902` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,679,736` bytes (`19.34 GB`):

```text
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch1.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch2.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch3.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch4.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch5.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch6.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch7.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch8.pth
1318719874	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch9.pth
1318728902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch10.pth
1318728902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_e19_e24_v3/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/diffusion_template/saved/E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,728,134` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,671,288` bytes (`19.34 GB`):

```text
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch1.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch2.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch3.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch4.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch5.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch6.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch7.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch8.pth
1318719106	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch9.pth
1318728134	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch10.pth
1318728134	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,754,182` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,957,816` bytes (`19.34 GB`):

```text
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch1.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch2.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch3.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch4.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch5.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch6.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch7.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch8.pth
1318745154	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch9.pth
1318754182	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch10.pth
1318754182	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds1_repeatdepth_balanced_24k_full96_r1/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,754,246` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,958,520` bytes (`19.34 GB`):

```text
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch1.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch2.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch3.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch4.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch5.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch6.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch7.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch8.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch9.pth
1318754246	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch10.pth
1318754246	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch12.pth` — `1,318,754,246` bytes (`1318.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,958,520` bytes (`19.34 GB`):

```text
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch1.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch2.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch3.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch4.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch5.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch6.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch7.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch8.pth
1318745218	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch9.pth
1318754246	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch10.pth
1318754246	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template/saved/BC_E13_ds3_large_anchor_2to1_24k_full96_r1/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch10.pth` — `768,133,062` bytes (`768.13 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch10.pth` — `256,035,186` bytes (`256.04 MB`)

Delete `18` files / `9,217,475,784` bytes (`9.22 GB`):

```text
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch1.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch1.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch2.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch2.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch3.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch3.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch4.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch4.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch5.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch5.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch6.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch6.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch7.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch7.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch8.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch8.pth
768129634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/checkpoint-epoch9.pth
256034342	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E11_large_ds_ba_sa_r128_20k_full96_r1/weights-epoch9.pth
```

### `runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch10.pth` — `809,252,614` bytes (`809.25 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch10.pth` — `269,753,458` bytes (`269.75 MB`)

Delete `18` files / `9,711,003,240` bytes (`9.71 GB`):

```text
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch1.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch1.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch2.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch2.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch3.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch3.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch4.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch4.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch5.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch5.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch6.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch6.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch7.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch7.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch8.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch8.pth
809248034	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/checkpoint-epoch9.pth
269752326	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804/diffusion_template/saved/E12_large_ds_ba_idca_up_r256_20k_full96_r1/weights-epoch9.pth
```

### `runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch12.pth` — `1,318,726,918` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,657,912` bytes (`19.34 GB`):

```text
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch1.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch2.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch3.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch4.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch5.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch6.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch7.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch8.pth
1318717890	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch9.pth
1318726918	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch10.pth
1318726918	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch12.pth` — `1,318,728,198` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,671,992` bytes (`19.34 GB`):

```text
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch1.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch2.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch3.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch4.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch5.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch6.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch7.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch8.pth
1318719170	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch9.pth
1318728198	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch10.pth
1318728198	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template/saved/E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,728,262` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,672,696` bytes (`19.34 GB`):

```text
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch1.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch2.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch3.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch4.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch5.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch6.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch7.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch8.pth
1318719234	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch9.pth
1318728262	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch10.pth
1318728262	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E15_large_ds_joint_persist_sa128_protected_24k_full96_r2/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch12.pth` — `1,318,729,286` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch12.pth` — `439,615,098` bytes (`439.62 MB`)

Delete `22` files / `19,341,686,776` bytes (`19.34 GB`):

```text
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch1.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch1.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch2.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch2.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch3.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch3.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch4.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch4.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch5.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch5.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch6.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch6.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch7.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch7.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch8.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch8.pth
1318720258	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch9.pth
439612854	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch9.pth
1318729286	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch10.pth
439615098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch10.pth
1318729286	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/checkpoint-epoch11.pth
439615098	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template/saved/E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch12.pth` — `1,383,770,598` bytes (`1383.77 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch12.pth` — `461,302,174` bytes (`461.30 MB`)

Delete `22` files / `20,295,694,184` bytes (`20.30 GB`):

```text
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch1.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch1.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch2.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch2.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch3.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch3.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch4.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch4.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch5.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch5.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch6.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch6.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch7.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch7.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch8.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch8.pth
1383761138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch9.pth
461299822	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch9.pth
1383770598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch10.pth
461302174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch10.pth
1383770598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/checkpoint-epoch11.pth
461302174	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template/saved/E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch12.pth` — `1,318,728,710` bytes (`1318.73 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch12.pth` — `439,614,842` bytes (`439.61 MB`)

Delete `22` files / `19,341,677,624` bytes (`19.34 GB`):

```text
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch1.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch1.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch2.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch2.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch3.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch3.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch4.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch4.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch5.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch5.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch6.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch6.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch7.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch7.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch8.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch8.pth
1318719682	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch9.pth
439612598	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch9.pth
1318728710	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch10.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch10.pth
1318728710	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/checkpoint-epoch11.pth
439614842	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template/saved/E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4/weights-epoch11.pth
```

### `runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch10.pth` — `559,750,758` bytes (`559.75 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch10.pth` — `186,601,446` bytes (`186.60 MB`)

Delete `18` files / `6,717,099,888` bytes (`6.72 GB`):

```text
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch1.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch1.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch2.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch2.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch3.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch3.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch4.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch4.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch5.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch5.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch6.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch6.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch7.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch7.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch8.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch8.pth
559744530	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/checkpoint-epoch9.pth
186599902	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/weights-epoch9.pth
```

### `runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch10.pth` — `376,909,478` bytes (`376.91 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch10.pth` — `125,658,982` bytes (`125.66 MB`)

Delete `18` files / `4,523,046,192` bytes (`4.52 GB`):

```text
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch1.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch1.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch2.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch2.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch3.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch3.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch4.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch4.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch5.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch5.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch6.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch6.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch7.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch7.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch8.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch8.pth
376903250	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/checkpoint-epoch9.pth
125657438	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E7_large_ds_generic_effective_r32_20k_full96_r1/weights-epoch9.pth
```

### `runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch10.pth` — `344,755,078` bytes (`344.76 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch10.pth` — `114,934,754` bytes (`114.93 MB`)

Delete `18` files / `4,137,144,840` bytes (`4.14 GB`):

```text
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch1.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch1.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch2.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch2.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch3.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch3.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch4.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch4.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch5.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch5.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch6.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch6.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch7.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch7.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch8.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch8.pth
344749410	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/checkpoint-epoch9.pth
114933350	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E8_large_ds_generic_ca_r32_20k_full96_r1/weights-epoch9.pth
```

### `runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1`

Retain:

- full: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch10.pth` — `225,193,126` bytes (`225.19 MB`)
- weights-only: `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch10.pth` — `75,064,618` bytes (`75.06 MB`)

Delete `18` files / `2,702,274,948` bytes (`2.70 GB`):

```text
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch1.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch1.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch2.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch2.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch3.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch3.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch4.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch4.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch5.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch5.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch6.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch6.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch7.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch7.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch8.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch8.pth
225189138	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/checkpoint-epoch9.pth
75063634	/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template/saved/E9_large_ds_shared_saout_r32_20k_full96_r1/weights-epoch9.pth
```

## Safety conditions before execution

- Re-run MLS status and regenerate this inventory immediately before deletion; abort if any excluded/live scope changed.
- Resolve each retained path and every deletion candidate under `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/`; reject symlinks or paths outside that root.
- Recompute the canonical deletion-list SHA-256 and require an exact match with the value above.
- Delete only the exact listed files—never use a broad recursive glob.
- Verify the retained checkpoint files remain readable in every run directory after cleanup, then report actual freed blocks with `df`.
