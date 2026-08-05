# Large Dataset r4 recovery, Serv two-GPU comparison, and E0 base

**Date:** 4 August 2026

**Status:** historical recovery complete; both E0 controls running on Serv

**Historical base:** `rhca_large_dataset_sameid_40k_full96_r4`
(`a99db1fb953d4511827672380e6c1645`)

**Historical Serv comparison:**
`rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu`
(`db32f157e75a4798b2dfa530477c66d6`)

## Decision

The two historical runs did **not** use different branched-attention model
code or different Hydra model settings. Their immutable Comet source assets
and recovered configuration are identical. The Serv result is not an exact
replication because it changed the optimization and data-exposure contract to
two ranks/global batch four, then required a replay and multiple resumptions
with different worker/RNG behavior. Later validation also used a one-record
bbox-policy difference. These effects, rather than a hidden BA architecture
change, are the credible explanation for its lower identity trajectory.

The observed `r4` run also contained a useful accidental signal: its failed
`train_ba_only` installation allowed generic rank-32 and PhotoMaker-default
rank-64 adapters to train alongside BA. That broad capacity improved identity
but hurt prompt and face-quality metrics and destroyed clean BA attribution.
Two matched E0 controls now isolate the full effect:

- `E0_large_ds_base_historical_r4_20k_full96_r1` deliberately reproduces the
  observed 171.29M fail-open ownership and legacy checkpoint behavior;
- `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1` preserves the intended
  hard-BA route but repairs ownership to the exact 31.95M BA state and uses
  complete schema-v2 checkpoints.

E2 remains the narrower controlled test of *which useful part* may explain a
positive historical-versus-fixed E0 gap: it adds output capacity only inside
the reference-face branch. E1-E6 are not changed by this work.

Both E0 controls use 20k steps so they match each other and E1-E6. Twenty
thousand, rather than 40k, is required for direct suite comparison. The
historical arm must be uninterrupted because its deliberately incomplete
legacy checkpoint is not resume-safe; the fixed arm has complete 2k
checkpoints.

## Recovery method and confidence

Neb was unavailable and was not contacted. Recovery used Git history,
immutable experiment JSONs, existing Serv recovery packages/checkpoints, and
read-only Comet API metadata/source assets.

| Evidence | Recovered result |
|---|---|
| Comet Git metadata | Both experiments recorded branch `test` and HEAD `c04970f342a186d1092f07f9a08d7d8a797383e8`. The commit alone is insufficient because the experiment files were then uncommitted machine-local work. |
| Preserved Neb snapshot | Commit `f0bf95b55eec0123f3aa290ffed4e1e4deb540b1` contains the r4 config, record, dataset loader, Neb launcher, and audited runtime snapshot. |
| Preserved Serv snapshot | Commit `547e93c6811e9cddb208464b486421fcd0cfa54a` contains the original two-GPU package plus its recovery, replay, continuation, and validation-sidecar packages. |
| Resolved Hydra config | `src/configs/large_dataset_rhca_40k.yaml` has SHA-256 `718b4fd365411122bd3e2ca236c95b33aa1399b8571b69fd27767a80a916bb15`; this is the exact hash stored in the Serv record and the preserved r4 snapshot. |
| Comet source asset `train.py` | Both runs logged SHA-256 `1ced08c517ab1dee4db0476d4cddd8a22294004170d7b38ab5e1112fec8da75f`, matching the preserved r4 snapshot. |
| Comet source asset `src/logger/cometml.py` | Both runs logged SHA-256 `179411e747f503d67cc4825a71b41e240cd2d007619944838e94232ed31bd161`, matching the preserved r4 snapshot. |
| Architecture lock | The historical launcher locks the core Apr implementation to commit `aede146e2e2a2dae1cb3d14a0ea5daed25ae9604` and separately hashes the validation/runtime patches. |

The scientific code/config recovery is therefore high confidence. The exact
Neb shell at process start is lower confidence: the later preserved active
launcher contains `TRAIN_EPOCHS=20`, while the immutable plan, resolved Comet
parameters, checkpoints, and the observed curve through 34k prove that r4
actually used the documented 80 × 500-step/40k endpoint. The resolved run
record is authoritative; the stale post-run shell value must not be replayed.

## Recovered historical contracts

### r4 on Neb

- one process and one GPU;
- local/global batch size 2;
- 40,000 planned optimizer updates, 500-step trainer epochs, 80 epochs;
- checkpoint and full-96 validation every 2,000 updates;
- two training data workers;
- adjusted Large Dataset with uniform distinct same-ID reference selection;
- rank-32 `noise_and_ref` branched self-attention at all 70 SA sites;
- branched CA disabled;
- `pipeline.pose_adapt_ratio=0.0` and
  `pipeline.ca_mixing_for_face=false`;
- LR `1e-4`, masked loss from step 1, seed 0;
- historical `legacy_full_copy` RealVisXL validation behavior.

The recovered entry chain is:

```text
launchers/neb/start_rhca_large_dataset_sameid_40k.sh
  -> launchers/active/run_rhca_large_dataset_40k_1gpu.sh
  -> launchers/active/run_rhca_apr2026_one_id_1gpu.sh
  -> train.py --config-name=large_dataset_rhca_40k
```

### Serv two-GPU run

The original package selected the same config and active launcher but set two
visible GPUs and `ACCELERATE_NUM_PROCESSES=2`. Local batch size remained two,
so DDP used global batch four. It also retained LR `1e-4` and 40,000 optimizer
updates.

The effective run was assembled from several processes:

1. the original job completed step-zero validation, stalled before its first
   optimizer update, and produced no checkpoint;
2. recovery restarted from step zero with initial validation skipped,
   `dataloaders.train.num_workers=0`, and
   `ddp_find_unused_parameters=true`; it reached 2k but produced a truncated
   checkpoint during a rank/checkpoint race;
3. replay rebuilt 0→2k from scratch, suppressed duplicate Comet events, enabled
   serialized distributed model initialization, then trained 2k→4k;
4. a new continuation resumed the intact 4k checkpoint with in-process
   validation disabled;
5. independent one-GPU sidecars evaluated later 2k checkpoints into the same
   Comet experiment.

`ddp_find_unused_parameters` and serialized initialization are engineering
controls and are not, by themselves, changes to the BA equation. Zero workers,
replay/resume, DDP sampling, and global batch do change which target/reference
pairs contribute to each optimizer update.

## Effective differences that matter

| Contract | r4 | Serv `_r1_2gpu` | Consequence |
|---|---|---|---|
| World/global batch | 1 rank / 2 | 2 ranks / 4 | Each update averages twice as many examples at the same LR and update count. This changes gradient noise, data exposure, and the number of examples processed. |
| Target stream | Single-process sampler | `DistributedSampler` partitions targets by rank | The target order is not the one-GPU stream. |
| Reference/flip RNG | Two worker-local Python RNG streams | Recovery and continuation use zero workers on two ranks | `large_dataset.py` calls `random.choice` for the same-ID reference and `random.random` for flips, so the effective pairs/augmentations differ. |
| Continuity | Uninterrupted through the recorded 34k gate | fresh 0→2k replay, resume at 2k, resume at 4k | Checkpoints save model/optimizer/scheduler/config but not Python, NumPy, Torch, CUDA, worker, or sampler RNG state. `train.py` reseeds each new process, so replay/resume cannot continue the exact r4 stream. |
| Validation execution | In-process one-GPU historical path | step 0/2k/4k in the training history, then one-GPU sidecars | Model checkpoints are comparable, but the execution history is not identical. |
| Later bbox input | Historical relative `pm96_bboxes_new.json` | sidecars pin `protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json` | The files differ for one record: `force_manual=false` versus `true`; later absolute metric comparisons are not byte-identical. |
| Initialization metric | ID `.30633` at step 0 | ID `.30187` at step 0 | A `.00446` gap existed before training, proving remaining machine/runtime or validation drift. It is too small to explain the later gap alone. |

Trainer epoch length is not a target-order difference here. The trainer wraps
the DataLoader in one persistent `inf_loop`; changing bookkeeping epochs from
500 to 2,000 does not recreate the loader every epoch. Both E0 controls
therefore use the repository-standard 2,000-step epoch so validation and
checkpoints coincide, without changing the continuous one-GPU sample stream.

## Identity evidence

All values below are the existing fixed-96 aggregate `manual_val/id_sim`.

| Step | r4 | Serv `_r1_2gpu` | r4 minus Serv |
|---:|---:|---:|---:|
| 0 | .3063 | .3019 | +.0045 |
| 2k | .2983 | .3078 | -.0094 |
| 4k | .3443 | .2556 | +.0887 |
| 8k | .3646 | .2663 | +.0983 |
| 12k | .3627 | .2863 | +.0763 |
| 16k | .3723 | .3039 | +.0684 |
| 20k | .3764 | .3129 | +.0634 |
| 24k | **.3904** | .3244 | +.0660 |
| 32k | .3871 | .3285 | +.0586 |
| 34k | .3797 | .3366 | +.0431 |

r4 peaks at `.39039` at 24k. The Serv run peaks at `.34509` at 26k and ends
at `.34192` at 40k. The step-zero gap explains only a small part of the
trained gap.

### Causal interpretation

Observed facts:

- model/config and logged core source assets match;
- global batch and rank count differ;
- the Serv job was replayed/resumed and used a different worker/sampler/RNG
  path;
- validation was later delegated to sidecars with a one-record bbox-policy
  difference;
- curves diverge sharply after the first recovery boundary.

Best-supported inference:

- the largest intentional optimization difference is global batch four at the
  same LR and optimizer-update budget; this was never a controlled one- versus
  two-GPU test;
- the replay/resume and changed random reference stream are a second strong
  confound, particularly because the first large gap appears at 4k;
- validation and step-zero drift add smaller measurement offsets.

It is not possible to assign a numerical share of the loss to any one factor.
The result should not be summarized as “two GPUs reduce ID similarity.” It
shows that changing world/global batch without retuning, while also changing
the sample trajectory and restart history, failed to reproduce r4.

## What r4 accidentally taught us

The historical `train_ba_only=true` setup failed open when the installation
loop called `.parameters()` on a plain `AttnProcessor2_0`; the broad exception
handler skipped the later freeze/allowlist stage. The observed r4 optimizer
therefore owned approximately 171.29M parameters rather than a clean 31.95M:

- intended BA Q/K/V state: 31.95M;
- unintended generic rank-32 U-Net adapter: 46.45M;
- unintended pretrained rank-64 PhotoMaker `default` adapter: 92.90M.

The generic adapter was saved, while the trained PhotoMaker `default` adapter
was not completely represented by the historical checkpoint schema. This is
an obvious correctness and attribution error and must not be reproduced.

The matched historical-versus-clean BA32 analysis nevertheless supplies a
useful direction. Through 14k, broad historical training gains roughly
`.026–.088` ID over clean BA32, while losing about `.50–.74` text score and
`.038–.067` TOPIQ-Face mean. The useful capacity signal is therefore real,
but broad generic adaptation has an unacceptable tradeoff.

E2 isolates the safest explanation: historical generic adaptation trained
shared `to_out`, while clean Q/K/V-only BA had to express the reference message
through a frozen output basis. E2 adds an exact-parity rank-32 output LoRA only
inside the reference-face hard branch. It keeps:

- target Q / explicit same-ID reference K/V BA as the core mechanism;
- no native/PhotoMaker face-output interpolation or residual mixer;
- branched CA disabled;
- both pose adaptation and CA face mixing disabled;
- generic and `default` adapters frozen;
- exact trainable ownership and complete schema-v2 checkpoints.

E2 thus gives a binary answer to “did the historical output-basis capacity
help?” without recreating the ownership bug. The historical E0 control now
measures the full broad-ownership effect because the user explicitly requested
that comparison; it is not a promotion proposal. If E2 is positive,
branch-local output capacity becomes part of the next base. If only historical
E0 is positive, the gain remains entangled with ordinary SA/CA and the
PhotoMaker-default adapter and needs a later controlled decomposition.

## Two E0 controls prepared for Serv

Both controls use one process on one A100, batch two, LR `1e-4`, 20k updates,
and fixed 96-image validation at step zero and every 2k. Both log to
`aug-large-ds`, including a variant-specific experiment comment and a 96-row
per-image ID table at every validation event. Hard target-Q/reference-KV BA
remains the face route; branched CA, `pose_adapt_ratio`, `ca_mixing_for_face`,
and PhotoMaker/native face-output interpolation are disabled in both.

| Control | Historical observed | Fixed BA-only |
|---|---|---|
| Run/Comet name | `E0_large_ds_base_historical_r4_20k_full96_r1` | `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1` |
| Hydra config | `src/configs/E0_large_ds_base_historical_20k.yaml` | `src/configs/E0_large_ds_base_fixed_20k.yaml` |
| Parent | `src/configs/large_dataset_rhca_historical_observed_20k.yaml` | `src/configs/large_dataset_rhca_hard_v1_audited_20k.yaml` |
| Experiment JSON | `experiments/large_dataset/E0_large_ds_base_historical_r4_20k_full96_r1.json` | `experiments/large_dataset/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1.json` |
| Serv entrypoint | `serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/start_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.sh` | `serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/start_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.sh` |
| MLS YAML | `serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/run_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.yaml` | `serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/run_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.yaml` |
| Startup ownership | exact 3,080 tensors / 171,294,720 parameters | exact 840 tensors / 31,948,800 parameters |
| Ownership detail | BA 31.95M + generic rank-32 46.45M + PhotoMaker-default rank-64 92.90M | BA Q/K/V only; generic/default adapters frozen |
| Checkpoint | historical legacy format; trained default adapter intentionally omitted | complete trainable schema-v2 |
| Resume policy | do not resume; restart the arm from step zero if interrupted | exact optimizer/checkpoint resume is supported |

The historical parent retains the original warning-and-continue installation
path, but adds an independent fail-closed ownership assertion. Startup must
match all three disjoint parameter groups and exact optimizer membership; if a
future bug fix changes the accidental behavior, the job stops rather than
silently becoming another fixed run. This is a historical control, not an
eligible promotion candidate.

The fixed control is an exact clean reconstruction of r4's **intended
scientific route**. Its fail-closed ownership and complete checkpoint repairs
are the same substrate inherited by E1-E6.

## E1-E6 base audit

No E1-E6 implementation was redone. All six inherit exactly one shared parent,
`large_dataset_rhca_hard_v1_audited_20k`, which is also fixed E0's parent. The
fail-closed composition gate compares every resolved leaf with that parent and
permits only the experiment comment plus the arm's declared scientific leaf:

- E1: true reference-key mask;
- E2: branch-local rank-32 output basis;
- E3: bbox-normalized reference ROI warp;
- E4: mid/up BA site selection;
- E5: inference-active timestep sampling;
- E6: FP32 BA trainable state.

The fixed gate also asserts hard reference replacement, SA on/CA off,
`pose_adapt_ratio=0`, `ca_mixing_for_face=false`, no validation PhotoMaker
LoRA adapter, one image for each of the fixed 96 items, and strict ownership.
Therefore the existing E1-E6 code is based on the correct clean intended-r4
base and each arm retains exactly one approved scientific delta.

## Serv submission record

The user explicitly authorized an eight-A100 exception for this E0 pair while
E1-E6 occupied the normal six-A100 ceiling. A read-only MLS check confirmed all
six existing jobs were `running` with one A100 each. Both E0 YAMLs were then
deployed by exact SHA-256, revalidated on Serv, and submitted once:

- historical job `lm-mpi-job-b7aed096-391a-4f54-b41b-6515ba895dc2`, Comet
  [`a5599bd06c9346978c1fca8b8087f634`](https://www.comet.com/nikolay-2104/aug-large-ds/a5599bd06c9346978c1fca8b8087f634);
- fixed job `lm-mpi-job-a0e91e1b-3e43-49c1-b65e-9f4992f33bc4`, Comet
  [`5b5cbd1584184ce1a9032dd6fafb91c5`](https://www.comet.com/nikolay-2104/aug-large-ds/5b5cbd1584184ce1a9032dd6fafb91c5).

Both jobs reached `running`, registered in `aug-large-ds`, and entered the
fixed step-zero 96-image validation. The historical runtime asserted exact
ownership `3,080/171,294,720`; the fixed runtime asserted
`840/31,948,800`; and both reported BA processor optimizer membership
`840/840`. The Comet API returned the intended `experiment_comment` for both
immutable keys.

The audited submission procedure was:

1. run `python3 local_scripts/serv_job.py check` locally and count this
   project's actual Running/Pending A100 requests;
2. verify the remote `test` checkout and SHA-256 of both E0 configs, the
   launcher, validator, `train.py`, experiment JSONs, and immutable packages; do not
   overwrite files being read by live jobs;
3. confirm both remote YAML paths exist and each requests exactly one
   A100/process;
4. submit each once with:

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/run_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.yaml \
  --comment "E0 historical r4 replay: exact observed 171.29M fail-open ownership versus fixed E0."

python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/run_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.yaml \
  --comment "E0 fixed r4 replay: strict 31.95M BA-only matched comparator."
```

The YAML's direct equivalent is:

```bash
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/run_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.yaml
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/run_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.yaml
```

5. record both MLS job names; inspect startup and verify historical ownership
   `3,080/171,294,720` with its three exact categories, and fixed ownership
   `840/31,948,800` plus optimizer membership `840/840`;
6. verify each `saved/<run_name>/comet_experiment.json`, copy both immutable
   keys into their experiment records, and check both Comet comments;
7. after step-zero validation completes, verify 96 images and the 96-row
   `id_sim__manual_val__step_000000.csv` table in each run;
8. if MLS rejects the request for an allocation/request-limit reason, do not
   retry unless the user asks.

The per-image table and experiment-comment API retrieval examples remain in
[`2026-08-03_large_dataset_hard_ba_six_arm_design.md`](2026-08-03_large_dataset_hard_ba_six_arm_design.md).
