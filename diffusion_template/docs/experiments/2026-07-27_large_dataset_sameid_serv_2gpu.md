# Large Dataset same-ID 40k Serv two-GPU mirror

Date: 27 July 2026

## Objective

Mirror the live Neb experiment
`rhca_large_dataset_sameid_40k_full96_r4` on a two-GPU Serv instance. The
model, dataset policy, optimizer schedule, per-GPU batch size, per-GPU
optimizer-step budget, validation panel, validation cadence, and metrics stay
fixed. The controlled differences are the machine, world size, global batch
size, and run identity.

## Fixed experiment contract

- eligible SA-only branched attention;
- branched cross-attention disabled;
- `branched_attn_weight_mode=noise_and_ref`;
- `pipeline.pose_adapt_ratio=0`;
- `pipeline.ca_mixing_for_face=false`;
- rank 32 and learning rate `1e-4`;
- 47,500-image adjusted Large Dataset with 2,561 explicit identities;
- uniformly sampled distinct same-ID reference for every target;
- batch size 2 on each GPU, hence global batch size 4;
- 40,000 synchronized optimizer updates on each rank;
- 500-step epochs and 80 epochs;
- checkpoint plus fixed full-96 validation at step 0 and every 2,000 steps;
- seven default face-quality curves plus the API-only per-image CSV.

The run therefore exposes each of the two GPUs to the same 40,000-update
budget as the one-GPU Neb run. It intentionally doubles global batch size
because the requested local batch remains 2 on both ranks.

## Reproducibility artifacts

- source template:
  `serv_run_packages/_sources/start_large_dataset_sameid_40k_2gpu.sh`
- rendered launcher:
  `serv_run_packages/rhca_large_dataset_sameid_40k_full96_serv_r1/start_rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu.sh`
- MLS YAML:
  `serv_run_packages/rhca_large_dataset_sameid_40k_full96_serv_r1/run_rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu.yaml`
- immutable local experiment JSON:
  `experiments/large_dataset/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu.json`
- Serv job:
  `lm-mpi-job-3809c1e1-9749-4dd6-9ef9-7fcc0f84e3e4`
- Comet:
  [`db32f157e75a4798b2dfa530477c66d6`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/db32f157e75a4798b2dfa530477c66d6)

## Startup evidence

- project allocation before submission: 0 Running/Pending `#nasilaev` GPUs;
- MLS accepted the two-GPU request and moved it from Pending to Running;
- exact manifest SHA-256 verified;
- 64/64 deterministic target samples decoded;
- ONNX Runtime 1.20.1 loaded `CUDAExecutionProvider`;
- PyIQA 0.1.15 loaded from the existing Serv overlay;
- historical architecture and audited validation runtime hashes passed;
- immutable Comet record was written before model startup;
- all 840/840 branched-processor tensors are in the optimizer;
- DDP synchronized rank 0 and rank 1 at epoch 1;
- fixed full-96 step-0 validation began successfully.

The benign historical catch-all message about an `AttnProcessor2_0` lacking a
`parameters` attribute remains visible, but the explicit 840/840 optimizer
gate passed.

## 28 July recovery

Observed evidence: the original job completed all 96 step-0 images and
face-quality scoring, but then remained on its first training batch with zero
optimizer updates and no checkpoint. The six-hour DDP timeout had not expired.
The stall began at the first training-loader/model boundary after rank-0-only
validation, so the original job was stopped:
`lm-mpi-job-3809c1e1-9749-4dd6-9ef9-7fcc0f84e3e4`.

The replacement reuses the completed step-0 artifacts and the same immutable
Comet key. It explicitly skips only the already-completed initial validation,
uses `num_workers=0` to avoid creating loader workers after rank-0 CUDA
validation, and enables DDP unused-parameter discovery for conditional BA
paths. Model weights, architecture, dataset/reference policy, batch size,
optimizer schedule, 40k step budget, and future full-96 validation cadence
are unchanged.

- recovery job:
  `lm-mpi-job-a8852c27-6027-4e4e-8b9c-515261ef687c`
- recovery package:
  `serv_run_packages/rhca_large_dataset_sameid_40k_full96_serv_r1_recover`
- recovery record:
  `experiments/large_dataset/rhca_large_dataset_sameid_40k_full96_serv_r1_recover_2gpu.json`

Verification passed: both ranks synchronized at epoch 1, all 840/840 processor
tensors remained in the optimizer, and the first gathered optimizer step
completed with reduced loss `0.057932`.

## Step-2000 incident and repaired continuation

The recovery job trained cleanly through four 500-step epochs and completed
all 96 step-2000 validation images and face-quality metrics. It then failed at
the checkpoint/epoch-5 transition. Rank 0 was logging and serializing
`checkpoint-epoch4.pth` while rank 1 entered the next backward, so NCCL
sequence 32048 saw incompatible collectives (rank 0: one-element all-reduce;
rank 1: 13,066,240-element gradient all-reduce). The six-hour watchdog aborted
the job. The checkpoint write was interrupted and its ZIP central directory
was missing, so it cannot be loaded.

The repair keeps every rank behind rank 0 before and after main-only epoch
logging/checkpoint work. Checkpoints now write to a same-directory temporary
file and use atomic replacement, so an interrupted write cannot masquerade as
a valid endpoint.

Because no valid optimizer checkpoint survived, the active recovery first
replays 0→2k with the original Comet initialization path but suppresses all
replay metrics/images/assets. Existing 0/2k full-96 validation remains
untouched. Fresh MLS containers also initialize model replicas serially:
rank 0 populates the model cache before rank 1 constructs the identical model,
eliminating the intermittent concurrent 891 MB artifact-cache race observed
in discarded zero-state attempts.

- failed step-2k job:
  `lm-mpi-job-a8852c27-6027-4e4e-8b9c-515261ef687c`
- then-active replay/continuation job:
  `lm-mpi-job-33f3c842-7d1c-4a39-bdec-d6acb69c7f23`
- recovery launcher:
  `serv_run_packages/_sources/replay2k_continue_large_dataset_sameid_40k_2gpu.sh`
- recovery record:
  `experiments/large_dataset/rhca_large_dataset_sameid_40k_full96_serv_r1_replay2k_continue_2gpu.json`

Startup verification passed: both serialized model replicas became ready,
both DDP ranks synchronized at epoch 1, 840/840 processor tensors remained in
the optimizer, and the first replay loss exactly matched the original
`0.057932`. After the atomic epoch-4 checkpoint passes a full
model/optimizer/scheduler load check, the same script automatically resumes
epoch 5 in immutable Comet experiment
[`db32f157e75a4798b2dfa530477c66d6`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/db32f157e75a4798b2dfa530477c66d6).

## Step-4000 Comet boundary stall and deferred-validation recovery

The replay/continuation job successfully reconstructed step 2,000, resumed
from its intact optimizer state, trained through step 4,000, generated all 96
validation images, calculated the complete face-quality panel, and atomically
wrote `checkpoint-epoch8.pth` and `weights-epoch8.pth`. The full checkpoint is
653,129,930 bytes, loads as epoch 8, and contains 2,240 optimizer states.

The next apparent DDP stall was not the previous checkpoint race. Rank 1
created its epoch-9 iterator (`0/500`) after the post-checkpoint barrier, while
rank 0 never created that iterator. The intervening rank-0-only operation is
the Comet writer `set_step`/scalar boundary after the 96-image asset stream.
There was no NCCL error and no new forward/backward had begun. The observed
root cause is therefore the rank-0 logging path blocking after full-96 Comet
asset logging, leaving rank 1 waiting at the next training boundary.

The replacement preserves the optimizer trajectory and removes that
rank-asymmetric operation from live DDP:

- two-GPU training resumes continuously from epoch 8/step 4,000 to step
  40,000 with online scalar logging and atomic checkpoints every 2,000 steps;
- full-96 generation and the seven face-quality metrics are deferred until
  training finishes, then each 6k–40k checkpoint is evaluated in a fresh
  single-process invocation;
- every scalar, image, metric, and per-image CSV continues to use immutable
  Comet key `db32f157e75a4798b2dfa530477c66d6`;
- model, dataset, reference selection, batch size, scheduler, step numbers,
  prompts, seeds, bboxes, and validation definitions are unchanged.

The active recovery job is
`lm-mpi-job-79007b8b-a9f0-41db-a15a-802ffea65658`. It passed the intact
step-4k checkpoint/output gate, loaded both model replicas, synchronized both
ranks at epoch 9, retained 840/840 processor tensors in the optimizer, and
completed its first new optimizer update with reduced loss `0.043631`.
Durable record:
`experiments/large_dataset/rhca_large_dataset_sameid_40k_full96_serv_r1_continue4k_deferred_val_2gpu.json`.

## Live validation restoration (28 July 2026)

The apparent Comet logging gap was not a general telemetry failure. The
running continuation was still logging training loss, learning rate, gradient
norms, throughput, and epoch through step 13,650 and beyond. Only validation
metrics and images stopped at step 4,000 because the recovery intentionally
set `validation_interval_steps=0` and deferred full-96 validation until after
training, avoiding the rank-0 Comet asset boundary that had stalled DDP.

Without stopping or changing the two-GPU trainer, two one-GPU validation-only
sidecars now append the fixed full-96 protocol to the same immutable Comet
experiment. Arm 0 handles steps 6k, 10k, ..., 38k; arm 1 handles 8k, 12k,
..., 40k. Each successful event also publishes its local completion artifacts
to the main run directory, so the original post-training validation loop skips
already completed steps rather than duplicating Comet assets.

- training job (unchanged):
  `lm-mpi-job-79007b8b-a9f0-41db-a15a-802ffea65658`
- live validation arm 0:
  `lm-mpi-job-2e42c27d-d4b0-4524-b728-2758be257aea`
- live validation arm 1:
  `lm-mpi-job-e2a7254f-1754-43d4-861a-fee26db1eabe`
- durable sidecar record:
  `experiments/large_dataset/rhca_large_dataset_sameid_40k_full96_serv_r1_live_validation_sidecars.json`

The first sidecar pair
(`lm-mpi-job-8cdff1e8-4c4e-4b55-bd32-1f9cd72abd5f` and
`lm-mpi-job-40b13177-d8ed-4910-be4f-7a4667651e23`) failed in a read-only
preflight because an empty output directory did not yet exist. They did not
load a model or write to Comet. The corrected pair creates the isolated output
directories before checking for partial state.
