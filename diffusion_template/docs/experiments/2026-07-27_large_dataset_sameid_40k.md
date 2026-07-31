# Large Dataset same-ID RHCA 40k run

Date: 27 July 2026

## Objective

Train the exact eligible model configuration used by the recent Cosmic Large
dataset-policy matrix on the adjusted identity-aware Large Dataset. This
changes the dataset contract, not the model architecture.

## Fixed model

- config parent: `cosmic_large_initial_usage_rhca`
- SA-only branched attention enabled
- branched cross-attention disabled
- `branched_attn_weight_mode=noise_and_ref`
- rank 32, learning rate `1e-4`
- `masked_loss_step=1`
- `pipeline.pose_adapt_ratio=0`
- `pipeline.ca_mixing_for_face=false`

## Dataset

- manifest: `filtered_ids3_adj.json`
- manifest SHA-256:
  `0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50`
- 47,500 1024px images
- 2,561 explicit identities
- every target samples a different reference image from the same identity
- target/reference path equality fails closed

## Schedule

- machine: Neb, one GPU
- run: `rhca_large_dataset_sameid_40k_full96_r4`
- 40,000 optimizer steps
- batch size 2
- 500-step epochs; checkpoint and validation gates every four epochs
- validation at step 0 and every 2,000 steps through step 40,000
- fixed `manual_val` full-96 panel
- face-quality metrics enabled for every validation gate
- Comet project: `jul-comet-large-testing-tr`

## Reproducibility records

- config: `src/configs/large_dataset_rhca_40k.yaml`
- dataset loader: `src/datasets/large_dataset.py`
- launcher: `launchers/neb/start_rhca_large_dataset_sameid_40k.sh`
- immutable plan/runtime record:
  `experiments/large_dataset/rhca_large_dataset_sameid_40k_full96.json`

The first startup, `rhca_large_dataset_sameid_40k_full96`, was stopped before
any optimizer step because Neb's ONNX Runtime CUDA provider could not resolve
`libcudnn_adv.so.9`. Its immutable Comet key is
`14ff135be57345f8a814aa5e80e2ba8a`. The replacement launcher adds the same
NVIDIA-library path and fail-closed provider gate used by the recent Neb
Cosmic Large baseline.

The second startup, `rhca_large_dataset_sameid_40k_full96_r2`, also reached
zero optimizer steps. It exposed that Neb had the new
`validation_interval_steps` config field but not its audited trainer
implementation. Its immutable Comet key is
`61dbc17a78d549df8103c5fc618994de`. The audited 27 July full-96/2k validation
runtime patch is synchronized before `r3`. Live PID, replacement Comet key,
and startup verification are added after launch.

The third startup, `rhca_large_dataset_sameid_40k_full96_r3`, generated and
logged all 96 step-0 images, but stopped before its first optimizer step
because the standalone PyIQA environment had CPU-only PyTorch and rejected
the requested CUDA device. Its immutable Comet key is
`47e5c507c4f34ffbbfae08b38c4382c9`. For `r4`, the existing PyIQA 0.1.15
packages are used as an overlay on the CUDA-enabled `photomaker_NS`
interpreter; a live GPU MUSIQ smoke test passed without changing either
environment.

## Final run status

- status: stopped by user request on 28 July 2026
- run: `rhca_large_dataset_sameid_40k_full96_r4`
- terminated Neb launcher PGID: `963959`
- terminated training/metric PGID: `964138`
- Comet:
  [`a99db1fb953d4511827672380e6c1645`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/a99db1fb953d4511827672380e6c1645)
- startup verification: 64/64 dataset pairs decoded, zero rsync dry-run
  differences, ONNX CUDA provider loaded, 840/840 processor tensors in the
  optimizer, 96/96 step-0 images generated, face-quality metrics completed on
  all 96 inputs, and optimizer steps observed at about 1.35 seconds/step
- log: `logs/rhca_large_dataset_sameid_40k_full96_r4.log`

SIGTERM removed the launcher, trainer, metric subprocesses, and GPU process;
no force kill was required. This historical run retains its original
500-step epoch/checkpoint semantics. New run defaults use 2,000-step epochs.
