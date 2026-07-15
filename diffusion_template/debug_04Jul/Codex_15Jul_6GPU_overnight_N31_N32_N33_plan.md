# Six-GPU overnight allocation: N31, N32, and N33

Date: 15 July 2026

## Decision

Use **one four-GPU run plus two single-GPU runs**, not six independent single-GPU
runs.

| Machine | Run | Purpose | Local/global train batch | Work tonight |
|---|---|---|---:|---:|
| 4 GPU | N31 | Make face improvement causally depend on BA identity memory | 2 / 8 | 10k optimizer steps, 80k samples |
| 2 GPU, GPU 0 | N32 | Test richer hard-face patch-token memory | 2 / 2 | 10k optimizer steps, 20k samples |
| 2 GPU, GPU 1 | N33 | Unchanged N29 duration control | 2 / 2 | resume 10k and finish at 40k, 60k additional samples |

This allocates 160k sample exposures, the same order as six 10k single-GPU runs,
but concentrates them on two unresolved architecture questions and one clean duration
control. N31 also performs an additional wrong-memory UNet pass on BA-active steps, so
it does more useful identity-attribution work than a plain large-batch N29 run.

## Why not six normal runs

There are not six equally strong, independent hypotheses after N29/N30. Six variants
would mostly tune token counts, objective weights, or gates before establishing whether
BA is causally carrying identity. That creates multiple-comparison noise and weakens the
interpretation of any isolated visual win.

Four GPUs are specifically useful for N31:

- each sample can select a confidently different BA memory from seven other samples,
  rather than the one alternative available in a single-GPU batch of two;
- each rank samples its own diffusion timestep, so one DDP update averages four schedule
  locations and reduces the unusually high timestep variance in this training scheme;
- global batch 8 gives a stable counterfactual ranking update without increasing the
  per-GPU batch, which matters because N31 retains both correct- and wrong-memory graphs.

A four-GPU plain N29 run would have no equivalent architectural reason. It would mainly
confound N29 with four times as many samples per nominal step. N32 is kept single-GPU
because its new representation needs a direct N29-scale comparison, while N31 is the
experiment that benefits directly from cross-rank candidates.

The learning rate remains `1e-4`; it is deliberately not linearly scaled with global
batch. The goal is reduced gradient variance and better negatives, not a different update
magnitude.

## N31: counterfactual BA identity dependence

N31 retains the complete N29 generation contract:

- frozen PhotoMaker and frozen PhotoMaker ID encoder;
- full-reference, two-token QFormer BA memory;
- target-face cross-attention residual only;
- hard generated-face bbox and hard PhotoMaker epsilon preservation;
- no spatial reference UNet, pose adaptation, or CAMIX.

On every forward pass, ranks exchange the frozen identity-memory tensors. For each local
sample, N31 chooses the lowest-cosine non-self memory as a confidently different identity.
On BA-active timesteps it evaluates:

```text
correct = PM(correct reference) + BA(correct memory)
wrong   = PM(correct reference) + BA(wrong memory)

L_depend = max(0, margin + face_MSE(correct) - face_MSE(wrong))
```

The PhotoMaker prompt/reference remains correct in both passes. Only BA memory changes,
and the loss is measured inside the existing hard target bbox. The PhotoMaker prediction
is computed once and reused, so the extra cost is one BA UNet pass rather than another PM
plus BA pair.

Opt-in switches in
[`lora2.py`](../src/model/photomaker_branched/lora2.py):

```yaml
model:
  ba_identity_dependence_mode: paired_wrong_reference  # default: none
  ba_identity_dependence_weight: 0.25
  ba_identity_dependence_margin: 0.02
  ba_identity_dependence_global_negatives: true
```

The configuration is
[`one_id_ba_identity_dependence_N31.yaml`](../src/configs/one_id_ba_identity_dependence_N31.yaml).

Why 10k rather than 20k: global batch 8 already gives N31 four times N29's sample exposure
at a fixed step count, and the second BA graph makes it about 40-50% slower than N29 on
average. Ten thousand updates should be decisive for whether correct/wrong separation
develops. Extending a failed dependence objective to 20k would waste the most expensive
machine.

## N32: hard-face patch-token memory

N32 changes representation, not the safe generation topology. It replaces N29's two
frozen QFormer tokens with eight trainable identity-specific tokens:

1. Run the normal full reference through the frozen PhotoMaker CLIP vision encoder.
2. Map the original hard reference bbox through CLIP's shorter-edge resize and center crop.
3. Retain only CLIP patch tokens whose centers fall in that bbox. This does not crop and
   rescale the image as N30 did.
4. Form eight queries from the frozen 512-D InsightFace embedding.
5. Use a small 256-D trainable cross-attention resampler over the selected patch tokens.
6. Send its eight 2048-D tokens only to N29's target-face CA residual.

The target output is still hard-masked, and no reference spatial latent is introduced.
This tests richer eyes/nose/mouth/face-shape evidence without reopening the N17/N24 pose
and geometry failure modes.

Opt-in switch:

```yaml
model:
  ba_identity_memory_mode: face_patch_resampler  # old modes remain available
  ba_identity_token_count: 8
  ba_identity_patch_padding: 0.0
  ba_identity_resampler_hidden_dim: 256
```

The resampler and bbox mapping are in
[`identity_memory.py`](../src/model/photomaker_branched/identity_memory.py). Its parameters
are included in optimizer groups and checkpoints, and the same module/mapping is attached
to inference pipelines in
[`br_pipeline_helpers.py`](../src/pipelines/br_pipeline_helpers.py). Thus N32 is not a
training-only feature. The configuration is
[`one_id_ba_facepatch_resampler_N32.yaml`](../src/configs/one_id_ba_facepatch_resampler_N32.yaml).

N31 and N32 are intentionally separate. Combining a new memory and a new objective now
would make a good or bad result impossible to attribute.

## N33: unchanged N29 to 40k

N33 resumes `ba_qformer_idtokens_N29/checkpoint-epoch5.pth` at step 10k and runs 15
more 2k epochs, ending at step 40k. It changes no architecture or loss. This answers
whether N29's mild 2k-to-10k visual evolution simply needed substantially more updates.

This is preferable to starting a fresh 40k run: 30k new steps are needed, and Comet step
numbering continues at 10k. The configuration is
[`one_id_ba_qformer_continue20k_N33.yaml`](../src/configs/one_id_ba_qformer_continue20k_N33.yaml).

Do not interpret more face change alone as success. N33 passes only if additional changes
move toward the reference across identities, rather than mainly adding age, beard, texture,
or stronger expression.

## Multi-GPU correctness changes

The following changes are global but preserve single-GPU behavior:

- checkpoint and weights-only writes are now main-rank-only in
  [`base_trainer.py`](../src/trainer/base_trainer.py), preventing concurrent writes after
  an interrupt or any future all-rank save call;
- a `val_smoke_test_limit=24` is treated as an aggregate distributed limit. On four ranks,
  each rank processes six smoke images, not 24;
- N31 validation uses local batch 3 on each rank, so 12 images are generated concurrently
  across the machine. Local batch 12 would retain the DDP training model and instantiate a
  validation model on every GPU, creating an unnecessary OOM risk;
- full validation remains all 96 samples. Accelerate shards the loader, and per-sample
  generators retain the dataset seed, so local batch 3 does not alter image seeds.

All new architecture behavior defaults off. Existing N29 and older configurations retain
their previous memory, loss, and inference paths.

## Launch commands

Activate the same environment and export the Comet key on both machines:

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
export COMET_API_KEY=YOUR_KEY
```

On the four-GPU machine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash serv_new_runs/start_ba_identity_dependence_4gpu_N31.sh
```

On the two-GPU machine, launch both commands:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash serv_new_runs/start_ba_facepatch_resampler_serv_N32.sh

CUDA_VISIBLE_DEVICES=1 \
N29_CHECKPOINT=/home/kolyangg/rsrch/diffusion_template/saved/ba_qformer_idtokens_N29/checkpoint-epoch5.pth \
bash serv_new_runs/start_ba_qformer_continue20k_serv_N33.sh
```

Each launcher creates its own timestamped file under `logs_new_runs/`, starts through
`nohup`, prints the detached PID, and returns to the shell. N33 fails early with a clear
message if the N29 checkpoint is not present; transfer that checkpoint to the two-GPU
machine or set `N29_CHECKPOINT` to its actual path.

Follow all runs:

```bash
tail -f logs_new_runs/ba_identity_dependence_4gpu_N31_*.log
tail -f logs_new_runs/ba_facepatch_resampler_N32_*.log
tail -f logs_new_runs/ba_qformer_continue40k_N33_*.log
```

Do not increase N31's local train batch above 2. It retains two trainable BA graphs per
active step; global batch 8 is already the intended experiment. If it OOMs, rerun with
`TRAIN_BATCH_SIZE=1`, yielding global batch 4, rather than changing the architecture.

## Validation schedule

- N31 and N32: 24-image smoke validation before training, then all 96 fixed-seed images at
  2k, 4k, 6k, 8k, and 10k.
- N33: all 96 fixed-seed images every 2k from 12k through 40k. There is no redundant
  step-10k smoke because that checkpoint was already evaluated.

Expected wall time from N29's approximately 7.2-hour 10k run is about 10-11 hours for N31,
roughly 7-8 hours for N32, and roughly 22-24 hours for N33's additional 30k. Its 15 full
validations and machine I/O can extend that estimate.

## What to check while running

N31:

- four ranks initialize and report global batch 8;
- `train/identity_dependence/wrong_minus_correct` moves positive and ideally exceeds the
  `0.02` margin; the ranking loss should not remain exactly at its initial margin;
- target-ID and face-delta norms remain finite;
- no non-face or alignment regressions appear at 2k/6k/10k.

N32:

- startup lists optimizer group `ba_identity_resampler_params`;
- `train/ba_norm/identity_resampler` is present and finite;
- 2k images preserve N29 geometry before judging identity change;
- checkpoint loading reproduces validation, proving resampler serialization works.

N33:

- log says `Resume training from epoch 6`;
- first new full validation is step 12k and the final one is step 40k;
- visual identity progress broadens across people rather than concentrating in facial hair,
  age, or expression intensity.

## Decision after the runs

Use aligned enlarged face crops and same-seed comparisons first; use aggregate ID metrics as
secondary evidence unless the difference is large.

1. If N31 separates correct from wrong memory and visibly improves identity without changing
   pose, it becomes the primary architecture. Test wrong/null-memory inference before a long run.
2. If N32 produces more reference-correct local changes while retaining N29 alignment, combine
   N32 memory with N31's objective in a short confirmation run.
3. If only N33 improves, use its 20k/30k/40k trajectory to decide whether 50k is justified;
   continue only if the visual gains are identity-specific and monotonic.
4. If N31 and N32 both fail while N33 plateaus, more hyperparameter variants are unlikely to fix
   the core problem. The next change should revisit how BA residuals are supervised or decoded.

No 50k run is started in this allocation. The purpose of tonight is to identify which topology,
if any, deserves that compute.

## Verification completed locally

- Python compilation of every modified module;
- Hydra composition for N31, N32, and N33;
- shell syntax checks for all three launchers;
- unit smoke checks for CLIP bbox-to-patch mapping, the eight-token resampler, ranking loss,
  old multi-reference QFormer behavior, and N32 training/inference identity-memory shape parity;
- a two-rank Gloo smoke test for cross-rank wrong-memory gathering and self-exclusion;
- `git diff --check`.

No full SDXL training step was run locally because that would require loading the complete base,
PhotoMaker, dataset, and validation stack on a GPU. The step-0 smoke validation in N31/N32 remains
the final server-side integration check before optimizer updates.
