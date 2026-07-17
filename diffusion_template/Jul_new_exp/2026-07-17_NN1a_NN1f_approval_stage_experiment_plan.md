# NN1a–NN1f approval-stage experiment plan

Date: 17 July 2026

Status: proposal only. This document and the architecture explorer may be
updated before approval, but model/trainer code, Hydra configs, and launch
scripts must not be created or changed until the experiment matrix is approved.

## Decision summary

Run six 10k-step experiments concurrently, one process and one GPU per run.
All six retain the original N3a full spatial branched-attention forward:

- one doubled `[target, reference]` U-Net call at active BA denoising steps;
- all 70 `BranchedAttnProcessor` self-attention sites;
- all 70 `BranchedCrossAttnProcessor` cross-attention sites;
- target-face Q attending masked reference-face spatial K/V;
- split target-generation/reference-face prompt conditioning;
- direct return of the target epsilon half;
- no compact identity memory, target-only residual, layer allowlist, separate
  PhotoMaker epsilon owner, or post-CFG residual.

The earlier NN1 draft assigned NN1b/c to broader stability and identity-loss
experiments. This plan supersedes that naming because the requested first
attribution is now:

- NN1b isolates audit issue 5, the train/inference timestep mismatch;
- NN1c isolates audit issue 6, zero-token attention sinks.

NN1d–f then test architecture-preserving improvements learned from the post-N3a
full-BA experiments.

## Common correctness package: audit issues 1–4

All six runs enable the same flag-gated correctness package. These controls
must not alter a valid N3a forward or its floating-point output.

1. **Strict processor installation**
   - installation exceptions are fatal;
   - assert exactly 70 `BranchedAttnProcessor` and 70
     `BranchedCrossAttnProcessor` instances;
   - log and assert trainable tensors by processor type and projection role
     before optimizer construction.
2. **Strict bbox validity**
   - reject missing, non-finite, inverted, empty-after-clamp, or empty-after-
     resize target/reference boxes;
   - never convert an invalid box to an all-ones mask;
   - skip the complete invalid microbatch before backward and count the reason;
   - do not advance the requested optimizer-step budget for a skipped batch.
3. **Strict reference identity validity**
   - no zero 512-D fallback when face recognition fails;
   - use the same invalid-microbatch path as bbox failures;
   - log target-bbox, reference-bbox, and reference-recognition rejection
     counters separately.
4. **Strict processor checkpoint restore**
   - save the expected processor-name and trainable-key manifests;
   - reject missing/unexpected selected processor names and trainable keys;
   - after every validation swap, assert that training reattaches the exact
     processor objects owned by the optimizer.

Proposed backward-compatible toggles:

```text
ba_correctness_guards: false                 # old behavior when false
ba_invalid_sample_policy: legacy             # legacy | error | skip_batch
ba_strict_processor_restore: false           # old strict=False route when false
```

Every NN1 launcher would explicitly set:

```text
ba_correctness_guards=true
ba_invalid_sample_policy=skip_batch
ba_strict_processor_restore=true
```

Because these are one-GPU runs, there is no cross-rank skip synchronization in
this matrix. The implementation should nevertheless use one whole-microbatch
decision so the same code remains safe if a future DDP run enables the guards.

## Common run protocol

| Item | Value |
|---|---|
| training allocation | one GPU and one Accelerate process per run |
| training batch | physical batch 2, effective batch 2, accumulation 1 |
| optimizer steps | 10,000 |
| validation/checkpoint cadence | every 2,000 optimizer steps |
| epochs | 5 × 2,000 optimizer steps |
| validation checkpoints | step 0, 2k, 4k, 6k, 8k, 10k |
| validation set | fixed full 96-image manual validation |
| inference schedule | text-only 0–9, PhotoMaker 10–14, spatial BA 15–49 |
| image size | 1024×1024 |
| seed/data order | identical across all six runs |
| N3a optimizer | LR `5e-5`, target/noise LR `1.25e-5`, clip `1.0`, weight decay `1e-2` |
| N3a objective | `masked_alternating` |
| N3a reference augmentation | crop margin 0.2–0.6, downscale jitter 0.5 |

Physical batch 2 deliberately matches N3a and avoids introducing the
microbatch/accumulation behavior as a seventh experiment. The future launchers
may retain N37-style environment overrides, but the default six-run comparison
must use batch 2 without accumulation.

The launcher format should follow
`Jul_new_exp/archived_post_n3a_examples/launchers/start_ba_identity_owner_hybrid_2gpu_N37.sh`:

- `set -euo pipefail`;
- detached-by-default execution and timestamped logs;
- `full_step0_val` positional argument;
- environment-overridable run, batch, validation, path, and port settings;
- an environment-only `COMET_API_KEY` requirement;
- explicit printed run summary;
- pass-through Hydra overrides.

## Six-run matrix

| Run | Isolated question | BA train-time schedule | Reference-half prompt | Trainable processors | Extra supervision |
|---|---|---|---|---|---|
| NN1a | Does guarded main_clean reproduce N3a? | BA on all sampled timesteps | legacy ID-only, 75 zero sinks | SA + CA, reference + target/noise clones | none |
| NN1b | Does fixing issue 5 reduce destructive drift? | sample only BA-active inference region | same as NN1a | same as NN1a | none |
| NN1c | Does fixing issue 6 strengthen useful ID conditioning? | same as NN1a | ID-only with explicit non-ID attention mask | same as NN1a | none |
| NN1d | Does N11’s frozen-CA lesson stabilize full BA? | same as NN1a | same as NN1a | SA reference + target/noise clones; CA active/frozen | none |
| NN1e | Can direct reference-ID supervision make NN1d identity-directed? | same as NN1a | same as NN1a | same as NN1d | decoded reference-ID loss 0.1 at `t≤400` |
| NN1f | Can selective reference K/V training preserve pose while learning identity? | same as NN1a | same as NN1a | SA `ref_to_k/v` only; CA active/frozen | same ID loss as NN1e |

## NN1a: guarded N3a replay

Purpose: establish that correctness fixes 1–4 do not change N3a on valid data.

Keep N3a’s:

- `noise_and_ref` LoRA clones in branched SA and CA;
- trainable reference and target/noise projections;
- `train_branched_ca_lora=true`;
- `train_ba_all_steps=true`;
- legacy `id_only` prompt implementation, including its zero-token sinks;
- `masked_alternating` loss;
- crop jitter and optimizer settings.

NN1a is not expected to be the winner. It is the parity reference for every
other run. Step-zero images must match N3a within deterministic tolerance, and
the run should reproduce N3a’s tendency to move too strongly unless a previous
silent correctness failure was materially affecting N3a.

## NN1b: schedule-matched BA training — audit issue 5

NN1b differs from NN1a only in training timestep exposure.

Do not implement this by merely taking the current
`train_ba_all_steps=false` branch. At text-only and PhotoMaker-only timesteps
that branch may bypass every trainable BA tensor, producing a no-gradient step
and making a 10k “optimizer-step” comparison ambiguous.

Instead add a flag-gated BA-active timestep sampler:

```text
ba_train_timestep_mode: all                  # all | inference_ba_region
```

For `inference_ba_region`, sample only timesteps satisfying the same current
ratio test as inference’s BA interval:

```text
denoise_progress >= branched_attn_start_step / num_inference_steps
```

With a 15/50 start and 1,000 training timesteps this is approximately
`t <= 699`. Every counted optimizer step therefore runs the unchanged doubled
BA forward, but BA is no longer trained in a denoising region where inference
uses text-only or ordinary PhotoMaker.

Expected useful signature: less rapid face/color/prop drift than NN1a without
making the branch visually inactive.

## NN1c: explicitly masked ID-only prompt — audit issue 6

NN1c differs from NN1a only in reference-half cross-attention masking.

Keep the same 77-token face-prompt tensor and the same two boosted ID-token
embeddings, but pass an additive attention mask to the reference-half
`BranchedCrossAttnProcessor` attention:

- conditional rows: ID-token positions are allowed and every non-ID position
  receives `-inf`;
- unconditional CFG rows: retain the existing plain negative-prompt context
  and do not apply the positive ID-token mask;
- target-half generation-prompt attention remains unchanged;
- use the identical behavior in training and validation.

Proposed toggle:

```text
model.ba_face_prompt_mode: id_only            # existing behavior
model.ba_face_prompt_attention_mask: false    # false = exact N3a
```

This tests whether probability mass currently lost to 75 zero K/V entries is
weakening the reference stream. It does not replace split cross-attention or
introduce compact identity tokens.

Expected useful signature: larger correct-vs-wrong reference sensitivity than
NN1a with comparable pose, color, and background.

## NN1d: full BA with active but frozen cross-attention

NN1d applies the strongest stable full-BA lesson from N11 while changing as
little as possible:

- both processor classes remain installed at all 70 sites;
- `BranchedCrossAttnProcessor` still performs target-generation/reference-face
  split conditioning;
- all cloned CA projections are frozen;
- branched SA remains `noise_and_ref` and trainable;
- keep NN1a’s loss, LR groups, crop jitter, all-timestep training, and legacy
  ID-only prompt.

This is one isolated trainability change, not a move to standard CA.

Expected useful signature: smaller chroma/background/prop drift and more
monotonic face movement than NN1a. A visually identical PhotoMaker trajectory
would mean SA training alone is too weak under the N3a objective.

## NN1e: NN1d plus low-noise reference identity supervision

NN1e differs from NN1d only by a direct identity objective:

- reconstruct predicted `x0` from the BA epsilon prediction;
- decode only when the shared sampled timestep is `t≤400`;
- crop the generated target face with the validated target bbox;
- compare its frozen differentiable FaceNet embedding with the validated
  reference-face embedding;
- use cosine-distance weight `0.1`;
- retain N3a’s diffusion objective unchanged.

Proposed toggles:

```text
model.use_id_loss: false
model.id_loss_weight: 0.1
model.id_loss_max_timestep: 400
model.id_loss_identity_source: reference
```

Only the minimal identity-loss implementation should be backported from
`main_clean_exp`; no compact-memory, residual-composition, layer-allowlist, or
causal-epsilon code may come with it.

Expected useful signature: NN1d-level stability with a consistent correct-
reference identity gain, especially in hard poses. Stop if the metric rises
through desaturation, expression collapse, or texture artifacts.

## NN1f: selective reference-K/V identity training

NN1f is the brave architecture-preserving option. It keeps NN1e’s forward and
identity loss but narrows trainability to the projections that directly supply
identity evidence to target-face self-attention:

- train `BranchedAttnProcessor.ref_to_k` and `ref_to_v` LoRA tensors at all 70
  SA sites;
- freeze target/noise Q/K/V, protecting target query geometry and background;
- freeze reference Q because it continues the reference stream but does not
  directly form target-face identity K/V;
- keep every branched SA processor active;
- keep every branched CA processor active with all CA weights frozen;
- do not change the hard target/reference masks or hidden-state composition.

Proposed toggle:

```text
model.ba_sa_train_mode: all                   # all | ref_kv_only
```

This is still the original branched-attention mechanism: target face Q attends
the full spatial reference-face K/V at every SA site. The change is optimizer
ownership, not a residual substitute.

Expected useful signature: less pose/scene drift than NN1e with comparable or
better reference identity. Failure mode: insufficient adaptation on extreme
pose differences, resembling the weak co-adaptation seen in reference-only
historical runs.

## Hardware and future launcher allocation

| Machine | Physical GPU | Run | Proposed launcher |
|---|---:|---|---|
| 2-GPU | 0 | NN1a | `jul_serv_runs/start_ba_NN1a_n3a_replay_1gpu.sh` |
| 2-GPU | 1 | NN1b | `jul_serv_runs/start_ba_NN1b_schedule_matched_1gpu.sh` |
| 4-GPU | 0 | NN1c | `jul_serv_runs/start_ba_NN1c_masked_id_prompt_1gpu.sh` |
| 4-GPU | 1 | NN1d | `jul_serv_runs/start_ba_NN1d_frozen_ca_1gpu.sh` |
| 4-GPU | 2 | NN1e | `jul_serv_runs/start_ba_NN1e_frozen_ca_id_loss_1gpu.sh` |
| 4-GPU | 3 | NN1f | `jul_serv_runs/start_ba_NN1f_ref_kv_id_loss_1gpu.sh` |

Each future script should default to its listed physical GPU but retain a
`CUDA_VISIBLE_DEVICES` override. Unique default master ports should be used
even though each run has one process.

## Mandatory startup and validation evidence

Before step zero:

- print processor class counts and exact processor names;
- print trainable tensor/parameter counts for SA target/noise Q/K/V, SA
  reference Q/K/V, CA target/noise Q/K/V, and CA reference Q/K/V;
- assert the counts implied by each run;
- report dataset rejection counters;
- hash or numerically fingerprint the step-zero target epsilon for NN1a parity.

At step 0, 2k, 4k, 6k, 8k, and 10k:

- full fixed 96-image validation;
- same-seed PhotoMaker comparison;
- BA enabled/disabled comparison;
- correct, wrong, and null reference canaries;
- face and background MAE versus PhotoMaker;
- identity similarity with face-detection success count;
- enlarged face sheets for hair, glasses, hats, hands, props, extreme pose, and
  non-frontal faces;
- strict checkpoint-copy and processor-object identity assertions.

## Interpretation order

1. NN1a must pass processor, mask, identity, checkpoint, and step-zero parity
   checks before trained comparisons are trusted.
2. Compare NN1b only with NN1a to attribute the schedule change.
3. Compare NN1c only with NN1a to attribute token masking.
4. Compare NN1d only with NN1a to attribute frozen CA updates.
5. Compare NN1e only with NN1d to attribute direct reference-ID supervision.
6. Compare NN1f only with NN1e to attribute reference-K/V-only trainability.

No model/trainer code, Hydra config, `jul_serv_runs` directory, or launcher is
authorized by this proposal. Create them only after explicit approval.
