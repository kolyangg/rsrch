# NN1: full branched-attention experiment family

Date: 17 July 2026

Status: architecture proposal only. No NN1 code, Hydra config, or launcher has
been created.

## Objective

Re-establish the original branched-attention mechanism as the non-negotiable
base, then test training changes that were successful after N3a without
replacing that mechanism with compact target residuals.

## Non-negotiable forward contract

Every NN1 option must retain:

1. a full VAE-encoded reference latent;
2. the same-timestep noised reference latent;
3. one doubled U-Net batch `[target, reference]`;
4. `BranchedAttnProcessor` at all 70 `attn1` sites;
5. `BranchedCrossAttnProcessor` at all 70 `attn2` sites;
6. target-background Q/K/V from the target stream;
7. target-face Q with reference-face spatial K/V;
8. target/reference split prompt conditioning in cross-attention;
9. the target half returned as the active BA prediction;
10. no compact identity residual, no independent PM epsilon owner, and no
    post-CFG additive residual.

“Frozen CA” means `BranchedCrossAttnProcessor` remains installed and active but
its cloned weights do not update. It does not mean replacing it with standard
cross-attention.

## Common two-GPU protocol

- two DDP processes;
- local batch 1 per GPU;
- effective global batch 2, matching N3a’s one-GPU local batch 2;
- identical seed and data ordering;
- 1024×1024 square training;
- full 96-image validation at step 0;
- short validation/checkpoints at 500, 1k, 2k, 3k, 4k, and 6k;
- same-seed PhotoMaker baseline;
- BA-on, BA-off, correct-reference, wrong-reference, and null-reference canary;
- stop immediately if processor counts or checkpoint-copy assertions fail.

Using local batch 2 on each GPU would double N3a’s effective batch and confound
the architecture replay. That can be tested later.

## NN1a: exact N3a DDP control

Purpose: prove that restored `main_clean` reproduces the runnable N3a behavior
before interpreting any improvement.

| Component | NN1a |
|---|---|
| topology | exact N3a full spatial BA |
| processor forward | 70 branched SA + 70 branched CA |
| trained weights | target/noise and reference clones in SA and CA |
| weight mode | `noise_and_ref`, LoRA clones |
| base LR | `5e-5` |
| target/noise LR | `1.25e-5` (`×0.25`) |
| loss | `masked_alternating` |
| reference crop | N3a jitter retained |
| schedule during training | BA at all sampled timesteps |
| purpose | baseline/reproduction, not expected winner |

Expected signature:

- step zero is clean and clearly BA-active;
- early training may reproduce N3a’s identity drop;
- if it does not reproduce, debug DDP/batch/checkpoint parity before NN1b/c.

NN1a should require no architecture implementation beyond approved correctness
assertions and a two-GPU launcher.

## NN1b: stable full-BA self-attention training

Purpose: apply the strongest architecture-preserving stability lesson from N11.

| Component | NN1b |
|---|---|
| topology | unchanged full spatial BA |
| processor forward | 70 branched SA + 70 branched CA |
| trained weights | branched SA clones only |
| cross-attention | branched and active, weights frozen |
| weight mode | `noise_and_ref` for SA |
| base LR | `1e-4` |
| target/noise LR | `1e-5` (`×0.1`) |
| weight decay / clipping | `1e-3` / `1.0` |
| loss | `blended_masked`, `lambda_face=0.15` |
| reference crop | fixed clean reference; no N3a crop jitter |
| schedule during training | keep all-steps behavior for parity with N11 |
| evidence | N11 rose cleanly and monotonically while trainable CA regressed |

This is the recommended stability anchor. It preserves both attention
processors but removes optimizer updates from the global prompt-conditioning
path that caused much of the N3a drift.

## NN1c: NN1b plus identity-directed supervision

Purpose: combine the stable full-BA route with the strongest identity-learning
lesson from N13/N14/N17.

NN1c is identical to NN1b except:

- add a decoded face identity loss with weight `0.1`;
- apply it only for sufficiently low-noise samples, initially `t <= 400`;
- use the trusted reference identity as target;
- keep all attention-forward math unchanged;
- keep branched CA active and frozen.

This is the highest-upside NN1 option, but it requires a minimal, flag-gated
backport of identity-loss plumbing from `main_clean_exp`. That backport must not
include compact identity memory, target residuals, layer allowlists, or new
epsilon composition.

## Why these three form a useful experiment

- NN1a answers whether the repository reset and two-GPU execution preserve N3a.
- NN1b isolates the proven stability recipe while retaining the full BA
  forward.
- NN1c differs from NN1b only by identity-directed supervision, giving a clean
  estimate of the loss’s value.

If resources permit, all three can run concurrently using three two-GPU pairs.
If only one experimental run is possible after the control, prioritize NN1c,
but retain NN1b as the cleaner attribution run.

## Mandatory diagnostics

At startup:

- assert 70 `BranchedAttnProcessor` instances;
- assert 70 `BranchedCrossAttnProcessor` instances;
- log exact trainable counts separately for SA target/noise, SA reference, CA
  target/noise, and CA reference;
- assert NN1b/c have zero trainable CA tensors while their CA processors remain
  active;
- assert target/reference masks are valid and nonempty;
- fail on invalid reference identity rather than substituting zero silently.

At every validation:

- verify all expected trained processor keys were copied;
- verify BA-off reproduces PhotoMaker;
- measure target-face and background MAE versus PhotoMaker;
- record correct/wrong/null-reference image differences;
- log face detection success, landmark displacement, saturation/chroma drift,
  and fixed identity similarity;
- retain enlarged visual sheets for pose, hair, goggles, hands, and props.

## Decision rules

1. Do not interpret NN1b/c unless NN1a processor counts and step-zero output are
   correct.
2. Stop any run whose face identity falls sharply while chroma/background drift
   rises.
3. Prefer NN1b over NN1c if identity loss improves the metric through expression,
   desaturation, or prop painting rather than recognizability.
4. Select the best checkpoint, not automatically the final checkpoint.
5. Do not add pose mixing, compact memory, layer restriction, or alternative
   CFG composition in this first family.

## Explicitly deferred

- schedule-matched training (`train_ba_all_steps=false`);
- small pose-adaptation ratios;
- explicit removal of zero-token prompt sinks;
- landmark/segmentation masks;
- training any branched CA weights;
- batch-size scaling;
- compact identity-token residuals.

Each is a plausible later isolated experiment, but combining them with NN1
would make the reset uninterpretable.

