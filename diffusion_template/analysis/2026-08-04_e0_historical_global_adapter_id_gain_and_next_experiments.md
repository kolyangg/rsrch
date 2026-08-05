# Why historical E0 wins on identity, and the next adapter-decomposition experiments

**Date:** 4 August 2026

**Evidence cutoff:** matched step-8k validation for the two live E0 runs; complete
step-0-through-20k histories for E1-E6

**Decision metric:** fixed-panel `manual_val/id_sim` is primary. Text metrics are
out of scope. Face-quality metrics are intentionally not used to rank these
runs; manual face inspection is only a catastrophic-failure guardrail.

**Implementation status:** E7-E10 were submitted to Serv on 4 August 2026 as
20k one-A100 jobs and all four are running. They use an isolated runtime so
the two live E0 jobs' shared checkout was not changed.

## Executive answer

Yes: the historical E0 run really does train non-BA adapters in addition to
the intended branched-attention state, and that is now the leading explanation
for its much better identity result. At step 8k, historical E0 reaches
`.36007`, versus `.27338` for fixed BA-only E0, a gap of `.08668`. The gain is
broad: historical E0 wins 88 of 96 matched images and every one of the eight
identities improves on average.

The result does **not** show that PhotoMaker output is mixed into the BA face.
There is no native/PhotoMaker face-output interpolation, residual mixer, or
gate. At every patched self-attention layer, the target face uses target
queries against masked spatial-reference keys/values; the area outside the
target face uses the native target branch. `pose_adapt_ratio=0` means that no
target-face K/V replaces the reference K/V, and branched CA plus
`ca_mixing_for_face` are disabled.

PhotoMaker nevertheless remains important conditioning. Its frozen ID encoder
inserts identity information into the prompt embeddings consumed by ordinary
cross-attention. During the `BOTH` portion of inference, hard branched SA and
PhotoMaker-conditioned ordinary CA operate at the same time. The doubled
reference latent branch is also cross-attended to an ID-token-only face prompt
before its hidden features become spatial-reference K/V. This is conditioning
inside the network, not a mix between two independently generated face
outputs.

The hypothesis that broad adapters help the hard-BA face fit the rest of the
image is plausible but not yet isolated. The historical generic adapter has
two effective components:

- a shared self-attention `to_out` projection applied after the hard
  face/background merge; and
- all ordinary cross-attention Q/K/V/output projections, which can strengthen
  the U-Net's response to PhotoMaker identity tokens.

The second explanation may be at least as important as “accommodation.” A
read-only audit of the exact historical step-8k checkpoint shows that all 210
self-attention Q/K/V LoRA-B tensors remain exactly zero, while all 70 shared
self-attention output tensors and all 280 cross-attention projection tensors
are nonzero. The next work should therefore decompose these two effective
paths, rather than add another BA routing variation.

## 1. Matched identity evidence

The two E0 controls have identical step-zero identity, which removes the
step-zero validation-family confound present in some older comparisons.

| Optimizer step | Historical broad E0 | Fixed BA-only E0 | Historical minus fixed |
|---:|---:|---:|---:|
| 0 | .30187 | .30187 | .00000 |
| 2k | .28186 | .20701 | +.07486 |
| 4k | .33287 | .28759 | +.04529 |
| 6k | .32316 | .25402 | +.06914 |
| 8k | **.36007** | **.27338** | **+.08668** |

Immutable runs:

- historical E0:
  `E0_large_ds_base_historical_r4_20k_full96_r1`, Comet
  [`a5599bd06c9346978c1fca8b8087f634`](https://www.comet.com/nikolay-2104/aug-large-ds/a5599bd06c9346978c1fca8b8087f634);
- fixed E0:
  `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1`, Comet
  [`5b5cbd1584184ce1a9032dd6fafb91c5`](https://www.comet.com/nikolay-2104/aug-large-ds/5b5cbd1584184ce1a9032dd6fafb91c5).

### Per-image breadth at 8k

Historical E0 is not winning because of a few extreme examples:

- historical wins: 88/96;
- fixed wins: 8/96;
- median per-image delta: `+.07609`;
- 10th-percentile delta: `+.01320`;
- 90th-percentile delta: `+.17916`;
- maximum delta: `+.29599`.

| Identity | Historical E0 | Fixed E0 | Delta | Historical wins |
|---|---:|---:|---:|---:|
| Keanu | .41294 | .27048 | +.14246 | 12/12 |
| Jennie | .40656 | .26993 | +.13664 | 12/12 |
| Elon | .43831 | .31992 | +.11839 | 12/12 |
| Marion | .26751 | .18164 | +.08587 | 12/12 |
| Jensen | .42974 | .35533 | +.07441 | 11/12 |
| Lex | .33646 | .27272 | +.06373 | 10/12 |
| Eddie | .16296 | .11886 | +.04409 | 11/12 |
| Jisoo | .42606 | .39819 | +.02788 | 8/12 |

The largest individual gains include Jennie night ride (`+.29599`), Jennie
reading (`+.28393`), Jennie skiing (`+.28050`), Elon chef (`+.27193`), and
Keanu kickboxing (`+.24847`). The largest loss is Jensen dancing (`-.10040`);
the other seven losses are small.

### Relation to E1-E6

Historical E0 at 8k is higher than every post-training E1-E6 observation.
Their best observed post-training identity values are approximately:

| Arm | Single clean-BA delta | Best post-training ID |
|---|---|---:|
| E1 | true reference-key mask | .29664 @16k |
| E2 | branch-local rank-32 output LoRA | .31625 @16k |
| E3 | reference ROI warp | .30082 @12k |
| E4 | mid/up BA sites only | .31186 @16k |
| E5 | inference-active training timesteps | .30892 @20k |
| E6 | FP32 BA trainables | .32313 @4k |

E4's `.36458` at step zero is not a trained result and comes from its changed
site selection, so it is not evidence that E4 learned past historical E0.
E2 is especially informative: adding a face-branch-local output basis helps
relative to some BA-only points, but it does not recover the historical E0
level. A face-only output adapter is therefore not the whole explanation.

## 2. What is actually trainable

### Fixed E0

Fixed E0 owns exactly 840 tensors / 31,948,800 parameters:

- rank-32 `noise_to_q/k/v` and `ref_to_q/k/v` state at all 70 branched-SA
  processors;
- no trainable generic U-Net adapter;
- no trainable PhotoMaker `default` adapter;
- exact optimizer membership and complete schema-v2 checkpointing.

### Historical E0

The historical installer fails before its intended freeze/allowlist stage and
leaves three groups marked trainable:

| Group | Tensors | Parameters |
|---|---:|---:|
| BA processor state | 840 | 31,948,800 |
| Generic `lora_adapter`, rank 32 | 1,120 | 46,448,640 |
| PhotoMaker `default` adapter, rank 64 | 1,120 | 92,897,280 |
| **Total** | **3,080** | **171,294,720** |

That is `5.36×` as many nominal trainable parameters as fixed E0. Raw capacity
alone is therefore a credible alternative explanation to any specific
architectural story.

### Which generic tensors actually changed by 8k

The exact saved historical `weights-epoch4.pth` was inspected read-only on
Serv. LoRA-B/up matrices start at zero, so an exactly zero matrix is a strong
indication that the projection was not exercised by this forward path.

| Generic rank-32 scope | A/B tensors | Parameters | LoRA-B state at 8k |
|---|---:|---:|---|
| SA Q/K/V, 70 sites | 420 | 15,974,400 | all 210 B tensors exactly zero |
| SA `to_out`, 70 sites | 140 | 5,324,800 | all 70 B tensors nonzero |
| CA Q/K/V/`to_out`, 70 sites | 560 | 25,149,440 | all 280 B tensors nonzero |
| **Effective generic path** | **700** | **30,474,240** | **all 350 B tensors nonzero** |

This follows directly from the hard-v1 implementation. With
`branched_attn_weight_mode=noise_and_ref`, self-attention Q/K/V come from the
processor-owned `noise_to_*` and `ref_to_*` layers; the outer U-Net SA Q/K/V
adapters are bypassed. The merged face/background result is still sent through
the shared outer `attn.to_out[0]`, and ordinary CA remains unbranched, so those
adapter projections receive gradients.

Consequently, future “generic adapter” tests should allowlist only the 700
effective tensors. Re-enabling 420 known-dead tensors adds optimizer state but
no scientific information.

### Important checkpoint/validation asymmetry

The historical schema saves the trained generic adapter and BA processors but
does **not** save the trained PhotoMaker `default` adapter. Full processor copy
transfers processor-owned BA buffers and deltas; it does not transfer the
outer trained `default` adapter. The alternate RealVis validation model reloads
the original pretrained PhotoMaker default state.

Therefore the 8k validation gain is directly expressed by:

1. historical BA weights learned while the broad adapters were active; and
2. the saved trained generic adapter.

It is **not** directly using the learned PhotoMaker-default update. That update
can still matter indirectly because it participated in the training forward
pass and changed the gradients received by BA and the generic adapter. This is
why the current result cannot yet distinguish “generic adapter is sufficient”
from “generic plus training-time default co-adaptation is required.”

## 3. Exact face routing: reference branch versus PhotoMaker

For a target-face mask `M`, the historical and fixed hard-v1 processors use
the same essential self-attention computation:

```text
Q_target = Q(noisy target hidden state)

H_background = Attention(Q_target outside M, K_target, V_target)
H_face       = Attention(Q_target inside M, K_reference-face, V_reference-face)

H_merged = (1 - M) * H_background + M * H_face
output   = shared_to_out(H_merged)
```

The generated-face bbox produces `M`; the reference-face bbox produces the
reference key/value mask. Masks are resized at each attention resolution.
With the current E0 settings:

- `pose_adapt_ratio=0`: `K_reference-face/V_reference-face` contain no
  target-native face substitution;
- `ca_mixing_for_face=false`: there is no extra cross-attention face-output
  mixer;
- `disable_branched_ca=true`: CA uses the ordinary SDXL attention path;
- `branched_start_mode=both`: once BA starts, PhotoMaker prompt conditioning
  stays active;
- there is no learned face gate, residual native-face addition, or
  PhotoMaker/BA image interpolation.

Inference uses three phases with the current 50-step schedule:

1. before PhotoMaker starts: text-only/native path;
2. PhotoMaker phase: native path with ID-fused prompt embeddings;
3. `BOTH` phase: hard branched SA plus the same ID-conditioned ordinary CA.

When BA is active, the generic adapter is active with the branched U-Net. When
BA is inactive, the pipeline selects only the PhotoMaker `default` adapter.
Thus the saved generic update is specifically expressed during the BA-active
portion of validation.

The most precise answer to “is the face just reference branch or a PhotoMaker
mix?” is:

- **At the self-attention routing decision:** the masked face is hard
  target-Q/reference-KV BA, not a PhotoMaker output mix.
- **For the final generated face:** it is not produced by SA in isolation.
  Ordinary CA supplies PhotoMaker ID-token conditioning to both halves of the
  doubled U-Net, and convolutions/residual blocks integrate the result across
  layers. The reference hidden features used as spatial K/V have themselves
  passed through PhotoMaker-conditioned CA.

## 4. Does global adaptation help the face accommodate the image?

### What supports that hypothesis

- The shared SA output adapter is applied after the hard per-token
  face/background merge. It can learn a common channel basis for both sides of
  that merge, after which later U-Net blocks can integrate them.
- Historical improvements occur on 88/96 images and all identities, not just
  on one face or prompt family.
- In matched 8k visual pairs, scene layout and body/background content are
  usually very similar while facial morphology and identity change. This is
  compatible with better representation of the imposed hard face branch.
- E2's branch-local output adapter does not match historical E0, suggesting
  that shared/global context may matter.

### What prevents that conclusion today

- The generic CA component is nearly five times larger than the shared-SA
  output component (`25.15M` versus `5.32M`) and directly controls response to
  PhotoMaker's identity-fused embeddings. It may simply make identity
  conditioning stronger.
- The historical PhotoMaker-default adapter also trains, even though its
  learned update is reset for validation. It can alter the optimization path.
- Historical E0 has much more capacity, so better reconstruction of the
  identity loss does not by itself identify a BA-specific mechanism.
- The selected visual pairs show face changes more clearly than a new
  face/background blending behavior. Similar attachment/layout in both E0
  arms means the current images do not prove “accommodation.”

The best current interpretation is therefore: **global adapter co-adaptation
very likely causes the identity gain, but the gain may come from stronger
PhotoMaker-conditioned CA, from a shared post-merge SA output basis, or from
their interaction.** The following four experiments give that ambiguity a
direct answer.

## 5. Four next experiments in priority order

All four are 20k, one-A100 runs with full-96 validation at step 0 and every
2k. This covers the delayed recovery seen in E1, E2, and E4 and gives every
arm the same endpoint as fixed E0 and the completed E1-E6 suite.

Every run must inherit the exact fixed E0 substrate and keep:

- hard target-Q/reference-KV branched SA at all 70 sites;
- `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, branched CA off;
- no native/PhotoMaker face-output mix, gate, or residual face path;
- the same data stream, batch 2, seed, LR, loss, masks/bboxes, validation
  images, prompts, scheduler, inference steps, and metric definition;
- strict allowlisted ownership, optimizer-membership assertions, and complete
  schema-v2 checkpointing;
- Comet project `aug-large-ds` and 96-row per-image ID tables.

| Arm | Hydra config | Experiment record | Serv start script and MLS YAML |
|---|---|---|---|
| E7 | `src/configs/E7_large_ds_generic_effective_20k.yaml` | `experiments/large_dataset/E7_large_ds_generic_effective_r32_20k_full96_r1.json` | `serv_run_packages/E7_large_ds_generic_effective_r32_20k_full96_r1/start_E7_large_ds_generic_effective_r32_20k_full96_r1_1gpu.sh`; `serv_run_packages/E7_large_ds_generic_effective_r32_20k_full96_r1/run_E7_large_ds_generic_effective_r32_20k_full96_r1_1gpu.yaml` |
| E8 | `src/configs/E8_large_ds_generic_ca_20k.yaml` | `experiments/large_dataset/E8_large_ds_generic_ca_r32_20k_full96_r1.json` | `serv_run_packages/E8_large_ds_generic_ca_r32_20k_full96_r1/start_E8_large_ds_generic_ca_r32_20k_full96_r1_1gpu.sh`; `serv_run_packages/E8_large_ds_generic_ca_r32_20k_full96_r1/run_E8_large_ds_generic_ca_r32_20k_full96_r1_1gpu.yaml` |
| E9 | `src/configs/E9_large_ds_shared_saout_20k.yaml` | `experiments/large_dataset/E9_large_ds_shared_saout_r32_20k_full96_r1.json` | `serv_run_packages/E9_large_ds_shared_saout_r32_20k_full96_r1/start_E9_large_ds_shared_saout_r32_20k_full96_r1_1gpu.sh`; `serv_run_packages/E9_large_ds_shared_saout_r32_20k_full96_r1/run_E9_large_ds_shared_saout_r32_20k_full96_r1_1gpu.yaml` |
| E10 | `src/configs/E10_large_ds_pmdefault_effective_20k.yaml` | `experiments/large_dataset/E10_large_ds_pmdefault_effective_r64_20k_full96_r1.json` | `serv_run_packages/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/start_E10_large_ds_pmdefault_effective_r64_20k_full96_r1_1gpu.sh`; `serv_run_packages/E10_large_ds_pmdefault_effective_r64_20k_full96_r1/run_E10_large_ds_pmdefault_effective_r64_20k_full96_r1_1gpu.yaml` |

### Serv submission record

All four jobs were accepted and reached `running`. Together with the two live
one-GPU E0 controls, this uses exactly the normal six-A100 project ceiling.
No eight-GPU exception was used for E7-E10.

| Arm | MLS job | Immutable Comet key |
|---|---|---|
| E7 | `lm-mpi-job-b90da1c7-9435-4aa7-a5de-00422c7c6022` | [`e3d540a8f5c84e9db960214a1342ca04`](https://www.comet.com/nikolay-2104/aug-large-ds/e3d540a8f5c84e9db960214a1342ca04) |
| E8 | `lm-mpi-job-153d81de-078d-4ba5-89ec-729ea8ca01db` | [`db1326c7591e484597f3009db63af42f`](https://www.comet.com/nikolay-2104/aug-large-ds/db1326c7591e484597f3009db63af42f) |
| E9 | `lm-mpi-job-c2cf07ab-eaf5-4176-8283-929682dc3ec8` | [`deb40502cfc849a0aecc8e48b4eec005`](https://www.comet.com/nikolay-2104/aug-large-ds/deb40502cfc849a0aecc8e48b4eec005) |
| E10 | `lm-mpi-job-01a36932-2be9-413c-8cb3-cadcca9ae5ad` | [`0375f172f75c482f840317ec5ae41c05`](https://www.comet.com/nikolay-2104/aug-large-ds/0375f172f75c482f840317ec5ae41c05) |

The immutable runtime is
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804`.
Before submission, 24 launch-critical local/remote files matched by SHA-256,
all four Hydra single-delta gates passed remotely, and all package scripts
passed remote shell syntax checks. Startup created each required
`saved/<run_name>/comet_experiment.json`, registered the exact display name in
project `aug-large-ds`, and logged the per-run experiment comment as required
Comet Other metadata. A read-only immutable-key API check retrieved all four
comments. Real-model startup also passed the exact ownership gates: E7
`1,540/62,423,040`, E8 `1,400/57,098,240`, E9 `980/37,273,600`, and E10
`1,540/92,897,280`; every arm reported all `840/840` BA processor parameters
in its optimizer and entered step-zero full-96 validation.

### E7 — effective generic adapter

**Run name:** `E7_large_ds_generic_effective_r32_20k_full96_r1`

**Config:** `src/configs/E7_large_ds_generic_effective_20k.yaml`

**One delta from fixed E0:** train the rank-32 generic adapter only at the
historically effective sites: all `attn2` Q/K/V/output projections and all
`attn1.to_out.0` projections. Keep PhotoMaker `default` frozen.

**Expected ownership:** 1,540 tensors / 62,423,040 parameters:

- BA: 840 / 31,948,800;
- effective generic adapter: 700 / 30,474,240.

**Binary question:** Is the saved generic path sufficient to recover the
historical identity benefit without training the PhotoMaker-default adapter?

**Interpretation:**

- strong recovery means the historical gain can be made clean and resumable
  with BA plus one explicit generic adapter family;
- weak recovery means training-time default-adapter interaction or some other
  historical fail-open behavior is necessary.

This is the highest-priority run because it most closely reproduces the state
that is actually present in historical validation.

### E8 — ordinary CA only

**Run name:** `E8_large_ds_generic_ca_r32_20k_full96_r1`

**Config:** `src/configs/E8_large_ds_generic_ca_20k.yaml`

**One delta from fixed E0:** train rank-32 generic LoRA only on ordinary
`attn2` Q/K/V/output projections. All generic SA projections and PhotoMaker
`default` remain frozen.

**Expected ownership:** 1,400 tensors / 57,098,240 parameters:

- BA: 840 / 31,948,800;
- generic CA: 560 / 25,149,440.

**Binary question:** Does adapting the U-Net response to PhotoMaker
identity-token conditioning explain most of historical E0's gain?

**Interpretation:** a result near E7 supports identity-conditioning
amplification rather than a post-merge accommodation mechanism.

### E9 — shared SA output only

**Run name:** `E9_large_ds_shared_saout_r32_20k_full96_r1`

**Config:** `src/configs/E9_large_ds_shared_saout_20k.yaml`

**One delta from fixed E0:** train rank-32 generic LoRA only on the 70 shared
`attn1.to_out.0` projections used after hard face/background merge. Ordinary
CA, generic SA Q/K/V, and PhotoMaker `default` remain frozen.

**Expected ownership:** 980 tensors / 37,273,600 parameters:

- BA: 840 / 31,948,800;
- shared SA output: 140 / 5,324,800.

**Binary question:** Does a shared post-merge output basis let the hard BA face
fit the surrounding native representation better?

This is deliberately different from E2. E2 adapts only the reference-face
branch output before the masked merge; E9 adapts the shared outer output
projection actually exercised by historical E0 for face, background, and
reference tokens.

**Interpretation:** a result near E7 supports the user's accommodation
hypothesis. A weak E9 with strong E8 rejects it as the main identity mechanism.

### E10 — PhotoMaker default adapter only, saved correctly

**Run name:** `E10_large_ds_pmdefault_effective_r64_20k_full96_r1`

**Config:** `src/configs/E10_large_ds_pmdefault_effective_20k.yaml`

**One delta from fixed E0:** train the pretrained rank-64 PhotoMaker `default`
adapter only at the effective sites (`attn2` Q/K/V/output plus
`attn1.to_out.0`). Keep the generic adapter frozen. Unlike historical E0,
save and load every trained default-adapter tensor during validation.

**Expected ownership:** 1,540 tensors / 92,897,280 parameters:

- BA: 840 / 31,948,800;
- effective PhotoMaker default: 700 / 60,948,480.

**Binary question:** Can direct, correctly checkpointed adaptation of the
pretrained PhotoMaker U-Net adapter improve identity with hard BA, independently
of the generic adapter?

This tests a potentially useful clean mechanism, not an exact reproduction of
historical validation. Historical validation reset this learned adapter.

## 6. Decision rules

At matched step 8k, define recovered historical gap as:

```text
recovery = (ID_arm - ID_fixed_E0) / (ID_historical_E0 - ID_fixed_E0)
```

Using the current 8k values, the denominator is `.08668`:

- `<25%` recovery: mechanism is weak/negative;
- `>=50%` recovery: meaningful mechanistic contribution (ID at least
  approximately `.31672` at 8k);
- `>=75%` recovery: promotion-strength contribution (ID at least
  approximately `.33840` at 8k).

Do not decide from the mean alone. Require the direction to persist at two
consecutive gates, including 12k or later, and inspect the fixed 96-row table:

- at least 65/96 wins versus fixed E0 for a meaningful positive;
- no result driven by only one identity;
- manual face crops contain no new catastrophic anatomy/attachment cluster.

Do not use text score or small face-quality-metric differences to rank these
arms. ID similarity is the optimization decision here.

Expected causal readout:

| Outcome | Conclusion |
|---|---|
| E7 strong; E8 strong; E9 weak | generic CA/PhotoMaker identity conditioning is the main mechanism |
| E7 strong; E9 strong; E8 weak | shared post-merge accommodation is the main mechanism |
| E7 strong; E8 and E9 both partial | both paths contribute; test their clean combination only after isolation |
| E7 weak; E10 strong | updating the pretrained PhotoMaker adapter directly is more important than a new generic adapter |
| E7 and E10 weak | historical benefit likely depends on simultaneous-adapter optimization or the historical train/validation mismatch |

For the best checkpoint, run the existing fixed-checkpoint spatial-reference
diagnostic without another training job: matched spatial reference versus
shuffled and zero spatial reference, while keeping PhotoMaker input images,
ID embeddings, prompt, seed, target bbox, and every other input fixed. A
candidate whose ID gain survives removal/shuffling of spatial K/V is mainly a
PhotoMaker/global-adapter result, not evidence that the BA reference path
caused the gain.

## 7. Implementation plan for the next agent

### 7.1 Add explicit, defaults-off adapter scopes

Add two audited model/config selectors; do not reuse the historical fail-open
path:

```yaml
model:
  generic_adapter_train_scope: none
  photomaker_default_train_scope: none
```

Allowed values should be explicit and fail closed:

```text
generic_adapter_train_scope:
  none | effective_all | cross_attention | self_attention_output

photomaker_default_train_scope:
  none | effective_all
```

The fixed-E0 default remains `none/none`. Reject unknown values and reject
trainable outer SA Q/K/V under hard `noise_and_ref`, because code inspection
and the 8k checkpoint prove that path is bypassed.

### 7.2 Extend the strict allowlist, not the legacy installer

In `src/model/photomaker_branched/lora2_helpers.py`, extend the exact expected
trainable-name construction after the existing BA allowlist:

- generic adapter names must contain `.lora_adapter.`;
- PhotoMaker adapter names must contain `.default.`;
- `cross_attention` selects `.attn2.` plus Q/K/V/`to_out.0` LoRA A/B;
- `self_attention_output` selects `.attn1.to_out.0.` LoRA A/B;
- `effective_all` is the union of those two scopes;
- every base weight, text encoder, ID encoder, VAE parameter, unrelated
  adapter, and plain U-Net parameter stays frozen.

Preserve the current BA allowlist unchanged. Add an independent category
partition assertion in each config/spec so a typo cannot pass merely because
the total count happens to match.

### 7.3 Make schema-v2 aware of the scopes

Add both selectors to the architecture manifest in
`src/model/photomaker_branched/lora2.py`. The existing trainable-v2 saver then
stores the exact requires-grad tensors, including `default` when E10 enables
it. Validation construction must receive the same selectors before
`prepare_for_training()`, and manifest equality must reject a mismatched load.

Do not use the historical legacy state format. Do not silently reset E10's
trained default adapter during validation.

### 7.4 Preserve active-adapter behavior

Keep both adapter modules installed and active in the existing branch-active
mode so frozen pretrained `default` conditioning and zero-initialized generic
state preserve step-zero parity. `requires_grad` controls ownership; adapter
activation controls the forward equation.

Assert that all four new arms reproduce fixed E0's step-zero ID within normal
deterministic tolerance before interpreting training deltas.

### 7.5 Configs, records, launch packages, and comments

The implementation provides one leaf config, immutable experiment JSON, Serv
package, start script, and one-GPU MLS YAML for each exact run name above. Each leaf inherits
`large_dataset_rhca_hard_v1_audited_20k` but overrides:

- `trainer.n_epochs: 10` with `trainer.epoch_len: 2000` for 20k;
- the single adapter-scope delta;
- its exact expected ownership partition;
- a concise Comet comment stating the isolated question.

Suggested Comet comments:

- E7: “BA + effective generic r32; default frozen; test whether the saved
  historical generic path alone recovers ID.”
- E8: “BA + ordinary CA r32 only; test whether stronger PhotoMaker ID-token
  response explains the historical ID gain.”
- E9: “BA + shared SA to_out r32 only; test post-merge face/background
  accommodation.”
- E10: “BA + correctly checkpointed effective PhotoMaker-default r64 only;
  test direct pretrained-adapter adaptation.”

All packages must request exactly one A100/process and use `aug-large-ds`.
The approved jobs were submitted only after rechecking current Serv
allocations under the normal project GPU ceiling.

### 7.6 Minimal verification

Only the following focused checks are needed:

1. Hydra composition for fixed E0 and E7-E10;
2. exact trainable and optimizer tensor/parameter counts for each arm;
3. schema-v2 round-trip for one representative generic arm and E10 default
   state;
4. one tiny forward/backward smoke proving selected LoRA-B gradients are
   nonzero and every excluded scope has no trainable parameter;
5. validation processor installation/load smoke preserving hard BA,
   `pose_adapt_ratio=0`, CA mixing off, and branched CA off;
6. shell syntax and MLS package integrity checks.

No broad refactor and no additional test suite is warranted.

Completed local smoke evidence:

- E7-E10 Hydra composition passed with exactly 20,000 steps, the fixed
  architecture/validation contract, one scientific selector delta, and exact
  audit metadata; regression composition also passed for both E0 controls and
  E1-E6;
- the scoped ownership helper passed representative generic/default CA,
  shared-SA-output, and excluded-SA-Q/K/V cases, exact optimizer-membership
  checks, and a schema-v2 selected-default state round-trip;
- changed Python modules compile; active/package Bash files pass `bash -n`;
- all four JSON/YAML/package triples agree on run name, one A100, 20k, start
  script, and asynchronous CUDA operation;
- all 18 audited runtime hashes match the current local files.

A full SDXL/InsightFace startup was deliberately not run locally; the strict
real-model tensor/parameter assertions remain startup gates on Serv. No live
Serv checkout was changed during preparation.

## 8. Recommendation

E7-E10 are now running in parallel. E7 remains the closest clean test of the
state that actually produces historical validation images; interpret the
decomposition only at matched validation gates.

If E7 reaches at least 75% recovery and remains broad through 12k/20k, the new
working base should become **hard BA plus explicitly allowlisted effective
generic adaptation**, not the historical fail-open run. E8 and E9 then tell us
whether to retain only CA or only shared SA output. If E7 falls well short,
the next controlled follow-up is a clean both-adapter run with all state saved,
paired with an evaluation that resets only `default` to pretrained state. That
single paired evaluation would quantify the historical training-time-default
interaction without making the broken checkpoint behavior the new base.
