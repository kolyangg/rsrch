# Loss objective and direct identity-supervision advice

**Date:** 5 August 2026

**Scope:** code/config audit, completed-run evidence, and research advice only

**Code changes:** none
**Primary question:** whether to restore alternating masked/full diffusion loss,
and whether an identity-similarity term should be added to training

## Executive recommendation

The current setup is not consistently using the alternating loss its name
suggests. E0-E13 compose `MaskedDiffusionLoss` with
`trainer.masked_loss_step=1`; in the trainer this makes **every optimizer batch
face-crop epsilon MSE**. True alternation is `masked_loss_step=2`: even batches
use face-crop MSE and odd batches use full-latent MSE.

I would **not globally revert E14-E18 to the old alternation**. The E14+ loss is
more suitable for jointly trainable global adapters because it retains
full-strength face supervision on every update while also applying small
full-image and face-boundary terms. E13 and E14 are already the important
loss-only comparison: E13 uses the historical face-only objective and E14 uses
the protected simultaneous objective on the same shadow-coadapter design. Do
not change either run mid-flight.

I would nevertheless run one future, strictly matched **true-alternating
control** if E13/E14 do not settle the question. It is the closest replay of
the original PhotoMaker/April training recipe, but it is not automatically the
best objective for maximizing ID similarity: half of its updates dilute the
face signal over the whole latent.

There is good evidence that the diffusion objective is **misaligned with
`ID_sim`**, but not enough evidence that it is the primary cause of the early
ceiling:

- nearly identical face-only training losses coexist with very different
  `ID_sim` results;
- the loss never asks whether the correct identity reference was used;
- face-only loss leaves globally active generic/default adapters without
  exterior supervision;
- but historical E0 and historical BigCelebs both achieved substantial,
  multi-gate identity growth under the same face-only objective;
- architecture and trainable-parameter ownership explain much larger
  differences than any completed loss ablation;
- a completed correct-versus-shuffled reference-ranking loss improved its own
  causal separation without improving validation identity.

Direct identity supervision is therefore worth testing as a **small auxiliary
to diffusion loss**, not as a replacement. E16 already implements the sensible
first version of this experiment, although it is a PhotoMaker-CLIP proxy rather
than the actual InsightFace validation metric. As of the live Serv check for
this report, E13 r3 and E14 r4 are running, while E15-E18 r2 were rejected by
the workspace GPU limit; consequently, **there is no E16 result yet**.

## 1. What loss is actually being optimized?

### 1.1 `MaskedDiffusionLoss`

[`src/loss/diffusion_loss.py`](../src/loss/diffusion_loss.py) contains two
branches:

```text
masked batch:  mean epsilon MSE inside bbox/8 in latent space
other batch:   mean epsilon MSE across the complete latent
```

The face term is averaged within each valid crop and then across valid samples;
it is not weighted by face area. An invalid/empty set of boxes falls back to
full-latent MSE.

[`PhotomakerLoraTrainer.process_batch`](../src/trainer/sdxl_trainers.py) selects
the branch deterministically:

```text
is_masked_loss = masked_loss_step > 0
                 and batch_idx % masked_loss_step == 0
```

This has an easy-to-miss consequence:

| `masked_loss_step` | Effective objective |
|---:|---|
| `1` | Every batch is face-only epsilon MSE |
| `2` | Face-only on even batches; full-latent on odd batches |
| `N > 2` | One face-only batch per `N`; all other batches full-latent |
| `0` | Full-latent on every batch |

The Hydra/config label `loss_kind: masked_alternating` does not establish that
alternation occurs. The trainer integer does.

### 1.2 Which experiments use which loss?

| Experiments | Effective training loss |
|---|---|
| E0 historical, E0 fixed, E1-E13 | Face-crop epsilon MSE on every update (`MaskedDiffusionLoss`, step `1`) |
| E14, E15, E17, E18 | `1.0 L_face + 0.1 L_full + 0.05 L_boundary` on every update |
| E16 | Same protected loss as E15, plus a scheduled predicted-x0 identity proxy |
| April one-ID historical/replay | True deterministic 50/50 face/full alternation (step `2`) |

The repository trainer default remains `masked_loss_step=2`, and the April
historical replay pins it explicitly. The August Large Dataset inheritance
chain overrides it to `1` in
[`large_dataset_joint_r128_24k.yaml`](../src/configs/large_dataset_joint_r128_24k.yaml).

The original PhotoMaker paper also reports using masked diffusion loss with a
50% probability. Thus the true-alternating form is historically grounded, but
our bbox crop is not necessarily pixel-for-pixel identical to PhotoMaker's
person-mask construction. See the [PhotoMaker paper](https://arxiv.org/abs/2312.04461).

### 1.3 The E14+ protected loss

[`BranchedReferenceLoss`](../src/loss/branched_reference_loss.py) computes all
three epsilon losses simultaneously:

```text
L_protected = 1.0 * L_face
            + 0.1 * L_full
            + 0.05 * L_boundary_ring
```

The current E14 weights are in
[`E14_large_ds_joint_shadow_sa128_protected_24k.yaml`](../src/configs/E14_large_ds_joint_shadow_sa128_protected_24k.yaml).
The two-latent-cell boundary ring is immediately outside the face bbox. The
correct-versus-shuffled reference branch is diagnostic only because
`reference_weight=0`.

This is not equivalent to alternating loss:

- the face receives a strong, area-normalized gradient on every update;
- the global scene and body receive a smaller preservation gradient on every
  update;
- the transition around the hard BA rectangle is explicitly supervised;
- component weights sum to more than one, so E13/E14 total gradient scale may
  differ as well as gradient direction. Component and per-role gradient norms
  must be considered when interpreting the A/B.

For adapters that can affect the full U-Net, this is a safer objective than
face-only loss. It directly addresses the failure exposed by E10, where the
persisted PhotoMaker-default path moved composition and invalidated fixed face
boxes while exterior pixels had no loss.

## 2. Should we revert to alternating masked/full loss?

### Decision

**Consider it as a controlled arm; do not make it the suite default yet.**

True alternation has two advantages:

1. it restores the repo/upstream historical recipe;
2. it makes half of the updates explicitly preserve scene, body, pose, and
   composition.

It also has two disadvantages for the stated goal of maximizing `ID_sim`:

1. on full-image updates, the small face occupies only a small share of the
   averaged latent, so identity-directed gradients are diluted;
2. it alternates between materially different gradient fields instead of
   optimizing a stable weighted compromise on every batch.

The protected E14 objective keeps the useful face emphasis without abandoning
global supervision. It is therefore the better default hypothesis for a
joint BA/generic/default model. If a later alternating arm is run, it should
change only the following relative to one frozen architecture/data/seed
control:

```yaml
loss_function:
  _target_: src.loss.diffusion_loss.MaskedDiffusionLoss
trainer:
  masked_loss_step: 2
```

Compare it against both face-only step `1` and protected simultaneous loss.
Keep parameter ownership, learning rates, timestep sampling, target/reference
schedule, validation, and total steps identical. A useful additional check is
to normalize or at least log total/per-role gradient RMS, because changing the
loss also changes the effective optimizer scale.

### What the earlier face-only versus alternating experiment tells us

The July Cosmic one-ID comparison was a useful matched loss ablation for visual
stability. Changing from alternating to every-batch face-only did **not** repair
displaced/duplicated facial anatomy. At 2k, its learned SA and CA projection
deltas remained about `1.48x` and `1.45x` the successful one-ID control. The
analysis concluded that data/reference geometry and global CA/target adapters,
not the alternation alone, dominated that failure. See
[`2026-07-24_cosmic_large_pipeline_performance_analysis.md`](2026-07-24_cosmic_large_pipeline_performance_analysis.md).

That experiment was short, on a pathological one-target Cosmic setup, and did
not isolate a long-horizon `ID_sim` ceiling. It is evidence against treating a
loss revert as a cure-all, not evidence that loss never matters.

## 3. Was the loss likely preventing further `ID_sim` progression?

### Evidence that supports objective mismatch

All E0-E12 August arms used the same every-batch face epsilon objective, yet
their identity endpoints differed enormously. Excluding E5, whose restricted
timestep distribution changes the raw loss scale, their mean logged training
losses occupy the narrow range `0.13823-0.14010`; their latest `ID_sim` spans
`0.16912-0.36889`.

Two concrete examples make the mismatch clear:

- historical E0: mean logged train loss `0.13823`, latest ID `0.36889`;
- E10: mean logged train loss `0.13857`, latest fixed-mask ID `0.16912`, with
  severe visible composition drift;
- E11: mean logged train loss `0.13869`, latest ID `0.32167`.

A low face epsilon error therefore does not imply a recognizably correct
identity, correct face/body association, or stable composition. The objective
can be reduced through target-image cues, generic face priors, or average
same-ID statistics without learning to distinguish the supplied reference
from another face.

Historical E0's own curves are also non-monotonic. The train-loss window that
led into the 8k validation averaged about `0.13405` and produced ID `0.36007`;
the window leading into the best 14k validation averaged a higher `0.13636`
and produced ID `0.37083`; the window leading into 20k fell to `0.13584` while
ID ended at `0.36889`. This rules out using raw stochastic train loss as an
early-stop proxy for identity.

Finally, face-only loss is structurally incomplete once global adapters are
trainable. Pixels outside the face provide no direct constraint even though
generic/default attention projections can move the body and scene. This is a
strong reason for E14's full and boundary terms regardless of whether they
raise the aggregate ID score.

### Evidence against claiming the loss caused the ceiling

The same face-only objective did not impose a universal early hard limit:

- August historical E0 rose from `0.30187` at step 0 to `0.37083` at 14k and
  ended at `0.36889` at 20k;
- historical BigCelebs dipped at 2k, reached `0.3723` at 10k, and peaked at
  `0.3817` at 18k under face-only step `1`;
- the older Large Dataset historical r4 reached `0.39039` at 24k.

Within the August suite, architecture/ownership changes under the same loss
explain the dominant differences. Historical E0's broad co-adaptation beats
fixed BA-only E0 at 20k by `+0.07161`, while E11's increased BA rank supplies a
smaller clean gain. This points first to conditioning capacity and optimization
trajectory, not a loss-imposed ceiling.

There is also no fixed-noise validation diffusion-loss curve. Logged training
loss averages random targets, references, crops, noise, and timesteps, so its
small oscillations cannot prove that the optimizer reached a meaningful
identity optimum.

### Bottom line

The current loss is **likely a contributing ceiling and definitely an
evaluation mismatch**, but “the loss prevented further ID progression” is
still a hypothesis. E13 versus E14 is the first appropriately controlled test
in the current high-capacity architecture. The decision should be made from
fixed-96 identity trajectories plus visual anatomy/composition, not from the
training loss alone.

## 4. Should `ID_sim` be directly in the loss?

### Use an identity embedding auxiliary, not the literal validation function

The literal validation implementation is a poor differentiable objective.
[`IDSimBest`](../src/metrics/id_sim_metric.py) runs InsightFace detection and
alignment, computes recognition embeddings, and takes the maximum similarity
over every detected generated face. The detector/ONNX path and discrete face
selection are not differentiable. More importantly, optimizing “best face
anywhere” could reward a duplicate or detached high-similarity face—the exact
metric loophole already observed visually.

A better training surrogate is:

```text
L_id = 1 - cosine(F(aligned predicted-x0 intended face),
                  stopgrad(F(same-ID reference/prototype)))
```

where `F` is a frozen differentiable face recognizer. Keep the unchanged
InsightFace `IDSimBest` as the primary historical evaluation metric, but do not
reproduce its best-anywhere selection inside training.

### Why it should remain auxiliary

An identity embedding is intentionally invariant to expression, pose, lighting,
and some appearance details. Optimizing it alone can produce an adversarially
recognizable but visually poor face, suppress prompt-requested expression, or
copy a canonical frontal appearance. The August images already show that high
`ID_sim` can coexist with bad goggles, hands, mouths, and face/body alignment.

The identity term should therefore be:

- low weight and ramped in after ordinary denoising has begun to learn;
- evaluated only on low/moderate-noise predicted-x0 estimates;
- applied to the intended fixed/aligned face, not the best detected face;
- compared with a distinct same-ID reference or multi-image identity centroid,
  rather than relying only on the target frame;
- accompanied by face/full/boundary diffusion loss and visual quality gates;
- monitored with its own gradient norm so it cannot silently dominate the BA
  and adapter updates.

### What E16 actually implements

[`E16_large_ds_joint_persist_sa128_idloss_24k.yaml`](../src/configs/E16_large_ds_joint_persist_sa128_idloss_24k.yaml)
adds a conservative predicted-x0 proxy to E15:

- weight `0` through 2k, linearly ramping to `0.05` at 6k;
- attempted every fourth optimizer step;
- only samples with diffusion timestep `t <= 400` are eligible;
- one eligible sample is decoded differentiably through the frozen VAE;
- generated and real target faces use the training bbox plus 25% padding and
  are resized to 224;
- the already loaded frozen PhotoMaker CLIP vision tower provides pooled
  embeddings;
- the loss is `1 - cosine(predicted_face, target_face)`.

This is a useful first safety/compute experiment, but it is **not direct
`ID_sim`**:

| Validation `ID_sim` | E16 auxiliary |
|---|---|
| InsightFace recognition embedding | PhotoMaker CLIP vision embedding |
| detector + face alignment | fixed padded training bbox |
| precomputed identity reference | the current real target image |
| best score over all detected faces | one intended crop |
| inference-only ONNX path | differentiable PyTorch path |

If E16 improves the proxy but not the fixed-96 InsightFace score, that is not a
surprise; it is evidence that the proxy is insufficient. A follow-up should
use a frozen differentiable ArcFace/AdaFace implementation with offline fixed
alignment and an identity centroid, while keeping the primary validation
metric unchanged for comparability.

## 5. Have identity losses been tried before?

### In this repository

There is **no completed run that directly optimizes an InsightFace/ArcFace-like
generated-face embedding similarity**.

- E16 is the first implemented predicted-x0 identity-proxy arm, but its latest
  Serv submission was rejected before startup because no workspace GPU was
  free.
- E17 adds an identity-token CA route; it is architectural conditioning, not
  identity loss.
- Earlier ArcFace scores were used for dataset/reference selection and
  curation, not backpropagated into the generator.
- The closest completed objective experiment is the BigCelebs E4
  matched-versus-shuffled spatial-reference rank loss. At 2k it scored
  `0.4639` versus `0.4779` for the otherwise matched no-rank E3. In the longer
  run, reference-error separation later improved while validation identity
  declined. The loss could satisfy itself partly by worsening the wrong
  reference branch. See
  [`2026-08-03_anchored_mix_sa_v3_rank_40k_through14k_results_and_e6_plan.md`](2026-08-03_anchored_mix_sa_v3_rank_40k_through14k_results_and_e6_plan.md).

That negative rank-loss result does not disprove embedding identity loss, but
it warns that a loss correlated with “reference sensitivity” is not necessarily
correlated with production identity quality.

### Published precedents

There are credible adjacent precedents for predicted-x0 face-recognition loss:

- [DCFace (CVPR 2023)](https://arxiv.org/abs/2304.07060) applies a pretrained
  face-recognition feature loss to a one-step predicted clean image. Its
  ablation found that a naive constant pull toward the canonical ID image hurt
  diversity; a timestep-dependent interpolation between ID and style targets
  worked better.
- [ReF-LDM (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html)
  uses ArcFace cosine distance on predicted clean faces. The authors report
  that naive identity loss at very noisy timesteps worsened image quality and
  introduce timestep scaling to suppress unreliable high-noise supervision.
- [Diff-ID (2026 preprint)](https://arxiv.org/abs/2607.25078) uses an ArcFace
  cosine-based pseudo-discriminator loss with timestep weighting. It reports a
  better identity/realism trade-off but does not exceed InstantID in raw
  ArcFace similarity, another reminder that direct optimization is not a
  guaranteed score breakthrough.

These tasks are face generation/restoration rather than our SDXL full-body
prompted personalization setup, so they are design precedents, not direct
evidence that the same weight will work here. Their common lesson strongly
supports E16's low-noise, scheduled, auxiliary design.

## 6. Recommended decision sequence

1. **Let E13 and E14 remain unchanged.** They are the current matched test of
   face-only versus protected diffusion loss in the historical shadow
   architecture. Compare full-96 ID at every gate, prompt/identity cells,
   body alignment, duplicate faces, boundary artifacts, and component/role
   gradient norms.
2. **Do not infer a winner from the scalar training loss.** A protected loss
   has a different scale and purpose. Success is higher `ID_sim` without the
   E10-style layout failure.
3. **Run E15 before interpreting E16.** E15 is the protected persistent
   control; E16 only answers the identity-auxiliary question relative to it.
4. **Run E16 unchanged as the first identity-loss experiment when capacity is
   available.** It is deliberately weak enough to test feasibility without
   making recognition loss the main generator objective.
5. **Only if E16 shows positive fixed-96 ID slope without visual regression,
   test a closer biometric surrogate.** Use frozen differentiable
   ArcFace/AdaFace, fixed offline alignment, a distinct-reference/identity
   centroid, low-noise predicted x0, and one isolated weight schedule.
6. **If E13 beats E14 or E14 protects quality but loses too much ID, add one
   true-alternating control.** Do not replace the whole suite based on the
   historical name alone.

### Promotion rule for an identity-loss arm

Promote only if all of the following hold against the matched no-ID-loss
control:

- aggregate and identity/prompt-cell `ID_sim` improve at multiple consecutive
  validation gates, not one noisy peak;
- intended face/body association remains correct and duplicate-face count does
  not increase;
- text similarity and face-quality tails do not materially regress;
- the gain remains when inspecting intended-box identity, not only
  best-detected-face identity;
- removing or shuffling reference evidence lowers identity, showing the model
  still uses the explicit BA/reference mechanism rather than only gaming the
  recognizer.

The most defensible current conclusion is therefore: **restore alternation as
one controlled historical arm if needed, retain the protected simultaneous
loss as the main joint-training candidate, and test direct identity only as a
scheduled low-noise auxiliary.**
