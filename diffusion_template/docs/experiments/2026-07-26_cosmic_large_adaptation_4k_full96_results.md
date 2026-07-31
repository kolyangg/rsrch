# Full-Cosmic 4k and full-96 adaptation results

**Date:** 26 July 2026
**Branch:** `test`
**Current repository commit:** `c01a73bd3759a712284c535aa4fc60259237244c`
**Decision:** the original 4k inference policy fails, but the fixed-checkpoint
target-native face-K/V intervention at ratio 1.0 passes the full-96 gate. A
bounded 4k training-path ablation and a historical-training/target-native-
validation 20k run are in progress.

This report follows the experiment plan in
[Cosmic Large experiment summaries and launch scripts](../../analysis/2026-07-25_cosmic_large_experiment_launch_plan.md).
It covers the final speed-corrected 4k controls and their sealed 96-image
validations. Earlier Task A-D evidence is summarized in
[Cosmic Large Tasks A-D results handoff](../../analysis/2026-07-25_cosmic_large_tasks_a_d_results_handoff.md).

## Observed evidence

### Runtime correction

The slow Serv runs combined three sources of avoidable overhead:

- production jobs inherited `CUDA_LAUNCH_BLOCKING=1`;
- InsightFace used CPU ONNX Runtime during conditioning;
- the full-manifest loader was not using the verified worker configuration.

The speed-corrected Serv packages used asynchronous CUDA, ONNX Runtime 1.20.1
with `CUDAExecutionProvider`, and two training workers. The two complete 4k
runs sustained approximately 2.0–2.1 seconds per optimizer step, compared with
approximately 5–7 seconds per step in the original Serv launches. Both
speed-corrected runs completed in about 2h50m, including validation and
finalization.

These runtime changes did not change the architecture, dataset manifest,
reference policy, validation prompts, seeds, inference scheduler, guidance
scale, or metric definitions.

### Speed-corrected 4k endpoints

| Policy | Run and immutable Comet key | Runtime | Checkpoint SHA-256 | 12-image text / ID |
|---|---|---:|---|---:|
| Pose-first capped caption | `rhca_cosmic_full_crop20_posefirst_4k_fast_r3`, `7839bf5f50924f3ab2bb848fd97837e0` | 2h49m44s | `2ea4544d1ba621e8ca6169d15b8c5c402ee56ceb8478fe4e0b19cfe903e13177` | 26.9857 / 0.1362 |
| Legacy caption | `rhca_cosmic_full_crop20_legacy_4k_fast_r2`, `f2cd04577b014e6bb2b98fbea5d5472e` | 2h52m31s | `1cb098146be50e2f6a087e8c81b7284ab6ad5e940aa57dfd9014d0865a771b64` | 26.7135 / 0.1343 |

Each endpoint produced all 12 expected step-4000 images with no Comet export
warning. The Eddie-only endpoint panels were coherent, so the 96-image
identity expansion was required before promotion.

### Sealed full-96 validations

All four rows below used the canonical `cosmic_full96_auto_v1` protocol:
eight sorted identities by 12 prompts, seed 0, RealVisXL V4.0 validation base,
50 inference steps, CFG 5, PhotoMaker start step 10, branched-attention start
step 15, 95 automatic reference boxes and the one documented forced-manual
Jensen box.

| Endpoint | Machine | Comet key | Exact images | Text similarity | ID similarity | Visual promotion |
|---|---|---|---:|---:|---:|---|
| Legacy batched 4k | Neb | `7b793baa279849928eef75143dd86071` | 96/96 | 27.2051 | **0.3634** | Fail |
| Pose-first original-runtime 4k | Serv | `4512aa3eb65e4f0c942c1b055446d737` | 96/96 | 26.8545 | **0.3636** | Fail |
| Pose-first fast 4k | Serv | `7c80400b23ba4a1683d4b034abdbb12c` | 96/96 | 27.0207 | 0.3538 | Fail |
| Legacy fast 4k | Serv | `0de9a9858a784373a8871e6b667316e1` | 96/96 | **27.1722** | 0.3374 | Fail |

For both fast runs:

- the final record contains exactly 96 downloaded Comet images at optimizer
  step 4,000;
- the first 12-image batch reproduces the corresponding trainer-saved source
  endpoint;
- the checkpoint, bbox maps, static validation inputs, and generated pixel
  manifest are hashed in `saved/<run>/comet_experiment.json`;
- `FULL96_RESULT_VERIFIED`, `FULL96_RECORD_FINALIZED`, and
  `FULL96_EVAL_COMPLETE` are present;
- no OOM, traceback, or fatal validation error occurred.

The fast pose-first pixel manifest is
`5380b47fde8a8959eeec2bd899e984963ab0468e5baf1b5a820cddccb820574a`.
The fast legacy pixel manifest is
`c9b7160b0350faed4554c29e7c4b957e66850a8bbdd27f463e5aeaa5210def1c`.

### Visual gate

The required gate is per identity, not an aggregate metric: at least 11 of 12
images for every identity must have a coherent, attached face without pasted
reference fragments.

Both speed-corrected candidates fail on Jisoo:

- pose-first has six clear failures: Chef, Crying, Kickboxing, Laughing,
  Night-ride, and Skiing;
- legacy has at least seven clear failures: Angry, Chef, Crying, Kickboxing,
  Laughing, Night-ride, and Skiing. Drumming is also borderline because of the
  small displaced mouth fragment.

| Pose-first fast | Legacy fast |
|---|---|
| ![Pose-first Jisoo full-96 panel](assets/2026-07-26_cosmic_large_adaptation/pose_fast_jisoo_panel.png) | ![Legacy Jisoo full-96 panel](assets/2026-07-26_cosmic_large_adaptation/legacy_fast_jisoo_panel.png) |

The same identity-specific failure family appeared in the original-runtime
pose-first run and the Neb legacy-batched run. It is therefore not caused by
the throughput correction.

An offline per-identity diagnostic for the Neb legacy-batched endpoint gave
Jisoo ID similarity 0.2692 while several other identities ranged from 0.3761
to 0.5075. Even that lower identity score does not express the severity of the
localized pasted-face failures. Aggregate text and ID similarity cannot
replace the visual anatomy gate.

### Fixed-checkpoint reference-contamination intervention

The pose-adapt diagnostic reused the exact fast pose-first checkpoint,
canonical 96-image protocol, prompts, seeds, scheduler, inference steps, CFG,
PhotoMaker schedule, and bbox maps. The only changed value was the blend used
by branched self-attention in the face region:

- ratio 0.0 is the historical spatial reference K/V route;
- intermediate ratios blend reference K/V with target-native K/V;
- ratio 1.0 uses target-native face K/V while retaining PhotoMaker identity
  conditioning and the remaining trained pipeline.

| Pose-adapt ratio | Comet key | Exact images | Text similarity | ID similarity | Visual result |
|---:|---|---:|---:|---:|---|
| 0.35 | `af04ca4c24a041449a8a730c5b746976` | 96/96 | 27.0094 | 0.3615 | Fail; Jisoo improves, but residual identity-specific failures remain |
| 0.65 | `955067a0b30341209afcad70dd0224db` | 96/96 | 26.9725 | 0.4016 | Fail; Jisoo passes, but Jensen Kickboxing and Skiing remain malformed |
| 1.00 | `90c8297973a7456496200e9f8c042755` | 96/96 | **27.1979** | **0.4421** | **Pass; every identity reaches at least 11/12 coherent faces** |

All three records were finalized against the same checkpoint SHA-256
`2ea4544d1ba621e8ca6169d15b8c5c402ee56ceb8478fe4e0b19cfe903e13177`.
Each has all 96 expected images and a verified immutable Comet record, with no
fatal error or OOM.

The representative panels below show the monotonic removal of copied or
misregistered facial structure. For Jisoo, the ratio-0.35 Skiing image still
contains an oversized displaced face, while ratio 1.0 is coherent. For Jensen,
ratios 0.35 and 0.65 retain severe Kickboxing/Skiing corruption; ratio 1.0
removes both failure modes.

| Jisoo ratio comparison | Jensen ratio comparison |
|---|---|
| ![Jisoo pose-adapt ratio comparison](assets/2026-07-26_cosmic_large_adaptation/jisoo_ratio_comparison.png) | ![Jensen pose-adapt ratio comparison](assets/2026-07-26_cosmic_large_adaptation/jensen_ratio_comparison.png) |

This is causal evidence that spatial reference K/V is the primary source of
the observed pasted-face corruption at inference. It is not yet evidence that
training with ratio 1.0 is superior: the passing diagnostic changes only
checkpoint inference and effectively removes spatial reference K/V from the
target face branch.

### Promoted follow-up runs

Commit `c01a73bd3759a712284c535aa4fc60259237244c` makes the validation-only
ratio explicit and backward-compatible. Historical runs omit the override and
retain one shared ratio.

| Run | Machine | Policy | Immutable identity | Startup evidence |
|---|---|---|---|---|
| `rhca_cosmic_full_crop20_posefirst_par100_4k_r2` | Neb | train 1.0 / validate 1.0, 4k | Comet `e6cfd6b676ba474fad5f97824ec3d37d` | complete; checkpoint SHA `a62ece86…76c80`; 12/12 coherent at step 4000 |
| `rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k` | Serv | train 0.0 / validate 1.0, 20k | MLS `lm-mpi-job-4196f81e-ac14-47ff-9c81-416396f275e5`; Comet `326fa0e0ea82490abdde08eb1b94eff9` | step 4,000 passed: 12/12 coherent, text 29.5339, ID 0.2249; run continues |
| `rhca_cosmic_full_crop20_posefirst_par100_4k_r2_full96_par100` | Serv | fixed Neb endpoint, validate 1.0, 96 images | MLS `lm-mpi-job-42829870-ae62-42c0-b8b6-1f79d201db00`; Comet `762db6cfcf654dfd93cb45122bc0ceef` | complete; 96/96 exact Comet images, text 26.2822, ID 0.5136, and all identities 12/12 coherent |

The earlier Neb `r1` record, Comet
`9c2e8eca36184c498c78596bc66a2fe6`, failed before validation or training
because the new trainer option was not accepted by the constructor. It is
preserved as failed-start evidence and is not reused.

#### Preliminary matched step-500 gate

The Neb ratio-1.0 arm completed its first post-training validation with all 12
images, text similarity 28.3294, ID similarity 0.2387, and no malformed or
detached face. It therefore passes the documented early-stop gate and
continues.

An immutable-key Comet export at exact step 500 returned 12/12 images, both
metric points, no fallback, and no warnings or errors. All 12 downloaded file
hashes match the corresponding trainer-saved PNGs.

The Serv train-0.0/validate-1.0 arm then supplied the missing matched control:

| Step-500 policy | Text similarity | ID similarity | Eddie visual result |
|---|---:|---:|---|
| train 0.0 / validate 0.0 | 27.3346 | 0.1104 | Fail; Reading and Rushing have clear localized face loss/misregistration |
| train 0.0 / validate 1.0 | **28.6628** | 0.2109 | Pass; 12/12 coherent |
| train 1.0 / validate 1.0 | 28.3294 | **0.2387** | Pass; 12/12 coherent |

The Serv step-500 Comet export also returned the exact 12 images and both
metrics with no fallback, warning, or error; all 12 file hashes match the
trainer-saved PNGs.

![Matched step-500 three-way comparison](assets/2026-07-26_cosmic_large_adaptation/step500_threeway.png)

This three-way result strengthens the causal interpretation: switching only
validation from ratio 0.0 to 1.0 removes the obvious anatomy failures. Training
at ratio 1.0 adds a modest ID gain of 0.0279 versus train-0.0/validate-1.0 at
this early point, with a text decrease of 0.3333 and no obvious Eddie anatomy
change. The final 4k/full-96 comparison is still required before attributing a
general benefit to ratio-1.0 training.

The Neb train-1.0/validate-1.0 arm also passed its step-1000 gate. All 12
images are coherent on visual inspection, with no pasted or misregistered
face. Text similarity is 28.6940 and ID similarity is 0.2294. The epoch-2
checkpoint SHA-256 is
`8ce4ec3602e3922910bdbe3ff043cdcd0ca14d14a296c227f6e8862810e1905d`.
An exact-step export through immutable Comet key
`e6cfd6b676ba474fad5f97824ec3d37d` returned both metrics and 12/12 images
without fallback, warning, or error; every downloaded PNG hash matches its
trainer-saved counterpart. The run therefore continues to 4k.

The Serv train-0.0/validate-1.0 control passed the same step-1000 visual gate
with 12/12 coherent images. Its text similarity is 29.1055 and ID similarity
is 0.2105; checkpoint-epoch2 SHA-256 is
`fb9e2d6a1882a15054e2564a604bb14b38bd5999f8d4042e3a63a8820006d500`.
An exact-step export through immutable Comet key
`326fa0e0ea82490abdde08eb1b94eff9` returned both metrics and 12 images with
no fallback, warning, or error. All 12 Comet PNG hashes match the trainer
outputs.

| train 1.0 / validate 1.0 | train 0.0 / validate 1.0 |
|---|---|
| ![Neb ratio-1.0 step-1000 validation](assets/2026-07-26_cosmic_large_adaptation/step1000_par100_panel.png) | ![Serv split-policy step-1000 validation](assets/2026-07-26_cosmic_large_adaptation/step1000_train0_val100_panel.png) |

The step-1000 relationship is consistent with step 500. Training at ratio
1.0 has 0.0190 higher ID similarity, while historical ratio-0.0 training has
0.4115 higher text similarity; neither arm shows an anatomy failure in the
matched 12-image panel. This remains an intermediate result rather than a
promotion decision.

At the Neb arm's halfway point, step 2000 remains stable: 12/12 images are
coherent, text similarity is 29.3372, and ID similarity is 0.2199.
Checkpoint-epoch4 SHA-256 is
`6c44b95b739f332efea93d750ecdf8c7896de4e59476bc7d1255493f1fd74437`.
The immutable-key Comet export resolved exact step 2000 with no fallback,
warning, or error, and all 12 PNG hashes match the trainer outputs.

The matched Serv train-0.0/validate-1.0 control also passes step 2000 with
12/12 coherent images, text similarity 29.8307, and ID similarity 0.2027.
Checkpoint-epoch4 SHA-256 is
`59dc65c97607883ed71802bff116d3ef80f7617456cc148b6c8f93d342410ba1`.
Its immutable-key Comet export likewise resolved exact step 2000 without
fallback, warning, or error, and all 12 PNG hashes match the trainer outputs.

| train 1.0 / validate 1.0 | train 0.0 / validate 1.0 |
|---|---|
| ![Neb ratio-1.0 step-2000 validation](assets/2026-07-26_cosmic_large_adaptation/step2000_par100_panel.png) | ![Serv split-policy step-2000 validation](assets/2026-07-26_cosmic_large_adaptation/step2000_train0_val100_panel.png) |

The direction remains consistent with the two earlier matched gates: ratio-1.0
training is 0.0172 higher in ID similarity, while historical ratio-0.0
training is 0.4935 higher in text similarity. The declining control ID metric
has not yet produced an obvious anatomy failure in the matched Eddie panel.

#### Serv train-0/validate-1 step-4,000 gate

The active long control reached step 4,000 without OOM or fatal error and
continued training. Checkpoint `checkpoint-epoch8.pth` has SHA-256
`bf50d8a1b205f6c59cddf156302245bcd34e4fb9aebb965ab550a08adda2a85f`.
An immutable-key Comet export for
`326fa0e0ea82490abdde08eb1b94eff9` resolved the exact requested step with no
fallback, warning, or error. All 12 downloaded PNG hashes match the
trainer-saved images.

The step-4,000 metrics are text similarity `29.533854166666668` and ID
similarity `0.22486064955592155`. Visual review found 12/12 coherent, attached
faces and no return of the reference-fragment failure.

![Serv train-0/validate-1 step-4000 panel](assets/2026-07-26_cosmic_large_adaptation/control_step4000_train0_val100/eddie_panel.png)

#### Neb ratio-1.0 endpoint

The Neb arm completed 4,000 optimizer steps with no OOM or fatal error. Its
step-4000 panel is 12/12 coherent, text similarity is 27.9870, and ID
similarity is 0.2185. Checkpoint-epoch8 SHA-256 is
`a62ece86f7da072863344dd7d7011bc8805222fbb3f45d99cd179f6a03976c80`.
The immutable-key Comet export resolved exact step 4000 with no fallback,
warning, or error; all 12 exported PNG hashes match the trainer outputs.

![Neb ratio-1.0 step-4000 validation](assets/2026-07-26_cosmic_large_adaptation/step4000_par100_panel.png)

The first full-96 submission, MLS
`lm-mpi-job-46414d9f-7357-4b9a-9678-9f40a83fd1cf`, failed before creating a
run or Comet record. The prerequisite checker incorrectly required the
already-finalized ratio-1.0 control to reproduce its ratio-0 source pixels,
although a fixed-checkpoint intervention must change those pixels. Commit
`3028064f9709ccb5c631c201ea760bf25e275849` makes exact reproduction the
backward-compatible default and adds an explicit, provenance-checked
intervention mode. The failed logs are archived on Serv, and the fresh retry
listed above started with a clean saved directory and new immutable Comet key.

#### Target-native-training full-96 result

The retry completed successfully in MLS job
`lm-mpi-job-42829870-ae62-42c0-b8b6-1f79d201db00`. The finalized record has:

- immutable Comet key `762db6cfcf654dfd93cb45122bc0ceef`;
- 96/96 images at optimizer step 4,000 across all eight identity batches;
- text similarity `26.2822265625` and ID similarity
  `0.5135621405206621`;
- source checkpoint SHA-256
  `a62ece86f7da072863344dd7d7011bc8805222fbb3f45d99cd179f6a03976c80`;
- pixel-manifest SHA-256
  `8c9a64d3927db73568d0fc46992aa251ab402635608a8b589a5f11003bddb0d3`;
- `FULL96_RESULT_VERIFIED`, `FULL96_RECORD_FINALIZED`, and
  `POSE_ADAPT_FULL96_COMPLETE` markers with no fatal error or OOM.

Visual review of all 96 exact PNGs found coherent, attached faces in 12/12
images for Eddie, Elon, Jennie, Jensen, Jisoo, Keanu, Lex, and Marion. The
minimum per-identity score is therefore 12/12, exceeding the required 11/12
gate.

| Eddie | Elon | Jennie | Jensen |
|---|---|---|---|
| ![Eddie target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/eddie_panel.png) | ![Elon target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/elon_panel.png) | ![Jennie target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/jennie_panel.png) | ![Jensen target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/jensen_panel.png) |

| Jisoo | Keanu | Lex | Marion |
|---|---|---|---|
| ![Jisoo target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/jisoo_panel.png) | ![Keanu target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/keanu_panel.png) | ![Lex target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/lex_panel.png) | ![Marion target-native-training full-96 panel](assets/2026-07-26_cosmic_large_adaptation/par100_train_full96/marion_panel.png) |

Compared with the fixed-checkpoint train-0/validate-1 control at ratio 1.0,
target-native training raises ID similarity from `0.4421` to `0.5136`
(`+0.0715`) while lowering text similarity from `27.1979` to `26.2822`
(`-0.9157`). Both pass the anatomy gate. This is controlled evidence that
training without spatial reference face K/V materially strengthens identity
retention, with a measurable prompt/text trade-off at 4,000 steps.

## Hypotheses

The following interpretations are plausible but are not established as facts:

1. The recurring Jisoo artifacts resemble content in `jisoo.webp`: dark hair
   crossing the face boundary, earrings, and a hand near the face. The
   fixed-checkpoint ratio sweep now strongly supports spatial reference K/V as
   the causal source, although the exact layer- and timestep-level mechanism
   remains unproven.
2. Because the failure repeats across caption policies, runtimes, and
   checkpoints, caption order and loader throughput are unlikely to be the
   primary cause.
3. The 20%-margin/256px training crop is useful in the controlled Task D
   comparison, but it is not sufficient to make every heterogeneous reference
   safe. Reference quality and spatial-reference routing need their own
   controls.
4. The strong aggregate ID scores may partly reward precisely the reference
   copying that produces bad anatomy, so optimizing only that metric could
   promote the wrong model.

## Conclusion and next decision

The throughput fix is successful and should remain the default Serv runtime.
The current full-Cosmic data path also passes its mechanical contract:
22,140 accepted records, deterministic reference transforms, valid bbox
propagation, target/reference inequality checks, and reproducible full-96
validation.

The original training recipe at its historical inference ratio is not suitable
for promotion: caption changes do not remove the reference-fragment failure.
Ratio 1.0 clears that visual failure by setting the spatial reference-face
contribution to zero. It is therefore a causal ablation of the failing route,
not an eligible implementation of reference-conditioned branched attention.

### Plain-PhotoMaker control for the ratio-1 endpoint

An exact 12-image control disabled branched attention on the ratio-1
step-4,000 checkpoint while holding the validation base, prompts, references,
seeds, scheduler, denoising steps, guidance, and PhotoMaker weights fixed.
The run is Comet `584eaec6c1fa4436a42a83477875d0bc`.

Observed evidence:

- all 12 filenames matched and `0/12` decoded images were pixel-equal;
- mean pixel MAE was `10.8996/255`, mean RMSE was `19.3240/255`, and mean PSNR
  was `22.7218 dB`, so the remaining BA processors do change the image;
- plain PhotoMaker produced 12/12 coherent faces, text similarity `28.40625`,
  and ID similarity `0.2214457`;
- the ratio-1 BA endpoint produced 12/12 coherent faces, text similarity
  `27.98698`, and ID similarity `0.21853`.

Thus the ratio-1 BA-specific processing is materially different in pixels but
does not outperform plain PhotoMaker in this matched panel. This confirms that
the earlier healthy-image gate was insufficient: it demonstrated removal of
reference contamination, not useful BA contribution.

On 26 July 2026, the active ratio-1/ratio-1 long run and the
train-0/validate-1 long run were both stopped. All subsequent experiments are
required to compose with `use_branched_attention=true`,
`pose_adapt_ratio=0.0`, and `ca_mixing_for_face=false`.

The next valid experiment family changes only reference formatting while
retaining 100% reference-face K/V: wider 40% and 60% context, a matched legacy
caption control, and a 512px reference-content control. Any winning 4k
checkpoint must pass a new canonical full-96 validation with the same
ratio-zero/CA-mixing-off invariant before longer training.
