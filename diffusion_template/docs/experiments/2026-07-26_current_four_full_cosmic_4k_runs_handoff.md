# Four full-Cosmic 4k runs: final full-96 handoff

Initial snapshot: **26 July 2026, 10:38 UTC / 11:38 Europe/London**

Full-96 update: **26 July 2026, 14:51 UTC / 15:51 Europe/London**

This handoff covers the four completed full-Cosmic experiments. It
supplements the broader
[ratio-zero reference-policy handoff](2026-07-26_ba_ratio_zero_reference_policy_runs_handoff.md)
and the
[4k/full-96 results report](2026-07-26_cosmic_large_adaptation_4k_full96_results.md).

## Executive status

- All four training jobs completed successfully on Serv at 4,000 steps. The
  matched multistep full-96 evaluations also generated all 480 expected images
  per arm: 96 each at steps 0, 1,000, 2,000, 3,000, and 4,000.
- All use the full 22,140-record accepted Cosmic training manifest and target
  4,000 optimizer steps.
- Training is healthy at approximately 2.06--2.10 seconds per step. There are
  no OOMs, tracebacks, or fatal runtime errors.
- All four full-96 endpoints retain globally coherent scenes and avoid the
  earlier catastrophic displaced or pasted-face failure.
- The best matched identity result is the 40%-margin, 256px, pose-first arm at
  step 3,000 (`0.3606`). Every arm peaks on identity at step 3,000 and declines
  by step 4,000.
- The 60% crop and 512px reference controls do not improve the trajectory.
  Legacy captions improve the text metric slightly but trade away identity.
- None of the arms fixes the identity-specific Jisoo face-corruption cluster.
  The recommended checkpoint from this matrix is therefore the 40% / 256px /
  pose-first step-3,000 checkpoint, with an explicit unresolved Jisoo gate.

## Shared architecture and data contract

All runs use branch `test` at commit
`cfa4bffebfbb46e324a7b503bdbfd786bea5e6e6`.

The immutable run records resolve:

- `use_branched_attention: true`;
- `pose_adapt_ratio: 0.0`;
- `ca_mixing_for_face: false`;
- `reference_face_kv_weight: 1.0`;
- `disable_branched_sa: false`;
- `disable_branched_ca: true`;
- `update_proc_weights_val: true`;
- alternate-base validation on `SG161222/RealVisXL_V4.0`;
- ONNX Runtime 1.20.1 with `CUDAExecutionProvider`;
- two training DataLoader workers.

The architecture startup audit reported 840/840 branched-attention processor
parameters in the optimizer. Because `disable_branched_ca` is true, these are
reference-conditioned **branched self-attention** experiments; the branched
cross-attention processor path is not active. This distinction must be
preserved in later interpretation: the matrix tests reference formatting,
caption policy, and resolution inside the current self-attention BA protocol,
not a full SA+CA factorial.

## What each experiment is trying to learn

| Arm | Experimental question | Serv job | Immutable Comet key |
|---|---|---|---|
| 40% margin, 256px, pose-first | Does the favorable one-ID 40%-margin result transfer to full Cosmic? This is the matrix baseline. | `lm-mpi-job-c31986ac-283b-4f4f-9d15-aef54890fc54` | `1a19fdf2793f413c9336379d3628874d` |
| 60% margin, 256px, pose-first | Does more real hair/head/shoulder/scene context improve face/body registration beyond 40%, or dilute identity? | `lm-mpi-job-86e0060d-c2a6-4f9d-a6fd-2ee8c0036285` | `a96bcbae3d2b4698a43d7ec80457586c` |
| 40% margin, 256px, legacy captions | At a fixed 40% crop, are results driven by the pose-first caption rewrite or by reference formatting itself? | `lm-mpi-job-a4a2cdb3-ffc5-45ec-9cb6-e0536494ab43` | `92572589d6594cd59749577fc51f5bba` |
| 40% margin, 512px, pose-first | At fixed margin and captions, does retaining twice the reference content resolution improve usable identity evidence? | `lm-mpi-job-29283861-49e2-4ee3-8572-d8ba3970fa31` | `c354369af45b4c9da84f1124cf3e9a88` |

The repository experiment plans are:

- [`rhca_cosmic_full_crop40_posefirst_4k_fast_r1.json`](../../experiment_specs/rhca_cosmic_full_crop40_posefirst_4k_fast_r1.json)
- [`rhca_cosmic_full_crop60_posefirst_4k_fast_r1.json`](../../experiment_specs/rhca_cosmic_full_crop60_posefirst_4k_fast_r1.json)
- [`rhca_cosmic_full_crop40_legacy_4k_fast_r1.json`](../../experiment_specs/rhca_cosmic_full_crop40_legacy_4k_fast_r1.json)
- [`rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1.json`](../../experiment_specs/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1.json)

The corresponding Serv launch packages are:

- [`rhca_cosmic_full_crop40_posefirst_4k_fast_r1/`](../../serv_run_packages/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/)
- [`rhca_cosmic_full_crop60_posefirst_4k_fast_r1/`](../../serv_run_packages/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/)
- [`rhca_cosmic_full_crop40_legacy_4k_fast_r1/`](../../serv_run_packages/rhca_cosmic_full_crop40_legacy_4k_fast_r1/)
- [`rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/`](../../serv_run_packages/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/)

## Interim 12-image progress and metrics

Metrics below come from the fixed 12-image validation set. They are suitable
for matched interim comparison among these four arms, but must not be compared
directly with historical 96-image aggregate metrics.

| Arm | Completed checkpoint at snapshot | Current training position | Step-2,000 text | Step-2,000 ID | Latest completed text / ID |
|---|---:|---:|---:|---:|---:|
| 40% / 256 / pose-first | 2,500 | approximately 2,550 | 27.2669 | 0.1437 | step 2,500: 27.2643 / 0.1623 |
| 60% / 256 / pose-first | 2,500 | approximately 2,550 | 27.1263 | 0.1402 | step 2,500: 27.6940 / 0.1536 |
| 40% / 256 / legacy | 2,500 | approximately 2,550 | **27.4297** | **0.1493** | step 2,500: 27.0977 / 0.1614 |
| 40% / 512 / pose-first | 2,000 | approximately 2,350 | 27.1081 | 0.1427 | step 2,000: 27.1081 / 0.1427 |

At the matched step-2,000 gate:

- identity spans only 0.1402--0.1493;
- text similarity spans only 27.1081--27.4297;
- all 48 expected PNGs exist, 12 per arm;
- all four panels retain the requested scenes and usable face/body
  registration;
- some local facial weaknesses remain, especially ski-goggle geometry,
  distant/small faces, and over-smoothed or stretched expressions;
- no image pair is pixel-identical across arms, but matched pairwise mean
  absolute differences are only 5.8--7.5/255, consistent with visible but
  modest changes rather than a qualitatively different regime.

The step-2,500 metrics show identity continuing to rise in the three completed
arms, but the ordering is unstable: crop-40 pose-first and legacy are almost
tied on ID, while crop-60 has the highest current text score. This is
preliminary evidence, not a promotion result.

## Validation warning and interpretation boundary

Every alternate-base validation emits:

```text
[PhotomakerBranchedLora] exception while installing branched processors:
'AttnProcessor2_0' object has no attribute 'parameters'
```

Code inspection shows that the self-attention BA processors are installed
before this warning. The warning occurs afterward when
`install_branched_processors_for_training()` iterates over the intentionally
unbranched cross-attention processors and assumes every processor is an
`nn.Module`. `BaseTrainer` then loads and explicitly copies the trained
processor weights into the temporary validation model. The warning therefore
does not by itself invalidate the saved self-attention BA panels, but the
catch-all should be removed or narrowed in future code so a real partial
installation failure cannot be hidden. Endpoint/full-96 evaluation should
also assert the exact installed processor count.

## Interim conclusion (superseded by the full-96 result)

The current evidence supports the basic direction: replacing tight or padded
Cosmic references with real surrounding context avoids the catastrophic
face-fragment behavior on this 12-image identity gate. It does **not** yet show
whether 40% or 60% context is better, whether pose-first captions help, or
whether 512px reference content earns its extra cost. The 512 arm has not
shown an interim metric advantage.

Do not promote any arm from these interim metrics. The meaningful decision
requires:

1. checkpoint epoch 8 / step 4,000 and 12 endpoint images for every arm;
2. matched visual review of anatomy, identity, prompt adherence, and
   face/body alignment;
3. Comet/image integrity checks by immutable key;
4. canonical 96-image validation for any candidate intended for promotion.

At the observed throughput, the first three jobs should reach 4k in roughly
one hour and the 512 arm in approximately 1.1--1.3 hours, assuming validation
and checkpoint overhead remain stable.

## Final multistep full-96 evaluation

### Evaluation contract and provenance

The final comparison uses validation code commit
`c04970f342a186d1092f07f9a08d7d8a797383e8`. Each validation run:

- evaluates the seeded initial state and checkpoints at steps 1,000, 2,000,
  3,000, and 4,000;
- uses the sealed `cosmic_full96_auto_v1` bbox protocol, fixed prompts, seeds,
  references, RealVisXL validation base, and batch size 12;
- produces eight 12-image batches, or 96 images per step and 480 images per
  arm;
- records all images and aggregate metrics at exact optimizer steps in one
  immutable Comet experiment; and
- reproduces the source trainer's first 12 step-4,000 images before accepting
  the result.

| Arm | Validation MLS job | Validation Comet key | Local immutable record |
|---|---|---|---|
| 40% / 256 / pose-first | `lm-mpi-job-60be615a-eeaa-460e-86ce-360b7edccbb5` | [`519f9ecac929417e8073e7b3cc953c2d`](https://www.comet.com/nikolay-2104/rsrch-jul/519f9ecac929417e8073e7b3cc953c2d) | [`comet_records/...crop40_posefirst...json`](../../comet_records/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 60% / 256 / pose-first | `lm-mpi-job-4e972ca2-019a-4013-a2c9-e2d840c45647` | [`df99f4b0bb9a4676bd6783d1bc611c6b`](https://www.comet.com/nikolay-2104/rsrch-jul/df99f4b0bb9a4676bd6783d1bc611c6b) | [`comet_records/...crop60_posefirst...json`](../../comet_records/rhca_cosmic_full_crop60_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 40% / 256 / legacy | `lm-mpi-job-6e454f41-4ac6-4eea-b7f8-81fa63550402` | [`dfb06576f4104d969b08c59b06ec7834`](https://www.comet.com/nikolay-2104/rsrch-jul/dfb06576f4104d969b08c59b06ec7834) | [`comet_records/...crop40_legacy...json`](../../comet_records/rhca_cosmic_full_crop40_legacy_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 40% / 512 / pose-first | `lm-mpi-job-6713df92-3d79-4289-9cd8-7873c827acf8` | [`00cfd945fdcf44dbbd8914b42f139300`](https://www.comet.com/nikolay-2104/rsrch-jul/00cfd945fdcf44dbbd8914b42f139300) | [`comet_records/...crop40_512_posefirst...json`](../../comet_records/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |

Observed integrity evidence:

- all four runs contain exactly 96 PNGs at each of the five steps;
- the 96 step-0 PNGs are byte-identical across all four arms;
- the Comet metrics use exact steps rather than nearest-step fallback;
- no OOM, traceback, or global image collapse occurred; and
- the source training and validation records preserve
  `use_branched_attention=true`, `pose_adapt_ratio=0.0`,
  `ca_mixing_for_face=false`, active branched self-attention, and disabled
  branched cross-attention.

### Full-96 metrics

Each cell is `text similarity / identity similarity`. Bold marks the best
identity score at a matched step.

| Step | 40% / 256 / pose-first | 60% / 256 / pose-first | 40% / 256 / legacy | 40% / 512 / pose-first |
|---:|---:|---:|---:|---:|
| 0 | 26.3205 / 0.2999 | 26.3205 / 0.2999 | 26.3205 / 0.2999 | 26.3205 / 0.2999 |
| 1,000 | 27.1279 / **0.2972** | 27.0369 / 0.2947 | **27.2619** / 0.2961 | 27.0129 / 0.2872 |
| 2,000 | 26.8722 / **0.3465** | 27.0072 / 0.3423 | **27.1172** / 0.3353 | 26.9036 / 0.3390 |
| 3,000 | 26.6846 / **0.3606** | 26.7720 / 0.3575 | **27.0054** / 0.3457 | 26.7827 / 0.3545 |
| 4,000 | 26.9992 / 0.3422 | 26.8936 / **0.3458** | **27.1810** / 0.3316 | 26.9494 / 0.3418 |

Observed metric conclusions:

1. The first 1,000 steps do not improve aggregate identity. The useful
   identity gain appears between 1,000 and 2,000 steps.
2. Every arm peaks on identity at step 3,000 and declines at step 4,000.
   Training to 4,000 is therefore past the best checkpoint for this matrix.
3. The 40% / 256px pose-first arm is best at both 2,000 and 3,000 and reaches
   the overall maximum, `0.3606`.
4. The 60% arm's small step-4,000 lead (`0.3458` versus `0.3422`) does not
   compensate for trailing the 40% arm at the higher-quality 2,000 and 3,000
   checkpoints.
5. The 512px arm never beats the otherwise matched 256px arm. At step 3,000
   it scores `0.3545` versus `0.3606`; at step 4,000 the two are effectively
   tied.
6. Legacy captions give the strongest text score at every trained checkpoint,
   but from step 2,000 onward they give the weakest identity score. The visual
   panels do not show a corresponding qualitative composition advantage.

The arms remain close in pixel space. Against the 40% / 256px pose-first arm,
matched 128px-thumbnail mean absolute differences after training are
approximately 3.9--4.4/255 for the 60% arm, 4.6--5.0/255 for the 512px arm,
and 4.9--5.6/255 for legacy captions. This is descriptive evidence that the
controls alter details rather than creating a new generation regime; it is
not a replacement for face-quality review.

### Visual evidence

The 40% / 256px pose-first trajectory shows the shared temporal pattern:

- [step 0 contact sheet](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step0.jpg)
- [step 2,000 contact sheet](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step2000.jpg)
- [step 3,000 contact sheet](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step3000.jpg)
- [step 4,000 contact sheet](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step4000.jpg)

Matched endpoint sheets:

- [60% / 256px / pose-first, step 4,000](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop60_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step4000.jpg)
- [40% / 256px / legacy, step 4,000](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop40_legacy_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step4000.jpg)
- [40% / 512px / pose-first, step 4,000](assets/2026-07-26_full96_multistep/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k__step4000.jpg)

Observed visual findings:

- All four arms preserve the requested scenes and global body layouts across
  the 96-image endpoint. None returns to the catastrophic pasted, displaced,
  or oversized face fragments seen with tight Cosmic references.
- From step 0 to steps 2,000--3,000, faces generally become more adult,
  sharper, and more identity-specific while prompt composition stays stable.
- Step 4,000 is not visibly better than step 3,000 and sometimes looks
  slightly harsher or less natural, consistent with the identity-metric drop.
- Differences among 40%, 60%, and 512px pose-first panels are subtle. The
  larger crop and higher resolution do not visibly repair failure cases.
- Jisoo is a reproducible outlier in every arm. At least five prompts per
  endpoint have unmistakable dark, duplicated, mask-like, or animal-like
  facial structures, especially angry, kickboxing, laughing, night-ride, and
  skiing. The clean frontal
  [Jisoo reference](assets/2026-07-26_full96_multistep/references/jisoo.webp)
  rules out an obviously malformed source photograph.
- Marion retains frequent hair-over-face or soft-face results, and skiing or
  boxing remain locally difficult for several identities. These are local
  face-quality issues, not global scene failures.

### Interpretation

Observed evidence supports the following:

- Real surrounding context in Cosmic references is compatible with stable BA
  training on the full 22,140-record manifest.
- A 40% crop with 256px reference content is sufficient; 60% context and
  512px content do not add measurable value here.
- Pose-first captions provide a better identity/text trade-off than legacy
  captions.
- A 3,000-step stopping point is better than the standardized 4,000-step
  endpoint for every arm in this matrix.

The following remains a hypothesis:

- Because Jisoo fails in the shared step-0 baseline and remains bad across all
  training controls, its failure is more likely an identity/reference,
  PhotoMaker, bbox-routing, or face-conditioning interaction than a crop
  margin or caption-policy effect. This matrix does not isolate those causes.

### Final decision and next action

Promote **40% margin, 256px content, pose-first captions at step 3,000** as the
best candidate from this matrix. Preserve the 4,000-step checkpoint for
reproducibility, but do not use it as the default endpoint.

Do not spend a long run on either 60% context or 512px reference content based
on these results. Before a long full-Cosmic run, add a small identity-specific
gate that:

1. evaluates Jisoo with the existing reference, a second clean reference, and
   a verified face bbox;
2. compares PhotoMaker-only, seeded step 0, and the promoted step-3,000
   checkpoint on the same 12 prompts; and
3. fails promotion if the mask-like face artifacts remain.

This recommendation is limited to the current branched-self-attention
protocol. Branched cross-attention is disabled in all four runs, so the matrix
does not establish the behavior of a combined SA+CA design.
