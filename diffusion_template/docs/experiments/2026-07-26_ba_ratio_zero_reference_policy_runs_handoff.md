# BA ratio-zero reference-policy runs handoff

Date: 26 July 2026

Branch: `test`

Deployed commit: `cfa4bffebfbb46e324a7b503bdbfd786bea5e6e6`

## Executive status

Monitoring by the current agent is intentionally finished at the user's
request. One Neb run and four Serv runs are live. The four Serv allocations
have MLS status `running`; each has passed its CUDA-provider and dataset
preflight checks, created an immutable Comet record, and entered model
initialization or step-0 validation. The Neb run is in real training at about
1.22 seconds per optimizer step.

Every live run preserves the required branched-attention design:

- `use_branched_attention=true`
- `pipeline.pose_adapt_ratio=0.0`
- `pipeline.ca_mixing_for_face=false`
- spatial reference face features remain available to the branched
  self-attention K/V path

No live run uses the rejected pose-adaptation or CA-mixing ablations.

## Experiment history and architectural reset

This section records the experiments that led to the live matrix. The linked
JSON files are the local immutable plans/packages; each actual run also wrote
its immutable runtime identity to
`saved/<run_name>/comet_experiment.json`. Comet keys below were read from
those runtime records, not inferred later from display names.

### Historical ratio-zero full-Cosmic controls

The first controlled full-Cosmic endpoints retained the intended spatial
reference-face K/V route (`pose_adapt_ratio=0`) and differed mainly in caption
policy. The CUDA-fast reruns confirmed that the throughput correction did not
cause the quality failure.

| Experiment | JSON plan | Comet key | Observed result |
|---|---|---|---|
| 20% crop, 256px, pose-first, 4k | [`rhca_cosmic_full_crop20_posefirst_4k_fast_r3.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_full_crop20_posefirst_4k_fast_r3.json) | `7839bf5f50924f3ab2bb848fd97837e0` | Completed in 2h49m44s; 12-image endpoint text 26.9857, ID 0.1362 |
| 20% crop, 256px, legacy captions, 4k | [`rhca_cosmic_full_crop20_legacy_4k_fast_r2.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_full_crop20_legacy_4k_fast_r2.json) | `f2cd04577b014e6bb2b98fbea5d5472e` | Completed in 2h52m31s; 12-image endpoint text 26.7135, ID 0.1343 |
| Pose-first endpoint, canonical full-96 | [`rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_eval.json`](../../serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_eval/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_eval.json) | `7c80400b23ba4a1683d4b034abdbb12c` | 96/96 images; text 27.0207, ID 0.3538; **failed** the visual gate because Jisoo had six clear malformed or misregistered faces |
| Legacy endpoint, canonical full-96 | [`rhca_cosmic_full_crop20_legacy_4k_fast_r2_full96_eval.json`](../../serv_run_packages/rhca_cosmic_full_crop20_legacy_4k_fast_r2_full96_eval/rhca_cosmic_full_crop20_legacy_4k_fast_r2_full96_eval.json) | `0de9a9858a784373a8871e6b667316e1` | 96/96 images; text 27.1722, ID 0.3374; **failed** the visual gate with at least seven clear Jisoo failures |

The same identity-specific pasted-face failure family appeared in earlier
runtime and batching variants. Therefore neither caption order nor the
throughput fix explains the failure. The aggregate ID metric also understated
the visible anatomy problem.

### Fixed-checkpoint target-native face-K/V sweep

The next diagnostic held the pose-first 4k checkpoint, all 96 validation
inputs, prompts, seeds, scheduler, CFG, bbox maps, and PhotoMaker settings
fixed. It changed only `pose_adapt_ratio`, progressively replacing spatial
reference-face K/V with target-native face K/V.

| Ratio | JSON plan | Comet key | Observed result |
|---:|---|---|---|
| 0.35 | [`full96_par35.json`](../../serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par35/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par35.json) | `af04ca4c24a041449a8a730c5b746976` | 96/96; text 27.0094, ID 0.3615; failed because residual identity-specific fragments remained |
| 0.65 | [`full96_par65.json`](../../serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par65/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par65.json) | `955067a0b30341209afcad70dd0224db` | 96/96; text 26.9725, ID 0.4016; Jisoo passed, but Jensen Kickboxing and Skiing remained malformed |
| 1.00 | [`full96_par100.json`](../../serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par100/rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par100.json) | `90c8297973a7456496200e9f8c042755` | 96/96; text 27.1979, ID 0.4421; every identity reached at least 11/12 coherent faces |

This sweep was valuable causal evidence: spatial reference-face K/V was the
primary source of the pasted or displaced facial structure. However, ratio
1.0 gives that source a weight of zero. It is therefore an ablation of the
reference-conditioned BA route, not a valid promotion of that route.

### Ratio-1 training and the drift toward plain PhotoMaker

A bounded training-path experiment then trained and validated with ratio 1.0:

| Experiment | JSON plan | Comet key | Observed result |
|---|---|---|---|
| Train 1.0 / validate 1.0, 4k | [`rhca_cosmic_full_crop20_posefirst_par100_4k_r2.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_full_crop20_posefirst_par100_4k_r2.json) | `e6cfd6b676ba474fad5f97824ec3d37d` | Completed; 12/12 coherent endpoint images, text 27.9870, ID 0.2185, checkpoint SHA `a62ece86f7da072863344dd7d7011bc8805222fbb3f45d99cd179f6a03976c80` |
| Ratio-1 trained endpoint, full-96 | [`rhca_cosmic_full_crop20_posefirst_par100_4k_r2_full96_par100.json`](../../serv_run_packages/rhca_cosmic_full_crop20_posefirst_par100_4k_r2_full96_par100/rhca_cosmic_full_crop20_posefirst_par100_4k_r2_full96_par100.json) | `762db6cfcf654dfd93cb45122bc0ceef` | 96/96; text 26.2822, ID 0.5136; all eight identities were 12/12 coherent |
| Train 0.0 / validate 1.0 long control | [`rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k.json`](../../serv_run_packages/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k/rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k.json) | `326fa0e0ea82490abdde08eb1b94eff9` | Reached the 4k gate with 12/12 coherent images, text 29.5339, ID 0.2249; later stopped because ratio-1 validation was not architecturally eligible |

These results looked strong if judged only by coherent faces and aggregate ID.
They nevertheless represented experimental drift toward plain PhotoMaker:
with `pose_adapt_ratio=1`, the target face no longer receives spatial identity
features through the reference-face K/V path that the project is intended to
study.

To test the degree of drift directly, a matched 12-image control disabled
branched attention entirely while preserving the ratio-1 checkpoint,
PhotoMaker weights, prompts, references, seeds, validation base, scheduler,
steps, and CFG:

| Experiment | JSON plan | Comet key | Observed result |
|---|---|---|---|
| Plain PhotoMaker matched control | [`plain_pm_step4000_12_r2.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_full_crop20_posefirst_par100_4k_r2_plain_pm_step4000_12_r2.json) | `584eaec6c1fa4436a42a83477875d0bc` | 12/12 coherent; text 28.4063, ID 0.2214 |
| Ratio-1 BA source panel | [`par100_4k_r2.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_full_crop20_posefirst_par100_4k_r2.json) | `e6cfd6b676ba474fad5f97824ec3d37d` | 12/12 coherent; text 27.9870, ID 0.2185 |

The panels were not pixel-identical (`0/12` equal; mean MAE
`10.8996/255`), so ratio-1 BA was not literally the same computation as plain
PhotoMaker. But plain PhotoMaker was slightly better on both matched metrics
and equally coherent. The evidence therefore did **not** show an effective
reference-conditioned BA contribution; it showed that removing the problematic
reference-face K/V route restored healthy generation.

The experiment program was reset at this point:

- active runs with `pose_adapt_ratio>0` were stopped;
- active or planned CA-mixing runs were rejected;
- `AGENTS.md` and active launchers were pinned to
  `use_branched_attention=true`, `pose_adapt_ratio=0.0`, and
  `ca_mixing_for_face=false`;
- new experiments were redirected to reference formatting, resolution, and
  caption controls while preserving 100% reference-face K/V.

### Ratio-zero reference-policy experiments after the reset

The controlled one-ID gates tested ways to retain reference identity while
reducing unsafe spatial copying:

| Experiment | JSON plan | Comet key | Observed result |
|---|---|---|---|
| 40% margin, 256px | [`rhca_cosmic_oneid_margin40_4k_r1.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_oneid_margin40_4k_r1.json) | `9a947bd85a7745e29ddf329b9be16763` | Completed; text 26.7409, ID 0.3076; mostly coherent, materially better than the canvas arm |
| Exact 256px crop centered on a 1024px canvas | [`rhca_cosmic_oneid_canvas1024_4k.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_oneid_canvas1024_4k.json) | `f03960bfb34a49bdba6e1503aafaf130` | Completed; text 26.2995, ID 0.1377; catastrophic displaced eyes/mouth or blank faces in about 10/12 images |
| 60% margin, 256px | [`rhca_cosmic_oneid_margin60_4k.json`](../../experiments/cosmic_large_adaptation/rhca_cosmic_oneid_margin60_4k.json) | `b2ef6ed73f164961b111e6c78c742eab` | Live at handoff; healthy real-training throughput around 1.22 seconds/step |

The canvas result rejects “reduce face occupancy by padding” as a safe
solution. The 40%-margin result supports adding real surrounding context,
which motivated the live full-Cosmic 40%/60%/caption/resolution matrix
documented below.

## Why the Serv runs were restarted

The first four Serv submissions used generic packages that did not inject the
known-good ONNX Runtime CUDA overlay. Their logs showed
`libonnxruntime_providers_cuda.so` failing to load because
`libcublasLt.so.11` was missing. One training loop stabilized near
4.0 seconds/step, and the other jobs were spending several minutes in
validation. All four were stopped before meaningful training:

| Superseded run | Comet key | MLS job |
|---|---|---|
| `rhca_cosmic_full_crop40_posefirst_4k` | `93fa7025e4ba44f38dc52cf2d1a2344d` | `lm-mpi-job-0552e075-0f2b-4c95-a99d-622d0f7124e1` |
| `rhca_cosmic_full_crop60_posefirst_4k` | `55724ec671fa422a823169380b9216bb` | `lm-mpi-job-388e4a8d-5cdf-44c1-b49b-4497dc2a0784` |
| `rhca_cosmic_full_crop40_legacy_4k` | `e60fcabe54d543299a21363f90fbf030` | `lm-mpi-job-fefbea7e-a382-4481-968b-9943ca38b007` |
| `rhca_cosmic_full_crop40_512_posefirst_4k` | `a3336edfbc434cd8a19dd091632e175c` | `lm-mpi-job-d39b6d6a-5c9c-449d-8a6d-392041108cf0` |

The replacements use ONNX Runtime 1.20.1 from the Serv-owned runtime overlay,
add the matching cuDNN and cuBLAS libraries, require
`CUDAExecutionProvider` before training, reject `CUDA_LAUNCH_BLOCKING`, and
set `dataloaders.train.num_workers=2`. A failure in this contract now stops a
job rather than silently falling back to CPU.

## Live Neb run

### 60%-margin one-ID gate

- Run: `rhca_cosmic_oneid_margin60_4k`
- PID/PGID: `4043420`
- Comet key: `b2ef6ed73f164961b111e6c78c742eab`
- Log: `/home/niko/rsrch/diffusion_template/logs/rhca_cosmic_oneid_margin60_4k.log`
- Output: `/home/niko/rsrch/diffusion_template/saved/rhca_cosmic_oneid_margin60_4k`
- Policy: full-scene face crop with 60% margin per side, 256-pixel content,
  no canvas, ratio-zero BA, no CA mixing
- Last observed state: real training, approximately 1.22 seconds/step,
  about 44.2 GiB GPU memory, no OOM or fatal error

This is the quick one-ID gate for whether more surrounding spatial context
retains the benefit seen in the completed 40%-margin run without reproducing
the catastrophic `canvas1024` failure.

## Live Serv matrix

All four jobs request one A100 each, so the current project allocation is
exactly the user-authorized maximum of four GPUs.

| Arm | Run | MLS job | Comet key |
|---|---|---|---|
| 40% margin, 256, pose-first | `rhca_cosmic_full_crop40_posefirst_4k_fast_r1` | `lm-mpi-job-c31986ac-283b-4f4f-9d15-aef54890fc54` | `1a19fdf2793f413c9336379d3628874d` |
| 60% margin, 256, pose-first | `rhca_cosmic_full_crop60_posefirst_4k_fast_r1` | `lm-mpi-job-86e0060d-c2a6-4f9d-a6fd-2ee8c0036285` | `a96bcbae3d2b4698a43d7ec80457586c` |
| 40% margin, 256, legacy captions | `rhca_cosmic_full_crop40_legacy_4k_fast_r1` | `lm-mpi-job-a4a2cdb3-ffc5-45ec-9cb6-e0536494ab43` | `92572589d6594cd59749577fc51f5bba` |
| 40% margin, 512, pose-first | `rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1` | `lm-mpi-job-29283861-49e2-4ee3-8572-d8ba3970fa31` | `c354369af45b4c9da84f1124cf3e9a88` |

Startup evidence observed for every arm:

- MLS status is `running`.
- ONNX Runtime reports version 1.20.1 with
  `CUDAExecutionProvider`.
- The 64-sample preflight passes 64/64 and resolves 22,140 accepted records
  from the full Cosmic manifest.
- `saved/<run>/comet_experiment.json` contains the immutable Comet key and
  commit `cfa4bffebfbb46e324a7b503bdbfd786bea5e6e6`.
- Resolved `config.yaml` records `pose_adapt_ratio: 0.0`,
  `ca_mixing_for_face: false`, `use_branched_attention: true`, and two train
  workers.
- The architecture audit reports 840/840 branched-attention processor
  parameters in the optimizer and 3,080 trainable tensors in total.
- No OOM, traceback, or fatal runtime error was observed.

At the final observation, the 40%-margin, 60%-margin, and legacy-caption arms
were generating their 12-image step-0 validation panels. The 512-reference
arm was still completing model initialization after its preflight and Comet
record. These are confirmed live experiment starts, but the next agent should
still capture the first real optimizer steps and seconds/step for all four.

## Resolved experiment controls

The matrix isolates three questions while holding BA routing fixed:

1. Does 40% surrounding reference context transfer the favorable one-ID result
   to the full Cosmic dataset?
2. Is 60% context better or worse than 40%?
3. At a fixed 40% crop, are failures driven by caption policy or loss of
   reference resolution?

The exact resolved policies are:

| Run suffix | Margin | Content size | Prompt mode | Prompt limit |
|---|---:|---:|---|---:|
| `crop40_posefirst` | 0.4 | 256 | `pose_first` | 55 words |
| `crop60_posefirst` | 0.6 | 256 | `pose_first` | 55 words |
| `crop40_legacy` | 0.4 | 256 | `legacy` | none |
| `crop40_512_posefirst` | 0.4 | 512 | `pose_first` | 55 words |

The package launchers and immutable specifications are under:

- `serv_run_packages/rhca_cosmic_full_crop40_posefirst_4k_fast_r1/`
- `serv_run_packages/rhca_cosmic_full_crop60_posefirst_4k_fast_r1/`
- `serv_run_packages/rhca_cosmic_full_crop40_legacy_4k_fast_r1/`
- `serv_run_packages/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/`
- `experiment_specs/rhca_cosmic_full_crop40_posefirst_4k_fast_r1.json`
- `experiment_specs/rhca_cosmic_full_crop60_posefirst_4k_fast_r1.json`
- `experiment_specs/rhca_cosmic_full_crop40_legacy_4k_fast_r1.json`
- `experiment_specs/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1.json`

## Next-agent checklist

1. Inspect all four MLS jobs and their stdout/stderr. Do not submit another
   Serv GPU while all four remain pending or running.
2. Confirm each arm finishes step-0 validation and advances through at least
   20 real optimizer steps. Healthy throughput should be near the corrected
   approximately 2 seconds/step range on Serv, not the superseded
   approximately 4 seconds/step CPU-fallback behavior.
3. If a job exits before real training, diagnose it before resubmitting. Use a
   fresh run name and preserve the failed Comet key; never reuse an existing
   `saved/<run>` identity.
4. Inspect the Neb margin-60 step-500 and step-1000 panels against the completed
   one-ID margin-40 and canvas-1024 runs. Continue to 4k unless every output is
   catastrophically malformed with no visible improvement.
5. For every completed full-Cosmic arm, require `checkpoint-epoch8.pth`, 12
   step-4000 validation images, no fatal errors, and its Comet metrics.
6. Compare matched prompts/seeds visually for facial anatomy, face/body
   registration, identity, prompt adherence, and global scene integrity.
   Metrics alone are not a promotion gate.
7. Select the strongest ratio-zero BA endpoint, then run its canonical full
   96-image validation as a separate Comet experiment on its respective
   machine. Do not compare a 12-image endpoint metric directly with historical
   96-image metrics.
8. Preserve machine-local bbox files, `.env`, credentials, and Serv files
   outside the `nasilaev` ownership area.

Useful local commands:

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template
python3 ../local_scripts/serv_job.py inspect <MLS_JOB_ID>
ssh neb 'cd /home/niko/rsrch/diffusion_template && tail -n 80 logs/rhca_cosmic_oneid_margin60_4k.log'
```

The broader interpretation and earlier full-96/plain-PhotoMaker control are
recorded in
`docs/experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md`.
That report and this handoff are intentionally uncommitted unless the user
separately authorizes committing experiment documentation.
