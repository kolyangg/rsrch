# Current project handoff

**Last updated:** 16 August 2026

**Repository:** `/home/kolyangg/rsrch_apr_test`

**Primary project:** `diffusion_template/`

**Branch:** `test`

**Current branch:** `test`; inspect live worktrees because active jobs are pinned
to their recorded launch commits.

This is the required starting document for a new agent. It summarizes the
research question, experiment history, reliable results, current code and
machine state, and the highest-value next work. Detailed evidence remains in
the linked reports.

## Read this first

The project tests whether PhotoMaker identity-conditioned generation can be
improved by explicit branched attention (BA). The core invariant is that
target queries must be able to consume identity/reference information through
the intended branched self-attention and cross-attention paths. A run that
looks good because it removes effective reference conditioning is a useful
ablation, but it is not a successful BA result.

For all currently eligible experiments:

```text
use_branched_attention = true
pipeline.pose_adapt_ratio = 0.0
pipeline.ca_mixing_for_face = false
reference_face_kv_weight = 1.0
```

The most recent full-Cosmic experiments have branched self-attention enabled
and branched cross-attention disabled. They establish results for the
reference-conditioned **SA-only BA protocol**, not a combined SA+CA design.

Do not change validation prompts, seeds, reference images, bboxes, validation
base, scheduler, inference steps, CFG, or metrics silently. Exact
comparability is part of correctness.

## Executive state

### CL15-CL20 final results and CL19 next architecture plan - completed, 13 August 2026

The audited report
`analysis/2026-08-13_cl15_cl20_results_cl19_next_architecture.md` supersedes
the earlier unmatched CL19/CL20 snapshot and the provisional 11 August CL15-
CL20 ranking. A fresh immutable-key Comet export contains exact, table-sealed
96-image 24k panels for CL14 through CL20 and step 0 for controlled PhotoMaker,
with no exporter errors:

- CL15 `d57604dc77334e0f9874ddd049e85a17`: ID `0.451067`;
- CL16 `7b71eb3dbb3a492e8fa9bb0d87343c28`: ID `0.453810`;
- CL17 `e05ce586c9364678a8370a774773341c`: ID `0.439219`;
- CL18 `f6530436bf22472c9fb7731d1696c5ab`: ID `0.451972`;
- CL19 `cfeda7b55c174b3c83e8d40537ebb6dd`: ID `0.506823`, text
  `26.370605`, mask IoU `0.914185`, peak ID `0.507105 @22k`;
- CL20 `b05488e2cce94476acc92bcaa21d7362`: ID `0.450064`;
- controlled PhotoMaker `74efd227d3f8488a98e83d815c77c07c`: ID
  `0.556580`.

CL19 is the only CL15-CL20 arm with a clear positive paired ID delta over
CL14: `+0.050707`, `74/96` wins, cell-bootstrap interval
`[+0.03730,+0.06464]`. It still trails controlled PhotoMaker by `-0.049757`,
wins `18/96`, interval `[-0.06483,-0.03531]`, and has lower face-quality
means. CL19 improves all twelve prompt means over CL14; Crying improves most
(`+0.0935`) and is largely visually repaired. Skiing remains the main
visibility-topology failure: one-reviewer inspection records CL19 as four
passes, one minor, and three clear failures versus eight PhotoMaker passes.
Future review must reject goggle/hand deletion as a fake identity improvement.

Read-only final-checkpoint inspection changes the interpretation of the failed
arms. CL15's ROI scale averaged about `0.00218`, CL16's clean-memory scale
about `0.00167`, and CL17 learned its synthetic ownership labels while its
routed effect stayed below `0.85%` of native. These runs do not prove that
their mechanisms are intrinsically useless; they prove that zero-retreat
gates/auxiliary labels can be optimized without materially controlling the
denoiser. New learned branches require a centered or floored contribution and
per-group `RMS(delta)/RMS(native)` telemetry.

The priority-ordered independent CL19 arms are now: (1) corrected bounded
residual identity-token CA v3; (2) explicit top-object/visible-face/background
visibility-order routing; (3) temporal-frequency routing of CL19's
reference-minus-native message; (4) sparse controlled-PhotoMaker boundary
distillation; (5) a four-step low-noise ArcFace reward continuation anchored
to frozen CL19; and (6) CL15's high-resolution ROI with a guaranteed active
gate. Design-only, non-runnable YAMLs and implementation gates are under
`analysis/blueprints/2026-08-13_cl19_next_six/`. Generic BigCelebs mixing is
not prioritized: CL20 and the historical BigCelebs arms are negative/neutral
for identity. No new training job was submitted for this report.

An additional immutable-key refresh found a complete interim 10k panel for
the corrected active `CL14_CA_optimized_r11`, key
`fafd7a61b06c4114b9dec2c21d29ca38`: ID `0.445238` versus matched CL14
`0.423661`, delta `+0.021577`, `58/96` wins, cell-bootstrap interval
`[+0.00800,+0.03521]`. This strengthens the first CL19-based CA experiment but
is not a final endpoint; full face-quality aggregates were not yet present at
the export cutoff. Recheck its immutable record before implementing or
launching CL21.

### PM0 / CL14 / CL19 / CL20 visual comparison snapshot — 12 August 2026

The latest exact-format Comet comparison PDF is
`output/pdf/comet_report_PM0_CL14_CL19_CL20_12Aug2026.pdf`; it supersedes the
earlier CL13 report for the requested comparison. It freezes the latest
complete validation panels available at export time: PhotoMaker V2 step 0
(`74efd227d3f8488a98e83d815c77c07c`), CL14 final step 24k
(`6fe0028be92242c38056b3d36665fdd6`), CL19 step 18k
(`cfeda7b55c174b3c83e8d40537ebb6dd`), and CL20 step 18k
(`b05488e2cce94476acc92bcaa21d7362`). Their subject-v2 ID means are
`0.556580`, `0.456116`, `0.506175`, and `0.443401`, respectively. CL14 peaks
at `0.457096` at step 22k; CL20 peaks at `0.446974` at step 16k. This remains
an explicitly unmatched-step snapshot; do not interpret it as a final
CL19/CL20 comparison.

The 24-page report contains all 96 matched images per run with the per-image
subject-v2 ID value in a filled top-right box, readable formula-based
architecture pages, exact critical attention-code excerpts, the eight fixed
reference identities and twelve prompts, a working full-history
`manual_val/id_sim` curve with three-decimal last/maximum labels, grouped
identity/prompt means, and a final table of the scientifically meaningful
configuration differences. The editable architecture/code/reference source is
`tools/comet/comet_report_pages_PM0_CL14_CL19_CL20.md`; reusable inputs are
`tools/comet/comet_runs_12Aug_PM0_CL14_CL19_CL20.json` and
`tools/comet/comet_pdf_config_12Aug_PM0_CL14_CL19_CL20.json`. The exporter uses
the per-image ID table as the completion seal, preventing a partially uploaded
next validation panel from being mistaken for the current step.

A later immutable-key refresh at `2026-08-12T15:46:37Z` found complete,
table-sealed 20k panels for CL19 and CL20; the PDF above remains the earlier
18k visual freeze. At the matched 20k gate, CL19 reaches subject-v2 ID
`0.503941`, text `26.3499`, and mask IoU `0.91473`; CL20 reaches `0.452543`,
`26.5938`, and `0.89226`; CL14 at 20k is `0.455268`, `26.2381`, and
`0.89596`. CL19 therefore leads CL14 by `+0.048673` ID at a matched step,
while CL20 remains effectively CL14-level. CL19's current peak remains
`0.506175` at 18k. No complete CL19/CL20 22k or 24k panel existed at that
cutoff.

The historical branched-cross-attention audit is
`analysis/2026-08-12_branched_cross_attention_disable_history_and_cl19_reintroduction.md`.
It concludes that the legacy `BranchedCrossAttnProcessor` must remain disabled:
July Cosmic evidence showed broader corruption, and direct code inspection
shows reference-hidden queries, identity output returned to the reference
half, unused spatial masks, and whole-lane replacement. This rejects the
legacy processor, not explicit CA as a project mechanism: the older leak-free
one-ID CA-on checkpoint was coherent and higher-ID, E12 rejects hard ID-only
face replacement, and E17's bounded residual CA was safe but inconclusive on
a weaker substrate without branch telemetry. The recommended next CA test is
a fresh single-delta CL19 successor using the existing corrected bounded
residual identity-token CA v3 in `up_blocks.0/1`, with native CA intact and
complete branch-use telemetry. Do not mutate or resume CL19/CL20 to add CA and
do not launch the conditional CL20 transfer unless the CL19-based arm passes.

The user subsequently selected CL14, rather than CL19, as the control for the
next CA ablation and fixed the new run name to exactly `CL14_CA`. The
implementation-ready design is
`analysis/2026-08-12_CL14_CA_implementation_plan.md`. It specifies a fresh
CL14-configured training run—not a continuation from CL14's 24k checkpoint—with
only the corrected residual identity-token CA v3 enabled in `up_blocks.0/1`,
rank 64, gate `0.02..0.20`, full branch telemetry, and an expected startup
contract of `2,348 / 224,624,676`. The immutable comparison control remains
CL14 key `6fe0028be92242c38056b3d36665fdd6`. This user-selected experiment is a
cleaner answer to “does CA add value to CL14,” although CL19 remains the
stronger absolute architecture substrate. The implementation and corrected
active scientific run now exist; see the final `CL14_CA implementation and
validation startup` section for immutable IDs, the telemetry fix, CL14-parity
deployment, and current status.

### Original PhotoMaker CL19-matched baseline — completed, 11 August 2026

The original pretrained PhotoMaker V2 baseline now has a complete, fail-closed
run on the exact CL19 validation contract. Serv job
`lm-mpi-job-0b15a0e4-5db3-48e6-bd98-450d5865ddd0` completed with exit code
zero; immutable Comet experiment `74efd227d3f8488a98e83d815c77c07c`
contains 96/96 images at exact step 0, all identity/text aggregates, all seven
face-quality aggregates, the 96-row per-image identity table, and the verified
96-row per-image face-quality CSV asset. A fresh immutable-key Comet export
reported zero warnings and zero errors.

The run uses CL19's byte-sealed prompts, references, bboxes, seed 0, full
96-image subject-v2 panel, RealVisXL validation base, DDIM 50, CFG 5, batch 12,
PhotoMaker start step 10, and metric definitions. It loads the pretrained
`PhotoMaker-V2/photomaker-v2.bin`, takes no optimizer step, loads no training
checkpoint, and disables branched attention so validation restores the native
attention processors plus the pretrained PhotoMaker default adapter. This is
the controlled plain-PhotoMaker baseline for CL19, not a BA experiment.

Step-0 observed metrics are: subject-v2 ID `0.5565802762`, legacy-best ID
`0.5014309009`, text similarity `26.0014648438`, mask IoU `0.8651533823`,
face count `1.1354166667`, no-face/unowned/ambiguous all `0`, face-detection
rate and TOPIQ-Face coverage both `1.0`, TOPIQ-Face mean `0.7531917117`,
TOPIQ-Face p10 `0.5918496251`, TOPIQ mean `0.6147096589`, MUSIQ mean
`73.0987923145`, and MANIQA-PIPAL mean `0.6437278905`.

Durable implementation and launch records are
`src/configs/PM0_original_photomaker_CL19_full96.yaml`,
`launchers/active/run_PM0_original_photomaker_CL19_full96_1gpu.sh`,
`serv_run_packages/PM0_original_photomaker_CL19_full96_r1/`, and
`experiments/baselines/PM0_original_photomaker_CL19_full96_r1.json`.

### CL9 fixed-checkpoint edge validations — completed, 11 August 2026

Both user-authorized one-A100 sidecars finished with MLS status `failed`, but
the ROI sidecar first produced the decisive baseline result: **96/96 fixed-panel
images replayed with exact RGB equality** from CL9's 24k checkpoint.

- `lm-mpi-job-bcff7ec7-e2b5-47de-87ac-ca37210da8dd` — the isolated Marion
  12-image baseline was `0/12` exact and the fail-closed gate prevented roll,
  similarity, and occluder arms from running. The full-96 success rules out a
  broadly wrong checkpoint/evaluator; the exact sequence-state cause is not
  established. Repair by replaying all 96 rows in historical order for each
  targeted arm, keeping indices 0-83 as sentinels.
- `lm-mpi-job-5dc7cc57-c622-48a5-a80d-16607107f151` — full-96 baseline was
  `96/96` exact. The first ROI arm then failed before saving an intervention
  image because the immutable DDIMScheduler does not accept a custom timestep
  list. Repair by retaining the standard DDIM50 grid and running its late suffix
  inside a bounded scheduler-compatible denoising loop.

Both use CL9 `weights-epoch12.pth` (step 24k, SHA-256
`5396993b16ace89908501bfddb2e412e755a3f6478a6449c502062d6ca7357c3`),
RealVis, `legacy_full_copy`, batch 12, CFG 5, DDIM 50, the exact active bbox
map, `pose_adapt_ratio=0`, and `ca_mixing_for_face=false`. Neither causal
intervention produced results; do not infer performance from the failed jobs.
The audited status report is
`analysis/2026-08-10_cl9_baseline_replay_and_validation_status.md`. Durable plans are
`experiments/diagnostics/CL9V_marion_occlusion_validation_24k_20260810_r1.json`
and `experiments/diagnostics/CL9V_smallface_roi_refine_24k_20260810_r1.json`.

Repaired `r2` packages are deployed and pass local plus remote shell/compile
checks. The Marion package replays full96 and changes only indices 84-95; the
ROI package uses a verified suffix of the unchanged DDIM50 grid. Submission
attempts at 16:14 local time were rejected with
`PROJECT_GPU_LIMIT_REACHED_ONLY_0_FREE`: the eight-GPU recovery job still owns
all project capacity. The user explicitly renewed the scoped ten-GPU exception,
but one fresh attempt per package at 16:20 was rejected by the Serv backend with
the same error before any MLS job was created. The active eight-GPU recovery
allocation was still at 12/25 verified runs, with five workers running and one
failed; do not stop it merely to make room. The user then explicitly authorized
recurring retries. Durable Serv scheduler PID `849511` will first try both jobs
at `2026-08-10 17:00 Europe/London`, then every 30 minutes. It tracks the two
YAMLs independently, never resubmits an accepted job, exits after both have MLS
IDs, and stops early if the agent creates the scheduler `STOP` flag. State is at
`analysis_sidecars/cl9v_r2_submission_scheduler_20260810`. Records:
`experiments/diagnostics/CL9V_marion_occlusion_validation_24k_20260810_r2.json`
and `experiments/diagnostics/CL9V_smallface_roi_refine_24k_20260810_r2.json`.

The scheduler was live-audited at 17:15 local time. It attempted both YAMLs at
17:00 as configured, counted four current `#nasilaev` GPUs, and reached MLS;
both were rejected before job creation with
`PROJECT_GPU_LIMIT_REACHED_ONLY_0_FREE`. A user-requested immediate attempt at
17:15 received the same response for both. Scheduler PID `849511` remains alive
and the next anchored attempt is 17:30 local time.

At 22:00 local, MLS accepted both `r2` YAMLs, but the original scheduler parser
did not recognize pretty-printed multi-line JSON and later duplicated Marion.
All three allocations failed without usable intervention outputs. Small-face
`r2` exposed a rectangular BA target (`seq_len 580 is not square`); Marion
`r2` exposed that the evaluator applies reference conditioning once to the
whole 96-row array, not once per batch. The parser now recognizes multi-line
responses and accepted IDs are sealed before subsequent retries.

Clean `r3` packages were built rather than reusing partial roots. Small-face
uses a square source crop/work canvas and passed local plus deployed all-16-row
geometry gates. Marion replaces only entries 84-95 in the evaluator's full96
conditioned-reference arrays and passed local plus deployed substitution gates
against the real transform manifest. Small-face `r3` job
`lm-mpi-job-131bc8e8-7a90-4d3c-acf3-1a2c13514ce6` completed successfully in
1,249 seconds with exit code zero after both small-face ROI arms and scoring.
It passed the reused `96/96` exact baseline gate and the 80-image sentinel gate
for each arm. Marion `r3` was initially rejected seconds later for zero free
capacity. The running MLS worker had no supported exec/attach continuation
hook, so it was not mutated. Scheduler PID `860799` detected small-face
completion at `2026-08-10T21:50:16Z` and immediately secured Marion job
`lm-mpi-job-193731be-cba3-4ae8-8606-53dbb0b59d47`; the scheduler then exited
normally. Marion passed its exact baseline gate, generated all 96 roll-arm
images, passed the 84-image roll sentinel gate, generated all 96 similarity and
all 96 occlusion-ownership images, and passed both remaining sentinel gates.
MLS job `lm-mpi-job-193731be-cba3-4ae8-8606-53dbb0b59d47` completed successfully
in 6,209 seconds with sidecar exit code zero. All three arms have subject-v2
and TOPIQ-Face outputs, and the final 338-file SHA-256 manifest is present.
Durable records are the two
`CL9V_*_20260810_r3.json` files under `experiments/diagnostics/`.

The completed paired analysis is
`analysis/2026-08-11_cl9_validation_interventions_results.md` with its rendered
PDF in `analysis/assets/2026-08-11_cl9_validation_interventions_results.pdf`.
The promotion summary excludes Eddie for the ROI and ownership arms because
the exact historical replay predates the repaired Eddie generation rows. The
18-step ROI suffix is the only positive result: non-Eddie ID improves by
`+0.097` with `11/14` wins while all 16 composites remain pixel-exact outside
the ROI; the 10-step suffix loses `-0.116` with `0/14` wins. Marion roll gives
a modest `+0.023` ID and `+0.015` TOPIQ but needs multi-seed replication;
five-point similarity is effectively neutral. The static occluder mask is
rejected: combined non-Eddie ID is flat, face scale/alignment regress, and the
largest Skiing gains move goggles away from the intended occlusion. Run the
14/16/18/20-step multi-seed ROI sweep first, then a per-image precise occluder
geometry diagnostic before considering training either mechanism.

The three promotion-gated follow-ups completed as one fail-closed,
single-A100 chain in
`serv_run_packages/cl9v_validation_chain_20260811_r4/`. The 30-minute monitor
submitted once capacity fell to four project GPUs; MLS job
`lm-mpi-job-09cb6478-b936-4ecb-b69b-a082742641c2` ran from
`2026-08-11T01:15:27Z` to `08:14:44Z` and exited at `final_hashes` with code
zero. All 26 stages completed. The chain passed `96/96` baseline replay,
`80/80` sentinels for every one of the 16 ROI arms, `16/16` exact-outside-ROI
composites per arm, `96/96` exact reproduction of r3 for ROI18 seed 0,
`80/80` precise-occluder sentinels, and `84/84` sentinels for all six new
Marion arms. Its final output seal contains 2,974 hashes.

The multi-seed result promotes the `18`-step ROI suffix: non-Eddie ID improves
`+0.09684` over 56 prompt-seed pairs with `43/56` wins and clustered 95%
interval `[+0.04354,+0.14659]`; all four seed means are `+0.094` to `+0.100`.
ROI20 also passes but has lower mean ID and more face/boundary movement. The
precise occluder arm is better than the static mask (`+0.03834`, `9/14` wins,
TOPIQ `+0.02743`, IoU `-0.00320`) but fails continuation because Skiing is
only `4/7` and visual goggle duplication/relocation remains. Marion roll does
not replicate: all-seed ID is `+0.00128`, `19/48` wins, interval
`[-0.00837,+0.01234]`; retire roll and five-point normalization as identity
fixes. The replicated ROI mechanism unlocks
`CL9T_shared_ba_highres_face_aux_24k_r1`; learned occluder-gate training stays
deferred. The updated evidence is in
`analysis/2026-08-11_cl9_validation_interventions_results.md`; the durable run
record is
`experiments/diagnostics/CL9V_validation_chain_24k_20260811_r4.json`.

### E11/E12 BA-capacity plan — 4 August 2026

The next two requested Large Dataset experiments are documented in
[`2026-08-04_e11_e12_large_ds_ba_capacity_plan.md`](../experiments/2026-08-04_e11_e12_large_ds_ba_capacity_plan.md).
E11 widens only the existing hard spatial-SA BA Q/K/V LoRA from rank 32 to
rank 128 (expected 127.80M BA parameters). E12 keeps rank-32 spatial BA and
adds a new corrected, hard face-local target-query/active-PhotoMaker-ID-token
CA branch at rank 256 in `up_blocks.0/1` (expected 134.58M total BA
parameters), with no native/ID face-output interpolation.
The legacy branched CA processor remains disabled: code inspection shows
that its identity-prompt result is returned to the reference half rather than
merged face-locally into target queries, and historical CA-on evidence was
worse. Both entries are implemented behind defaults-off controls with exact
ownership/configuration gates and local one-A100 MLS packages. They have not
been deployed, registered in Comet, or submitted.

### Neb unavailable — 3 August 2026

Neb is not working and must not be accessed or used. Do not attempt SSH/SCP,
rsync, job inspection, launches, termination, file retrieval, or Comet
downloads through Neb. Use the local machine or Serv when appropriate. If a
user asks to use Neb, first obtain separate explicit confirmation that the
machine is working again and that they want a connection attempt; the request
itself is not confirmation. After confirmation, begin with a read-only
connectivity check. Historical Neb paths, launchers, PIDs, and operational
instructions below remain provenance only while this restriction is active.

### Architecture audit warning — 1 August 2026

The Large Dataset and BigCelebs plateau audit found that their advertised
`train_ba_only=true` contract did not execute. Every preserved startup and
alternate-validation model log contains
`'AttnProcessor2_0' object has no attribute 'parameters'`; the exception is
swallowed before `configure_branched_trainables()` runs. The optimizer therefore
contained 171.29M requires-grad parameters rather than the intended 31.95M
rank-32 BA processor state. It included the pretrained rank-64 PhotoMaker
`default` adapter and the new rank-32 generic U-Net adapter as well as BA.

Checkpointing saves the new adapter and BA deltas but not the unintentionally
trained `default` adapter. In-training alternate-base validation also performs
a legacy full processor copy, transferring training-base effective Q/K/V
buffers into the RealVis validation U-Net. Existing within-run validation
curves remain useful and consistently produced, but they are not clean BA-only
rank-32 capacity measurements and do not contain the complete live training
state.

Do not launch a rank-64 or broader-trainable architecture experiment until
processor installation fails closed, exact trainable ownership is asserted,
all trainables round-trip through checkpoints, and validation processor-base
semantics are explicit. The detailed evidence, fixes, implementation diffs,
and prioritized residual-BA architecture are in
[the 1 August architecture recommendations](../../analysis/2026-08-01_large_dataset_big_celebs_ba_architecture_recommendations.md).

### Clean BA32 correctness arm — stopped on Neb, 2 August 2026

The defaults-off Priority-0 repair is implemented locally. Strict installation
skips the historical processor-wide enable loop, derives an exact BA allowlist,
and audits both requires-grad and optimizer membership on every rank. The new
schema-v2 checkpoint format saves the exact trainable U-Net tensors plus an
architecture manifest and loads them with exact name/shape checks; schema-v1
checkpoints retain their historical loader. Alternate-base validation now
records one of `legacy_full_copy`, `validation_native`, or
`no_processor_update`. Absent the new setting, the old
`update_proc_weights_val` behavior is preserved.

The first controlled arm intentionally keeps the historical hard-routing
attention math, rank 32, loss, LR, dataset schedule, and explicitly audited
`legacy_full_copy` validation behavior. Its config is
`src/configs/big_celebs_scheduled_rhca_clean_ba32_40k.yaml`; its prepared Neb
launcher is
`launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh`; and its
prepared immutable plan is
`experiments/big_celebs/rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1.json`.
The concise implementation overview is
[`docs/experiments/2026-08-01_clean_ba32_priority0_implementation.md`](../experiments/2026-08-01_clean_ba32_priority0_implementation.md).
The launcher defaults to Neb GPU 0. The reviewed files were synchronized onto
Neb after preserving the machine-specific originals under
`/home/niko/rsrch/runtime_backups/clean_ba32_20260801_1805`.

Hydra composition, Python compilation, shell syntax, launcher hash-locks, a
mock reproduction of the old `AttnProcessor2_0` failure, exact optimizer
membership, strict processor copying, and schema-v2 tensor round-trip all pass.
The real startup gate remains exactly 840 tensors / 31,948,800 parameters.

`rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1` was stopped cleanly at
the user's request after its complete 36k validation/checkpoint gate and before
40k:

- immutable Comet key
  [`700240d8f90b48cfa2cc16f8ff2886b6`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/700240d8f90b48cfa2cc16f8ff2886b6);
- all sealed-dataset, sampling-plan, ONNX CUDA, Hydra, shell/Python, and 13/13
  audited-runtime checks passed;
- exact ownership is 840 tensors / 31,948,800 parameters, all under branched
  SA processors, with optimizer membership 840/840;
- explicit `legacy_full_copy` validation copied 70/70 stateful processors;
- step-0 validation completed 96 images and face-quality scoring with 94
  detected faces;
- training began at approximately `2026-08-01T18:23:09Z`; its complete
  step-32k/epoch-16 and step-36k/epoch-18 checkpoints are available. TERM was
  sent only to the verified training and launcher process groups; both exited
  without KILL and Neb's GPU was empty before fixed-checkpoint validation was
  launched.

The 32k result materially updates the architecture decision. Identity peaks at
`0.3347` at 18k and oscillates through `0.3273` at 32k, while TOPIQ-Face mean
peaks at `0.7472` at 22k, p10 reaches `0.6281` at 32k, and text stays near
`27.6–28.0`. Between the identity peak and 32k, the pinned schedule supplies
26,177 additional unique targets and 5,320 additional identities. The clean
plateau therefore persists without the former 171M-parameter ownership bug and
is not explained by exhausting the data.

The exact same-schedule historical arm has materially higher identity but
lower text and face-IQA through 14k, showing that its unintended generic U-Net
adapter bought identity at the cost of prompt/quality behavior. Direct 32k
checkpoint inspection finds all trainables and Adam moments in BF16 and all 32
LoRA singular directions numerically active but strongly top-heavy. The next
model should not globally increase `noise_and_ref` rank. First implement a
true-key-masked, gated residual reference-SA path with frozen target Q/K/V, a
branch-local output adapter, FP32 trainables, inference-aligned timesteps, and
correct-versus-shuffled spatial-reference diagnostics. Then test rank 64 only
inside reference K/V and branch output. Detailed evidence and implementation
diffs are in
[`analysis/2026-08-02_clean_ba32_32k_architecture_recommendations.md`](../../analysis/2026-08-02_clean_ba32_32k_architecture_recommendations.md).

### Residual SA-v2 implementation — prepared locally, 2 August 2026

The critical architecture changes recommended above are now implemented behind
defaults-off toggles. `residual_sa_v2` preserves frozen native target SA and
adds target-Q/reference-KV attention as a true reference-key-masked, bounded
face-local residual with a branch output adapter. The prepared rank-32
mid/up-block configuration owns exactly 414 tensors / 10,567,818 parameters,
all FP32, with separate reference-KV, output, and gate optimizer roles. It uses
inference-active DDIM timesteps and logs a detached 25%-probability spatial
reference shuffle diagnostic; its causal auxiliary loss weight is zero.

Historical `hard_replace_v1`, inherited dtype, uniform-all timestep behavior,
and no shuffle remain the defaults. The prepared Neb training launcher is
`launchers/neb/start_rhca_big_celebs_scheduled_residual_sa_v2_40k.sh`; its
default run is
`rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r1`. The fixed
clean-32k D0 diagnostic launcher is
`launchers/neb/run_clean_ba32_32k_d0_validation_matrix.sh`; it prepares legacy
versus native processor-base and matched versus zero/shuffled spatial-reference
arms while keeping PhotoMaker identity inputs matched. Neither launcher was run
or synchronized to Neb during the initial implementation session. Exact changes,
verification, limitations, and ladder mapping are in
[`docs/experiments/2026-08-02_residual_sa_v2_critical_changes.md`](../experiments/2026-08-02_residual_sa_v2_critical_changes.md).

The D0 evaluator, Comet publisher, arm specifications, and launcher were later
synchronized deliberately to Neb. The launcher records one immutable Comet
experiment per arm and uses the trainer-equivalent cached generation-bbox map.
The matrix completed under historical PGID `3407672` from the 32k checkpoint
(SHA-256
`99a1491e32ea58b1262eb17799457cc1fcce37defec3a8048bbeff8b4312a30c`):

- completed arm: `d0_clean_ba32_32k_legacy_matched`; immutable Comet key:
  [`736a47373f0a43508fd5a3b2e32ac2a4`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/736a47373f0a43508fd5a3b2e32ac2a4);
- completed arm: `d0_clean_ba32_32k_native_matched`; immutable Comet key:
  [`82ba5adaddc44cc1b0a25f7f5c9f57e1`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/82ba5adaddc44cc1b0a25f7f5c9f57e1);
- completed arm: `d0_clean_ba32_32k_native_zero_spatial`; immutable Comet key:
  [`d13388ecfdda435fa4786b5dd3038cb2`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/d13388ecfdda435fa4786b5dd3038cb2);
- completed arm: `d0_clean_ba32_32k_native_shuffle_spatial`; immutable Comet key:
  [`fa42aea6b0954816bbca69cbe200b2c4`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/fa42aea6b0954816bbca69cbe200b2c4);
- bbox audit: 96/96 cached entries resolved from
  `pm96_bboxes_new_auto.json`, SHA-256
  `5e4983bdb9fe75e4fb75568d24f1126ea2b8c8ae3c07857fb14bae130702a14c`;
- the first attempt failed before producing an image because the standalone
  evaluator sanitized bbox-key spaces. The corrected run resumed the same
  Comet key; the first arm completed all 96 images, metrics, face-quality
  scoring, and Comet upload without an OOM.

The matrix log is `logs/d0_clean_ba32_32k_validation_matrix.log`;
per-arm logs and outputs are under `diagnostics/d0_clean_ba32_32k/`. The
residual-SA-v2 training launcher was subsequently launched; see the live-result
subsection below.

All four D0 immutable keys were moved to Comet project
`jul-comet-large-testing-tr`. The matrix completed all four arms. The detached
end-of-matrix audit in
`logs/d0_clean_ba32_32k_comet_move_verification.log` verifies 15 metric names
and exactly 96 image assets on every arm in the intended project.

### Residual SA-v2 first live result — failed causal-use gate, 2 August 2026

The first successfully started live residual arm is
`rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6`, immutable
Comet key
[`4d6186f56cd24ba3a907fa35406c284e`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/4d6186f56cd24ba3a907fa35406c284e).
Its fixed-96 step-0 and step-2k results are downloaded under
`comet_data/rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6/`.

The run rules out installation, optimizer, save, and validation-load failures:
the 2k weights contain the exact 414 tensors / 10,567,818 FP32 parameters; all
46 reference-K, reference-V, and reference-output LoRA-B sites are nonzero;
all three optimizer roles receive gradients; and all 96 step-2k images differ
from step zero. However the architecture fails the functional reference-use
test:

- step zero is exactly native PhotoMaker by construction because the only
  reference output is a zero-initialized, base-free low-rank delta;
- the mean gate remains only `0.1034` at 2k;
- identity similarity falls `0.5236 -> 0.5086` and TOPIQ-Face mean falls
  `0.7473 -> 0.7359` while text similarity rises;
- the shuffled-spatial-reference face-error gap is centered near zero
  (`-2.60e-6` including unshuffled diagnostic zeros);
- the reference-causal loss weight and margin are both zero, and the wrong
  branch is detached.

The correct interpretation is that v2 is mechanically active but learns a
small generic PhotoMaker correction without demonstrated causal dependence on
which spatial reference is supplied. Do not treat more steps or rank 64 as the
primary repair.

The next recommended architecture is a versioned anchored interpolation BA-v3:
use frozen target Q and explicit masked reference K/V, project the reference
message through the frozen native output projection plus a trainable delta,
and interpolate inside the target face with a nonzero bounded strength
(`alpha_init` about `.50`, floor about `.25`). This bridges exact PhotoMaker at
`alpha=0` and old forced reference routing at `alpha=1`. Add actual
branch-contribution telemetry and then isolate a differentiable
correct-versus-shuffled reference-ranking objective. Full evidence, code diffs,
reversible toggles, and the E0-E8 experiment ladder are in
[`analysis/2026-08-02_residual_sa_v2_2k_plain_photomaker_failure_analysis.md`](../../analysis/2026-08-02_residual_sa_v2_2k_plain_photomaker_failure_analysis.md).

At the last read-only machine audit, r6 remained active beyond 3k on Neb and
used about 28.1 GiB GPU memory. This is a transient resource snapshot, not an
instruction to assume the process is still live later. It was not stopped by
the analysis session.

### Anchored interpolation BA-v3 — E3 completed and analyzed, 2 August 2026

The recommended repair is now implemented behind the new explicit
`anchored_mix_sa_v3` selector. It keeps frozen target Q and native target
self-attention, applies true-key-masked reference K/V, projects the reference
message through frozen native `to_out` plus a zero-initialized rank-32 output
delta, and interpolates inside the target face with a bounded nonzero mix.
Shared defaults remain `hard_replace_v1`; `residual_sa_v2` tensor names and
zero-start behavior are unchanged.

The completed E3 arm used the same rank 32, 46 mid/up sites, inference-active
timesteps, pinned BigCelebs schedule, and fixed-96 validation contract as r6,
but ran for only 2,000 optimizer steps. Its mix started at `.50`, had a `.25`
floor and `.90` maximum, and used detached clipped RMS matching. Exact
ownership remains 414 tensors / 10,567,818 FP32 parameters, with roles
`ref_kv`, `ref_output`, and `mix`. Schema-v2 manifests now distinguish v3 and
fail on routing/mix/RMS mismatches.

Runtime telemetry logs detached matched-forward mix and actual
reference/native and contribution/native RMS ratios by `mid`, `up0`, and
`up1`; a shuffled diagnostic cannot overwrite it. Correct and shuffled
forwards explicitly reuse the same per-target reference noise. The old
detached diagnostic is preserved, while a separate defaults-off
`differentiable_rank` mode supplies gradients through both predictions.

Prepared artifacts:

- config:
  `src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_2k.yaml`;
- Neb launcher:
  `launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_2k.sh`;
- completed run:
  `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2`;
- experiment JSON:
  `experiments/big_celebs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2.json`;
- implementation record:
  [`docs/experiments/2026-08-02_anchored_mix_sa_v3_implementation.md`](../experiments/2026-08-02_anchored_mix_sa_v3_implementation.md).

Local `photomaker` checks pass exact alpha-zero native parity (including
residual/rescale), outside-mask invariance, invalid-key exclusion,
matched/shuffled separation, first-backward gradients for every v3 role,
Hydra composition for v1/v2/v3, exact optimizer membership, checkpoint
round-trip, and manifest mismatch rejection. The launcher fails closed on a
busy Neb GPU and forces one 2k epoch.

At user request, residual-SA-v2 r6 was stopped gracefully: TERM was sent only
to verified launcher PGID `3422557` and training PGID `3422710`, both exited,
and the GPU was empty without using KILL. The first v3 submission (`r1`) then
failed before model construction, GPU use, or Comet experiment creation because
the runtime hash gate found Neb's older `train.py`; its null-key record was
preserved. After backing up the prior runtime under
`/home/niko/rsrch/runtime_backups/anchored_mix_sa_v3_20260802_1455` and syncing
the audited files, fresh run `r2` passed the integrity gate and entered
`train.py` under launcher PGID `3439415` and training PID/PGID `3439630`.
Its immutable Comet key is
[`de23193eeac9433fa090bc009f10e752`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/de23193eeac9433fa090bc009f10e752).
The run subsequently completed both fixed-96 validations and saved schema-v2
weights/full checkpoints. The local export has no warnings or errors and
resolves exactly to step 2,000. A later read-only audit found no active Neb
training process or GPU workload.

E3 fixed residual-v2's plain-PhotoMaker/dead-causal-path problem but did not
produce a quality promotion. Identity changed `0.494456 -> 0.477912`, text
similarity `25.8003 -> 26.8457`, TOPIQ-Face mean `0.717769 -> 0.722528`, and
TOPIQ-Face p10 `0.622961 -> 0.588201`. Visual changes are much stronger than
v2 and remain broadly coherent, but are predominantly prompt/expression and
generic rendering changes rather than clear identity-morphology improvements.

The training counterfactual now gives strong evidence of actual spatial
reference use: after correcting the current logger's unshuffled-zero dilution,
all 39 shuffled windows through step 1,950 have positive wrong-minus-correct
face error. Approximate conditional relative gap is `1.21%` over the run and
`0.88%` over the last 500 steps; correct/wrong prediction delta is about
`8.06%` and `6.84%` of prediction RMS. Mix falls only `.500 -> .464` while
contribution/native RMS rises `.302 -> .489`, so the branch is active and
growing rather than collapsing.

Treat E3 r2 as a positive causal control but a negative identity/quality
candidate. The next controlled training arm is E4: keep architecture, rank,
sites, data, optimizer, mix, and validation fixed; change only to
`differentiable_rank`, shuffle probability `.50`, reference weight `.10`, and
relative margin `.02`. Before launch, fix the standalone evaluator's missing
v3 processor recognition, expose an audited alpha override for the existing
checkpoint's matched/shuffle/zero/alpha0/alpha1 matrix, and log conditional
counterfactual metrics. Full evidence, code findings, planned diffs, gates,
and the updated ladder are in
[`analysis/2026-08-02_anchored_mix_sa_v3_2k_results_and_e4_plan.md`](../../analysis/2026-08-02_anchored_mix_sa_v3_2k_results_and_e4_plan.md).

### Anchored interpolation BA-v3 E4 — completed and analyzed, 2 August 2026

The controlled E4 objective arm is implemented. It inherits the exact E3 v3
architecture, rank 32, 46 sites, mix `.50/.25/.90`, data schedule, optimizer,
inference-active timestep policy, and fixed-96 validation. Its only objective
changes are matched model/loss `differentiable_rank` mode, spatial shuffle
probability `.50`, reference weight `.10`, and relative margin `.02`.

Reference diagnostics now retain the historical zero-diluted curves and add
shuffle-conditional companions. V3 sampled telemetry also logs direct
reference/native cosine and merged/native RMS. The standalone checkpoint
evaluator now recognizes v3 and exposes an audited `--ba-mix-override` for the
planned matched/shuffle/zero/alpha0/alpha1 matrix. Old E3 detached mode and
historical non-reference logging remain available.

The first E4 submission identity (`r1`) stopped before model construction,
GPU use, or Comet creation because the integrity gate had not yet reclassified
the intentionally changed trainer as an audited runtime file. Its null-key
record was preserved. After pinning both new and previous hashes, fresh run
`rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2`
passed the integrity gate under launcher PGID `3454285`. Training PID/PGID at
startup was `3454498` / `3454498`. Its immutable Comet key is
[`f72ea55eb0af44828cd6511a15ba5933`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/f72ea55eb0af44828cd6511a15ba5933).

Startup resolved the intended E4 values, constructed v3 processors on
`cuda:0` in 5.75 seconds, passed the exact 414 / 10,567,818 trainable contract,
and completed step-zero `validation_native` full-96 generation on RealVisXL in
12m12s. Face quality detected 96/96 faces. A direct Comet API audit found the
identity, text, and seven face-quality metrics plus exactly 96 image assets.
All 96 E4 step-zero PNG hashes are identical to E3 r2 step zero, so the
training-only objective toggle preserved inference initialization exactly.
Training then passed the first-three-batch gate and continued beyond batch 21
with finite loss, about `2.0-2.6 s/it`, and about 43.4 GiB GPU memory. The new
cosine/merged-RMS and shuffle-fraction series are present in Comet.
At the first complete step-50 window, effective shuffle fraction was `.54`,
conditional relative error gap `1.89%`, conditional prediction delta `10.88%`,
reference/native cosine `.806`, and merged/native RMS `.950`.
Machine originals were preserved under
`/home/niko/rsrch/runtime_backups/anchored_mix_sa_v3_e4_20260802_180639`.
Implementation and verification details are in
[`docs/experiments/2026-08-02_anchored_mix_sa_v3_e4_rank_implementation.md`](../experiments/2026-08-02_anchored_mix_sa_v3_e4_rank_implementation.md).

E4 subsequently completed all 2,000 optimizer updates, its step-2k fixed-96
validation, face-quality scoring, and checkpoint saves without an export
warning, traceback, OOM, or non-finite loss. Identity changed
`0.494456 -> 0.463905`, text `25.8003 -> 27.0073`, TOPIQ-Face mean
`0.717769 -> 0.717843`, and TOPIQ-Face p10 `0.622961 -> 0.568263`. Against
the otherwise identical E3 endpoint, E4 is lower by `0.01401` identity and
`0.01994` p10; every logged IQA mean is also lower, while text is `0.16162`
higher. The differentiable rank objective has no observed quality benefit at
2k.

The objective is nevertheless active. All 39 shuffle-conditional windows
have positive correct-reference advantage; whole-run relative separation is
`1.299%` and the final 400-step interval is about `1.081%`. Only 3/39 windows
reach the requested 2% margin. The route is not collapsing to PhotoMaker:
final branch contribution is about `.483` of native RMS, reference/native
cosine is about `.457`, and mix remains `.467`. Linear interpolation of the
increasingly rotated messages reduces merged/native RMS to `.856`, exposing a
separate quality-risk mechanism.

Do not use the below-step-zero 2k identity as a terminal rejection. Both
available hard-BA histories lost identity at 2k and recovered only around
4–8k; the clean hard arm did not cross its own step-zero identity until 8k.
Their absolute scores are not clean E4 controls because routing, ownership,
data order, and validation processor-base semantics differ, but their curve
shape invalidates a 2k-only promotion rule.

The next controlled work is: (1) run E4's existing-checkpoint
matched/shuffle/zero/alpha0/alpha1 matrix to measure held-out spatial
causality and anchoring, then (2) run the fresh E4-exact E5-L40 configuration
with a 40k process ceiling, full-96 validation/checkpoints every 2k, and 8k as
the first scientific decision point. The implemented config, Neb launcher,
and prepared record are respectively
`src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_40k.yaml`,
`launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_40k.sh`,
and
`experiments/big_celebs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1.json`.
Select the best intermediate checkpoint rather than assuming step 40k is best.
If identity is flat or falling through 8k despite persistent causality,
query-adaptive BA-v4 remains the next architecture: add a zero-initialized
branch-only target-Q LoRA while keeping native target Q/K/V frozen. The full
evidence, high-priority loss/fusion issues, key diffs, gates, and revised
D2/E5/E6 ladder are in
[`analysis/2026-08-02_anchored_mix_sa_v3_rank_2k_results_and_e5_plan.md`](../../analysis/2026-08-02_anchored_mix_sa_v3_rank_2k_results_and_e5_plan.md).
The exact launch instructions and runtime expectation are in
[`docs/experiments/2026-08-02_anchored_mix_sa_v3_e5_l40_overnight.md`](../experiments/2026-08-02_anchored_mix_sa_v3_e5_l40_overnight.md).

E5-L40 was launched on Neb at `2026-08-02T20:59:39Z` as
`rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1`.
Its immutable Comet key is
[`f5b5a7054e854137abe53c47f34ebae0`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/f5b5a7054e854137abe53c47f34ebae0).
Launcher PID/PGID is `3468188` / `3468188`; accelerate PID is `3468397`;
training PID/PGID is `3468434` / `3468434`; and the log is
`logs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1.log`.
Both sealed dataset preflights, ONNX CUDA, the audited runtime, and immutable
Comet-record creation passed. Processor construction took 5.785 seconds; exact
ownership is `414 / 10,567,818` with optimizer membership `414/414`.
Step-zero `validation_native` fixed-96 generation started on RealVisXL and
reached image `00/96`; no traceback, OOM, non-finite value, or integrity error
was present at the `2026-08-02T21:01:30Z` handoff. Leave the run active unless
a real failure or user request requires intervention.

### Anchored interpolation BA-v3 E5-L40 — no promotion through 14k, 3 August 2026

The downloaded E5-L40 package now contains complete aggregate fixed-96 metrics
through step 14k and all 96 images at steps 0, 2k, and 14k. This is an evidence
cutoff, not a completed 40k claim. Identity briefly recovers from `0.464927` at
2k to `0.497282` at 6k, only `0.002826` above initialization, then declines at
every gate to `0.447258` at 14k. TOPIQ-Face p10 remains below initialization
even at 6k. By 14k, text is up `1.4725`, generic TOPIQ is up `.02490`, and
TOPIQ-Face mean is up `.00718`, while identity is down `.04720`. The run is
improving prompt/rendering behavior rather than identity.

The architecture is mechanically healthy and causally active: exact ownership
remains `414 / 10,567,818` FP32 parameters with optimizer membership `414/414`,
face detection/coverage remain 96/96, the correct-versus-shuffled training gap
is positive, and the late branch contribution remains about `.41x` native RMS.
The decisive telemetry is learned route retreat. Phase-mean mix falls from
about `.478` in 0–2k to `.353` in 12–14k and `.345` after 14k, increasing the
native coefficient to about `.65`. Conditional reference separation later
recovers while validation identity falls, so causal sensitivity is not aligned
with useful identity generation. Do not increase the current two-sided rank
loss or branch rank as the primary repair.

The historical `rhca_big_celebs_sameid_40k_full96_r1` remains useful only as an
architectural clue: its forced/reference-dominant routing and target-side
adaptation produced a clearer multi-thousand-step identity curve. Do not resume
or exactly replay its incomplete live state; its swallowed installation error
trained about 171.29M broad parameters, including state absent from its saved
checkpoint.

The historical `sameid` processor did not mix native and reference face
self-attention outputs. It used a hard mask merge—native/background attention
outside the face and target-query/reference-KV attention inside it—with
`pose_adapt_ratio=0`. On 3 August the user explicitly selected the same
mechanistic rule for the successor: mixing can hide the intended BA failure in
the same way as target-K/V pose adaptation.

The next controlled training arm is therefore E6-H, a defaults-off
`query_adaptive_hard_sa_v4`: hard reference replacement inside the target-face
self-attention route, no alpha/gate/mix parameters, branch-only target-Q rank
16, frozen native target projections, reference K/V/output rank 32, current
rank loss disabled, detached shuffle diagnostics, CA off,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false`. Preserve v1/v2/v3
unchanged behind their existing selectors. An optional E5 checkpoint
alpha-one/shuffle/zero matrix is diagnostic only and must not gate E6-H because
those v3 projections trained while the mix was retreating. Full evidence,
prioritized code issues, implementation diffs, experiment gates, and the
updated ladder are in
[`analysis/2026-08-03_anchored_mix_sa_v3_rank_40k_through14k_results_and_e6_plan.md`](../../analysis/2026-08-03_anchored_mix_sa_v3_rank_40k_through14k_results_and_e6_plan.md).

E6-H was implemented and launched on Neb on 3 August 2026 as
`rhca_big_celebs_scheduled_v1_hard_ba_v4_q16_r32_20k_full96_r1`. The full
model startup installed all 46 intended processors and passed the exact
trainable contract: 368 tensors / 12,328,960 parameters, with only
`ref_query`, `ref_kv`, and `ref_output` roles. The prior E5 process stopped
cleanly on SIGTERM; its synchronized runtime files are recoverable under
`/home/niko/rsrch/runtime_backups/hard_ba_v4_20260803_093826`. E6-H has
launcher PGID `3518720`, training PGID `3518939`, and immutable Comet key
[`408606871a5b40c6b75d2da855b83a44`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/408606871a5b40c6b75d2da855b83a44).
The fixed step-0 full-96 validation started successfully with the
`validation_native` RealVisXL path and the run was left active; no later-step
result is implied by this startup record.
The implementation and reversibility summary is
[`docs/experiments/2026-08-03_query_adaptive_hard_sa_v4_implementation.md`](../experiments/2026-08-03_query_adaptive_hard_sa_v4_implementation.md).

### Query-adaptive hard BA-v4 — partial result through 12k, 3 August 2026

The local export for E6-H contains complete fixed-96 validation metrics through
12k, all 96 images at steps 0, 2k, and 12k, and training telemetry through
13.3k. It has zero export warnings/errors and no logged traceback, OOM, or
non-finite diagnostic. The evidence package ends during epoch 7 before a
completed 14k validation; it does not establish the reason or final live
machine state.

The clean hard route works mechanically: exact ownership remains 368 tensors /
12,328,960 FP32 parameters; target-query delta grows to about `.088`; the
reference message grows from `1.18×` to roughly `1.45×` native RMS; every role
has stable nonzero gradients; and native-face leakage is exactly zero for all
267 telemetry samples. Identity falls from `.1488` at initialization to
`.1148` at 2k, recovers to a `.2213` peak at 8k, and is `.2054` at 12k. The
0→12k gain of `.0566` is close to the historical same-ID run's `.0638` gain,
showing real clean-BA learning, but the absolute 12k historical score is still
`.3701` under its different `legacy_full_copy` validation path.

Text improves by `.524`; face detection reaches 96/96; TOPIQ-Face mean rises
by `.087`; and generic quality improves strongly. Full-panel visual review
nevertheless finds persistent colored face strips, hard-mask texture seams,
occlusion failures, duplicated glasses/goggles/features, and exaggerated
mouths/expressions. Conditional correct-versus-shuffled reference advantage
remains positive but small and declines while branch magnitude grows. The
result is: **mechanism validated and relative identity learning demonstrated,
but absolute identity, correct-reference specificity, and face integration are
not sufficient for promotion through 12k**. Detailed evidence and no-plan
analysis:
[`analysis/2026-08-03_query_adaptive_hard_sa_v4_through12k_results.md`](../../analysis/2026-08-03_query_adaptive_hard_sa_v4_through12k_results.md).

### Consolidated Large Dataset / BigCelebs BA comparison — 3 August 2026

The August result reports are consolidated in
[`analysis/2026-08-03_large_dataset_bigcelebs_ba_run_comparison.md`](../../analysis/2026-08-03_large_dataset_bigcelebs_ba_run_comparison.md).
It treats `rhca_large_dataset_sameid_40k_full96_r4` (`a99db1f…`) as the Large
Dataset base and `rhca_big_celebs_sameid_40k_full96_r1` (`569cc68…`) as the
BigCelebs base, compares nine architecture/run families, and keeps
`legacy_full_copy` and `validation_native` metrics separate. It includes exact
step-zero/2k/final metric stages, post-8k identity trajectories, representative
matched image grids, and a diagnostic final-image distance comparison against
validation-native plain PhotoMaker. The consolidation adds no successor
experiment recommendation.

A read-only Neb audit during this analysis found E5-L40 still active beyond
15k with the expected process group and GPU use. This is a transient resource
snapshot, not an instruction to assume the process remains active later.

### Large Dataset audited hard-BA suite — E1-E6 complete; E0 pair running, 4 August 2026

The controlled suite was pushed as commit `e860f9e`; all six one-A100 arms
produced their complete step-0-through-20k validation sequence on Serv. MLS
labels the terminal jobs `failed` with `error_code=0`, but every saved run has
all eleven expected 96-row ID-sim tables through step 20k. The shared parent reconstructs the
intended hard-routing behavior of
`rhca_large_dataset_sameid_40k_full96_r4` on the adjusted Large Dataset, while
adding the correctness substrate that the historical run lacked: fail-closed
BA-only ownership, schema-v2 complete trainable checkpoints, and explicit
strict `legacy_full_copy` validation. Historical `r4` remains a contextual
target rather than a perfectly matched clean control.

A 4 August source recovery found that the historical Neb and two-GPU Serv
runs used matching core code/config assets; the Serv run's lower ID trajectory
is instead confounded by global batch four at unchanged LR/update count,
distributed sampling, zero-worker random reference selection, replay/resume
with non-checkpointed RNG, and later one-GPU validation sidecars. The same
audit separates two E0 controls:

- `E0_large_ds_base_historical_r4_20k_full96_r1` deliberately reproduces the
  observed 3,080-tensor/171,294,720-parameter fail-open ownership and legacy
  incomplete checkpoints, guarded by a new exact ownership partition;
- `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1` preserves the intended
  route with strict 840-tensor/31,948,800-parameter BA-only ownership and
  complete schema-v2 checkpoints.

The user explicitly authorized an eight-A100 exception and both one-A100 E0
jobs are running on Serv alongside E1-E6. Historical E0 is MLS job
`lm-mpi-job-b7aed096-391a-4f54-b41b-6515ba895dc2`, immutable Comet key
[`a5599bd06c9346978c1fca8b8087f634`](https://www.comet.com/nikolay-2104/aug-large-ds/a5599bd06c9346978c1fca8b8087f634).
Fixed E0 is MLS job `lm-mpi-job-a0e91e1b-3e43-49c1-b65e-9f4992f33bc4`,
immutable Comet key
[`5b5cbd1584184ce1a9032dd6fafb91c5`](https://www.comet.com/nikolay-2104/aug-large-ds/5b5cbd1584184ce1a9032dd6fafb91c5).
Both passed their exact trainable-ownership and `840/840` BA optimizer gates,
their Comet comments were retrieved by immutable key, and both entered
step-zero full-96 validation. The historical arm is an attribution control and
must run uninterrupted; it is not an eligible promotion candidate. E1-E6 were
not redone. Recovery evidence,
metrics, exact files, and submission commands are in
[`docs/experiments/2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md`](../experiments/2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md).

At the matched 8k gate, historical E0 reaches ID `.36007` versus `.27338` for
fixed E0 (`+.08668`), after identical `.30187` step-zero values. The historical
arm wins 88/96 matched images and improves the mean for all eight identities.
A read-only audit of its step-8k legacy checkpoint finds that the saved generic
rank-32 update is effective only at shared SA `to_out` and ordinary CA
Q/K/V/output: all 210 outer-SA-Q/K/V LoRA-B tensors remain exactly zero, while
all 350 SA-output/CA LoRA-B tensors are nonzero. The trained PhotoMaker
`default` adapter is omitted by the historical state format and reset to its
pretrained value in alternate-base validation, so it can affect the observed
result only indirectly through training-time co-adaptation. There is still no
PhotoMaker/BA face-output mix: hard target-Q/reference-KV SA owns the target
face, while PhotoMaker ID embeddings continue to condition ordinary CA. The
four proposed next diagnostics isolate effective generic-all, ordinary-CA-only,
shared-SA-output-only, and correctly saved PhotoMaker-default-only adaptation;
ID is the primary decision metric and face-quality metric differences are not
used for ranking. Full evidence and implementation-ready design:
[`analysis/2026-08-04_e0_historical_global_adapter_id_gain_and_next_experiments.md`](../../analysis/2026-08-04_e0_historical_global_adapter_id_gain_and_next_experiments.md).

E7-E10 are implemented and were submitted to Serv on 4 August 2026 as 20k
one-A100/full-96 experiments. All four reached `running`. The defaults-off model selectors
`generic_adapter_train_scope` and `photomaker_default_train_scope` extend the
strict hard-v1 allowlist only at historically exercised outer sites: ordinary
CA Q/K/V/output and/or shared SA `to_out`. Outer SA Q/K/V remain frozen,
hard-BA routing is unchanged, and new schema-v2 manifests persist selected
generic/default tensors; defaults-off E0-E6 manifests remain compatible. Exact
planned ownership is E7 `1,540/62,423,040`, E8 `1,400/57,098,240`, E9
`980/37,273,600`, and E10 `1,540/92,897,280` tensors/parameters. Each run has a
leaf config under `src/configs/`, a durable JSON under
`experiments/large_dataset/`, and a generated one-GPU start script plus MLS
YAML under `serv_run_packages/<run_name>/`. All composition gates resolve to
20,000 steps and one scientific selector delta. They run from isolated runtime
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804`,
so the shared checkout used by live E0 jobs was not mutated. The two E0 and
four E7-E10 one-GPU requests use exactly the normal six-A100 ceiling.

Immutable startup records:

- E7: MLS `lm-mpi-job-b90da1c7-9435-4aa7-a5de-00422c7c6022`, Comet
  `e3d540a8f5c84e9db960214a1342ca04`;
- E8: MLS `lm-mpi-job-153d81de-078d-4ba5-89ec-729ea8ca01db`, Comet
  `db1326c7591e484597f3009db63af42f`;
- E9: MLS `lm-mpi-job-c2cf07ab-eaf5-4176-8283-929682dc3ec8`, Comet
  `deb40502cfc849a0aecc8e48b4eec005`;
- E10: MLS `lm-mpi-job-01a36932-2be9-413c-8cb3-cadcca9ae5ad`, Comet
  `0375f172f75c482f840317ec5ae41c05`.

Each required `saved/<run_name>/comet_experiment.json` exists, and startup
logged the planned experiment comment to project `aug-large-ds`; a read-only
immutable-key API check retrieved all four comments. Every arm passed its
exact real-model ownership count, reported all `840/840` BA processor
parameters in the optimizer, and entered step-zero full-96 validation.

An E10 visual/config/hash audit through step 4k found severe person-position
drift and duplicated/ghost faces. This is not seed drift: all 96 step-zero PNGs
are byte-identical across E7-E10, and all four saved configs use validation
seed zero. E10 alone trains the pretrained rank-64 PhotoMaker `default`
adapter at global ordinary-CA and shared-SA-output sites. Meanwhile full-96
validation has `automatic_bboxes_every_val=false`, so it reuses the original
96 cached face boxes after E10 moves the subject. The layout tendency is
learned model behavior, but training masks remain aligned because each real
Large Dataset target supplies its own transformed bbox; the stale-mask failure
is specific to fixed-mask validation/inference. Treat E10's post-drift
fixed-mask ID metrics as confounded. Recommended diagnosis is a separately
named BA-off plus dynamic-bbox validation sidecar, not a seed change or an
overwrite of canonical metrics. Full evidence:
[`analysis/2026-08-04_e10_face_position_and_static_mask_drift.md`](../../analysis/2026-08-04_e10_face_position_and_static_mask_drift.md).

E11 and E12 were then submitted on 4 August under a new user-authorized,
experiment-specific eight-A100 exception while the six E0/E7-E10 jobs remained
running. They use one A100 each and the separate isolated runtime
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E11_E12_20260804`;
no live checkout was mutated. The project therefore has exactly eight Running
or Pending A100 requests. The ceiling returns to six after E11/E12 finish or
are removed.

- E11 `E11_large_ds_ba_sa_r128_20k_full96_r1`: MLS
  `lm-mpi-job-f0ba530e-5398-4e45-982b-3e130ae0fca3`, immutable Comet key
  [`e748a5e136b3441688aaf968294612a1`](https://www.comet.com/nikolay-2104/aug-large-ds/e748a5e136b3441688aaf968294612a1).
  Its real-model gate verified exactly 840 trainable tensors and 127,795,200
  parameters.
- E12 `E12_large_ds_ba_idca_up_r256_20k_full96_r1`: MLS
  `lm-mpi-job-6590eff3-244c-4df7-bba5-1c1a5aaa9be4`, immutable Comet key
  [`d06ab51afbff4cacac1877632e26cf24`](https://www.comet.com/nikolay-2104/aug-large-ds/d06ab51afbff4cacac1877632e26cf24).
  Its real-model gate verified exactly 1,128 trainable tensors and 134,578,176
  parameters.

Both online Comet records are from `CometMLWriter`, use project
`aug-large-ds`, and contain the planned experiment comments. Before the
successful jobs, one attempt per arm failed the package's bbox existence gate
because full-96 metadata was copied one directory too deep in the new runtime.
Those attempts (`lm-mpi-job-4c25106f-...` and `lm-mpi-job-7a8dfc1d-...`)
created neither Python training processes nor Comet experiments; their empty
logs and failed local job records are preserved. Correcting the deployment
path did not change either scientific config. Detailed design and launch audit:
[`docs/experiments/2026-08-04_e11_e12_large_ds_ba_capacity_plan.md`](../experiments/2026-08-04_e11_e12_large_ds_ba_capacity_plan.md).

All arms use one A100, 20,000 optimizer steps, the fixed 96-image panel at step
0 and every 2,000 steps, and Comet project `aug-large-ds`. Branched SA is the
hard face route; branched CA is disabled, `pose_adapt_ratio=0`,
`ca_mixing_for_face=false`, and there is no PhotoMaker/native face-output mix,
gate, or residual reference blend. The six single-delta arms, in priority
order, are:

1. true reference-key masking;
2. a rank-32 branch-local output projection;
3. bbox-normalized reference-ROI warping into the target face box;
4. BA only at mid block and the first two up blocks;
5. training on the exact 50 inference-active DDIM timesteps;
6. FP32 BA trainables and optimizer state with the frozen U-Net in BF16.

Every run/package/config name carries its experiment number and key change
(`E1_large_ds_truekey` through `E6_large_ds_fp32`); Comet receives that exact
run name.

The exact configs, run names, hypotheses, decision gates, and Comet retrieval
examples are in
[`docs/experiments/2026-08-03_large_dataset_hard_ba_six_arm_design.md`](../experiments/2026-08-03_large_dataset_hard_ba_six_arm_design.md).
Each run has a one-GPU package under `serv_run_packages/`, an immutable
pre-registration JSON under `experiments/large_dataset/`, and the active
launcher `launchers/active/run_E_large_ds_hard_v1_20k_1gpu.sh`.
Startup validation rejects contract drift before training. Every validation
event retains and is configured to upload a deterministic 96-row
`id_sim__manual_val__step_<six-digit>.csv` table, and each child config logs its
delta/hypothesis as Comet Other metadata `experiment_comment`. Local config,
processor, dtype, gradient, strict-allowlist, logger/table, Python, shell, and
package smoke checks passed.

A 4 August live-log audit found that the first implementation passed `step=`
to Comet `log_table`. Comet 3.53.1 treated that unknown keyword as a pandas
CSV-format option and emitted `NDFrame.to_csv() got an unexpected keyword
argument 'step'`; training and local CSV retention continued. The local writer
now omits that keyword, encodes the validation step only in the deterministic
filename, serializes with `index=False`, and fails closed when a required table
returns no Comet asset. The production launcher's audited logger SHA now pins
that corrected local file, so the old Serv logger fails the integrity gate for
a future launch. Neither corrected file has been copied into the checkout used
by the two live E0 jobs; do not mutate that shared checkout while they run.

The saved tables were copied read-only from Serv to
`comet_data/aug_large_ds_per_image_id_backfill_20260804/` and uploaded from the
local machine through the direct Comet API, without resuming or ending any live
experiment. Byte-exact readback now verifies 71 dataframe assets on the
original keys: E1-E6 each have steps 0-20k (11 each); historical E0 has steps
0 and 2k; fixed BA-only E0 has steps 0, 2k, and 4k. Every table has 96 rows.
Use
`tools/comet/backfill_per_image_id_tables.py` after copying any later saved
tables locally; its default is a read-only dry run and `--write` is explicit.

Immutable E1-E6 records:

- E1 true-key mask: MLS
  `lm-mpi-job-a686e213-b211-48e2-bc0b-7a26ae06f307`, Comet
  `ce0c9b918d79449b92fa83ef970285c3`, exact ownership 840 tensors /
  31,948,800 parameters and optimizer membership 840/840;
- E2 branch-output LoRA: MLS
  `lm-mpi-job-555ea214-95e9-41f6-a470-68587451dcd6`, Comet
  `4c8af4e867b14377b69fa250fae5cde9`, exact ownership 980 tensors /
  37,273,600 parameters and optimizer membership 980/980.
- E3 reference ROI warp: MLS
  `lm-mpi-job-404c8887-7a3f-49c7-aa6c-7c23eebe485b`, Comet
  `9c5cbe4e49254134b4763ff7a4554c9b`, exact ownership 840 tensors /
  31,948,800 parameters and optimizer membership 840/840;
- E4 mid/up-only sites: MLS
  `lm-mpi-job-5160fbfc-be6e-478f-8099-b6dfb161880e`, Comet
  `2d77f35256844e0399c1834859a45dc7`, exact ownership 552 tensors /
  21,135,360 parameters and optimizer membership 552/552;
- E5 inference-active timesteps: MLS
  `lm-mpi-job-ce87c84b-cf29-4570-8c52-c6c2cf438bdc`, Comet
  `4a107cbc30a04a858de3e3b5c411cdca`, exact ownership 840 tensors /
  31,948,800 parameters and optimizer membership 840/840;
- E6 FP32 BA state: MLS
  `lm-mpi-job-5fbd78a3-fd27-479a-9660-aa81813db9c9`, Comet
  `9f3e20a75a0a4304b12d724693e13fbf`, exact ownership 840 tensors /
  31,948,800 parameters and optimizer membership 840/840.

All six passed config, decoded-dataset, CUDA ONNX, PyIQA, runtime-hash, Comet
registration/comment, processor-ownership, and optimizer gates, then completed
strict `legacy_full_copy` validation through step 20k on the fixed 96-image
panel. Only the two one-A100 E0 controls remain running, so the temporary
eight-A100 exception is no longer active. Serv's
preserved dirty checkout reports historical HEAD `c04970f`; the 49 suite files
were backed up, selectively synchronized from `e860f9e`, and individually
SHA-256 verified. The durable experiment JSONs carry both values so the source
provenance is explicit.

- Tasks A–D and the subsequent full-Cosmic reference-policy experiments are
  complete.
- Four final 4,000-step full-Cosmic training arms and their matched
  0/1k/2k/3k/4k full-96 validations are complete and integrity-verified.
- The initial-usage baseline and four dataset-policy arms are complete through
  20k, with exactly 96 validation images and one identity/text metric at all
  13 requested steps.
- All five initial-usage validation IDs now have seven decision-oriented
  `face_quality/` curves at all 13 steps plus an API-only per-image CSV, with
  no per-step table assets.
- The complete 20k comparison finds no promotion candidate. Top-three
  score-weighted distinct references are the best visual/identity compromise;
  highest-score distinct is the lower-confidence second. Every arm finishes
  below its peak identity score, and action/small-face attachment failures
  remain widespread.
- The strongest result in the final matrix is the **complete existing 256px
  reference asset (configured as 40% margin), pose-first captions, step
  3,000**. It reaches full-96 identity similarity `0.3606`.
- This is a matrix winner, not a production promotion. All four arms retain a
  repeatable Jisoo-specific malformed-face cluster.
- The nominal 40%/60% context and 256px/512px controls did not add source
  information: 40% and 60% are almost always the same full 256px input, and
  512px is an upscale. Legacy captions slightly improved text similarity but
  reduced identity similarity.
- Every final arm peaks on identity at step 3,000 and declines at step 4,000.
- The full-Cosmic data pipeline is mechanically healthy: 22,140 accepted
  records, deterministic reference transforms, propagated bboxes,
  target/reference inequality checks, CUDA ONNX Runtime, and reproducible
  full-96 validation.
- The best next dataset experiment is a clean highest-versus-top-three,
  fixed-256-versus-scale-curriculum factorial with one accepted-target
  manifest and no self-reference fallback. If longer training is requested,
  probe top-three first with early full-96 gates; do not run unchecked to 50k.

## Large Dataset same-ID 40k run — 27 July 2026

### New curated-dataset singleton switch — 31 July 2026

The training-ready successor is sealed at
`/home/niko/rsrch/dataset_publish/releases/v2`; the portable relative link
`dataset_publish/current -> releases/v2` selects it. Source
`current/pytorch_default.env` for the default 192px, no-singleton,
fixed-full96-disjoint manifest: 349,348 images / 68,648 identities. The
explicit 256px alternative contains 295,867 / 62,673. All selected captions
fit both SDXL tokenizers within 77 tokens and contain exactly one lowercase
`img`; all 386,092 release images were fully decoded and SHA-256 sealed.
`dataset_manifest.json` records every relative image path, size, hash, policy,
split hash, and validation audit. The include-singletons file is ablation-only.

The curated 486,103-image release is available on Neb at
`/home/niko/rsrch/dataset_publish/releases/v1`. It has one hard-linked
1024-square image tree and two loader-compatible manifests:

- `filtered_ids3_exclude_singletons.json`: 449,600 images / 77,050 IDs;
- `filtered_ids3_include_singletons.json`: 486,103 images / 113,553 IDs,
  including 36,503 true one-image identities.

Source `pytorch_exclude_singletons.env` or
`pytorch_include_singletons.env` from that release to set the manifest, image
root, and `LARGE_DATASET_SINGLETON_REFERENCE_POLICY` together. The
`LargeDatasetTrain` default remains fail-closed (`error`). The explicit `self`
mode retains distinct references for multi-image identities and uses the
target itself only where a true singleton has no alternative. A local/Neb
smoke test verified exclude mode selects a distinct reference, include mode
fails under the default policy, and include+`self` loads a singleton.

### Big Celebs 40k control — stopped on Neb, 1 August 2026

`rhca_big_celebs_sameid_40k_full96_r1` was stopped by user request after its
complete step-32k validation/checkpoint and after entering epoch 17. The
terminated launcher and training PGIDs were `3228642` and `3228842`; the GPU
process list was empty afterward. `BigCelebsTrain` reuses the Large Dataset target/reference and
transform behavior while failing closed on the sealed-release contract:
distinct same-ID references, exact `{new_face_crop, text}` records, in-bounds
faces with minimum side 192px, and exactly one lowercase `img` trigger.

The Neb launcher
`launchers/neb/start_rhca_big_celebs_sameid_40k.sh` pins release `v2`, manifest
SHA-256 `f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa`,
and `dataset_manifest.json`; it does not follow the movable `current` symlink.
Its preflight verifies the READY seal, sealed variant counts and policy, the
selected and full image-tree path sets, every metadata record, and 64 decoded
target/distinct-reference pairs before creating a Comet run. The complete v2
preflight passed for 349,348 images / 68,648 identities, and the actual loader
initialized the full manifest and loaded three boundary samples successfully.

`src/configs/big_celebs_rhca_40k.yaml` inherits the Large Dataset model and
fixed full-96 validation configuration and changes only
`train_dataset_name=big_celebs`. The launcher supplies the current standard
2,000-step epoch length × 20 epochs (40k total), validation/checkpoint gates
every 2,000 steps, `pose_adapt_ratio=0`, and `ca_mixing_for_face=false`. The
prepared immutable experiment spec is
`experiments/big_celebs/rhca_big_celebs_sameid_40k_full96_r1.json`.

The run remained healthy through the completed 32,000-step validation and
checkpoint with no traceback or CUDA OOM:

- immutable Comet key
  [`569cc685ff9144f5a9b42bf70e14e040`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/569cc685ff9144f5a9b42bf70e14e040);
- full sealed-release preflight and 64/64 decoded target/reference pairs;
- historical architecture/runtime hashes and CUDA ONNX provider passed;
- 840/840 branched-processor tensors are present in the optimizer;
- step-0 fixed validation produced exactly 96 images, followed by a complete
  96-row face-quality CSV (94 detected faces);
- identity similarity rose from `0.2841` at 2k to `0.3723` at 10k, peaked at
  `0.3817` at 18k, then declined to `0.3762` at 20k, `0.3651` at 28k, and
  `0.3552` at 30k;
- text similarity declined from `27.8118` at 2k to `26.5120` at 20k;
- the latest complete recoverable state is `checkpoint-epoch16.pth` and
  `weights-epoch16.pth` at step 32,000.

The preserved log is `logs/rhca_big_celebs_sameid_40k_full96_r1.log`.

### BigCelebs scheduled policy v1 — implemented on Neb, 1 August 2026

The opt-in implementation plan is
`docs/experiments/2026-08-01_big_celebs_dataset_usage_plan.md`. It preserves
`BigCelebsTrain` and the current launcher as the exact control, and adds:

- an offline ArcFace-centroid score and deterministic 40k schedule builder;
- `BigCelebsScheduledTrain`, which consumes pinned target/reference/flip rows;
- a separate `big_celebs_scheduled` registry entry, Hydra config, plan
  preflight, and active one-GPU launcher;
- explicit sequential DataLoader selection only for datasets that require it.

The corrected loader audit established that `BaseTrainer.inf_loop` preserves
the same shuffled DataLoader iterator across 2,000-step validation boundaries.
An uninterrupted batch-size-2 run therefore sees exactly 40,000 distinct
targets by 20k; the earlier estimated 2k-boundary resampling does not occur.

The offline score build routes each selected image through the authoritative
provenance recorded in `final_assets.jsonl`: 146,557 legacy, 200,420 EQR6,
and 2,371 Neb-incremental embeddings. These are existing read-only curation
caches for the same sealed images, not training inputs; the runtime dataset
does not open them. The complete score sidecar covers all 349,348 images and
has SHA-256 `3b7010fb9d8e05fc274c4cd5ecb1c861df7e70cc3869bc1f80d175d8d40aa8ba`.

The pinned Neb plan is
`/home/niko/rsrch/dataset_publish/sampling_policies/big_celebs_v2_policy_v1/train_40k_bs2.jsonl`,
80,000 rows with SHA-256
`e7041ca446331aeeff89baffe7bf2d678a1763722cd995cb33df8fd19b063b24`.
It contains zero self/cross-identity pairs, 76,677 unique targets, and 78,574
unique ordered pairs. Its 64-pair real-image decode preflight passed. Synthetic
regeneration is deterministic, plan corruption and unaligned recovery offsets
fail closed, a two-worker scheduled dataset batch passed, and Hydra composition
preserves the fixed model and full-96 contract.

`rhca_big_celebs_scheduled_v1_40k_full96_r1` was stopped by user request after
its complete step-14k validation/checkpoint and after entering epoch 8:

- launcher PGID `3304878`; training PID/PGID `3305027`;
- immutable Comet key
  [`7c8b04738250479aac2a186ee3c96942`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/7c8b04738250479aac2a186ee3c96942);
- both dataset preflights, CUDA ONNX, historical architecture/runtime hashes,
  and 840/840 processor-in-optimizer checks passed;
- step-0 fixed validation produced 96/96 images and a 96-row face-quality CSV
  with 94 detected faces;
- training began at `2026-08-01T11:14:34Z`, completed full-96 validation at
  step 14,000, and reached batch 176 of the following epoch;
- the latest recoverable state is `checkpoint-epoch7.pth` and
  `weights-epoch7.pth` at step 14,000;
- at approximately `2026-08-01T18:05Z`, SIGTERM was sent only to verified
  launcher PGID `3304878` and training PGID `3305027`; both groups exited and
  GPU memory returned to zero before the clean BA32 launch.

Two earlier startup attempts stopped before GPU training and before obtaining
a Comet key because the shared launcher's audited hash table had not yet been
updated for the new resume-position assertion and Neb's existing runtime
patches. Their logs are preserved as `startup_failed_integrity*`; the final
Neb integrity table passed all 12 audited files. The preserved log is
`logs/rhca_big_celebs_scheduled_v1_40k_full96_r1.log`. No commit has been made.

A separate batch-4 variant is prepared but not launched. Its pinned
160,000-row plan is `train_40k_bs4.jsonl`, SHA-256
`ff373204841cec5d06014faa7d3932442bfc256adf7fa02c63ca1e010ed2cbb8`,
with zero self/cross-identity rows and 141,685 unique targets. Config
`big_celebs_scheduled_rhca_40k_bs4` changes the train batch size to 4 while
pinning the same 2,000-step epochs, 40k budget, model, and full-96 validation.
The Neb launcher is
`launchers/neb/start_rhca_big_celebs_scheduled_sameid_40k_bs4.sh`; its default
Comet run name is `rhca_big_celebs_scheduled_v1_40k_bs4_full96_r1`. Exact
regeneration, 64 decoded plan pairs, Hydra composition, and a real two-worker
batch-4 loader smoke test passed. This is not evidence of GPU-memory safety;
the first launch still requires a monitored optimizer-batch memory gate.

`rhca_large_dataset_sameid_40k_full96_r4` was stopped by user request on
28 July 2026. It kept the
exact eligible SA-only BA model used by the recent Cosmic Large matrix and
changes only the dataset to the adjusted identity-aware Large Dataset:
47,500 1024px images, 2,561 explicit identities, and a uniformly sampled
distinct same-ID reference for every target.

- 40,000 optimizer steps, batch size 2
- validation at step 0 and every 2,000 steps through 40k
- fixed full-96 panel and default face-crop quality metrics
- terminated Neb launcher PGID `963959`; terminated training/metric PGID
  `964138`; the GPU process list was empty after SIGTERM
- Comet
  [`a99db1fb953d4511827672380e6c1645`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/a99db1fb953d4511827672380e6c1645)

Startup passed a 64/64 dataset decode preflight, exact transfer
reconciliation, ONNX CUDA, 840/840 processor-in-optimizer, 96/96 step-0
generation, face-quality scoring on all 96 inputs, and multiple optimizer
steps. Three preserved zero-step startup records exposed and fixed,
respectively, the missing Neb CUDA library path, an unsynchronized audited
validation runtime patch, and a CPU-only PyIQA subprocess. Full details and
all immutable failed IDs are in
[the run report](../experiments/2026-07-27_large_dataset_sameid_40k.md).

### Historical two-GPU Serv mirror

`rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu` completed as a
two-GPU mirror with the same model, dataset policy, local batch size 2,
40,000 synchronized optimizer updates per rank, and full-96 validation every
2,000 steps. It was originally described as differing only by Serv, world size
2, and global batch size 4, but its recovery history introduced additional
sample-order, RNG, worker, and sidecar-validation confounds.

- historical Serv continuation job:
  `lm-mpi-job-79007b8b-a9f0-41db-a15a-802ffea65658`
- Comet:
  [`db32f157e75a4798b2dfa530477c66d6`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/db32f157e75a4798b2dfa530477c66d6)
- startup passed the exact manifest and 64/64 decode preflight, CUDA ONNX
  Runtime, PyIQA 0.1.15, 840/840 processor-in-optimizer, and rank-0/rank-1
  DDP epoch synchronization;
- the original job completed all 96 step-0 images and face-quality metrics but
  stalled on its first training batch with zero updates and no checkpoint;
- the original job `lm-mpi-job-3809c1e1-9749-4dd6-9ef9-7fcc0f84e3e4`
  was stopped, while its step-0 Comet artifacts were preserved;
- the first recovery trained through step 2,000 and completed all 96 validation
  images and face-quality metrics, but rank 1 entered epoch 5 while rank 0
  wrote the checkpoint; mismatched NCCL sequence 32048 timed out and left a
  truncated, unloadable epoch-4 checkpoint;
- the repaired trainer holds all ranks around main-only checkpoint/logging and
  writes checkpoints by atomic replacement;
- the replay job reconstructed 0→2k without duplicate validation or Comet
  events, verified the full optimizer checkpoint, resumed epoch 5, and
  completed training plus full-96 validation at step 4,000;
- fresh-container model initialization is opt-in serialized across ranks to
  prevent the observed concurrent 891 MB artifact-cache race;
- after the complete step-4k validation and intact atomic epoch-8 checkpoint,
  rank 1 entered its epoch-9 iterator while rank 0 blocked at the next
  rank-0-only Comet writer boundary; this was a logging-boundary stall, not
  the prior checkpoint/NCCL race;
- the final recovery kept two-GPU training continuous from step 4k to 40k
  with 2k checkpoints, then evaluates every 6k–40k checkpoint in fresh
  single-process full-96 invocations that append to the same immutable Comet
  key;
- on 28 July, the missing live validation was traced to that intentional
  deferred-validation mode rather than loss of all Comet telemetry: training
  curves were present through step 13,650+, while validation assets stopped at
  4k. Two non-disruptive one-GPU sidecars restored live full-96 validation
  in the same Comet run: arm 0
  `lm-mpi-job-2e42c27d-d4b0-4524-b728-2758be257aea` covers
  6k,10k,...,38k and arm 1
  `lm-mpi-job-e2a7254f-1754-43d4-861a-fee26db1eabe` covers
  8k,12k,...,40k. They leave the two-GPU trainer untouched and publish
  completion markers that prevent the deferred loop from duplicating work;
- both ranks synchronized at epoch 9, 840/840 processor tensors remained
  in the optimizer, and the first recovery update completed with reduced
  loss `0.043631`.

The final Serv identity curve peaks at `.34509` at 26k and ends at `.34192` at
40k. Historical r4 peaks at `.39039` at 24k; at 32k it is `.3871` versus
`.3285` on Serv. Matching Comet source assets rule out a hidden model-code or
Hydra-model difference. The credible causes are global batch four at the same
LR/update count plus distributed sampling and replay/resume with a changed
zero-worker random reference stream. Later sidecar validation and a one-record
bbox-policy difference add measurement drift. Do not interpret the result as a
controlled claim that two GPUs alone reduce identity similarity.

The exact launcher/YAML hashes and live startup evidence are in
[the Serv mirror report](../experiments/2026-07-27_large_dataset_sameid_serv_2gpu.md).
The recovered source hashes, full metric comparison, and causal limits are in
[the 4 August forensic report](../experiments/2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md).

## Cosmic Large initial-usage continuations — 27 July 2026

The initial-usage baseline plus four dataset-policy arms completed their
4,000-step training and sealed 0/1k/2k/3k/4k full-96 validation. They continued
from the full epoch-8 optimizer/scheduler checkpoints to 20,000 steps. The old
12-image validation every 500 steps was disabled. Training runs
in 2,000-step segments, followed by sealed 96-image validation at
6k/8k/10k/12k/14k/16k/18k/20k. Training metrics append to the original
training Comet keys; validation metrics/images append to the existing
validation-only Comet keys.

| Arm | Machine/job history | Training / validation Comet keys |
|---|---|---|
| Initial self-reference baseline `_r2` | Neb PGID `387209` | `aa982105aad148bf9b2a30d3fc2149f1` / `658d22341cf24accb5a3890869e76c28` |
| Uniform distinct reference | Serv `lm-mpi-job-bb07b32f-2e2e-4c63-943c-d880274e92eb`, recovered on Neb PGID `637653` | `288ebfe3ccf74d5ea328a55b3abe31cb` / `ced6658b5b12484a9e003fe47cd0c2bf` |
| Highest-score distinct reference | Serv `lm-mpi-job-50741d46-69db-4fc3-a467-64b419230efe` | `fc3dec2223e84d49aa7c711fda968135` / `ddaeb234353b45a1ae6763f5d8a1c81f` |
| Top-three score-weighted distinct reference `_r2` | Serv `lm-mpi-job-6f171c44-2c62-4ea1-a9f0-891906b09d52` | `b7821337e24e49f388450c103553a9da` / `b9751dc78c3b460c9b2ebc50d61b2036` |
| Self-reference with 256px minimum target face | Serv `lm-mpi-job-a958a020-cd0e-4623-b428-98c5b07a0d5e` | `c6979abd46754e4ca43fae87df77eeff` / `e44bd0b7434348fa868844e96d704fca` |

All five passed immutable source/evaluation checks and a 64/64 dataset
preflight, resumed their exact existing training Comet experiments, and
entered model startup. Startup monitoring then stopped as requested. Details,
including preserved failed pre-training submissions caused by an older
full-96 record-field assumption, are in
[the 20k continuation report](../experiments/2026-07-27_cosmic_large_initial_usage_20k_continuations.md).

All five are now complete through 20k. Direct immutable-key audits found
exactly 96 images and exactly one identity/text metric at every requested step
from 0 through 20k. No completed uniform step was relogged because every
server-side step was already intact.

Uniform stopped after producing 10k because a
post-upload REST export transiently downloaded one truncated PNG. Its
server-side asset, all 96 local pixels, and both 10k metrics subsequently
verified exactly in the same validation ID. Uniform recovered from its
byte-identical 10k checkpoint on Neb under PGID `637653` and completed 20k,
retaining training key `288ebfe3ccf74d5ea328a55b3abe31cb` and validation key
`ced6658b5b12484a9e003fe47cd0c2bf`.

The minimum-face training key `c6979abd46754e4ca43fae87df77eeff`
intentionally contains the old 12-image panels. Its single canonical full-96
trajectory is validation key `e44bd0b7434348fa868844e96d704fca`.
The local continuation launcher now retries full Comet pixel/metric
verification, including transiently truncated downloads and delayed metrics.

All five validation keys now have the 27 July offline face-quality metrics at
all 13 steps. Their separate `face_quality/` sections each have exactly seven
curves: face-detection rate, TOPIQ-Face mean/p10/coverage, and
TOPIQ/MUSIQ/MANIQA means. Each also has one API-only 1,248-row per-image CSV;
there are no legacy `manual_val/face_quality/` series or per-step table assets.

The completed five-run comparison uses all 96 images at
0/4k/8k/12k/16k/20k. Quantitatively:

- top-three is the best distinct arm at 20k on identity (`0.2703`) and
  TOPIQ-Face mean (`0.6531`);
- highest has the strongest late broad distinct quality and the only
  meaningful 14–20k identity rebound (`0.2324 -> 0.2646`);
- uniform reaches the best text score (`27.1631`) but the worst 20k identity
  (`0.2428`);
- the 256px self-reference arm peaks at `0.3467` identity at 4k and falls to
  `0.2647` at 20k despite leading face-IQA means;
- every 20k endpoint is below that arm's best identity gate.

A matched visual audit of all 480 20k images ranks top-three first and highest
second, but neither is promotion-quality. Jumping, dancing, skiing, and crying
still show pasted, stretched, duplicated, or misplaced facial regions.
TOPIQ-Face coverage and p10 are more useful coherence guards than its mean;
the generic TOPIQ/MUSIQ/MANIQA models, although evaluated on the same padded
face crop, and saturated face detection can reward a crisp but grossly
malformed face. None of these IQA values was calculated on the whole image.
The full decision record and 97-page comparison PDF are in
[the 20k analysis](../experiments/2026-07-27_cosmic_large_initial_usage_20k_analysis.md).

## Completed initial-usage Cosmic Large 4k matrix — 26 July 2026

A controlled matrix reproduced only the Cosmic Large
portion of the initial `test` branch at
`6782e9d62345fe910633cc8ceec0e2fda6ec2fd1`: legacy captions, historical bbox
gate, target-as-reference, and no minimum face size. The current eligible
SA-only BA model is fixed across all arms; a composed-config comparison against
the current adapted run found no architecture, optimizer, loss, pipeline, or
BA-flag differences.

| Arm | Machine/job | Immutable training Comet key |
|---|---|---|
| Initial self-reference baseline `rhca_cosmic_initial_selfref_4k_baseline_r2` | Neb PID `196928`, PGID `196733` | `aa982105aad148bf9b2a30d3fc2149f1` |
| Uniform distinct reference | `lm-mpi-job-8f161a20-3303-40e2-8884-8c137348d9bb` | `288ebfe3ccf74d5ea328a55b3abe31cb` |
| Highest-score distinct reference | `lm-mpi-job-acd898ba-b09a-46e4-a8b5-4becae1b1280` | `fc3dec2223e84d49aa7c711fda968135` |
| Top-three score-weighted distinct reference `_r2` | `lm-mpi-job-f2a4b83f-ab44-4717-82b8-cd085307db3f` | `b7821337e24e49f388450c103553a9da` |
| Self-reference with 256px minimum target face | `lm-mpi-job-ca0acbd0-7433-42da-bcc1-39ab72a38272` | `c6979abd46754e4ca43fae87df77eeff` |

Every run passed a 64/64 decode preflight, explicitly loaded CUDA ONNX Runtime,
verified all 840/840 branched-processor tensors in the optimizer, completed
initial validation, and entered its optimizer loop. Startup monitoring then
stopped. Baseline and distinct-reference arms accept 74,754 examples; 59,143
have audited distinct-reference candidates and 15,611 retain the historical
self-reference fallback. The 256px target-face arm accepts 16,168 examples.

Each trainer targets 4,000 optimizer steps with checkpoints every 1,000 steps.
The same machine job then creates a separate Comet experiment and evaluates
steps 0/1k/2k/3k/4k on every sample in sealed
`cosmic_full96_auto_v1` (96 images per step, batch 12).

The Neb baseline's automatic evaluation chain initially stopped before Comet
creation because the configured historical bbox source did not match the
sealed SHA-256. Its manual restart subsequently completed all five endpoints:
40 batches and 480 images under Comet
`658d22341cf24accb5a3890869e76c28`.

Two failed startup identities are intentionally preserved:

- Neb `rhca_cosmic_initial_selfref_4k_baseline`,
  `a42206ee6fd241a4914aabdb436eca7f`, was stopped before step 1 because the
  CUDA provider could not load `libcudnn_adv.so.9`.
- Serv `rhca_cosmic_initial_distinct_top3softmax_4k`,
  job `lm-mpi-job-5295c0a9-49b9-43b0-8013-feabeeebe687`, Comet
  `ec43ee00375f4563b353bf701720c9eb`, stalled in model initialization and was
  deleted before processor installation or step 1. The `_r2` retry has the
  same experiment semantics and disables only the optional C++ stack
  symbolizer.

The canonical design and live provenance are in
[Cosmic Large initial-usage baseline matrix](../experiments/2026-07-26_cosmic_large_initial_usage_baseline_matrix.md).
## Dataset-policy audit addendum — 26 July 2026

A fresh live-manifest geometry audit materially narrows the interpretation of
the final reference-policy matrix:

- Full-Cosmic `face_paths` are already 256x256 face-focused assets.
- The 40% and 60% policies produce the same crop for `99.9922%` of the
  180,623 valid reference candidates; 40% already returns the full source for
  all but 14 candidates.
- The 512px arm upsamples the same at-most-256px source and therefore does not
  test additional native reference detail.
- The final matrix supports using the complete existing 256px reference asset,
  but it does not establish an optimal real-context margin or source
  resolution.
- The manifest still has no stable identity IDs joining 1024px targets:
  22,140 accepted targets map to 22,140 unique target-specific reference
  groups.

Do not run another numeric margin or 512px upscale arm on these assets. For
dataset-policy work, prioritize an audited reference-selection factorial,
stable multi-target identity grouping, target-scale/quality curricula, and
native full-scene references. The full analysis and experiment designs are in
[Cosmic Full dataset usage recommendations](../../analysis/2026-07-26_cosmic_full_dataset_usage_recommendations.md).

## Why Cosmic Large was initially unsuitable and how it became trainable

This is a central project result, not merely data-loader cleanup. The raw
Cosmic Large package could not safely be substituted into the historical
training path.

### Problems found

1. **The historical loader represented the wrong data contract.**
   `src.datasets.cosmic.CosmicDoubledTrain` combines older Cosmic metadata,
   defaults to using the target itself as the reference unless a separate
   mapping is supplied, and cannot consume the new manifest's `face_paths`,
   per-reference bboxes, and scores. It also does not include reference
   transforms in conditioning-cache identity. It remains historical replay
   code and must not be used for new full-Cosmic training.
2. **The raw manifest needed filtering and validation.** It contains 59,143
   input records, small target faces, invalid target/reference boxes, and
   records without a usable reference after filtering. The audited loader
   retains 22,140 targets with a target face of at least 192px. It removed 137
   invalid reference-bbox entries; accepted samples have 2–10 valid reference
   candidates, mean `8.158`.
3. **Target/reference leakage had to fail closed.** Self-reference lets the
   network copy the target rather than learn identity transfer. The new path
   requires a different reference path and raises an error on a collision.
   The earlier one-ID `51.jpg` training/validation overlap was also removed
   for leak-free endpoint comparisons.
4. **Tight reference geometry was unsafe at inference.** Task B showed that a
   tight 256px Cosmic reference can be copied into the target as an oversized,
   displaced, or incomplete face. Centering the same crop on a blank 1024px
   canvas did not fix occupancy; it caused catastrophic failures in about
   10/12 one-ID images. Real surrounding image context, rather than padding,
   was required.
5. **Reference image and bbox transforms could not diverge.** Cropping,
   resizing, and flipping a reference without applying the exact same
   transform to its face box corrupts the spatial BA mask. The policy and flip
   state also have to be part of the conditioning cache key.
6. **Caption order was poorly matched to long Cosmic captions.** The legacy
   order starts with facial appearance, so pose and background can be weakened
   by token truncation. The controlled pose-first mode emits
   `<class> img, pose, background, remaining appearance` and caps at 55 words.
7. **The apparent data throughput problem was partly runtime configuration.**
   `CUDA_LAUNCH_BLOCKING=1`, CPU InsightFace/ONNX Runtime fallback, and the
   initial worker settings made Serv training take 5–7 seconds/step.
8. **The package has a remaining identity-structure limitation.** The 22,140
   accepted targets resolve to 22,140 fallback reference-parent groups, so the
   manifest does not prove multiple target views per stable explicit identity.
   This is not fixed by the loader and limits claims about multi-view identity
   learning.

### Implemented, backward-compatible fix

The trainable path is isolated rather than replacing historical behavior:

- `src/datasets/cosmic_large_adapted.py` reads the real full-Cosmic manifest,
  validates boxes, filters target face size, samples a distinct valid
  reference, exposes paths/identity IDs for audits, and supports legacy or
  pose-first prompts.
- `src/datasets/reference_policy.py` performs one deterministic square crop,
  adds real context, resizes once with exact bbox propagation, optionally
  applies a diagnostic canvas, and returns a cache descriptor that includes
  the policy.
- `src/configs/cosmic_large_adapted_rhca.yaml` selects the isolated loader,
  keeps CA disabled and masked loss at every step, disables the ineffective
  one-ID-style conditioning LRU, and batches frozen conditioning preparation.
- `tools/datasets/preflight_cosmic_large_adapted.py` must pass before a run is
  registered. It checks decoded dimensions, target/reference inequality,
  bboxes, face-area fractions, exactly one PhotoMaker trigger, prompt policy,
  and cache keys on a deterministic sample.
- The successful final recipe uses diverse 1024px scene targets, a different
  same-identity **complete existing 256px reference asset** (reached through
  the nominal 40% policy), pose-first captions, ratio-zero branched SA,
  branched CA off, and full-scene references for validation/inference.
- Serv production uses asynchronous CUDA, ONNX Runtime 1.20.1 with
  `CUDAExecutionProvider`, and two loader workers. It now trains at roughly
  2.06–2.10 seconds/step and fails closed instead of silently accepting CPU
  fallback.
- The canonical full-96 protocol batches 12 prompts for one shared
  identity/reference at a time. Do not use a heterogeneous-reference batch
  until the remaining first-reference spatial setup is made truly per-sample.

Validation of the fix includes a deterministic 64/64 decode preflight, real
two-sample instantiate/collate checks, 22,140 accepted records, four complete
4k training runs without OOM, and four integrity-verified 480-image multistep
full-96 evaluations. This establishes that Cosmic Large is mechanically
trainable through the adapted path. It does not erase the remaining Jisoo
model-quality failure or the dataset's weak explicit multi-view identity
structure.

The full code audit and rationale are in
[Cosmic Large training recommendations](../../analysis/2026-07-25_cosmic_large_training_recommendations_and_experiments.md).

## Current machine and worktree snapshot

This snapshot is informational; always recheck live state before launching.

- Local checkout: `test` at `c04970f...`.
- Serv checkout:
  `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test`, `test` at
  `c04970f...`; the four recorded Cosmic initial-usage continuation jobs are
  complete.
- Neb is unavailable as of 3 August 2026. Do not attempt to inspect its
  checkout, processes, GPU, files, or credentials without the separate user
  confirmation required above.
- No resource availability is implied by this snapshot. Recheck Serv
  Running/Pending jobs before launching there.
- The worktree contains untracked experiment reports, image assets, and the
  controlled dataset artifact from prior work. They are intentional evidence;
  do not delete or overwrite them. The user has not authorized a blanket
  commit of those files.

Operational entry points:

- [Project tools](../../TOOLS.md)
- [Neb operations](../../LOCAL_NEB_SERVER_OPERATIONS.md) — dormant during the
  current outage
- [Serv/MLS operations](../../../local_scripts/serv_instructions.MD)
- [Repository rules](../../../AGENTS.md)

## Experiment history and what was learned

### 1. Original one-ID overfit and validation leakage

The original Cosmic one-ID experiment established the malformed or displaced
face failure. A validation image (`51.jpg`) was also present in the training
set. A leak-free launcher was created and the holdout reproduced correctly.
The leakage was a real comparability issue, but removing it did not explain
the main anatomy failure.

Start with the historical
[one-ID handoff](../../2026-07-24_test_branch_one_id_overfit_handoff.md) only
when reconstructing that baseline.

### 2. Tasks A–C: architecture and fixed-checkpoint diagnostics

The consolidated evidence is in
[Tasks A–D results](../../analysis/2026-07-25_cosmic_large_tasks_a_d_results_handoff.md).

| Task | Intervention | Result | Decision |
|---|---|---|---|
| A | Disable branched cross-attention while retaining branched self-attention | Scenes and bodies improved, but about 9/12 faces remained malformed; text `24.7982`, ID `0.1418` | CA amplified global corruption but was not the primary face-local cause |
| B | Reproduce fixed checkpoints exactly, then vary inference reference, CA, CFG, and identity | Tight 256px Cosmic references recreated the pathology on a healthy checkpoint; a full-scene wrong-identity reference produced mostly attached anatomy; CFG 1 collapsed to haze; null identity destroyed the face | Strongest causal evidence that the spatial reference path is active and unsafe for some tight references |
| C | Train only reference-path SA processors with CA disabled | About 9/12 anatomy failures remained; text `24.4779`, ID `0.1484` | Target/noise projection drift was not the primary cause |

Detailed reports:

- [Task A](../experiments/2026-07-25_task_a_cosmic_faceonly_noca_4k_results.md)
- [Task B](../experiments/2026-07-25_task_b_checkpoint_diagnostic_matrix_results.md)
- [Task C](../experiments/2026-07-25_task_c_cosmic_faceonly_noca_refonly_4k_results.md)

Task B passed its reproduction gates at exact filename, file-hash, and decoded
pixel equality before its interventions. Treat its causal conclusion as more
reliable than an uncontrolled visual comparison.

### 3. Task D: controlled target/reference factorial

Task D used one sealed woman-class identity and isolated training target
diversity from training reference format. Every arm used the same full-scene
reference at validation:

| Arm | Training targets | Training references | Text / ID at 4k | Visual result |
|---|---|---|---:|---|
| `multi_full` | Eight distinct scenes | Full scenes | `25.7448 / 0.2357` | Roughly 6–7/12 coherent |
| `multi_cosref` | Eight distinct scenes | Deterministic tight 256px crops | **`26.9297 / 0.3375`** | Best; two hard failures plus milder defects |
| `single_full` | One repeated scene | Full scenes | `25.0182 / 0.1853` | Worst; repeated eye/missing-feature failures |

Immutable Comet keys:

- `multi_full`: `d6363cba32e444469cde81b1d6e291af`
- `multi_cosref`: `3738f67625894b1ba583d3c7eff06c51`
- `single_full`: `ce3256602a7b4f09a82a30db616c3c3e`

Local immutable records:

- [multi_full JSON](../../comet_records/rhca_controlled_identity_factorial_multi_full_4k.json)
- [multi_cosref JSON](../../comet_records/rhca_controlled_identity_factorial_multi_cosref_4k.json)
- [single_full JSON](../../comet_records/rhca_controlled_identity_factorial_single_full_4k.json)

Task D reconciles with Task B by separating stages:

```text
tight crops used during training
    can focus identity learning and suppress nuisance scene context

tight crops injected through the current spatial path at inference
    can be copied or misregistered as literal face geometry
```

This is supported by controlled interventions but is not yet a
layer-by-layer mechanistic proof. Target diversity clearly helped. No Task D
checkpoint passed a 12/12 anatomy gate.

### 4. Initial full-Cosmic adaptation and runtime correction

The full dataset contains 59,143 input rows and 22,140 accepted training
records after the documented filters. Early Serv runs were slow because:

- production inherited `CUDA_LAUNCH_BLOCKING=1`;
- InsightFace fell back to CPU ONNX Runtime;
- the loader did not use the verified worker configuration.

The corrected runtime uses asynchronous CUDA, ONNX Runtime 1.20.1 with
`CUDAExecutionProvider`, and two training workers. Training improved from
roughly 5–7 seconds/step to roughly 2.0–2.1 seconds/step on Serv and about
1.2 seconds/step on Neb. Production jobs must fail closed if CUDA ONNX Runtime
is unavailable.

The 20%-margin pose-first and legacy full-Cosmic endpoints both failed their
canonical 96-image visual gates, primarily on Jisoo, despite plausible
aggregate metrics:

| Run | Comet key | Full-96 text / ID | Result |
|---|---|---:|---|
| 20% pose-first fast | `7c80400b23ba4a1683d4b034abdbb12c` | `27.0207 / 0.3538` | Fail: six clear Jisoo failures |
| 20% legacy fast | `0de9a9858a784373a8871e6b667316e1` | `27.1722 / 0.3374` | Fail: at least seven clear Jisoo failures |

See the
[full-Cosmic 4k/full-96 report](../experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md)
for hashes, panels, and the runtime investigation.

### 5. Drift toward plain PhotoMaker and the architectural reset

A fixed-checkpoint `pose_adapt_ratio` sweep progressively replaced spatial
reference-face K/V with target-native face K/V:

| Ratio | Full-96 text / ID | Visual result |
|---:|---:|---|
| 0.35 | `27.0094 / 0.3615` | Residual identity-specific fragments |
| 0.65 | `26.9725 / 0.4016` | Jisoo improved; Jensen still failed |
| 1.00 | `27.1979 / 0.4421` | Every identity at least 11/12 coherent |

Ratio 1.0 looked attractive, and a train-1/validate-1 run reached full-96 ID
`0.5136` with 12/12 coherent images for all eight identities. However, ratio
1.0 gives spatial reference-face K/V zero weight. A matched plain PhotoMaker
control was equally coherent and slightly better on the 12-image text and ID
metrics. The pixels differed, but there was no evidence of useful
reference-conditioned BA contribution.

This was experimental drift toward plain PhotoMaker, not a BA promotion. The
program was reset:

- runs with `pose_adapt_ratio > 0` were stopped;
- CA-mixing experiments were rejected;
- `AGENTS.md`, launchers, and run records were pinned to ratio zero and no CA
  mixing;
- subsequent experiments changed reference formatting, caption policy, or
  resolution while retaining the reference-face K/V path.

The full chronology and JSON/Comet IDs are in the
[ratio-zero reference-policy handoff](../experiments/2026-07-26_ba_ratio_zero_reference_policy_runs_handoff.md).

### 6. Ratio-zero one-ID reference-policy gates

These gates retained the intended BA route:

| Policy | Comet key | Text / ID | Result |
|---|---|---:|---|
| 40% real context, 256px | `9a947bd85a7745e29ddf329b9be16763` | `26.7409 / 0.3076` | Mostly coherent; strong improvement |
| Exact crop centered on blank 1024px canvas | `f03960bfb34a49bdba6e1503aafaf130` | `26.2995 / 0.1377` | Catastrophic in about 10/12 |
| 60% real context, 256px | `b2ef6ed73f164961b111e6c78c742eab` | See immutable record/report | Completed; motivated the full-data margin control |

The canvas experiment rejects padding as a substitute for real context.
Surrounding image content matters.

### 7. Final four-arm full-Cosmic matrix

All four source runs completed on Serv at commit
`cfa4bffebfbb46e324a7b503bdbfd786bea5e6e6`. They used all 22,140 accepted
records, ratio-zero BA, no CA mixing, active branched SA, disabled branched
CA, 840/840 processor parameters in the optimizer, and approximately
2.06–2.10 seconds/step.

| Arm | Source Comet key | Source experiment JSON |
|---|---|---|
| 40% / 256 / pose-first | `1a19fdf2793f413c9336379d3628874d` | [JSON](../../experiment_specs/rhca_cosmic_full_crop40_posefirst_4k_fast_r1.json) |
| 60% / 256 / pose-first | `a96bcbae3d2b4698a43d7ec80457586c` | [JSON](../../experiment_specs/rhca_cosmic_full_crop60_posefirst_4k_fast_r1.json) |
| 40% / 256 / legacy | `92572589d6594cd59749577fc51f5bba` | [JSON](../../experiment_specs/rhca_cosmic_full_crop40_legacy_4k_fast_r1.json) |
| 40% / 512 / pose-first | `c354369af45b4c9da84f1124cf3e9a88` | [JSON](../../experiment_specs/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1.json) |

The corresponding Serv packages are under
`serv_run_packages/<run_name>/`.

### 8. Final multistep full-96 result

Validation commit `c04970f...` evaluates steps 0, 1,000, 2,000, 3,000, and
4,000 in one Comet run. Each arm produced 96 images at every step using batch
size 12. Step 0 is byte-identical across all arms; all source reproduction and
Comet image checks passed.

| Step | 40% / 256 / pose-first | 60% / 256 / pose-first | 40% / 256 / legacy | 40% / 512 / pose-first |
|---:|---:|---:|---:|---:|
| 0 | `26.3205 / 0.2999` | `26.3205 / 0.2999` | `26.3205 / 0.2999` | `26.3205 / 0.2999` |
| 1,000 | `27.1279 / 0.2972` | `27.0369 / 0.2947` | `27.2619 / 0.2961` | `27.0129 / 0.2872` |
| 2,000 | **`26.8722 / 0.3465`** | `27.0072 / 0.3423` | `27.1172 / 0.3353` | `26.9036 / 0.3390` |
| 3,000 | **`26.6846 / 0.3606`** | `26.7720 / 0.3575` | `27.0054 / 0.3457` | `26.7827 / 0.3545` |
| 4,000 | `26.9992 / 0.3422` | `26.8936 / 0.3458` | `27.1810 / 0.3316` | `26.9494 / 0.3418` |

Validation provenance:

| Arm | Validation Comet key | Immutable local record |
|---|---|---|
| 40% / 256 / pose-first | `519f9ecac929417e8073e7b3cc953c2d` | [JSON](../../comet_records/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 60% / 256 / pose-first | `df99f4b0bb9a4676bd6783d1bc611c6b` | [JSON](../../comet_records/rhca_cosmic_full_crop60_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 40% / 256 / legacy | `dfb06576f4104d969b08c59b06ec7834` | [JSON](../../comet_records/rhca_cosmic_full_crop40_legacy_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 40% / 512 / pose-first | `00cfd945fdcf44dbbd8914b42f139300` | [JSON](../../comet_records/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |

The complete review and contact-sheet links are in
[the final four-run handoff](../experiments/2026-07-26_current_four_full_cosmic_4k_runs_handoff.md).

Observed conclusions:

1. The useful identity gain appears between 1k and 2k.
2. Every arm peaks at 3k and regresses at 4k.
3. The 40%-labelled full-256 / pose-first arm is the best identity/text
   trade-off.
4. The 60% and 512px arms add no native data information: the former is
   almost always the same crop and the latter is an upscale.
5. Legacy captions trade identity for a small text-score gain.
6. All arms avoid the original widespread pasted/displaced face failure.
7. All arms retain a strong Jisoo-specific failure cluster. Marion and small
   action faces also remain weaker, but less catastrophically.

## What the current results do and do not establish

### Established by observed evidence

- Tight reference formatting at inference can causally trigger copied or
  misregistered face structure.
- Real surrounding context is much safer than blank padding.
- Cropped references can still be beneficial during training.
- Diverse target views are better than a repeated single target.
- Aggregate ID similarity can reward identity fragments and cannot replace a
  per-image anatomy review.
- The full-Cosmic loader, crop/bbox propagation, checkpoint evaluation, Comet
  export, and multistep full-96 path work correctly under the audited
  protocol.
- Using the complete existing 256px reference asset is sufficient for this
  matrix. The current assets cannot test wider real context or higher native
  resolution through larger margins or output resizing.
- A 3k stopping point is better than 4k for this matrix.

### Not established

- That the complete-256 reference policy generalizes beyond the eight full-96
  identities.
- That the Jisoo issue is caused by one specific reference image, bbox,
  PhotoMaker embedding, BA layer, or timestep.
- That branched cross-attention cannot be made useful. It was disabled in the
  successful matrix because earlier CA-on runs caused additional corruption.
- That combined SA+CA BA is healthy.
- That a long run will outperform the 3k candidate.
- That the current candidate beats a fully matched plain PhotoMaker baseline
  on full-96 while retaining a demonstrably useful reference-conditioned BA
  contribution.

## Recommended next experiments

### Priority 1 — clean reference-selection × target-scale factorial

Use one immutable accepted-target manifest with **no self-reference fallback**.
Compare highest-score versus top-three score-weighted distinct references,
crossed with:

1. target face ≥256px throughout; and
2. a scale-balanced curriculum that oversamples ≥256px faces for 4–6k, then
   introduces 192–255px faces in balanced bins.

Audit reference candidates jointly on ArcFace score, pose difference,
occlusion, blur, and native resolution. Highest ArcFace alone may select a
near-duplicate view that encourages literal spatial copying. Do not repeat
40%/60% margin or 512px-upscale arms on the current 256px face assets.

### Priority 2 — bounded top-three continuation

If the user wants to test whether the late recovery continues, resume
top-three from its exact 20k checkpoint. Highest-score is an optional second
arm. Give the run a 50k maximum budget but validate at 22/24/28/32k and stop
unless identity exceeds 20k, TOPIQ-Face coverage/p10 do not regress, and the
fixed jumping/dancing/skiing/reading hard set visibly improves. Do not run
either arm unchecked to 50k.

### Priority 3 — reference-conditioned BA routing/alignment

First localize the failure with fixed-checkpoint branched-SA
layer/resolution/timestep-window ablations. Then test one-variable changes:

- a bounded per-layer/timestep gate on the reference-branch residual merge,
  regularized against collapsing the reference contribution to zero; or
- bbox-relative coordinate remapping of reference K/V and branch masks into
  the target-face frame.

Preserve target queries, explicit reference K/V, reference-face K/V weight
1.0, `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and CA-off. Do not use
target K/V substitution as a fix.

### Priority 4 — matched plain PhotoMaker and broader identity gates

Run exact full-96 inputs as plain PhotoMaker against step 0 and the selected BA
checkpoint, then add identities outside full-96 with difficult
hair/occlusion and small/action faces. This separates failures inherited from
PhotoMaker/reference preparation from failures amplified by spatial BA.

## Code and launch entry points

Run all Hydra and training commands from `diffusion_template/`.

Current data/reference policy:

- `src/datasets/cosmic_large_adapted.py`
- `src/datasets/reference_policy.py`
- `src/configs/cosmic_large_adapted_rhca.yaml`
- `tools/datasets/preflight_cosmic_large_adapted.py`

Current training launcher:

- `launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh`

Current full-96 evaluation:

- `launchers/active/run_rhca_cosmic_full96_eval_1gpu.sh`
- `src/configs/cosmic_large_adapted_full96_eval_rhca.yaml`
- `src/configs/cosmic_large_adapted_full96_multistep_eval_rhca.yaml`
- `tools/inference/full96_protocol.py`
- `tools/inference/finalize_multistep_full96_eval_record.py`

Controlled one-ID reference policies:

- `launchers/active/run_rhca_cosmic_one_id_reference_policy_4k_1gpu.sh`
- `src/configs/controlled_identity_reference_policy_rhca.yaml`

Task D controlled factorial:

- `launchers/active/run_rhca_controlled_identity_factorial_4k_1gpu.sh`
- `src/configs/controlled_identity_factorial_rhca.yaml`
- `src/datasets/controlled_identity_factorial.py`

Before changing the attention subsystem, search the relevant files for
`AICODE-NOTE:`, `AICODE-TODO:`, and `AICODE-QUESTION:` anchors. Keep new
behavior behind toggles and verify both old and new composition.

## Experiment and Comet protocol

Every new experiment must have:

1. a unique run name and output directory;
2. a local experiment JSON describing the hypothesis, fixed controls,
   changed variables, machine, launcher/package, gates, and status;
3. `saved/<run_name>/comet_experiment.json` created at startup;
4. an immutable Comet experiment key copied into the local JSON;
5. metrics and images retrieved by immutable key, never by display name;
6. exact image counts and exact requested steps;
7. visual review separated from metric evidence;
8. preserved failed-start records rather than reused contaminated run names.

Use:

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template
python tools/comet/comet_experiment.py --help
```

The local `comet_records/` cache is ignored by Git but contains the current
immutable validation records. The durable experiment specifications and Serv
packages are under `experiment_specs/`, `experiments/`, and
`serv_run_packages/`.

## Machine rules

### Neb

- Unavailable as of 3 August 2026; do not access or use it for any purpose.
- If the user asks to use Neb, obtain separate explicit confirmation that it
  is working again before attempting a read-only connectivity check.
- The older Neb capacity, environment, and synchronization guidance is
  dormant historical information until the outage restriction is superseded.

### Serv

- Write only below
  `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/`.
- Count this project's own Running and Pending A100 requests by actual GPU
  count. The ceiling is six GPUs; other users do not count.
- Pending is a successful submission.
- If MLS rejects/discards a request for allocation limits, do not retry unless
  the user asks.
- Do not set `CUDA_LAUNCH_BLOCKING=1` in production.
- Require the CUDA ONNX Runtime overlay and fail closed rather than accepting
  CPU fallback.
- Do not sync or mutate a code checkout being read by live jobs; previous NFS
  stale-handle failures came from changing shared files during execution.
  Use immutable run packages or wait for jobs to finish.

## Known implementation caveats

- Alternate-base validation previously installed processors before
  propagating architecture flags. Commit `5e55450b...` fixed flag propagation
  and the controlled-validation DataLoader.
- Some historical validation emitted an
  `AttnProcessor2_0 ... has no attribute parameters` warning after installing
  self-attention processors. The SA path was present, but the catch-all can
  hide future partial-installation failures. New critical evaluations should
  assert exact installed processor counts.
- The original 12-image endpoint is not enough for promotion. Eight-identity
  full-96 exposed failures that the Eddie-only panel missed.
- Step-0 images must be identical across matched arms. If they differ, the
  validation contract has drifted.
- Do not invent CA-on weights for a CA-off checkpoint.
- Do not compare 12-image aggregate metrics directly with 96-image metrics.

## Face-quality metric backfill status (27 July 2026)

- Neb baseline validation key `658d22341cf24accb5a3890869e76c28`
  has seven compact `face_quality/` curves at all 13 full-96 steps. Its
  1,248-row per-image CSV is retained locally under
  `analysis/assets/face_quality/neb_baseline_658d22341cf24accb5a3890869e76c28/`
  and attached as API asset `26160c7a6a18404a8087de4bdb67290e`.
- The four Serv validation keys passed preflight with exactly 96 images at all
  13 steps and no existing face-quality metrics/tables. Both planned Serv GPU
  submissions were rejected before job creation with
  `WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE`.
- Under an explicit exception, Serv staged all 4,992 images and transferred
  them to Neb with per-file size/SHA-256/PIL verification. Neb processed the
  four runs sequentially under PGID `812861`.
- All four validation keys now have exactly seven `face_quality/` curves × 13
  steps plus one API-only 1,248-row per-image CSV asset; the independent audit
  found zero legacy metrics and zero table assets.
- A post-run audit found no credential or key file/pattern in either staging
  tree; Serv and Neb retained distinct machine-local `.env` files.
- A bounded Serv compatibility smoke test was prepared for the uniform
  step-0/full-96 panel using one A100 and no Comet writes. Its single submission
  at 14:07 UTC was rejected before job creation with
  `WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE` and cancelled without retry, as
  requested. Record:
  `experiments/cosmic_large_continuation/serv_face_quality_uniform_step0_smoke.json`.
- Durable status:
  `experiments/cosmic_large_continuation/serv_four_validation_face_quality_backfill.json`.

## Default in-pipeline validation (27 July 2026)

- The standard trainer configuration now validates at step 0 and every 2,000
  optimizer steps. The interval must divide `trainer.epoch_len` exactly.
- New training launchers default to 2,000 optimizer steps per epoch, so each
  epoch ends at a validation/checkpoint gate. Historical scripts that address
  immutable 500-step checkpoint epoch numbers pin `TRAIN_EPOCH_LEN=500`.
- Standard Cosmic Large configs use the fixed 96-image `manual_val` panel and
  one generated image per item. Explicit historical one-identity protocols
  remain 12-image exceptions.
- The canonical seven face-quality metrics run at every actual validation
  event by default behind `trainer.face_quality.enabled`. They use the same
  standalone PyIQA 0.1.15 scorer and definitions as the completed backfill.
- Comet receives seven `face_quality/` scalar curves and one API-only
  per-image CSV asset per validation step; no table is created.
- Full configuration and machine-environment behavior are documented in
  `docs/validation_protocol.md`.

## Detailed document index

- [Tasks A–D implementation request](../../analysis/2026-07-24_cosmic_large_next_steps_implementation_handoff.md)
- [Tasks A–D consolidated results](../../analysis/2026-07-25_cosmic_large_tasks_a_d_results_handoff.md)
- [Cosmic Large recommendations and experiment design](../../analysis/2026-07-25_cosmic_large_training_recommendations_and_experiments.md)
- [Experiment launch plan](../../analysis/2026-07-25_cosmic_large_experiment_launch_plan.md)
- [Full-Cosmic 4k/full-96 report](../experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md)
- [Architectural reset and ratio-zero runs](../experiments/2026-07-26_ba_ratio_zero_reference_policy_runs_handoff.md)
- [Final four-run multistep full-96 report](../experiments/2026-07-26_current_four_full_cosmic_4k_runs_handoff.md)
- [Initial-usage five-run 20k analysis and full-96 PDF](../experiments/2026-07-27_cosmic_large_initial_usage_20k_analysis.md)
- [Initial-usage 20k continuation provenance](../experiments/2026-07-27_cosmic_large_initial_usage_20k_continuations.md)
- [Serv face-quality backfill](../experiments/2026-07-27_cosmic_large_serv_face_quality_backfill.md)

Representative validation images are stored beside those reports under
`docs/experiments/assets/`. The final four-run multistep contact sheets are
under `docs/experiments/assets/2026-07-26_full96_multistep/`.

## New-agent startup checklist

1. Read this file completely.
2. Read `AGENTS.md` and `TOOLS.md`.
3. Check branch, commit, and dirty status locally and on the target machine.
4. Do not access Neb during the current outage; recheck only Serv
   Running/Pending allocations when Serv work is requested.
5. Identify experiments only through their JSON plus immutable Comet key.
6. Inspect the five-run 20k PDF, especially the jumping/dancing/skiing hard
   cases, before proposing a long run.
7. State whether a proposed experiment preserves effective
   reference-conditioned BA.
8. Start with the clean dataset factorial unless new user direction supersedes
   it; if longer training is requested, gate top-three early.
9. Update this file when a material result changes the current decision.

## E10 dynamic-mask correction (5 August 2026)

- Completed source run `E10_large_ds_pmdefault_effective_r64_20k_full96_r1`
  has ten intact checkpoints (2k-20k) and immutable Comet key
  `0375f172f75c482f840317ec5ae41c05`.
- The fixed-mask validation is confounded because E10's trained effective
  PhotoMaker-default adapters move the generated subject while cached step-0
  boxes remain fixed. This is not seed drift.
- Sidecar `E10V_large_ds_dynamicmask_reval_2k20k_full96_r1` regenerates each
  checkpoint with a checkpoint-current BA-off locator pass and a fresh CPU
  face box, then runs BA and recalculates the full-96 metrics.
- Publication is transactional at the workflow level: all ten steps stage and
  pass integrity checks before any Comet mutation. The publisher preserves
  step 0, backs up original metric histories/asset IDs, replaces only 2k-20k
  validation images/tables and the nine affected validation curves, and
  verifies the resulting Comet state.
- Exact design and paths:
  `analysis/2026-08-05_e10_dynamic_mask_checkpoint_revalidation.md`.
- First submission `lm-mpi-job-733198c2-2c48-4f37-a67c-4f9d0f663610`
  failed after completing the full step-2k dynamic-mask generation because
  the face-quality subprocess could not see CUDA. No Comet mutation occurred.
  The corrected resume reuses the staged 96 images/boxes/ID table and runs
  face-quality scoring on CPU; image generation remains on one A100.

## E0-E12 completed full-96 analysis and next parallel suite (5 August 2026)

The two E0 controls and E1-E12 now have a single audited local export and a
full per-image/visual analysis:

- export index:
  `comet_data/aug-large-ds_E0-E12_20260805/README.md`;
- report, charts, selected comparisons, and implementation-ready six-run plan:
  `comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS.md`.

Current observed conclusion:

- historical E0 remains the identity winner at `.37083 @14k` and
  `.36889 @20k`;
- E11 rank-128 spatial BA is the best clean arm at `.32704 @8k` and
  `.32167 @20k`;
- historical E0 beats E11 by `.04723` on the paired 20k panel and wins 72/96
  images, but its gain is concentrated in Keanu/Jennie and
  skiing/laughing/night-ride cells rather than being uniform;
- E10 fixed-mask identity values are not rankable after subject-position
  drift; the visual layout failure is real even when boxes are corrected;
- E12 is incomplete at 12k but conclusively rejects hard ID-token-only face CA
  replacement because of severe face-patch artifacts and falling identity;
- the leading explanation for historical E0 is joint BA/generic/PhotoMaker
  training-time co-adaptation. E7-E10 and E11 show that the isolated effective
  adapters or wider BA alone do not reproduce it.

For the requested next-six design, the report recommends prioritizing the new
joint-mechanism suite over the earlier data-factorial-first ordering. E13/E14
are explicit shadow-co-adaptation mechanism arms; E15-E18 are fully persisted
promotion arms covering protected joint training, a predicted-x0
PhotoMaker-CLIP identity proxy, bounded residual identity-token CA, and
deterministic decoupled multi-reference training. Every arm keeps
target-query/same-ID-reference-KV spatial BA, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`.

Implementation entry points:

- configs: `src/configs/E13_*` through `src/configs/E18_*`, all inheriting
  `large_dataset_joint_r128_24k.yaml`;
- controlled launcher: `launchers/active/run_E13_E18_large_ds_24k_1gpu.sh`;
- immutable records: `experiments/large_dataset/E13_*` through `E18_*`;
- one-A100 MLS YAMLs and startup wrappers:
  `serv_run_packages/E13_*` through `serv_run_packages/E18_*`;
- E17's residual processor is defaults-off and separate from E12's hard v2;
- E18 uses an exact 48,000-row deterministic sequential dataset and passes all
  identity references to PhotoMaker while preserving `ref_images[0]` as the
  sole spatial latent/KV reference.

Launch state on 5 August 2026:

- E13 r1 (`lm-mpi-job-be2c88f8-7008-4828-a41e-abaaa7f47839`, immutable Comet
  key `ce847065760c47e7bc16530238b39792`) passed both exact ownership gates at
  2,240 tensors / 219,217,920 parameters and generated the deterministic
  96-image step-0 panel, but then failed before optimizer step 1 because its
  rank-0 PyIQA subprocess requested unavailable CUDA.
- E13 r2 (`lm-mpi-job-7f4b3b5b-77d7-496d-990d-8f7d5a36b9f4`, immutable
  Comet key `251d7696031846a0a89f6dcaabf47d47`) was stopped during its slow
  step-0 CPU face-quality pass at the user's request. E13 r3
  (`lm-mpi-job-c5cc58ca-aaab-4e7f-b8ed-bd2706859f79`, immutable Comet key
  `2397182200d64c56bf70b85398144cb9`) passed both exact ownership gates and
  generated all 96 step-0 images, then failed before optimizer step 1 because
  its scorer child reported CUDA unavailable.
- E14 r2 (`lm-mpi-job-f1132bd5-d531-4cf0-b7e7-f2179f0b1240`, immutable Comet
  key `f3ccb0c866b343198ce936cf342e8633`) failed at the same post-step-0
  face-quality CUDA boundary after generating all 96 images; no optimizer
  step ran.
- E12 proves that the same four-metric PyIQA 0.1.15 scorer repeatedly works on
  CUDA in this Serv environment. The launcher-level parent/child CUDA probe
  also passed on E13 r3 and E14 r4, but it ran before Accelerate and the large
  joint validation model were initialized and therefore did not reproduce the
  failure boundary. The replacement contract stages the exact 96 validation
  PNGs plus checksummed manifests at every validation event without importing
  or running PyIQA. Only after Accelerate exits successfully does a standalone
  CUDA scorer combine all 13 steps, compute the unchanged canonical metrics,
  and backfill the seven Comet curves. Its `--nonfatal` wrapper writes
  `post_training_face_quality/status.json` and returns success on a scorer or
  Comet error, so it cannot invalidate completed training or checkpoints.
- E14 r3 (`lm-mpi-job-3c9b21e1-6101-4d86-8d81-d51d4bc8174d`, immutable
  Comet key `230ad9dedc674d54884fcb150ac8446b`) was stopped during its slow
  step-0 CPU face-quality pass at the user's request. E14 r4
  (`lm-mpi-job-bfbdc73f-fb12-4ce5-862c-d1be4d37ff64`, immutable Comet key
  `90b31a80d20d4083b8d32873ac170881`) passed its matching ownership gates and
  generated all 96 step-0 images, then failed at the same scorer-child CUDA
  boundary before optimizer step 1.
- E13 r4 and the E14 deferred retries retain the exact 24k/full-96 training
  contracts and do not construct PyIQA models at startup or during training.
  Both exact local/Serv config gates pass. Their original 19:20/19:21 attempts
  were rejected while the workspace was full.
  After explicit user reauthorization, E13 was accepted once as
  `lm-mpi-job-62127e33-1dec-4daa-a9ed-02e30b9b0f8f` but failed before Comet or
  training because an interrupted fast-forward left the exact `ebf1ac8` tree
  staged over an `84fb6e9` HEAD. The tree equality was verified and the ref
  update completed without changing files. Corrected E13 is Running as
  `lm-mpi-job-57f10bf3-5010-4c97-aac6-26164c84defb` from clean commit
  `ebf1ac8`, immutable Comet key `1cc0a02371094b24a6a02a4cc649f10c`.
  E14 r5 was later accepted as
  `lm-mpi-job-57f41a1b-d3d6-49ee-88a7-a7b84b6fb4ca`, but failed before Comet
  or training because E13's live validation created untracked bbox-cache files
  in their shared checkout and the clean guard correctly rejected it. E14 r6
  uses its own clean runtime and is Running as
  `lm-mpi-job-e813fc81-beb6-4421-8e30-d175392e82a6`, immutable Comet key
  `f53c2a2f130247a1b817c820ba7615ae`.
- E15-E18 r2 were packaged earlier with in-process rank-0 CUDA scoring from
  commit `57257ac68ae2b9503e4899d43a082e92cf4cb1c7`. Their isolated runtime is
  `runtime_worktrees/rsrch_test_E15_E18_gpu_20260805`. Already-running E15/E16
  processes cannot inherit the later deferred implementation.
- The user explicitly reauthorized another E15-E18 submission attempt after
  checking E13/E14. E15 was accepted and is Running as
  `lm-mpi-job-3406b3fc-8369-48c9-9a20-c838417f4e92`, immutable Comet key
  `f320234a54624aa6a1a100307691b627`; its exact four-model in-process CUDA
  smoke, config gate, and 64-image dataset gate passed. E16 was accepted and
  is Running as `lm-mpi-job-f195abb1-c656-4d5f-b1fc-fd2f7d78f504`, immutable
  Comet key `4561fb0de8c64b3da8663e3f4c37589c`. Both completed the full-96 step-0
  panel and four-model in-process CUDA face-quality scoring (95 detected
  faces), then entered training; latest observed progress was at least step
  300 for E15 and step 50 for E16. Earlier E17/E18 r2 attempts were rejected
  before job creation while the workspace was full. Their deferred replacements
  use separate clean runtimes to prevent concurrent bbox-cache writes from
  tripping another job's clean guard. E17 r3
  (`lm-mpi-job-58b43e25-2c54-4c03-be75-9d4b46cd4418`, immutable Comet key
  `ed09370a12f8429e8e0556c387afdb1d`) failed before optimizer step 1 during
  the first step-0 validation denoising call: its validation U-Net correctly
  contained 36 residual identity-CA v3 processors, but the temporary pipeline
  omitted all v3 selector/rank/gate attributes and computed `expected=[]`.
  Fix commit `de34683` propagates those five attributes before denoising. E17
  r4 (`lm-mpi-job-363a85af-651e-45b1-9e4c-41740fe14cf0`, immutable Comet key
  `2b0581a252ae45ee9e6d24eb6fbad9c4`) passed the exact
  2,348-tensor/224,624,676-parameter contracts and full-96 step-0 validation,
  then failed on training batch 2 after batch 1 reported finite loss 0.067981.
  Its zero-initialized residual output used `sqrt(mean(delta^2))` before an RMS
  clamp: forward was finite, but `sqrt'(0)` produced NaN gradients on the first
  backward pass. Commit `1a88f6a` clamps mean-square before `sqrt`, preserving
  the 1e-6 RMS floor while making zero-init gradients finite; an actual
  processor forward/backward smoke passed on Serv in FP32. E17 r5 is Running
  as `lm-mpi-job-46422bb5-61e0-45e1-a188-002ba7d0edf3`, immutable Comet key
  `08ecedf8e058461abe952077f9623ab8`, from isolated runtime
  `runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805`. E18 r4 is Running as
  `lm-mpi-job-53666768-76c5-46f4-b488-895d2d7c74ab`, immutable Comet key
  `b9e118da6dc94cd9b3849566e18c67ff`. The unsubmitted shared-runtime E18 r3
  package is superseded and must not be launched.
- E14-E18 r1 are failed/stopped startup attempts and must not be interpreted as
  experiment results. Their immutable keys and failure provenance are kept in
  their JSON records. The root cause was a missing `loss_kind:
  branched_reference`, which made the training entry point replace the
  configured protected loss with `MaskedDiffusionLoss` while retaining
  protected-loss kwargs. The r2 configs fail closed on both selector and
  target and explicitly instantiate successfully in the Serv environment.

The original E11/E12 worktree was left untouched because it was dirty and E12
was still Running. E13 r2 used clean isolated worktree
`runtime_worktrees/rsrch_test_E13_r2_20260805`; its failed r1 remains in
`runtime_worktrees/rsrch_test_E13_E18_20260805`. E14 r3 and launch-ready
E15-E18 used clean worktree
`runtime_worktrees/rsrch_test_E14_E18_cpu_20260805`; E13 r3 and E14 r4 run
from `runtime_worktrees/rsrch_test_E13_E14_gpu_20260805`. The prior E14 r2 artifacts
remain in `runtime_worktrees/rsrch_test_E14_E18_r2_20260805`. The canonical fixed full-96
bbox file was copied without modification to
`datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json`
with SHA-256
`a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614` so clean
runtime clones can use the exact protocol.

### E13-E18 completion update — 6 August 2026

Live MLS and log inspection found no crash in any selected final revision.
E13 r4, E15 r2, and E16 r2 completed their 24k jobs with `error_code=0`; E13
also completed deferred face-quality finalization. E14 r6 completed 24k
training and remained in deferred face-quality finalization. E17 r5 remained
healthy beyond 16k, and E18 r4 remained healthy beyond 21k. The immutable
Comet IDs, experiment records, scientific-description report, and exact
startup scripts/YAMLs are indexed in
[`analysis/2026-08-06_e13_e18_successful_run_index.md`](../../analysis/2026-08-06_e13_e18_successful_run_index.md).

### E13-E18 metric/visual result and next-suite recommendation — 6 August 2026

This result supersedes the earlier E0-E12 section's statement that historical
E0 is the current fixed-panel leader. On the unchanged 96-image `manual_val`
panel:

- E13 shadow co-adaptation reaches **`.39980 @24k`**. Against historical
  E0's `.37083 @14k`, the matched best-checkpoint delta is `+.02897`, with
  67/96 wins and a paired panel-bootstrap interval of `+.01756` to `+.04029`.
- E14 shadow co-adaptation plus protected reconstruction reaches
  **`.39185 @20k`**. Its E0 delta is `+.02102`, with 65/96 wins and interval
  `+.00856` to `+.03306`.
- Both results also beat E0 at the common 8k and 20k checkpoints. The new tier
  is therefore not an artifact of selecting different peak steps.
- Persisting the trained PhotoMaker-default path is the main failure: E15's
  best `.31787` is `-.07398` versus E14. Visuals keep body position but show
  strong unprompted open-mouth/expression drift.
- E16's predicted-x0 PhotoMaker-CLIP proxy and E17's bounded residual ID-CA do
  not improve E15 on mean `ID_sim`. E17 also lacks its intended gate/residual
  Comet telemetry because the writer metric list was not extended.
- E18's identity-balanced multi-reference package is the only strong
  persistent-route positive: `.35522 @12k`, `+.03735` versus E15, 65/96 wins,
  and improvements for 7/8 identities and 11/12 prompts. The bundle should be
  transferred onto E13's shadow route; E18 itself remains below E13 because
  it persists the damaging trained default path.
- E13/E14 faces remain attached to the intended bodies without E10-like
  relocation or E12-like face plates in the reviewed best panels. The main
  recurring visual failure is skiing eyewear/goggle duplication; crying
  hand-eye fusion and small jumping faces also remain hard.

Comet data were refreshed through 16:42 UTC. E17's latest complete validation
was 16k (`.30773`), and E18's was 22k (`.35401`); neither changes its earlier
best checkpoint.

The full aggregate, identity, prompt, and identity-by-prompt tables; visual
contact sheets; reproducible analysis assets; and implementation-ready E19-E24
parallel one-GPU plan are in
[`analysis/2026-08-06_e13_e18_results_and_next_experiments.md`](../../analysis/2026-08-06_e13_e18_results_and_next_experiments.md).
The priority order is: (1) E13 plus E18 balanced multi-reference conditioning,
(2) E13 plus branch-local output rank32, (3) their 2x2 combination, (4) a
verified ArcFace-like predicted-x0 loss rather than E16's CLIP proxy, (5)
earlier LR decay beginning at 8k, and (6) exact every-other-step masked/full
loss.

### E19-E24 implementation and launch update — 6 August 2026

The recommended suite is implemented and its corrected `r2` jobs are Running
on Serv. All six use separate hash-verified source trees under
`runtime_sources_e19_e24_v3`, the fixed full-96 protocol, one A100 each, and
immutable Comet records. The user explicitly authorized the suite-specific
eight-A100 exception; with E17/E18 still Running, the six new jobs bring the
project total to exactly eight.

- E19 r2: job `lm-mpi-job-8ad95723-ea0a-4bfb-a920-6904d91eb993`, Comet
  `3280232a45ef4ea2ae68c8deff3b81c1`.
- E20 r2: job `lm-mpi-job-51cd67d6-c28c-4185-9595-b37a273e71c1`, Comet
  `4084c35600ae4ad3904446e5f4d2de92`.
- E21 r2: job `lm-mpi-job-e11a4015-6493-4313-8e82-4c6525e02fec`, Comet
  `3ef78907f60a4f5cbd7727fc5be7143e`.
- E22 r2: job `lm-mpi-job-69206471-725e-4a97-b33f-a088e8fb6576`, Comet
  `5a91be0df76f4966be5c77eee26cfc29`.
- E23 r2: job `lm-mpi-job-48c7efd6-517d-400d-9eac-d77cba398853`, Comet
  `9b6942c0ee6740c7aa4d3fe74effee93`.
- E24 r2: job `lm-mpi-job-6f9ec18e-2c47-4b1e-ad97-4a29f16a31b5`, Comet
  `5b64f84f134441b791e7c3ffbd6fe4f7`.

All six passed Hydra/spec composition and exact optimizer ownership. E20/E21
resolved the proposed branch-output contract at 2,380 tensors / 224,542,720
parameters; the other arms match E13 at 2,240 / 219,217,920. E22 additionally
passed exact Buffalo-L ONNX/PyTorch parity (`cosine=1.0`, max absolute error
`3.73e-6`) and nonzero input-gradient verification.

The `r1` jobs are failed startup records, not experiment results: their
isolated runtime linked a training-only dataset mirror and therefore lacked
the fixed-96 reference images. No optimizer step ran. The failure, two
immediately deleted duplicate E23/E24 submissions caused by MLS visibility
lag, implementation details, YAMLs, and complete job/key mapping are recorded
in
[`analysis/2026-08-06_e19_e24_implementation_and_launch.md`](../../analysis/2026-08-06_e19_e24_implementation_and_launch.md).

### BigCelebs E13 transfer launch — 8 August 2026

`BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1` transfers E13 r4's
shadow-coadapter contract to the sealed BigCelebs v2 training dataset, with
the fixed full-96 validation protocol unchanged. Its isolated Serv runtime is
`runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808`, built from commit
`ad194a026ab701dd979712d415c487dd536a4645`. The gated 05:00 BST attempt saw
six existing project A100s under the explicitly authorized eight-GPU ceiling
and was accepted as job `lm-mpi-job-7d361838-6faa-43be-a44c-ea6df1871233`.
The immutable Comet key is `c138db7c41ae435c8a7560f40cf5f58d`.

The required delayed startup check began after 35 minutes and stopped after
observing 98 completed training batches. A later live check found the job
Running with MLS `error_code=0`, no traceback or CUDA-OOM signature, durable
epoch-1 checkpoints at 2,000 steps, and progress beyond optimizer step 3,350.
The exact config, launcher, Serv package, dataset paths, and launch evidence
are recorded in
[`experiments/big_celebs/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1.json`](../../experiments/big_celebs/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1.json).

### E13 versus BigCelebs analysis and dataset-only follow-ups — 9 August 2026

`BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1` completed successfully
with MLS `error_code=0`. Its controlled comparison with E13 is documented in
[`analysis/2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md`](../../analysis/2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md),
with a visually verified 15-page PDF under
`output/pdf/2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.pdf`.
The report's metric freeze preceded BC_E13's deferred face-quality finalizer;
its ID/text conclusions and paired full-96 analysis are complete, while the
report deliberately does not invent unavailable face-quality values.

Observed result:

- E13 reaches `.399799 @24k`; BC_E13 peaks at `.399010 @16k` but falls to
  `.389430 @24k`. The paired final-panel mean delta is `-.010369`, median
  `-.01733`, 40/96 BC wins, with a 95% panel-bootstrap interval of
  `[-.02488, +.00441]`. This is a modest aggregate loss with large
  identity/prompt interactions, not a uniform failure.
- BC_E13 is persistently weak on Jisoo/Crying, Eddie/Night ride,
  Eddie/Crying, Jisoo/Rushing, Keanu/Dancing, and Lex/Jumping, while Jensen
  and Skiing improve. The report contains matched images and exact per-cell
  metrics for representative wins and losses.
- BigCelebs has 349,348 images / 68,648 IDs versus Large's 47,500 / 2,561,
  but the unchanged 48k-row budget reaches only about 31,480 BigCelebs IDs;
  roughly 21,382 are expected once and only about 10,097 at least twice.
  Median identity depth is 4 instead of 18.
- The domain also shifts sharply toward portraits: 83.97% portrait/close-up
  versus 0.324%, with much less standing, hands/holding, and multi-person
  content. Unrestricted image-proportional sampling and random same-ID
  references therefore spend the fixed budget on breadth and close-ups rather
  than repeated, scene-rich identity supervision.
- Horizontal flips leave directional captions unchanged. The measured
  directional-caption rate implies about 1,846 additional wrong-direction
  rows over 48k for BigCelebs relative to Large. The release itself is
  mechanically healthy; the auxiliary unused `caption_changes.jsonl` is
  defective because all 71,321 rows repeat one path.

Three opt-in dataset-only policies were implemented by the requested
GPT-5.6-Sol High agent. Historical Large and BigCelebs loaders remain
unchanged. The implementation entry points are
`src/datasets/bc_e13_schedule_policy.py`,
`src/datasets/big_celebs_e13_scheduled.py`,
`tools/datasets/build_bc_e13_dataset_schedule.py`, and
`launchers/active/run_BC_E13_dataset_experiments_24k_1gpu.sh`; exact configs,
specs, and Serv packages use the `BC_E13_ds1`, `BC_E13_ds2`, and
`BC_E13_ds3` prefixes. The sequential loader fails closed on source and
schedule hashes, preserves schedule order, uses distinct same-ID references,
and disables directional-caption flips. Full Hydra comparison permits only
`train_dataset_name` and `writer.experiment_comment` to differ from BC_E13;
all three retain E13's 2,240 trainable tensors / 219,217,920 parameters,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false`.

The isolated Serv runtime is
`runtime_worktrees/rsrch_test_BC_E13_dataset_20260809`, branch `test` at
`ad194a026ab701dd979712d415c487dd536a4645`. Before submission, each real
48k schedule was regenerated twice with byte-identical hashes, fully scanned,
and decoded at 64 distributed target/reference pairs. The accepted one-A100
jobs are:

- `BC_E13_ds1_repeatdepth_balanced_24k_full96_r1`: 2,561 strict deep IDs,
  18/19 visits each; schedule
  `b4fbceebe4dc76bbe5f5430cdf608f961266c495b092e245f8d3f836a6b88f73`;
  job `lm-mpi-job-7e1b163c-d16f-445c-896f-a313e12e2cf0`; Comet
  `b5b23b0ca4b449bc8f4703d6a7334be1`.
- `BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1`: the same identity
  order with 32k scene targets / 16k unrestricted targets and canonical
  top-three references; schedule
  `3300e2ea9ecfc23c60d9c94056df2c2ed0dad495211d892dcf6b33f39a25e9be`;
  job `lm-mpi-job-dc1dae0c-b373-47e1-a84a-96593da966f9`; Comet
  `5db54d7d4557487e94251656736843db`.
- `BC_E13_ds3_large_anchor_2to1_24k_full96_r1`: deterministic 32k Large /
  16k BigCelebs interleave, with 10,667 scene-role BigCelebs targets;
  schedule
  `b61b42a282c4db2b15d3ae25d6b47a1bcd6ab5e739051d81034214207f40c711`;
  job `lm-mpi-job-439c7f1b-d339-4077-a871-c52e4f264caa`; Comet
  `43adf33cf7174e89b8fde1cdd640a052`.

Immediately before the three submissions the project used four A100s. The
successive audits saw totals of four, five, and six; the new requests brought
the project to seven, within the user's scoped eight-GPU exception. All three
immutable `saved/<run>/comet_experiment.json` records were verified after the
complete source/schedule/decode gates, and all three jobs remained Running at
the final startup check.

### Problematic validation audit and corrected Eddie intervention — 9 August 2026

The cross-run audit covering E13, BC_E13, CL10 and CL11 is finalized in
[`analysis/2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.md`](../../analysis/2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.md).
It uses the controlled fixed-96 18k panels, historical endpoint panels, 41
checkpoint histories, detector/fixed-mask geometry, and a completed local
12-prompt corrected-Eddie final-checkpoint intervention. The sidecar is an
analysis-only protocol exception and must not be joined to historical Comet
curves as if it were a normal validation event.

Two independent validation defects are proven:

- Eddie's reference contains foreground Eddie plus a small background face.
  Historical `faces[0]` selection uses the bystander for the validation target
  and PhotoMaker conditioning while the spatial BA crop already uses Eddie.
  The stored embedding has cosine `-0.0078` to the intended foreground face.
- `IDSimBest` can reward identity on another body because it maximizes over all
  generated detections without mask ownership. All four 18k Chef/Lex images
  show this failure; the winning identity face has fixed-mask IoU `0.000`.

The corrected Eddie chain completed successfully for E13 24k, BC_E13 24k and
the requested CL11 20k checkpoint. Replacing only the historical bystander
embedding with foreground Eddie, while preserving reference pixels/bbox,
prompts, seeds, fixed generation masks, scheduler, inference steps, CFG and
checkpoint, raises intended-Eddie ID similarity:

- E13 24k: `0.0653 -> 0.3407`, paired `+0.2754`, 12/12 wins;
- BC_E13 24k: `0.0626 -> 0.2842`, paired `+0.2216`, 11/12 wins;
- CL11 20k: `0.0741 -> 0.2992`, paired `+0.2251`, 11/12 wins.

The correction is necessary but not sufficient. Corrected median mask IoU is
only `0.799-0.835`; E13/BC_E13 Jumping move the face off the fixed mask (IoU
`0.000`), and all three Kickboxing outputs fuse or duplicate face/body anatomy.
E13 Kickboxing still scores `0.449` with mask IoU `0.122`, demonstrating that
ArcFace plus binary detection can reward a visibly invalid face. CL11 preserves
Jumping placement best (`0.744` IoU) but has lower Eddie identity. Skiing stays
low (`0.102-0.142`) with severe goggle/eye artifacts despite reasonable mask
alignment.

Sidecar outputs are under
`analysis/assets/problematic_validation_20260809/final_checkpoint_sidecar/`;
paired tables and figures are generated by
`analysis/assets/problematic_validation_20260809/analyze_final_corrected_sidecar.py`.
Checkpoint SHA-256 values are E13
`4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab`,
BC_E13 `99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e`,
and CL11 `e65972c8c14b5031f879e1ee8b1e11a707823e0cfccdb80553219fc8069dbb83`;
the preserved generation-bbox protocol hash is
`4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099`.

E13 remains the base. Priority order is now: (P0) a versioned subject selector
and mask-owned ID metric with legacy metrics preserved; (P1) an E13
PhotoMaker identity-onset/cross-attention isolation sweep to prevent global
composition drift; (P2) a separately normalized native/reference face-message
mixture for occlusion and anatomy. Target-scale matching is next after those.
Keep `pose_adapt_ratio=0` and `ca_mixing_for_face=false` throughout.

The final report PDF is
`analysis/assets/2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.pdf`
(`32,615,925` bytes; SHA-256
`fee2995303a6f283353f9f5f92f717b337ac43abf102d1b1cdda8e98e3f0ccbd`).
All 19 pages were rendered and visually checked; the Dropbox upload returned
content-integrity OK.

A separate image-by-image before/after report is
[`analysis/2026-08-09_eddie_validation_pre_vs_post_reference_fix.md`](../../analysis/2026-08-09_eddie_validation_pre_vs_post_reference_fix.md),
with PDF
`analysis/assets/2026-08-09_eddie_validation_pre_vs_post_reference_fix.pdf`
(`33,108,650` bytes; SHA-256
`e238c81350c72a44984b71e5ed3f984d2b9e3640fa4903a4b8dec43f27f6e401`).
It shows all 36 historical/corrected pairs with fixed-mask and detected-face
overlays. The key paired summary is 34/36 identity wins versus only 4/36 mask
IoU wins, with five new post-fix mask IoUs below `0.30`. All 12 PDF pages were
rendered and visually checked; Dropbox integrity verification passed.

### Correction: Eddie sidecar validation contract was not reproduced — 9 August 2026

The generated corrected-Eddie evidence in the immediately preceding section is
**withdrawn** after a user-raised image audit and code/config trace. Do not use
the `34/36` win count, corrected alignment deltas, counterfactual full-96 means,
or E13/BC_E13/CL11 corrected-image rankings. The historical Comet images,
wrong-reference-selector diagnosis, mask-ownership diagnosis, and non-Eddie
measurements remain valid.

The old corrected sidecar did load the intended RealVis base and preserved the
recorded checkpoints, prompts, seed 0, references, reference boxes, cached
generation boxes, DDIM scheduler, 50 steps and CFG 5. It nevertheless differed
from all three in-training validation contracts in material ways:

- it forced `processor_base_mode=validation_native`; E13, BC_E13 and CL11 all
  resolve to strict `legacy_full_copy` (70 stateful BA processors);
- `evaluate_rhca_checkpoint.py` did not implement
  `validation_shadow_photomaker_default=true`, so 700 trained checkpoint
  `.default.` tensors remained active instead of restoring the pretrained
  PhotoMaker default on RealVis; and
- it forced batch size 1 and rewrote the validation dataset limit to 12, while
  training generated the 12 Eddie rows as one batch inside configured full-96.

The old sidecar also changed the ArcFace vector fused into global PhotoMaker
prompt tokens at denoising step 10; spatial BA begins at step 15. It was not a
face-local BA-mask change, and the foreground Eddie BA bbox was already
correct. Thus an exact future replay may still show global composition response
to the selector correction, but only after an unchanged baseline reproduces the
historical images.

The evaluator now derives and enforces base, processor, shadow, CFG, CA and
batch semantics from the experiment config; mirrors strict processor copying
and the pretrained-default snapshot/restore; propagates the trainer's full BA
runtime attribute set; retains the configured full-96 context; and records the
contract in its manifest. The local contract-v2 chain first generates an
unchanged historical replay, requires exact pixels for all 12 Eddie images, and
only then permits the corrected global-ID arm. The old Serv launcher is blocked
and the old analysis script rejects non-v2 manifests. No corrected validation
was launched during this audit. See
[`analysis/2026-08-09_eddie_revalidation_contract_audit.md`](../../analysis/2026-08-09_eddie_revalidation_contract_audit.md).

### Eddie contract-v2 Serv replay launched — 9 August 2026

The guarded one-GPU chain was submitted on Serv at
`2026-08-09T19:37:53+01:00` as
`lm-mpi-job-baea4903-7f8d-4785-a67d-f153df3299da`. Six project A100s were
Running and none Pending immediately before submission; this job is the seventh
GPU under the user's scoped ten-GPU exception for the corrected Eddie chained
revalidation. Serv accepted it and the job completed successfully in 2,111
seconds at `2026-08-09T20:13:25+01:00`.

The chain uses the immutable E13, BC_E13, and CL11 runtime snapshots and sealed
checkpoint hashes, with the patched evaluator supplied as an external overlay;
the experiment runtime trees are not edited. For each model it runs the
unchanged 12-image historical batch first, requires exact RGB pixels against
the original Comet images, and only then permits the foreground-Eddie global
PhotoMaker identity-vector arm. Any failed replay gate stops the entire chain.
Outputs and stage markers are under
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_sidecars/eddie_revalidation_contract_v2_serv_20260809_r1`.
The audit record is
`experiments/diagnostics/eddie_revalidation_contract_v2_serv_20260809_r1.json`.

All three gates passed: E13, BC_E13 and CL11 each reproduced 12/12 historical
Eddie images RGB pixel-exact, with no failed pairs or contract mismatches. The
chain then produced all 36 corrected images. The local
`analysis/assets/problematic_validation_20260809/final_checkpoint_sidecar_contract_v2/`
copy verified 105/105 output files against Serv's SHA-256 manifest.

The valid analysis reverses the invalid sidecar's composition conclusion:
intended-Eddie identity improves in 36/36 prompt pairs (`+0.360/+0.291/+0.289`
mean for E13/BC_E13/CL11), corrected median mask IoU remains
`0.891/0.875/0.880`, and no corrected image falls below `0.30`. Kickboxing and
Jumping retain body layout. P1 is therefore the E13 dual native/reference
face-message diagnostic for residual anatomy/occlusion; P2 is target-scale
matching for small faces. The PhotoMaker-onset sweep is demoted.

The rebuilt main PDF is
`analysis/assets/2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.pdf`
(`32,237,957` bytes; SHA-256
`e5a5638eb367e202ccfe85082714975f8b080dd21622a401739ce32b76b20981`).
The rebuilt pre/post PDF is
`analysis/assets/2026-08-09_eddie_validation_pre_vs_post_reference_fix.pdf`
(`32,738,165` bytes; SHA-256
`e81a81977c12d3b23418324ce6d813b3aad62e6a3fc97e5ba2161dded5fdb7c8`).
All 19 and 12 pages respectively were rendered and visually checked; both
Dropbox uploads returned content-integrity OK.

### Subject-v2 training-validation repair and selective Comet backfill — 9 August 2026

The Eddie defect is now fixed in the shared E13-family **validation** path.
`bbox_overlap_v2` selects the InsightFace detection with maximum overlap to the
declared reference box and fails closed on missing/non-overlapping/ambiguous
detections. The training model retains `legacy_first`, so this repair does not
silently change the E13 optimization trajectory. Historical configs also keep
the legacy default for exact replay. E13, BC_E13 and CL11 compose to validation
policy v2, batch 12, full 96, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`.

The versioned reference preflight passes 12/12 identities. Eddie is the only
multi-face reference and selects detection index 1 with IoU `0.896066`; all 11
other references select index 0 and remain numerically equal to their legacy
vectors. The new embedding artifact is
`dataset_full/val_dataset/id_embeds_manual_val_subject_v2.pth`, SHA-256
`e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465`;
its provenance manifest binds both new and legacy embedding hashes.

The generated-face metric now receives the exact bbox resolved by the trainer
and passed into BA. This corrected a second, analysis-side inconsistency. The
earlier Chef/Lex claim used the dataset manual bbox `[590,413,694,543]`, while
in-training BA used cached auto bbox `[223,380,447,668]`. On the E13 18k image,
the actual-box mask-owned result equals historical best ID `0.365892` with IoU
`0.898726`; the unused manual box selects the background chef at ID `0.139086`.
Chef/Lex is therefore not a proven live BA ownership failure. It remains a
useful negative check showing why metrics must consume the resolved box.
`IDSimMaskMatched` is the corrected primary score; historical max-over-any-face
ID is retained as `manual_val/id_sim_legacy_best`, with mask IoU, face count,
no-face, unowned and ambiguity diagnostics in curves and per-image CSVs. The
main report Markdown contains a prominent erratum; its previously uploaded PDF
predates that erratum.

`tools/comet/backfill_subject_v2_validation.py` is prepared for retrospective
repair of CL0-CL10, E13, BC_E13 and related runs. It supports one checkpoint or
all deserializable checkpoints with complete fixed-96 Comet panels, discovers
changed identities from the hash-bound selector manifest, expands generation
to original batch boundaries, and publishes only requested identity/prompt
rows. Before correction it replays the historical batch and requires exact RGB
pixels. It merges corrected rows into the other historical images, rescoring
all 96 with batched CLIP, mask-owned and legacy ID, and the canonical seven
face-quality metrics. Dry-run is the default; `--write` backs up exact assets
and complete metric histories, waits for Comet deletions, restores/replaces the
series and tables, then downloads every replacement and verifies SHA-256.
`--generation-bbox-map` pins run-specific cached maps such as CL11's distinct
historical protocol. Local `BACKFILL_ETA` lines report current-checkpoint and
whole-job remaining time, while evaluator batch lines retain inference ETA.

No historical Comet experiment was mutated while implementing this repair.
Run the script from each experiment's exact immutable source/config tree; using
a current Hydra tree can mis-derive historical epoch/step, RealVis,
processor-copy, shadow-adapter, batch, scheduler, or bbox semantics.

The requested retrospective E13, BC_E13, BC_E13_ds1/ds2/ds3 and CL4-CL9 job
was submitted on Serv as one five-worker binary allocation. The first
production job was `lm-mpi-job-af9c6f65-e450-4edb-bd62-1d24379228c7`; it entered Running at
`2026-08-09T22:12:48Z` with five one-A100 nodes. Three project A100 jobs were
active before submission, so the total request is eight under the user's
scoped ten-GPU exception. Worker 0 owns E13 -> CL4 -> CL6; workers 1-4 own
BC_E13 -> CL5, ds1 -> CL7, ds3 -> CL8, and CL9 -> ds2 respectively. Every
initial run discovered all 12 safe checkpoints, 12 affected Eddie rows, batch
12, and began the step-2k exact historical replay gate. First rolling estimates
put one 12-checkpoint run near 2.3-2.9 hours; the three-run tail gives an early
wall estimate of roughly 7-9 hours, subject to full-96 IQA and Comet write time.

The deployed package is
`serv_run_packages/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/`.
The launcher uses an updated backfill tool SHA-256
`51b8ef9839a6ed77210a163e35fed77f57b7c95feb3958f255faab41a9bd57d1`.
A read-only live audit passed exactly 96 normalized canonical image assets at
all 12 saved steps in all 11 immutable Comet experiments. Three earlier
startup/preflight attempts produced no staging files and no Comet writes: the
first exposed the machine-local legacy-embedding location, the second was
stopped after internal worker shells lacked a Conda bootstrap, and the third
exposed Comet's removed `.png` suffix plus nondeterministic ` (N)` counters.
The production package now resolves legacy embeddings from each immutable
runtime, directly enters the pinned environment on worker nodes, and uses the
existing export/report filename normalization for discovery, exact deletion,
and post-upload verification. The audit record is
`experiments/diagnostics/subject_v2_historical_backfill_11runs_5gpu_20260809_r1.json`.

The first production allocation was then stopped after its fail-closed replay
gate found 12/12 pixel mismatches for both E13 and BC_E13 at step 2k. It made
zero corrected images, zero job manifests, and zero Comet writes. The cause
was a launcher-only contract error: it passed the sealed canonical manual bbox
seed (`a39645e2...`) as `--generation-bbox-map`, while training used each
immutable runtime's derived `pm96_bboxes_new_auto.json`. The corrected launcher
separates the manual seed from the active cache and seals E13/BC-family maps at
`4db6344d...` and CL4-CL9 maps at `b33cf026...`. All 12 Eddie boxes differ
between the manual seed and active cache, explaining the complete replay
rejection.

After re-auditing Serv at three active project A100s and zero Pending, the
corrected five-worker package was resubmitted at `2026-08-09T22:31:01Z` as
`lm-mpi-job-44b99a20-a6ad-4023-b3c6-f249b1abe83d`. Its five GPUs bring the
project total to eight under the user's scoped ten-GPU exception. The failed
partial staging is preserved separately at
`staging_failed_manual_seed_af9c6f65`; the corrected job starts from an empty
staging directory.

All five initial step-2k historical replay gates subsequently passed exact RGB
and advanced into corrected-Eddie generation: E13, BC_E13, BC_E13_ds1,
BC_E13_ds3, and CL9. This directly validates both sealed active-cache families
before any Comet write. The job remains in dry staging; Comet mutation is still
blocked until every checkpoint of a whole run has staged successfully.

### E14-E22 subject-v2 backfill schedule — 10 August 2026

At `2026-08-09T23:41Z`, the active five-GPU backfill remained healthy with 15
fully staged checkpoints, every worker processing step 8k, no traceback, and
current-run remaining estimates of 2.0-2.4 hours. Worker 0 still owns the
three-run E13→CL4→CL6 bottleneck; measured throughput implies whole-job
completion around 09:00-10:00 Europe/London, with normal uncertainty from
full-run Comet replacement.

The selected completed revisions for the requested extension are E14 r6, E15
r2, E16 r2, E17 r5, E18 r4, and E19-E22 r2. Each owns exactly 12 weights
checkpoints and an immutable Comet record. E14-E18 seal active generation
cache `4db6344d...`; E19-E22 seal `b33cf026...`.

An immediate two-GPU E14/E15 request was attempted only after counting eight
project Running+Pending GPUs, but MLS rejected it before job creation with
`PROJECT_GPU_LIMIT_REACHED_ONLY_1_FREE`. Per the allocation-rejection rule it
was not retried. All nine runs are now assigned to one delayed five-worker
wave: E14→E19, E15→E20, E16→E21, E17→E22, and E18. Serv-side scheduler PID
`832912` requires current job
`lm-mpi-job-44b99a20-a6ad-4023-b3c6-f249b1abe83d` to succeed, waits 20 minutes,
then recounts project Running+Pending GPUs and makes exactly one submission
attempt only if adding five remains within the authorized ceiling of ten.
Companion PID `832935` monitors the submitted job through Running and then
exits. State is under
`analysis_jobs/subject_v2_historical_backfill_e14_e22_20260810_r1/scheduler`;
the local audit is
`experiments/diagnostics/subject_v2_historical_backfill_e14_e22_20260810_r1.json`.

At the user's subsequent request, completed CL10 r2 and CL11-CL14 r1 were
inserted at the head of that still-unsubmitted delayed wave. All five were
live-audited on Serv as Completed with exactly 12 weights checkpoints, immutable
Comet records, the legacy embedding hash `23ae9707...`, and active bbox hash
`b33cf026...`. The priority chains are now CL10→E14→E19,
CL11→E15→E20, CL12→E16→E21, CL13→E17→E22, and CL14→E18. This
puts one CL run first on every A100; E14-E22 start only after the corresponding
CL run is transactionally published and hash-verified. The scheduler and
20-minute dependency gate are unchanged.

The original five-GPU job later failed at `2026-08-10T09:12:56Z` after eight
runs had been transactionally published and verified. CL6 encountered a
120-second Comet asset-list timeout after staging through 20k; BC_E13_ds3
encountered an asset-download HTTP 502 after staging through 16k; CL8 had not
started. No unfinished run had a job manifest or any Comet mutation. The
success-gated E14-E22 scheduler therefore exited without submitting, as
intended.

Recovery job `lm-mpi-job-7df3819d-7fdc-4ca0-a50a-b058bb254f03` was submitted
under the user's scoped ten-GPU exception and entered Running with eight A100
workers on 10 August. Workers 0-2 own CL6 resume, BC_E13_ds3 resume, and CL8;
workers 3-7 own CL10→E14→E19, CL11→E15→E20, CL12→E16→E21,
CL13→E17→E22, and CL14→E18. The tool now retries idempotent Comet reads and
downloads up to eight times with exponential backoff, downloads atomically,
hash-validates and reuses completed step manifests, and preserves incomplete
steps under `incomplete_recovery/` before regenerating them. Initial live
checks found all eight distinct claims Running with zero tracebacks: CL6 had
resumed at 22k, BC_E13_ds3 at 18k, and CL8/CL10-CL14 at 2k. Multiple fresh
Comet 502s were observed and recovered by the new retry path.

Serv-side monitor PID `839332` polls that recovery job every 60 seconds, logs
job/worker status plus staged and verified counts, and alerts after 45 minutes
without worker-log progress. Its first observation at `2026-08-10T10:39:33Z`
was Running, `8/25` total runs verified, 114 step manifests, eight workers
Running, zero workers failed, and 21 seconds since log progress. State is under
`analysis_jobs/subject_v2_historical_backfill_recovery_priority_8gpu_20260810_r1/monitor`.

CL6 later failed while scoring its incomplete 22k step because OpenAI CLIP
downloaded `ViT-L-14-336px.pt` into the worker-local `~/.cache/clip` and the
result failed its checksum. The traceback itself confirms execution under the
pinned Nasilaev `photomaker_NS` environment; CLIP's cache is independent of
Conda and `TORCH_HOME`. The metric file is now pinned under
`metric_cache/clip/ViT-L-14-336px.pt` (934,088,680 bytes, SHA-256
`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`),
passed explicitly through `CLIP_CACHE_DIR`, and the worker fails startup unless
both `python` and `CONDA_PREFIX` resolve exactly to
`nasilaev/conda_env/photomaker_NS`. A one-GPU CL6 recovery request was rejected
before job creation with `PROJECT_GPU_LIMIT_REACHED_ONLY_0_FREE` and was not
retried.

At the user's request to avoid idle allocated GPUs, the fixed-chain recovery
job was stopped after 12 runs were verified and 177 complete checkpoint
manifests were preserved. A replacement uses an NFS-claimed dynamic queue so
each worker takes the next unfinished CL6/CL12-CL14/E14-E22 run immediately.
The subsequent eight-GPU submission was rejected before job creation with
`PROJECT_GPU_LIMIT_REACHED_ONLY_4_FREE`. Per the no-retry-after-limit rule it
was not resubmitted in the same turn. A validated four-GPU fallback is deployed
as `analysis_jobs/subject_v2_dynamic_remaining_8gpu_20260810_r1/package/run_dynamic_4workers.yaml`
and awaits an explicit post-rejection user request. At this handoff no project
GPU job is Running or Pending; Comet and staging state remain intact.

The user then explicitly requested independent one-GPU submissions. The worker
now accepts a shared dynamic NFS claim root across distinct MLS job IDs; this
prevents separate jobs from all selecting CL6 while retaining job-isolated logs
and status. Four one-A100 jobs were submitted individually—the full live Serv
capacity reported after another user's four-GPU allocation—and all four entered
Running with distinct claims and zero startup tracebacks:

- `lm-mpi-job-a528e879-358d-486d-9f17-8dc655309eb7`: CL6, resumed at 22k;
- `lm-mpi-job-55fcaba9-351d-4c95-a901-725fa25180f3`: CL12;
- `lm-mpi-job-c532a193-92fb-41c1-8978-7051a5d87cf9`: CL13;
- `lm-mpi-job-d96b3825-0065-4c63-a0a4-2b11483e6cb1`: CL14.

Each worker claims another E14-E22 run immediately after verifying its current
run, so CL6 failure cannot block the other queue. Four additional one-GPU YAMLs
are deployed but not submitted while no further Serv GPU is free. The package
is `analysis_jobs/subject_v2_dynamic_remaining_1gpu_20260810_r2/package`; the
local audit is
`experiments/diagnostics/subject_v2_dynamic_remaining_1gpu_20260810_r2.json`.
Durable Serv monitor PID `850568` polls every 30 seconds and submits workers
05-08 individually only if measured global capacity and unclaimed work permit;
it exits without retry after any allocation rejection. Its first two snapshots
showed all eight Serv GPUs allocated (four to this queue, four to another user),
four distinct project jobs Running, four claims, and nine runs still unclaimed.

On 11 August, after 19 of 25 runs were transactionally verified and only E21
and E22 remained unclaimed, the user requested use of newly available GPUs.
Two additional one-A100 jobs were submitted separately and both entered
Running with distinct claims:

- `lm-mpi-job-212a5d08-b329-4c0d-9157-b5ab6fc62fc7`: E21;
- `lm-mpi-job-a500307e-4e4c-4bcb-8ae5-8f23e61f07fe`: E22.

The queue now uses six project A100s, all 13 dynamic runs are claimed, and no
additional worker can accelerate the remaining work. The earlier four workers
own E17-E20. Startup showed the pinned `photomaker_NS` environment and no
traceback, OOM, checksum, or Comet error.

The historical subject-v2 replacement wave completed on 11 August. All six
independent MLS jobs reached Completed with `error_code=0`; the monitor exited
normally after observing zero project jobs and all 13 dynamic runs claimed.
The final Serv audit found exactly 25 `replacement_verified.json` records,
zero invalid audits, the complete 2k-through-24k saved-checkpoint sequence for
every run, and `job_manifest.status=verified_on_comet` throughout. Thus the
corrected images and recalculated per-image/aggregate metrics are now stored in
the original immutable Comet IDs for E13, BC_E13 and ds1-ds3, CL4-CL14, and
E14-E22. No project GPU remains allocated to this wave.

### CL14 hard-case architecture review — 11 August 2026

The detailed architecture/data/literature review requested after CL14 became
the best corrected run is complete in
`analysis/2026-08-11_cl14_hard_cases_architecture_research_and_experiment_plan.md`.
No training job was launched. Six implementation-gated, non-runnable YAML
blueprints are under `experiments/designs/cl14_next_20260811/`; they must not be
copied into `src/configs/` or submitted until their listed parity, checkpoint,
causality, memory, and validation preflights pass.

Immutable comparison keys are CL14
`6fe0028be92242c38056b3d36665fdd6` and CL9
`81bb311ed70545eda3281c64bc48be47`. Corrected step-24k subject-v2 identity is
`0.456116` versus `0.447997` (`+0.008120`, `63/96` paired wins). Marion improves
by `+0.038795` but remains lowest at `0.349982`; Jumping/Dancing remain the
small-face floor, while Skiing/Crying regress against CL9. CL14 has 96/96 face
detection, median detected/requested face-size ratio `1.0145`, and no IoU below
`0.3`, so its small-face issue is absolute local resolution rather than box
underfill.

The review found a material CL14 interpretation discrepancy. CL14 writes a
two-cell target training feather (`1/3`, `2/3`), but the installed BA processors
default `force_binary_masks=true` and threshold at `>0.5`. The intended soft
ramp is therefore a hard one-cell erosion. Do not “fix” this by setting the flag
false: current processors mask Q and output, so that would not implement a
single continuous native/reference blend. Blueprint P5 specifies separate
target routing and reference key masks with full lanes and one convex blend.
Standalone true-reference-key masking remains excluded because E1 failed and
removing zero sinks changes reference-softmax amplitude.

Recommended independent experiment order is:

1. P1 / CL15 shared high-resolution ROI BA—the training analogue of the
   replicated CL9 18-step ROI gain (`+0.09684`, `43/56` non-Eddie wins).
2. P2 / CL16 bounded clean multi-scale reference memory alongside CL14.
3. P3 / CL17 semantic native/reference ownership for glasses, hair, hands,
   tears, and goggles, with a clean-skin reference floor and causality telemetry.
4. P4 / CL18 cross-view consistency in the spatial BA lane while target Q,
   target noise, and PhotoMaker tokens remain fixed.
5. P5 / CL19 the correct continuous full-query router described above.
6. P6 / CL20 a 20% curated deep-ID/hard-case BigCelebs curriculum for 20k plus
   a 4k Cosmic-only re-anchor; broad BigCelebs replacement is rejected because
   all corrected 24k BigCelebs/ds arms remain below CL14 overall.

Nineteen primary papers, extracted text, eight immutable official-README
snapshots, and a SHA-256 manifest are archived under
`analysis/sources/2026-08-11_cl14_architecture_review/`. The highest-value
connections are RealisID-style local crop/upscale/scatter implemented with
shared BA, DreamCache-style clean feature memory, DynamicID/PersonaHOI-style
query ownership, and same-ID cross-view regularization. PuLID-style accurate-x0
ID loss is deliberately reserved for later because naive ID-loss arms already
failed and editability contamination remains a high risk.

### CL15-CL20 implementation and Serv startup — 11 August 2026

The six CL14 hard-case blueprints were implemented as defaults-off additions on
the CL14 PhotoMaker + branched-attention base. The shared config is
`src/configs/CL15_CL20_hardcase_base_24k.yaml`; the runnable arm configs are
`src/configs/CL15_cosmic_shared_highres_roi_ba_24k.yaml` through
`src/configs/CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml`. Submit with
`launchers/active/run_CL15_CL20_hardcases_24k_1gpu.sh`. All arms preserve
target Q/reference K/V BA, `pipeline.pose_adapt_ratio=0`,
`pipeline.ca_mixing_for_face=false`, 24k optimizer steps, batch 2,
`epoch_len=2000`, and the sealed CL14 step-0/every-2k manual-val-96 DDIM50
contract. `tools/validate_CL15_CL20_config.py` fail-closes these invariants.

The defaults-off implementation adds: CL15 fixed 32x32 high-resolution ROI BA;
CL16 detached clean t=1/null-text multiscale reference memory; CL17 semantic
native/reference ownership plus deterministic occluder supervision; CL18
alternate same-ID cross-view consistency; CL19 one-time cosine full-query
blend; and CL20 an exact sealed 48k-sample Cosmic/BigCelebs schedule (40k
Cosmic, 8k BigCelebs, with BigCelebs strata 2667 small / 2667 occlusion / 2666
action). CL20 schedule SHA-256 is
`783eb1729871e4ac423c770042315572ee7ea24171797402fc4a565999dd5289`.

The first r1 submissions are non-scientific startup failures and must not be
used as experiment results: CL15-CL19 used a packaging symlink to a
protocol-only dataset mirror, and CL20 lacked the subject-v2 embedding mount.
Their immutable Comet records are retained with
`status=failed_startup_packaging`. The corrected r2 source snapshot revision is
`ad194a026ab701dd979712d415c487dd536a4645+cl15-cl20-r2-mountfix-valseal-metricseal-20260811`.
It pins the CL14 validation assets and subject-v2 embedding SHA-256 and mounts
the full dataset root.

All corrected r2 jobs were submitted one by one after a capacity check, reached
the fixed step-0 96-image validation, completed it, and emitted a training-loop
marker with zero monitored startup failures:

- CL15: job `lm-mpi-job-9cbf24b7-543a-47ad-b790-c30a36e09303`, Comet
  `d57604dc77334e0f9874ddd049e85a17`;
- CL16: job `lm-mpi-job-7c6a1db0-4d26-4339-8871-e99b2366f395`, Comet
  `7b71eb3dbb3a492e8fa9bb0d87343c28`;
- CL17: job `lm-mpi-job-15b25e73-47da-4ab1-a71f-efec8f50ed0c`, Comet
  `e05ce586c9364678a8370a774773341c`;
- CL18: job `lm-mpi-job-1c4dd150-9688-4ca0-b678-8f74134a70e7`, Comet
  `f6530436bf22472c9fb7731d1696c5ab`;
- CL19: job `lm-mpi-job-f1b9d006-208c-4b35-8e4a-ab0ab2f030a9`, Comet
  `cfeda7b55c174b3c83e8d40537ebb6dd`;
- CL20: job `lm-mpi-job-1e0f08fd-b0d3-4b26-9167-5d55103f442d`, Comet
  `b05488e2cce94476acc92bcaa21d7362`.

Monitoring stopped at the requested terminal condition at
2026-08-11T15:59:05+03:00: Comet 6/6, validation 6/6, training 6/6, failed 0.
Use the immutable local records under `experiments/cosmic_large/` for all
subsequent retrieval and comparison.

### CL14_CA implementation and validation startup — 12 August 2026

`CL14_CA` now runs the corrected residual identity cross-attention v3 on the
CL14 control: target Q attends active PhotoMaker ID-token K/V in
`up_blocks.0/1`, rank 64, zero-initialized output, gate 0.02 bounded by 0.20.
Native CA remains intact; legacy branched CA remains disabled;
`pose_adapt_ratio=0` and `ca_mixing_for_face=false`. The exact ownership gate
passed at 2,348 tensors / 224,624,676 parameters for both trainables and
optimizer membership.

The first architecture-capable run, job
`lm-mpi-job-cf7eda84-ad0b-4d50-af17-c3d9f19e5315`, Comet
`0cfe3c874d75448789acc0a5c9b4bc63`, completed fixed-96 step-zero validation
and then failed on its first training batch. The model returned new route
telemetry nested under `ba_telemetry`, but CL14's unchanged masked loss did not
flatten it; `SDXLTrainer` then indexed the absent top-level key
`ba/identity_ca_token_count/up0`. The minimal fix in
`src/trainer/sdxl_trainers.py` promotes `output["ba_telemetry"]` immediately
after `batch.update(output)`. It changes logging only and leaves CL14's loss
and scientific objective unchanged.

Subsequent startup-only failures came from package drift, especially custom
InsightFace roots and incomplete default worker caches; they are not
architecture evidence. The final wrapper follows CL14: analyzer constructors
remain unchanged, the sealed five-file `buffalo_l` cache is copied into the
default `~/.insightface/models/buffalo_l` location, its hash is checked, and
the normal CL14 validation/offload/reinstall lifecycle is retained. A
deliberately corrupted partial cache was repaired and both detection and
recognition initialized with network access disabled.

The active scientific run is `CL14_CA_r7`: Serv job
`lm-mpi-job-244ef7b2-3943-4998-a82e-ae1be2208169`, immutable Comet key
`4d96dc8e776b4039b1116acc5cdcf706`, runtime
`runtime_sources_cl14_ca_v8/CL14_CA_r7`, manifest SHA-256
`00cdb22e122cc94f64496808a3146cf452cae60a71f6ac033e9f3d7cfd13339c`.
It completed all 96 step-zero images, wrote the 96-row ID table, staged all 96
face-quality inputs, restored the CL14 training base, and advanced beyond
optimizer step 224 with finite logged losses. Immutable-key Comet
telemetry at step 0 records two active identity tokens in both groups, gate
0.02, finite native-face RMS, and the expected zero-initialized residual. It
must remain running unless a later error is observed.

The user-requested operational one-batch smoke uses 12 fixed-panel items at
CL14's inherited batch size 12 and sets the matching face-quality expected
count to 12. It is not scientifically comparable. Smoke r1 generated all 12
images but stopped at the inherited 96-image assertion. Corrected smoke r2 is
Serv job `lm-mpi-job-05738b32-5978-4559-96cb-6ac7ea38cd2d`, sealed runtime
`runtime_sources_cl14_ca_v11/CL14_CA_onebatch_smoke_r2`, manifest SHA-256
`ac2d5574ac22a13c3686c1dfa65a13e7ea3ab87c21d2871d9acfca45a76cb09d`.
Its immutable Comet key is `f808676f2ad54e5e928d92b6650053ca`.
It completed the 12-image validation, wrote its ID table and face-quality
staging manifest, restored the training base, and advanced through optimizer
step 7 with finite loss `0.064705`. Immutable-key telemetry confirmed two
identity tokens per group, gate 0.02, finite native-face RMS, and the expected
zero-initialized residual. The requested MLS stop succeeded; final status is
Stopped. The scientific `CL14_CA_r7` job remained Running.

The detailed failure analysis, exact code, reproduction gates, and live run
ledger are in
`analysis/2026-08-12_CL14_CA_startup_failure_fix_and_relaunch.md`.

### CL14_CA throughput optimization and corrected-Eddie relaunch — 13 August 2026

The original scientific `CL14_CA_r7` remains active and untouched: Serv job
`lm-mpi-job-244ef7b2-3943-4998-a82e-ae1be2208169`, immutable Comet key
`4d96dc8e776b4039b1116acc5cdcf706`. Its recent 100-sample training-rate median
was `3.480 s/it`; its matched first-epoch steps 21--120 median was
`3.590 s/it`. Historical CL14 key `6fe0028be92242c38056b3d36665fdd6`
has median `2.190 s/it` over the same 100-step window across all 12 epochs
(`1,200` samples).

Safe execution-only optimizations were implemented with the user-requested
`12 Aug 2026 - Training optimization` comments: all 19 telemetry scalars are
stacked for one synchronization with a one-GPU collective bypass; fixed
identity-token indices are built once per U-Net call and shared by the CA
processors; and independent native target/reference CA rows are fused on the
batch axis before being split. Q/K/V ownership, residual-CA coverage and gates,
loss, loader, optimizer, and exact `2,348 / 224,624,676` contract remain
unchanged. Pinned/persistent workers, nonblocking transfers, and speculative
manual device re-homing were removed.

Speed smoke `CL14_CA_optimized_speed_smoke_r12`, Serv job
`lm-mpi-job-b3db33aa-5b8e-45fd-a48e-e7fe7d7ab9af`, immutable Comet key
`75ab71fc2d4c44a5b4b625ebf20b89ed`, completed one unchanged 12-image startup
validation batch and finite optimizer steps. Steps 21--120 have median displayed
rate `3.230 s/it`; step-20 to step-120 elapsed time independently gives
`3.23 s/it`. This is `7.2%` lower latency than r7's latest median and `10.0%`
lower than its matched first-epoch window, but still `47.5%` slower than CL14
because the residual identity CA is real additional compute. The smoke was
stopped deliberately after observing step 126.

Do not use a completely skipped step-zero validation as the normal startup
smoke. Multiple cold attempts, including an exact healthy-r7-derived source
(job `lm-mpi-job-60b5e876-20d8-44f9-b7f3-4baa7a1c1c20`, Comet
`cc096b408f184faf9f1618afc4eb9588`), segfaulted on the first native autograd
backward. A single standard 12-image validation batch safely initializes the
established CL14 validation/offload/reinstall lifecycle. The exact lower-level
CUDA cause is not established.

The new scientific production run is `CL14_CA_optimized_r11`: Serv job
`lm-mpi-job-26dc8f54-1b96-4129-9151-a4fb066a7ff7`, immutable Comet key
`fafd7a61b06c4114b9dec2c21d29ca38`. It uses one A100 and runtime
`runtime_sources_cl14_ca_v23`, revision
`live-r7-v8+a65ffcb2c95f+ca-optimization+cl20-validation`, manifest SHA-256
`8d03af06bcc32b306ce3cbf83d56180573cac709467b02512b3cf817ef7999d5`.
It completed the canonical step-zero `96/96` panel in `25:18`, wrote the 96-row
ID table, staged 96 face-quality inputs, restored the training base, logged
finite loss `0.064709`, and advanced through at least optimizer step 10 at
`3.23 s/it`. Monitoring then stopped as requested; the job remains Running.

This production run also ports CL20's validation-only Eddie repair:
`bbox_overlap_v2` selection in both dataset and pipeline, subject-v2 metrics,
and sealed embedding SHA-256
`e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465`.
The selector audit chooses Eddie detector index 1 with IoU `0.896066`; all other
identities use index 0. The production step-zero Eddie rows have mean subject-v2
similarity `0.243214`, versus `0.097842` under r7's historical wrong-face
contract. This is a contract sanity check, not a model-quality comparison,
because the correction intentionally changes both generation conditioning and
the scoring target. `pose_adapt_ratio=0` and `ca_mixing_for_face=false` remain
fixed. Full details are in
`analysis/2026-08-13_CL14_CA_training_throughput_optimization.md`.

### CL21-CL26 CL19 follow-ups: implementation and Serv training startup — 13 August 2026

The six priority experiments from the CL15-CL20/CL19 follow-up report are now
implemented as localized, defaults-off additions over CL19. The configs are
`src/configs/CL21_cosmic_true_soft_router_resididca_v3_24k.yaml` through
`src/configs/CL26_cosmic_anchored_highres_roi_ba_24k.yaml`; the fail-closed
launcher is `launchers/active/run_CL21_CL26_cl19_followups_1gpu.sh`, and
`tools/validate_CL21_CL26_config.py` locks the fixed 96-image validation,
optimizer-step budgets, trainable ownership, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`.

The one-delta arms are: CL21 rank-64 bounded residual identity CA at up0/up1;
CL22 three-state top-object/visible-face/background visibility routing with
deterministic synthetic supervision; CL23 deterministic denoising-progress
low/high-frequency routing; CL24 training-only native-PhotoMaker boundary
distillation; CL25 a 4k weights-only CL19 continuation with sparse low-noise
ArcFace three-reference-centroid reward plus a frozen CL19 prediction anchor;
and CL26 a late bounded 32x32 small-face ROI residual added over the CL19 soft
router. The exact trainable contracts are respectively
`2348/224624676`, `2384/224652396`, `2240/219217920`,
`2240/219217920`, `2240/219217920`, and `2276/219217956`
(tensors/parameters). CL25 pins CL19 `weights-epoch12.pth` at SHA-256
`707cff809414414c0c85e6fcdf52845d3655284a68d54ecb0d657236634492d5`.

The active scientific run ledger at the requested monitoring terminal is:

- CL21 r2: job `lm-mpi-job-fba7a7ca-ce8f-4b65-a7e5-f139cb3187af`, Comet
  `6670db89c44a489388b8f09b91423b0d`, observed through training batch 104;
- CL22 r2: job `lm-mpi-job-84855e01-da1a-4066-b2b3-e71d4904f66e`, Comet
  `b181feb6c54644e69fb7e8709a59f32e`, observed through batch 185;
- CL23 r1: job `lm-mpi-job-f9160c9d-2b18-401d-844c-1e1116f17c3e`, Comet
  `a9ec9c59d1624c68acb98737dcd65298`, observed through batch 300;
- CL24 r1: job `lm-mpi-job-caae3dad-99ab-40ac-80f2-6ebb106f813a`, Comet
  `a18e22ae9f0e4a24b6252f6b392fab62`, observed through batch 428;
- CL25 r2: job `lm-mpi-job-893096da-e633-40cc-9a28-cde68fd4e813`, Comet
  `120b72df8134474ca094e6162d085eb0`, observed through batch 268;
- CL26 r3: job `lm-mpi-job-e07a2b02-6f5b-4ad8-bf80-e1f36c24cd4b`, Comet
  `e9c0a9b505f041a68a183ca3cb4ca0af`, observed through batch 9.

All six completed the canonical step-zero manual-val-96 run, entered training,
and produced finite losses without a post-validation traceback. Monitoring
then stopped as requested. Six project A100s were active, within the explicit
ten-GPU exception for this submission set.

Retain the superseded records as non-scientific startup evidence. CL21 r1
failed before validation on an old one-argument ID-token-mask setter; CL22 r1
and CL26 r1 failed before validation on inherited optimizer category labels;
CL25 r1 failed before validation on a defaults-off strict-manifest field; and
CL26 r2 exposed an fp32-gated ROI residual rejoining a bf16 UNet on its first
validation batch. The replacements change no scientific YAML setting. CL21
r2 seals the compatible two-argument setter. CL26 r3 casts the bounded
residual back to the UNet activation dtype and passed a focused bf16 Serv
processor smoke before submission. Sealed replacement source revisions are
`cl19-cfeda7b5+cl21-cl26-20260813-r2-identity-mask-fix` and
`cl19-cfeda7b5+cl21-cl26-20260813-r3-activation-dtype-fix`.

One submission-loop output interruption also created duplicate CL24/CL25/CL26
requests. The CL25/CL26 duplicates were killed immediately; the CL24 duplicate
failed closed on its non-empty output directory. Peak requested capacity was
nine A100s, below the authorized ten, and the final scientific set is the six
runs above.

Serv NFS had only about 25 GB free at launch and 19 GB at the monitoring stop.
The historical every-epoch
checkpoint semantics were deliberately not changed, so six complete runs can
outgrow that space. Before their first 2k checkpoint boundary, inspect storage
and obtain explicit approval for any old-run cleanup or checkpoint-retention
change; do not silently delete data or change checkpoint comparability.

### Serv checkpoint-retention cleanup — 13 August 2026

The user approved the exact policy “keep the newest full checkpoint and newest
weights-only file in every finished/inactive run directory.” The sealed local
manifest is
`analysis/2026-08-13_serv_checkpoint_deletion_manifest_keep_last_full_and_weights.md`;
its canonical deletion-list SHA-256 is
`a6bc07d632d0a75a16194746616381479ff505ba42566525ae7411485c8f9cfa`.

A fail-closed guard at
`tools/storage/guarded_delete_pth_list.sh` was deployed read-only-equivalent to
Serv as `nasilaev/tmp/guarded_delete_pth_list_20260813.sh` (SHA-256
`c6de32a40f630bb04ae8cf34c0d2ee626d8e32fe39aaae6f2492e3072a5f8838`).
It rejects any candidate outside `nasilaev/`, any non-`.pth`, symlink,
non-regular/missing file, size mismatch, duplicate, count/byte-total mismatch,
or list-hash mismatch. A deliberate non-`.pth` input exited `64`; the sealed
list then passed a complete dry run before execution.

The guard deleted exactly `2,149` approved `.pth` files with logical size
`1,145,599,523,788` bytes and reported `GUARD_DELETE_OK`. Post-delete checks
found zero listed candidates remaining and exactly `120` inactive run
directories retaining `239` endpoint files / `113,501,316,764` bytes. Every
directory has exactly one full checkpoint and at most one weights-only file;
the one documented historical run never had a weights-only endpoint. No
validation images or non-`.pth` files were deleted.

All eight CL14_CA and CL21-CL26 scientific roots were excluded; their MLS jobs
were rechecked as Running immediately before deletion. NFS available space
rose from `12,477,792,256` bytes immediately before cleanup to
`1,153,820,524,544` bytes after reclamation, a net observed increase of
`1,141,342,732,288` bytes while active jobs continued writing. Recheck live
state rather than treating this free-space figure as permanent.

### CL21–CL26 interim result and CL23 successor decision — 14 August 2026

The current-results report is
`analysis/2026-08-14_cl21_cl26_current_results_cl23_fairness_and_next_experiments.md`.
Its reproducible Comet snapshot and hard-case sheets are under
`analysis/assets/cl21_cl26_20260814_current/`; the evidence cutoff is 11:34 UTC
on 14 August 2026. CL21, CL22, CL23, CL24, and CL26 were still running; CL25
had completed its intended 4k continuation. Treat later complete validation
gates as new evidence.

CL23 (`a9ec9c59d1624c68acb98737dcd65298`) is the only CL21–CL26 arm with a
clear persistent matched-step ID gain over CL19. Its current peak is
`0.525002` at 10k. The newly complete 12k gate is `0.518674`, versus matched
CL19 `0.480898`: paired delta `+0.037776`, 73/96 wins, 95% bootstrap interval
`[+0.026193,+0.049563]`. CL23 12k also exceeds CL19 24k by `+0.011851`, but
remains below controlled PhotoMaker step zero (`0.556580`) by `-0.037906`.
Text softens to `26.2713` at 12k, about `0.200` below matched CL19, so the 10k
ID peak is not evidence of monotonic improvement.

The latest one-reviewer hard-case rubric marks CL23 12k Skiing as 5 pass / 1
minor / 2 fail and Crying as 8 / 0 / 0. Skiing mean ID is `0.367513`, below
CL19 final `0.379294` and PhotoMaker `0.464005`; Jisoo and Lex remain the clear
topology failures. Crying is visually resolved on this fixed seed, but its
mean `0.546830` remains below both final CL19 and PhotoMaker.

CL23 is a fair cold-start PhotoMaker+BA arm, not a checkpoint leak:
`trainer.from_pretrained`, `trainer.resume_from`, and `saved_checkpoint` are
all null; branched SA is enabled, branched CA disabled, pose adaptation is
zero, and CA face mixing is false. Its high step zero comes from a deterministic
temporal-frequency merge of nonzero reference-minus-native messages whose base
projections clone the effective PhotoMaker/U-Net weights while branch LoRA B
starts at zero. The absolute score is not BA-only because the generic and
PhotoMaker-default adapters are intentionally co-trained as in CL19. No held-
out native-endpoint/zero-spatial/shuffled-spatial panel exists yet, so causal
attribution of the learned gain to spatial BA remains provisional. Run that
four-arm fixed-96 diagnostic before a promotion claim.

The three priority successors are design-only under
`analysis/blueprints/2026-08-14_cl23_next_three/` and must not be submitted
until implemented, validated, and packaged:

1. CL27: a training-only top-object versus visible-face routed-frequency
   energy loss; exact CL23 inference and ownership.
2. CL28: zero-initialized bounded per-processor corrections to three CL23
   schedule endpoints; predicted ownership `2310/219218130`, to be rederived
   and pinned after implementation.
3. CL29: same-ID positive versus wrong-ID negative contrastive supervision on
   low-band reference messages with detached auxiliary target Q; no ArcFace or
   final-prediction consistency.

All three must cold-start for 24k on Cosmic, preserve the fixed manual-val-96
contract and CL23 routing invariants, and change exactly one scientific
mechanism. Do not repeat CL22's direct dense ownership router, CL24's boundary
epsilon distillation, or CL25's global/decoded ArcFace reward. Keep BigCelebs
out of these mechanism arms; CL20 is current negative evidence for generic
mixing, and a curated hard-case data arm belongs only after a Cosmic mechanism
passes.

### CL27–CL29 implementation, corrected reruns, and live Serv state — 14 August 2026

The three CL23 successors are now implemented behind explicit defaults-off
toggles, with configs at `src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml`
through `CL29_cosmic_lowband_causal_contrastive_24k.yaml`. The shared fail-closed
launcher is `launchers/active/run_CL27_CL29_cl23_followups_1gpu.sh`; the config
gate is `tools/validate_CL27_CL29_config.py`. All retain the cold 24k Cosmic
contract, manual-val-96 step zero/every 2k, batch 2, DDIM50,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false`.

CL27 adds the training-only top-object/high-frequency suppression and
visible-face energy-floor objective at up0/up1, plus 0.25 deterministic
semantic-occlusion sampling. CL28 adds three zero-initialized bounded schedule
corrections per installed SA processor; low-early remains fixed at 0.50 and
step zero is exactly CL23. CL29 adds a sampled branch-local low-band InfoNCE
using a distinct same-ID spatial positive, an in-batch wrong-ID negative, and
a detached target-query anchor. Exact trainable contracts are respectively
`2240/219217920`, `2310/219218130`, and `2240/219217920`
(tensors/parameters).

Retain all superseded attempts as startup evidence. CL27 r1 and CL29 r1 failed
before the first validation image because the alternate RealVis pipeline did
not mirror the new processor-extension map. CL28 r1 failed the exact ownership
category gate. CL27 r2 then showed that alternate-base validation keeps modules
in train mode under `no_grad()`; its training-only surface loss incorrectly
requested an ownership mask. CL28 r2 also retained the old validation map.
CL28 r3 completed all 96 step-zero images but failed before its first optimizer
batch because `writer.loss_names` requested nonexistent SDXL attention groups
`down0` and `up2` (`KeyError: ba/frequency_low_scale/down0`). CL29 r2 was not a
runtime failure: its sole submission attempt was rejected by the Serv
allocation gate before job creation.

The corrected shared source revision is
`cl28-cl29-training-transition-fix-20260814-v5`. CL28 now logs only the five
installed SDXL attention groups (`down1`, `down2`, `mid`, `up0`, `up1`), and
the validator pins that exact set. CL29's sampled dual-reference auxiliary now
requires both `self.training` and `torch.is_grad_enabled()`, so alternate-base
validation cannot enter the training-only path merely because modules retain
train mode. The validation pipeline mirrors all three extension maps.

Current immutable scientific records at the live monitoring point are:

- CL27 r3: Serv `lm-mpi-job-6af73e51-e281-4356-adde-767f15cc7607`, Comet
  `dbfbf40c3bdd4f70bedc58bda3dfb9cd`; step-zero full96 completed and training
  was observed through optimizer batch 28 with finite progress.
- CL28 r4: Serv `lm-mpi-job-6681ed16-1d71-4175-a5a0-fca7b3a1b632`, Comet
  `3d8aca3b4cbb4ddc9338f14952c5bd0e`; exact ownership gates and step-zero
  full96 completed, then training was observed through optimizer batch 18 at
  approximately 7.1 seconds per iteration with no telemetry error.
- CL29 r3: Serv `lm-mpi-job-4977ec3e-aab0-4793-9330-116e721801f5`, Comet
  `2981820837564d01b1cefbf52c4dabd0`; exact gates and the 64/64 alternate-ref
  dataset preflight passed; step-zero full96 completed after a slow but
  non-failing model-load phase, then training was observed through optimizer
  batch 27 at approximately 7.1 seconds per iteration.

The user explicitly authorized up to ten A100s for CL27–CL29. The two corrected
reruns were submitted in priority order after a live project count of six;
both were accepted, bringing the conservative project request count to eight.
Monitoring stopped after all three scientific runs completed step-zero full96
and multiple finite training batches. Recheck live MLS state rather than
treating the resource snapshot as permanent.

### CL14–CL29 training-throughput diagnosis and optimization plan — 16 August 2026

The audited plan is
`analysis/2026-08-16_cl14_cl29_training_throughput_optimization_plan.md`.
Warmed cross-epoch Serv stderr medians are CL14 `2.19 s/it`, CL19 `3.94`,
CL23 `5.50`, CL26 `6.12`, CL27 `7.23`, CL28 `7.02`, and CL29 `7.37`; the
latest arms are `2.79x–3.37x` slower than CL14. CL14 and CL26 are complete;
CL27–CL29 were still running at the 16 August 12:25 Europe/London evidence
cutoff. No training job was launched and no training implementation was
changed for the diagnosis.

The highest-priority common regression is an every-step post-backward
`_record_active_gradient_norms()` pass introduced in the CL19 source era. It
casts, squares, and reduces all `219,217,920` trainable gradients even though
CL19/CL23/CL26–CL29 do not request the `active_grad_norm_*` metrics and
`max_grad_norm=null`. The exact CL14 clean-port source has no such pass. Other
definite costs are CL26's discarded `_call_legacy()` result before rebuilding
its CL19 baseline; CL27's Python `bool(eligible.any())` CUDA synchronization at
each of 36 up0/up1 processors; CL23+ full-fp32 5x5 Gaussian filtering and
uncadenced full-activation telemetry at all 70 BA processors; and CL29's GPU
`.item()` sampling, unconditional alternate-reference loading, and sampled
second branched U-Net forward.

Treat seconds recovered per item as unmeasured until matched CUDA-event and
profiler speed smokes run. Implement the four pipeline-neutral P0 changes
behind explicit compatibility toggles: skip unrequested active-gradient
metrics, remove CL26's dead baseline, tensorize CL27 eligibility without host
sync, and separate loss state from telemetry while preserving historical
metric definitions. Keep the fixed manual-val-96 contract, seeds, prompts,
references, boxes, DDIM50, batch 2, trainable ownership,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false` unchanged.

Source provenance must not be reduced to a false single commit. CL14's Git
ancestor is `c04970f342a186d1092f07f9a08d7d8a797383e8` plus the sealed
`cl12-cl14-snapshot-v1-20260809` overlay; CL19's is
`ad194a026ab701dd979712d415c487dd536a4645` plus the sealed 11 August
overlay. CL23 and later revision labels are not Git SHAs; their exact source is
the per-file SHA-256 manifest. Future runtime packaging should record Git HEAD
and dirty-overlay identity in addition to the existing manifest.

### CL29 throughput implementation and active qualification — 16 August 2026

The throughput plan's pipeline-neutral changes are implemented and pushed on
branch `test`. Commit `20d32369f35521460689e6adf9b87bae099a607d` adds an
opt-in CL29 speed configuration and launcher. It skips the unused 219M-parameter
active-gradient scan, suppresses unrequested hard-case activation telemetry,
removes CL26's discarded legacy attention result, tensorizes CL27 eligibility,
and moves CL29 sampling/permutation decisions off the active CUDA stream. The
CL29 objective, two-pass sampled auxiliary, trainable contract
`2240/219217920`, batch 2, 24k budget, fixed manual-val-96 every 2k, DDIM50,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false` remain unchanged. Per the
user's explicit request, only initial step-0 validation is omitted.

Retain two non-scientific startup records. R4, Serv
`lm-mpi-job-21052a15-2089-4215-9442-7e6c80eb2066`, Comet
`92ca4b49bfb24af997093a9de7278b3b`, was stopped before GPU placement after a
35-minute cold initialization stall. R5, Serv
`lm-mpi-job-24402a1d-f1cb-4e61-92e1-83328a75320e`, Comet
`8aac42e2e7334c8e82f9665d35cbd6a8`, populated a persistent Hugging Face cache
but exited on a native SIGSEGV at its first training batch before yielding a
timing sample.

The active exact retry is
`CL29_cosmic_lowband_causal_contrastive_24k_full96_r6`, Serv
`lm-mpi-job-f92ed6b8-cf48-4885-a7f0-648f0689b825`, immutable Comet key
`2c5d2e18558249138e5edf7b6be0b01f`, and sealed source commit
`8dec793afcd78d5b550960f91908a443578f4bd1`. It passed source, config, dataset,
Comet-record, and exact trainable-ownership gates; no step-0 validation ran.
Warm-cache base preparation fell from R5's `31.016 s` to `5.903 s` and
processor installation completed at `19.226 s`.

At the early qualification cutoff, batches 21-80 had a displayed-rate median
of **6.21 s/iteration** over 60 samples and mean integer wall-clock deltas of
`6.35 s/iteration`. The matched CL29 r3 warmed median is `7.37 s/iteration`,
so the requested run-rate check passes with an approximately **15.7%** median
improvement. Ordinary steps settle near 6 seconds; sampled contrastive steps
remain slower because the unchanged CL29 objective performs its second
branched U-Net pass. R6 was finite and still Running after batch 80; leave it
active, then verify sustained throughput plus the fixed 2k validation gate
before any scientific promotion claim. Recheck live MLS state rather than
treating this snapshot as permanent.
