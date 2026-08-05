# Current project handoff

**Last updated:** 4 August 2026

**Repository:** `/home/kolyangg/rsrch_apr_test`

**Primary project:** `diffusion_template/`

**Branch:** `test`

**Current local and Serv commit:** `c04970f342a186d1092f07f9a08d7d8a797383e8`

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
joint-mechanism suite over the earlier data-factorial-first ordering. The suite
is now implemented and named E13-E18, but remains unsubmitted. E13/E14 are
explicit shadow-co-adaptation mechanism arms; E15-E18 are fully persisted
promotion arms covering protected joint training, a predicted-x0
PhotoMaker-CLIP identity proxy, bounded residual identity-token CA, and
deterministic decoupled multi-reference training. Every arm keeps
target-query/same-ID-reference-KV spatial BA, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`.

Implementation entry points:

- configs: `src/configs/E13_*` through `src/configs/E18_*`, all inheriting
  `large_dataset_joint_r128_24k.yaml`;
- controlled launcher: `launchers/active/run_E13_E18_large_ds_24k_1gpu.sh`;
- immutable pre-launch records: `experiments/large_dataset/E13_*` through
  `E18_*`, status `ready_not_submitted`;
- one-A100 MLS YAMLs and startup wrappers:
  `serv_run_packages/E13_*` through `serv_run_packages/E18_*`;
- E17's residual processor is defaults-off and separate from E12's hard v2;
- E18 uses an exact 48,000-row deterministic sequential dataset and passes all
  identity references to PhotoMaker while preserving `ref_images[0]` as the
  sole spatial latent/KV reference.

No E13-E18 job or Comet experiment has been created. Before simultaneous
submission, inspect current Serv Running/Pending allocations and keep this
project at the normal six-A100 ceiling. The existing E11/E12 Serv worktree must
be updated to the final pushed `test` branch and clean before package startup.
