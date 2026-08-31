# CL39N architecture implementation and Serv launch log

Date: 31 August 2026
Branch: `test` (implementation pushed through `8bb2fb1`)
Sealed source revisions: `ff2b01b` (first qualifications), `cd50401`
(N6R recovery), `8bb2fb1` (direct N7–N9 production)

## Implemented architecture leaves

| Arm | Single architectural delta from CL39 | Production run name |
|---|---|---|
| CL39N6R | Disable only the frozen `up_blocks.1` low-band group selected by the predeclared audit rule; optimizer membership is unchanged. | `CL39N6R_cosmic_up1_low_pruned_24k_full96_r1` |
| CL39N7 | Route confidence from posterior invalid-key mass while preserving zero-key SDPA and a bounded native fallback. | `CL39N7_cosmic_posterior_null_router_24k_full96_r1` |
| CL39N8 | Remove only the positive native-parallel component of high-band BA output in `up_blocks.0/1`. | `CL39N8_cosmic_native_orthogonal_highband_24k_full96_r1` |
| CL39N9 | Add a raw InsightFace-token sidecar through a `512→1024→4×2048` projector and rank-32 local K/V/output adapters in the 36 `up_blocks.0/1` cross-attention processors. | `CL39N9_cosmic_intrinsic_id_sidecar_24k_full96_r1` |

All leaves preserve the fixed 96-image validation panel, prompts, references,
boxes, seeds, scheduler, 24k schedule, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`. N6R's frozen map SHA-256 is
`858c4663083ccffbd461e94215d4e9951f2765b59b4f49ce454de92c5910904f`.

## Throughput recovery and gates

The prior 6–7 s/iteration regression occurred after the full-validation
lifecycle. The new defaults-off trainer option
`preserve_training_model_during_validation` keeps the training model resident
while the alternate validation base is temporary, avoiding the full
training-model CPU/GPU offload-and-restore round trip. Historical configs keep
the old default; only the new validated smokes and production leaves opt in.

Every production allocation runs two fail-closed qualifications before Comet
registration:

1. sealed source/config/Cosmic preflight and 100 optimizer steps without
   validation;
2. the real fixed-96 step-zero validation lifecycle followed by 100 optimizer
   steps, requiring the resident-through-validation marker and exactly 96
   valid images;
3. only after both gates pass, create a fresh immutable Comet experiment and
   begin the 24k run.

The first sealed N7–N9 allocation used `validation_interval_steps: null`,
which this trainer interprets as periodic validation every epoch rather than
disabled. Their 100-step timing windows occur before that extra validation and
remain usable; the live jobs were not stopped or modified. Commit `461c207`
sets all future smoke intervals to `0` and makes the checker assert the
mode-specific absence/presence of the validation-start marker. Validated
smokes retain their initial fixed-96 panel but no longer repeat it at step 100.

At the first live no-validation observation, N7 was approximately `3.9–4.2`,
N8 `4.2–4.5`, and N9 `3.8–4.1` s/iteration after warm-up. These measurements
show that the direct training hot path is back near its intended range; the
second gate remains authoritative for the former post-validation slowdown.

The completed fixed-96 qualifications passed every scientific, finite,
image-count, mechanism-activity, and resident-model check, but failed their
post-validation speed ceilings: N7 `5.62` s/it (ceiling `5.0`), N8 `5.94`
(ceiling `5.0`), and N9 `5.63` (ceiling `5.5`). No qualification created a
Comet experiment. At the user's explicit direction to make the production
runs live, N7–N9 were submitted directly from a fresh sealed runtime; the
throughput shortfall remains an operational issue and is not represented as
fixed.

The current immutable production mappings are:

| Arm | Serv job | Comet key |
|---|---|---|
| N7 | `lm-mpi-job-082bb824-370a-4778-8024-f3d6bf746148` | `d8c2aacddd0d465d9892ec19da1a7e06` |
| N8 | `lm-mpi-job-8df80bfa-1389-4097-b627-d003dc92a4cd` | `78cd414ea37343cc88cf2136b14a9f70` |
| N9 | `lm-mpi-job-1979ecab-cdc7-407b-8f6a-0c3690cfe212` | `bd29402f968b49ce8a64c31ba380c48b` |

All three were Running and had passed source-manifest, configuration,
Cosmic-preflight, exact trainable-contract, and immutable-Comet startup gates
at the final audit. They were executing the standard step-zero validation;
this is not yet a claim about optimizer progress or final scientific quality.
The direct-production archive SHA-256 is
`6d0d2926881b8be4479913b267fb999962d4c3d451d56234798ff0c1f485264d`.

## X12 replacement fourth architecture

N6R's independent seed-1 intervention reduced subject-v2 identity by
`-0.005748`, increased face SSIM to PhotoMaker by `+0.002140`, and reduced
TOPIQ-Face by `-0.003733`; pruning a routed band is therefore not promoted.
`[measured]` X01 valid-key attention was a matched statistical tie rather than
an improvement, but it established that excluding invalid zero keys is stable.
Its confidence was recalculated on valid keys, increasing and recalibrating the
correction budget at the same time. `[report]`

The replacement arm is
`CL39X12_cosmic_valid_kv_legacy_confidence_24k_full96_r1`. Its single change
from CL39 is valid-only reference K/V message support; confidence remains the
original detached CL39 masked-full entropy statistic. `[code]` It adds no
parameters and retains the exact `2,240 / 219,217,920` trainable contract,
fixed manual-val96 inputs, scheduler, data, losses, 24k schedule,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false`.

Primary decision metric is paired subject-v2 identity versus matched CL39;
PhotoMaker face SSIM is a dissimilarity constraint, not the optimization
target. `[hypothesis]` X12 should retain X01's useful message support without
forcing its larger valid-only confidence budget. Promote only with paired ID
interval lower bound above zero, Skiing within `0.005`, no identity below
`-0.015`, and TOPIQ-Face/MANIQA within `0.005` of CL39.

| Claim | Confidence | Basis |
|---|---|---|
| X12 message equals X01 | High | Focused tensor gate, maximum absolute difference `0` |
| X12 confidence equals CL39 | High | Focused tensor gate, maximum absolute difference `0` |
| X12 will improve identity | Not established | X01 tied CL39 and X12 has not trained |
| X12 avoids N6R's pruning failure | Medium | It removes no routed band, but final metrics remain unknown |

Not established: X12 has no trained checkpoint, so neither identity gain nor
visual difference from PhotoMaker can be claimed from its construction gates.

X12 was submitted on Serv as
`lm-mpi-job-b5f9916a-904f-4a66-842d-3c75c43e5f77` from source commit
`36f6d27ec64b5b570ef7c8ff4e437107da85ff65`; its immutable Comet key is
`d2e3d71bc1824959886ece2ba6ddb157`. At the startup audit it was Running and
had passed the source-manifest, CPU architecture, exact trainable-contract,
64-record Cosmic data, and immutable-Comet gates. Model initialization was in
progress, and a subsequent audit verified the installed processor/optimizer
ownership and the intended NO_ID → PhotoMaker → BOTH routing switches during
the first item of standard step-zero validation. This is not yet evidence of
optimizer activity or scientific quality. The source archive SHA-256 is
`0b0f126a67513db0b38442f316b02eb894fd33fa116a84f3a2efd646c72a0def`.

The r1 job later completed all 96 step-zero images and entered finite training,
then failed on batch 3 with
`KeyError: ba/valid_kv/valid_fraction/all`. `[measured][code]` The scientific
forward was valid; the detached valid-KV diagnostics are cadence sampled, but
the writer requires every declared scalar on every step. Recovery r2 retains
the exact architecture and emits zero placeholders only when those diagnostics
are unsampled. A focused sampled/unsampled schema check passes. The ONNX CUDA
fallback warnings and Comet status timeout were not the terminal cause.

Recovery r2 is the proper unchanged 24k fixed-96 run: Serv
`lm-mpi-job-f599e54b-87a1-49c5-85c4-b29cbac69d12`, Comet
`e7e614c6f9a84f01a2acdb7ac4da234d`, source commit `8effa14`. `[measured]`
It passed sealed-source/config/data/Comet startup and is generating the
step-zero panel. Crossing the former third-batch crash and first later sampled
diagnostic remain the integrity gates.

## N3 failure and separately named successor

N3R2 r3 completed exact resume and fixed-96 step-2k validation, then failed
its third hard-reference activity attempt: cadence `9/9`, achieved-margin mean
`0.009998`, hinge mean `0.000030`, BA/native gradient ratio `0.002311`, and
calibration saturated at `0.075`. `[measured]` The squared-hinge objective is
therefore rejected; it was not weakened or silently restarted under N3.

The separately named N3S successor changes only that rank penalty to
temperature-`0.02` softplus. Its 101-step qualification passed with exact
applications `11/11`, nonzero losses `11/11`, achieved-margin mean `0.009578`,
and calibrated weight mean `0.005674`; throughput was `5.05` s/iteration.
`[measured]` Proper 24k fixed-96 production then started from the exact common
pre-auxiliary checkpoint SHA-256
`c995a10102ec746474d6a3bf7652afec0846d72c5a0a45a786b9d0a7f38492aa`:
Serv `lm-mpi-job-0a9cbd11-cafb-40df-a709-331ca4458472`, Comet
`aa08052261f74d9bb3aa65334aa36c27`. It passed source/config/data/Comet and
checkpoint-hash startup checks; fixed-96 validation and production activity
remain pending.

## Serv submissions

The pre-submit recount was four Running and zero Pending project A100s. The
four initial one-GPU requests were accepted, for eight total requests under
the user's run-scoped ten-GPU exception.

| Purpose | Serv job | State at launch audit | Comet |
|---|---|---|---|
| N6R independent seed-1 confirmation r1 | `lm-mpi-job-f9f16542-d31e-4d66-9f5d-2403c92dc95f` | Failed before validation; preserved | None (console-only confirmation) |
| N7 qualification → production | `lm-mpi-job-8d26b5da-9c18-4216-b8f3-69a7af99e133` | Running; sealed preflight passed and mechanism active | Deliberately deferred until both gates pass |
| N8 qualification → production | `lm-mpi-job-12ec7cf3-a2ee-4d2f-95ec-df6000849664` | Running; sealed preflight passed and mechanism active | Deliberately deferred until both gates pass |
| N9 qualification → production | `lm-mpi-job-60ebd9fd-b324-4eb0-b597-1f7bb08344cc` | Running; sealed preflight passed and mechanism active | Deliberately deferred until both gates pass |
| N6R seed-1 confirmation recovery r2 | `lm-mpi-job-e5450e00-be90-4693-a805-93edcb1cf14d` | Failed before model load: inherited Cosmic environment omitted | None (console-only confirmation) |
| N6R seed-1 confirmation recovery r3 | `lm-mpi-job-20eae07e-5f14-44b9-8e42-3b6f1b4a82f4` | Running; source/data and legacy-manifest gates passed, first all-on validation active | None (console-only confirmation) |

N7–N9 use sealed archive SHA-256
`a42fbf071ca63b1d87a4b330000ede173bfad2bcb17a94fd7a6febc0854adb67`.
The N6R recovery uses sealed archive SHA-256
`a9e31e22fc22de7951a3e865353752f9e098cca75e01877272e73eb0c6a90e65`.

## N6R failure and prevention

The immutable CL39 epoch-8 checkpoint stored the parameter-free null-key
router using the historical exact
`hardcase_route.cl38_cl44_extension` marker. Current code records the same
CL39 defaults under the explicit `null_key_confidence_router` schema, so strict
raw manifest equality rejected r1. Commit `cd50401` canonicalizes only that
exact legacy marker to the immutable full CL39 defaults. Different groups or
thresholds still fail closed. Each recovery uses fresh package, runtime,
confirmation, and log paths; r3 must emit
`CL39_LEGACY_ROUTER_MANIFEST_CANONICALIZED` before checkpoint loading is
accepted.

Recovery r2 then exposed an independent launch-wrapper omission: a
validation-only trainer still instantiates the inherited Cosmic train dataset,
but the wrapper had not exported its root and manifest. Recovery r3 restores
and verifies both variables before launch. This is an operational environment
fix; the N6R architecture, checkpoint, map, prompts, seeds, and validation
contract are unchanged.

N6R production remains blocked. Its exact seed-1 join completed for all 96
cells, but the predeclared quantitative gate failed: identity mean delta
`-0.005748`, PhotoMaker face-SSIM delta `+0.002140`, TOPIQ-face mean delta
`-0.003733`, and TOPIQ-face p10 delta `-0.025925`. Registration or job state
alone is not promotion evidence.
