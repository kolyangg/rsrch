# CL39N architecture implementation and Serv launch log

Date: 31 August 2026  
Branch: `test` (pushed through `461c207`)  
Sealed source revisions: `ff2b01b` (N7–N9 and N6R r1), `cd50401` (N6R recovery)

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
| N6R seed-1 confirmation recovery r3 | `lm-mpi-job-20eae07e-5f14-44b9-8e42-3b6f1b4a82f4` | Accepted from fresh paths with complete inherited dataset environment | None (console-only confirmation) |

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

N6R production remains correctly blocked until seed-1 quantitative gates and
the eight-page visual review both pass. Registration or job state alone is
not promotion evidence.
