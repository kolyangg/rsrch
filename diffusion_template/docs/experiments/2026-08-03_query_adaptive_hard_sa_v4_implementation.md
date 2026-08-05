# Query-adaptive hard BA v4 implementation

## Outcome

E6-H removes the learned native/reference face mixer that retreated toward
PhotoMaker in E5. Inside each target-face self-attention mask it now uses a
branch-adapted target query against true-mask-filtered reference K/V. Outside
the face it retains native target self-attention. There is no alpha, mix, gate,
RMS matching, pose adaptation, or cross-attention face mixing.

The historical `hard_replace_v1`, `residual_sa_v2`, and `anchored_mix_sa_v3`
paths remain available. The new behavior is selected only by
`model.ba_architecture_version=query_adaptive_hard_sa_v4`.

## Trainable architecture

- 46 self-attention processors: `mid`, `up0`, and `up1`.
- branch-only target Q: rank 16, FP32;
- explicit reference K and V: rank 32, FP32;
- reference output residual: rank 32, FP32, zero initialized;
- exact startup contract: 368 tensors / 12,328,960 parameters;
- optimizer roles: `ref_query`, `ref_kv`, and `ref_output` only;
- forbidden and absent: `mix`, `gate`, generic U-Net LoRA, and the default
  PhotoMaker adapter.

The processor logs route telemetry including reference/native RMS and cosine,
branch-query change, valid reference-key fraction, denoising progress, and the
hard invariant `hard_face_native_leakage=0`.

## Experiment configuration

- Config: `src/configs/big_celebs_scheduled_rhca_query_adaptive_hard_sa_v4_20k.yaml`
- Neb launcher: `launchers/neb/start_rhca_big_celebs_scheduled_query_adaptive_hard_sa_v4_20k.sh`
- Run: `rhca_big_celebs_scheduled_v1_hard_ba_v4_q16_r32_20k_full96_r1`
- Budget: 20,000 steps, ten 2,000-step epochs
- Validation: fixed manual full-96 at step 0 and every 2,000 steps
- Data: the same sealed BigCelebs scheduled policy-v1 source
- Required routing controls: `pose_adapt_ratio=0` and
  `ca_mixing_for_face=false`
- Comet: `jul-comet-large-testing-tr`, immutable experiment
  `408606871a5b40c6b75d2da855b83a44`

## Verification and launch

The focused numerical check proved exact hard replacement inside the mask,
exact native output outside it, zero native-face leakage, identity-at-init for
the query clone, and finite nonzero gradients in all three optimizer roles.
A full PhotoMaker construction then verified all 46 processors and the exact
trainable contract. Python compilation, Hydra composition, shell syntax, and
the audited runtime hash gate also passed locally and on Neb.

The previous E5 process was stopped cleanly with `SIGTERM`. Its runtime files
were backed up under
`/home/niko/rsrch/runtime_backups/hard_ba_v4_20260803_093826`. E6-H started on
Neb on 3 August 2026 under launcher PGID `3518720`; its training PGID is
`3518939`. The immutable Comet record exists and the strict startup contract
passed.
