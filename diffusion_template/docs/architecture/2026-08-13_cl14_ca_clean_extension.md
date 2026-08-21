# CL14_CA clean extension

## Outcome

`CL14_CA_cosmic_residual_identity_ca_24k` is a minimal, fail-closed delta over
`CL14_cosmic_joint_shadow_sa128_softmask_24k`. It reproduces the latest test
production design, `CL14_CA_optimized_r11`, without importing the test branch's
older CA variants, smoke matrix, runtime snapshots, or relaunch history.

Source provenance:

- test branch revision inspected: `ceb34c3`;
- latest production run: `CL14_CA_optimized_r11`;
- immutable Comet key: `fafd7a61b06c4114b9dec2c21d29ca38`;
- production runtime: `runtime_sources_cl14_ca_v23`;
- observed startup contract: 2,348 tensors / 224,624,676 parameters;
- observed startup evidence: complete 96-image step-0 validation, finite first
  loss `0.064709`, and training through at least step 10 at about 3.23 s/it.

No job was launched while preparing this clean port.

## The one scientific change from CL14

CL14's spatial self-attention, dataset, two-cell training-mask feather, loss,
LR schedule, 24k step budget, prompt conditioning, and native PhotoMaker/text
cross-attention remain unchanged. Only cross-attention processors in
`up_blocks.0` and `up_blocks.1` add:

```text
native_target_CA
+ target_face_mask
  * (0.20 * sigmoid(gate_logit))
  * RMS_normalize(rank64_delta(target_Q attends active PhotoMaker ID-token K/V))
```

The output delta is zero-initialized, the gate starts at `0.02`, and its maximum
is `0.20`. The reference lane remains native. Legacy branched CA stays disabled,
as do pose adaptation and face CA mixing. Consequently a newly initialized
CL14_CA processor produces the exact CL14 CA output before its first update.

The implementation lives in
`src/model/photomaker_branched/residual_identity_ca_processor_v3.py`; installation
and exact `up_blocks.0/1` selection are the only CA routing additions in
`branched_runtime.py`. All new source logic is marked with dated `CL14_CA-*`
comments explaining the invariant or benefit.

## Trainable and checkpoint contract

CL14 owns 2,240 tensors / 219,217,920 parameters. CL14_CA adds exactly 108
FP32 tensors / 5,406,756 parameters: two rank-64 output matrices plus one scalar
gate for each selected CA processor. Total ownership is therefore:

| Role | Tensors | Parameters |
|---|---:|---:|
| Branched SA rank 128 | 840 | 127,795,200 |
| Residual identity CA rank 64 | 108 | 5,406,756 |
| Generic effective adapter rank 32 | 700 | 30,474,240 |
| PhotoMaker default effective adapter rank 64 | 700 | 60,948,480 |
| Total | 2,348 | 224,624,676 |

`e13_contract.py` includes the residual selector, rank, gate, routing equation,
zero-init invariant, exact names, shapes, and dtypes in schema-v2 checkpoints.
Loading a CL14 checkpoint as CL14_CA, or a CL14_CA checkpoint without its CA
route, fails before generation.

## Validation-only repair

Latest production used the corrected subject-v2 Eddie contract. The clean leaf
therefore inherits the existing CL18-CL20 validation wrapper: `bbox_overlap_v2`
binds the PhotoMaker identity vector to the declared reference face, while the
primary subject-v2 metric evaluates the generated face owned by the exact BA
box. The fixed 96 prompts, seeds, references, bboxes, DDIM50 scheduler, CFG 5,
RealVis base, batch 12, checkpoint cadence, and legacy processor-base copy stay
unchanged. This repair changes validation identity selection, not CL14_CA's
training architecture or Cosmic loader.

## Execution-only improvements

The latest safe speed improvements are kept separately from model semantics:

- target and reference native CA rows are concatenated for one projection/SDPA
  call, then split; the attention equation and row independence are unchanged;
- active ID-token indices are validated once per U-Net call and reused by all
  selected CA processors;
- the 19 training scalars are stacked for one device synchronization, and the
  one-GPU Serv path bypasses the unnecessary distributed gather.

These optimizations do not change parameter ownership, loss, random sampling,
or generated outputs.

## Exact files and launch path

- Hydra config:
  `src/configs/CL14_CA_cosmic_residual_identity_ca_24k.yaml`
- active launcher: `launchers/active/run_e13_family_24k_1gpu.sh`
- Serv entry point: `launchers/serv/start_cl14_ca_1gpu.sh`
- exact one-A100 YAML:
  `serv_run_packages/CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1/run_CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1_1gpu.yaml`
- config gate: `tools/validate_cl14_ca_config.py`

From the Serv checkout, populate the gitignored `diffusion_template/.env` with
the existing PhotoMaker, Cosmic corrected-r2 manifest/root, subject-v2 asset,
Comet, and deferred face-quality paths. Pull `clean`, require an
empty worktree, and inspect this project's Running/Pending MLS jobs so the
normal six-A100 request ceiling is respected. Then submit the exact YAML:

```bash
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_e13_family_clean/diffusion_template/serv_run_packages/CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1/run_CL14_CA_cosmic_residual_identity_ca_24k_full96_clean_r1_1gpu.yaml
```

The YAML refuses a dirty or wrong branch and records the source commit. The
shared launcher verifies source/config parity, Cosmic and subject-v2 hashes,
dataset decoding/policy, ONNX Runtime CUDA, a fresh run directory, and an
immutable 32-character Comet key before allowing training to continue.

## Verification gates

Before promotion, run:

```bash
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py
python tools/validate_cl14_ca_config.py
bash -n launchers/active/run_e13_family_24k_1gpu.sh
bash -n launchers/serv/start_cl14_ca_1gpu.sh
```

Also compile the changed Python files, compare the clean residual processor
against the latest test implementation with identical weights/inputs, verify
zero-init equality to native CL14 CA, and exercise schema-v2 save/load and the
2,348-tensor ownership gate on an A100 startup before treating a new run as
promoted. The prepared job has not yet performed that final A100 gate.

Local verification on 13 August 2026 passed Python compilation, shell syntax,
YAML parsing, the three-family composition gate, all CL18-CL20 regression
gates, the CL14 sealed-source/fixed-96 parity gate, and the dedicated CL14_CA
diff gate. A deterministic processor fixture compared this clean implementation
to test revision `ceb34c3` after loading identical nonzero weights: maximum
absolute output, input-gradient, prompt-gradient, and processor-gradient error
were all `0.0`. A separate zero-init fixture matched ordinary native CL14 CA
with maximum absolute error `0.0`. A memory-free full RealVisXL U-Net topology
audit installed the real processor map and reproduced every ownership category,
including the exact total of 2,348 tensors / 224,624,676 parameters. The real
A100 startup and checkpoint round-trip remain intentionally unexecuted until
the prepared job is launched.
