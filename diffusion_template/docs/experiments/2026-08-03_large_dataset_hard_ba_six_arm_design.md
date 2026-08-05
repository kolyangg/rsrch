# Large Dataset hard-BA E0 baseline and six-arm experiment design

**Date:** 3 August 2026

**Status:** E1-E6 running on Serv; historical and fixed E0 controls prepared locally and awaiting approval

**Training dataset:** adjusted Large Dataset, 47,500 images / 2,561 identities

**Historical anchor:** `rhca_large_dataset_sameid_40k_full96_r4`

**Historical Comet key:** `a99db1fb953d4511827672380e6c1645`

The source recovery and the explanation for the weaker historical two-GPU
Serv result are in
[`2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md`](2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md).
That audit also establishes E2 as the controlled test of the useful output
capacity accidentally present in historical r4.

### Prepared matched E0 controls

Two one-A100, 20k controls are prepared locally, have no MLS or Comet identity,
and must not be submitted until the user approves and two slots are free:

- `E0_large_ds_base_historical_r4_20k_full96_r1` reproduces r4's observed
  171.29M fail-open ownership and incomplete legacy checkpoint behavior;
- `E0_large_ds_base_fixed_baonly_r32_20k_full96_r1` materializes the shared
  audited parent with exact 31.95M BA-only ownership and complete checkpoints.

The pair directly measures the full historical-ownership effect. E2 remains
the clean branch-local test of whether output-basis capacity is its useful
component. E1-E6 are unchanged.

### Live six-arm submission record

| Run | MLS job | Immutable Comet key | Verified startup ownership |
|---|---|---|---|
| `E1_large_ds_truekey_r32_20k_full96_r1` | `lm-mpi-job-a686e213-b211-48e2-bc0b-7a26ae06f307` | `ce0c9b918d79449b92fa83ef970285c3` | 840 tensors / 31,948,800 parameters; optimizer 840/840 |
| `E2_large_ds_branchout_r32_20k_full96_r1` | `lm-mpi-job-555ea214-95e9-41f6-a470-68587451dcd6` | `4c8af4e867b14377b69fa250fae5cde9` | 980 tensors / 37,273,600 parameters; optimizer 980/980 |
| `E3_large_ds_roiwarp_r32_20k_full96_r1` | `lm-mpi-job-404c8887-7a3f-49c7-aa6c-7c23eebe485b` | `9c5cbe4e49254134b4763ff7a4554c9b` | 840 tensors / 31,948,800 parameters; optimizer 840/840 |
| `E4_large_ds_midup_r32_20k_full96_r1` | `lm-mpi-job-5160fbfc-be6e-478f-8099-b6dfb161880e` | `2d77f35256844e0399c1834859a45dc7` | 552 tensors / 21,135,360 parameters; optimizer 552/552 |
| `E5_large_ds_infersteps_r32_20k_full96_r1` | `lm-mpi-job-ce87c84b-cf29-4570-8c52-c6c2cf438bdc` | `4a107cbc30a04a858de3e3b5c411cdca` | 840 tensors / 31,948,800 parameters; optimizer 840/840 |
| `E6_large_ds_fp32_r32_20k_full96_r1` | `lm-mpi-job-5fbd78a3-fd27-479a-9660-aa81813db9c9` | `9f3e20a75a0a4304b12d724693e13fbf` | 840 tensors / 31,948,800 parameters; optimizer 840/840; FP32 BA state |

All six jobs passed the exact config delta, 64-image decoded dataset preflight,
CUDA ONNX Runtime, PyIQA, audited runtime hash, online Comet registration,
experiment-comment, exact ownership, and optimizer-membership gates. Every arm
entered strict `legacy_full_copy` step-zero validation on the fixed 96-image
panel. The four later submissions filled the project ceiling exactly: six
Running/Pending one-A100 requests after E6 was accepted. Serv's preserved
checkout reported historical HEAD
`c04970f`, so the 49 launch files were selectively synchronized from pushed
commit `e860f9e`, backed up first, and SHA-256 verified individually. This
explicit source-sync record is retained in all six experiment JSONs.

## Decision summary

The six experiments below preserve the historical Large Dataset face-attention
equation: target queries use same-identity reference K/V inside the target face,
while native target attention is used outside the face. There is no
native/PhotoMaker face-attention interpolation, learned mix, residual reference
gate, `pose_adapt_ratio`, or branched cross-attention. Each arm changes exactly
one scientific element relative to the audited baseline configuration.

All six arms have a 20,000-step process ceiling, use one A100 each, and validate
at step 0 and every 2,000 optimizer steps on the unchanged fixed 96-image
`manual_val` panel. Twenty thousand steps covers the historical recovery region
and the BigCelebs 18k peak without repeating the uninformative 32–40k tail.
Checkpoints remain available every 2k so a dominated or broken arm can be
stopped at an earlier decision gate.

## Evidence that determines this design

Observed results, not hypotheses:

- The Large Dataset historical run is still the strongest long hard-route run:
  identity peaked at `.3904` at 24k and was `.3797` at its last complete 34k
  gate.
- The 7.35-times-larger BigCelebs dataset did not raise the ceiling. Its base
  peaked at `.3817` at 18k and regressed while most of the dataset remained
  unseen. Data exhaustion is therefore not a credible primary explanation.
- The historical `train_ba_only=true` installation failed open. About 171.29M
  parameters trained, including the pretrained PhotoMaker adapter and a generic
  U-Net adapter, and not all live state was checkpointed. The historical curve
  is a result anchor, not a clean BA-only attribution result.
- Clean hard BA32 retained good prompt and face-quality behavior but reached a
  lower identity plateau. Its reference K/V attention normalized over masked
  zero keys, its face message had no branch-local output basis, and its BF16
  trainables/Adam state were strongly top-heavy.
- Residual SA-v2 stayed close to PhotoMaker and failed the spatial-reference
  causal-use gate. Anchored mix v3 became causally active but learned to retreat
  toward the native PhotoMaker path. Neither is eligible here because the user
  requires hard, core BA with no PhotoMaker face-output mixing.
- Query-adaptive hard BA-v4 proved that clean no-mix BA can learn: identity rose
  `.0566` from initialization to 12k. Its absolute identity remained low and
  the full panel showed colored face strips, seams, occlusion failures, and
  duplicated glasses/features. This makes attention normalization, output
  basis, spatial alignment, and semantic placement higher priorities than
  another native/reference mixer or a global rank increase.

## Audited baseline substrate

The shared parent config is `large_dataset_rhca_hard_v1_audited_20k.yaml`.
It reconstructs the *intended* `r4` experiment, not its swallowed installation
failure:

- `ba_architecture_version=hard_replace_v1`;
- all 70 self-attention sites, rank-32 `noise_and_ref` Q/K/V LoRA;
- hard reference-face replacement, with the ordinary U-Net residual retained;
- branched SA enabled and branched CA disabled;
- `pose_adapt_ratio=0.0` and `ca_mixing_for_face=false` in training and
  validation;
- `train_ba_only=true`, fail-closed installation, exact trainable allowlist,
  and schema-v2 complete trainable checkpointing;
- explicit, strict `legacy_full_copy` validation to preserve the historical
  validation family;
- historical masked face epsilon-MSE, LR `1e-4`, warmup, batch size 2, seeds,
  prompts, reference images, bboxes, RealVisXL validation base, DDIM scheduler,
  50 inference steps, CFG, and metric definitions;
- 2,000-step epochs, 10 epochs, checkpoint/validation every epoch;
- Comet project `aug-large-ds`.

The ownership/checkpoint repairs are correctness requirements common to every
arm, not experimental elements. Because they remove unintended broad trainable
state, absolute new-run metrics must be compared primarily within this new
suite and by within-run change. Historical `r4` remains a contextual target,
not a perfectly matched clean control.

Fixed E0 inherits this parent directly and changes only the required Comet
comment. Historical E0 inherits an observed-r4 parent that deliberately
preserves the old ownership/checkpoint bug but adds an exact assertion that it
is reproduced. E1-E6 inherit the clean parent and each change one declared
scientific leaf.

## Six experiments in priority order

| Priority | Run | The only scientific change | Why it is high priority | Binary decision at or before 20k |
|---:|---|---|---|---|
| 1 | `E1_large_ds_truekey_r32_20k_full96_r1` | Exclude non-face reference tokens from the face-attention softmax with a true boolean key mask. | The baseline multiplies invalid reference features by zero but leaves their keys in the softmax denominator. This makes branch gain depend on face area and dilutes valid identity tokens. | **Yes:** materially better paired ID and no tail-quality loss means true-key normalization becomes the next baseline. **No:** key dilution is not the main plateau cause. |
| 2 | `E2_large_ds_branchout_r32_20k_full96_r1` | Add a rank-32 output projection used only by the hard reference-face branch; it is initialized to exact base-output parity. | The historical broad run unintentionally trained generic `to_out`, while clean BA froze the shared output basis. This arm tests whether missing output-basis capacity explains that identity gap without allowing generic U-Net drift. | **Yes:** better ID with stable text/background promotes branch-local output capacity. **No:** Q/K/V-to-output basis mismatch is not limiting at rank 32. |
| 3 | `E3_large_ds_roiwarp_r32_20k_full96_r1` | Bbox-normalize and bilinearly map the masked reference ROI into the target-face bbox before reference K/V projection. | Hard v4's strips, duplicated accessories, and occlusion failures are consistent with reference features arriving in their source spatial frame. This is a direct alignment test while retaining reference-only K/V. | **Yes:** fewer fixed hard-case alignment failures plus higher paired ID supports bbox-frame alignment. **No:** explicit ROI warping damages pose or does not solve misregistration. |
| 4 | `E4_large_ds_midup_r32_20k_full96_r1` | Patch only `mid_block`, `up_blocks.0`, and `up_blocks.1`; all other baseline settings and rank remain unchanged. | The baseline forces the same face route through all 70 sites, including down blocks that establish pose/layout. This isolates whether early hard BA causes the later face/body integration failures. | **Yes:** comparable/higher ID with better p10/hard cases promotes semantic site selection. **No:** the removed down/high-resolution sites carry necessary identity capacity. |
| 5 | `E5_large_ds_infersteps_r32_20k_full96_r1` | Sample per-example training timesteps only from the fixed DDIM-50 timesteps at which BA is active during validation. | The baseline trains on one uniformly sampled scalar timestep for the whole batch, including noise levels never used by the inference BA route. | **Yes:** earlier recovery or a higher 12–20k ID plateau supports inference-aligned optimization. **No:** timestep-support mismatch is not the dominant ceiling. |
| 6 | `E6_large_ds_fp32_r32_20k_full96_r1` | Store only the rank-32 BA trainables and their Adam moments in FP32; the frozen U-Net remains BF16. | Clean BA32's BF16 weights and moments were numerically active but strongly top-heavy. The later FP32 architectures changed too many other elements to isolate precision. | **Yes:** better late ID or less oscillation promotes FP32 BA state. **No:** numerical precision is not the important capacity bottleneck. |

None of the arms changes rank globally, activates cross-attention, adds an
identity projector, changes reference selection, or introduces a shuffled-rank
loss. Those would obscure the answer to the higher-priority mechanisms above.
The exact run strings in the table are also the Comet display names and Serv
package identifiers; each carries both its experiment number and key change.

## Exact files for each run

The builder-standard `run_` and `start_` filename prefixes identify the MLS
file and entrypoint; the run/package/Comet identifier itself starts with the
requested `E<n>_large_ds_` prefix. Each row is self-contained and maps one
Hydra config to one immutable experiment record and one one-GPU Serv package.

| Experiment | Hydra config | Experiment JSON | Generated Serv entrypoint | MLS YAML |
|---|---|---|---|---|
| E0 historical observed | `src/configs/E0_large_ds_base_historical_20k.yaml` | `experiments/large_dataset/E0_large_ds_base_historical_r4_20k_full96_r1.json` | `serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/start_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.sh` | `serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/run_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.yaml` |
| E0 fixed BA-only | `src/configs/E0_large_ds_base_fixed_20k.yaml` | `experiments/large_dataset/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1.json` | `serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/start_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/run_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.yaml` |
| E1 true-key mask | `src/configs/E1_large_ds_truekey_20k.yaml` | `experiments/large_dataset/E1_large_ds_truekey_r32_20k_full96_r1.json` | `serv_run_packages/E1_large_ds_truekey_r32_20k_full96_r1/start_E1_large_ds_truekey_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E1_large_ds_truekey_r32_20k_full96_r1/run_E1_large_ds_truekey_r32_20k_full96_r1_1gpu.yaml` |
| E2 branch-only output | `src/configs/E2_large_ds_branchout_20k.yaml` | `experiments/large_dataset/E2_large_ds_branchout_r32_20k_full96_r1.json` | `serv_run_packages/E2_large_ds_branchout_r32_20k_full96_r1/start_E2_large_ds_branchout_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E2_large_ds_branchout_r32_20k_full96_r1/run_E2_large_ds_branchout_r32_20k_full96_r1_1gpu.yaml` |
| E3 reference ROI warp | `src/configs/E3_large_ds_roiwarp_20k.yaml` | `experiments/large_dataset/E3_large_ds_roiwarp_r32_20k_full96_r1.json` | `serv_run_packages/E3_large_ds_roiwarp_r32_20k_full96_r1/start_E3_large_ds_roiwarp_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E3_large_ds_roiwarp_r32_20k_full96_r1/run_E3_large_ds_roiwarp_r32_20k_full96_r1_1gpu.yaml` |
| E4 mid/up sites | `src/configs/E4_large_ds_midup_20k.yaml` | `experiments/large_dataset/E4_large_ds_midup_r32_20k_full96_r1.json` | `serv_run_packages/E4_large_ds_midup_r32_20k_full96_r1/start_E4_large_ds_midup_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E4_large_ds_midup_r32_20k_full96_r1/run_E4_large_ds_midup_r32_20k_full96_r1_1gpu.yaml` |
| E5 inference-active timesteps | `src/configs/E5_large_ds_infersteps_20k.yaml` | `experiments/large_dataset/E5_large_ds_infersteps_r32_20k_full96_r1.json` | `serv_run_packages/E5_large_ds_infersteps_r32_20k_full96_r1/start_E5_large_ds_infersteps_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E5_large_ds_infersteps_r32_20k_full96_r1/run_E5_large_ds_infersteps_r32_20k_full96_r1_1gpu.yaml` |
| E6 FP32 BA state | `src/configs/E6_large_ds_fp32_20k.yaml` | `experiments/large_dataset/E6_large_ds_fp32_r32_20k_full96_r1.json` | `serv_run_packages/E6_large_ds_fp32_r32_20k_full96_r1/start_E6_large_ds_fp32_r32_20k_full96_r1_1gpu.sh` | `serv_run_packages/E6_large_ds_fp32_r32_20k_full96_r1/run_E6_large_ds_fp32_r32_20k_full96_r1_1gpu.yaml` |

All packages use the shared fail-closed dispatcher
`launchers/active/run_E_large_ds_hard_v1_20k_1gpu.sh`, generated from
`serv_run_packages/_sources/start_E_large_ds_hard_v1_20k_1gpu.sh`. The audited
parent is `src/configs/large_dataset_rhca_hard_v1_audited_20k.yaml`; E0 is its
explicit fixed runnable comparator. The historical E0 parent is
`src/configs/large_dataset_rhca_historical_observed_20k.yaml`.

## Fixed run contract

Fixed E0 and every E1-E6 arm must resolve to all of the following before
submission:

```text
machine                         Serv
GPU request                     1 × A100
optimizer steps                 20,000
epoch length / epochs           2,000 / 10
train batch size                2
validation                      step 0 + every 2,000
validation images               fixed manual_val 96, one image each
checkpoint interval             2,000
branched self-attention         enabled
branched cross-attention        disabled
pose_adapt_ratio                0.0
ca_mixing_for_face              false
face route                      hard target-Q/reference-KV replacement
PhotoMaker/native face mixer    none
Comet project                   aug-large-ds
```

Historical E0 keeps the same run contract except for its explicit broad
ownership and legacy checkpoint format. It has its own exact 3,080-tensor /
171,294,720-parameter startup partition and must not be resumed after an
interruption.

The Large Dataset manifest/image paths remain Serv machine-local environment
inputs and are preflighted with 64 decoded distinct same-ID target/reference
pairs before Comet registration. No credential or machine-local secret is
stored in a config, experiment JSON, or Serv package.

## Decision gates

- **Clean startup:** exact processor list, exact trainable names/count/dtype, zero
  trainable CA/default/generic adapter tensors, optimizer membership, schema-v2
  manifest, ONNX CUDA, PyIQA 0.1.15, and immutable Comet record must pass.
- **Historical-E0 startup:** exact optimizer membership and the complete
  3,080/171,294,720 partition must pass: BA 840/31,948,800, generic rank-32
  adapter 1,120/46,448,640, and PhotoMaker-default rank-64 adapter
  1,120/92,897,280.
- **Step 0:** 96 images, unchanged prompts/seeds/references/bboxes and complete
  aggregate/per-image ID logging. Arms whose change is zero-initialized must
  match the audited baseline pixels; true-key masking and ROI warping are
  expected to alter step zero because they change routing immediately.
- **4k:** stop only for a dead route, non-finite training, catastrophic face
  coverage, or clear widespread corruption. Historical hard routes commonly
  dip at 2k.
- **8k/12k:** compare ID slope, text, TOPIQ-Face p10/coverage, and the fixed
  angry/kickboxing/night-ride/skiing hard cases. An arm dominated on all of
  those dimensions may be stopped while retaining its complete checkpoint.
- **20k:** a practical promotion requires at least `.01` paired mean ID gain
  over the most appropriate clean comparator, a paired bootstrap interval that
  excludes zero, no material text or TOPIQ-Face p10/coverage regression, and a
  visible reduction rather than relocation of hard-route artifacts.

The `.01` threshold is an auxiliary screening rule, not a change to the
canonical metric definition.

## Per-image ID similarity in Comet

Each validation event logs the existing aggregate `manual_val/id_sim` scalar
unchanged and also one Comet table named:

```text
id_sim__manual_val__step_<six-digit-step>.csv
```

The table has exactly 96 rows and includes validation step, image index,
output key, identity, prompt, seed, generated-image count, and individual
`id_sim`. This is a Comet table asset in the Tables section, separate from the
scalar curves. A local copy is retained below the run's saved directory.

Retrieve one table by immutable run ID:

```python
from io import BytesIO
import pandas as pd
from comet_ml import APIExperiment

run_id = "<32-character immutable Comet experiment key>"
step = 20000
experiment = APIExperiment(previous_experiment=run_id)
filename = f"id_sim__manual_val__step_{step:06d}.csv"
payload = experiment.get_asset_by_name(filename, return_type="binary")
if payload is None:
    raise RuntimeError(f"missing Comet table: {filename}")
table = pd.read_csv(BytesIO(payload))
print(table[["image_index", "output_key", "id_sim"]])
```

List all ID tables for the run with
`experiment.get_asset_list()` and filter `fileName` by the `id_sim__` prefix.

## Per-run Comet comment

Every child config supplies one short `writer.experiment_comment` stating the
single delta, its baseline, and its decision question. `CometMLWriter` records
it as the experiment-level Other value `experiment_comment` at registration.
Retrieve it by immutable run ID:

```python
from comet_ml import APIExperiment

experiment = APIExperiment(previous_experiment="<run-id>")
values = experiment.get_others_summary("experiment_comment")
comment = values[-1] if values else None
print(comment)
```

## Serv submission plan

Implementation creates one non-secret experiment JSON and one generated
one-GPU MLS package per run. All six change-arm packages were deployed and
submitted only after explicit user approval; E0 remains local and unsubmitted.
The commands below remain the E1-E6 launch audit.

These are the exact direct commands printed inside the six YAML files:

```bash
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E1_large_ds_truekey_r32_20k_full96_r1/run_E1_large_ds_truekey_r32_20k_full96_r1_1gpu.yaml
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E2_large_ds_branchout_r32_20k_full96_r1/run_E2_large_ds_branchout_r32_20k_full96_r1_1gpu.yaml
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E3_large_ds_roiwarp_r32_20k_full96_r1/run_E3_large_ds_roiwarp_r32_20k_full96_r1_1gpu.yaml
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E4_large_ds_midup_r32_20k_full96_r1/run_E4_large_ds_midup_r32_20k_full96_r1_1gpu.yaml
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E5_large_ds_infersteps_r32_20k_full96_r1/run_E5_large_ds_infersteps_r32_20k_full96_r1_1gpu.yaml
mls job submit --config /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E6_large_ds_fp32_r32_20k_full96_r1/run_E6_large_ds_fp32_r32_20k_full96_r1_1gpu.yaml
```

The repository-preferred equivalent is `python3 local_scripts/serv_job.py
submit <REMOTE_YAML> --comment "<experiment objective>"`; it invokes the MLS
submission while retaining the local job JSON/comment audit record.

The approved submission followed this sequence:

1. verify Serv's `test` branch/worktree and deploy only to an unused package
   directory;
2. inspect this project's Running and Pending MLS jobs;
3. keep the total request at or below six A100 GPUs;
4. submit each package once with `local_scripts/serv_job.py submit` and the
   same objective as its Comet comment;
5. verify `saved/<run_name>/comet_experiment.json`, copy the immutable key into
   the experiment JSON, and confirm the comment; the step-0 96-row ID table is
   verified after the full validation panel finishes;
6. do not retry an allocation-limit rejection unless the user asks.

After separate approval and only when two project A100 slots are free, submit
the E0 pair through the same audited helper:

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E0_large_ds_base_historical_r4_20k_full96_r1/run_E0_large_ds_base_historical_r4_20k_full96_r1_1gpu.yaml \
  --comment "E0 historical r4 replay: exact observed 171.29M ownership versus fixed E0."

python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/E0_large_ds_base_fixed_baonly_r32_20k_full96_r1/run_E0_large_ds_base_fixed_baonly_r32_20k_full96_r1_1gpu.yaml \
  --comment "E0 fixed r4 replay: strict 31.95M BA-only matched comparator."
```
