# CL14 parity removes the CL14_CA startup failures

**Date:** 12 August 2026  
**Scope:** CL14_CA startup diagnosis, corrective implementation, and Serv relaunch  
**Evidence cutoff:** 12 August 2026 21:52 Europe/London

## Executive conclusion

The original CL14 training path was not broken. The first CL14_CA attempt that reached training failed because new residual-cross-attention telemetry was returned under `ba_telemetry`, while the trainer tried to log the same values as already-flattened top-level loss outputs. [code][measured] The minimal scientific fix is to promote this model telemetry immediately after `batch.update(output)`; CL14's masked loss remains unchanged. [code]

The later InsightFace failures came from deployment drift away from CL14, not from the CL14_CA architecture. [code][measured] The final package restores CL14's unchanged default `FaceAnalysis2(...)` constructors and copies the sealed five-file `buffalo_l` cache into the worker's default `~/.insightface/models/buffalo_l` directory before startup. It also retains CL14's validation/offload/reinstall lifecycle. [code][measured]

The scientific run is `CL14_CA_r7`; it uses the canonical fixed 96-image step-zero panel and remains active unless an error occurs. The operational smoke uses exactly one existing CL14 validation batch (12 items), then proves forward, backward, telemetry logging, and optimizer progress before being stopped. The smoke is not scientifically comparable. [code]

## Run ledger

| Run | MLS job | Immutable Comet key | Evidence and disposition |
|---|---|---|---|
| CL14 control | historical | `6fe0028be92242c38056b3d36665fdd6` | Immutable 24k comparison baseline. [report] |
| CL14_CA corrected v2 | `lm-mpi-job-cf7eda84-ad0b-4d50-af17-c3d9f19e5315` | `0cfe3c874d75448789acc0a5c9b4bc63` | Completed 96-image step-zero validation, then failed on the first training batch with missing flattened telemetry. [measured] |
| CL14_CA normal | `lm-mpi-job-244ef7b2-3943-4998-a82e-ae1be2208169` | `4d96dc8e776b4039b1116acc5cdcf706` | Completed canonical 96-image step-zero validation and advanced beyond optimizer step 224; left running. [measured] |
| one-batch smoke r1 | `lm-mpi-job-968d8fa9-86cc-476a-8701-3b3ce458cedd` | `ad3c24addbca4cf2acfcc123c25faace` | Generated all 12 images, then correctly failed because inherited face-quality expected 96. [measured] |
| one-batch smoke r2 | `lm-mpi-job-05738b32-5978-4559-96cb-6ac7ea38cd2d` | `f808676f2ad54e5e928d92b6650053ca` | Completed 12/12 validation and optimizer step 7; stopped after successful proof. [measured] |

Earlier attempts that failed before model training are retained in the experiment/job audit. They are neither scientific runs nor evidence against residual CA: r3/r1 omitted the sibling dataset mount; r4/r2 and r5/r3 used an unnecessary custom InsightFace path; r6/r4 encountered a partially populated default cache; the skip-validation smoke bypassed CL14's normal model lifecycle and segfaulted; and a one-item attempt violated CL14's validation batch-size assertion. [measured][code]

## Root cause and minimal code fix

The model output has this shape:

```python
{
    "model_output": ...,
    "ba_telemetry": {
        "ba/identity_ca_token_count/up0": ...,
        # other route-usage values
    },
}
```

CL14 uses `MaskedDiffusionLoss`, which computes the training objective but does not flatten arbitrary model telemetry. Therefore the CL14_CA trainer's configured `writer.loss_names` could not find `ba/identity_ca_token_count/up0` after the first forward/backward pass. [code]

The exact correction in `src/trainer/sdxl_trainers.py` is:

```python
output = self.model(**batch, do_cfg=do_cfg)
batch.update(output)

# 12 Aug 2026 - AICODE-NOTE: BA telemetry is model output, not loss
# output. Promote it independently so CL14's unchanged masked loss can
# log the residual-CA route without changing the scientific objective.
ba_telemetry = output.get("ba_telemetry")
if ba_telemetry:
    batch.update(ba_telemetry)
```

This changes logging data flow only. It does not modify the loss, optimizer, model forward values, gradients, dataset, validation, seeds, scheduler, or step budget. [code]

## CL14 parity in the Serv package

The runtime wrapper keeps the original CL14 analyzer calls unchanged. Before Python starts it repairs any partial worker-local default cache from a sealed NFS copy:

```bash
INSIGHTFACE_SOURCE="${OWNER_ROOT}/metric_cache/insightface/models/buffalo_l"
INSIGHTFACE_DEFAULT="${HOME}/.insightface/models/buffalo_l"
mkdir -p "${INSIGHTFACE_DEFAULT}"
cp -a "${INSIGHTFACE_SOURCE}/." "${INSIGHTFACE_DEFAULT}/"
```

It then checks a deterministic aggregate SHA-256 over the five expected files. A local simulation deliberately corrupted one default-cache ONNX file, ran this repair, disabled network access, and successfully initialized both InsightFace detection and recognition. [measured]

The final CL14_CA scientific contract remains:

- fresh CL14 initialization, not a resume from CL14's 24k checkpoint;
- residual identity CA v3 only in `up_blocks.0/1`, rank 64, gate initialized at 0.02 and bounded by 0.20;
- target Q attends active PhotoMaker identity-token K/V through a zero-initialized output delta while native cross-attention remains intact;
- legacy branched CA disabled, `pose_adapt_ratio=0`, and `ca_mixing_for_face=false`;
- exact ownership: 2,348 trainable tensors / 224,624,676 parameters, all present in the optimizer;
- CL14 batch size 2, 24k optimizer steps, and fixed 96-image validation at step 0 and every 2k steps. [code]

## Operational one-batch smoke

The smoke inherits `CL14_CA.yaml` and changes only:

```yaml
datasets:
  val:
    manual_val:
      limit: 12

trainer:
  face_quality:
    expected_images: 12
```

The inherited `manual_val.batch_size` is 12, so this is exactly one normal CL14 validation batch. The matching face-quality count is necessary to allow the validation finalizer to return to the training lifecycle. This run cannot be compared with the fixed-96 scientific panel. [code]

## Verification gates

| Gate | Result | Confidence |
|---|---|---|
| Hydra composition and single-delta diff against CL14 | Pass; only residual-CA, telemetry, and explicitly declared smoke fields differ. [measured] | High |
| Shell syntax and Python compile | Pass. [measured] | High |
| Sealed default-cache repair with network disabled | Detection and recognition initialized. [measured] | High |
| Trainable/optimizer ownership | Exact 2,348 / 224,624,676 match. [measured] | High |
| Normal fixed-96 startup | Final state recorded below. [measured] | High after training-step proof; medium while only validation is observed |
| Smoke forward/backward/optimizer telemetry | Final state recorded below. [measured] | High after at least two finite steps |
| Scientific benefit over CL14 | Not yet measured; compare subject-v2 ID and face quality at matched checkpoints, primarily 24k. [hypothesis] | Low until matched validation completes |

## Reproduction and audit commands

Run from `diffusion_template/` in `photomaker` locally or `photomaker_NS` on Serv:

```bash
python tools/validate_CL14_CA_config.py \
  --config-name CL14_CA \
  --run-name CL14_CA_r7 \
  --experiment-spec experiments/cosmic_large/CL14_CA_r7.json

python tools/validate_CL14_CA_config.py \
  --config-name CL14_CA_onebatch_smoke \
  --run-name CL14_CA_onebatch_smoke_r2 \
  --experiment-spec experiments/cosmic_large/CL14_CA_onebatch_smoke_r2.json

bash -n launchers/active/run_CL14_CA_24k_1gpu.sh \
  serv_run_packages/CL14_CA_relaunch_common/start_CL14_CA_variant_1gpu.sh
```

Retrieve future scientific results only by immutable Comet key, not display name. The primary decision is paired subject-v2 ID similarity on the fixed 96 outputs at matched step 24k; also examine prompt similarity, artifacts, face quality, and face/body alignment. [report]

## Final live state

`CL14_CA_r7` completed all eight fixed-validation batches (96 images), wrote the 96-row per-image ID table, staged 96 face-quality inputs, restored the CL14 training base, and advanced beyond optimizer step 224. Logged losses remained finite; the first was `0.064705`. An immutable-key Comet query at training step 0 found two active identity tokens in both installed groups, gate `0.0200000014`, native-face RMS `0.2548573` in up0 and `0.0834144` in up1, and zero residual-face RMS as expected from the zero-initialized output projection. The telemetry is therefore present and finite at the exact code path that previously raised `KeyError`. The job remains Running as requested. [measured]

`CL14_CA_onebatch_smoke_r2` completed its one validation batch, wrote the 12-row ID table, staged 12 face-quality inputs, restored the CL14 training model, and advanced through optimizer step 7. Its first logged loss was the same finite `0.064705`. An immutable-key Comet query independently returned the same step-zero route telemetry as the normal run: two identity tokens per group, gate `0.0200000014`, finite native-face RMS, and expected zero-initialized residual-face RMS. MLS stop succeeded and the final job status is Stopped. [measured]

## Evidence sources

- `src/trainer/sdxl_trainers.py` and `src/model/photomaker_branched/residual_identity_ca_processor_v3.py`. [code]
- `src/configs/CL14_CA.yaml` and `src/configs/CL14_CA_onebatch_smoke.yaml`. [code]
- Sealed Serv stdout/stderr and local `local_scripts/serv_job_records/` records for the MLS IDs above. [measured]
- `analysis/2026-08-12_CL14_CA_implementation_plan.md` and `analysis/2026-08-12_branched_cross_attention_disable_history_and_cl19_reintroduction.md`. [report]
