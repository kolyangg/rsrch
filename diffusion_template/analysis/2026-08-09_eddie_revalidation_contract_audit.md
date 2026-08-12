---
title: "Eddie revalidation contract audit"
subtitle: "Why the first corrected sidecar was not comparable, what was fixed, and how exact replay verified the repair"
date: "9 August 2026"
status: "FINAL: audit fixed; guarded Serv replay completed 36/36 exact"
---

# Executive conclusion

The concern is valid. The existing “historical versus corrected” Eddie panels
do **not** isolate the Eddie subject-selection change. RealVis was loaded
correctly, and the recorded checkpoint, prompts, seeds, references, boxes,
DDIM scheduler, 50 inference steps and CFG 5 were correct. However, the
corrected sidecar diverged from the validation path used during training in
three material ways:

1. it forced `validation_native`, while E13, BC_E13 and CL11 all validated with
   strict `legacy_full_copy` processor semantics;
2. it loaded the checkpoint's trained PhotoMaker `default` adapter, while the
   experiments' shadow-validation contract restores the pretrained default
   adapter after checkpoint loading; and
3. it generated one image at a time, while the in-training `manual_val` loader
   generated the 12 Eddie rows as one batch inside the full 96-image panel.

The second mismatch changes a global conditioning path with 700 trained
tensors and is the most plausible source of wholesale scene differences. The
first mismatch changes the processor bases used by all 70 stateful BA
processors. Either is sufficient to invalidate a causal pre/post comparison.
The previous corrected images, their `34/36` win count, alignment deltas and
cross-model rankings are therefore withdrawn. **[code] [artifact audit]**

There is also an interpretation error: the sidecar did not “change only the
face with branched attention.” It injected a new ArcFace vector into global
PhotoMaker prompt-token conditioning at step 10; spatial BA starts at step 15.
That intervention can legitimately change pose and composition even under an
otherwise exact replay. The BA reference crop and its bbox already selected
foreground Eddie. A face-local BA-only mask correction was not what the old
sidecar performed. **[code]**

No inference was launched during the audit phase itself. A later user-approved
guarded Serv run completed after the fixes; its result is recorded below.

# Visual symptom

The disputed screenshots changed bodies, props and composition, not just face
pixels. They demonstrated that the pair was not comparable, but could not
identify the cause because processor, shadow-adapter and batch contracts also
changed. Those corrected images are retained only as invalidated provenance.

The difference is global quantitatively: outside the immutable generation
mask, `99.96%` of CL11 Kickboxing pixels and `100.00%` of BC_E13 Jumping pixels
change; mean absolute RGB differences are `53.19` and `29.40` on the 0-255
scale. These numbers describe the invalid pair and are not model-quality
metrics. **[measured]**

![](assets/problematic_validation_20260809/eddie_reference_metric_error.png)

*Figure 1. The detector-result-0 ArcFace vector belongs to the small bystander,
whereas the registered BA reference bbox already encloses foreground Eddie.
Correcting the vector changes global PhotoMaker conditioning; changing the BA
reference mask would be a different intervention.*

# Contract comparison

| Validation component | In-training E13 / BC_E13 / CL11 | Old corrected sidecar | Audit result |
|---|---|---|---|
| Validation base | `SG161222/RealVisXL_V4.0` | `SG161222/RealVisXL_V4.0` | matched |
| Scheduler / steps / CFG | RealVis DDIM / 50 / 5.0 | RealVis DDIM / 50 / 5.0 | matched |
| Checkpoints | E13 24k, BC_E13 24k, CL11 20k saved weights | same SHA-256 files | matched |
| Prompts / seeds | fixed manual-val rows, seed 0 | same | matched |
| Reference pixels / reference bbox | Eddie image; foreground BA bbox | same | matched |
| Generation masks | cached fixed full-96 auto boxes | same recorded hash | matched |
| Processor base mode | `legacy_full_copy`, strict | forced `validation_native` | **mismatch** |
| PhotoMaker default adapter | restore pretrained default after loading state | trained checkpoint default left active | **mismatch** |
| Eddie execution batch | 12 rows together | 12 calls of batch 1 | **mismatch** |
| Dataset context | configured 96, Eddie is first batch | config node overwritten to limit 12 | **mismatch in replay context** |
| Changed scientific input | normal detector-result-0 global ID vector | foreground global ID vector | not BA-only |

The three composed configs independently resolve to the same relevant values:
RealVis, `legacy_full_copy`, strict copying, shadow default restore, 96 samples,
batch 12, 50 steps, CFG 5, CA off, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`. **[config composition]**

# What was fixed

The standalone evaluator now:

- derives RealVis, processor mode, CFG, CA state and validation batch size from
  the composed experiment config;
- refuses silent base/processor/CFG/batch/CA overrides unless an explicit
  ablation escape hatch is provided;
- mirrors the training validator's PhotoMaker-default shadow snapshot/restore;
- mirrors strict full processor copying and records copied/restored counts;
- propagates the same complete versioned BA runtime-attribute set as the
  trainer;
- keeps the configured full-96 dataset intact while selecting the first 12
  Eddie rows; and
- records the resolved training-validation contract in `run_manifest.json`.

The Eddie wrapper now labels the corrected vector accurately as a **global
PhotoMaker conditioning intervention**, derives batch 12 rather than forcing
batch 1, and can generate a historical replay arm. The old Serv launcher is
blocked so it cannot accidentally reproduce the invalid protocol. The old
analysis script now rejects manifests missing the corrected contract and the
700-default-tensor / 70-processor restore audit. **[implemented]**

The prepared local chain writes to a new namespace,
`final_checkpoint_sidecar_contract_v2/`; it does not overwrite the invalidated
assets. For each checkpoint it is ordered as:

1. generate an unchanged historical replay under the exact contract;
2. compare all 12 replay images to the downloaded historical validation pixels;
3. stop immediately if any pair or contract field differs; and
4. only after that gate passes, generate the foreground-Eddie global-ID arm.

# Interpretation after a valid replay

The unchanged historical replay is the necessary control. If it does not
reproduce the source images, no corrected image may be compared with them.
If it does reproduce them, a foreground-vector correction may still alter the
base image because PhotoMaker ID tokens are global and start five denoising
steps before BA. That would then be a valid measured effect of the global
selector correction—not evidence of a seed or base-model mismatch.

If the intended experiment is instead “only change spatial BA inside the face
mask,” the PhotoMaker ID vector must remain historical and fixed. For Eddie,
the existing spatial BA reference bbox is already foreground-correct, so there
is no known BA-mask selector defect to repair in that arm. A new face-local
intervention would need a separately specified architecture change and should
not be labeled as the subject-selector correction.

# Current artifact status

- Historical Comet images and historical metrics remain immutable evidence.
- `final_checkpoint_sidecar/` corrected images are invalidated and retained
  only for provenance.
- `final_checkpoint_sidecar_contract_v2/` contains the verified replays,
  corrected images and three exact-pixel gate records.
- The two decision reports were rebuilt from contract-v2 evidence; older PDFs
  remain stale and must not be used.

# Verification performed without inference

- Hydra composition for all three configs confirmed the same RealVis/shadow/
  processor/batch/full-96 contract.
- All three checkpoint files retained their recorded SHA-256 values and each
  contains 2,240 trainable tensors: 840 BA, 700 generic adapter and 700
  PhotoMaker-default tensors.
- Python compilation and shell syntax checks were run on the corrected tools.
- No model generation, GPU validation, Serv submission or Comet mutation was
  performed during the audit phase.

# Subsequent guarded replay result

Serv job `lm-mpi-job-baea4903-7f8d-4785-a67d-f153df3299da` completed in 2,111
seconds. For E13 24k, BC_E13 24k and CL11 20k, the unchanged arm reproduced
all 12 historical Eddie PNGs exactly: 36/36 RGB pixel-identical, no failed
pairs and no contract mismatches. Only then did each corrected arm run.

The valid result reverses the invalid sidecar's layout conclusion. Intended
Eddie improves in 36/36 pairs, while median fixed-mask IoU remains
`0.891/0.875/0.880` and no corrected image is below `0.30`. Kickboxing and
Jumping keep their source body layout. This confirms that the old wholesale
scene changes came from the standalone validation mismatches, not from the
corrected Eddie selector itself. **[measured] [visual]**
