# E10 face-position drift versus the fixed full-96 BA mask

**Date:** 4 August 2026  
**Run:** `E10_large_ds_pmdefault_effective_r64_20k_full96_r1`  
**Immutable Comet key:** `0375f172f75c482f840317ec5ae41c05`  
**Scope:** diagnosis only; no code or live-run changes

## Conclusion

This is not a seed change. E10 starts from exactly the same validation image
as E7-E9, but training its effective rank-64 PhotoMaker `default` adapter
changes global attention paths. The generated person/layout can therefore move
even with identical initial noise. Validation continues to use the cached
pre-E10 face boxes, so hard BA writes identity information into the old face
location. The visible foreground face plus secondary/ghost face is the
expected failure signature of this mismatch.

The layout change is learned model behavior and therefore affects validation
and ordinary inference with the E10 weights. The **static-mask misalignment**
is validation-specific in the current setup: Large Dataset training constructs
the target mask from each real training image's own face box, including the
corresponding horizontal-flip adjustment, so its target mask remains aligned
to the teacher image.

## Observed evidence

### 1. Seeds and initial validation state did not drift

The saved resolved configs for E7-E10 all contain:

- `trainer.seed=0`;
- `datasets.val.manual_val.seeds=[0]`;
- 50 inference steps and guidance scale 5;
- the same prompts, references, scheduler, panel, and bbox files.

More strongly, SHA-256 comparison of the 96 saved step-0 PNGs found:

| Comparison with E7 step 0 | Pixel files equal |
|---|---:|
| E8 | 96/96 |
| E9 | 96/96 |
| E10 | 96/96 |

Thus E10 receives the same noise and produces the same initial outputs. Its
later geometric change must follow weight updates, not a changed validation
seed or item ordering.

### 2. E10's one controlled delta is global, not face-local

`src/configs/E10_large_ds_pmdefault_effective_20k.yaml` enables only:

```text
model.photomaker_default_train_scope=effective_all
```

The real-model ownership gate verified 700 trainable PhotoMaker-default LoRA
tensors / 60,948,480 parameters in addition to the normal 840 hard-BA tensors.
The allowlist in `lora2_helpers.py` defines `effective_all` as:

- Q/K/V/output at every ordinary cross-attention (`attn2`) site; and
- the shared output projection at every self-attention (`attn1`) site.

Those adapter outputs are evaluated for all spatial tokens. The name
"PhotoMaker default" does not make them face-local. They can change subject
count, body position, scale, pose, and background layout before the hard BA
face replacement is applied.

E10 is especially aggressive because it updates the pretrained, already
nonzero rank-64 PhotoMaker adapter at the same `1e-4` LR. E7-E9 instead train
new rank-32 generic adapters from their zero-output initialization and/or a
narrower scope.

### 3. Validation deliberately uses cached boxes

The resolved E10 validation config contains:

```text
automatic_bboxes=true
automatic_bboxes_every_val=false
bbox_mask_gen=../dataset_full/val_dataset/pm96_bboxes_new.json
```

Startup logged:

```text
[AutoBboxGen] using existing: ../dataset_full/val_dataset/pm96_bboxes_new_auto.json (96 entries)
```

Because all 96 cached entries exist and recomputation is disabled, the code
does not perform a fresh PhotoMaker-only pass at 2k or 4k. The masks therefore
remain tied to the original step-0 composition. For the two reported cases,
the cached inner face boxes are:

- `Rushing ma_lex.png`: `[490, 197, 751, 528]`;
- `Angry man _eddie.png`: `[447, 179, 668, 486]`.

At E10 step 2k and 4k, the principal person moves left while a second/ghost
face remains near these original central boxes. E9 at step 4k retains the
single-person step-0 layout for the same two items. The screenshots are
therefore consistent with global E10 layout drift plus BA continuing to act in
the cached old location.

One qualification: a BA-disabled E10 checkpoint inference was not run in this
audit. Therefore, saying the BA-disabled "base image" itself moves is a
high-confidence causal inference from the controlled delta and the large
outside-mask changes, rather than a directly saved BA-off image.

## Training versus validation impact

| Question | Answer |
|---|---|
| Does E10 training cause the layout tendency? | **Yes.** Global default-adapter weights are updated during training. |
| Are training BA masks stale? | **No.** `LargeDatasetTrain` supplies the actual target `new_face_crop`; it mirrors the bbox when the target is flipped, and training builds the latent mask from that bbox. |
| Are current validation BA masks stale? | **Yes, once E10 moves the generated subject.** They remain cached from the original composition. |
| Would ordinary E10 inference also move layout? | **Yes.** The global trained adapter is part of the checkpoint behavior. |
| Would dynamic validation boxes restore the original layout? | **No.** They can align BA to the moved face, but they do not undo the global composition change. |

Current E10 fixed-mask ID similarity after layout drift is confounded: the
face scorer may select the moved foreground face or the BA-created secondary
face, while BA is not necessarily conditioning the face being scored.

## Recommended fix and clean diagnostic

Do not change seeds. For the existing E10 checkpoints, run a separately named
validation sidecar with `automatic_bboxes_every_val=true`. For each checkpoint
and sample it should:

1. generate a BA-disabled PhotoMaker image with the same prompt/reference/seed;
2. detect the current principal face;
3. run BA with that freshly detected box;
4. log the box/overlay and results to a separate Comet experiment.

**5 August superseding execution decision:** the user explicitly requested an
in-place correction of the completed E10 run. The implemented sidecar stages
and verifies every replacement first, then replaces steps 2k-20k on immutable
Comet key `0375f172f75c482f840317ec5ae41c05` while preserving step 0 and a local
pre-replacement manifest. See
`analysis/2026-08-05_e10_dynamic_mask_checkpoint_revalidation.md`.

This is the smallest way to answer whether E10 has useful identity behavior
once BA follows its moved face. It must be labelled as a dynamic-mask protocol
and must not overwrite or be compared naively with canonical fixed-mask
metrics. A BA-off image should also be retained to directly measure how much
layout drift comes from the trained default adapter before BA.

For a future training arm that must preserve the original layout, dynamic
validation alone is insufficient. The training change should instead make the
default-adapter update face-local or penalize deviation from frozen-E0
predictions outside the target face. Merely lowering the seed or regenerating
the fixed boxes does not address the cause; lowering rank/LR is only an
ablation, not a guarantee of layout preservation.

## Audited source locations

- `src/configs/E10_large_ds_pmdefault_effective_20k.yaml`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/trainer/sdxl_trainers.py`
- `src/datasets/large_dataset.py`
- `src/model/photomaker_branched/lora2_helpers.py::prepare_branched_training_inputs`
- Serv saved configs and validation PNGs under the isolated E7-E10 runtime
