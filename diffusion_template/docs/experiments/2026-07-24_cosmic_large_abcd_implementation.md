# CosmicLarge A–D implementation and deployment plan

**Date:** 24 July 2026
**Branch:** `test`

## Isolation decision

The historical April replay model, trainer, pipeline, and launchers remain
unchanged. Copying those implementations into a second model tree would make
training/inference fixes drift and would weaken checkpoint comparability.

The experiments are isolated at their narrowest stable interfaces:

- Task A and Task C are thin launchers with explicit Hydra overrides.
- Task B is a standalone evaluation-only entry point that reuses the existing
  model, pipeline, validation dataset, and metric constructors.
- Task D has a separate manifest-backed dataset, config, and launcher.
- Existing behavior is restored by selecting an existing replay launcher.

## Task A: face-only, branched CA off

Launcher:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh
```

It wraps the existing face-only launcher and changes only:

```text
disable_branched_ca=true
train_branched_ca_lora=false
model.train_branched_ca_lora=false
trainer.n_epochs=8
```

`branched_attn_weight_mode=noise_and_ref` remains unchanged. At startup, the
trainable summary must contain self-attention `ref_to_*`/`noise_to_*` LoRA
parameters and zero `attn2.processor` trainables.

## Task B: evaluation-only checkpoint diagnostics

Tool:

```text
tools/inference/evaluate_rhca_checkpoint.py
```

The tool accepts full or weights-only RHCA checkpoints and writes:

```text
resolved_config.yaml
command_manifest.json
run_manifest.json
per_image.json
images/*.png
```

`validation_native` installs processors from the validation base before loading
saved trainable deltas. It snapshots every processor `base_weight` buffer
before/after delta loading and fails if any changes.

`legacy_full_copy` deliberately initializes/loads the training-base model,
copies every processor state to CPU, then applies those full states to the
validation model. The manifest labels and audits this historical hybrid-base
path.

Example reproduction row:

```bash
python tools/inference/evaluate_rhca_checkpoint.py \
  --config saved/rhca_apr2026_cosmic_large_one_id_faceonly_8k/config.yaml \
  --checkpoint saved/rhca_apr2026_cosmic_large_one_id_faceonly_8k/checkpoint-epoch8.pth \
  --output-dir diagnostics/cosmic_faceonly_4k/row01_legacy_cfg5_matched \
  --validation-dataset cosmic_large_one_id_val \
  --guidance-scale 5 \
  --processor-base-mode legacy_full_copy \
  --reference-condition matched \
  --limit 12
```

After row 1 reproduces, run this matrix for each CA-on endpoint:

| Row | CFG | CA | Processor base | Reference |
|---:|---:|---|---|---|
| 1 | 5 | on | `legacy_full_copy` | `matched` |
| 2 | 5 | on | `validation_native` | `matched` |
| 3 | 1 | on | `validation_native` | `matched` |
| 4 | 5 | off | `validation_native` | `matched` |
| 5 | 5 | off | `validation_native` | `wrong` |
| 6 | 5 | off | `validation_native` | `null` |

Rows 4–6 pass `--disable-branched-ca`; row 3 passes
`--guidance-scale 1`. Run the Cosmic face-only 4k and leak-free one-ID 4k
endpoints first, then the Task A endpoint when available.

For a CA-off checkpoint, pass `--disable-branched-ca`. The tool rejects an
attempt to enable CA when the checkpoint has self-attention processor state but
no saved CA processor state.

Wrong-reference evaluation requires a real wrong-identity image and bbox when
the fixed dataset has only one reference:

```text
--reference-condition wrong
--wrong-reference /path/to/wrong_identity.jpg
--wrong-reference-bbox X0 Y0 X1 Y1
```

The null intervention uses a zero PhotoMaker identity embedding, neutral
spatial reference, and zero reference face mask. Both wrong and null are
labelled end-to-end conditioning interventions.

## Task C: CA off, reference-only projections

Launcher:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k_1gpu.sh
```

It wraps Task A and changes only:

```text
branched_attn_weight_mode=ref_only
model.branched_attn_weight_mode=ref_only
```

Run Task C only if Task A reduces exterior/global corruption but leaves
pasted, duplicated, or misregistered facial fragments. Its startup summary
must contain `attn1.processor.ref_to_*` parameters and no `noise_to_*` or
`attn2.processor` trainables.

## Task D: controlled identity factorial

Files:

```text
src/datasets/controlled_identity_factorial.py
tools/datasets/build_controlled_identity_factorial.py
src/configs/controlled_identity_factorial_rhca.yaml
launchers/active/run_rhca_controlled_identity_factorial_4k_1gpu.sh
```

The builder requires an explicit split and records the source metadata/image
roots and hashes, selected IDs, bboxes, prompts, source/artifact hashes,
selection seed, duplicate rejects, face-embedding audit, deterministic crop
parameters, derived-reference hashes/cache keys, and holdouts.

The 256 reference transform is fixed:

```text
20% margin per side around a square face box
boundary-clamped deterministic square crop
PIL bicubic resize to 256×256
JPEG quality 95, subsampling 0
```

The runtime dataset checks the manifest schema, hashes, target/reference
inequality, image dimensions, and both target/reference bbox bounds. The same
eight source image IDs back all arms.

Candidate `nm0004960` was inspected locally:

- 19 metadata/image records;
- `52.jpg` rejected as a perceptual duplicate of `43.jpg`;
- 18 images passed the face-embedding threshold;
- InsightFace cosine to the medoid: minimum `0.454`, median `0.751`;
- training IDs: `0,19,20,27,31,49,56,59`;
- recurring validation ID: `69`;
- final untouched holdout ID: `92`;
- single-target arm target ID: `0`.

The first build is deliberately unsealed because generated validation bboxes
do not exist yet. On a GPU machine:

```bash
python tools/datasets/generate_cosmic_large_one_id_photomaker.py \
  --dataset-dir /path/to/nm0004960_v1_preflight \
  --photomaker-dir /home/niko/models/PhotoMaker-V2
```

Inspect all 12 generated images and bbox overlays. Then rebuild the immutable
artifact with:

```text
--generation-bboxes /path/to/inspected/photomaker_generated_bboxes.json
```

The final manifest must report `generation_bboxes_status: sealed`.
The builder seals both the index-keyed
`photomaker_generated_bboxes.json` used by the validation dataset and the
filename-keyed `photomaker_generated_bboxes_auto.json` cache required by the
historical trainer. The factorial config loads this cache but never recomputes
it.

Run an arm with:

```bash
FACTORIAL_ARM=multi_full \
CONTROLLED_FACTORIAL_ROOT=/path/to/sealed/artifact \
bash launchers/active/run_rhca_controlled_identity_factorial_4k_1gpu.sh
```

Valid arm names are `multi_full`, `single_full`, and `multi_cosref`.

## Deployment order

### Neb

Neb is the preferred machine for these 4k decision runs. Use one job at a time:
validation can consume about 72 GB even when training usage is lower.

1. Fast-forward the clean remote `test` checkout and preserve existing
   untracked bbox files.
2. Run Task A from fresh initialization.
3. Inspect steps 0, 500, and 1,000 against the CA-on face-only run. Stop early
   only under the handoff's catastrophic/no-improvement rule; otherwise finish
   epoch 8 / step 4,000.
4. When the GPU is free, run Task B row 1 first and require reproduction before
   the remaining checkpoint matrix.
5. Schedule Task C only if Task A meets its trigger.
6. Generate/inspect/seal the Task D validation package once, transfer only the
   small immutable artifact, then run `multi_full` and `multi_cosref`.
   `single_full` is the third arm.

Use `photomaker_NS`, the tested `ENV_FILE=/dev/null`/`PM_PATH` ordering from
`LOCAL_NEB_SERVER_OPERATIONS.md`, unique run/log directories, and `setsid`.

### serv

The serv context check succeeds for:

```text
environment: /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS
checkout:    /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test
branch:      test
```

Do not use the queue for the first 4k causal arm unless Neb becomes
unavailable. Reserve serv for promoted multi-identity confirmation or runs
extended beyond 4k. Build an explicit launcher/YAML package with
`local_scripts/serv_run_builder/create_serv_run.py`, deploy it, and submit via
`local_scripts/serv_job.py submit ... --comment ...`. Never package `.env`.

Task C remains conditional and no >4k extension should be queued before the
4k anatomy comparison is reviewed.
