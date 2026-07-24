# 23Jul NN3a training lab

Everything created by this lab stays inside this directory.

Prepare the fixed selected-ID split:

```bash
/home/niko/miniconda3/envs/photomaker_NS/bin/python prepare_selected_id.py
```

Inspect a fully resolved run without starting it:

```bash
./run_architecture.sh E00_control --dry-run
```

Start an arm:

```bash
./run_architecture.sh E00_control
```

The valid distinct-target/reference OneIDTrain ablation is rerunnable with:

```bash
./run_architecture.sh E18_oneid8_distinct_projection_split
./run_architecture.sh E19_oneid8_distinct_projection_split_blended20
```

Both use the immutable eight-image `nm0005092` subset under
`data/one_id_nm0005092/`; the validation reference `51.jpg` is excluded from
training, and every training reference is guaranteed to use a different
filename from its target. E15/E16 used a pixel-identical target as reference
and are retained only as leakage audits; they must not be promoted.

If a shared-GPU arm is interrupted after a 200-step boundary, resume the
latest full checkpoint in place (same run directory, optimizer/scheduler
state, and Comet experiment):

```bash
./run_architecture.sh E01_active_up \
  --resume-run-dir experiments/<existing-run-folder>
```

When the production job is sharing GPU 0, queue the arm immediately after its
next high-memory validation:

```bash
nohup ./schedule_after_production_validation.sh E00_control \
  > scheduler/E00_control.log 2>&1 &
```

Chain another arm immediately after a local launcher exits; the helper launches
at once if production is training, or falls back to the validation-aware
scheduler:

```bash
./chain_after_local_run.sh <local-launcher-pid> E04_projection_split
```

Architecture toggles and their exact recipes live in
`architecture_registry.json`. Every invocation creates an immutable folder
under `experiments/` containing its manifest, resolved command, log,
checkpoints, validation images, metrics, and report.

After an arm completes, validate the requested checkpoints:

```bash
./run_validation_suite.sh experiments/<run-folder>
```

Validation renders with the console writer, resolves the training arm's real
Comet experiment key, verifies it, and uploads the finished files directly to
that experiment. All steps and modes therefore appear in the same Comet run
as training, with collision-proof names such as
`canonical50__step0200__...`. The GPU validation process never initializes a
Comet experiment.

The first BA validation performs the PhotoMaker bbox-control pass and freezes
its four detected generation boxes in `validation/fixed_gen_bboxes.json`.
Every later checkpoint/mode reuses those exact boxes, so checkpoint comparisons
do not pay for or vary an extra PhotoMaker pass.

Runs rendered before unified logging was enabled can consolidate their local
images into the training experiment without regenerating them:

```bash
/home/niko/miniconda3/envs/photomaker_NS/bin/python \
  migrate_validation_to_comet.py experiments/<run-folder>
```

To calibrate memory with the cheapest stream first:

```bash
./run_validation_suite.sh experiments/<run-folder> \
  --steps 0 --modes pmControl50
```

After all validation streams finish:

```bash
/home/niko/miniconda3/envs/photomaker_NS/bin/python \
  summarize_run.py experiments/<run-folder>
```

Upload the resulting checkpoint metrics and PDF into the same verified Comet
training experiment:

```bash
/home/niko/miniconda3/envs/photomaker_NS/bin/python \
  upload_report_to_comet.py experiments/<run-folder>
```

Build a CPU-only PDF over every fully validated run. Different identities and
dataset loaders are placed in separate sections, never in the same grid:

```bash
CUDA_VISIBLE_DEVICES='' /home/niko/miniconda3/envs/photomaker_NS/bin/python \
  build_consolidated_report.py --dataset-profile all \
  --output visual_reports/<name>.pdf
```

Processor drift can be inspected as soon as any checkpoint exists:

```bash
/home/niko/miniconda3/envs/photomaker_NS/bin/python \
  checkpoint_diagnostics.py experiments/<run-folder>
```
