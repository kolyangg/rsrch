# Project purpose

This repository researches whether PhotoMaker identity-conditioned image
generation can be improved with a branched-attention (BA) mechanism. The main
goal is to increase identity similarity while retaining PhotoMaker's pose,
composition, prompt adherence, and image quality. Preserve the core BA idea:
target queries should be able to use identity/reference information through
explicit branched self-attention and cross-attention paths.

Treat experiment comparability as part of correctness. Do not silently change
the validation model, seeds, prompts, reference images, face masks/bounding
boxes, scheduler, inference steps, or metric definitions.

# Repository layout

- `diffusion_template/` is the primary training and validation project.
- `diffusion_template/src/` contains datasets, models, attention processors,
  pipelines, trainers, losses, metrics, and Hydra configuration.
- `diffusion_template/launchers/active/` contains supported experiment
  launchers. `launchers/archive/` is historical evidence, not the default
  place for new runs.
- `diffusion_template/docs/` contains architecture and experiment handoffs.
  Read only the documents relevant to the current task.
- `diffusion_template/tools/` contains Comet export, inference, reporting, and
  dataset utilities.
- `diffusion_template/bbox_utils/` contains face-box utilities used by
  validation.
- `diffusion_template/setup/` contains environment snapshots and setup helpers.
- `dataset_full/` contains training datasets and manual validation metadata.
  Its position beside `diffusion_template/` is significant because some Hydra
  paths use `../dataset_full/...`.
- `_other_models/` contains PhotoMaker, PuLID, PersonaGen, and comparison code.
  Treat these primarily as external/reference implementations unless a task
  explicitly targets them.

Run training and Hydra commands from `diffusion_template/` so relative paths
resolve consistently.

# Environment and credentials

Use an existing Conda environment named `photomaker` or `photomaker_NS`, as
available on the current machine. Prefer `photomaker_NS` when a launcher or
handoff explicitly requires it. Do not create or substantially alter an
environment unless requested.

Machine-local credentials and paths belong in `diffusion_template/.env`.
Active launchers load that file automatically. Never commit `.env`, API keys,
tokens, proxy credentials, or machine-specific secrets. Keep
`diffusion_template/.env.example` free of real credentials.

# Working practices

- Check the current branch, worktree, and dirty status before editing. Preserve
  unrelated user changes.
- Keep code changes concise and localized. Avoid broad cleanup, refactors, or
  formatting changes during an experiment fix.
- Preserve old behavior behind explicit configuration toggles when changing an
  architecture or inference path. Defaults should remain backward-compatible
  unless the task explicitly changes the default.
- Do not replace branched attention with an unrelated conditioning mechanism.
  Make Q/K/V routing, target/reference batch layout, masks, residual merges,
  temporal schedules, and trainable/frozen parameters explicit.
- Keep training and validation code paths aligned. A checkpoint is not
  validated correctly if its trained processors, gates, or routing are absent
  or replaced during inference.
- Do not change experiment names, Comet project IDs, checkpoint semantics,
  dataset paths, batch sizes, or step counts unless requested or documented.
- Do not commit or push unless the user explicitly asks.

For important architecture changes or critical bug fixes, add a concise dated
comment close to the changed logic. Use the actual date and explain the
invariant, for example:

```python
# 24 Jul 2026 - Fixed branched-attention reference K/V routing; target Q stays unchanged.
```

Do not add dated comments to trivial edits. Comments should explain why the
change exists, not merely restate the code.

# Verification

Use the smallest checks that can catch a real regression: configuration
composition, shell syntax, import/compile checks, focused smoke tests, processor
installation checks, checkpoint load checks, and deterministic validation
comparisons where relevant. Verify architecture toggles in both old and new
modes.

Important: do not add vanity tests without permission. Bad tests become bad
requirements and can cripple research iteration. When asking permission to add
tests, explain the proposed testing design and what failure it would detect.

# Anchor comments

Add specially formatted comments where appropriate as inline knowledge that can
be found with `rg`.

- Before scanning a relevant subsystem, first search it for existing
  `AICODE-NOTE:`, `AICODE-TODO:`, and `AICODE-QUESTION:` anchors.
- Use `AICODE-NOTE:` for important invariants or non-obvious behavior.
- Use `AICODE-TODO:` for concrete follow-up work.
- Use `AICODE-QUESTION:` for unresolved design questions.
- Update or remove stale anchors when changing the associated code.
- Add anchors sparingly around complex, critical, or bug-prone logic; do not
  annotate routine code.

# Experiment documentation

Document material architecture/configuration changes and the exact launchers
used. Reports should separate observed evidence from hypotheses and should note
whether conclusions come from visuals, metrics, logs, or code inspection.
When comparing runs, prioritize controlled validation and visual face quality,
while also checking identity similarity, prompt adherence, artifacts, and
face/body alignment.

# Tool index and Comet run records

Use `diffusion_template/TOOLS.md` as the entry point for repository tools and
server-operation helpers.

For every new Comet-tracked experiment, verify during startup that
`saved/<run_name>/comet_experiment.json` exists and contains the experiment
key written by `CometMLWriter`. Use
`diffusion_template/tools/comet/comet_experiment.py` and that immutable key to
retrieve metrics or images. Do not identify an experiment later only by its
display name.
