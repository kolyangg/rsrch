# Clean-branch removal candidates

**Date:** 12 August 2026

**Branch:** `kit/e13-family-clean`

**Scope:** files irrelevant to E13, BC_E13, CL14, CL18, CL19 and CL20 clean
training/validation, including their retained dataset, reporting and Serv tools

**Action taken:** approved Batch A removed from the branch worktree on
12 August 2026; conditional Batch B remains

**Recovery proof:** all 573 removed tracked paths exist on
`origin/main_clean` commit `19a812f9c842153f412b88182f9beae2b4b9c7aa`;
58 also exist on `origin/test` commit
`ad194a026ab701dd979712d415c487dd536a4645`. No approved path was absent from
both recovery branches before deletion.

## Executive recommendation

Cleanup is split into two passes. The approved conservative Batch A below has
now been removed: its paths are outside the clean launch/config/import closure
and consist of old schedulers, old Serv packages, upstream comparison
checkouts, explicitly named source backups, and generated debug images. Batch B
still waits because it contains historical comparison/reporting material that
is not needed by clean runs but has stale documentary or default-path
references.

The completed first pass removed **573 tracked files** and approximately
**110 MB** from this branch. The confirmed branches above and Git history remain
the recovery mechanisms. File size is informational only and is not evidence
that a path is safe.

## Evidence and method

Observed evidence:

1. The production entrypoint is
   `diffusion_template/launchers/active/run_e13_family_24k_1gpu.sh`, reached on
   Serv through `launchers/serv/start_e13_family_1gpu.sh`.
2. The six supported leaves resolve through `e13_family_24k.yaml`, CL14's
   dataset/mask leaf, or `cl18_cl20_extension_24k.yaml`; none imports a parent
   `PhotoMaker`, `PuLID`, `persongen`, `_clearml_examples`, old scheduler, or
   `serv_new_runs` path.
3. PhotoMaker model code used by the clean runs is vendored under
   `diffusion_template/src/model/` and `src/pipelines/`. Model weights resolve
   from the Hugging Face cache or the `.env` `PM_PATH`, not from root
   `PhotoMaker/`.
4. Static searches found no production imports into any explicitly named
   `_old`, `_old2`, `_old3`, `_backup`, or trainer/pipeline archive directory.
   The only cross-reference found was historical documentation, plus one old
   pipeline importing another old backup.
5. `serv_new_runs/` contains 50 pre-clean YAML/shell files. The clean launchers
   do not call them. One retained dataset-analysis notebook mentions two old
   paths as historical provenance; that is not an executable dependency.
6. There are 24 tracked sbatch files: three at `diffusion_template/` and 21
   inside `diffusion_template/_old/`. Clean Serv jobs use MLS YAML and neither
   launcher invokes Slurm.

Interpretation: Batch A can be removed without changing clean-run model output,
dataset selection, validation protocol, metrics, checkpoint semantics, or Serv
startup. This conclusion is **high confidence for the six clean recipes**. It
does not claim that legacy PhotoMaker/PuLID comparison commands or historical
one-ID experiments will continue to work after cleanup; removing those flows is
the purpose of the clean branch.

## Batch A — removed after approval

### A1. Old Slurm launchers and their adjacent archived trainers

Remove the three root-level sbatch files:

- `diffusion_template/18Apr_cometL.sbatch`
- `diffusion_template/18Apr_cometL_rocky.sbatch`
- `diffusion_template/23Mar_all.sbatch`

Remove the entire directory:

- `diffusion_template/_old/` — 27 files, approximately 140 KB. This includes
  the other 21 sbatch files and six October/November backup training or
  inference scripts.

This removes all 24 tracked sbatch files in the branch. No clean launcher or
config references them.

### A2. Pre-clean Serv packages

Remove the entire directory:

- `diffusion_template/serv_new_runs/` — 50 files, approximately 284 KB.

It contains only the older `run_*.yaml`, `start_ba_*.sh`, and its `_old/`
subdirectory. The replacement is the exact clean package set documented in
`diffusion_template/serv_run_packages/README.md` and the shared fail-closed
launcher.

### A3. Parent-folder external model copies and their setup residue

Remove these complete tracked directories:

- `PhotoMaker/` — 176 files, approximately 85 MB;
- `PuLID/` — 96 files, approximately 5.5 MB;
- `persongen/` — 133 files, approximately 4.2 MB;
- `_clearml_examples/` — 3 files, approximately 16 KB.

Remove these associated standalone files:

- `setup_pulid_NS3.sh`
- `pl_requirements.txt`
- `pm_requirements.txt`
- `diffusion_template-main.zip`
- `README_cluster.md`

Why safe for clean runs: these are upstream/reference implementations, an old
PuLID environment bridge, old dependency lists, an unused source archive, and
obsolete Slurm-era setup text. Clean code uses the internal PhotoMaker/BA
implementation and `diffusion_template/hpc_requirements.txt` plus the retained
environment snapshots. The root `README.md` is deliberately not in Batch A.
Replacement content is staged as `README_E13_FAMILY_CLEAN.md`; after approval
it can become the conventional root README without leaving the branch
undocumented.

### A4. Explicitly archived source/config copies

Remove these complete directories:

- `diffusion_template/src/configs/_old/` — 17 files;
- `diffusion_template/src/configs/_old2/` — 6 files;
- `diffusion_template/src/configs/datasets/_old/` — 1 file;
- `diffusion_template/src/configs/pipeline/_old/` — 4 files;
- `diffusion_template/src/model/photomaker_branched/_backup/` — 4 files;
- `diffusion_template/src/model/photomaker_branched/_old/` — 6 files;
- `diffusion_template/src/model/photomaker_branched/_old2/` — 3 files;
- `diffusion_template/src/model/photomaker_branched/_old3/` — 1 file;
- `diffusion_template/src/model/photomaker_branched/_old_attn_pr_model/` — 1
  file;
- `diffusion_template/src/model/photomaker_branched/_old_masking/` — 4 files;
- `diffusion_template/src/pipelines/_old2/` — 3 files;
- `diffusion_template/src/trainer/_old/` — 6 files.

Also remove the two standalone backups:

- `diffusion_template/src/configs/ddp/accelerate_backup.yaml`
- `diffusion_template/src/datasets/cosmic_backup.py`

These 58 files are explicitly archived copies. No active clean target imports
them. Do **not** remove similarly named active files such as
`src/configs/lr_scheduler/warmup_hold_cosine_24k.yaml`; it is part of every
clean 24k schedule despite containing the word `old` inside `hold`.

### A5. Checked-in debug outputs

Remove these complete directories:

- `diffusion_template/hm_debug_orig/` — 21 PNG files, approximately 15 MB;
- `diffusion_template/dim_debug/` — one historical notebook, approximately
  1.2 MB.

They are static diagnostic outputs, not inputs to training or validation.
Current debug/preflight tools remain under `diffusion_template/tools/`.

## Batch B — likely removable, but not in the first deletion

These are irrelevant to clean execution but have references that should be
retired in the same change, so they are not labeled unconditional Batch A.

### B1. `compare/` and the legacy comparison landing page

Candidate directory:

- `compare/` — 131 files, approximately 13 MB.

Associated legacy material:

- root `README.md`, which is currently a PhotoMaker/PuLID comparison runbook;
- `diffusion_template/src/configs/inference/ba_testing_new_oneid.yaml`;
- `diffusion_template/src/configs/inference/ba_testing_vs_original.yaml`;
- the inactive `import_mask="../compare/..."` default in
  `src/pipelines/photomaker_branched_clean.py`;
- historical `compare/` examples embedded in `setup_pulid_NS3.sh` (already in
  Batch A).

The clean fixed-96 path uses bbox masks in `dataset_full`, not `compare/`.
However, deleting `compare/` alone would leave stale examples and an obsolete
default string. Recommended second-pass action: replace the root README with a
clean branch landing page, remove the two legacy inference configs if they are
also approved, and change the unused pipeline default to `None` in one reviewed
line. Then `compare/` is safe to remove.

### B2. Superseded reporting utilities

Potentially removable directories:

- `diffusion_template/comet_utils/` — 17 files, approximately 128 KB;
- `diffusion_template/infer_tools/` — 3 files, approximately 40 KB.

The maintained equivalents live under `diffusion_template/tools/comet/`,
`tools/inference/`, and `tools/reports/`. Nevertheless, earlier instructions
explicitly asked to retain useful tools and skills, so these two directories
should remain until their historical JSON/report formats are confirmed
unneeded.

### B3. Older narrative READMEs

Potentially removable after the new clean README is accepted:

- `diffusion_template/README_ba.md`
- `diffusion_template/README_ba_new.md`

They describe older one-ID commands and mention archived processors. They are
not runtime dependencies, but may still be useful architectural history.

### B4. Further pruning of non-family configs or version-numbered modules

Not approved by this audit. Many apparently old config files participate in
Hydra inheritance, and some version-numbered processors are still reached by
the clean model. A later aggressive pruning pass would need a generated Hydra
composition closure plus import tracing for all six recipes and retained tools.
Filename age or numbering alone is not enough evidence.

## Paths that must stay

Retain at least:

- `dataset_full/`, including fixed-96 protocol files, BigCelebs, and helper
  metadata;
- current `diffusion_template/src/` model, pipeline, trainer, dataset, metric,
  loss, and Hydra dependency files not explicitly listed in Batch A;
- `diffusion_template/bbox_utils/` and its detector asset;
- `diffusion_template/launchers/active/` and `launchers/serv/`;
- `diffusion_template/serv_run_packages/`;
- `diffusion_template/tools/`, `scripts/`, `setup/`, `docs/`, and `analysis/`;
- `diffusion_template/.env.example`, `.gitignore`, `TOOLS.md`, `train.py`, and
  `hpc_requirements.txt`;
- root `.claude/`, `.codex/`, `.gitignore`, and `AGENTS.md`.

In particular, do not remove `dataset_full/` merely because external dataset
roots are also configured through `.env`: default Hydra paths and fixed-96
validation assets deliberately resolve through the sibling directory.

## Verification after deletion

Before committing a cleanup, run:

```bash
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py
python tools/validate_cl18_cl20_config.py \
  --config-name CL18_cosmic_crossview_spatial_consistency_24k
python tools/validate_cl18_cl20_config.py \
  --config-name CL19_cosmic_true_soft_fullquery_router_24k
python tools/validate_cl18_cl20_config.py \
  --config-name CL20_cosmic_bigcelebs_hardcase_curriculum_24k
bash -n launchers/active/run_e13_family_24k_1gpu.sh
bash -n launchers/serv/start_e13_family_1gpu.sh
```

Also compile/import the active Python targets and repeat the static scan for
deleted path names. No training or Serv submission is needed merely to verify a
file-only cleanup.

Post-deletion result on 12 August 2026:

- the staged deletion set contains exactly 573 paths and no path outside the
  approved Batch A scope;
- all 573 deletion paths still resolve on `origin/main_clean`;
- active model/pipeline/trainer Python targets compile;
- E13, BC_E13 and CL14 composition/shared-projection checks pass;
- sealed CL14 pipeline, denoising and fixed-96 input parity passes;
- CL18, CL19 and CL20 config/architecture checks pass at 24k, 2,240 trainable
  tensors and 219,217,920 trainable parameters;
- both active/Serv launchers pass shell syntax checks;
- all six exact clean Serv YAMLs parse.

## Limitations and what is not established

- The 573 Batch A tracked paths were removed. Batch B was not removed.
- The Serv NFS checkout is not mounted locally, so the new YAML paths can be
  checked structurally but not existence-tested from this machine.
- No A100 job, model construction, checkpoint replay, or fixed-96 RGB replay
  was run. This inventory therefore establishes dependency separation, not a
  new empirical generation-parity result.
- Historical workflows outside the six clean recipes are intentionally not
  guaranteed after Batch A.
