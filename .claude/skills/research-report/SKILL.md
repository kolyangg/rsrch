---
name: research-report
description: Write an experiment analysis/report for this project (BA / PhotoMaker / cosmic_large / large_dataset runs) as a Markdown file, optionally rendered to PDF and uploaded to Dropbox. Use whenever asked to analyse run results, compare experiments, diagnose a failure, propose next experiments, or produce a report/writeup/PDF of findings. Covers the required report structure, evidence discipline, measurement conventions, the MD→PDF→Dropbox pipeline, and the silent data-joining traps in this repo.
---

# Research report

Produce reports that a colleague can audit and act on without rerunning the
analysis. The structure below is the house style; follow it unless the user asks
for something else.

## Before writing: get real evidence

A report is only worth the measurements behind it. Do not write conclusions from
config inspection alone when data is reachable.

- **Identify runs by immutable Comet key, never display name.** Keys live in
  `saved/<run_name>/comet_experiment.json`, or pull with
  `python tools/comet/comet_experiment.py pull --run-name <R> --host serv --remote-project <abs path>`.
- **Measure, then claim.** If asserting a run "looks worse", quantify it —
  per-image tables, distributions, counts of a defined failure. Aggregate means
  routinely hide the effect the user is describing; the tail is usually the story.
- **Check the obvious cause first and say so when you rule it out.** A ruled-out
  hypothesis is a finding.
- **Compare at matched steps.** Runs at different training steps are not
  comparable; if you must, label it explicitly as indicative.
- **Prefer an interventional arm as proof.** An existing run that changed exactly
  the suspected variable is stronger than any correlation.

## Report structure

```text
# <Specific title: the finding, not the topic>

**Date / Scope / Evidence cutoff**        what was and was not touched
<run table: arm | immutable Comet key | step | headline metric>

## Executive conclusion
## 1. <The measured phenomenon>           tables; which items are affected
## 2. <Which runs are better/worse>       ranked, with caveats on step mismatch
## 3. Root cause                          incl. "what is NOT the cause"
### Confidence                            per-claim table
## 4. Proposed experiments                1-2, priority ordered
## 5. Implementation plan                 steps a coding agent can follow
## 6. Reproducing                         exact commands + known traps
## 7. References
```

Rules that make it useful:

- **Lead with the conclusion.** First paragraph states the finding and its
  strength.
- **Separate observed evidence from hypothesis.** Tag claims `[measured]`,
  `[code]` (verified by reading source), `[report]` (prior audit), `[hypothesis]`.
- **Always include a confidence table** with a one-line basis per claim.
- **Always include "Not established"** or "what is NOT the cause".
- **State when you were wrong.** If the data contradicts an earlier
  recommendation of yours, say so plainly in the executive conclusion.
- **Numbers in backticks**, bold only the decisive ones. Tables over prose for
  anything with more than three values.
- **Every proposed experiment needs**: config name, single scientific change,
  hypothesis, prediction, risk, and decision gates.
- **Name the primary metric explicitly**, especially when it differs from the
  project default (`id_sim`). Say which metric would mislead and why.

## Project invariants to preserve

Keep these fixed in any proposed experiment unless the user overrides:
`use_branched_attention=true`, `pipeline.pose_adapt_ratio=0`,
`pipeline.ca_mixing_for_face=false`, the fixed 96-image `manual_val` panel,
seeds, prompts, references, scheduler, inference steps, and metric definitions.
State the trainable contract (e.g. `2,240 tensors / 219,217,920 parameters`) and
whether the arm changes it.

## Build and publish

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh && conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template

# Markdown -> PDF -> (optional) Dropbox, in one call
python tools/reports/publish_report.py analysis/<YYYY-MM-DD>_<slug>.md --upload
```

`publish_report.py` renders with pandoc/xelatex into `analysis/assets/`, copies
any referenced figures next to the PDF, and calls the project Dropbox tool.

Conventions:

- Report path `analysis/<YYYY-MM-DD>_<slug>.md`; PDF into `analysis/assets/`.
- Embed figures with a path relative to `analysis/`, e.g.
  `![caption](assets/problem_grid.png)`. Build a comparison grid when a visual
  claim is made — annotate it (draw the mask box, label each column with the run
  and its metric).
- **A Dropbox upload is not complete without the temporary link.** Always paste
  the exact link into the reply and note it expires in ~4 hours.

## Known silent traps in this repo

These fail quietly — they return plausible partial results rather than erroring.

1. **Space/underscore key mismatch.** Validation bbox keys and per-image
   `id_sim` `output_key` are built from `prompt[:10]` and contain **spaces**;
   exported PNG filenames use **underscores**. Joining literally silently drops
   ~83% of images. Normalise with `name.replace(" ", "_")` on both sides.
2. **Comet API returns figure names for image assets**, not output keys. For
   anything joining to bbox or `id_sim` tables use
   `tools/comet/comet_experiment.py fetch`, and give **each step its own
   `--output-dir`** or successive fetches overwrite each other.
3. **`loss_kind` overrides `loss_function._target_` in `train.py`.** Testing
   `config.loss_function` directly instantiates the wrong class. Replicate the
   mapping in `train.py` when validating a loss change. A missing `loss_kind` has
   silently degraded runs to `MaskedDiffusionLoss` before.
4. **`pgrep -f "<pattern>"` matches the wrapper shell** whose command line
   contains that pattern, so `until ! pgrep -f ...` never terminates. Poll for
   the artifact, or use `run_in_background` and the completion notification.
5. **`python` may not exist** outside the conda env; always activate
   `photomaker` first.
6. **Local `onnxruntime` is the CPU build.** For GPU face embedding use the
   PyTorch ArcFace at `src/model/photomaker_branched/arcface_identity_aux.py`
   (same `w600k_r50` graph, verified cosine 1.0) rather than InsightFace.

## Useful measurement tools

- `tools/datasets/measure_face_body_alignment.py` — detected face vs the fixed
  mask box: centre offset, size ratio, IoU. Size ratio `<0.8` = undersized face.
- `tools/comet/comet_experiment.py` — pull/fetch by immutable key.
- `tools/datasets/build_cosmic_identity_assets.py` — ArcFace embedding/grouping.
- `tools/dropbox/upload_to_dropbox.py` — enforces `/rsrch/YYYY-MM-DD/<file>` and
  verifies the content hash.

## Checklist before returning

- [ ] Every run identified by immutable Comet key
- [ ] Claims tagged measured / code / report / hypothesis
- [ ] Confidence table present
- [ ] "What is NOT the cause" present
- [ ] Comparisons at matched steps, or the mismatch flagged
- [ ] Primary metric named; misleading metrics called out
- [ ] Proposed experiments have gates and predictions
- [ ] Reproduction commands included
- [ ] PDF built; Dropbox link pasted with its ~4h expiry noted
