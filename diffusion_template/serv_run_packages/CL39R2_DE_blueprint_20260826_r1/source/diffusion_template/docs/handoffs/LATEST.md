# Current handoff: clean_full

Updated 26 August 2026. Read this file before changing code, interpreting
results, or launching a job from `diffusion_template`.

## Current code boundary

`clean_full` is the supported unified branch. It intentionally contains one
trainer/model/pipeline implementation, one Serv launcher, a small Hydra config
inheritance closure, and four training dataset implementations. The support
set is PM0; CL14, CL19, CL23, CL27, CL39; BC39; CL40-CL45; E13; BC_E13; and the
BC_E13 ds1-ds3 dataset arms. The allowlist and immutable historical Comet keys
are in `src/configs/clean_full_runs.json`.

Use only:

```bash
python tools/validate_clean_full_config.py --list
CONFIG_NAME=<allowlisted-config> RUN_NAME=<new-name> \
  bash launchers/active/run_clean_full_config_1gpu.sh
```

The config selects scientific behavior. Ad-hoc Hydra overrides are rejected.
The launcher selects a dataset preflight from the resolved config, seeds the
canonical run record, verifies the live Comet key, and runs deferred fixed-96
face quality only after Accelerate succeeds.

The full file/class/function and exclusion map is
`analysis/2026-08-22_clean_full_code_structure_and_run_inventory.md`.

## Scientific invariants

- Run Hydra from `diffusion_template/`; sibling `../dataset_full` paths are
  intentional.
- Training and validation use hard-replacement branched self-attention with
  reference K/V and target Q. Branched cross-attention is disabled.
- `pipeline.pose_adapt_ratio=0` and `pipeline.ca_mixing_for_face=false` are
  mandatory.
- Validation is the sealed 96-image `manual_val` panel at step 0 and every
  2,000 optimizer steps, one generated image per item. PM0 is validation-only
  at step 0 and disables branched attention.
- Identity curves are `id_sim_best_legacy` and mask-matched
  `id_sim_subject_v2`; the latter uses the sealed subject-v2 embeddings.
- All new training must preserve the optimized processor lookup/collector
  pipeline documented in
  `analysis/2026-08-16_training_pipeline_processor_lookup_fix.md`.
- Immutable Comet keys, not display names, identify historical runs.

## Current result context

The completed CL38-CL45 comparison found CL39 (null-key confidence router) to
be the strongest current system and CL44 (semantic window gate) the secondary
candidate. CL38 is deliberately excluded from the clean support set because
its recovery history is more complicated and it was not requested as a main
target. CL40-CL45 are retained as the latest six requested configs even where
their result did not beat CL39.

Canonical keys for all supported runs are recorded in
`src/configs/clean_full_runs.json`. The 21 August code/visual/metric synthesis
remains available from branch `test`; its decision-relevant conclusion is
summarized here so generated figure assets do not re-enter `clean_full`.

## Reusable six-run comparison report

The 23 August report extends the fixed PM0/CL14/CL19/CL23/CL27 comparison with
authoritative CL39 r4 (`b1ca0b3da679401c85b991f1bbdf0b2a`) at the completed
24k endpoint. The reusable report config is
`tools/comet/comet_pdf_config_23Aug_PM0_CL14_CL19_CL23_CL27_CL39_faces.json`;
the immutable-key export manifest is
`tools/comet/comet_runs_23Aug_PM0_CL14_CL19_CL23_CL27_CL39.json`. It keeps eight
image columns, expands to six experiment rows, and retains the sealed fixed-box
face-closeup pages.

`build_comet_report_pdf.py` now accepts JSON `metric_reference_lines`. For
`manual_val/id_sim`, PhotoMaker's step-0 value `0.556580` is drawn as a dashed
horizontal line rather than a one-point curve. CL39's curve labels its
`0.570124 @16k` peak and `0.566342 @24k` endpoint. The refreshed export has
exactly 96 images and one 96-row ID table for each of six immutable runs, with
no warnings or fallback steps.

The original 36-page PDF and the prior 47-page detailed PDF remain available.
The current reordered version is
`output/pdf/comet_report_PM0_CL14_CL19_CL23_CL27_CL39_reordered_appendix_23Aug2026.pdf`
(`94,402,257` bytes; SHA-256
`51dca00b31ae13bab786639d4367a0772aaf248b93f3924d60274d6220169547`).
It has 56 pages. Pages 1-3 are the ID curve, grouped identity/prompt means, and
a 20-row architecture comparison covering processor class/mode, target query,
spatial routing, reference support, output-projection order, frequency bands,
denoising gains, confidence, surface supervision, inference delta, trainables,
attn2, and fixed controls. Page 4 is the fixed references/prompts contract;
pages 5-16 are full images; pages 17-28 are fixed-face crops. Page 29 is an
Appendix flysheet, followed by 27 pages of cumulative architecture analysis,
fully annotated equations, and current on-disk `clean_new` code excerpts.

## BC39 launch state

On 24 August 2026, BC39 was added as the controlled BigCelebs dataset-transfer
arm of CL39. The original config
`BC39_big_celebs_null_key_confidence_router_24k` inherited CL39 and changed
only `train_dataset_name` plus the experiment comment; its resolved
architecture, optimizer, schedule, trainable contract, and fixed-96 validation
controls remained CL39. The original Serv/Comet run name was
`BC39_big_celebs_null_key_confidence_router_24k_full96_r1`.

The sealed Serv runtime source revision is
`4ac98179dc9f9d78f969954da22e975fa8bdc2e2+BC39.0bcb7a2f4bb2972151871404e0604b04d472c4fcbda4809324664a9d3c24408b`.
Its BigCelebs v2 manifest and download-completion preflight passed. Three MLS
submission attempts on 24 August were rejected before job creation with
`WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE`. The second and third attempts
followed explicit user retry requests while the workspace still reported 14
running GPUs.

On 25 August, a fourth explicitly requested submission was accepted as MLS UID
`40f75a2c-913a-42d7-800f-e53fe172c6ff`. The immutable Comet key is
`7f28fd59e7f8432ab94f7cb2b447d9e6`. BigCelebs preflight passed, source and
trainable contracts matched exactly, and step-zero validation completed all 96
images; it wrote the 96-row ID table and staged the face-quality manifest.
Immediately afterward, the first training batch failed before an optimizer
step with `RuntimeError: Frequency-surface loss requires an ownership mask`.
The MLS job is Failed and no training batch completed. BC39 inherits CL27's
frequency-surface objective, but the then-current BigCelebs loader did not emit
the `ba_occluder_mask` supplied by the Cosmic loader's deterministic semantic
occlusion policy. Do not relaunch r1 unchanged. Recovery required an explicit
scientific choice between porting CL27's synthetic ownership-mask augmentation
to BigCelebs (preserves the full CL39 objective) or disabling/changing that
objective (not a dataset-only CL39 transfer).

The user explicitly selected the full-objective recovery. On 25 August,
BigCelebs gained a backward-compatible, default-off target-augmentation hook
and the exact CL27 deterministic semantic-occlusion families, probability
`0.25`, and seed `150017`. Enabled samples always emit `ba_occluder_mask`, with
a zero tensor on deterministically unsampled examples; existing BigCelebs/E13
configs remain off. BC39 alone enables this policy, and its resolved config
differs from CL39 only by the dataset selection, the two augmentation fields,
and experiment comment. A focused dataset smoke check verified deterministic
nonzero/zero masks and default-off compatibility; both BC39 and historical
BC_E13 passed the fail-closed config validator.

The recovery run is
`BC39_big_celebs_null_key_confidence_router_24k_full96_r2`, using sealed Serv
source revision
`4ac98179dc9f9d78f969954da22e975fa8bdc2e2+BC39r2.f4695837c8e31c0c64c763a435cf1b5a159501843f85ac25687944e8a75f6cf4`.
MLS accepted it as UID `a94c9def-5f00-4fbd-a568-b6a9e6b7d5dc` / job
`lm-mpi-job-b94c7816-f219-42d7-a189-b66b573ece7e`; its immutable Comet key is
`96cfa64b72934afc870432a243cd4a55`. The sealed BigCelebs preflight passed all
349,348 records and 64 decoded target/reference pairs. Step-zero validation
completed all 96 images in eight batches, wrote the 96-row ID table, and staged
a 96-item deferred face-quality manifest. Training then crossed the r1 failure
point and reached 32/2,000 batches with no error signature. Monitoring stopped
there as requested; the MLS job remained Running and training was left active.

`build_comet_report_pdf.py` now accepts JSON `page_order`, Markdown
`report-group` selectors, and a `flysheet` layout. The report config removes
the word `final` from run labels and explicitly encodes the requested section
order. The Markdown records its source-worktree boundary so historical Comet
results are not presented as fresh clean-branch reruns. This is a reporting
change, not a new experiment.

## CL39 24k attention/confidence audit

On 24–25 August 2026, the canonical CL39 r4 checkpoint (epoch 12 / 24k,
immutable Comet key `b1ca0b3da679401c85b991f1bbdf0b2a`) was audited through
the historical `train.py` validation-only/YAML path on Serv A100s. The
checkpoint is `1,318,771,270` bytes with SHA-256
`74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07`.
The exact sealed config, runtime source, validation grouping, CUDA generator,
RealVisXL base, 1024 resolution, DDIM50, CFG5, seed0, prompts, references, and
face boxes were retained. A 12-image fail-closed smoke matched the sealed PNGs
byte-for-byte; the final instrumented actual arm then matched all 96 sealed
outputs with RGB MAE/max error `0` and `0%` pixels changed above `1/255`.

The initial three complete fixed-96, batch-12 arms ran as ordinary Serv YAML jobs: actual
plus telemetry `lm-mpi-job-79af3dd2-f662-48bc-ac83-c18adc33d490`, forced
`C=1` `lm-mpi-job-ba15e767-18cb-4000-aa8a-8b1613c683a1`, and explicit BA
correction `=0` `lm-mpi-job-01f46a9a-d2cb-4f4b-b279-0842bf1a2718`. Metrics
and final figures were assembled/rendered on Serv by
`lm-mpi-job-20808f03-0104-44dd-a0ae-b21901ccdc49`. Compact all-layer telemetry
was retained for deterministic indices
`1,7,13,17,33,35,38,40,51,55,63,69,78,80,87,93`; the hooks add no checkpoint
keys and never persist an `L×L` tensor.

A fourth fixed-96, batch-12 Serv YAML arm directly visualized branch behavior:
`lm-mpi-job-1d94aa46-9197-466a-8f22-1abcce4e4312` set target attention to
`N + soft_face_router * (R-N)` at all 36 shipped CL39 processors, so raw `R`
replaced `N` inside the existing face router while `N` remained outside. The
job completed all 96 outputs and rendered 16 detailed panels plus four overview
grids. Against the existing N-only arm on the selected 16 cells, routed raw `R`
changes `76.60%` of all pixels and `95.24%` of fixed-face-crop pixels; mean RGB
MAE is `0.02163` globally and `0.07253` on the face. The signed differences
trace facial structure, but raw `R` frequently duplicates or warps facial
parts. This is a whole-denoising evaluation intervention, not a direct VAE
decode of an intermediate attention tensor or a trained operating point.

On those 16 cells, routed-face confidence is `0.49485` with mean p10/p50/p90
`0.39668/0.47726/0.54157`; effective low/high weights are
`0.24793/0.36627`, and applied correction/native magnitude is `0.22099`.
BA-off changes `60.71%` of pixels above `1/255` and lowers subject-v2 ID from
`0.55754` to `0.51925`. Actual-minus-off is `+0.03829`, with `15/0/1`
wins/ties/losses and paired bootstrap 95% interval `[+0.02093,+0.05774]`.
Forced `C=1` scores `0.49984`; actual is `+0.05770`, with `14/0/2` and interval
`[+0.03432,+0.07949]`. The explicit `(R-N)` lane and the entropy attenuation
are therefore both causally active on this selected exact Serv panel.

Both bands are live: relative `D-(D_low+D_high)` reconstruction error is
`0.001240`; the high band supplies `37.15%` of summed applied-band magnitude
and has `3.817x` the low band's magnitude-normalized spatial total variation.
Reference-face queries place `51.44%` of key mass inside boxes occupying
`16.68%` of the reference grid, with mean per-sample uniform-area enrichment
`4.643x`; `48.56%` remains outside because shipped CL39 does not exclude those
positions from the attention softmax.

The report is
`analysis/2026-08-25_cl39_entropy_confidence_attention_audit.md`; records are
under `artifacts/cl39_attention_24k_serv_a100/` and
`artifacts/cl39_attention_24k_serv_a100_branch_faces/`, figures under
`analysis/assets/cl39_attention_24k_serv_a100/` and
`analysis/assets/cl39_attention_24k_serv_a100_branch_faces/`, scripts under
`tools/analysis/`, and the playground is
`notebooks/CL39_attention_analysis.ipynb`. The strongest next causal test is a
correct-ID-token/spatial-reference-shuffle arm. The standard publisher created
the visually inspected 24-page PDF at
`analysis/assets/2026-08-25_cl39_entropy_confidence_attention_audit.pdf`
(`52,323,195` bytes; SHA-256
`60ed1a53a3e242e45724ab85c49dab133eb79c3ae860c6de89dbfd425f23cca8`) and
uploaded it with verified integrity to
`/rsrch/2026-08-25/2026-08-25_cl39_entropy_confidence_attention_audit.pdf`.
A temporary direct-download link was issued on 25 August 2026 and expires in
approximately four hours.

## CL39 raw-R diagnosis and R2 architecture recommendation

On 25 August 2026, the raw `R-on-face` audit was compared with the clean-new
CL14/E13 hard reference-face route and the completed CL40-CL44 mechanisms. The
important scope correction is that the retained CL39 stress, forced-`C=1`, and
correction-zero arms act on the 36 null-key processors in `up_blocks.0/1`;
the other 34 transferred BA processors retain their normal CL23/27 route. The
correction-zero result is therefore not a global 70-processor BA-off result.

The decision is two-part. Raw R is genuinely underconstrained and frequently
misregisters facial geometry, and forced `C=1` proves that over-trusting the
same up0/up1 residual harms combined denoising. However, raw R is about `3.80x`
the mean magnitude of the actual routed residual, while ordinary CL39 remains
the strongest fixed-panel end-to-end result. The stress images should be used
as a robustness/headroom diagnostic, not as evidence that ordinary CL39
generations generally fail.

The recommended first build is CL39-R2-A: training-only coherent R-route
dropout, initially on the audited 36 processors, sampled once per forward with
a warm ramp and disabled for validation/inference. It borrows CL14's requirement
that the reference lane sometimes own the face without restoring CL14's hard
inference handover. Bounded zero-init low/high reliability corrections and
separately switchable RMS tail caps are later independent arms; combine only
after each passes. Promotion requires the group-scoped correction-zero arm, a
new global 70-processor BA-off arm, and correct-ID-token/spatial-reference
shuffle so a clean native-path collapse cannot be called an R improvement.

The full evidence, equations, architecture diagram, file-level implementation
plan, verification steps, and experiment gates are in
`analysis/2026-08-25_cl39_r_branch_artifact_diagnosis_and_r2_architecture.md`;
the PDF is
`analysis/assets/2026-08-25_cl39_r_branch_artifact_diagnosis_and_r2_architecture.pdf`.
The report-specific figure renderer is
`tools/analysis/render_cl39_r2_report_figures.py`. The visually inspected PDF
is `13,107,001` bytes with SHA-256
`0e90174d0bf7532536ae3df4e1d285b4ff69f9049bdef29a21d7e20cc0050223`;
the publisher uploaded it with verified integrity to
`/rsrch/2026-08-25/2026-08-25_cl39_r_branch_artifact_diagnosis_and_r2_architecture.pdf`.
A temporary direct-download link was issued on 25 August 2026 and expires in
approximately four hours.

The attached CL14-versus-CL39 query diagram was then audited specifically for
the hypothesis that CL39 R should use CL14-style `q_face=q*M`. Binary q masking
does not change the reference message at any routed face token because the
CL39 soft router satisfies `S>0 => M=1`. In temporal-frequency CL39 it is not
fully byte-inert: changing R outside M can enter the 5x5 Gaussian split and
affect a two-latent-cell boundary strip before S is applied. It is therefore a
cheap boundary ablation, not a plausible direct repair for face-core duplicated
eyes/glasses/noses. The clean-new lesson remains direct face-lane ownership
during training. The refined plan first evaluates full/binary/soft query modes
on the sealed checkpoint, then runs the previously recommended ownership arm
only if appropriate; the ownership arm makes `F=Attn(q*M,K_r,V_r)` explicit
but its scientific change is the occasional full face route, not multiplication
by M. The plan is
`analysis/2026-08-25_cl39_cl14_qface_hypothesis_and_experiment_plan.md`.

## Completed CL19/CL23/CL27/CL39 branch-lineage audit

The 25–26 August controlled audit is complete. It generated 18 validation-only
arms × 96 images = 1,728 Serv A100 images from immutable CL19/CL23/CL27/CL39
24k checkpoints. Fixed controls were RealVisXL, DDIM50, CFG5, seed0, batch 12,
the sealed `manual_val` prompts/references/boxes/masks, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`. It created no Comet experiment. All four
instrumented actual arms reproduced their sealed 96 PNGs byte-for-byte: mean
and maximum RGB MAE were zero, maximum absolute difference was zero, and no
pixel changed. The selected diagnostic set is deterministic stratified two per
identity, seed `390024`, 16 total.

Primary subject-v2 mask-matched ID means on those 16 were: CL19 actual/trained
reference route `0.48975`; CL23 actual/native/raw-R/low/high
`0.51368/0.48998/0.44318/0.50078/0.52027`; CL27
`0.52132/0.49972/0.46396/0.51091/0.52977`; and CL39
actual/native/raw-R/low/high/C=1
`0.55754/0.52184/0.42241/0.52793/0.54047/0.49984`. CL39 actual beat native by
`+0.03570` with paired bootstrap interval `[+0.01204,+0.06613]`, raw R on all
16 by `+0.13513` with interval `[+0.09178,+0.18101]`, low-only by `+0.02961`
with interval `[+0.01258,+0.04730]`, and C=1 by `+0.05770` with interval
`[+0.03443,+0.07939]`. High-only was a smaller intervention than low-only and
had higher ID in CL23/27/39; CL39 actual minus high-only was `+0.01707` with an
interval that nearly touches zero `[-0.00003,+0.03826]`.

The conclusion is that CL39 raw R is a fragile standalone stress route, while
the normal N-anchored frequency/confidence correction is active and beneficial
on this panel. Raw R bypasses the Gaussian split, so D-low/D-high do not cause
its artifacts. Both isolated bands are active; confidence is not cosmetic,
because forcing C=1 measurably and visibly regresses the final denoising path.
CL19's coherent trained reference-owned route remains evidence that R can be
made more self-sufficient, but it is not an apples-to-apples checkpoint quality
comparison against off-operating-point CL39 raw R.

The report is
`analysis/2026-08-26_ba_lineage_r_frequency_confidence_audit.md`; scored local
assets are under `artifacts/ba_lineage_branch_audit_20260826/`. The sealed Serv
task root is
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/BA_lineage_branch_audit_serv_r1`.
The selected scoring archive SHA-256 is
`7d0aaea3569b7397e6f9cabfa5d5664984aecc1e8cfcfe786e1825fbae6fcaf4`.
The visually inspected eight-page PDF is
`analysis/assets/2026-08-26_ba_lineage_r_frequency_confidence_audit.pdf`
(`21,024,886` bytes; SHA-256
`ecbfcd33b7520adfdec0fc010739f92576e32fc110703ba84b929f0306d714da`).
The publisher uploaded it with verified integrity to
`/rsrch/2026-08-26/2026-08-26_ba_lineage_r_frequency_confidence_audit.pdf`.

Jobs were sequential `lm-mpi-job-a64df24d-350d-4cae-bdf5-8b31d2a5af29`,
parallel CL27/CL39 `lm-mpi-job-46454c8f-2967-4d12-9a28-a75d7232cf86`, and final
five-worker historical-runtime CL23 recovery
`lm-mpi-job-c7456c95-415d-4ce4-9d3e-5026b5196440`. The first unified clean
CL23 replay was rejected because it differed from the sealed endpoint (mean
RGB MAE `0.00712`, maximum image MAE `0.01549`); the immutable historical CL23
runtime restored exact 96/96 parity. The recovery also fixed analysis-mode
attachment through the stable processor map and disabled a platform C++
symbolizer hang, without changing the model equation or validation controls.
The audit-specific ten-GPU exception has ended; the normal six-GPU ceiling is
again in force.

## Machine and credentials

Neb is unavailable. Do not access it, use it as a proxy, or submit work to it.
Use the local machine or Serv as authorized. Before submitting on Serv, inspect
this project's running and pending jobs and respect the normal six-A100
concurrent request ceiling.

Machine paths and credentials belong only in `.env`; never commit them. Every
new Comet run must produce `saved/<run_name>/comet_experiment.json` with a live
32-character experiment key during startup.

## Historical recovery

Removed configs, external model mirrors, per-job sealed snapshots, launchers,
generated reports/assets, and alternate training families remain recoverable
from branch `test` at base commit
`97e0364d6fa6ee6b1b8c3d99aa547805b18ad47f`. Do not copy them back into
`clean_full`; use a historical checkout when exact replay of an excluded run is
required.
