# Codex 9 Jul: PhotoMaker vs N17 Runtime Ablations and Next Training Runs

Date completed: 2026-07-10

## Artifacts

- Comparison PDF: `full_validation_results/full_val_report_pm_vs_N17_runtime_ablation.pdf`
- PDF config: `infer_tools/full_val_report_pm_n17_runtime.yaml`
- Updated generator: `infer_tools/pdf_full_val.py`
- N23 config/script: `src/configs/one_id_ba_camix_train_N23.yaml` and
  `serv_new_runs/start_ba_camix_train_vast_N23.sh`
- N24 config/script: `src/configs/one_id_ba_dualgate_train_N24.yaml` and
  `serv_new_runs/start_ba_dualgate_train_vast_N24.sh`

The PDF has 10 pages: summary, key config differences, then one 12-prompt page for each of the
eight identities. All five result columns were verified as **96/96 matched images**.

Rebuild it with:

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
python infer_tools/pdf_full_val.py --config infer_tools/full_val_report_pm_n17_runtime.yaml
```

## Report Filename Bug Fixed

The old PDF code silently omitted 10 images per run. Inference resolves `<class>` as `man img`, so
the first ten characters of the two affected male prompts produce names such as:

- `Angry man _elon.png`
- `Chef man i_elon.png`

The report resolved `<class>` as only `man`, looked for `Angry man,_elon.png` and
`Chef man, _elon.png`, and displayed missing cells. This affected two prompts for each of five male
identities: 10 missing grid cells per run. The metrics were not affected because their script globs
the actual image files directly.

`pdf_full_val.py` now:

- uses trigger-word-aware filename candidates;
- indexes actual run files rather than assuming one exact reconstructed filename;
- supports metrics stored in separate/nested result folders;
- supports inference-only config overlays in the config table;
- prints matched counts and, for this report, fails unless every run matches all 96 cells.

## Runs Compared

| Short name | Result | Inference face-source behavior |
|---|---|---|
| PM | PhotoMaker V2 baseline | Original PhotoMaker only; no BA |
| N17 | N17 step 26000 | BA reference face only; PAR=0; CAMIX off |
| PAR | N17 step 26000 | PAR=0.25; CAMIX off |
| CAMIX | N17 step 26000 | PAR=0; CAMIX on |
| BOTH | N17 step 26000 | PAR=0.25; CAMIX on |

Important terminology: `ca_mixing_for_face` is a misleading old name. In the active code it changes
the **branched self-attention face K/V source**, not the branched cross-attention (`attn2`) weights.

## Metric Summary

All existing metric files contained 96 images and detected 96 faces, so no metric recomputation was
needed.

| Run | Mean ID sim | Delta vs N17 | Paired wins vs N17 | Delta vs PM |
|---|---:|---:|---:|---:|
| PM | **0.4886** | +0.1404 | 87/96 | - |
| N17 | 0.3482 | - | - | -0.1404 |
| PAR | 0.3527 | +0.0045 | 52/96 | -0.1359 |
| CAMIX | 0.3795 | +0.0313 | **70/96** | -0.1091 |
| BOTH | **0.3886** | **+0.0404** | 68/96 | -0.1000 |

PAR is effectively neutral: its median paired change is only +0.0030 and it wins almost exactly
half the images. CAMIX is the meaningful switch. BOTH has the best mean among BA variants, but its
gain is not uniform and PM remains substantially ahead.

### Per-identity ID similarity

| Identity | PM | N17 | PAR | CAMIX | BOTH | BOTH - N17 |
|---|---:|---:|---:|---:|---:|---:|
| eddie | 0.2053 | 0.1192 | 0.1458 | 0.1605 | 0.1905 | +0.0713 |
| elon | 0.5462 | 0.4993 | 0.4527 | 0.5074 | 0.5060 | +0.0067 |
| jennie | 0.5898 | 0.4361 | 0.4336 | 0.4663 | 0.4725 | +0.0364 |
| jensen | 0.5572 | 0.4448 | 0.4047 | 0.3891 | 0.4096 | **-0.0352** |
| jisoo | 0.5768 | 0.2417 | 0.2912 | 0.3684 | 0.3764 | **+0.1347** |
| keanu | 0.4997 | 0.3947 | 0.3700 | 0.4153 | 0.4100 | +0.0153 |
| lex | 0.4727 | 0.3792 | 0.3844 | 0.4116 | 0.4287 | +0.0495 |
| marion | 0.4611 | 0.2708 | 0.3398 | 0.3170 | 0.3151 | +0.0443 |

This variation matters more than the mean. Jisoo accounts for a large fraction of the aggregate
gain because CAMIX repairs several severe N17 face corruptions. Jensen moves in the opposite
direction and exposes the instability of applying one global mixing rule to every layer and prompt.

## Visual Comparison

### Base PhotoMaker

PM is the strongest overall result here. Faces are usually anchored correctly to the head and body,
neck length is plausible, expression follows the prompt, and face/background boundaries are clean.
It is not perfect: hats, hands, goggles, and distant faces make the ID metric unreliable, especially
the Chef rows. Still, its 0.4886 mean is consistent with the visual advantage rather than being only
a metric artifact.

### Plain N17

N17 keeps most of PM's global scene because all runs use the same seeds and staged generation, but
the BA face branch often fails to integrate with the PM-generated head. Repeated symptoms are:

- a small or high face pasted into a larger head/hair silhouette;
- elongated necks and a face center shifted relative to the shoulders;
- hair, goggles, hands, or headwear cutting through the injected face;
- a recognizable but rigid/canonical face that ignores the target expression or pose;
- severe local corruption on difficult examples, notably Jisoo skiing/kickboxing/night-ride.

### PAR=0.25 only

PAR-only remains visually close to N17 and does not consistently repair the integration problem.
Its implementation directly interpolates `ref_face_hidden` and `noise_face_hidden` at the same
sequence index. Those tensors come from different images with different face boxes, so index `i`
does not necessarily describe the same semantic face location in both branches. This makes the
blend geometrically questionable.

The results match that concern: the mean changes by only +0.0045, while Jensen skiing becomes a
major corruption (`-0.0223` ID sim) and several dancing/jumping images regress. PAR helps some
Marion/Jisoo cases, but it is not a reliable mechanism by itself.

### CAMIX only

CAMIX is the useful inference intervention. It appends current-generation face tokens to the
reference-derived face K/V set. This lets the face query recover geometry already established by
PhotoMaker while retaining access to the reference branch. It repairs many N17 failures:

- Keanu subway: the face returns to the head/shoulder geometry and the long-neck artifact is gone;
- Jisoo skiing: the N17 goggle/face corruption becomes a coherent face;
- multiple angry, crying, reading, and rushing examples become cleaner and more expressive;
- ID sim improves on 70/96 paired images.

It is still uncontrolled. Concatenating both token sets puts them into one attention softmax, so
source competition can change abruptly by layer/head/prompt. Jensen kickboxing and night-ride are
clear counterexamples, and Jensen's identity mean falls from 0.4448 to 0.3891.

### PAR + CAMIX

BOTH gives the best BA mean (0.3886), with especially large gains for Jisoo, Eddie, Lex, angry
faces, crying women, reading/rushing, kickboxing, and skiing women. It also retains prompt-specific
failures: Jensen crying/kickboxing, some dancing/jumping images, and several chef rows regress.

The conclusion is not "turn both knobs on everywhere." The result says that restoring access to the
current PhotoMaker-conditioned face is important, but a fixed global interpolation/concatenation is
too coarse.

## Keanu Subway Example

| PM | N17 | PAR only | CAMIX only | BOTH |
|---|---|---|---|---|
| ![PM](<../full_validation_results/photomaker_baseline/Rushing ma_keanu.png>) | ![N17](<../full_validation_results/ba_longrun_N17/Rushing ma_keanu.png>) | ![PAR](<../full_validation_results/par025_no_camix_ablation/ba_longrun_N17_step26000/Rushing ma_keanu.png>) | ![CAMIX](<../full_validation_results/pose0_camix_ablation/ba_longrun_N17_step26000/Rushing ma_keanu.png>) | ![BOTH](<../full_validation_results/par_camix_ablation/ba_longrun_N17_step26000/Rushing ma_keanu.png>) |

N17 places a small, flat face high on the head and stretches the neck into the shirt collar. PAR
does not solve it. CAMIX and BOTH re-anchor the face to the PM head geometry. This is strong evidence
that the failure is not simply insufficient identity training; the reference-only SA K/V pathway is
overriding the already-correct target geometry.

## Does PM Identity Overshadow BA Identity?

I measured, for every paired image, cosine similarity between the generated-face InsightFace
embedding and the corresponding PM/N17 generated-face embeddings. This is not a causal attribution
metric, but it is a useful paired indication of which output identity each variant resembles.

| Variant | Cosine to PM output | Cosine to N17 output | Images closer to PM than N17 |
|---|---:|---:|---:|
| PAR | 0.5194 | **0.6327** | 17/96 |
| CAMIX | 0.5797 | **0.6216** | 34/96 |
| BOTH | **0.6251** | 0.5760 | **54/96** |

Whole-image SSIM tells a complementary story:

| Variant | SSIM to PM | SSIM to N17 | Images structurally closer to PM |
|---|---:|---:|---:|
| PAR | 0.8412 | **0.9018** | 3/96 |
| CAMIX | 0.8529 | **0.8812** | 25/96 |
| BOTH | 0.8526 | **0.8635** | 33/96 |

Therefore:

- PAR does not make PM dominate; it stays strongly N17-like.
- CAMIX moves the face toward PM, but it is still slightly closer to N17 on average.
- BOTH tips the **face identity** toward PM in a small majority (54/96), especially for Marion,
  Jisoo, Eddie, and Keanu.
- The overall image remains more N17-like, and BOTH is still 0.1000 below PM in reference ID sim.
  Only 14/96 BOTH images beat PM's reference ID score.

So PM identity is not completely overshadowing BA identity, but BOTH does shift the face balance
too far toward PM for a mechanism intended to add strong BA identity. The desired solution is a
learned, layer/head-specific balance, not a single global switch.

## Would Training with CAMIX/PAR Be Different?

Yes. The flags themselves are not trainable, but they change the forward graph under which the SA
adapters are optimized.

Training calls `run_branched_forward_pass()` in `lora2_helpers.py`, which calls the same
`two_branch_predict()` used by inference. `patch_unet_attention_processors()` copies
`pose_adapt_ratio`, `ca_mixing_for_face`, and the runtime gate from the model/pipeline to every
processor. In `attn_processor_cleanest.py`, these values directly choose the face K/V inputs before
the trainable `ref_to_*` and `noise_to_*` projections are evaluated.

Consequences:

- N17 weights were optimized with reference-only face K/V, then CAMIX was introduced only at
  inference. The weights never learned how to normalize or use the extra generation-face tokens.
- Training with CAMIX can let the SA adapters co-adapt to the mixed token distribution and may
  reduce seams and catastrophic source switching.
- It can also collapse toward the easier PM path, because PM already gives stronger identity on this
  validation set. The N23 experiment is needed to measure this rather than assuming improvement.
- PAR-only is not recommended for training. Its same-index cross-image interpolation is the weaker
  mechanism both conceptually and empirically.

One separate mismatch remains: N17 uses `train_ba_all_steps=true`, while inference runs text-only,
then PhotoMaker, then BA. N23 deliberately retains this to isolate one change. N24 uses
`train_ba_all_steps=false` because its goal is the more coherent end-state design.

## Should the Staged Inference Schedule Stay?

Yes, for now. The first text-only steps establish global composition; PhotoMaker then establishes
the target face/head/pose; BA is best treated as a later identity refinement. N17's stuck faces are
consistent with injecting reference geometry too strongly, not evidence that BA should start even
earlier.

A trainable fusion layer should operate **inside the BA phase**, not replace the phase schedule.
Learning the stage boundary or applying BA from step zero would introduce a second large variable
before source fusion is stable. If N24 succeeds, a later experiment can make the gate timestep-aware
(small PM contribution early in the BA phase, larger only when local detail is formed).

## New Trainable Fusion: N24

The new mode is behind `ba_face_fusion_mode`; default `legacy` preserves all old behavior.

`dual_attention_gate` does the following in each branched self-attention processor:

1. Compute face attention once against reference-face K/V.
2. Compute face attention separately against current-generation-face K/V.
3. Blend the two **attention outputs** with a learned gate per attention head and layer.

This is better targeted than a new dense layer:

- reference and target face spatial grids are never interpolated index-by-index;
- the two sources do not compete inside one concatenated-token softmax;
- different heads/layers can learn different identity-versus-pose roles;
- only a few gate parameters are added per processor;
- N24 starts at 0.20 PM contribution and caps it at 0.50, preventing immediate PM domination.

The gate is installed before optimizer construction, marked trainable, saved in processor state, and
restored by inference when the same fusion flags are supplied. It adds a second face-attention call,
so some training/inference slowdown is expected, but it does not duplicate the full UNet.

## Recommended Run Order

### 1. N23: controlled CAMIX-during-training experiment

Run first. It changes the N17 mechanism only by enabling CAMIX throughout training; PAR remains 0,
CA (`attn2`) remains frozen, and the N17 loss/ID settings remain unchanged. Stop at 10k and compare
2k/4k/6k/8k/10k against the corresponding N17 checkpoints.

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
export COMET_API_KEY=...
bash serv_new_runs/start_ba_camix_train_vast_N23.sh
```

Full validation:

```bash
export N23_INFER_OVERRIDES="pipeline.ba_enable_runtime_sa_knobs=true model.ba_enable_runtime_sa_knobs=true pipeline.pose_adapt_ratio=0 pipeline.ca_mixing_for_face=true pipeline.ba_face_fusion_mode=legacy model.ba_face_fusion_mode=legacy"
RESULTS_DIR=full_validation_results/ba_camix_train_N23_steps \
EXTRA_INFER_OVERRIDES="${N23_INFER_OVERRIDES}" \
bash serv_new_runs/run_full_validation_steps.sh ba_camix_train_N23 2000 4000 6000 8000 10000
```

Decision criterion: not just mean ID sim. N23 should repair Keanu/Jisoo-like placement failures
without introducing Jensen-style prompt-specific collapses. If it simply becomes PM-like, do not
extend it beyond 10k.

### 2. N24: learned dual-source gate

Run second, or in parallel if a second GPU is available. It replaces both fixed PAR and CAMIX,
keeps branched CA frozen, caps PM contribution, and aligns the sampled training stages to the
text-only -> PM -> BA inference schedule.

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
export COMET_API_KEY=...
bash serv_new_runs/start_ba_dualgate_train_vast_N24.sh
```

Full validation must create the gate modules before checkpoint loading:

```bash
export N24_INFER_OVERRIDES="pipeline.ba_enable_runtime_sa_knobs=false model.ba_enable_runtime_sa_knobs=false pipeline.pose_adapt_ratio=0 pipeline.ca_mixing_for_face=false pipeline.ba_face_fusion_mode=dual_attention_gate pipeline.ba_face_fusion_gate_init=0.2 pipeline.ba_face_fusion_gate_max=0.5 model.ba_face_fusion_mode=dual_attention_gate model.ba_face_fusion_gate_init=0.2 model.ba_face_fusion_gate_max=0.5"
RESULTS_DIR=full_validation_results/ba_dualgate_train_N24_steps \
EXTRA_INFER_OVERRIDES="${N24_INFER_OVERRIDES}" \
bash serv_new_runs/run_full_validation_steps.sh ba_dualgate_train_N24 2000 4000 6000 8000 10000
```

Inspect learned gates as well as images. A healthy outcome is heterogeneous gates across heads and
layers, with neither all-zero (pure BA) nor all-at-cap (PM-dominated) saturation.

## Final Recommendation

Do **N23 first** because it directly answers whether the strong CAMIX inference gain becomes more
stable when the adapters train under that graph. N24 is the more promising architecture if N23
confirms that access to PM face geometry matters but fixed concatenation remains unstable.

Do not launch the previous N22 as currently written before these tests. It combines PAR+CAMIX,
ID-embedding injection, alternating loss, no ID loss, and a schedule change, so the new ablation data
would be hard to interpret. The present results support isolating CAMIX first and replacing the
fixed fusion second.

## Verification Completed

- Report generator compiled and rebuilt the PDF with 96/96 matches for all five runs.
- Existing metrics verified at 96 images and 96 detected faces per run.
- N23/N24 Hydra configs composed with the intended flags.
- Both shell scripts pass `bash -n` and are executable.
- Modified Python modules pass `py_compile`.
- Dual-gate processor smoke test produced the expected tensor shape, initialized all four test
  heads at 0.20, propagated a nonzero gate gradient, and included the gate in `state_dict()`.

## 10 Jul N24 Startup Fix

The first N24 server attempt failed on a normal PhotoMaker training phase with:

```text
Invalid branched batch: total=2, generation=2, reference=0
```

Cause: N24 sets `train_ba_all_steps=false`. The text-only and PhotoMaker-only timestep branches
called the UNet with a normal batch, but optimizer-owned branched processors were still attached and
expected a doubled `[generation, reference]` batch.

The training forward now temporarily selects the original processors for those two plain phases,
restores the exact same branched processor instances before leaving the forward, and adds a
zero-valued dependency on inactive trainable BA parameters. The dependency changes neither outputs
nor weights, but keeps backward/DDP valid on intentionally non-BA steps. Branched phases continue to
use the doubled batch and learned N24 gates.

Verified locally with processor-identity/output checks and a one-process DDP regression that
alternated plain and branched steps with `find_unused_parameters=false`; all backward/optimizer
steps completed and inactive BA gradients remained exactly zero.
