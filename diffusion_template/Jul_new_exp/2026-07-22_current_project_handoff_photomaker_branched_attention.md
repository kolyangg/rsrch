# PhotoMaker branched-attention research: current project handoff

**Date:** 22 July 2026  
**Repository:** `diffusion_template`  
**Current branch when written:** `main_clean`  
**Current commit when written:** `5e75690`  
**Current research frontier:** `NN7a_init-v2`

## 1. Purpose of the project

The project tries to improve identity fidelity in PhotoMaker V2 conditional
image generation by adding a branched-attention (BA) route.

The target behavior is:

1. preserve the composition, body pose, head location, expression compatibility,
   lighting, clothes, background, occluders, and rendering quality of ordinary
   PhotoMaker;
2. replace or refine the generated face so it is more similar to the supplied
   reference identity;
3. improve the face identity-similarity score without exploiting the face
   recognizer or creating visually broken images;
4. make the result causally depend on the reference: changing only the
   reference identity should move the output toward the changed identity.

The project is not trying to maximize arbitrary image difference from
PhotoMaker. A changed face is useful only if the change is identity-directed,
spatially aligned, and visually coherent.

The desired ownership policy is approximately:

```text
ordinary PhotoMaker
    owns pose, scene, composition, body, background and safe fallback

branched attention
    adds or takes bounded ownership of reference-specific facial identity

final output
    PhotoMaker outside the protected face core
    identity-improved BA result inside the face core
```

## 2. The central scientific problem

The project has repeatedly reached one of two bad extremes.

### 2.1 Strong BA changes the person but breaks geometry

The original N3a/full-spatial mechanism was powerful enough to make faces very
different from PhotoMaker. Target-face queries attended spatial K/V from a
separate reference stream at many U-Net attention sites.

This proved that BA can have real authority. It also produced severe failures:

- reference frontal geometry imposed on a target with different yaw or pose;
- duplicated, folded, displaced, or erased facial features;
- goggles, glasses, hair, hands, hats, and other occluders copied or crossed;
- stretched face/neck boundaries and pasted-face behavior;
- lighting, color, expression, hair, and background leakage from the reference;
- recognizer scores that could improve while visual anatomy became worse.

N3a and its guarded NN1 replay are therefore evidence of branch strength, not a
usable final architecture.

### 2.2 Safe residual BA preserves geometry but remains PhotoMaker-dominated

Later architectures protected an ordinary PhotoMaker prediction and applied a
bounded target-face residual. These runs usually kept the face attached,
preserved the body and scene, and avoided large artifacts. However, their faces
often remained almost identical to ordinary PhotoMaker at every checkpoint.

When amplified, these branches commonly changed generic face attributes—gaze,
eye opening, expression, skin smoothness, contrast, or color—without moving
identity toward a swapped reference.

This is the current bottleneck: obtain enough branch ownership to change
identity, while keeping the target geometry and PhotoMaker safety envelope.

## 3. Do not treat every historical “BA” run as the same mechanism

Three architecture lineages appear in the repository and reports.

### 3.1 Original full spatial BA: N3a and NN1

Core properties:

- one doubled `[target, reference]` U-Net call at active BA steps;
- `BranchedAttnProcessor` at all 70 self-attention sites;
- `BranchedCrossAttnProcessor` at all 70 cross-attention sites;
- target face Q attends spatial K/V from the evolving noised reference stream;
- the target half is returned as the absolute prediction;
- broad direct spatial ownership inside the target face mask.

This line produces large face changes but has an ill-posed correspondence
problem: a target coordinate and the same normalized coordinate in an
unaligned reference image do not necessarily describe the same facial part.

Relevant historical code snapshots cited by older reports include
`2157eada14824d14019e80f9416e6d736c837306` and the audited N3a baseline
`e42c96604ee73b8b073b3def268beead8c8af684`. Check the exact report and launcher
before assuming they are identical snapshots.

### 3.2 Protected packed-residual/PPR BA: NN2–NN6

Core properties:

- ordinary target self-attention or an ordinary PhotoMaker prediction remains
  the base;
- target queries attend packed reference ROI, learned-null, or compact identity
  memory;
- a connector and scalar gate produce a bounded residual;
- the residual is restricted to an eroded/feathered target face core;
- the final epsilon is anchored to ordinary PhotoMaker outside the core;
- branched CA is usually active/frozen or fully disabled rather than trained.

This line is much safer, but progressively moved away from the original N3a
ownership equation. NN4 and NN5 demonstrate that a branch can be active,
reference-sensitive, and well trained while still failing to transfer identity.

### 3.3 Current clean-reference takeover BA: NN7

NN7 tries to recover richer spatial identity evidence without restoring the
unsafe noised reference-U-Net as the active memory.

Current NN7a uses:

- a clean, bbox-cropped reference face;
- CLIP or PhotoMaker-V2 Perceiver-context patch tokens;
- target queries with local reference K/V attention;
- bbox-relative local `5 x 5` correspondence;
- a direct full-dimensional reference-versus-target candidate difference;
- selected `up_blocks.1.attn1` sites only;
- a bounded gate and RMS cap;
- the protected PhotoMaker epsilon outside the target core.

NN7a is still a simplified correspondence experiment. It does not yet contain
landmark/UV registration, semantic face-part masks, visibility reasoning, or an
occluder parser.

## 4. Current NN7a variants

### 4.1 NN7a: clean-patch direct takeover

Config:

```text
src/configs/one_id_ba_NN7a_clean_patch_takeover_up1.yaml
```

Key settings:

```yaml
ba_spatial_memory_mode: clean_clip_patches
ba_spatial_patch_dim: 1024
ba_spatial_local_window: 5
ba_spatial_mix_mode: direct_candidate_takeover
ba_site_policy: up_blocks1_attn1
ba_spatial_gate_max: 0.80
ba_gate_init_logit: -3.6635616461  # effective alpha about 0.02
ba_spatial_delta_rms_cap: 0.45
ba_total_delta_rms_cap: 0.45
ba_output_anchor_mode: base_outside_core
```

The clean patches are information-rich, but the original NN7a spatial K/V
starts from Xavier-random projections and the initial authority is small.

### 4.2 NN7a_init v1: partial PhotoMaker warm start

Config:

```text
src/configs/one_id_ba_NN7a_init.yaml
```

Changes from NN7a:

- uses 2048-D frozen PhotoMaker-V2 Perceiver-context patch tokens;
- initializes reference K/V from sibling `attn2.to_k/to_v`;
- trains rank-32 K/V LoRA deltas with zero-initialized LoRA B;
- starts with effective alpha about `0.05`.

Its limitation is that it is a hybrid operator:

```text
attn1 target Q
+ sibling attn2 reference K/V
+ attn1 output projection
```

It also caps first and gates second. With a `0.45` cap and `0.05` gate, the
final attention perturbation is bounded near `2.25%` of target-candidate RMS.
This explains why it still looks very close to PhotoMaker.

### 4.3 NN7a_init-v2: current recommended frontier

Config:

```text
src/configs/one_id_ba_NN7a_init_v2.yaml
```

V2 uses a complete frozen sibling-attn2 attention space:

```text
target hidden
  -> sibling attn2 Q

PMv2-context reference patches
  -> sibling attn2 K/V base
  -> trainable rank-32 K/V LoRA

local Q/K/V attention
  -> sibling attn2 output projection
  -> reference candidate

ordinary attn1 target candidate
  -> target candidate

output = target + cap(alpha * (reference - target))
```

Default V2 authority:

```text
alpha = 0.10
final local RMS cap = 0.20
gate position = before cap
```

Only the spatial K/V LoRA tensors and scalar gate are trainable. Sibling Q,
output projection, norms, PhotoMaker backbone, and branched CA remain frozen.

The architecture manifest distinguishes v1 and v2 checkpoints. Do not load a
v1 checkpoint into v2 or vice versa by bypassing the strict check.

## 5. Current forward path

For the normal 50-step validation schedule, the intended phase structure is:

```text
steps 0–9:   text-only SDXL
steps 10–14: ordinary PhotoMaker identity conditioning
steps 15–49: PhotoMaker plus active branched-attention route
```

At active BA steps:

1. prepare the target latent and reference inputs;
2. extract reference memory (reference U-Net ROI, compact identity tokens, or
   clean spatial patches depending on config);
3. install and configure the selected attention processors;
4. run the doubled or protected branched prediction;
5. obtain the target-half prediction and branch diagnostics;
6. combine with ordinary PhotoMaker using the target core mask:

```text
epsilon_final = epsilon_PM + M_core * (epsilon_BA - epsilon_PM)
```

The exact implementation differs by architecture toggle, so inspect the active
config rather than inferring behavior from a run name.

## 6. Training objective and causal supervision

Current NN7 configs inherit the NN5 counterfactual training scaffold.

For low enough timesteps, training constructs matched and wrong-reference rows
that share exactly the same:

- target latent;
- target noise;
- timestep;
- prompt and PhotoMaker target identity;
- target mask;
- reference-noise realization.

Only the reference identity changes. The decoded predictions are supervised
with absolute and directional identity losses, plus a boundary/ring penalty.

Relevant inherited settings include:

```yaml
ba_counterfactual_enabled: true
ba_counterfactual_max_timestep: 300
ba_counterfactual_abs_id_weight: 0.05
ba_counterfactual_direction_weight: 0.10
ba_counterfactual_direction_margin: 0.03
ba_counterfactual_ring_weight: 0.05
use_id_loss: true
id_loss_weight: 0.025
```

This supervision is necessary but has not yet been sufficient. NN5a showed
finite gradients and substantial branch movement while the swapped-reference
identity gain remained statistically indistinguishable from chance.

Do not interpret lower diffusion loss, larger parameter norms, a nonzero
residual, or larger face MAE as proof of identity learning.

## 7. Experiment history and transferable lessons

| Run family | Main result | Lesson |
|---|---|---|
| N3a / NN1a–c | Very different faces; mask/collage, duplicated and displaced features | Full spatial BA has authority but unsafe correspondence |
| N11 / NN1d | Freezing branched CA greatly improves color and stability | Keep CA forward semantics, but do not broadly train CA first |
| N13/N17 / NN1e–f | Identity loss can raise metrics while smoothing or breaking faces | Identity metrics and decoded losses can find shortcuts |
| N25–N33 | Compact identity memory and target residual are aligned and stable | Safe residual arbitration works, but compact memory/authority is weak |
| N34–N38 | “Identity owner” configs remain nearly PhotoMaker-identical | Architecture labels do not guarantee actual ownership |
| NN2/NN3 | Packed target-Q/reference-KV residual opens but remains weak or clipped | More training does not repair a non-causal residual direction |
| NN4 | Active and spatially safe, but swapped references do not transfer identity | Reference sensitivity is not identity causality |
| NN5a | Counterfactual loss increases visible movement; directional gain remains near chance | Objective is active, but memory/output remains semantically contaminated |
| NN5b/NN6 | Clean identity-token and factorized identity-lane experiments | Compact identity is safer, but risks returning to PhotoMaker-like weak control |
| NN7a | Restores rich clean spatial memory with bounded local takeover | Tests whether spatial capacity can return without N3a geometry leakage |
| NN7a_init-v2 | Full sibling-attn2 warm space and visible pre-cap authority | Current most principled step-zero/4k experiment |

The broad conclusion is:

```text
strong unaligned spatial ownership -> different but broken faces
safe compact residual ownership    -> aligned but PhotoMaker-like faces

needed next:
strong, identity-directed, geometrically registered local ownership
```

## 8. Latest NN7 debug-lab evidence

Notebook:

```text
Jul_new_exp/22Jul_debug/NN7_branched_attention_debug_lab_v3.ipynb
```

The executed notebook tested:

```text
CONFIG_NAME = one_id_ba_NN7a_init
CHECKPOINT_PATH = None
20 inference steps (fast schedule)
```

It did **not** test NN7a_init-v2 or a trained checkpoint.

Main findings:

- clean same-ID reference patch memories differ strongly;
- the processor responds to A1/A2 reference changes;
- response scales with alpha;
- processor changes are exactly zero outside the target core;
- the full pipeline preserves and displays a strong BA perturbation;
- normal v1 applied authority is only about `2.06%` per selected site;
- the strong positive control changes rendering substantially;
- the strong same-ID and wrong-ID outputs look similarly flatter/desaturated;
- the strong wrong-ID output does not move measurably toward the wrong identity.

Representative numbers:

| Measurement | Result |
|---|---:|
| Same-ID patch A1/A2 RMS difference | 0.441 |
| Processor A2−A1 RMS at alpha about 0.05 | 0.0040 |
| Processor A2−A1 RMS at alpha 0.50 | 0.0396 |
| Outside-core processor change | exactly 0 |
| Current full-pipeline applied ratio | about 0.0206 |
| Strong-control applied ratio | about 0.457 |
| Current A1/A2 final-image MAE | 0.00388 |
| Strong A2 versus PM image MAE | 0.0293 |

Interpretation:

1. reference extraction is not dead;
2. processor wiring is not dead;
3. the target core mask and output anchor work locally;
4. v1 is too conservative at normal authority;
5. high authority exposes a generic appearance direction more clearly than an
   identity direction;
6. V2 must be tested before deciding that the warm-start idea has failed.

One control caveat: `BA_off_A1` differs slightly from separately generated
`PM0`, because one uses the doubled branched execution path and the other uses
the ordinary path. Use zero-scale BA as the baseline for branch attribution,
and compare repeated zero-scale A1/A2 runs to detect real reference leakage.

The notebook generated an extreme face close-up whose core occupied about 64%
of latent space. It is useful for branch activity but not for judging body
alignment, background preservation, seams, or normal face-size behavior.

## 9. Current unresolved issues

### 9.1 Identity direction remains unproven

Recent branches can distinguish reference images internally, but a changed
reference does not consistently move the output toward the changed identity.
This is the main scientific failure criterion.

### 9.2 PhotoMaker remains the effective identity owner

Normal validation images in NN3–NN6 and many later runs remain close to the
step-zero PhotoMaker face. A branch may be nonzero while still being too weak,
clipped, zero-initialized, or semantically unable to compete with target
PhotoMaker conditioning.

### 9.3 Spatial memory mixes identity with nuisance information

Reference patch features carry pose, lighting, crop, expression, hair,
occluders, and style. The latest notebook's one-example token cosine was not
more same-identity selective than wrong-identity selective. This needs a larger
representation audit, not a conclusion from one pair.

### 9.4 Bbox-relative correspondence is only approximate

The same normalized face-crop coordinate can correspond to different anatomy
under yaw, expression, occlusion, or crop errors. NN7a's local window improves
over global copying but does not solve semantic registration.

### 9.5 Stronger residuals can amplify the wrong signal

NN4/NN5 and the NN7 debug strong control show that raising scale, cap, gate, or
site count can expose generic expression/chroma changes rather than identity.
Do not treat branch strength as the default fix.

### 9.6 Validation controls can be misleading

- matched PhotoMaker and matched BA both receive the same identity, so similar
  outputs do not prove a dead branch;
- comparing BA to a separately executed PM path includes execution-path
  numerical differences;
- a fixed face metric can reward distorted identity-correlated texture;
- RealVis manual target bboxes must not be used with a different validation
  backbone;
- large face bboxes do not test body and boundary preservation well.

### 9.7 “Branched attention” is overloaded terminology

Before analyzing a run, establish:

- installed processor class and count;
- active SA and CA sites;
- memory source;
- whether the branch is absolute or residual;
- gate initialization and location relative to the cap;
- output ownership/anchor;
- training versus inference denoising schedule;
- trainable parameter manifest;
- checkpoint architecture manifest.

## 10. Key code structure

### Entry point, config and training

| Path | Role |
|---|---|
| `train.py` | Hydra entry point, model/trainer construction, resume and validation-only dispatch |
| `src/configs/one_id_ba_NN*.yaml` | Architecture inheritance and experiment toggles |
| `src/configs/datasets/all_datasets.yaml` | `cosmic_large_neb` and 96-image `manual_val` definitions |
| `src/configs/dataloaders/all_dataloaders.yaml` | Train and validation DataLoader defaults |
| `src/trainer/base_trainer.py` | Training/validation loops, checkpoint loading and PPR preflight |
| `src/trainer/sdxl_trainers.py` | Diffusion/auxiliary loss composition, BA metric gathering and evaluation routing |

### Model and attention implementation

| Path | Role |
|---|---|
| `src/model/photomaker_branched/lora2.py` | Main `PhotomakerBranchedLora` model, architecture flags, reference preparation, training forward, causal A/B rows and decoded identity losses |
| `src/model/photomaker_branched/lora2_helpers.py` | Processor installation, strict trainable manifest, spatial/identity memory preparation and branched forward wrapper |
| `src/model/photomaker_branched/packed_residual_attn_processor.py` | Current protected packed-residual processor, identity/spatial lanes, local clean-patch attention, gates, caps and NN7 warm operators |
| `src/model/photomaker_branched/attn_processor_cleanest.py` | Historical `BranchedAttnProcessor` and `BranchedCrossAttnProcessor` used by the full N3a/NN1 mechanism and legacy paths |
| `src/model/photomaker_branched/branched_runtime.py` | Runtime processor patching, mask/token routing, doubled target/reference U-Net logic and processor runtime scales |
| `src/model/photomaker_branched/model_v2_NS.py` | PhotoMaker V2 ID encoder, QFormer/Perceiver identity tokens, and extraction of raw CLIP or PMv2 Perceiver-context patch tokens |
| `src/loss/id_loss.py` | Differentiable absolute and counterfactual directional identity losses |
| `src/loss/diffusion_loss.py` | Diffusion and core-normalized loss paths |

### Inference and diagnostics

| Path | Role |
|---|---|
| `src/pipelines/photomaker_branched_clean.py` | Main PhotoMaker branched pipeline and denoising integration |
| `src/pipelines/br_pipeline_helpers.py` | Schedule switching, clean patch extraction, BA step, PhotoMaker/BA output anchor and tensor diagnostics |
| `src/trainer/ppr_reference_noise.py` | Five-condition PM/reference/noise causal matrix, metrics, integrity assertions and reports |
| `src/trainer/ppr_diagnostic.py` | Diagnostic generation helpers and randomness fingerprints |
| `src/trainer/ppr_scale_sweep.py` | Checkpoint residual/authority sweeps |
| `ba_architecture_explorer/index.html` | Interactive visual comparison of historical and proposed architectures |
| `infer_tools/pdf_full_val.py` | PDF comparison generation for downloaded validation images |
| `comet_utils/` | Comet metric/image download and export utilities |

Avoid editing files under `_old*` or `_backup` unless deliberately reconstructing
historical behavior. The active implementation is in the non-archived paths.

## 11. Data, environment and server conventions

### Primary server

Common remote paths:

```text
project:       /home/niko/rsrch/diffusion_template
environment:   /home/niko/miniconda3/envs/photomaker_NS
PhotoMaker:    /home/niko/models/PhotoMaker-V2/photomaker-v2.bin
dataset JSON:  /home/niko/datasets/gathered_data_cosmic_large_filtered.json
images:        /home/niko/datasets/LAION-5B-Filtered-Large-Faces/laion1B-nolang
```

The training dataset name for current primary-server launchers is:

```text
cosmic_large_neb
```

### Local analysis checkout

The local project commonly lives at:

```text
/home/kolyangg/rsrch/diffusion_template
```

The local Conda environment available for focused tests is commonly named
`photomaker`; remote launchers use `photomaker_NS`.

### Secondary/NFS server

Older combined-job YAMLs use paths under:

```text
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/diffusion_template
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS
```

Do not copy primary-server paths blindly into NFS YAMLs. Follow an existing
recent one-/two-GPU YAML and its `.env` conventions.

## 12. Validation protocol

### Ordinary in-run validation

- validation backbone: `SG161222/RealVisXL_V4.0`;
- fixed 96 images/prompts/seeds;
- fixed reference bboxes from `ref_bboxes.json`;
- fixed generation bboxes from `pm96_bboxes_new.json`;
- normal validation batch size: 12 per process where memory permits;
- validation every 2,000 optimizer steps;
- full 96-image step-zero validation is commonly enabled.

The bbox files were measured for RealVis. Always keep RealVis validation unless
new boxes are explicitly generated for another backbone.

### Five-condition causal matrix

```text
PM0   ordinary PhotoMaker / BA scale zero
R1N1  matched reference R1, reference noise N1
R2N1  swapped identity R2, same noise N1
R1N2  matched reference R1, alternate noise N2
R2N2  swapped identity R2, same alternate noise N2
```

This separates reference-image effects from nuisance reference-noise effects.
For current 4k approval runs, a deterministic 24/96 subset is used:

```text
subset seed: 20260722
indices: 5, 6, 8, 10, 14, 17, 18, 22, 31, 35, 36, 47,
         51, 52, 53, 64, 70, 72, 74, 77, 81, 84, 89, 94
```

### Evidence hierarchy

Use evidence in this order:

1. enlarged fixed face crops and complete images;
2. reference-swap direction at fixed target/noise;
3. pose, attachment, occluder, seam and background stability;
4. reference-image effect versus reference-noise effect;
5. identity similarity and text similarity;
6. training loss and parameter norms.

The decisive identity statistic is not only original-ID similarity. For an R1
to R2 swap, measure whether the output becomes more similar to R2 and less
similar to R1, consistently across both noise seeds.

## 13. Immediate recommended next steps

### Step 1: run NN7a_init-v2 step-zero alpha sweep

```bash
cd /home/niko/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0 \
  BATCH_SIZE=12 \
  SUBSET_SIZE=24 \
  bash jul_serv_runs/start_ba_NN7a_init_v2_step0_alpha_sweep_1gpu.sh
```

This evaluates alpha `0.05`, `0.10`, and `0.20`, all with final cap `0.20`, on
the same deterministic 24-image RealVis subset.

### Step 2: rerun the NN7 debug notebook with V2

Set:

```python
CONFIG_NAME = "one_id_ba_NN7a_init_v2"
CHECKPOINT_PATH = None
```

Add or inspect these controls:

- repeated `BA_off_A1` for determinism;
- `BA_off_A1` versus `BA_off_A2` to detect zero-scale reference leakage;
- current-authority wrong-ID reference, not only the strong condition;
- direct pairwise metrics relative to the zero-scale branched baseline;
- shuffled and zero patch controls;
- at least one normal manual-validation sample with a smaller face bbox;
- one exact 50-step validation-schedule rerun.

### Step 3: train V2 only if the step-zero route is semantically promising

Default 4k train-and-diagnose launcher:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_init_v2_train_then_diagnose_1gpu.sh
```

Use the 2k checkpoint as an early gate even if the job budget is 4k or longer.
Continue only if reference swaps become identity-directed while geometry stays
stable.

### V2 approval criteria

Proceed if:

- R1/R2 patch memories and candidate tensors differ as expected;
- zero-scale A1/A2 controls remain invariant;
- reference-image effects clearly exceed reference-noise effects;
- swapped-reference identity direction is positive and stable across N1/N2;
- alpha gives visible local authority without desaturation, part crossing,
  seams, face displacement, or body/background changes;
- the applied ratio is meaningful rather than approximately 2%;
- checkpoints restore with the exact V2 architecture manifest.

Stop or redesign if a strong V2 branch again produces the same generic grey,
smoothed, expression-shifted face for both same-ID and wrong-ID references.

## 14. If NN7a_init-v2 fails

Do not respond by only raising the gate, cap, residual scale, training duration,
GPU count, or number of sites. Those changes amplify the existing direction.

The next architecture should retain the BA core but separate identity from
spatial nuisance more explicitly. A promising structure is:

```text
identity lane
    clean PhotoMaker/recognition identity tokens
    global or part-level identity evidence
    dedicated gate and cap

spatial lane
    clean reference patch grid
    local pose/texture/expression evidence
    target-geometry correspondence and confidence
    dedicated gate and cap

fusion
    target self-attention fallback
    shared final cap
    semantic/occluder restrictions
    exact PhotoMaker epsilon outside the core
```

Useful isolated changes after a failed V2 test include:

- add low-rank trainable adapters to sibling Q/output as well as K/V;
- condition reference values on an explicit ArcFace/PhotoMaker identity token;
- compare raw CLIP and PMv2-context patch identity separability over many
  same/different pairs;
- add semantic eye/nose/mouth/contour ownership zones;
- add landmark- or UV-registered local correspondence;
- preserve target-owned hair, mouth cavity, eye whites, occluders, and face
  boundary;
- supervise correspondence confidence and reference-swap identity direction.

These should be architecture ablations, not simultaneous wholesale changes.

## 15. Common pitfalls for a new agent

1. **Do not infer branch activity from visual similarity alone.** Inspect
   processor counts, trainable keys, checkpoint manifests, candidate hashes,
   applied ratios and swapped-reference effects.
2. **Do not infer identity success from face MAE.** Generic color/expression
   changes can produce large differences.
3. **Do not use the fixed RealVis bboxes with SDXL validation.** Either retain
   RealVis or regenerate target boxes.
4. **Do not load an architecture-mismatched checkpoint with `strict=False`.**
   Current manifests intentionally reject incompatible variants.
5. **Do not compare only matched BA to PM.** Use R1/R2 and N1/N2 causal controls.
6. **Do not make an amplified diagnostic scale the training default.** It is a
   route-activity test, not evidence of a useful operating point.
7. **Do not assume old run names map to current source.** Use the recorded commit,
   config, launcher, log and architecture manifest.
8. **Do not edit archived `_old*` files for current experiments.** Confirm the
   imported active path first.
9. **Do not ignore target/reference mask dimensions.** Verify masks at every
   selected attention resolution and reject invalid boxes rather than using a
   full-image fallback.
10. **Do not train longer after a failed 2k/4k causal gate.** Several historical
    runs show that more training strengthens the wrong behavior.

## 16. Suggested reading order

1. `Jul_new_exp/2026-07-17_post_N3a_experiment_summary.md`
2. `Jul_new_exp/2026-07-17_NN1a_NN1f_results_and_NN2_architecture_plan.md`
3. `Jul_new_exp/2026-07-18_branched_attention_photomaker_research_review_brief.md`
4. `Jul_new_exp/2026-07-21_NN4_results_analysis_and_next_architecture.md`
5. `Jul_new_exp/2026-07-22_NN5a_4k_results_analysis_and_next_training_recommendation.md`
6. `Jul_new_exp/2026-07-22_N3a_vs_NN6a_and_NN7_architecture_proposal.md`
7. `Jul_new_exp/2026-07-22_NN7a_implementation_and_primary_launch.md`
8. `Jul_new_exp/2026-07-22_NN7a_init_audit_and_visible_warm_start_fixes.md`
9. `Jul_new_exp/2026-07-22_NN7a_init_v2_visible_warm_start_implementation.md`
10. `Jul_new_exp/22Jul_debug/NN7_branched_attention_debug_lab_v3.ipynb`

For older N29–N38 evidence, use:

- `debug_04Jul/Codex_16Jul_project_handoff_fresh_architecture_review.md`;
- `debug_04Jul/Codex_16Jul_N31_N32_N33_visual_architecture_analysis.md`;
- `debug_04Jul/Codex_17Jul_N36_N37_N38_results_and_failure_analysis.md`.

For visual architecture comparison, open:

```text
ba_architecture_explorer/index.html
```

## 17. One-paragraph handoff summary

This repository tries to improve PhotoMaker V2 identity fidelity with branched
attention. The old N3a/full-spatial path proved that target-face queries can use
reference K/V strongly enough to produce a different person, but it copied
unaligned reference pose, parts, lighting and occluders, causing severe face
artifacts. Later protected PPR/residual runs fixed alignment and preserved the
PhotoMaker scene, but usually left the face PhotoMaker-dominated or learned
generic gaze/texture/expression changes rather than reference identity. Current
NN7 restores rich clean spatial patch memory and local target/reference
attention while retaining target fallback, a face-core mask and exact
PhotoMaker epsilon outside the core. NN7a_init v1 is confirmed active but only
about 2% authoritative and not clearly identity-directed. The immediate task is
to evaluate NN7a_init-v2's complete sibling-attn2 warm operator at step zero,
then train only if R1/R2 causal tests show movement toward the swapped identity
without N3a-like geometry failures.
