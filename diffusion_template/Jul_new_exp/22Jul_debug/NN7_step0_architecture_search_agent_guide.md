# Agent guide: NN7 step-zero architecture search

**Notebook:** `NN7_step0_architecture_search.ipynb`  
**Expected project location:** `diffusion_template/Jul_new_exp/22Jul_debug/`  
**Result root:** `diffusion_template/Jul_new_exp/22Jul_debug/experiments/`  
**Scope:** step-zero, untrained architecture search only  
**Core-code rule:** do not edit `src/`, `train.py`, or existing repository configs during this search

## 1. Objective

Find a branched-attention configuration that, at step zero:

1. produces a face visibly different from ordinary PhotoMaker;
2. moves the generated face toward the supplied reference identity;
3. preserves PhotoMaker target pose, expression, head placement, body and scene;
4. avoids N3a's duplicated landmarks, pasted face boundaries, hair/hand/glasses collisions and neck mismatch.

The target is not maximum distance from PhotoMaker. The target is a safe middle region:

```text
more reference ownership than NN5/NN6/NN7a default
less unaligned authority than N3a
```

## 2. Architectural principle to preserve

Stay as close as possible to the original branched-attention idea:

```text
target-coordinate face query
    attends
spatial reference face K/V
```

N3a proved that this route can make the generated face come from the reference. Its failure was the ownership policy around that route:

```text
full noised reference grid
+ reference-only face candidate
+ all self-attention layers
+ split branched cross-attention
+ no target fallback
+ no correspondence
+ no protected PhotoMaker epsilon
```

The notebook therefore starts from exact N3a as a positive control and introduces repairs one at a time.

## 3. What the notebook does

For the first four records of the repository's current `manual_val` dataset, it:

1. generates one ordinary PhotoMaker baseline per sample;
2. rebuilds each step-zero architecture from its Hydra base config;
3. applies only in-memory overrides or processor mutations;
4. generates BA images using the same prompt, seed, PhotoMaker ID, fixed RealVis bbox and schedule;
5. calculates:
   - exact RGB equality;
   - full, fixed-face and outside-face MAE versus PM;
   - boundary-ring MAE;
   - generated-face embedding cosine versus PM output;
   - similarity to the input reference;
   - reference-similarity gain over PM;
   - detected-face bbox IoU versus PM;
   - normalized five-landmark displacement;
   - face-detection rate;
6. saves the exact compact recipe, metrics and images;
7. updates a master leaderboard;
8. verifies that core repository code did not change.

## 4. Output structure

Every experiment receives a unique folder:

```text
Jul_new_exp/22Jul_debug/experiments/
  <timestamp>__<experiment_id>/
    experiment_spec.json
    compact_config.json
    architecture_signature.json
    metrics_per_sample.csv
    metrics_per_sample.json
    metrics_summary.json
    contact_sheet.png
    README.md
    images/
      sample_00_reference.png
      sample_00_PM0.png
      sample_00_BA.png
      sample_00_PM0_face.png
      sample_00_BA_face.png
      sample_00_diff.png
      ...
```

The root also contains:

```text
registry.jsonl
leaderboard.csv
<timestamp>__validation_samples.json
<timestamp>__PM0/
```

Do not place temporary code or outputs in `src/`.

## 5. Environment

Run from the PhotoMaker project environment on a free GPU:

```bash
cd /home/niko/rsrch/diffusion_template

source /home/niko/miniconda3/etc/profile.d/conda.sh
conda activate photomaker_NS

CUDA_VISIBLE_DEVICES=0 \
  python -m jupyter lab \
  Jul_new_exp/22Jul_debug/NN7_step0_architecture_search.ipynb
```

Confirm these paths in the first cell:

```python
REPO_ROOT
PHOTOMAKER_PATH
EXPERIMENTS_ROOT
```

The notebook defaults to:

```text
RealVisXL V4.0
BF16
rank 32
20 inference steps
PM start step 4
BA start step 6
first four manual-validation records
```

## 6. First run

Use:

```python
RUN_PROFILE = "quick"
SELECTED_EXPERIMENT_IDS = None
```

The quick suite includes:

- exact N3a;
- N3a with branched CA disabled;
- normalized ROI;
- up-block-only reference ownership;
- reference-only + protected output;
- core-ring ownership;
- 75% dual target/reference attention;
- confidence-gated reference residual;
- a minimal full-grid repaired candidate;
- NN7a_init v1/v2 controls;
- a stronger NN7a_init-v2 control.

Expect the full run to take time because every architecture is rebuilt.

For a short test:

```python
SELECTED_EXPERIMENT_IDS = [
    "n3a_exact",
    "n3a_roi_up_core_ring_anchor",
    "n3a_roi_up_dual75_anchor",
]
```

Restart the kernel before a materially different suite.

## 7. Decision criteria

The notebook's automatic labels are screening heuristics.

### Required visible activity

A useful candidate should satisfy approximately:

```text
median fixed-face MAE versus PM >= 0.012
```

and should not be byte-identical to PM.

Current weak NN7 paths can produce small numerical differences below this threshold without meaningful face ownership.

### Required safety

Prefer:

```text
face detection                 = 4/4
median outside-face MAE        <= 0.015
median normalized landmark RMSE <= 0.08
median detected-bbox IoU       >= 0.60
median face MAE                <= 0.12
```

The contact sheet remains mandatory. A face recognizer can score malformed faces.

### Required identity direction

Prefer:

```text
median(sim(BA, reference) - sim(PM, reference)) >= 0.003
positive reference gain on at least 3 of 4 samples
```

A face that only moves away from PM is not success.

### Automatic decisions

```text
too_close_to_photomaker
active_but_not_reference_improving
n3a_like_unsafe
promising_step0_candidate
invalid_face_detection
error
```

Sort first by decision, then inspect every component metric and image. Do not rely only on `screen_score`.

## 8. Highest-value architecture search, in order

### A. Repaired N3a core-ring

Start with:

```text
target Q -> evolving spatial reference K/V
normalized ROI
up blocks only
reference owns inner face core
target owns face boundary ring
branched CA off
PM epsilon outside core
```

Sweep:

```text
ba_sa_core_ratio = 0.50, 0.60, 0.68, 0.75, 0.82
ba_sa_roi_grid_size = 6, 8, 12, 16
BA start = 20%, 30%, 40%, 50% of steps
```

This is the closest strong repair to N3a.

### B. Dual target/reference candidate

Use:

```text
A_face = (1 - g) A_target + g A_reference
```

Sweep:

```text
g = 0.35, 0.50, 0.65, 0.75, 0.85
```

Keep:

```text
ROI
up-only
CA off
PM anchor outside core
```

This is the cleanest way to retain reference ownership while restoring target fallback.

### C. Confidence residual

Use the notebook's in-memory mutation:

```python
runtime_mutations={
    "legacy_confidence_gain": 0.25,
}
```

Sweep `0.25, 0.50, 0.75`.

This remains target-Q/reference-KV, but confidence entropy is only a provisional signal; wrong matches can be confidently sharp.

### D. Full-grid versus ROI

Retain one full-grid repaired candidate to determine whether ROI normalization removes useful face structure.

Do not promote full-grid solely because it changes the face more. It retains absolute reference layout and zero-token sinks.

### E. Layer and time scope

Compare:

```text
all SA layers
up blocks only
```

Then vary BA start time.

Down/mid reference ownership is the likely source of pose overwrite. Up-only should be the default for serious candidates.

### F. Cross-attention

Run exact N3a with branched CA once as a historical positive control.

For all serious candidates:

```text
disable_branched_ca = true
```

Historical experiments showed that trainable split CA caused color-mask and collage collapse.

### G. Strict routing

Test `strict_face_routing=true` only after a candidate is otherwise promising. It may prevent target-face leakage into the background branch, but it can also remove target continuity around the face.

## 9. Modern clean-patch controls

NN7a_init is useful as a comparison, but the main search should remain close to original BA.

Useful controls:

```text
alpha = 0.10, 0.25, 0.40, 0.60
cap = 0.20, 0.35, 0.50
local window = 5, 9, 15
sites = up1 or all up blocks
```

The completed earlier debug run showed that a strong clean-patch condition can change the face while still failing to improve identity. Therefore, more authority alone is not enough.

## 10. How to add an experiment

Append an `ExperimentSpec` in the notebook:

```python
spec(
    "n3a_roi_up_dual65_anchor",
    family="legacy_spatial_ba",
    role="authority_ablation",
    base_config=N3A_BASE,
    description="65% reference ownership with ROI, up-only and PM anchor.",
    overrides={
        "disable_branched_ca": True,
        "strict_face_routing": False,
        "model.ba_sa_ref_token_mode": "roi",
        "model.ba_sa_roi_grid_size": 8,
        "model.ba_sa_face_mode": "dual",
        "model.ba_sa_mix_init": 0.65,
        "model.ba_sa_ref_layer_scope": "up",
        "model.ba_target_core_erode_frac": 0.10,
        "model.ba_output_anchor_mode": "base_outside_core",
    },
)
```

Do not create a repository YAML for a step-zero screen. The notebook logs the exact recipe.

For a processor value that lacks a config field, use a supported in-memory mutation:

```python
runtime_mutations={
    "legacy_confidence_gain": 0.50,
    "legacy_scale": 1.0,
    "force_binary_masks": True,
}
```

If a new local experimental processor is required, define it inside the notebook or a file inside the experiment folder. Do not edit the production processor during search.

## 11. Promotion process

A four-image step-zero screen is only a filter.

Promote no more than three candidates and follow this sequence:

1. repeat at 50 inference steps on the same four cases;
2. add a fixed wrong-reference A→B control;
3. ensure the output moves toward B, not merely away from PM/A;
4. run the deterministic 24-case RealVis matrix;
5. inspect hard pose, occlusion, glasses, hats, hair and hands;
6. only then create a production training YAML and launcher.

A sensible final step-zero candidate should lie between:

```text
N3a:
    clearly reference-owned but geometrically unsafe

NN7a default:
    geometrically safe but nearly PhotoMaker-owned
```

## 12. What not to do

- Do not train a configuration just because its face MAE is large.
- Do not re-enable pose adaptation or CA face mixing as part of this screen.
- Do not change backbone and architecture together.
- Do not use SDXL bboxes for RealVis or vice versa.
- Do not edit core code to make one notebook result work.
- Do not overwrite experiment folders.
- Do not omit failed experiments from the registry.
- Do not push Git changes unless explicitly authorized.

## 13. Completion report for the next agent

For each iteration, report:

```text
experiment IDs run
Git commit
leaderboard path
top three candidates
PM face change
reference-similarity gain
landmark displacement
outside-face MAE
face detection count
visual failure notes
next single-axis ablation
```

Attach the relevant `metrics_summary.json` and `contact_sheet.png` files. Do not summarize only the best-looking sample.
