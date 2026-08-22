# From the 2 June clean baseline to the E13-family experiments

**Date:** 18 August 2026

**Branch:** `clean`

**Evidence cutoff:** clean branch commit `7b7579bf91f378321103cf1a9d367f6906e7e0e1`

**Baseline source:** `main_clean` commit
`2157eada14824d14019e80f9416e6d736c837306`, `code clean-up - restore 1 ref
only`, 2 June 2026 21:58 BST

**Historical baseline recipe:** `cosm_new1_vast`, immutable Comet key
[`b7602f92bca54ba5aa0c189192d17165`](https://www.comet.com/nikolay-2104/rsrch-30oct/b7602f92bca54ba5aa0c189192d17165),
started 3 July 2026 13:37:53 BST

**Scope:** code lineage, architecture formulas, dataset contracts, validation
repairs, and execution-only training improvements; no job was launched and no
new model-quality measurement was made for this document.

## Executive conclusion

The correct historical base is now defined by two inseparable records: the 2
June `main_clean` source snapshot and the exact `cosm_new1_vast` configuration
recorded by Comet. No later committed `main_clean` revision existed when that
run started. The run instantiated the June single-reference mechanism with
rank-32 `noise_and_ref` LoRA branches, branched SA and CA, hard masks,
`pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and the original
`CosmicLargeTrain` policy. **[code] [measured]**

That source/config pair is not a safe experiment contract. Its processor
installer catches every exception and continues; its optimizer then accepts
every parameter still marked trainable; its checkpoint saves the new generic
adapter and selected processor state but omits the trained PhotoMaker
`default` adapter; and alternate-base validation copies processor state
without an explicit, checked semantic contract. **[code] [report]**

The clean line therefore has two kinds of changes, in this order:

1. **Make the 2 June mechanism auditable and reproducible.** Fail closed on
   processor installation, prove exact optimizer ownership, save every trained
   tensor in schema-v2 checkpoints, make validation shadow/copy behavior
   explicit, seal the fixed-96 inputs, and remove output-irrelevant slow paths.
2. **Add named scientific experiments as small deltas.** E13 is the first
   complete architecture on that substrate. BC_E13 changes only the dataset.
   CL14 keeps E13 attention, adopts the corrected Cosmic Large data policy, and
   constructs a two-cell training mask. CL14_CA, CL18, CL19, and CL20 are
   separate children of CL14. CL23 is a child of CL19; CL27 is a child of
   CL23.

This is a dependency graph, not one linear sequence:

```text
2 June source + cosm_new1_vast historical recipe
+-- correctness + checkpoint + validation + execution substrate
    +-- E13: rank-128 hard BA + effective generic/default co-adaptation
        +-- BC_E13: E13 on sealed BigCelebs
        +-- corrected Cosmic Large input policy
            +-- CL14: E13 + two-cell target training-mask construction
                +-- CL14_CA: bounded residual ID-token CA
                +-- CL18: training-only cross-view consistency
                +-- CL19: full-query cosine router
                |   +-- CL23: fixed temporal-frequency route
                |       +-- CL27: training-only frequency-surface objective
                +-- CL20: deterministic Cosmic/BigCelebs curriculum
```

The previously existing implementation ledger is
[`2026-08-10_e13_family_clean_implementation.md`](2026-08-10_e13_family_clean_implementation.md).
It is the original detailed record of changes against the 2 June base. The
formula reference is
[`2026-08-13_e13_family_architecture_reference.md`](2026-08-13_e13_family_architecture_reference.md).
This new document consolidates both into a baseline-first, dependency-ordered
reference and incorporates CL23/CL27.

## 1. Exact baseline: 2 June source plus `cosm_new1_vast` config

The latest 2 June commit on `main_clean` is uniquely:

```text
2157eada14824d14019e80f9416e6d736c837306
2026-06-02T21:58:25+01:00
code clean-up - restore 1 ref only
```

### 1.1 Why the run is assigned to the 2 June commit

The Comet metadata does not contain a Git SHA, so exact source attribution is
not cryptographically sealed. The independent evidence nevertheless makes
`2157eada...` the only supported committed source assignment. **[measured]
[code]**

| Evidence | Observed value | Implication |
|---|---|---|
| Comet start time | `2026-07-03 13:37:53.110 BST` | run began from the July checkout before any July commit existed |
| preceding `main_clean` commit | `2157eada...`, 2 June 21:58 BST | last committed source available at run start |
| next `main_clean` commit | `2df2b895...`, 3 July 22:54 BST | committed 9 h 16 min after the run started |
| historical launcher history | no change after `144843f...`, 26 April | launch script at `2157eada...` is byte-identical to the later `main_clean` copy |
| Comet parameters | exact match to `start_ba_cosm_new1_vast.sh` | identifies the launcher/config used |
| 3 July worklog | later F1/B1/jitter/processor fixes described as uncommitted responses to the old run | later July fixes are descendants, not the run's starting architecture |

Therefore this document uses `main_clean@2157eada...` as the historical source
commit. A dirty, unrecorded source edit at launch cannot be absolutely excluded
because this old run predates automatic `comet_experiment.json` Git capture;
no evidence found supports such an edit. A later commit must not be substituted
merely because the run finished after that commit. **[measured]**

### 1.2 Exact historical configuration from Comet

The immutable Comet experiment is `cosm_new1_vast` in project
`nikolay-2104/rsrch-30oct`, key
[`b7602f92bca54ba5aa0c189192d17165`](https://www.comet.com/nikolay-2104/rsrch-30oct/b7602f92bca54ba5aa0c189192d17165).
It reached logged `curr_step=28000`. The table below uses Comet's 336 recorded
parameters rather than reconstructing values from present-day defaults.
**[measured]**

| Contract area | Historical value |
|---|---|
| Hydra root / dataset group | `one_id_09Feb_testing` / `all_datasets` |
| training dataset | `cosmic_large_vast` -> `src.datasets.cosmic.CosmicLargeTrain` |
| reference policy | one reference; `const_ref=true`; `origtarget_genref=true`; `ref_similar=false`; `train_on_separate_image=false` |
| image policy | 1024 target; `upscale_to_1024=true`; `min_face_res=192`; no target 256 crop; no reference crop |
| branch weights | `noise_and_ref`; new weights are LoRA; model rank 32 |
| attention training | branched SA plus CA; `train_branched_ca_lora=true`; `train_ba_only=true`; `non_ba_train=false`; `train_ba_all_steps=true` |
| routing safeguards | `pose_adapt_ratio=0`; `ca_mixing_for_face=false`; hard mask (`expansion=1.0`, `softness=0`) |
| denoising schedule | PhotoMaker/merge at step 10; BA at step 15; DDIM-style configured 50-step validation |
| objective | `masked_alternating`; masked step period 2 |
| optimizer / LR | AdamW; LoRA LR `1e-4`; weight decay 0; no grad clipping; linear scheduler warmup 2,000 |
| execution | bf16; batch 2; 12 workers; no gradient accumulation; epoch length 2,000 |
| validation | `manual_val_two`, limit 24, batch 4, one image/prompt, CFG 5, RealVisXL V4.0 |
| boxes and metrics | automatic boxes once, forced first-box log, `all_metrics`, seed 0 |

This is the baseline *run recipe*. Raw constructor defaults that the launcher
overrode are not treated as historical run behavior.

The example file named by the user is
[`lora2.py` at the current clean branch](../../src/model/photomaker_branched/lora2.py).
For an exact historical comparison, use:

```bash
git show 2157eada14824d14019e80f9416e6d736c837306:\
diffusion_template/src/model/photomaker_branched/lora2.py
```

### 1.3 Baseline attention concept

Let $H_t\in\mathbb{R}^{B\times L\times d}$ be target/noisy latent tokens,
$H_r\in\mathbb{R}^{B\times L\times d}$ a separate reference latent, and
$M_t,M_r$ their target/reference face masks. The U-Net receives the doubled
batch

$$
H=[H_t;H_r].
$$

With scaled dot-product attention

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V,
$$

the central branched-SA idea is a target-face message whose queries come from
the target and whose keys/values come from the reference:

$$
F=\operatorname{Attn}
\left(W_q^tH_t\odot M_t,
W_k^r(H_r\odot M_r),
W_v^r(H_r\odot M_r)\right).
$$

The target background is produced from target features, and a mask selects
between the messages. This explicit target-Q/reference-KV route is the project
invariant retained by every supported experiment. **[code]**

For the historical `noise_and_ref` setting, target/noise and reference paths
receive separate rank-32 LoRA deltas, schematically

$$
W_x^t=W_x+\Delta W_x^{noise},\qquad
W_x^r=W_x+\Delta W_x^{ref},\quad x\in\{q,k,v,o\}.
$$

Both branched self-attention and branched cross-attention were enabled. The
hard target/reference masks and `train_ba_all_steps=true` were active, while
the eligible reference-routing safeguards were already explicitly set to
`pose_adapt_ratio=0` and `ca_mixing_for_face=false`. **[code] [measured]**

### 1.4 Source defaults versus the instantiated run

The 2 June constructor exposes many independent switches. Important defaults
include `pose_adapt_ratio=0.25`, `ca_mixing_for_face=true`, shared/full branch
weights, branched CA training enabled, and `train_ba_all_steps=false`.
`cosm_new1_vast` overrode these to the values in Section 1.2, including the
correct zero pose-adaptation and disabled face CA mixing. The problem was not
that every run value came from unsafe defaults; it was that neither source nor
launcher enforced a fail-closed, self-describing contract. The later clean
family keeps one reference and those two safeguards, disables legacy branched
CA, and pins every architecture/schedule field in named Hydra leaves.
**[code] [measured]**

### 1.5 Baseline failure modes

The six baseline failure/repair pairs are:

- **Fail-open installation.** The 2 June
  `install_branched_processors_for_training()` wrapped installation, parameter
  iteration, and selection in one broad `try/except` and only printed the
  exception. A plain `AttnProcessor2_0` could raise during `.parameters()` and
  training would continue with the wrong `requires_grad` set. The E13 profile
  makes installation fatal and asserts the exact processor/name map before
  optimizer creation.
- **Broad optimizer ownership.** `get_trainable_params()` returned every U-Net
  parameter still marked trainable in one group, so a failed BA-only
  reconfiguration silently trained broad generic/default adapters. The repair
  freezes all parameters first, enables the exact 840/700/700 allowlist, and
  proves disjoint optimizer membership.
- **Incomplete checkpoints.** `get_state_dict()` saved `lora_adapter` plus
  selected processor parameters, but omitted the trained PhotoMaker `default`
  adapter. Schema-v2 instead saves every exact trainable name, tensor, shape,
  dtype, route, rank, and processor hash.
- **Permissive validation loads.** Processor loads used `strict=False`, and a
  boolean optionally copied processor state. The clean contract names
  `legacy_full_copy`, requires exactly 70 stateful processors, and makes any
  manifest mismatch fatal.
- **Wrong-person identity selection.** `faces[0]` could bind a bystander. New
  reporting uses bbox-owned subject-v2 selection, while historical replay
  remains explicitly available.
- **Avoidable frozen work.** Conditioning was prepared sample by sample and
  text-only embeddings were computed even when unreachable. The named E13
  performance profile batches conditioning and skips only proven-dead work.

The later architecture audit observed the installer failure in historical run
logs and measured the fail-open state at about `171.29M` trainable parameters,
rather than the intended BA-only `31.95M`. That observation proves the failure
was not merely theoretical. It does not make those historical runs useless; it
means they must be described as joint co-adaptation rather than clean BA-only
experiments. **[report]**

## 2. Repair layer: changes that precede architecture claims

These repairs are part of correctness or execution. They are not themselves a
claim that a new attention equation is better.

### 2.1 Fail-closed processor installation and exact ownership

[`e13_contract.py`](../../src/model/photomaker_branched/e13_contract.py) is the
single ownership switchboard. It freezes the U-Net and then enables exactly:

| Role | Tensors | Parameters | Rank |
|---|---:|---:|---:|
| hard spatial BA | 840 | 127,795,200 | 128 |
| effective generic adapter | 700 | 30,474,240 | 32 |
| effective PhotoMaker `default` adapter | 700 | 60,948,480 | 64 |
| **E13-family total** | **2,240** | **219,217,920** | - |

Every expected name must exist, every trainable must have exactly one optimizer
owner, no unexpected base-U-Net parameter may train, and the semantic processor
list is hashed. The installation path is in
[`branched_runtime.py`](../../src/model/photomaker_branched/branched_runtime.py)
and
[`lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py).
**[code]**

### 2.2 Complete checkpoint and validation semantics

Schema-v2 checkpoints save the complete trainable U-Net state and an
architecture manifest. Loads fail on missing/unexpected names, shape or dtype
drift, rank/scope drift, routing drift, or processor-map drift. The legacy
schema remains only for labelled historical recovery. **[code]**

E13 deliberately trains the pretrained PhotoMaker `default` adapter as part of
co-adaptation, but validation restores the 700-tensor pretrained snapshot in
the temporary RealVis model. This is the E13 **shadow** mechanism: the trained
default path influences optimization but is not persisted into the validation
forward. The 70 stateful attention processors are copied under an explicit
`legacy_full_copy` contract. The implementation is in
[`base_trainer.py`](../../src/trainer/base_trainer.py). **[code]**

### 2.3 Fixed validation inputs and subject ownership

All recipes retain the fixed 96-image `manual_val` panel, one image per item,
RealVisXL V4.0, DDIM 50, CFG 5, batch 12, seed/prompt/reference stability,
PhotoMaker onset 10, and BA onset 15. E13/BC_E13 use active generation-box
cache `4db6344d...`; the CL14 family uses `b33cf026...`; both preserve the
canonical Jensen manual override from `a39645e2...`. **[code]**

The later subject-v2 repair is a measurement/validation fix, not a model
architecture change. `bbox_overlap_v2` binds the identity vector to the face
overlapping the declared reference bbox, and `IDSimMaskMatched` scores the
generated face owned by the exact BA generation box. Historical
best-over-any-face ID remains as an audit metric. See
[`face_subject_selector.py`](../../src/face_subject_selector.py),
[`id_sim_metric.py`](../../src/metrics/id_sim_metric.py), and the isolated
validation wrapper
[`photomaker_branched_subject_v2.py`](../../src/pipelines/photomaker_branched_subject_v2.py).
**[code]**

### 2.4 Pipeline corrections kept separate from datasets

The generic pipeline receives only final reference pixels, final reference
bbox, target generation bbox, prompt, seed, and optional precomputed identity
embedding. It does not inspect Cosmic or BigCelebs manifests. It prepares one
spatial reference latent/mask/noise state per identity batch and reuses it
through the denoising call. The sealed files are
[`photomaker_branched_clean.py`](../../src/pipelines/photomaker_branched_clean.py)
and
[`br_pipeline_helpers.py`](../../src/pipelines/br_pipeline_helpers.py).
**[code]**

This separation matters: a crop, caption, flip, curriculum, or manifest change
belongs to a dataset; target-Q/reference-KV routing and denoising activation
belong to the model/pipeline.

### 2.5 Training-efficiency layer

The clean family preserves the scientific computation while avoiding work that
cannot affect the selected loss/output:

| Improvement | Removed work | Scientific status |
|---|---|---|
| batch frozen text/PhotoMaker/VAE conditioning | repeated per-sample encoder calls | historical E13 setting; semantically equivalent, not bit-identical to scalar bf16 GEMMs |
| skip unreachable text-only conditioning | text encode and `timestep.item()` when BA trains at all timesteps | output-neutral for this route |
| disable ineffective diverse-pair LRU | lookup/bookkeeping with almost no reuse | data/runtime only |
| cache prepared masks within one forward | repeated resize/preparation | exact tensor reuse |
| disable unconsumed branch-debug outputs and zero-valued post-backward touches | tensor creation and full parameter scans | output-neutral |
| request gradient norms only when logged/consumed | unnecessary reductions | telemetry-only |
| defer PyIQA until successful training | model construction and scoring inside the optimizer lifecycle | validation scoring timing only |
| asynchronous CUDA and verified ORT-GPU | blocking execution and silent CPU face analysis | runtime only; provider is logged and fail-closed |
| cache `unet.attn_processors` once per collector | repeated recursive full-U-Net dictionary rebuilds inside layer loops | execution-only |
| return early from disabled collectors | processor lookup and unused 1024-square mask allocation | execution-only |
| keep CL27 eligibility on device | Python-bool CUDA synchronization per processor | execution-only |
| disable undeclared full-activation BA telemetry | detached activation retention/collection | telemetry-only |

The original E13-family benchmarking measured batched conditioning at roughly
`0.9 s/step` versus roughly `5 s/step` for the scalar path on its fixture, with
about `0.074%` one-step loss difference from bf16 batching order. The separate
16 August processor-map fix measured current CL14 falling from `3.56` to
`2.06 s/it` while preserving the exact `2,240 / 219,217,920` ownership
contract. These are historical measurements, not fresh benchmarks on this
clean branch. **[report]**

The optimized collector pattern is visible in
[`lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py):
resolve `unet.attn_processors` once outside each per-layer loop, and do not
resolve it at all when the collector is disabled.

## 3. Dataset repair layer

Dataset improvements finish before the generic model/pipeline input. They
should never be interpreted as an attention-formula change.

### 3.1 Large Dataset for E13

[`LargeDatasetTrain`](../../src/datasets/large_dataset.py) uses 47,500 adjusted
1024-pixel scenes over 2,561 identities. A target is paired with a distinct
same-identity reference; target flips propagate to the target bbox; the
reference is not independently mirrored. Singleton self-reference fails
closed. **[code]**

### 3.2 Sealed BigCelebs for BC_E13

[`BigCelebsTrain`](../../src/datasets/big_celebs.py) keeps the E13 model and
requires the sealed v2 release: 349,348 images / 68,648 identities, exact
manifest fields, a distinct same-ID reference, valid bboxes, minimum face side
192 pixels, and exactly one lowercase `img` trigger. BC_E13 changes only the
training dataset and explanatory metadata. **[code]**

### 3.3 Why the baseline Cosmic Large policy had to be repaired

`cosm_new1_vast` used the original `CosmicLargeTrain` path described in
Section 1.2. It had several independent problems that were fixed before the
later CL14-family claims. The corrected loader is
[`cosmic_large_adapted.py`](../../src/datasets/cosmic_large_adapted.py), with
geometry in
[`reference_frame.py`](../../src/datasets/reference_frame.py).

The first two failures below are the critical ones. They are related but not
identical: the first is the original **scale and resolution mismatch**; the
second is the **degenerate exact-alignment shortcut introduced by the first
attempt to repair that mismatch**.

#### 3.3.1 Critical failure 1: a 256px crop became a 2.12x oversized spatial memory

The old loader supplied a tight `256 x 256` reference crop, while the target
was a `1024 x 1024` scene. The model did not treat that crop as a scale-free
identity token. In
[`PhotomakerBranchedLora._encode_reference_latents`](../../src/model/photomaker_branched/lora2.py),
the whole reference image was bilinearly fitted to `1024 x 1024` and then
encoded by the frozen VAE as the reference half of the doubled U-Net batch.
Consequently, a roughly `142-144px` face in the source JPEG became roughly
`568-576px` wide in the image presented to the VAE. Upscaling created no new
facial detail; it spread the same low-resolution pixels across four times as
many pixels per axis. **[code] [measured]**

**Exact old-code quotation - crop returned by the 2 June loader.** At
`main_clean@2157eada...`, `CosmicLargeTrain.get_ref_image()` opened the
face-focused asset, cropped around its bbox, and returned that crop directly;
it did not first place the face into a calibrated 1024px scene
(`src/datasets/cosmic.py`, historical lines 1299-1309):

```python
ref_img = Image.open(self._face_full_path(face_path)).convert("RGB")
face_bbox = self._get_face_bbox(img_data, face_path)
ref_face, ref_bbox = self._get_bigger_crop_with_bbox(ref_img, face_bbox)
if ref_bbox is None:
    raise ValueError(f"Invalid reference face bbox after crop: {face_path}")
if random.random() < 0.5:
    w, _ = ref_face.size
    ref_face = ImageOps.mirror(ref_face)
    x0, y0, x1, y1 = ref_bbox
    ref_bbox = [w - x1, y0, w - x0, y1]
return ref_face, ref_bbox
```

**Exact old-code quotation - enlargement immediately before the VAE.** The
same 2 June model then fitted the entire returned crop to `target_size=1024`
and only afterwards encoded it (`lora2.py`, historical lines 662-678):

```python
ow, oh = ref_image.size
scale = min(self.target_size / ow, self.target_size / oh)
rw = max(8, int(round(ow * scale)) // 8 * 8)
rh = max(8, int(round(oh * scale)) // 8 * 8)
pl = (self.target_size - rw) // 2
pr = self.target_size - rw - pl
pt = (self.target_size - rh) // 2
pb = self.target_size - rh - pt
ref_resized = ref_image.resize((rw, rh), Image.BILINEAR)
ref_np = np.array(ref_resized).astype(np.float32) / 255.0
ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)
ref_tensor = (ref_tensor - 0.5) / 0.5
ref_tensor = F.pad(ref_tensor, (pl, pr, pt, pb), value=0.0)
ref_tensor = ref_tensor.to(device=self.device, dtype=self.vae.dtype)

with torch.no_grad():
    latents = self.vae.encode(ref_tensor).latent_dist.mode()
```

Together these two contiguous excerpts are the exact mechanism behind the
old geometry: return a tight face crop, expand that crop to the full spatial
conditioning frame, then VAE-encode it. The bbox mask was constructed after
the same resize; it selected the enlarged face tokens but did not undo the
enlargement. **[code]**

The population geometry made the mismatch explicit:

| Quantity | original Cosmic training reference | filtered Cosmic target |
|---|---:|---:|
| frame size before model normalization | `256 x 256` | `1024 x 1024` |
| median face-area fraction | `42.6%` | `9.49%` |
| representative face short side | `144px`, then `4x` enlarged | `272px` |
| approximate face span on the `128 x 128` latent grid | `83.5` cells | `39.4` cells |

Thus the reference/target linear scale ratio was

$$
r_{linear}=\sqrt{\frac{0.426}{0.0949}}\simeq 2.12,
\qquad
r_{area}=\frac{0.426}{0.0949}\simeq 4.49.
$$

Figure 1 reconstructs that forward-path mismatch with **real Cosmic Large
target/reference pairs**, not generated examples or a schematic. The three
targets are records `339886005404.jpg`, `1223294003468.jpg`, and
`1087567013854.jpg` from
`gathered_data_cosmic_large_filtered_sample_two.json`; all three pass the
later 192px minimum target-face-side gate. The middle column applies the old
model's actual normalization: resize the entire 256px crop to 1024px with
bilinear interpolation, while multiplying its bbox coordinates by four. The
right column calls the production `compose_target_frame_reference` function
in its scale-parity mode. It is deliberately a **scale-only controlled
reconstruction**: it makes the short side match the target before the VAE,
while leaving exact-centre alignment visible so Figure 2 can isolate why that
first repair was still insufficient. **[code] [measured]**

![Real Cosmic Large pairs showing the historical oversized reference
memory and the pre-VAE scale repair.](../../analysis/assets/cosmic_scale_mismatch_before_after.png)

The plotted measurements come directly from the manifest boxes:

- `339886005404.jpg`: target `[320,103,594,464]`, `9.43%`; reference
  `[55,35,207,236]`; old reference area `46.62%`; old ref/target short-side
  ratio `2.22x`.
- `1223294003468.jpg`: target `[577,411,920,921]`, `16.68%`; reference
  `[64,31,201,246]`; old reference area `44.94%`; old ref/target short-side
  ratio `1.60x`.
- `1087567013854.jpg`: target `[542,64,920,601]`, `19.36%`; reference
  `[55,41,193,236]`; old reference area `41.06%`; old ref/target short-side
  ratio `1.46x`.

The examples also make the mechanism concrete. In the middle column the face
mask covers most of the VAE input and the source pixels have been magnified
fourfold in each axis. In the repaired column the face has target-scale
granularity and the remainder of the 1024px reference frame is supplied by
edge padding. The visible horizontal/vertical colour bands are therefore the
literal configured `edge` fill, not image-generation artifacts. They contain
no invented structure, and the hard reference mask excludes them from the
identity-bearing reference K/V set. The figure's cyan box is the propagated
reference mask that is actually passed downstream. **[code]**

The reference bbox mask did **not** correct this. It selected which reference
features could supply K/V, but it did not rescale, warp, landmark-align, or
otherwise register what those features encoded. With
`pipeline.pose_adapt_ratio=0`, target-face queries had no target-feature
fallback: the hard face branch consumed the mis-scaled reference-face K/V and
wrote its message into the target face box. A target query corresponding to an
eye-sized region therefore attended a reference grid whose local features were
roughly half that physical granularity. The result was a strong route carrying
blurred, spatially misregistered facial structure rather than merely an
identity descriptor. See the exact hard-route quotation in Section 11.7.
**[code]**

This failure was not inferred only from dimensions. The fixed-96 validation
panel measured generated-face short side divided by the fixed required mask
short side; `<0.8` means the rendered face is at least 20% too small. At the
matched step `8,000`, changing target-image composition alone did nothing,
while changing the *reference* face-scale distribution removed nearly the
entire failure tail:

| Arm | training reference policy | ratio p10 | `<0.8` faces |
|---|---|---:|---:|
| E13 | full-scene reference, about `8.6%` face area | `0.937` | `0` |
| CL4 | original tight reference, about `42%` | `0.791` | `11` |
| CL8 | same tight reference; full-body targets restored | `0.779` | `12` |
| CL9 | reference face fraction sampled in `[0.06,0.30]` | `0.933` | `1` |

Immutable Comet keys are E13 `1cc0a02371094b24a6a02a4cc649f10c`, CL4
`0dd86b436b224f939efa3887ad6acbe2`, CL8
`a6b5970aa1a24d3490ad08e7994b5f1e`, and CL9
`81bb311ed70545eda3281c64bc48be47`.

The affected low-scale validation identities were also sharply separated:
Jensen (`6.22%` reference face area), Jisoo (`7.11%`), and Keanu (`7.84%`)
accounted for the old undersized-face cluster; CL9 reduced all three to zero.
CL8 restored the target full-body distribution but left the reference at about
`42%`, and its count stayed `12`. This is strong interventional evidence that
the reference scale, not the target framing, caused this particular failure.
`id_sim` was not a safe primary metric here because a small but recognizable
face could still score well. **[measured] [report]**

The corrected data path performs the normalization *before* the VAE. For
reference-face area $A_r$, canvas side $S=1024$, and requested face fraction
$u \sim U(0.06,0.30)$, it uses

$$
s=\sqrt{\frac{uS^2}{A_r}},
$$

resizes the reference and bbox together, crops or edge-pads the result to a
`1024 x 1024` frame, and validates that the realized area is within 10% of the
requested draw. CL14 inherits exactly this corrected loader policy. The fix
changes the spatial representation consumed by the frozen VAE/U-Net; it does
not change the E13 attention equation, trainable ownership, or inference
pipeline. **[code]**

**Exact fix quotation - calculate the requested face area before encoding.**
The repaired compositor derives the resize factor from the requested *area*
and propagates the same factor to pixels and bbox (`reference_frame.py`, clean
lines 87-97):

```python
rx0, ry0, rx1, ry1 = [float(v) for v in reference_bbox]
reference_area = (rx1 - rx0) * (ry1 - ry0)
if reference_area <= 0:
    raise ValueError(f"degenerate reference face box: {list(reference_bbox)}")
scale = ((fraction * canvas_size * canvas_size) / reference_area) ** 0.5
```

```python
scaled_w = max(1, int(round(reference.width * scale)))
scaled_h = max(1, int(round(reference.height * scale)))
scaled = reference.convert("RGB").resize((scaled_w, scaled_h), BICUBIC)
scaled_bbox = [float(v) * scale for v in reference_bbox]
```

**Exact fix quotation - invoke the compositor before the model sees the
reference.** The adapted dataset replaces `reference` and `reference_bbox`
with the composed 1024px result before returning the batch
(`cosmic_large_adapted.py`, clean lines 562-576):

```python
(
    reference,
    reference_bbox,
    policy_descriptor,
    frame_telemetry,
) = compose_target_frame_reference(
    reference,
    reference_bbox,
    target_bbox,
    canvas_size=1024,
    fill=self.reference_frame_fill,
    gray_level=self.reference_canvas_fill,
    target_face_fraction=requested_fraction,
    position_offset=position_offset,
)
```

This ordering is the critical repair: the frozen VAE receives the already
calibrated 1024px frame, rather than being asked to encode an oversized tight
crop that a downstream attention mask cannot rescale. **[code]**

#### 3.3.2 Critical failure 2: exact scale and position created an in-place copy shortcut

The first repair attempt, CL2 `facecanon`, removed the `2.12x` mismatch too
literally. It used

$$
s_{CL2}=\frac{\text{target face short side}}
               {\text{reference face short side}}
$$

and placed the resized reference-face centre exactly on the target-face centre
for every training pair. This made the reference and target face occupy the
same scale and absolute canvas coordinates on every sample. In code terms,
`target_face_fraction` was absent and `position_offset=(0,0)`, so
`compose_target_frame_reference` selected target-scale parity and exact centre
coincidence. **[code]**

**Exact CL2 quotation - the first repair retained deterministic geometry.**
CL2 selected the target-frame compositor, while its inherited controls left
scale and position randomization disabled. The immutable CL2 source record is
`d903b2c9...+cl1-cl3-snapshot-v1-20260806`; its effective config excerpts were:

```yaml
reference_frame_mode: target_face_frame
reference_frame_fill: edge
```

```yaml
reference_scale_jitter: null
reference_position_jitter: 0.0
```

Those values select the compositor's exact-parity branch and zero offset
(`reference_frame.py`, clean lines 92-110):

```python
else:
    scale = target_short / reference_short
```

```python
ref_cx, ref_cy = _center(scaled_bbox)
tgt_cx, tgt_cy = _center(target_bbox)
tgt_cx += float(position_offset[0]) * canvas_size
tgt_cy += float(position_offset[1]) * canvas_size
offset_x = ref_cx - tgt_cx
offset_y = ref_cy - tgt_cy
```

With the effective `position_offset=(0,0)`, the last four lines reduce to
`offset_x=ref_cx-tgt_cx` and `offset_y=ref_cy-tgt_cy`: the crop window is
therefore chosen to place the reference centre on the target centre. **[code]**

That geometry was an overly easy solution for hard spatial attention. The
target queries and the identity-bearing reference K/V were already registered
cell-for-cell; minimizing the face-only diffusion objective no longer required
the branch to learn that identity must survive a change of scale or position.
It could instead learn a near-position-preserving, in-place transfer. Real
validation references do not share the generated target's scale and centre, so
that shortcut did not generalize. This is the project's "positional-copy
shortcut" diagnosis. It is strongly supported by the intervention below, but
scale and position were randomized together, so their individual contributions
were not isolated by separate runs. **[report] [hypothesis]**

CL2 proved both halves of the diagnosis. It almost eliminated the undersized
face problem (`1` case at 24k), confirming that exact scale registration worked,
but its identity learning regressed: immutable Comet run
`be7b7a2acf174b69b5e361490926140e` moved from `id_sim=0.30187` at step zero to
`0.24273` at 2k and `0.27219` at 4k, ending at `0.28255` at 24k. The network was
being optimized, yet the learned solution transferred identity worse than its
own initialization. **[measured]**

CL9 kept the same target-frame compositor and the same edge-filled surround,
but broke the deterministic correspondence in two ways:

- draw face area independently as $u \sim U(0.06,0.30)$, rather than copying
  the target's exact scale;
- draw horizontal and vertical offsets independently from
  $U(-0.15,0.15)$ of the 1024 canvas, or up to about `154px` per axis, rather
  than copying the target's exact centre.

**Exact two-change quotation - CL9/CL14 configuration.** The promoted dataset
policy turns on the two controls together:

```yaml
reference_frame_mode: target_face_frame
reference_frame_fill: edge
reference_scale_jitter: [0.06, 0.30]
reference_position_jitter: 0.15
```

**Exact two-change quotation - independent draws in the loader.** The loader
samples face area separately from the two position coordinates
(`cosmic_large_adapted.py`, clean lines 544-554):

```python
requested_fraction = None
position_offset = (0.0, 0.0)
if self.reference_scale_jitter is not None:
    low, high = self.reference_scale_jitter
    requested_fraction = random.uniform(low, high)
if self.reference_position_jitter > 0.0:
    jitter = self.reference_position_jitter
    position_offset = (
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    )
```

The sampled values then enter the compositor through the exact call quoted in
Section 3.3.1. The area draw selects the area-based scale branch; the two
offset draws are applied to the target centre before the crop window is
computed. This is why the fix changes both size correspondence and absolute
cell correspondence, while leaving the attention equation untouched.
**[code]**

Figure 2 shows why those two random draws matter. It uses the same three real
pairs as Figure 1 and overlays the masks that control the spatial branch. The
CL2 column is produced with `target_face_fraction=None` and
`position_offset=(0,0)`: the reference short side equals the target short side
and their centres differ by only `0.3-0.4px`, solely because the compositor
rounds its integer crop window. At a 128px latent resolution that sub-pixel
image-space difference vanishes; the target-query and reference-K/V regions
are effectively registered cell for cell. **[code] [measured]**

![The CL2 exact-alignment shortcut and the CL14 independent scale and
position policy on the same real Cosmic Large pairs.](../../analysis/assets/cosmic_positional_shortcut_before_after.png)

\clearpage

The CL14 column uses three fixed, valid draws solely to make the illustration
reproducible: `(u, dx, dy) = (0.06,-0.13,+0.11)`,
`(0.17,+0.12,-0.10)`, and `(0.29,-0.10,-0.13)`. The production loader draws
new values independently for every pair. The requested area fractions are
realized as `0.060`, `0.170`, and `0.290`; the reference-mask centres move by
`(-133,+113)px`, `(+107,-102)px`, and `(-102,-5)px` relative to the target
mask. The third vertical draw is partially constrained by the
keep-the-whole-face-inside-canvas rule, hence its requested `-133px` vertical
shift realizes as `-5px`; this is expected boundary protection and is recorded
by compositor telemetry. Green dashed boxes are target-query masks, coloured
solid boxes are reference-K/V masks, and each white segment joins their
centres. The non-overlap makes the removed correspondence visually explicit:
the branch can no longer assume that a target cell retrieves the same facial
location from the reference cell at the same absolute coordinate. **[code]
[measured]**

These pictures demonstrate the *training inputs and masks*, not generated
quality. The causal evidence that the repairs improved generated geometry and
avoided CL2's identity regression remains the controlled validation comparison
below. The figures are reproducible from the checked script
[`build_cosmic_geometry_figures.py`](../../tools/reports/build_cosmic_geometry_figures.py):

```bash
cd diffusion_template
python3 tools/reports/build_cosmic_geometry_figures.py \
  --dataset-root /home/kolyangg/rsrch/dataset_full
```

At the matched 8k comparison, CL9 retained the scale benefit (`1` undersized,
ratio p10 `0.933`) while reaching `id_sim=0.38619`, close to CL4's `0.39391` and
without CL2's collapse. Because CL2's 24k endpoint and CL9's cited 8k point are
not step-matched, their absolute endpoint scores must not be ranked directly;
the robust findings are CL2's regression relative to its own initialization
and the matched CL9/CL4 face-scale result. The shared edge fill also makes a
fabricated-surround-only explanation much less plausible, although no
dedicated fill ablation was run. **[measured] [report]**

Finally, the realized scale, crop window, requested fraction, and position
offset are included in the conditioning-cache descriptor. Otherwise two
targets sharing one reference path could incorrectly reuse conditioning from
different geometry and silently reintroduce a spatial mismatch. The relevant
source quotation is in Section 11.6. **[code]**

The remaining independent Cosmic issues and their fixes were:

| Problem | Corrected behavior | Category |
|---|---|---|
| reference was independently mirrored | disable reference flip; retain configured target flip | dataset sampling |
| long appearance-first captions lost pose/background to tokenizer truncation | pose-first prompt, exactly one `img`, maximum 50 words | dataset prompt policy |
| transformed pixels and reference bbox/cache identity could diverge | propagate the same transform to the bbox and bind realized geometry into the cache key | dataset correctness |
| self-reference leakage | require a different reference path and fail on collision | dataset correctness |
| silent CPU InsightFace and `CUDA_LAUNCH_BLOCKING=1` caused 5-7 s/step | require CUDA provider and asynchronous execution | runtime, not dataset science |

The adapted CL14 policy accepts 22,140 of 59,143 input records with minimum
target-face side 192 pixels. It does **not** repair the remaining lack of stable
multi-view identity groups, and it does not restore the full-body records
removed by the 192-pixel gate. **[code] [report]**

## 4. E13: first architecture on the repaired substrate

E13 keeps hard target/background replacement but widens the branch projections
to rank 128 and deliberately co-trains the effective generic rank-32 and
PhotoMaker-default rank-64 paths under exact ownership.

For target projections

$$
Q_t=W_q^tH_t,\quad K_t=W_k^tH_t,\quad V_t=W_v^tH_t,
$$

the target background and reference-face messages are

$$
B=\operatorname{Attn}(Q_t\odot(1-M_t),K_t,V_t),
$$

$$
F=\operatorname{Attn}
\left(Q_t\odot M_t,W_k^r(H_r\odot M_r),W_v^r(H_r\odot M_r)\right).
$$

The required `pose_adapt_ratio=0` means no target feature is substituted for
reference face K/V. The target output is

$$
Y_t=W_o\left((1-M_t)\odot B+M_t\odot F\right).
$$

The reference lane retains ordinary reference self-attention. The binary
reference mask zeros unsupported K/V vectors but leaves their token positions
in the softmax; those historical zero sinks are retained. Native SDXL/
PhotoMaker cross-attention remains active, while the old branched-CA processor
is disabled. See
[`attn_processor_cleanest.py`](../../src/model/photomaker_branched/attn_processor_cleanest.py).
**[code]**

All recipes use 24,000 optimizer steps, batch 2, 2,000-step epochs, base LR
`1e-4`, and a warmup/hold/cosine schedule. Their primary diffusion objective is
face-box MSE:

$$
L_{face}=\frac1B\sum_b
\operatorname{mean}_{c,x,y\in\mathcal B_b}
\left(\epsilon_{\theta,b,c,x,y}-\epsilon_{b,c,x,y}\right)^2.
$$

The E13 leaf is
[`E13_large_ds_joint_shadow_sa128_24k.yaml`](../../src/configs/E13_large_ds_joint_shadow_sa128_24k.yaml).

## 5. Dataset siblings of E13

### 5.1 BC_E13 = E13 architecture, BigCelebs dataset

BC_E13 changes no model, optimizer, loss, schedule, pipeline, or validation
equation:

$$
\theta_{BC\_E13}
=\operatorname{Train}(A_{E13},D_{BigCelebs}).
$$

Its leaf is
[`BC_E13_big_celebs_joint_shadow_sa128_24k.yaml`](../../src/configs/BC_E13_big_celebs_joint_shadow_sa128_24k.yaml).
Any BC_E13 result is therefore evidence about dataset usage under E13, not a
new architecture. **[code]**

### 5.2 CL14 = E13 architecture, corrected Cosmic data, training-mask delta

CL14 selects the corrected Cosmic policy from Section 3.3 and constructs a
two-cell inward target-mask feather during training. For width $k=2$, ring
$j\in\{1,2\}$ receives

$$
w_j=\frac{j}{k+1},
$$

so the rings are `1/3` and `2/3`, with interior 1. The hard processor applies
`M_t > 0.5`; therefore the current effective SA route is still binary: the
outer ring becomes background and the inner ring becomes face. This is a
one-cell boundary contraction, not a continuous inference blend. Reference
and inference masks remain binary. **[code]**

The construction is in
[`lora2.py`](../../src/model/photomaker_branched/lora2.py), and the leaf is
[`CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`](../../src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml).

## 6. Direct CL14 children

### 6.1 CL14_CA: bounded residual identity-token cross-attention

CL14_CA keeps CL14 self-attention, dataset, loss, and schedule. In
`up_blocks.0/1`, target queries additionally attend active PhotoMaker identity
tokens. Native CA remains intact:

$$
Y_t=N_t+M_t\odot g\widehat{\Delta},\qquad
g=0.20\sigma(\gamma),\quad g_{init}=0.02.
$$

The rank-64 residual output is zero-initialized and RMS-normalized after
clamping the mean square, so a fresh processor is exactly CL14 native CA and
has finite zero-init gradients. CL14_CA adds 108 tensors / 5,406,756
parameters, for **2,348 / 224,624,676** total. See
[residual CA processor](../../src/model/photomaker_branched/residual_identity_ca_processor_v3.py)
and
[CL14_CA config](../../src/configs/CL14_CA_cosmic_residual_identity_ca_24k.yaml).
**[code]**

### 6.2 CL18: training-only same-ID cross-view consistency

CL18 has exactly CL14 inference. On 25% of training batches it evaluates a
second distinct same-ID spatial reference while fixing target latent, noise,
timestep, prompt, identity tokens, and paired reference noise. With detached
primary teacher (T) and alternate-reference student (S):

$$
L_{cv}=\operatorname{SmoothL1}_{M_t}(S,T)
+0.10\left(1-\cos(\operatorname{vec}S,\operatorname{vec}T)\right),
$$

$$
L=L_{face}+0.05L_{cv}
$$

on sampled batches. Inference remains single-reference. See the CL18 block in
[`lora2.py`](../../src/model/photomaker_branched/lora2.py) and the
[CL18 config](../../src/configs/CL18_cosmic_crossview_spatial_consistency_24k.yaml).
**[code]**

### 6.3 CL19: full-query two-cell cosine router

CL19 replaces CL14's query masking/hard merge with two complete messages:

$$
N=\operatorname{Attn}(W_q^tH_t,W_k^tH_t,W_v^tH_t),
$$

$$
R=\operatorname{Attn}
\left(W_q^tH_t,W_k^r(H_r\odot M_r),W_v^r(H_r\odot M_r)\right).
$$

Two eroded boundary rings receive cosine weights `0.25` and `0.75`; the face
interior is 1 and exterior 0. With router (C):

$$
Y_t=W_o\left((1-C)\odot N+C\odot R\right).
$$

The full messages are blended exactly once. CL19 adds no parameters. See
[CL19 router implementation](../../src/model/photomaker_branched/attn_processor_cleanest.py)
and
[CL19 config](../../src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml).
**[code]**

### 6.4 CL20: dataset-only hard-case curriculum

CL20 returns to the CL14 model and loss. Its only scientific variable is a
sealed sequential 48,000-row schedule: 80/20 Cosmic/BigCelebs through step
19,999, followed by 4,000 Cosmic-only re-anchor steps. BigCelebs rows rotate
across synthetic-small-face, occlusion-caption, and action-caption strata.

$$
D(n)=
\begin{cases}
BigCelebs,&\lfloor n/2\rfloor<20{,}000\ \land\ n\bmod5=0,\\
Cosmic,&\text{otherwise}.
\end{cases}
$$

The loader verifies exact row order, manifests, target/reference inequality,
and resume offset. See
[CL20 curriculum loader](../../src/datasets/cl20_hardcase_curriculum.py),
[schedule builder](../../tools/datasets/build_cl20_hardcase_schedule.py),
and
[CL20 config](../../src/configs/CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml).
**[code]**

## 7. CL19 descendants

### 7.1 CL23: deterministic temporal-frequency routing

CL23 keeps CL19's (N), (R), and router (C). It splits (D=R-N) with a
fixed separable 5x5 Gaussian kernel `[1,4,6,4,1]/16`:

$$
D_L=G*D,\qquad D_H=D-D_L.
$$

For real denoising progress (p=1-t/(T-1)):

$$
a_L=0.50+0.35p,\qquad a_H=0.75+0.50p,
$$

$$
Y_t=N+C\odot(a_LD_L+a_HD_H).
$$

The schedule is fixed and adds no parameters. Training uses progress from the
sampled diffusion timestep; validation uses the live scheduler timestep. See
[`_call_temporal_frequency`](../../src/model/photomaker_branched/attn_processor_cleanest.py)
and
[`CL23_cosmic_temporal_frequency_router_24k.yaml`](../../src/configs/CL23_cosmic_temporal_frequency_router_24k.yaml).
**[code]**

### 7.2 CL27: CL23 inference plus a training-only surface objective

CL27 inference is exactly CL23. On deterministic 25% semantic-occluder samples,
only `up_blocks.0/1` add a frequency-surface objective. Let $M_T$ be the
top-object/face overlap, $M_V$ the visible face, $L=C\odot a_LD_L$,
$H=C\odot a_HD_H$, and $\Delta=L+H$:

$$
E_T=\operatorname{mean}_{M_T}(H^2)
+0.25\operatorname{mean}_{M_T}(L^2),
$$

$$
r=\frac{\operatorname{RMS}_{M_V}(\Delta)}
{\operatorname{stopgrad}(\operatorname{RMS}_{M_V}(N))},
$$

$$
L=L_{face}+0.02E_T+0.005\max(0,0.35-r)^2.
$$

The occluder mask is supervision only and never enters validation/inference
routing. Ownership remains **2,240 / 219,217,920**. See
[`_frequency_surface_loss`](../../src/model/photomaker_branched/attn_processor_cleanest.py),
[`collect_frequency_surface_aux_loss`](../../src/model/photomaker_branched/lora2_helpers.py),
and
[`CL27_cosmic_frequency_surface_energy_24k.yaml`](../../src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml).
**[code]**

## 8. Recipe matrix: what changes and what must remain fixed

| Recipe | Parent | Architecture delta | Dataset/objective delta | Trainables |
|---|---|---|---|---:|
| E13 | repaired `cosm_new1_vast` / 2 June base | hard BA rank 128 + effective generic/default co-adaptation | Large Dataset | 2,240 / 219,217,920 |
| BC_E13 | E13 | none | BigCelebs only | same |
| CL14 | E13 | effective one-cell hard boundary contraction during training | corrected Cosmic policy | same |
| CL14_CA | CL14 | bounded residual identity-token CA in up0/up1 | none | 2,348 / 224,624,676 |
| CL18 | CL14 | none at inference | sampled cross-view loss | 2,240 / 219,217,920 |
| CL19 | CL14 | full-query cosine SA router | none | same |
| CL20 | CL14 | none | deterministic Cosmic/BigCelebs curriculum | same |
| CL23 | CL19 | fixed temporal-frequency gains | none | same |
| CL27 | CL23 | none at inference | semantic-occluder surface loss | same |

Non-negotiable shared invariants:

```text
use_branched_attention = true
pose_adapt_ratio = 0
ca_mixing_for_face = false
legacy branched CA = disabled
target Q retains an explicit spatial-reference K/V path
fixed validation prompts, seeds, references, bboxes, scheduler and metrics
checkpoint processor/trainable manifest matches exactly
```

## 9. Files with the largest live changes versus 2 June

The following counts use `git diff --numstat 2157eada..7b7579b` and include
only files that still exist. They are review size, not a claim that every line
is scientific logic.

- `src/datasets/cosmic_large_adapted.py` (`+717 / -0`): corrected Cosmic
  policy, CL18 alternate view, and CL27 supervision.
- `src/model/photomaker_branched/lora2.py` (`+527 / -23`): E13 integration,
  batching, CL14 mask, CL18 objective, and CL27 aggregation.
- `src/model/photomaker_branched/e13_contract.py` (`+524 / -0`): exact
  ownership, manifest, and checkpoint contract.
- `src/model/photomaker_branched/attn_processor_cleanest.py` (`+354 / -1`):
  hard route plus localized CL19/CL23/CL27 equations.
- `src/model/photomaker_branched/branched_runtime.py` (`+264 / -68`):
  processor installation, flags, doubled-batch route, and progress.
- `src/model/photomaker_branched/residual_identity_ca_processor_v3.py`
  (`+330 / -0`): isolated CL14_CA delta.
- `src/model/photomaker_branched/lora2_helpers.py` (`+276 / -0`):
  conditioning/runtime helpers and optimized collectors.
- `src/trainer/base_trainer.py` (`+200 / -24`): validation shadow, processor
  copy, sequential offsets, and fixed protocol.
- `src/datasets/cl20_hardcase_curriculum.py` (`+219 / -0`): sealed sequential
  curriculum consumer.
- `src/datasets/reference_frame.py` (`+188 / -0`): target-frame reference
  composition.
- `src/pipelines/br_pipeline_helpers.py` (`+51 / -137`): concise sealed
  single-reference pipeline helpers.
- `src/pipelines/photomaker_branched_subject_v2.py` (`+157 / -0`): isolated
  corrected validation and extension installation.

The core remains much smaller than the experimental `test` switchboard. New
logic is marked with dated `E13C-*`, `CL14_CA-*`, or 12/18 August comments so
`rg` can recover the rationale beside the code.

## 10. Exact configs, launcher, and server preparation

### 10.1 Historical and descendant provenance

The baseline row is an immutable run-to-source attribution. For later rows,
the Git column records the source revision used to reconstruct the clean leaf;
it must not be misread as Git metadata emitted by every historical Comet run.
The detailed ledgers in References retain checkpoint hashes, failed attempts,
and runtime-specific fixes. **[record] [code]**

| Run | Immutable Comet key | Git/source provenance |
|---|---|---|
| `cosm_new1_vast` | [`b7602f92...`](https://www.comet.com/nikolay-2104/rsrch-30oct/b7602f92bca54ba5aa0c189192d17165) | `main_clean@2157eada...`; attribution evidence in Section 1.1 |
| E13 r4 | [`1cc0a023...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/1cc0a02371094b24a6a02a4cc649f10c) | successful runtime `ebf1ac8295f363adb0055cd74db1a96c2ff03a35` |
| BC_E13 r1 | [`c138db7c...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/c138db7c41ae435c8a7560f40cf5f58d) | runtime `ad194a026ab701dd979712d415c487dd536a4645` |
| CL14 r1 | [`6fe0028b...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/6fe0028be92242c38056b3d36665fdd6) | `c04970f342a186d1092f07f9a08d7d8a797383e8` plus sealed `cl12-cl14-snapshot-v1-20260809` |
| CL14_CA r11 | [`fafd7a61...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/fafd7a61b06c4114b9dec2c21d29ca38) | test source inspected at `ceb34c3` |
| CL18 r2 | [`f6530436...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/f6530436bf22472c9fb7731d1696c5ab) | test `ad194a0...` plus recorded corrected-r2 snapshot |
| CL19 r2 | [`cfeda7b5...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/cfeda7b55c174b3c83e8d40537ebb6dd) | same corrected-r2 source boundary as CL18 |
| CL20 r2 | [`b05488e2...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/b05488e2cce94476acc92bcaa21d7362) | same corrected-r2 source boundary as CL18 |
| CL23 | [`a9ec9c59...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/a9ec9c59d1624c68acb98737dcd65298) | committed test snapshot `6eb6613` |
| CL27 r3 | [`dbfbf40c...`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/dbfbf40c3bdd4f70bedc58bda3dfb9cd) | committed test snapshot `6eb6613` |

The clean implementation itself is currently audited through
`clean@7b7579bf91f378321103cf1a9d367f6906e7e0e1`.

### 10.2 Launch provenance and supported commands

The historical invocation was:

```bash
git switch --detach 2157eada14824d14019e80f9416e6d736c837306
cd diffusion_template
bash serv_new_runs/start_ba_cosm_new1_vast.sh
```

This is a forensic command, not a supported relaunch command. The historical
script contains an obsolete embedded credential and `/workspace` dataset
paths. Do not execute it unchanged; the Comet table in Section 1.2 is the
authoritative resolved configuration. **[code]**

Every clean descendant uses the same fail-closed launcher, with exactly one
leaf from the table below and no ad-hoc Hydra overrides:

```bash
cd /absolute/path/to/diffusion_template
RUN_NAME=<unique_run_name> \
CONFIG_NAME=<exact_leaf_below> \
bash launchers/active/run_e13_family_24k_1gpu.sh
```

On Serv, the supported path is the corresponding checked-in one-A100 YAML:

```bash
mls job submit --config /absolute/path/to/run_<clean_run_name>_1gpu.yaml
```

No command in this document authorizes a submission.

### 10.3 Clean recipe leaves

| Recipe | Hydra leaf |
|---|---|
| E13 | `E13_large_ds_joint_shadow_sa128_24k` |
| BC_E13 | `BC_E13_big_celebs_joint_shadow_sa128_24k` |
| CL14 | `CL14_cosmic_joint_shadow_sa128_softmask_24k` |
| CL14_CA | `CL14_CA_cosmic_residual_identity_ca_24k` |
| CL18 | `CL18_cosmic_crossview_spatial_consistency_24k` |
| CL19 | `CL19_cosmic_true_soft_fullquery_router_24k` |
| CL20 | `CL20_cosmic_bigcelebs_hardcase_curriculum_24k` |
| CL23 | `CL23_cosmic_temporal_frequency_router_24k` |
| CL27 | `CL27_cosmic_frequency_surface_energy_24k` |

All clean recipes use
[`run_e13_family_24k_1gpu.sh`](../../launchers/active/run_e13_family_24k_1gpu.sh).
Exact one-A100 Serv YAMLs are indexed in
[`serv_run_packages/README.md`](../../serv_run_packages/README.md). Machine
paths and secrets belong only in the ignored `.env`.

Before any submission:

```bash
cd /absolute/path/to/diffusion_template
git switch clean
git pull --ff-only origin clean
test -z "$(git status --porcelain)"

python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py
python tools/validate_cl14_ca_config.py
python tools/validate_cl18_cl20_config.py
python tools/validate_cl23_cl27_config.py
bash -n launchers/active/run_e13_family_24k_1gpu.sh
```

The Serv wrapper additionally checks the exact dataset/subject hashes, decoded
samples, ONNX Runtime CUDA provider, branch cleanliness, unique run directory,
ownership totals, and creation of `saved/<run>/comet_experiment.json`. Inspect
Running and Pending MLS requests before submission and stay within the normal
six-A100 project ceiling. This document does not authorize or perform a launch.

## 11. Critical code quotations: fixes and run-to-run deltas

These are deliberately short quotations of the lines that carry the important
behavior. Historical excerpts are from
`main_clean@2157eada14824d14019e80f9416e6d736c837306`; clean excerpts are from
the source audited at `clean@7b7579bf91f378321103cf1a9d367f6906e7e0e1`.
Line numbers identify those immutable snapshots. Comments, relative indentation,
and physical line breaks inside the blocks are preserved from source; leading
function indentation is normalized for display. The PDF renderer may wrap a
long displayed line visually. Context omitted between separate blocks is stated
explicitly. **[code]**

### 11.1 Error fix: warning-and-continue processor installation becomes fatal

The June installer iterated over every processor and swallowed any exception,
including a plain Diffusers processor not implementing `.parameters()`:

Source: `lora2_helpers.py`, June lines 103-121, inside the `try` begun at line 89.

```python
if hasattr(model.unet, "attn_processors"):
    for proc in model.unet.attn_processors.values():
        for p in proc.parameters():
            p.requires_grad_(True)

if model.face_embed_strategy == "id_embeds" and not model.use_attn_v2:
    for name, proc in model.unet.attn_processors.items():
        if not name.endswith("attn1.processor"):
            continue
        if getattr(proc, "id_to_hidden", None) is None and hasattr(proc, "hidden_size"):
            proc.id_to_hidden = torch.nn.Linear(2048, proc.hidden_size, bias=False).to(
                model.unet.device, dtype=model.unet.dtype
            )
            with torch.no_grad():
                proc.id_to_hidden.weight.mul_(0.1)

configure_branched_trainables(model)
except Exception as e:
    print(f"[PhotomakerBranchedLora] exception while installing branched processors: {e}")
```

The clean E13 branch exits the strict path only after installation and exact
ownership both succeed; there is no catch around it:

Source: `lora2_helpers.py`, clean lines 175-196.

```python
def install_branched_processors_for_training(model) -> None:
    """Install branched attention processors once before optimizer creation."""
    if bool(getattr(model, "e13_family_contract", False)):
        # 10 Aug 2026 - E13C-CORE-01: Strict installation must propagate any
        # processor/ownership failure. The historical warning-and-continue path
        # could silently leave the base U-Net or the wrong adapters trainable.
        h = model.target_size // int(model.vae_scale_factor)
        w = model.target_size // int(model.vae_scale_factor)
        zero_ctx = torch.zeros(
            1, 1, h, w, device=model.unet.device, dtype=model.unet.dtype
        )
        patch_unet_attention_processors(
            pipeline=model,
            mask=zero_ctx,
            mask_ref=zero_ctx,
            scale=1.0,
            id_embeds=None,
            class_tokens_mask=None,
        )
        configure_e13_trainables(model)
        assert_e13_trainable_contract(model)
        return
```

### 11.2 Error fix: broad `requires_grad` selection becomes an exact allowlist

June returned every U-Net tensor left trainable after the potentially failed
installer:

Source: `lora2.py`, June lines 258-263.

```python
# Default behavior: train all UNet parameters with requires_grad=True (LoRA + processors).
lora_params = filter(lambda p: p.requires_grad, self.unet.parameters())
trainable_params = [
    {"params": lora_params, "lr": config.lr_for_lora, "name": "lora_params"},
]
return trainable_params
```

The clean contract computes names first, freezes everything else, and compares
actual ownership to the allowlist:

Source: `e13_contract.py`, clean lines 240-247.

```python
def configure_trainables(model) -> None:
    expected = set(expected_trainable_names(model))
    if not expected:
        raise RuntimeError("E13 trainable allowlist is empty")
    for name, parameter in model.unet.named_parameters():
        parameter.requires_grad_(name in expected)
    model._ba_expected_trainable_names = tuple(sorted(expected))
    assert_trainable_contract(model)
```

Source: `e13_contract.py`, clean lines 258-274.

```python
def assert_trainable_contract(model, optimizer=None) -> dict:
    expected = set(expected_trainable_names(model))
    actual = {
        name for name, parameter in model.unet.named_parameters()
        if parameter.requires_grad
    }
    non_unet = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("unet.")
    }
    if actual != expected or non_unet:
        raise RuntimeError(
            "E13 trainable ownership mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, "
            f"non_unet={sorted(non_unet)}"
        )
```

The optimizer is checked against the same object identities, preventing a
correct `requires_grad` set from being wired into an incorrect optimizer:

Source: `e13_contract.py`, clean lines 289-297.

```python
if optimizer is not None:
    expected_ids = {id(named[name]) for name in expected}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }
    if optimizer_ids != expected_ids:
        raise RuntimeError("Optimizer membership differs from the E13 allowlist")
```

### 11.3 Error fix: incomplete/permissive checkpoints become schema-v2 exact

June saved only `lora_adapter` plus selected processor tensors; the trained
PhotoMaker `default` adapter was absent:

Source: `lora2.py`, June lines 265-269.

```python
def get_state_dict(self):
    lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
    state = {
        'lora_weights': lora_weights,
    }
```

Its processor load also accepted missing state:

Source: `lora2.py`, June lines 301-304.

```python
for name, sd in state_dict.get("attn_processors", {}).items():
    proc = self.unet.attn_processors.get(name)
    if proc is not None and hasattr(proc, "load_state_dict"):
        proc.load_state_dict(sd, strict=False)
```

The clean saver stores the entire exact trainable allowlist and its
architecture manifest:

Source: `e13_contract.py`, clean lines 419-432.

```python
def get_state_dict(model) -> dict:
    # 10 Aug 2026 - E13C-CORE-04: Save the complete requires-grad allowlist,
    # including both outer adapters; the June subset saver lost trained paths.
    assert_trainable_contract(model)
    named = dict(model.unet.named_parameters())
    names = expected_trainable_names(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "state_format": STATE_FORMAT,
        "architecture": architecture_manifest(model),
        "trainable_unet": {
            name: named[name].detach().cpu().clone() for name in names
        },
    }
```

Load rejects a wrong schema, architecture, or tensor-name set before copying:

Source: `e13_contract.py`, clean lines 503-515.

```python
def load_state_dict(model, state: dict) -> None:
    if int(state.get("schema_version", 1)) != SCHEMA_VERSION:
        raise RuntimeError("The clean E13 family accepts schema-v2 checkpoints only")
    if state.get("state_format") != STATE_FORMAT:
        raise RuntimeError(f"Unknown E13 state format: {state.get('state_format')!r}")
    current = architecture_manifest(model)
    _validate_compatible_manifest(state.get("architecture") or {}, current)
    received = state.get("trainable_unet")
    if not isinstance(received, dict):
        raise RuntimeError("Schema-v2 checkpoint is missing trainable_unet")
    expected = set(current["trainable_names"])
    if set(received) != expected:
        raise RuntimeError("Schema-v2 checkpoint trainable names do not match E13")
```

### 11.4 Error fix: E13 validation shadow is explicit and counted

The alternate RealVis validation model snapshots the pretrained PhotoMaker
default before loading the complete checkpoint, requires exactly 700 tensors,
and restores that snapshot afterward:

Source: `base_trainer.py`, clean lines 591-612.

```python
shadow_default = None
if bool(getattr(self.config, "validation_shadow_photomaker_default", False)):
    # 10 Aug 2026 - E13C-CORE-05: E13 validation measures
    # BA + generic adapters over the pretrained PhotoMaker
    # default, not over the jointly trained default adapter.
    shadow_default = _photomaker_default_snapshot(_val_model)
    if len(shadow_default) != 700:
        raise RuntimeError(
            "E13 validation shadow expected 700 PhotoMaker tensors, "
            f"got {len(shadow_default)}"
        )
try:
    state = self.accelerator.unwrap_model(self.model).get_state_dict()
except Exception:
    state = self.model.get_state_dict()
if hasattr(_val_model, "load_state_dict_"):
    _val_model.load_state_dict_(state)
if shadow_default is not None:
    _restore_photomaker_default(_val_model, shadow_default)
```

### 11.5 Speed fix: resolve Diffusers processors once, skip disabled work

The optimized collector returns before any processor lookup when CL27 is off,
then resolves Diffusers' recursive property once outside the layer loop:

Source: `lora2_helpers.py`, clean lines 55-68.

```python
def collect_frequency_surface_aux_loss(model):
    """Return CL27's live loss graph and already-required detached metrics."""
    if not bool(getattr(model, "ba_frequency_surface_loss_enabled", False)):
        return None, {}
    # 18 Aug 2026 - The fixed pipeline resolves Diffusers' recursive processor
    # property once, never once per selected attention layer.
    processors = model.unet.attn_processors
    grouped: dict[str, list[dict[str, torch.Tensor]]] = {}
    top_losses, floor_losses, applied = [], [], []
    for name in getattr(model, "_ba_patched_processor_names", ()):
        processor = processors.get(name)
        if not bool(getattr(processor, "frequency_surface_loss_enabled", False)):
            continue
```

### 11.6 `cosm_new1_vast` to corrected Cosmic: break positional copying

The corrected target-frame compositor displaces the reference face instead of
always placing it exactly over the target face:

Source: `reference_frame.py`, clean lines 102-110.

```python
ref_cx, ref_cy = _center(scaled_bbox)
tgt_cx, tgt_cy = _center(target_bbox)
# CL9: displacing the paste centre breaks the positional copy shortcut. With
# the reference face landing exactly on the target face every sample, the
# branch can satisfy training by copying in place, which does not transfer to
# validation where the composition differs.
tgt_cx += float(position_offset[0]) * canvas_size
tgt_cy += float(position_offset[1]) * canvas_size
offset_x = ref_cx - tgt_cx
offset_y = ref_cy - tgt_cy
```

It also binds realized target-dependent geometry into the conditioning cache
key, preventing two targets sharing one reference path from sharing the wrong
cached conditioning:

Source: `reference_frame.py`, clean lines 177-183.

```python
# AICODE-NOTE: This descriptor must reach `reference_cache_key`. The composed
# reference depends on the *target* box, so two samples sharing a reference
# path do not share conditioning.
descriptor = (
    f"target_face_frame;canvas={canvas_size};fill={fill};"
    f"scale={scale:.6g};win={window_x0},{window_y0}"
)
```

### 11.7 Repaired baseline to E13: hard spatial reference route and ownership

E13 fixes the scientific route at zero pose substitution and no face CA
mixing:

Source: `attn_processor_cleanest.py`, clean lines 641-644 and 670-691.

```python
runtime = cross_attention_kwargs if isinstance(cross_attention_kwargs, dict) else {}
POSE_ADAPT_RATIO = 0.0 # hardcoded to 0.0 for simplicity
CA_MIXING_FOR_FACE = False # hardcoded to False for simplicity
```

```python
# Extract face regions from both noise and reference
noise_face_hidden = noise_hidden * mask_flat  # Face from current noise
ref_face_hidden = ref_hidden * ref_mask_flat

# Blend them to allow pose adaptation while preserving identity
# Higher POSE_ADAPT_RATIO = more pose flexibility, less identity preservation
face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden

# Just use the blended face directly (previously had option for CA_MIXING_FOR_FACE but removed for simplicity)
key_face = self._k_ref(attn, face_hidden_mixed)
value_face = self._v_ref(attn, face_hidden_mixed)

key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

if mask_gate is  None:
    raise ValueError("Branched attention requires a mask for the face branch")

q_face = q * mask_gate # face area of noise_hidden
hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, dropout_p=0.0, is_causal=False)
```

The configuration/ownership change around that equation is rank-128 branched
SA plus effective generic/default adapters, represented by the exact allowlist
and three named optimizer groups from Section 11.2.

### 11.8 E13 to BC_E13: dataset-only inheritance

BC_E13 inherits the complete E13 leaf and changes only the dataset selector:

Source: `BC_E13_big_celebs_joint_shadow_sa128_24k.yaml`, clean lines 1-6.

```yaml
defaults:
  - E13_large_ds_joint_shadow_sa128_24k
  - _self_

# 10 Aug 2026 - E13C-CFG-02/DATA-02: Dataset-only E13 transfer.
train_dataset_name: big_celebs
```

The shared loader fails closed when a distinct same-ID reference is absent:

Source: `large_dataset.py`, clean lines 121-131.

```python
if self.train_on_separate_image:
    candidates = [
        path
        for path in self.same_id_ref_map[identity]
        if path != target_path
    ]
    if not candidates:
        if self.singleton_reference_policy != "self":
            raise ValueError(
                f"No distinct same-ID reference for {target_path!r}"
            )
```

### 11.9 E13 to CL14: corrected Cosmic data plus training-mask feather

The leaf exposes the two and only two named changes:

Source: `CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`, clean lines 1-10.

```yaml
defaults:
  - E13_large_ds_joint_shadow_sa128_24k
  - _self_

# 10 Aug 2026 - E13C-CFG-02/DATA-03/CORE-06: CL14 changes the training
# dataset policy and feathers the target training mask by two latent cells.
# Its validation/inference pipeline is otherwise exactly the sealed CL14 path.
train_dataset_name: cosmic_large_adapted
model:
  ba_training_mask_feather: 2
```

The feather is constructed only in the training mask:

Source: `lora2.py`, clean lines 1083-1099.

```python
mask[:, :, y_start:y_end, x_start:x_end] = 1.0
feather = int(getattr(self, "ba_training_mask_feather", 0))
if feather > 0:
    # 10 Aug 2026 - E13C-CORE-06: CL14 feathers only the target mask
    # used by training. Reference masks and inference masks remain
    # unchanged, preserving the historical CL14 generation path.
    for step in range(1, feather + 1):
        weight = step / float(feather + 1)
        ys, ye = y_start + step - 1, y_end - step + 1
        xs, xe = x_start + step - 1, x_end - step + 1
        if ye <= ys or xe <= xs:
            break
        mask[:, :, ys, xs:xe] = weight
        mask[:, :, ye - 1, xs:xe] = weight
        mask[:, :, ys:ye, xs] = weight
        mask[:, :, ys:ye, xe - 1] = weight
return mask
```

### 11.10 CL14 to CL14_CA: bounded residual ID-token CA

The zero-init safety fix clamps the mean square before `sqrt`, and the only
scientific output delta is gated, face-local, and added over native CA:

Source: `residual_identity_ca_processor_v3.py`, clean lines 278-300.

```python
identity_delta = self.id_delta_out(identity_hidden)
# The clamp prevents sqrt'(0) from producing NaN on the zero-init step.
delta_rms = identity_delta.float().square().mean(
    dim=-1, keepdim=True
).clamp_min(self.rms_epsilon**2).sqrt()
normalized_delta = identity_delta / delta_rms.to(identity_delta.dtype)
gate = torch.sigmoid(self.gate_logit) * self.gate_max
target_mask = self._prepare_spatial_mask(
    target_len=target_hidden.shape[1],
    batch_size=batch_size,
    device=native_target.device,
    dtype=native_target.dtype,
)
# 13 Aug 2026 - AICODE-NOTE: CL14_CA-CORE-01 keeps native
# PhotoMaker/text CA complete; this bounded face-local ID term is the
# only scientific delta.
residual_message = (
    target_mask * gate.to(native_target.dtype)
    * normalized_delta.to(native_target.dtype)
)
hidden_states = torch.cat(
    [native_target + residual_message, native_reference], dim=0
)
```

### 11.11 CL14 to CL18: training-only cross-view consistency

CL18 freezes the primary prediction as teacher and penalizes a second
same-identity spatial-reference prediction only inside the face mask:

Source: `lora2.py`, clean lines 768-786.

```python
face = mask4.float()
if face.shape[-2:] != noise_pred.shape[-2:]:
    face = F.interpolate(face, size=noise_pred.shape[-2:], mode="nearest")
teacher_face = noise_pred.detach().float() * face
student_face = student_pred.float() * face
smooth_map = F.smooth_l1_loss(
    student_face,
    teacher_face,
    reduction="none",
)
smooth = (smooth_map * face).sum() / (
    face.sum() * student_face.shape[1]
).clamp_min(1.0)
cosine = F.cosine_similarity(
    student_face.flatten(1),
    teacher_face.flatten(1),
    dim=1,
).mean()
crossview_loss = smooth + 0.10 * (1.0 - cosine)
```

The leaf pins sampling probability `0.25` and loss weight `0.05`; both default
off outside CL18.

### 11.12 CL14 to CL19: compute two full messages, blend exactly once

CL19 replaces the hard query/merge route with complete native and reference
messages followed by the two-cell cosine router:

Source: `attn_processor_cleanest.py`, clean lines 471-488.

```python
def _call_soft_router(self, attn, hidden_states, temb) -> torch.Tensor:
    # 12 Aug 2026 - AICODE-NOTE: CL19 computes full native and full
    # target-Q/reference-KV messages, then applies one cosine blend. The
    # reference key mask remains binary, preserving the historical sinks.
    residual = hidden_states
    target, reference, input_ndim, spatial = self._normalized_halves(
        attn, hidden_states, temb
    )
    native_out, reference_out = self._full_target_lanes(
        attn, target, reference
    )
    router = self._soft_router_mask(
        self.mask, target.shape[1], target.shape[0], native_out.dtype
    )
    target_out = native_out * (1.0 - router) + reference_out * router
    return self._finish_full_router(
        attn, residual, target_out, reference, input_ndim, spatial
    )
```

### 11.13 CL14 to CL20: model unchanged, sealed curriculum substituted

The leaf changes only dataset selection and disables shuffling so row order is
the experiment:

Source: `CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml`, clean lines 1-7.

```yaml
defaults:
  - subject_v2_extension_24k
  - _self_

# 12 Aug 2026 - CL20 changes only the sealed training schedule.
train_dataset_name: cl20_hardcase_curriculum
train_dataloader_shuffle: false
```

The consumer rejects hash drift, malformed step ownership, unknown sources,
and self-reference:

Source: `cl20_hardcase_curriculum.py`, clean lines 73-83.

```python
actual_hash = _sha256(self.schedule_path)
if actual_hash != str(expected_schedule_sha256).lower():
    raise RuntimeError(
        f"CL20 schedule hash mismatch: expected={expected_schedule_sha256}, "
        f"actual={actual_hash}"
    )
summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
if summary.get("schedule_sha256") != actual_hash:
    raise RuntimeError("CL20 summary does not seal the schedule bytes")
```

Source: `cl20_hardcase_curriculum.py`, clean lines 93-104.

```python
rows = []
with self.schedule_path.open(encoding="utf-8") as handle:
    for index, line in enumerate(handle):
        row = json.loads(line)
        if set(row) != FIELDS or int(row["index"]) != index:
            raise ValueError(f"Malformed CL20 schedule row {index}")
        if int(row["optimizer_step"]) != index // 2:
            raise ValueError(f"CL20 row {index} has the wrong optimizer step")
        if row["source"] not in {"cosmic", "big_celebs"}:
            raise ValueError(f"Unknown CL20 source at row {index}")
        if row["target_path"] == row["reference_path"]:
            raise ValueError(f"CL20 self-reference at row {index}")
```

### 11.14 CL19 to CL23: fixed temporal-frequency routing

CL23 keeps CL19's two full messages and router, but splits their difference and
applies fixed early-to-late gains from real denoising progress:

Source: `attn_processor_cleanest.py`, clean lines 498-513.

```python
native_out, reference_out = self._full_target_lanes(attn, target, reference)
router = self._soft_router_mask(
    self.mask, target.shape[1], target.shape[0], native_out.dtype
)
low, high = self._gaussian_split(reference_out - native_out)
progress = self._progress(target)
low_scale = self.hardcase_frequency_low_early + progress * (
    self.hardcase_frequency_low_late - self.hardcase_frequency_low_early
)
high_scale = self.hardcase_frequency_high_early + progress * (
    self.hardcase_frequency_high_late - self.hardcase_frequency_high_early
)
low_component = router * low_scale * low
high_component = router * high_scale * high
routed_delta = low_component + high_component
target_out = native_out + routed_delta
```

The leaf pins low `0.50 -> 0.85`, high `0.75 -> 1.25`, disables full-activation
telemetry, and adds no parameters.

### 11.15 CL23 to CL27: inference unchanged, surface loss added in training

The processor computes the two differentiable CL27 terms only with training
gradients and an ownership mask:

Source: `attn_processor_cleanest.py`, clean lines 395-413.

```python
eligible = (top.sum(dim=(1, 2)) > 0.0) & (visible.sum(dim=(1, 2)) > 0.0)
eligible_float = eligible.float()
eligible_count = eligible_float.sum().clamp_min(1.0)
top_high = self._masked_mean_square(high_component, top)
top_low = self._masked_mean_square(low_component, top)
routed_rms = self._masked_mean_square(routed_delta, visible).clamp_min(1e-12).sqrt()
native_rms = self._masked_mean_square(native_out, visible).clamp_min(1e-12).sqrt()
ratio = routed_rms / native_rms.detach().clamp_min(1e-6)
# 18 Aug 2026 - AICODE-NOTE: CL27 eligibility remains on-device; a
# Python bool here synchronizes CUDA once per selected processor.
top_loss = (
    (top_high + self.frequency_surface_top_low_band_factor * top_low)
    * eligible_float
).sum() / eligible_count
floor_loss = (
    F.relu(ratio.new_tensor(self.frequency_surface_visible_floor_ratio) - ratio).square()
    * eligible_float
).sum() / eligible_count
self._frequency_surface_aux_loss = (top_loss, floor_loss)
```

The model adds them after the unchanged CL23 primary forward at weights `0.02`
and `0.005`. The feature is defaults-off, so CL23 inference and all earlier
leaves remain unchanged.

## 12. Evidence and confidence

| Claim | Confidence | Basis |
|---|---|---|
| `2157eada...` is the last 2 June `main_clean` commit | high | local Git log and full commit timestamp |
| `cosm_new1_vast` is the historical config baseline | high | immutable Comet key, name, start time and 336 resolved parameters match the historical launcher |
| `cosm_new1_vast` used committed `2157eada...` rather than a later commit | high, not cryptographic | run predates the next commit by 9 h 16 min; no intervening commit; launcher is unchanged; Comet did not record Git SHA |
| the 2 June installer can fail open and leave unintended trainables | high | direct source inspection plus later historical log/ownership audit |
| the 2 June checkpoint omits trained PhotoMaker-default tensors | high | direct `get_state_dict()` inspection |
| E13/BC_E13/CL14 share the same hard-SA equation | high | clean config, contract and processor inspection |
| BC_E13 differs from E13 only by the training dataset | high | leaf/config and loader inspection |
| CL14 separates Cosmic data fixes from its target training-mask construction | high | dataset/model call-site inspection |
| the original Cosmic reference scale caused the undersized-face tail | high | `2.12x` code/geometry audit plus matched CL4/CL8/CL9 intervention |
| CL2's exact scale-and-position alignment enabled a positional-copy shortcut | medium-high | CL2 self-regression and CL9 recovery with the same compositor/fill; scale and position were randomized together |
| CL18 and CL27 are training-only inference-preserving additions | high | guards, validation wrapper and leaf inspection |
| CL19 and CL23 change inference routing without adding parameters | high | processor formulas and ownership validator |
| optimized processor lookup is scientifically neutral | high for code dependency; not freshly benchmarked here | detached/disabled collector inspection and prior matched speed report |
| clean branch generations equal historical CL14/CL23/CL27 pixels | not established by this document | static/fixture parity exists; no fresh A100 checkpoint/full-96 replay was run |

## 13. What is not established

- This document does not claim that every later experiment improves image
  quality. It describes exact code differences and lineage.
- The old Comet experiment did not record a Git SHA. The attribution to
  `2157eada...` is strongly supported by chronology and config/source matching,
  but cannot exclude an uncommitted local edit that left no surviving record.
- `cosm_new1_vast` is the baseline configuration, not a claim that its old
  Cosmic dataset policy, branched CA, optimizer ownership, checkpointing, or
  validation behavior should be reused in a new run.
- The branch has static, config, ownership, source-hash, and focused
  processor-fixture evidence, but no fresh A100 full-96 RGB replay for all
  recipes.
- CL14's constructed `1/3,2/3` mask is not a continuous inference router; the
  hard threshold makes its effective route binary.
- CL14 does not fix Cosmic's missing stable multi-view identity structure or
  restore full-body examples rejected by `min_face_res=192`.
- Subject-v2 changes validation identity ownership; it must not be mixed with
  legacy metrics as though the metric contract were unchanged.
- Batched bf16 conditioning is the historical E13 execution path and is
  semantically equivalent, but it is not bit-identical to the slower scalar
  conditioning order.
- CL27's promoted historical checkpoint was 16k; the clean YAML intentionally
  describes the full 24k training recipe and does not silently fetch or resume
  that checkpoint.

## 14. Reproducing the lineage audit

```bash
git show -s --format=fuller \
  2157eada14824d14019e80f9416e6d736c837306

git log main_clean \
  --since='2026-06-02 00:00:00 +0100' \
  --until='2026-07-04 23:59:59 +0100' \
  --format='%H %cI %s'

git diff --quiet \
  2157eada14824d14019e80f9416e6d736c837306 main_clean -- \
  diffusion_template/serv_new_runs/start_ba_cosm_new1_vast.sh

set -a
source .env
set +a
COMET_KEY=b7602f92bca54ba5aa0c189192d17165
curl -fsS -H "Authorization: ${COMET_API_KEY}" \
  "https://www.comet.com/api/rest/v2/experiment/metadata?experimentKey=${COMET_KEY}"
curl -fsS -H "Authorization: ${COMET_API_KEY}" \
  "https://www.comet.com/api/rest/v2/experiment/parameters?experimentKey=${COMET_KEY}"

git log --oneline --reverse \
  2157eada14824d14019e80f9416e6d736c837306..clean

git diff --numstat \
  2157eada14824d14019e80f9416e6d736c837306..clean \
  -- diffusion_template/src diffusion_template/train.py

rg -n 'E13C-|CL14_CA-|AICODE-NOTE|CL18|CL19|CL20|CL23|CL27' \
  diffusion_template/src/model/photomaker_branched \
  diffusion_template/src/datasets \
  diffusion_template/src/pipelines
```

## References

- [2 June baseline commit](https://github.com/kolyangg/rsrch/commit/2157eada14824d14019e80f9416e6d736c837306)
- [Historical baseline launcher at the 2 June commit](https://github.com/kolyangg/rsrch/blob/2157eada14824d14019e80f9416e6d736c837306/diffusion_template/serv_new_runs/start_ba_cosm_new1_vast.sh)
- [`cosm_new1_vast` immutable Comet experiment](https://www.comet.com/nikolay-2104/rsrch-30oct/b7602f92bca54ba5aa0c189192d17165)
- [Original clean implementation ledger](2026-08-10_e13_family_clean_implementation.md)
- [Original port plan and June comparison](2026-08-10_e13_bc_e13_cl14_clean_port_plan.md)
- [Cosmic scale-mismatch root-cause audit](https://github.com/kolyangg/rsrch/blob/a61cb773cdc6b947072f3ddd8476cada33dc0ce5/diffusion_template/analysis/2026-08-06_cosmic_large_vs_large_dataset_root_cause_and_cl1_cl3_plan.md)
- [Cosmic face-scale and CL9 experiment design](https://github.com/kolyangg/rsrch/blob/a61cb773cdc6b947072f3ddd8476cada33dc0ce5/diffusion_template/analysis/2026-08-08_cl_face_scale_root_cause_and_cl8_cl9.md)
- [Matched CL8/CL9 face-scale results](https://github.com/kolyangg/rsrch/blob/a61cb773cdc6b947072f3ddd8476cada33dc0ce5/diffusion_template/analysis/2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md)
- [Formula-level family reference](2026-08-13_e13_family_architecture_reference.md)
- [CL18/CL19/CL20 extension ledger](2026-08-12_cl18_cl19_cl20_clean_extension.md)
- [CL14_CA extension ledger](2026-08-13_cl14_ca_clean_extension.md)
- [CL23/CL27 extension ledger](2026-08-18_cl23_cl27_clean_extension.md)
- [Current handoff](../handoffs/LATEST.md)
- [Validation protocol](../validation_protocol.md)
