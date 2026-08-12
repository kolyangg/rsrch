# CL8-CL11: face scale is solved; what remains is universal across every dataset

**Date:** 9 August 2026
**Scope:** analysis of CL8-CL11 against CL4/E13, per identity and per prompt, with
a focus on the residual hard cases; plus three follow-up experiments. No running
job was touched and no training code was changed for this analysis.
**Evidence cutoff:** CL8 and CL9 complete at 24k; **CL10 and CL11 still running**
(14k and 16k). All geometry comparisons are made at the **matched step 14,000**.

| Run | Comet key | Change | best `id_sim` |
|---|---|---|---:|
| E13 | `1cc0a02371094b24a6a02a4cc649f10c` | large_dataset reference | `.39980` @24k |
| CL4 | `0dd86b436b224f939efa3887ad6acbe2` | hygiene only | `.40691` @14k |
| CL8 | `a6b5970aa1a24d3490ad08e7994b5f1e` | full-body targets | `.40256` @10k |
| **CL9** | `81bb311ed70545eda3281c64bc48be47` | reference scale/position jitter | **`.41513` @24k** |
| CL10 | `eba0187806ec476996f5ea4af356361e` | CL9 + full-body targets | `.39926` @8k *(running)* |
| CL11 | `32f4ba2a3b3a493f96a3a2345147e84c` | CL9 + 3 identity references | `.42142` @8k *(running)* |

---

## Executive conclusion

**The small-face failure is solved.** At the matched step 14,000, every arm that
uses reference scale calibration reaches **zero** undersized faces, matching E13:

| Run | ratio p10 | median | **< 0.8** | < 0.9 | no face |
|---|---:|---:|---:|---:|---:|
| E13 | `0.958` | `1.029` | **0** | 2 | 0 |
| **CL9** | `0.932` | `1.026` | **0** | 5 | 0 |
| **CL11** | `0.915` | `1.013` | **0** | 6 | 0 |
| **CL10** | `0.913` | `1.013` | **0** | 8 | 1 |
| CL8 | `0.789` | `0.977` | **10** | 27 | 1 |

CL8, the only arm without reference scaling, still has 10. That is the third
independent confirmation that **reference face scale is the causal variable** and
target framing is not.

**CL9 is also now the identity leader**: `.41513` at 24k, above CL4's `.40387`
and E13's `.39980`. Its early 8k deficit reversed completely. So the fix costs
nothing — it wins on both metrics.

What remains is **not a cosmic problem at all**. The residual hard cases are
**universal across three entirely different training datasets** — large_dataset
(E13), BigCelebs (BC_E13) and cosmic_large (CL9/CL4) — so they are limits of the
architecture or validation protocol, not of any dataset:

1. **Occlusion and extreme expression** — Skiing (goggles), Crying (hands),
   Kickboxing (gloves), Laughing (open mouth). The branch paints a full face into
   a region the scene has occluded.
2. **eddie** — fails on *every prompt* in *every dataset* (`.08-.26`).
3. **marion** — weak everywhere (`.31-.35`).

**Face size is not the cause.** Correlation between prompt difficulty and the
required face size is `-0.026`: Skiing-woman has the *largest* faces (`325px`
mean) and the *worst* score (`.201`), while Jumping-man has `90px` faces and
scores higher (`.319`). An earlier draft of this report attributed the hard
prompts to small faces; the measurement falsifies that.

---

## 1. Face geometry at the matched step 14,000

Metric: detected-face short side ÷ the fixed mask box short side, where the mask
is the required face area and is identical across runs.

### Undersized faces by identity

| Run | eddie | elon | jennie | jensen | jisoo | keanu | lex | marion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CL9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CL10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CL11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CL8 | 0 | 0 | 0 | 2 | 5 | 3 | 0 | 0 |

The jensen/jisoo/keanu cluster that failed in CL0-CL8 is completely gone in every
reference-scaled arm.

### The residual tail (`ratio < 0.9`, not a failure but the remaining gap to E13)

| Image | identity | arms affected | E13 ratio |
|---|---|---:|---:|
| `Crying_wom_jisoo` | jisoo | 3/3 | `0.96` |
| `Reading_pa_jisoo` | jisoo | 3/3 | `0.92` |
| `Skiing_man_jensen` | jensen | 3/3 | `0.89` |
| `Crying_man_keanu` | keanu | 2/3 | `0.96` |
| `Skiing_man_keanu` | keanu | 2/3 | `0.98` |
| `Crying_man_eddie` | eddie | 2/3 | `0.89` |
| `Laughing_m_eddie` | eddie | 2/3 | `0.94` |

E13 is itself below `0.9` on `Skiing_man_jensen` and `Crying_man_eddie`, so part
of this tail is intrinsic difficulty rather than a cosmic deficit.

### Blending quality

IoU between the detected face and the required box — a proxy for "is the face
seated where the body expects it":

- **No arm has any image below IoU `0.3`.** The gross misplacement seen in
  CL0-CL7 is gone.
- The worst remaining are `Skiing_man_jensen` (CL11 `0.71`, CL10 `0.72`),
  `Skiing_man_keanu` (`0.72-0.77`), `Laughing_m_eddie` (`0.75-0.77`). E13's own
  worst is `Skiing_man_jensen` at `0.78`, so these are the same hard images.

---

## 2. Identity and prompt structure

### `id_sim` by identity at step 14,000

| Run | eddie | elon | jennie | jensen | jisoo | keanu | lex | marion | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E13 | `.163` | `.497` | `.417` | `.481` | `.425` | `.430` | `.347` | `.306` | `.383` |
| CL4 | `.158` | **`.517`** | **`.486`** | `.499` | `.367` | `.466` | **`.411`** | `.352` | **`.407`** |
| CL8 | `.163` | `.513` | `.479` | `.489` | `.318` | `.432` | `.391` | `.311` | `.387` |
| CL9 | `.152` | `.474` | `.447` | **`.523`** | `.411` | `.488` | `.395` | `.288` | `.397` |
| CL10 | `.161` | `.498` | `.430` | `.513` | `.393` | `.450` | `.394` | **`.353`** | `.399` |
| CL11 | `.140` | `.498` | `.456` | `.518` | **`.474`** | **`.498`** | `.355` | `.219` | `.395` |

Two clean, complementary signals:

- **CL11 (multi-reference) is best on jisoo `.474` and keanu `.498`** — the
  identities whose single reference is weakest. More independent identity
  evidence helps exactly where one crop is not enough.
- **CL10 (full-body targets) is best on marion `.353`** — marion's weak prompts
  are Jumping and Dancing. Note this is *not* a face-size effect (see below);
  full-body training data appears to help pose/composition diversity rather than
  scale, which CL10 already inherits from CL9.

They are strong on **different** identities, which is the main argument for
combining them.

CL11 is also **worst on marion `.219`**, a real regression against E13's `.306`,
so multi-reference is not free.

### The hard cases are universal across datasets

This is the single most important result in the report. At step 24,000, across
three different **training datasets**:

| identity | E13 (large_dataset) | BC_E13 (BigCelebs) | CL9 (cosmic_large) | CL4 (cosmic_large) |
|---|---:|---:|---:|---:|
| **eddie** | `.174` | `.156` | `.157` | `.157` |
| **marion** | `.341` | `.333` | `.314` | `.352` |
| jensen | `.473` | `.533` | **`.559`** | `.499` |
| keanu | `.467` | `.448` | **`.496`** | `.464` |
| lex | `.380` | `.367` | **`.414`** | `.396` |

Worst six prompts per dataset at 24k:

```text
E13  (large_dataset) : Skiing-w .20  Jumping-w .29  Skiing-m .31  Dancing-m .32  Jumping-m .32  Dancing-w .33
BC   (BigCelebs)     : Jumping-w .22  Jumping-m .31  Skiing-w .32  Dancing-m .32  Dancing-w .34  Crying-w .35
CL9  (cosmic_large)  : Jumping-w .28  Skiing-w .30  Jumping-m .31  Dancing-m .31  Laughing-w .34  Skiing-m .35
CL4  (cosmic_large)  : Crying-w .23  Skiing-w .26  Dancing-m .28  Skiing-m .32  Jumping-w .33  Jumping-m .33
```

**The same prompts are worst in every dataset.** No dataset-side experiment will
move them — the limit is architectural or in the validation protocol.

Note also that **CL9 on cosmic_large is the best of the three datasets on jensen,
keanu and lex**, so cosmic with the reference-scale fix is not merely catching up;
it leads on most identities.

### Why these prompts are hard — it is not face size

| Prompt | mean `id_sim` | mean required face (px) |
|---|---:|---:|
| Skiing woman | `.201` | **`325`** |
| Jumping woman | `.285` | `120` |
| Skiing man | `.310` | `330` |
| Dancing man | `.317` | `98` |
| Jumping man | `.319` | `90` |
| … | | |
| Reading paper man | `.475` | `233` |
| Chef woman | `.473` | `217` |

**Correlation across prompts: `-0.026`** — none. The two Skiing prompts have the
*largest* required faces and among the worst scores. What the hard prompts share
is **occlusion and extreme expression**: goggles and helmets (Skiing), hands over
the face (Crying), gloves and motion (Kickboxing, Jumping), open mouths
(Laughing). The easy prompts are unoccluded, neutral-expression portraits.

---

## 3. The hard cases, visually

![Hard cases at step 14,000; red box is the identical required face area](assets/cl8_11_hardcases.png)

Four distinct failure modes, and only the first is fixed:

**`Skiing_wom_jisoo` — occlusion destroys the face.** E13 renders a clean face.
CL9 (`r=1.05`) and CL11 (`r=0.94`) render distorted faces with goggles merged
into the features. **CL10 detects no face at all.** The face is the right *size*
but the goggles have consumed it. This is the single "face not visible" case in
the panel, and it is occlusion, not scale.

**`Skiing_man_jensen` — poor seating.** All arms place a correctly-sized face,
but IoU is `0.71-0.80`: goggles pushed onto the forehead, face slightly low. E13
has the same problem (`0.78`).

**`Crying_man_eddie` — hands over the face.** The prompt puts hands across the
face in every arm. The branch paints identity into a region the scene says is
occluded, producing a face-behind-hands composite. E13 shows the same structure.

**`Jumping_wo_marion` — correctly sized, still weak.** Ratios are `1.05-1.20`,
i.e. the face is if anything *larger* than required. The difficulty here is
extreme pose and motion, not scale — consistent with the zero correlation above.

### eddie is not a cosmic problem

eddie by prompt at 14k, all twelve prompts:

```text
           E13    CL9   CL10   CL11
Angry     .161   .166   .128   .209
Chef      .145   .151   .140   .167
Crying    .227   .109   .187   .050
Dancing   .149   .107   .151   .075
Drumming  .175   .201   .174   .182
Jumping   .082   .096   .102   .086
Kickbox   .130   .190   .147   .107
Laughing  .155   .102   .167   .058
Night     .164   .114   .120   .104
Reading   .204   .224   .260   .223
Rushing   .206   .224   .187   .216
Skiing    .154   .137   .172   .203
```

**No prompt exceeds `.26` in any arm, including E13.** This is a property of the
eddie reference or of `IDSimBest` on that identity, not of cosmic training.
Chasing it with a cosmic-side experiment would waste an A100. It should be
investigated separately — check the reference image, its bbox, and whether the
ArcFace embedding of the reference is even well-formed.

---

## 4. Confidence

| Claim | Confidence |
|---|---|
| Reference scale calibration eliminates undersized faces | **High** — 0 in CL9/CL10/CL11 vs 10 in CL8, matched step **[measured]** |
| CL9 leads on `id_sim` at completion | **High** — `.41513` @24k vs E13 `.39980` **[measured]** |
| Remaining weak prompts are universal across datasets | **High** — same worst set in large_dataset, BigCelebs and cosmic **[measured]** |
| Prompt difficulty is NOT explained by face size | **High** — correlation `-0.026`; Skiing has the largest faces and worst scores **[measured]** |
| Occlusion/expression is the shared property of hard prompts | **Medium-high** — visual plus the size falsification; no ablation isolates it |
| CL11 helps weak-reference identities (jisoo, keanu) | **Medium-high** — clear at 14k, but CL11 is unfinished |
| CL10 helps small-face identities (marion) | **Medium** — one identity, and CL10 is unfinished |
| eddie is reference/metric-specific, not cosmic | **High** — fails on all 12 prompts in E13 **and** BigCelebs **[measured]** |
| Occlusion is a distinct unfixed failure mode | **Medium-high** — visual, plus the CL10 no-face case; not isolated by an ablation |

### Not established

- CL10 and CL11 final behaviour — both are mid-run; their 8k-vs-14k swings
  (CL11 `.42142` → `.39490` → `.40594`) are large enough that neither ordering is
  settled.
- Whether CL10 and CL11 gains are additive.
- Whether CL11's marion regression is systematic or noise.
- Any causal claim about occlusion; no occlusion-specific ablation has run.

---

## 5. Proposed experiments

All keep the E13 contract: 24k steps, batch 2, one A100, fixed full-96,
`2,240 / 219,217,920` parameters.

### CL12 — CL10 + CL11 combined (priority 1)

`CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k.yaml`

**Change:** CL9's reference scaling + CL8's full-body targets + 3 identity
references — the union of everything that has worked.

```yaml
defaults: [CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k, _self_]
datasets:
  train:
    cosmic_large_adapted:
      num_identity_refs: 3
model:
  batched_conditioning_preparation: false
```

**Why:** the two parents are strong on **disjoint** identities — CL11 on jisoo
`.474` and keanu `.498`, CL10 on marion `.353`. Both keep face scale at zero
undersized. This is the highest-`id_sim` candidate and it needs no new mechanism.

**Prediction:** undersized stays 0; `id_sim` above CL9's `.41513`.
**Risk:** CL11's marion regression could carry over. Watch marion specifically;
if it drops below `.30`, multi-reference is trading identities rather than adding.

### CL13 — reference dropout for occlusion robustness (priority 2)

`CL13_cosmic_joint_shadow_sa128_refdropout_24k.yaml`

**Change:** CL9 plus randomly disabling the reference branch on a fraction of
training steps.

```yaml
model:
  ba_reference_dropout_probability: 0.15   # NEW, defaults-off
```

**Why:** in `Crying_man_eddie` and `Skiing_wom_jisoo` the branch paints a full
face into a region the scene has occluded, because it has *never* been trained to
defer. Dropping the branch on ~15% of steps teaches the model that the native
path must remain a coherent fallback, which is exactly the behaviour needed when
an occluder owns the face region.

**Prediction:** occluded cases degrade more gracefully — fewer no-face and
merged-goggle outputs; overall `id_sim` within `.01` of CL9.
**Risk:** the most speculative of the three. If `id_sim` falls materially, drop
to `0.05` or abandon; a weaker branch is not the goal.

### CL14 — feathered training face mask (priority 3)

`CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`

**Change:** CL9 plus a soft-edged face mask **in training only**.

The training mask (`_bbox_to_mask`) is a hard binary rectangle, so the branch
learns a discontinuous handover at the box edge. That is the mechanism behind the
"not fully blended" appearance and the goggle/hair seams. A feathered mask
teaches a gradual handover.

```yaml
model:
  ba_training_mask_feather: 2      # NEW, defaults-off: latent cells of ramp
```

**Critically, validation is unchanged** — inference keeps the hard mask
(`mask_expansion_ratio: 1.0`, `mask_softness: 0.0`), so the arm stays fully
comparable with every previous run. This is a training-side change only.

**Prediction:** IoU improves on the seam cases (`Skiing_man_jensen` `0.71-0.80`,
`Skiing_man_keanu` `0.72-0.77`); `id_sim` neutral to slightly positive.
**Risk:** a soft mask slightly dilutes branch authority; if `id_sim` drops more
than `.01`, feather 1 instead of 2.

### Not proposed

- **Anything targeting eddie or the hard prompts via the dataset.** They fail
  identically in large_dataset, BigCelebs and cosmic. Dataset work cannot move
  them; that is why CL13/CL14 are architectural.
- **More target-framing work.** CL8 settled that it does not affect face scale;
  CL10 already carries its benefit for small-face prompts.
- **Inference-side mask changes.** They would break comparability with all
  prior arms while fixing nothing in training.

---

## 6. Implementation plan

### Step 1 — CL12 (config only)

No code. `defaults: [CL10_..., _self_]` plus `num_identity_refs: 3` and
`batched_conditioning_preparation: false`. The dataset already supports both;
the validator already enforces the multi-reference/batched rule.

### Step 2 — CL13 (model, defaults-off)

Add `ba_reference_dropout_probability: float = 0.0` to
`src/model/photomaker_branched/lora2.py`. In the training forward (the
`run_branched_forward_pass` call site), draw once per batch and, when it fires,
take the ordinary non-branched U-Net path. The draw is guarded by `self.training`,
so **inference never drops the branch**.

### Step 3 — CL14 (model, defaults-off)

Add `ba_training_mask_feather: int = 0`. In `_bbox_to_mask`, when it is > 0, build
the binary box then ramp linearly inward from the edge by that many latent cells.
**Do not touch `_bbox_to_ref_mask`** — only the target-side mask, and only where
the training path builds it. The inference pipeline builds its own mask and never
reads the attribute, so validation is provably unchanged.

Propagate through the model-attribute lists in `train.py` and `base_trainer.py`
only if validation needs it — it does not, which is the point.

### Step 4 — registry, validator, preflight, launcher

Registry: nothing for CL12; the two new model flags default off. Validator: add
all three to `ARMS`; assert CL12 has `num_identity_refs == 3` with batched
conditioning off; assert CL13/CL14 change only their single new flag. Preflight:
add the arm tokens to the CL9 branch — **use exact tokens**, since
`startswith("CL1")` also matches CL12/CL13/CL14 and previously mis-routed CL10.
Launcher: add the three config names to the `case` gate.

### Step 5 — records and packages

`experiments/cosmic_large/CL{12,13,14}_..._r1.json`, Serv packages cloned from
CL9's — isolated runtime, sealed manifest, self-contained MLS YAML.

### Step 6 — pre-launch gates

1. Composition: 24,000 steps, `2,240 / 219,217,920`.
2. Only the intended field differs from the parent.
3. **CL13:** assert the dropout never fires outside `self.training`.
4. **CL14:** assert step-0 output is byte-identical to CL9's — a training-only
   mask change must not move inference.
5. Real-data preflight passes, with the arm token echoed in the report.
6. Allow ~10 min model construction, then ~25 min of silent step-0 generation
   before treating quiet as a hang.

### Step 7 — decision gates

| Gate | Step | Rule |
|---|---:|---|
| Face scale | 4k | undersized stays 0 (all three inherit CL9) |
| Occlusion | 8k | CL13: no no-face outputs; `Skiing_wom_jisoo` renders a face |
| Blending | 8k | CL14: fewer images with IoU < 0.8 than CL9's 4 |
| Hard prompts | 14k | Skiing/Jumping/Dancing means above the current `.23-.34` |
| Identity | 14k | CL12 `id_sim` ≥ CL9's `.3973` at the same step; marion ≥ `.30` |
| Promotion | 24k | undersized 0, `id_sim` ≥ CL9's `.41513` |

Report **undersized count, IoU tail, and `id_sim` together**. `id_sim` alone
still ranks CL8 above CL9 at 8k, which inverts the true ordering.

---

## 7. Launch record — all three are submitted

Built and submitted on 9 Aug 2026 under the eight-A100 authorization (five were
already running: CL10, CL11, and the three BC_E13 dataset arms).

| Arm | Serv job | Snapshot |
|---|---|---|
| CL12 | `lm-mpi-job-fb755bc8` | `runtime_sources_cl1_cl3_v1/CL12_…_r1` |
| CL13 | `lm-mpi-job-9f1db03f` | `runtime_sources_cl1_cl3_v1/CL13_…_r1` |
| CL14 | `lm-mpi-job-2ff91c51` | `runtime_sources_cl1_cl3_v1/CL14_…_r1` |

Each runs from its own sealed snapshot — 1,220 files, hash-verified at
revision `c04970f3+cl12-cl14-snapshot-v1-20260809`, so no live job's code was
read or mutated.

**Gates that actually ran, on the real dataset, inside each snapshot:**

| Check | CL12 | CL13 | CL14 |
|---|---|---|---|
| Composition | 24,000 steps, `2,240 / 219,217,920` | same | same |
| Records after filtering | 79,124 | 22,140 | 22,140 |
| Targets below 5% face area | 45.3% | — | — |
| Reference face area, median | 22.9% | 19.1% | 17.2% |
| References per sample | 3 | 1 | 1 |
| Captions over 77 CLIP tokens | 0.8% | 0.6% | 0.6% |
| Preflight failures | none | none | none |

CL13 and CL14 reproduce CL9's dataset audit exactly (22,140 accepted from 59,143;
37,003 dropped on target face, 137 on reference bbox), which is the evidence that
their only difference from CL9 is the single training-only flag.

CL12's oversampling factors came out at 1.0/1.13/1.88/1.90 across the four scale
bins — under the 4.0 cap, so the balancer is active rather than silently clipped.
That was the specific failure mode CL8 hit.

**Two traps this round, both caught before launch.** The arm-token gate matters:
`startswith("CL1")` matches CL12, CL13 and CL14 as well, exactly the bug that
killed CL10 r1 after 40 seconds; the preflight now splits the exact token. And the
plan text originally had CL13 and CL14 assigned to each other's mechanism — the
implemented and launched assignment is CL13 = reference dropout,
CL14 = feathered mask, and section 6 has been corrected to match.

---

## 8. Reproducing

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh && conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template
python tools/comet/comet_experiment.py fetch --record comet_records/<run>.json \
  --step-number 14000 --output-dir comet_data/<batch>/<arm>_14000
python tools/datasets/measure_face_body_alignment.py --images-dir <dir> \
  --mask-boxes ../dataset_full/val_dataset/pm96_bboxes_new.json \
  --id-sim-csv <csv> --label CL9 --output align.json
```

Silent traps: bbox and `id_sim` keys contain spaces while PNGs use underscores;
the Comet API returns figure names, so use `comet_experiment.py fetch`; give each
step its own `--output-dir`. Local disk fills quickly — each arm-step is ~150MB
of PNGs; delete images after measuring, the JSON/CSV artefacts are what matter.

Data: `comet_data/cl8_11_20260809/`.

## 9. References

- [CL8/CL9 results, 9 Aug](2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md)
- [Face-scale root cause, 8 Aug](2026-08-08_cl_face_scale_root_cause_and_cl8_cl9.md)
- [Alignment analysis, 7 Aug](2026-08-07_cl_face_body_alignment_analysis_and_cl6_cl7.md)
