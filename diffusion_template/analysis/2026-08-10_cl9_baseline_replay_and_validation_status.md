# CL9 baseline replay is exact; intervention jobs failed closed before causal results

**Date:** 10 August 2026  
**Evidence cutoff:** live Serv inspection on 10 August 2026  
**Source run:** `CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1`  
**Immutable Comet experiment:** [`81bb311ed70545eda3281c64bc48be47`](https://www.comet.com/nikolay-2104/aug-large-ds/81bb311ed70545eda3281c64bc48be47)  
**Checkpoint:** exact step `24,000`, `weights-epoch12.pth`, SHA256 `5396993b...7357c3`  
**Scope:** status and results of the fixed-96 baseline replay, Marion conditioning sidecar, and small-face ROI sidecar. The two sidecars intentionally did not create Comet experiments. No training result is reported.

Evidence labels are **[measured]**, **[visual]**, **[code]**, **[report]**, and **[hypothesis]**.

| Arm | Tracking identity | Step | Final status | Usable result |
|---|---|---:|---|---|
| Source CL9 | Comet `81bb311ed70545eda3281c64bc48be47` | `24,000` | complete | current fixed-96 baseline metrics and images |
| Full-96 replay + ROI | MLS `...107f151` | `24,000` | failed after baseline | **`96/96` exact RGB baseline replay** |
| Marion + occlusion | MLS `...210da8dd` | `24,000` | failed at baseline gate | transforms prepared; no valid intervention output |

---

## Executive conclusion

The baseline finished successfully and is exact. The Serv evaluator regenerated all `96` fixed-panel images from the saved CL9 24k checkpoint, and every image matched the historical output pixel-for-pixel: **`96/96` exact RGB, `0` mismatches**. **[measured]** This validates checkpoint loading, the immutable evaluator, seeds, prompt order, references, boxes, batch size, scheduler, and the installed branched-attention processors as a complete sequence.

The baseline result is therefore the existing CL9 24k result, not a new statistical sample. Its primary current mask-owned subject-v2 identity score is **`0.447997`**, face detection is **`96/96`**, and TOPIQ-Face is `0.679632`. **[measured]** The exact replay is strong evidence that later intervention deltas can be attributed to a controlled change, once those interventions run.

Neither causal intervention produced a result. The Marion job generated only a standalone 12-image replay and stopped because it matched `0/12` historical Marion images. The full-96 success rules out a corrupt checkpoint or broadly wrong evaluator; it instead shows that extracting the last 12 examples into a separate sequence is not replay-equivalent. **[measured]** The small-face job passed the full-96 gate, then failed before its first refined image because the installed DDIM scheduler does not accept a custom timestep list. **[code, measured]** Consequently, there is no evidence yet for or against reference roll normalization, occluder-aware routing, or ROI refinement.

---

## 1. Exact baseline result

### 1.1 Reproduction contract

| Fixed item | Replayed value |
|---|---|
| Validation model | `SG161222/RealVisXL_V4.0` |
| Panel | fixed 96-image `manual_val`, original order |
| Loading | `legacy_full_copy`, strict copy, pretrained PhotoMaker shadow enabled |
| Batch / CFG / scheduler | `12` / `5` / DDIM, `50` steps |
| BA invariants | `use_branched_attention=true`, `pose_adapt_ratio=0`, `ca_mixing_for_face=false` |
| Active generation-box SHA256 | `b33cf026...013a2f1c7d` |
| Replay gate | RGB array equality at `1024 x 1024`, every image |

The replay took `1006.5` seconds for generation after model initialization. The recorded gate checked `96` expected outputs and found `96` exact, with no skipped indices. **[measured]** This is stricter than metric agreement: any visible or numerical pixel difference would fail.

### 1.2 Baseline endpoint

The primary metric is `manual_val/id_sim`, using the current mask-owned subject-v2 selection contract. Legacy best-face ID can select a bystander and is not the promotion metric. TOPIQ-Face measures detected-face quality, but can be high when identity is wrong or an occluder hides the eyes.

| Endpoint metric | CL9 at 24k |
|---|---:|
| `manual_val/id_sim`, mask-owned subject-v2 | **`0.447997`** |
| `manual_val/id_sim_legacy_best` | `0.399063` |
| `manual_val/id_sim_mask_iou` | `0.895067` |
| face detection rate | **`1.000` (`96/96`)** |
| no-face / ambiguous / unowned | `0 / 0 / 0` |
| TOPIQ-Face mean / p10 | `0.679632 / 0.580499` |

| Prompt family | n | ID | TOPIQ-Face | face short side |
|---|---:|---:|---:|---:|
| Clean prompts | `48` | **`0.4955`** | `0.6961` | `207.6 px` |
| Crying | `8` | `0.4733` | `0.6664` | `203.3 px` |
| Skiing | `8` | `0.3622` | **`0.7736`** | `316.0 px` |
| Jumping / Dancing | `16` | **`0.3389`** | `0.5424` | **`120.9 px`** |

![CL9 family-level identity, face quality, and rendered face scale](assets/cl9_edge_cases_20260810/fig_cl9_family_profile.png)

The family pattern is unchanged by this replay. Skiing faces are large and sharp but identity drops under eye-region goggles; small-face prompts have far less absolute face resolution; Crying stays closer to clean identity despite hands, hair, tears, and closed eyes. **[measured, visual]**

### 1.3 Marion baseline

Marion remains the difficult reference edge case: mean ID is `0.3112` over her 12 prompts and `0.3653` over her six clean prompts. **[measured]** Her source has a `-7.65` degree eye-line roll and a strong 3/4-view yaw proxy (`0.368`, versus at most `0.093` for the other seven panel identities). **[report]** A deterministic same-file roll correction and five-point similarity transform were prepared, but neither was passed through a valid generation arm.

![Marion source, deterministic roll preview, and representative exact CL9 baseline outputs](assets/cl9_edge_cases_20260810/fig_marion_reference_and_outputs.png)

The transformed reference panel is an input preview, not an intervention result. A same-file transform cannot recover facial evidence hidden by yaw.

---

## 2. Intervention job outcomes

### 2.1 Marion normalization and occluder routing

The job prepared the roll and five-point similarity references, initialized CL9, and generated a 12-image original-reference Marion replay in `128.2` seconds. Its strict gate found **`0/12` exact RGB** and stopped before running either transformed-reference arm or any occlusion arm. **[measured]** This fail-closed behavior is correct: comparing interventions against a non-equivalent baseline would create false deltas.

The exact mechanism behind the 12-image sequence difference is **not established**. Plausible sources include sequence-level random state, pipeline state, or batching context before Marion's final batch. **[hypothesis]** The full-96 replay proves that the source checkpoint, evaluator, and full fixed-panel contract are valid; therefore, broad checkpoint corruption and a general scheduler/config mismatch are not the cause.

### 2.2 Small-face ROI refinement

This job first passed the full-96 exact gate. It then reloaded the model for `CL9V_smallface_roi_refine_24k_r1` and failed before saving the first ROI output. The immutable runtime's `DDIMScheduler.set_timesteps` does not support the custom timestep sequence passed by the sidecar. **[code, measured]** No ROI or gentle-ROI image exists, so no identity, quality, seam, or background-drift conclusion can be drawn.

### 2.3 Comet status

The validation sidecars were intentionally filesystem-only diagnostics, not new Comet-tracked experiments. They have MLS IDs and evidence manifests, but no `comet_experiment.json`. **[code]** The immutable Comet identity for all baseline metrics remains the source CL9 key shown above.

---

## 3. Root cause and confidence

| Claim | Confidence | Basis |
|---|---|---|
| Full-96 baseline is an exact CL9 24k replay | **Very high** | `96/96` RGB arrays match; no skipped indices |
| The saved checkpoint and full evaluator are valid | **High** | exact end-to-end replay under the historical contract |
| The isolated Marion subset is not a controlled baseline | **Very high** | `0/12` exact against the same historical rows |
| The subset difference is caused by RNG or pipeline sequence state | Medium | consistent with full-sequence success, but not isolated experimentally |
| ROI failure is a DDIM API incompatibility | **Very high** | direct traceback before the first refined image |
| Roll normalization, occluder routing, or ROI refinement improves CL9 | Not established | no intervention arm passed its baseline gate and generated results |

### What is not the cause

- The failed Marion subset is not evidence that CL9's checkpoint is broken: the full-96 replay is exact. **[measured]**
- The ROI failure is not evidence that local refinement harms identity or quality: denoising never began. **[measured]**
- The job-level MLS status `failed` does not invalidate the completed full-96 baseline artifact. The exact replay manifest was written before the later controlled failure. **[measured]**
- No result supports changing training yet. The missing evidence is interventional validation, not another aggregate baseline metric. **[report]**

---

## 4. Priority repairs and experiments

### 4.1 `CL9V_marion_occlusion_validation_24k_20260810_r2`

**Single scientific changes:** same-file Marion roll normalization; then frozen-mask face-vs-native ownership on the predefined Crying/Skiing set. Each remains a separate named arm.  
**Implementation:** replay all `96` examples in their historical order for every arm. Keep indices `0-83` as sentinels. Apply the Marion reference transform only at indices `84-95`; apply the occlusion routing change only to its frozen target rows. Do not run a standalone 12-row baseline.  
**Hypothesis:** conditioning roll contributes to Marion's deficit; native ownership of occluder pixels preserves the object while preventing it from replacing face-branch evidence.  
**Prediction:** Marion clean ID improves without global image drift; Skiing/Crying retention improves without removing the requested occluder.  
**Risk:** full-96 arms are slower, and a 2D roll will not correct yaw.  
**Decision gates:** baseline arm must remain `96/96` exact; all 84 sentinel rows must remain exact in targeted arms; no-face or unowned count cannot rise; promote only with issue-set improvement and blinded visual confirmation of prompt/occluder fidelity.

### 4.2 `CL9V_smallface_roi_refine_24k_20260810_r2`

**Single scientific change:** deterministic late local denoising inside the fixed Jumping/Dancing face ROI; retain separate standard and gentle strengths.  
**Implementation:** keep the standard DDIM 50-step grid and scheduler definition. Build the late suffix from that grid inside a bounded sidecar denoising loop instead of calling `set_timesteps(timesteps=...)`. First run one ROI smoke case and verify output shape, scheduler state, determinism, and unchanged pixels outside the blend support. Then run the gated fixed panel.  
**Hypothesis:** extra local pixel/latent budget improves identity at a fixed final face size.  
**Prediction:** Jumping/Dancing ID and TOPIQ-Face improve, with low seam energy and negligible background drift.  
**Risk:** visible blending seams, texture oversharpening, or composition drift.  
**Decision gates:** exact full-96 baseline first; deterministic repeat of the smoke case; no new face-count failures; outside-ROI drift below the predefined threshold; promote only if both ID and visual seam review pass.

Both repaired jobs preserve RealVisXL, the fixed 96 prompts/seeds/references/boxes, DDIM50, CFG `5`, BA routing, `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and the CL9 trainable contract. They are validation-only and do not change trainable parameters.

---

## 5. Reproducing and auditing

From `diffusion_template/`:

```bash
ROI_JOB='lm-mpi-job-5dc7cc57-c622-48a5-'\
'a80d-16607107f151'
MARION_JOB='lm-mpi-job-bcff7ec7-e2b5-47de-'\
'87ac-ca37210da8dd'

python3 ../local_scripts/serv_job.py inspect "$ROI_JOB" --lines 30
python3 ../local_scripts/serv_job.py inspect "$MARION_JOB" --lines 30

jq '{expected_count, checked_count, exact_count, mismatch_count}' \
  tmp/cl9_baseline_status_20260810/smallface/replay_verification.json

jq '{expected_count, checked_count, exact_count, mismatch_count}' \
  tmp/cl9_baseline_status_20260810/marion/replay_verification.json
```

Expected outputs:

```text
full96: expected=96 checked=96 exact=96 mismatches=0
marion subset: expected=12 checked=12 exact=0 mismatches=12
```

The downloaded evidence directory contains run manifests and per-image records. The authoritative baseline metrics and images are identified by the immutable CL9 Comet key, not by display name.

---

## 6. References

1. CL9 source experiment: immutable Comet key `81bb311ed70545eda3281c64bc48be47`, step `24,000`.
2. `analysis/2026-08-10_cl9_marion_occlusion_small_faces.md` - baseline edge-case measurements and original gated experiment design.
