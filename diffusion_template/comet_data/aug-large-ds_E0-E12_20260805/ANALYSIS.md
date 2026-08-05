# E0-E12 identity-similarity and visual analysis

**Date:** 5 August 2026  
**Scope:** the two E0 controls and E1-E12 in the immutable
[`aug-large-ds` export](README.md)  
**Primary decision signal:** `manual_val/id_sim` on the fixed 96-image panel,
interpreted together with full-image and face-crop inspection  
**Status:** analysis and experiment design only; no job was launched

## Executive conclusion

`E0_large_ds_base_historical_r4_20k_full96_r1` remains the clear identity
winner. It reaches **0.37083 at 14k** and ends at **0.36889 at 20k**. The best
clean contender, E11 rank-128 spatial BA, peaks at **0.32704 at 8k** and ends
at **0.32167**. At 20k, historical E0 beats E11 on 72/96 matched images by a
mean `+0.04723` (paired bootstrap 95% interval `+0.03246` to `+0.06219`). It
beats fixed BA-only E0 on 78/96 by `+0.07161`.

The headline average is materially incomplete:

- Historical E0's largest identity gains over fixed E0 are Keanu `+0.134`,
  Jennie `+0.113`, Elon `+0.109`, and Marion `+0.082`. Its gains for Eddie
  (`+0.013`) and Lex (`+0.020`) are small.
- The largest prompt gains are laughing `+0.172`, skiing `+0.144`, night ride
  `+0.109`, and kickboxing `+0.092`. Dancing and jumping improve only `+0.025`.
- Eddie remains uniformly weak (`0.141` even in historical E0), while skiing
  remains the hardest prompt (`0.277` in historical E0 and `0.141` averaged
  over the clean completed suite).
- E11 is already equal to historical E0 for Jisoo and rushing/chef, and it is
  slightly better on dancing. Historical E0's lead is therefore a set of
  large identity-by-prompt interactions, not a uniform improvement.

The visual conclusion is similarly qualified. Historical E0 usually keeps the
face attached to the correct body and changes facial morphology without moving
the scene, but it is not artifact-free. Skiing repeatedly produces duplicated
goggles/glasses; crying produces hand/eye/face fusion; small jumping faces stay
generic; and some laughing/kickboxing mouths are exaggerated. A high
`ID_sim` can coexist with an anatomically bad face.

The most likely explanation for historical E0 is **joint training-time
co-adaptation**, not BA rank alone. It trained hard spatial BA together with a
generic rank-32 adapter and the pretrained PhotoMaker rank-64 adapter. The
trained PhotoMaker-default update influenced the optimization path but was
reset in historical alternate-base validation, while the BA and generic
updates were expressed. E7-E10 show that the effective generic path, generic
cross-attention, shared self-attention output, or persisted PhotoMaker-default
path alone does not reproduce the gain. E11 shows that four-times wider BA
alone also does not reproduce it. The interaction is the important clue.

The early plateau is more consistent with an **objective and optimization
ceiling** than a dataset-size ceiling. The current loss directly optimizes
face-region diffusion error, not identity. BigCelebs was 7.35 times larger by
accepted images and still peaked early. The next suite should therefore
combine the historical interaction explicitly, protect layout with a
full/face/boundary objective and differential learning rates, and dedicate
arms to a direct identity objective, safe residual identity-token CA, and
better reference evidence.

## Evidence and comparability

The export contains 14 immutable runs, 150 per-image CSVs with 14,400 rows,
and the requested 4,032 validation PNGs. Every per-image table has exactly 96
rows: eight identities crossed with 12 prompt families.

The analysis uses:

- every available per-image `ID_sim` table at step 0 and every 2k;
- latest images at 20k for E0-E11 and 12k for incomplete E12;
- full-panel and enlarged fixed-bbox face contact sheets for every latest
  run, plus focused 8k comparisons for E10/E12;
- matched per-image differences, identity means, prompt means, and
  identity-by-prompt cells;
- a 20,000-resample paired bootstrap over the 96 fixed-panel differences.

The bootstrap interval describes uncertainty across this fixed panel. It is
not a population-generalization interval: the 96 observations have the known
8-by-12 identity/prompt structure.

Three interpretation boundaries are important:

1. E1, E3, E4, and E12 alter the step-zero route, so their step-zero values are
   not interchangeable initialization controls. E4's `0.36458` at step zero,
   for example, is not a learned result.
2. E10's trained PhotoMaker-default adapter moved the subject away from the
   cached step-zero generated-face boxes. Its fixed-mask BA metrics are
   confounded and must not be ranked. The full images still establish a real
   layout/duplication failure. Later dynamic-box revalidation corrects where
   BA is applied and measured; it does not make the underlying composition
   drift acceptable.
3. E12 ends at 12k, not 20k. Its conclusion is nevertheless clear because it
   stays below its own initialization and displays the same severe face-patch
   failure across many prompts by 8k-12k.

The primary metric also has an important semantic limitation. The configured
`IDSimBest` implementation selects the best-matching detected face anywhere
in the generated image; it does not require that face to intersect the
intended body/face box. Duplicate people or a detached identity fragment can
therefore score better than the intended subject. This report does not change
the metric definition; visual body association remains a separate required
gate.

All derived files are reproducible with
[`analysis_assets/build_report_assets.py`](analysis_assets/build_report_assets.py).
The normalized per-image endpoint data are in
[`derived_latest_per_image.csv`](analysis_assets/derived_latest_per_image.csv)
and the numerical summary is in
[`derived_summary.json`](analysis_assets/derived_summary.json).

## Aggregate `ID_sim` result

![ID similarity trajectories](analysis_assets/id_trajectories.png)

![Initialization, latest, and best post-training checkpoints](analysis_assets/id_endpoint_and_peak.png)

| Arm | Scientific change | Step 0 | Best after training | Latest | Result |
|---|---|---:|---:|---:|---|
| **E0 historical** | Fail-open BA + broad generic + PM-default co-adaptation | .30187 | **.37083 @14k** | **.36889 @20k** | Clear winner; historical control, not clean ownership |
| E0 fixed | Exact hard BA r32 only | .30187 | .29965 @14k | .29728 @20k | Training does not beat initialization |
| E1 | True reference-key mask | .10310 | .29664 @16k | .27739 @20k | Different initial route; no recovery to E0 |
| E2 | Branch-local output r32 | .29937 | .31625 @16k | .30250 @20k | Small, visually stable positive element |
| E3 | Reference ROI warp | .32788 | .30082 @12k | .28126 @20k | Does not solve misregistration/plateau |
| E4 | Mid/up BA sites only | .36458 | .31186 @16k | .29623 @20k | High route-dependent step zero; learning regresses |
| E5 | Inference-active timesteps | .30187 | .30892 @20k | .30892 @20k | Modest late gain only |
| E6 | FP32 BA trainables | .30187 | .32313 @4k | .30226 @20k | Useful transient peak, no sustained gain |
| E7 | Effective generic adapter | .30187 | .26720 @14k | .25275 @20k | Generic path alone hurts identity |
| E8 | Generic ordinary CA | .30187 | .25068 @14k | .24581 @20k | Strong negative result |
| E9 | Shared SA output | .30187 | .31174 @10k | .30385 @20k | Small positive element, far below E0 |
| E10* | Persisted PM-default effective path | .30187 | .21557 @4k | .16912 @20k | Metrics confounded; visual layout failure |
| **E11** | Hard spatial BA r128 | .30187 | **.32704 @8k** | **.32167 @20k** | Best clean arm; capacity helps but is insufficient |
| E12** | Hard ID-token CA r256 in up0/up1 | .26209 | .19800 @10k | .18305 @12k | Incomplete but categorical architectural failure |

`*` E10 fixed-mask values are shown for provenance, not ranking.  
`**` E12 latest is 12k.

Historical E0's advantage remains broad in paired comparisons:

| Comparator | Comparison step | Historical minus comparator | 95% paired interval | Historical wins |
|---|---:|---:|---:|---:|
| E11 SA-r128 | 20k | **+.04723** | +.03246 to +.06219 | 72/96 |
| E5 inference timesteps | 20k | +.05998 | +.04491 to +.07530 | 73/96 |
| E2 branch output | 20k | +.06640 | +.05012 to +.08335 | 73/96 |
| E6 FP32 | 20k | +.06663 | +.05161 to +.08197 | 75/96 |
| E0 fixed | 20k | **+.07161** | +.05471 to +.08877 | 78/96 |
| E12 ID-CA | matched at 12k | +.16770 | +.14764 to +.18766 | 93/96 |

The matched suite target is E0 historical's `0.37083` at 14k. The older
Large Dataset historical r4 reached `0.39039` at 24k; that is the stronger
aspirational target. More steps alone are not the answer: both the old Large
Dataset and much larger BigCelebs histories oscillated or regressed after
their peaks.

## The average hides identity structure

![Endpoint ID similarity by identity](analysis_assets/id_by_identity_latest.png)

The following table isolates historical E0 against the two most informative
controls at 20k:

| Identity | E0 historical | E0 fixed | E11 | Hist - fixed | Hist - E11 |
|---|---:|---:|---:|---:|---:|
| Eddie | .141 | .128 | .137 | +.013 | +.004 |
| Elon | **.464** | .356 | .392 | **+.109** | **+.072** |
| Jennie | .382 | .269 | .310 | **+.113** | **+.072** |
| Jensen | .452 | .391 | .425 | +.061 | +.027 |
| Jisoo | .423 | .382 | .421 | +.041 | +.002 |
| Keanu | .420 | .286 | .329 | **+.134** | **+.091** |
| Lex | .372 | .352 | .334 | +.020 | +.038 |
| Marion | .296 | .214 | .225 | **+.082** | **+.071** |

Key implications:

- Eddie is an evaluation/training blind spot rather than an E0 success. Across
  the completed clean runs, his mean endpoint is only `0.121`; historical E0
  barely changes that operating point. All ten of historical E0's weakest
  individual cells are Eddie prompts.
- Marion remains the second weakest identity, but historical co-adaptation
  lifts her broadly. This is qualitatively different from Eddie's flat curve.
- Jennie is the most run-sensitive identity. Expression/occlusion cells swing
  by more than 0.3 across clean arms.
- E2 is competitive for Jennie (`0.367`) and slightly exceeds historical E0
  for Jisoo (`0.426` versus `0.423`), despite its much lower overall mean.
  Branch-local output capacity is therefore worth retaining as an optional
  component, but it is not sufficient by itself.

![Historical E0 identity-by-prompt matrix](analysis_assets/e0_historical_id_prompt_matrix_20k.png)

Historical E0's weakest individual cells include Eddie jumping `.079`, Eddie
angry `.087`, Eddie night ride `.106`, Eddie drumming `.109`, and Marion
skiing `.194`. Its strongest include Elon laughing `.586`, Elon rushing
`.566`, Jensen rushing `.564`, Jisoo night ride `.563`, and Jensen crying
`.562`. A single mean combines qualitatively different regimes.

## The average hides prompt structure

![Endpoint ID similarity by prompt](analysis_assets/id_by_prompt_latest.png)

| Prompt family | E0 historical | E0 fixed | E11 | Hist - fixed | Hist - E11 |
|---|---:|---:|---:|---:|---:|
| Reading | .435 | .391 | .365 | +.044 | +.070 |
| Rushing | .422 | .384 | .416 | +.038 | +.005 |
| Skiing | .277 | .133 | .159 | **+.144** | **+.118** |
| Drumming | .387 | .355 | .370 | +.032 | +.017 |
| Kickboxing | .368 | .275 | .308 | **+.092** | +.060 |
| Dancing | .297 | .273 | **.304** | +.025 | **-.007** |
| Angry | .374 | .315 | .330 | +.058 | +.043 |
| Laughing | .398 | .226 | .318 | **+.172** | **+.080** |
| Crying | .381 | .307 | .324 | +.074 | +.057 |
| Chef | .383 | .336 | .374 | +.046 | +.009 |
| Night ride | .416 | .308 | .332 | **+.109** | **+.084** |
| Jumping | .289 | .265 | .261 | +.025 | +.029 |

Skiing is the clearest systematic failure. It averages only `0.141` over the
completed clean suite, versus rushing `0.380` and reading `0.364`. Historical
E0 recovers a large fraction of the identity score, but the images show that
the recovery often coexists with doubled eyewear. Jumping and dancing have
small faces and remain weak even when attachment is coherent. Laughing has the
largest historical gain, but it also has the strongest open-mouth/teeth
exaggeration risk.

![Historical E0 advantage by identity and prompt](analysis_assets/e0_advantage_by_identity_and_prompt_20k.png)

The selected trajectories make the different regimes visible:

![Key identity and prompt trajectories](analysis_assets/id_key_group_trajectories.png)

- Eddie stays near `0.10-0.16` in every arm.
- Marion receives a broad historical E0 lift.
- Jennie and Keanu receive large, sustained historical-specific gains.
- Skiing collapses at 2k in every clean route and never fully recovers.
- Dancing is weak but E11 is competitive, so historical co-adaptation is not
  universally better.

## Highest-discrepancy image groups

Among the scientifically usable completed clean arms at 20k (E0 fixed,
E1-E9, and E11; excluding historical E0, E10, and E12), the largest per-image
ranges are concentrated in crying, angry, kickboxing, and other expressive or
occluded faces:

| Fixed-panel item | Lowest clean arm | Highest clean arm | Range |
|---|---:|---:|---:|
| Jennie crying | E3 .012 | E6 .387 | **.375** |
| Jensen crying | E8 .168 | E6 .521 | **.352** |
| Jennie kickboxing | E7 .153 | E2 .480 | **.327** |
| Jennie drumming | E8 .093 | E2 .417 | **.324** |
| Jennie angry | E8 .085 | E4 .403 | **.318** |
| Jisoo crying | E1 .121 | E11 .401 | .280 |
| Jennie skiing | E1 .047 | E4 .282 | .235 |
| Jennie jumping | E8 .122 | E11 .345 | .223 |

Historical E0's largest 20k leads over E11 are also interaction-specific:

| Item | Historical E0 | E11 | Delta |
|---|---:|---:|---:|
| Jennie night ride | .428 | .199 | **+.229** |
| Keanu skiing | .417 | .191 | **+.226** |
| Elon reading | .494 | .304 | **+.190** |
| Jisoo skiing | .203 | .033 | **+.170** |
| Jennie angry | .420 | .252 | **+.168** |

E11 nevertheless wins important cells:

| Item | Historical E0 | E11 | Historical delta |
|---|---:|---:|---:|
| Jisoo jumping | .302 | .415 | **-.114** |
| Jennie jumping | .255 | .345 | -.090 |
| Jisoo chef | .476 | .560 | -.084 |
| Marion chef | .265 | .349 | -.084 |
| Eddie drumming | .109 | .191 | -.082 |

This is why a future result should be evaluated with paired cells and fixed
hard subsets, not only a mean.

## Visual comparison: identity, body attachment, and artifacts

The full-image grid below shows high historical-E0 identity deltas. Scene,
pose, and body placement are nearly fixed across E0 fixed, E2, E11, and
historical E0; the main change is facial morphology. This is evidence that
historical E0's advantage is not generally coming from a different scene.

![High historical-E0 deltas, full images](analysis_assets/visual_high_e0_deltas_latest_full.png)

[Open enlarged face crops for the same items](analysis_assets/visual_high_e0_deltas_latest_faces.png).

Representative gains include Keanu laughing (`.410` versus E0 fixed `.080`
and E11 `.279`), Elon skiing (`.311` versus `.047` and `.215`), Elon chef
(`.555` versus `.320` and `.420`), and Keanu skiing (`.417` versus `.163` and
`.191`). These are genuine identity changes, but the skiing rows also show the
central visual warning: the face can score better while retaining two layers
of orange goggles/glasses.

The next grid deliberately shows difficult or contrary cases:

![Hard and contrary cases, full images](analysis_assets/visual_hard_cases_latest_full.png)

[Open enlarged face crops for the hard cases](analysis_assets/visual_hard_cases_latest_faces.png).

Observed recurring failures:

- **Skiing/accessories:** repeated goggles, glasses below goggles, lens/eye
  overlap, and face content copied into the eyewear region. Historical E0 is
  often more recognizable, but it does not solve the geometry.
- **Crying/hand occlusion:** fingers merge into the eye/cheek; eyelids and
  mouths shift; Jisoo and Jennie are especially sensitive. Jisoo crying in
  historical E0 scores `.426` despite a visibly malformed hand/eye boundary.
- **Small faces:** jumping and some dancing images keep the face attached but
  remain generic. ID differences can be driven by a few pixels.
- **Extreme expressions:** laughing and kickboxing can acquire oversized open
  mouths, teeth, or highly saturated skin. E7/E8 show this especially often.
- **Identity-specific non-gains:** historical E0 is worse than several clean
  arms for Eddie angry, Lex dancing, and Jisoo jumping. The average hides these
  regressions.

E10 and E12 are qualitatively different failure classes:

![E10 and E12 failures at 8k](analysis_assets/visual_e10_e12_failures_8k_full.png)

[Open enlarged E10/E12 face crops](analysis_assets/visual_e10_e12_failures_8k_faces.png).

- **E10:** the persisted PhotoMaker-default update changes pose, subject
  location, expression, and sometimes the number of people. The old fixed box
  then applies BA at the old location, producing ghost or displaced faces.
  This is both a model-layout failure and a fixed-mask measurement confound.
- **E12:** hard replacement of native face cross-attention by an ID-token-only
  message creates colored rectangular/plate-like faces, mask seams, missing
  facial regions, and implausible high-contrast features. The native prompt CA
  contained essential face structure. Rank 256 and more training do not fix
  the wrong merge equation.

For complete review, see the generated
[`full-image contact sheets`](analysis_assets/review_contacts_latest/) and
[`face contact sheets`](analysis_assets/review_contacts_faces_latest/).

## What the experiments establish

### Elements worth carrying forward

1. **The historical joint interaction is the strongest lead.** Future work
   should explicitly train the effective generic and PhotoMaker-default paths
   together with BA, rather than repeat isolated E7-E10 components.
2. **More core BA capacity helps clean identity.** E11 is the only clean arm
   with a sustained, meaningful lift, and its visuals remain broadly coherent.
   Rank 128 is a sensible substrate, but not a complete solution.
3. **Branch output/shared output capacity is low-risk but small.** E2 and E9
   provide modest gains and generally stable layouts. They can be considered
   after the joint mechanism is reproduced, not used as the main hypothesis.
4. **Optimization details can move the early peak.** E6 peaks at `.32313` at
   4k and E5 ends at `.30892`; neither solves the plateau, but FP32/different
   timestep support are not categorical failures.
5. **Fixed per-cell analysis is essential.** Keanu/Jennie and
   skiing/laughing/night ride are where the historical mechanism helps;
   Eddie, small jumping faces, and hand/accessory occlusion are the required
   stress tests.

### Elements not supported

1. True-key masking, ROI warping, or mid/up site restriction alone does not
   raise the identity ceiling.
2. Generic ordinary CA or the effective generic adapter alone is harmful at
   the current LR/objective.
3. Persisting a fully trained PhotoMaker-default adapter at the current LR and
   face-only objective is unsafe; it moves global composition.
4. Wider spatial BA alone is not enough: E11 remains `0.04379` below the
   historical best even at its own peak.
5. Hard ID-only CA replacement is rejected. Any future identity-token CA must
   retain native CA and add a bounded residual message.
6. More data or unchecked longer training is unlikely to break the ceiling.
   The larger BigCelebs base plateaued similarly, and most current arms peak
   well before their endpoint.

## Why historical E0 is still superior

Fixed E0 trains exactly 840 BA tensors / 31,948,800 parameters. Historical E0
accidentally left three broad groups trainable:

| Historical group | Tensors | Parameters | Effective role |
|---|---:|---:|---|
| Hard spatial BA processors | 840 | 31,948,800 | Target Q uses same-ID reference-face K/V |
| Generic rank-32 adapter | 1,120 | 46,448,640 | Effective at shared SA output and ordinary CA |
| PhotoMaker-default rank-64 adapter | 1,120 | 92,897,280 | Changes the training forward/gradient field |
| **Total** | **3,080** | **171,294,720** | **5.36x fixed-E0 capacity** |

Not all nominal generic tensors were active. Historical checkpoint inspection
showed that generic SA Q/K/V LoRA-B tensors stayed exactly zero because hard BA
bypasses those outer projections. The effective generic path was:

- shared self-attention `to_out` after the face/background merge: 5.3248M;
- ordinary cross-attention Q/K/V/output: 25.14944M.

The PhotoMaker-default update was trained but omitted/reset in historical
alternate-base validation. Consequently, historical validation directly used
the trained BA and generic states with the original pretrained default
adapter. The default update could still be important indirectly because it
changed the gradients that produced those saved BA/generic weights.

The completed decomposition sharpens the conclusion:

- E7/E8 show that saved generic adaptation without default co-training is not
  sufficient.
- E9 shows that the shared output basis alone is not sufficient.
- E10 shows that expressing the trained default update alone is destructive
  under the current objective/LR.
- E11 shows that raw BA capacity alone is not sufficient.

The evidence therefore favors a **co-adapted optimization trajectory**: the
default adapter, generic ordinary CA/shared output, and hard reference BA shape
one another during training; resetting the default adapter at validation then
removes its large global-layout drift while retaining the identity-beneficial
BA/generic solution. This is scientifically informative but not yet a clean
deployable recipe. The next work needs both an explicit reproduction of that
"shadow co-adapter" behavior and a protected, fully persisted alternative.

## Common training and evaluation issues

| Issue | Observed/code evidence | Consequence | Recommended treatment |
|---|---|---|---|
| **No direct identity objective** | Training minimizes diffusion epsilon MSE; `ID_sim` exists only in validation | A low denoising loss can plateau while recognition does not improve | Add one bounded predicted-x0 identity-loss arm; keep BA the principal identity route |
| **Face-only loss on every batch** | The inherited config sets `trainer.masked_loss_step=1`; `MaskedDiffusionLoss` then uses only the bbox crop | Generic/default adapters affect the whole U-Net while pixels outside the face receive no loss, enabling E10 drift | Use `face + 0.1*full + 0.05*boundary` for protected arms |
| **One LR for very different parameter groups** | BA, zero-init generic LoRA, and nonzero pretrained default updates all receive `1e-4` | The pretrained default path moves composition before BA/generic settle | Add exact optimizer roles; use lower generic/default LRs and a late decay |
| **Uniform random single reference** | `LargeDatasetTrain` uses `random.choice` over distinct same-ID images; no pose, blur, occlusion, resolution, or embedding-quality score | Noisy/misaligned spatial K/V and duplicated accessories; resume streams are not exact | Pin a deterministic schedule and separate pose-compatible spatial references from canonical identity references |
| **Image-frequency rather than identity-balanced sampling** | The dataset index contains images, so identities with more images appear more often | Uneven identity learning can contribute to weak tails | Make the data arm identity-balanced while keeping a fixed row count |
| **Hard rectangular routing masks** | Masks are bbox rectangles resized with nearest-neighbor at attention resolutions | Hands, hair, goggles, and background enter/exit abruptly; seams and duplicated structure persist | Preserve the primary fixed boxes, but test soft boundary weighting in the loss and a safe residual CA branch; do not silently change validation masks |
| **Best-face metric is not body-associated** | `IDSimBest` maximizes over every detected generated face | Duplicates/detached faces can inflate identity | Keep primary `ID_sim` unchanged; add a secondary intended-box/largest-subject ID and face-count diagnostic |
| **Resume RNG is not exact** | Python reference selection/flip state is not part of the training checkpoint | A resumed run does not reproduce the same target/reference/augmentation stream | Use a hashed deterministic schedule with explicit optimizer-step offset |
| **Batch size is small, but not the leading cause** | Current one-GPU batch is 2; a historical global-batch-4 Serv run was worse at the same LR, though DDP/resume differences confound it | Blindly increasing batch can change optimization without fixing identity | Do not spend one of the next six GPUs on a batch-size-only arm; retune LR if accumulation is tested later |
| **Dataset quantity is not the main bottleneck** | Large Dataset has 47,500 images/2,561 IDs; BigCelebs had 349,348/68,648 and still plateaued early | More uncurated images alone are unlikely to help | Improve objective/reference quality and identity balance before seeking another large dataset |

The face-only-loss issue is especially important. `_masked_face_mse` is a
reasonable BA-local objective when only a local processor changes, but it is
not sufficient supervision when generic or PhotoMaker-default adapters can
move the entire image. Historical E0 partly hid this because its trained
default state was reset for validation; E10 exposed it when the state was
correctly persisted.

## Recommended next six one-GPU experiments

These runs are designed as a parallel performance suite, not six isolated
single-variable architecture ablations. E13/E14 explicitly test the historical
shadow-co-adapter mechanism; E15-E18 are fully persisted promotion candidates.
Each arm must remain independently launchable.

### Shared contract

- One A100 per run; six simultaneous one-GPU requests use the normal six-GPU
  project ceiling. Inspect current Running/Pending MLS jobs before submission.
- Hard cap 24k optimizer steps: `trainer.epoch_len=2000`, `n_epochs=12`,
  validation/checkpoint at step 0 and every 2k. The exact 20k point remains
  directly comparable; 22k/24k test the older r4 late peak without an
  unchecked 40k continuation.
- Fixed 96-image `manual_val`, seeds, prompts, references, generated/reference
  boxes, RealVis validation base, scheduler, 50 inference steps, CFG, and
  primary metric definition.
- `pipeline.pose_adapt_ratio=0` and `pipeline.ca_mixing_for_face=false` in
  training and validation. Target queries must retain explicit same-ID
  reference K/V through hard spatial BA.
- Rank-128 hard spatial BA at all 70 SA sites, effective generic rank-32
  adapter, and effective PhotoMaker-default rank-64 adapter, all explicitly
  allowlisted and fully saved in schema v2.
- Warm up for 20 steps, keep each group's LR constant through 14k, then cosine
  decay to 10% of its base LR by 24k. This preserves the historical learning
  window and reduces late oscillation.
- Keep the primary fixed-mask validation canonical. A dynamic-box pass may be
  logged only as a clearly labeled secondary diagnostic; a candidate that
  requires it because layout moved has failed the primary composition gate.
- Log unchanged `IDSimBest` plus secondary intended-subject ID, detected face
  count, and per-image CSVs. Promotion still depends on visual full-image and
  face-crop review.

| Priority | Proposed run | Core question | Expected role |
|---:|---|---|---|
| 1 | `E13_large_ds_joint_shadow_sa128_24k_full96_r1` | Does E0 joint co-adaptation plus E11 capacity exceed E0 when the default update is intentionally shadow-only? | Highest probability of a numerical ID gain; mechanism run, not deployable |
| 2 | `E14_large_ds_joint_shadow_sa128_protected_24k_full96_r1` | Can full/boundary supervision retain the shadow gain while reducing artifacts? | Safer shadow mechanism run |
| 3 | `E15_large_ds_joint_persist_sa128_protected_24k_full96_r1` | Can differential LR and full-state validation make the joint mechanism deployable? | Main clean promotion candidate |
| 4 | `E16_large_ds_joint_persist_sa128_idloss_24k_full96_r1` | Does direct recognition supervision break the denoising-loss plateau? | Highest-upside objective arm |
| 5 | `E17_large_ds_joint_persist_sa128_resididca_24k_full96_r1` | Can a bounded residual ID-token CA add identity without E12's face plates? | Corrected cross-attention arm |
| 6 | `E18_large_ds_joint_persist_sa128_multiref_24k_full96_r1` | Does identity-balanced, pose/quality-aware reference evidence improve weak cells? | Data/reference arm |

### E13 - explicit shadow co-adaptation with wider BA

Configuration:

```yaml
model:
  ba_hard_v1_lora_rank: 128
  generic_adapter_train_scope: effective_all
  photomaker_default_train_scope: effective_all
  branched_state_dict_mode: trainable_v2
validation_adapter_state_policy: pretrained_default_with_trained_ba_generic
optimizer_roles:
  ba_lr: 1.0e-4
  generic_effective_lr: 1.0e-4
  photomaker_default_effective_lr: 1.0e-4
loss_function:
  _target_: src.loss.diffusion_loss.MaskedDiffusionLoss
trainer:
  masked_loss_step: 1
```

Expected effective ownership is 2,240 tensors / 219,217,920 parameters:
127.7952M BA + 30.47424M effective generic + 60.94848M effective default.
Derive and assert the count during implementation rather than trusting this
document if processor selection changes.

The checkpoint must save all three groups for exact resume. The validation
policy deliberately applies trained BA/generic state while restoring the
pretrained PhotoMaker-default state. This converts the historical omission
into explicit, auditable behavior. It is a mechanism/performance control, not
a deployable promotion candidate. Its upside is the most direct combination
of the two positive signals: historical interaction and E11 capacity.

### E14 - shadow co-adaptation with protected diffusion loss

E14 is identical to E13 except for the existing composite objective:

```yaml
loss_function:
  _target_: src.loss.branched_reference_loss.BranchedReferenceLoss
  face_weight: 1.0
  full_weight: 0.1
  boundary_weight: 0.05
  boundary_ring_width: 2
  reference_weight: 0.0
```

This is the cleanest test of whether the every-batch face-only loss enables
the historical/default drift. It preserves full-strength face supervision,
adds a small global constraint, and directly penalizes the seam ring. Do not
enable the shuffled-reference ranking loss; prior rank-loss experiments did
not improve the plateau.

### E15 - fully persisted protected joint model

E15 uses the E14 architecture/objective, but validation loads every trained
group and the optimizer uses differential rates:

```yaml
validation_adapter_state_policy: trained_full_state
optimizer_roles:
  ba_lr: 1.0e-4
  generic_effective_lr: 5.0e-5
  photomaker_default_effective_lr: 1.0e-5
```

This is the main clean candidate. The lower rates address E7/E8 degradation
and E10 layout drift while preserving joint co-adaptation. If its primary
fixed boxes become invalid, treat the arm as failed even if a secondary
dynamic-box score is high. A successful checkpoint must reproduce identical
pixels after schema-v2 save/load with all 2,240 trainable tensors present.

### E16 - E15 plus predicted-x0 identity supervision

Add a low-weight differentiable PhotoMaker-CLIP cosine proxy on the generated
predicted-x0 face. The repository and Serv environment have the ONNX
InsightFace validation model but no runnable differentiable PyTorch ArcFace
checkpoint, so this arm is deliberately named and logged as a proxy rather
than claiming metric-backend parity:

- use the already-loaded frozen PhotoMaker CLIP vision tower and compare the
  predicted face to the real training target face; no validation image or
  validation-identity centroid enters training;
- reconstruct predicted x0 from the current noisy latent, timestep, and
  epsilon prediction; apply the fixed target-face alignment transform and
  differentiate through the frozen VAE decoder/recognizer to BA/adapters;
- evaluate one sample every four optimizer steps and only for `t <= 400` to
  bound memory and avoid meaningless high-noise identity gradients;
- use cosine loss weight 0 through 2k, linearly ramp to `0.05` at 6k, then
  keep it fixed; log raw cosine loss, eligible-batch fraction, and gradient
  norms separately;
- keep the E15 face/full/boundary diffusion objective as the main loss.

This supervision is not the primary InsightFace `IDSimBest` metric. Its value
must therefore be decided by the unchanged full-96 metric and visual review,
not by the proxy loss curve alone.

### E17 - E15 plus safe residual identity-token cross-attention

Implement `ResidualIdentityCrossAttnProcessorV3` instead of modifying E12 in
place. At selected `up_blocks.0` and `up_blocks.1` CA sites:

```text
native = CA(target_Q, full PhotoMaker/text prompt_KV)
id_msg = delta_out(CA(branch_target_Q, active_ID_token_KV))
output = native + face_mask * bounded_gate(layer) * rms_norm(id_msg)
```

Required invariants:

- native CA remains intact inside and outside the face;
- the ID branch uses target queries and only active PhotoMaker ID tokens;
- rank 64, delta-only zero-initialized output, gate initialized to `0.02` and
  bounded to `[0, 0.20]`;
- no hard replacement, no legacy branched CA, and no
  `ca_mixing_for_face`; the new manifest must say
  `native_plus_bounded_identity_residual`;
- log gate, native-face RMS, ID-message RMS, and residual/native ratio per
  block; fail if the step-zero output is not numerically near E15.

This directly addresses E12's error: identity tokens add information, but do
not own the entire face representation.

### E18 - E15 plus deterministic, decoupled reference evidence

Build a deterministic-by-algorithm 48,000-row schedule (24k steps x batch 2)
over the existing Large Dataset:

- balance target rows by identity rather than raw image count;
- require target/reference inequality and exact same identity;
- select one **spatial BA reference** by available quality metadata (falling
  back to face resolution) plus bbox geometry diversity;
- supply up to three additional **PhotoMaker identity references** selected by
  deterministic quality ranking and a seeded row offset; these condition the
  ID tokens but do not replace the explicit single spatial reference K/V path;
- derive identity, target, references, and flip from the fixed seed and row;
  the dataset is sequential and has exactly 48,000 rows;
- preserve the standard fixed validation references unchanged.

`LargeDatasetBalancedMultiRefTrain` returns the spatial reference first and
the extra PhotoMaker references after it. Existing conditioning already uses
all references for PhotoMaker tokens but only `ref_images[0]` for the spatial
latent and bbox, so the two evidence paths remain explicitly decoupled. Do not
turn this into pure token conditioning or self-reference.

## Implementation map for another agent

1. **Create an implementation branch/worktree only after checking dirty
   state.** Preserve existing changes. Read the handoff and search the touched
   subsystem for `AICODE-*` anchors first.
2. **Add optimizer roles** in
   `src/model/photomaker_branched/lora2.py` and the exact allowlist helpers in
   `lora2_helpers.py`. Simultaneous effective generic/default scopes already
   exist; split them into named groups without changing default behavior.
3. **Add explicit validation adapter policy** in the alternate-base validation
   load path. Both policies must load a complete schema-v2 training
   checkpoint; the shadow policy selectively restores only the pretrained
   default state after load and records that choice in the manifest/log.
4. **Reuse `BranchedReferenceLoss`** for E14-E18. Add only the Hydra configs and
   writer loss names needed for its existing full/face/boundary outputs.
5. **Add the identity auxiliary** as a defaults-off trainer/model hook plus a
   small composite loss. Return the noisy latent and timesteps only when the
   hook is enabled. Verify frozen VAE/recognizer parameters never enter the
   optimizer while gradients reach the intended adapter groups.
6. **Create `residual_identity_ca_processor_v3.py`** for E17. Do not alter E12's v2 class
   or checkpoint semantics. Register v3 behind new defaults-off model flags,
   ownership roles, schema manifest, training install, validation install, and
   telemetry.
7. **Create a deterministic dataset mode** for E18. Reuse existing sequential
   schedule handling and audit identity balance, path inequality, bbox bounds,
   reference counts, and repeatability across all 48,000 algorithmic rows.
8. **Add six Hydra configs** inheriting the audited fixed-E0 parent, six
   experiment JSON records, and six one-GPU launch packages/scripts. Pin every
   invariant explicitly, including epoch length, total steps, validation
   interval, batch size 2, LR groups, masks, `pose_adapt_ratio=0`, and
   `ca_mixing_for_face=false`.
9. **Run focused preflight only:** Hydra composition, shell syntax, import
   compile, exact trainable/optimizer membership, one-batch forward/backward,
   nonzero/forbidden gradient audit, checkpoint round trip, processor install
   parity, and deterministic step-zero validation. E16 additionally needs
   identity-proxy and memory smoke checks; E17 needs zero-init parity;
   E18 needs schedule audit.
10. **Before submitting**, inspect Serv Running/Pending jobs and keep the
    project at or below six requested A100s. During every startup, verify
    `saved/<run_name>/comet_experiment.json` exists and contains the immutable
    key. No run may be identified later only by display name.

## Decision gates

A future arm is not promoted merely because one average crosses E0.

1. **Numerical target:** exceed matched-suite E0's `0.37083` at two consecutive
   full-96 gates. Treat the older r4 `0.39039` as the stronger target.
2. **Paired breadth:** compare the candidate checkpoint to historical E0's
   14k per-image table; require positive median delta, more than 48/96 wins,
   and a paired-bootstrap interval whose lower bound is above zero.
3. **Weak-group guardrail:** do not regress historical E0 by more than `.02`
   for Eddie (`.141`), Marion (`.296`), skiing (`.277`), or jumping (`.289`).
   Report Jennie and Keanu separately because they drive much of the mean.
4. **Visual guardrail:** review all 96 full images and face crops, with an
   explicit hard subset covering all skiing images, Jennie/Jisoo crying,
   small jumping/dancing faces, and laughing/kickboxing mouths. Count body-face
   dislocations, duplicated faces/people, eyewear duplication, mask seams,
   hand-face fusion, and severe mouth/eye artifacts.
5. **Causal/architecture gate:** verify target-Q/reference-KV spatial BA is
   active and checkpoint-identical in training and validation. For E17, also
   verify the residual ID-token branch is nonzero but bounded.
6. **Promotion boundary:** E13/E14 can validate the historical mechanism but
   cannot be promoted as deployable because their default adapter is
   intentionally different between training and inference. Promotion must
   come from E15-E18 or a subsequent fully persisted confirmation.

This design gives the parallel batch two hedges: E13/E14 maximize the chance of
recovering and extending the known high-ID basin, while E15-E18 try to make that
gain complete, stable, and attributable without abandoning explicit
reference-conditioned branched attention.
