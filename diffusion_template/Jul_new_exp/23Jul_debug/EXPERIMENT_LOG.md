# 23Jul NN3a_new1 one-ID experiment log

Selected identity: `id_00081_1017318003459`  
Training references: 8  
Held-out references: 2 (`holdout_A` recurring; `holdout_B` final audit)  
Optimizer budget: 600 steps  
Checkpoint/validation points: 0, 200, 400, 600  
Recurring prompts: 4  
Physical/effective batch: 1/1 (the training dataset contains one record)  
Training environment: `photomaker_NS`

## Shared-GPU execution decision

The production `ba_N3a_new1_1gpu` process occupies approximately 35 GB while
training and approximately 76 GB while validating. A second native trainer
cannot safely enter validation while that process exists. Therefore:

1. each local arm trains in three 200-step epochs with native validation
   disabled;
2. checkpoints are saved at steps 200, 400, and 600;
3. the experiment-local validation harness evaluates step 0/200/400/600 after
   the training process releases its model;
4. all four checkpoint results are logged with explicit step and validation
   stream names.

This preserves the requested 200-step evaluation cadence while avoiding a
predictable validation-time GPU OOM.

## Run ledger

| Run | Change | Status | Comet name | Result |
|---|---|---|---|---|
| E00_control | Exact NN3a_new1 control | complete | `23Jul_E00_control_id00081_s0_600__20260723T181448Z` | Severe core blur at step 0 and face-coordinate warping already visible at step 200 |
| E01_active_up | Train active up-block processors only | complete | `23Jul_E01_active_up_id00081_s0_600__20260723T185401Z` | Up-only pruning accelerates the same coordinate failure; rejected |
| E02_up1_detail | Train the six high-resolution up1 processors only | complete | `23Jul_E02_up1_detail_id00081_s0_600__20260723T192610Z` | Duplicated/displaced inner-face features across checkpoints; rejected |
| E03_staged_up1_up0 | Up1-only to step 99, then up0 at 0.35x | complete | `23Jul_E03_staged_up1_up0_id00081_s0_600__20260723T200451Z` | Grossly duplicated/displaced face geometry persists through step 600; rejected |
| E04_projection_split | Active-up; faster ref K/V, slower ref Q/noise/up0 | complete | `23Jul_E04_projection_split_id00081_s0_600__20260723T194413Z` | Better identity metric at 600 but persistent horizontal landmark smear; rejected visually |
| E05_blended20 | 80/20 full/face loss on every step | pending | `23Jul_E05_blended20_id00081_s0_600` | — |
| E07_schedule_matched_up | Active-up with BA-region timestep sampling | pending | `23Jul_E07_schedule_matched_up_id00081_s0_600` | Audited N3a train/inference mismatch isolated behind a registry toggle |
| E11_active_up_blended20 | E01 active-up plus 80/20 anchored loss | pending | `23Jul_E11_active_up_blended20_id00081_s0_600` | Combination arm retained for comparison |
| E12_projection_split_schedule | E04 plus BA-region timestep sampling | ready | `23Jul_E12_projection_split_schedule_id00081_s0_600` | Single-mechanism follow-up if E04 images confirm its favorable weight trajectory |
| E13_projection_split_blended20 | E04 plus 80/20 anchored loss | ready | `23Jul_E13_projection_split_blended20_id00081_s0_600` | Single-mechanism loss follow-up to E04 |
| E14_projection_split_pm_teacher20 | E04 plus matched frozen PhotoMaker epsilon anchor at 0.20 | complete; rejected | `23Jul_E14_projection_split_pm_teacher20_id00081_s0_600__20260723T212654Z` | Teacher leaves E04's severe horizontal landmark displacement intact at all trained checkpoints |
| E15_oneid8_projection_split | Exact E04 through `OneIDTrain`, same-image pairing | complete; **invalid leakage audit** | `23Jul_E15_oneid8_projection_split_nm0005092_oneid8_s0_600__20260723T204711Z` | Every target is its own pixel-identical reference; optimistic step-200 result is not promotable |
| E16_oneid8_pm_teacher20 | Exact E14 through `OneIDTrain`, same-image pairing | complete; **invalid leakage audit** | `23Jul_E16_oneid8_pm_teacher20_nm0005092_oneid8_s0_600__20260723T210320Z` | Same target leakage as E15; teacher conclusion is valid only within the invalid control |
| E17_oneid8_projection_split_blended20 | Same-image OneIDTrain follow-up | canceled before launch | — | Stopped immediately after the pairing audit; no GPU training or Comet run occurred |
| E18_oneid8_distinct_projection_split | Corrected E15 with guaranteed different same-ID reference | complete; rejected | `23Jul_E18_oneid8_distinct_projection_split_nm0005092_oneid8_distinct_s0_600__20260723T220805Z` | Step 200 duplicates landmarks; step 600 geometry recovers by drifting toward PhotoMaker identity |
| E19_oneid8_distinct_projection_split_blended20 | Corrected E18 plus always-anchored 80/20 loss | training complete; short validation stopped | `23Jul_E19_oneid8_distinct_projection_split_blended20_nm0005092_oneid8_distinct_s0_600__20260723T222432Z` | Step-200 images uploaded; stopped when user superseded 600-step protocol with 4k runs |
| E20_oneid8_distinct_ref_value_only | Corrected distinct pairs; train reference V only, freeze Q/K and noise route | stopped by protocol change | `23Jul_E20_oneid8_distinct_ref_value_only_nm0005092_oneid8_distinct_s0_600__20260723T224441Z` | Reached first checkpoint; superseded by paired 4k reference-V-only arms |

## E00 live observations

- First successful run folder:
  `experiments/20260723T181448Z__23Jul_E00_control_id00081_s0_600__20260723T181448Z`
- Concurrent memory after warm-up: approximately 22.8 GB for E00 and 34.7 GB
  for the production run.
- Step 200 checkpoint saved successfully.
- Step 200 LoRA-B drift: reference L2 `4.8122`, noise L2 `1.0354`.
- Step 200 block L2 (reference and noise combined): up0 `3.8069`, up1
  `1.1724`, down `2.3109`, mid `1.7385`.
- Step 400 LoRA-B drift: reference L2 `6.3748`, noise L2 `1.6120`.
- From step 200 to 400 the noise norm grew about 56%, faster than the
  reference norm’s 32%. Up0 grew to `5.2570`, while up1 grew only to `1.3981`.
  This is consistent with testing stronger noise damping and a more targeted
  high-resolution learning allocation.
- Interpretation pending images: substantial optimizer movement is already
  spent in down/mid processors even though reference ownership is restricted
  to up blocks. This makes E01 active-up pruning a well-motivated isolated
  follow-up.
- The canonical step-zero Reading and Rushing cases preserve the PhotoMaker
  scene and outer head but replace the inner face with a visibly blurred,
  low-detail BA core. This selected held-out reference is therefore a harder
  initialization case than the broad 22Jul grid.
- At canonical step 200, the first Reading case becomes sharper but develops
  severe non-rigid eye/nose/mouth displacement inside the core. Training is
  amplifying a coordinate/alignment error rather than simply converging too
  slowly. This visual failure agrees with the early up0/down/mid norm growth
  and strengthens the case for active-up and high-resolution/reduced-up0
  variants.

## Protocol revision

The approved protocol supersedes the original plan’s 500-step/100-step
cadence: all arms now use 600 optimizer steps and are evaluated every 200
steps.

## Comet validation logging policy

- Generated-face masks are now pinned to
  `data/id_00081_1017318003459/pm_generated_bboxes_holdout_A_seed0.json`.
  The trainer produced these boxes in its explicit PhotoMaker-only precursor
  pass using the new held-out reference and seed 0. E00/E01/E02 independently
  produced semantically and byte-identical masks, and E04's independent
  Reading recomputation matched exactly. Therefore their completed results do
  not require rerendering; all future jobs consume the shared file directly.
- All validation GPU jobs now use the console writer. After render completion,
  the launcher resolves and verifies the real training experiment key and
  uploads local files directly to that experiment. No future checkpoint-level
  Comet experiments are created by the training framework.
- Image names include validation stream and optimizer step, for example
  `canonical50__step0200__...`, so canonical/early-BA images at the same
  checkpoint cannot collide.
- E00 jobs launched before this correction created separate Comet experiments.
  Their already-rendered images are re-logged into the E00 training experiment
  with `migrate_validation_to_comet.py`; the local originals remain intact.
- The first E01 unified-validation preflight failed before checkpoint loading
  or image generation because the local logging subclass inherited
  `SDXLTrainer` rather than the production `PhotomakerLoraTrainer` that owns
  `masked_loss_step`. The inheritance was corrected and compiled; this did not
  consume a validation result, although the framework writer had already
  initialized one of the continuation artifacts listed below.
- A second framework-writer audit showed that Comet created continuation
  experiments even when Hydra held the correct training key. Two E01
  preflight/validation continuation keys (`dad3351...` and `2b02ced...`) are
  retained as failed logging artifacts, renamed
  `FAILED_23Jul_E01_validation_writer_preflight`, tagged accordingly, and are
  not used for results. The
  console-render/direct-upload path was then verified against the original E01
  training key `97459743...`; its four PM-control images are present there.

## E01 live observations

- Run folder:
  `experiments/20260723T185401Z__23Jul_E01_active_up_id00081_s0_600__20260723T185401Z`
- Forward graph remains all 70 installed processors, but the optimizer retains
  only 432 tensors across 36 up-block processors: 30 in up0 and six in up1.
- Step 200 reference/noise LoRA-B L2 norms are `4.2599` / `0.8927`.
  Active up0 reference drift is `4.0847`, versus `3.7232` for E00 at the same
  step; concentrating the optimizer does accelerate the directly active path.
  The checkpoint images will determine whether this is useful learning or
  faster coordinate distortion.
- Step 400 reference/noise norms are `5.7057` / `1.4375`; step 600 norms are
  `6.8537` / `1.9644`.
- E01's sampled scalar losses are almost identical to E00 at every logged
  25-step point despite the optimizer scope changing by nearly half. Denoising
  loss alone therefore cannot distinguish useful branch routing from the
  visually severe alignment failure; checkpoint images and geometry metrics
  remain the promotion gate.
- E01 reached step 600 and released its GPU immediately before the production
  run entered validation, so checkpoint interruption was not required.
- Canonical E01 images reject the isolated active-up hypothesis: Reading at
  step 200 has a sharper but severely displaced/doubled mouth and compressed
  right eye; at step 400 the core is smoother but one eye is effectively
  erased. Freezing down/mid does not solve the coordinate failure and, by
  concentrating updates into up0, can accelerate it.
- The local launcher now supports checkpoint-boundary interruption and
  in-place resume with the same run directory and Comet run ID. This is used
  to protect the production run's high-memory validation window.

## E02 training observations

- Run folder:
  `experiments/20260723T192610Z__23Jul_E02_up1_detail_id00081_s0_600__20260723T192610Z`
- The forward retains all 70 NN3a processors, but only the six up1 processors
  are trainable (72 tensors).
- Reference/noise LoRA-B norms progress from `1.7095/0.3368` at step 200 to
  `2.1723/0.4698` at 400 and `2.5387/0.6051` at 600.
- At step 200, up1 reference movement is about 41% larger than E01's up1
  movement (`1.7095` versus `1.2093`), as expected when the same denoising
  objective is concentrated into the high-resolution route. Validation will
  determine whether this produces useful detail without up0 coordinate warp.
- The first canonical step-200 Reading image rejects the strongest version of
  that hypothesis: compared with the already-blurred step-zero core, training
  adds eye/nose/mouth texture but in duplicated and displaced coordinates.
  Up1-only training reduces overall parameter drift but does not by itself
  correct the branch/reference alignment. The remaining three prompts and
  later checkpoints confirm the same failure; horizontal feature smearing is
  stronger in the Reading case at step 400.
- Median reference similarity is `0.2528`, `0.2211`, and `0.2688` at steps
  200/400/600, versus `0.2549` at step zero. Thus the visually malformed
  result does not even deliver a reliable identity improvement. The full
  metrics, four-prompt checkpoint PDF, and images are uploaded to the original
  E02 Comet experiment (`a3202002...`).

## E04 live observations

- Run folder:
  `experiments/20260723T194413Z__23Jul_E04_projection_split_id00081_s0_600__20260723T194413Z`
- Six optimizer groups are active: ref K/V, ref Q, and noise Q/K/V, each split
  between up0 and up1 (`other`).
- At step 200, reference/noise LoRA-B norms are `2.3954/0.1662`. Noise movement
  is 81% lower than E01's `0.8927`.
- The step-200 up0 combined norm is `1.7910`, 57% below E01's `4.1724`;
  up1 is `1.5994`, retaining most of E02's focused high-resolution movement.
  Reference K/V norms (`1.4511/1.8623`) dominate reference Q (`0.4052`), which
  is the intended content-over-coordinate signature. This is the strongest
  weight-space trajectory so far, pending images.
- E04 completed all 600 steps. Reference/noise norms progress
  `2.3954/0.1662` → `3.6779/0.2741` → `4.6233/0.3582`. At step 600,
  reference K/V remain dominant (`2.8987/3.5302`) while reference Q is only
  `0.7145`; up1 reaches `2.3809` and up0 `3.9793`. The intended optimizer
  separation therefore remains stable rather than collapsing late in the
  short run.
- Canonical validation does not promote E04 despite its more disciplined
  weights. Median reference similarity improves from `0.2549` at step 0 to
  `0.3533` at step 600, but steps 200 and 400 still contain the characteristic
  horizontally duplicated/displaced inner-face landmarks. Geometry is the
  promotion gate, so the result is a useful optimizer direction rather than a
  successful architecture.

## E03 staged run

- Run folder:
  `experiments/20260723T200451Z__23Jul_E03_staged_up1_up0_id00081_s0_600__20260723T200451Z`
- Started automatically at `20:04:51Z` immediately after E04 released its
  training allocation. The first 100 global optimizer steps update only up1;
  the named up0 optimizer groups are then enabled at their configured `0.35x`
  scale without changing the forward graph.
- Training completed through step 600. Reference/noise LoRA-B norms are
  `1.8750/0.2309`, `2.9985/0.3970`, and `3.9966/0.5376` at steps
  200/400/600. Canonical checkpoints were rendered in the first safe window
  after the production run's 96-image validation.
- Canonical validation rejects staging decisively. The Reading step-600 image
  has horizontally duplicated eyes/nose/mouth compressed into the upper face;
  Rushing has the same displaced mouth/teeth failure. Median reference
  similarity is only `0.2023`, `0.2709`, and `0.2780` at steps 200/400/600,
  while landmark displacement remains `0.0269`, `0.0248`, and `0.0223`.
  The slight late metric recovery does not correspond to acceptable geometry.
  Results and the PDF are uploaded to the original Comet key `5f57bf4c...`.

## Matched PhotoMaker teacher and dataset-loader ablation

### Target/reference leakage audit (critical correction)

- A post-run provenance audit found that E15/E16 were launched with
  `train_on_separate_image=false`. In `OneIDTrain.__getitem__`, that branch
  sets `ref_images = [deepcopy(img)]` after applying the same horizontal flip
  and copies the same bbox. This is target leakage, not a distinct same-ID
  reference.
- An actual Hydra-instantiated sample audit covered all eight subset records.
  In every case the target and reference filename, pixels, and bbox were
  identical: `83→83`, `109→109`, `38→38`, `57→57`, `104→104`, `36→36`,
  `1→1`, and `116→116`; each pixel comparison had `max_abs=0`.
- Consequently, the strong-looking E15 step-200 result and the E15/E16
  comparison cannot establish identity generalization or a dataset-loader
  benefit. Both runs and PDFs are retained as clearly labeled leakage audits.
  E17 was canceled before launch.
- The earlier E00–E04 CosmicLarge runs are not affected. Their saved config
  instantiates the newer `CosmicLargeTrain`, whose constructor discards the
  legacy `train_on_separate_image` flag and whose `__getitem__` always loads
  the target from `image_path` and the reference from a separately sampled
  `face_paths` file. Seeded live samples confirmed different target/reference
  paths, e.g. target `.../1017318003459.jpg` versus refs `.../6.jpg`,
  `.../5.jpg`, and `.../2.jpg`. The clean E14 retry uses the same valid loader.
- E18/E19 use a new
  `one_id_nm0005092_subset8_distinct` profile with
  `train_on_separate_image=true`. `OneIDTrain` then samples the reference from
  all subset indices except the target index, guaranteeing different files
  from the same identity.

- E14 keeps E04's NN3a core, scope, and projection-split optimizer unchanged,
  but adds `0.20 * MSE(epsilon_BA, epsilon_PhotoMaker)` using the same noisy
  target latent, timestep, prompt, reference augmentation, and restored RNG
  state. This directly penalizes destructive branch corrections while still
  allowing identity-distinct residual learning.
- E14's first preflight launch failed before optimizer step 1 because a
  one-row PhotoMaker teacher was initially sent through NN3a processors that
  require a doubled `[target, reference]` batch. The local implementation now
  replaces only those processors with vanilla `AttnProcessor2_0` during the
  no-grad teacher pass and restores the exact original objects before the BA
  pass. The failed run is retained as an auditable artifact; E16 and the clean
  E14 retry use the corrected code.
- A console-only one-step smoke run of the corrected teacher completed a full
  forward, auxiliary loss, backward, and optimizer update. The preservation
  loss was finite (`0.0017985`, weighted `0.0003597`), total loss was
  `0.044944`, total gradient norm was `0.004286`, and all six projection-split
  optimizer groups had finite gradients. This is the implementation used by
  the queued E16 and clean E14 runs.
- The user's dataset-type hypothesis is isolated with two paired arms:
  E15 is exact E04 and E16 is exact E14, both loaded by
  `src.datasets.cosmic.OneIDTrain`.
- Both one-id arms use the same fixed eight records:
  `83.jpg`, `109.jpg`, `38.jpg`, `57.jpg`, `104.jpg`, `36.jpg`, `1.jpg`,
  and `116.jpg`. The native validation reference `51.jpg` is excluded.
  `train_on_separate_image=false`, so each target is also its own reference,
  matching the native one-id behavior.
- Validation uses reference `one_id/ref/51.jpg`, seed 0, and the first four
  native one-id prompts. Generated-face masks are the corresponding records
  from the repository's PhotoMaker-only automatic bbox pass and are snapshotted
  locally with source/reference checksums and prompt/ID/seed metadata. No BA
  image is used to determine a validation mask.
- Hydra composition plus an actual CPU dataset sample verified eight train
  items, four validation items, 1024×1024 train tensors, the native stochastic
  horizontal flip/bbox transform, and all four resolved PhotoMaker-generated
  face boxes before scheduling GPU work.
- E15 completed 600 steps. Its reference/noise LoRA-B norms progress
  `1.6458/0.0967` → `2.7415/0.1641` → `3.5069/0.2205`, substantially below
  E04's `2.3954/0.1662` → `3.6779/0.2741` → `4.6233/0.3582` under the
  same architecture and optimizer recipe.
- In retrospect, E15's clean-looking early validation demonstrates how
  strongly same-image leakage can conceal the coordinate-collapse failure.
  Step-zero Reading is already broadly aligned; at step 200 it becomes sharper
  without duplicated eyes/mouth, and Rushing is likewise coherent. These are
  invalid memorization artifacts, not evidence for identity generalization.
- The complete four-prompt grid makes step 200 the cleanest E15 leakage
  checkpoint, but it is not eligible for promotion.
  Reading and Rushing are sharp, anatomically coherent, and subject-specific;
  Skiing remains aligned despite the difficult double-goggle composition; and
  Drumming is coherent with only mild eye asymmetry. Steps 400 and 600 begin
  to drift (eye/glasses stretching and softness) but remain far less damaged
  than the Cosmic-loader runs.
- E15 canonical median metrics at steps 0/200/400/600 are:
  reference similarity `0.3434/0.3249/0.3307/0.3607`,
  BA-versus-PhotoMaker face difference `0.4963/0.6137/0.6353/0.6299`,
  face-box IoU `0.9271/0.9111/0.9168/0.9175`, and landmark displacement
  `0.0323/0.0377/0.0340/0.0367`. The scalar composite incorrectly favors
  step 600; visual anatomy is the required primary gate and favors step 200.
- The PhotoMaker-derived mask overlay was audited after rendering. All four
  masks cover the correct generated face without a coordinate offset; the
  ski mask intentionally includes the face and goggles. The checkpoint PDF,
  metric JSON/CSV, images, and bbox debug report were uploaded to the original
  E15 Comet experiment key `1bb75f6fd1aa4ebabf9dee6aeca98a6c`.
- E16 completed the paired OneIDTrain ablation and was validated against
  exact materialized copies of E15's step-zero and PhotoMaker controls. Its
  reference/noise LoRA-B norms are `1.6477/0.0970` → `2.7236/0.1638` →
  `3.4865/0.2204`, only marginally different from E15. Canonical step-200
  medians are reference similarity `0.3201`, face difference from PhotoMaker
  `0.6037`, IoU `0.9091`, landmark displacement `0.0376`, and outside MAE
  `0.0161`. Step-600 reference similarity rises to `0.3678`, but visible
  eye/glasses warping makes that metric improvement non-promotable.
- Visual review finds E16 effectively tied with E15 at step 200 and still
  degraded at steps 400/600. The `0.20` matched PhotoMaker epsilon teacher
  does not solve late coordinate drift and adds a second denoiser pass during
  training. Neither E15 nor E16 is promotable because both leak the target
  image into the reference branch. E16's report and metrics were uploaded to
  its original Comet key
  `6898cc23883445a28dd2701374e5f35f`.
- CPU-only comparison PDFs:
  `visual_reports/20260723T2139_oneid_dataset_ablation_comparison.pdf` is the
  six-page E15/E16 leakage report; and
  `visual_reports/20260723T2203_all_23Jul_runs_cpu_visual_summary_LEAKAGE_CORRECTED.pdf`
  is the corrected 13-page all-run report. The latter explicitly labels
  E15/E16 invalid and includes the target/reference audit.

### E18 corrected distinct-reference result

- E18 passed the automatic 64-pair launch audit and the extended 128-pair
  audit with zero cases in which a target was observed as its own reference.
- Step 200 recreates the valid-run failure: Reading has a displaced/duplicated
  eye, Skiing has duplicated glasses and mouth landmarks, and Drumming has
  severe inner-face coordinate distortion. The clean leaked E15 step-200 grid
  was therefore an optimistic artifact.
- Geometry progressively recovers by step 600, but the identity trajectory
  moves in the wrong direction. Median reference similarity at steps
  0/200/400/600 is `0.3434/0.2827/0.2765/0.2931`; reference gain over
  PhotoMaker is `+0.0689/-0.0035/-0.0164/-0.0175`.
- Step-600 median landmark displacement improves to `0.0143` and bbox IoU to
  `0.9514`, while face similarity to PhotoMaker rises to `0.5489`. This is
  late PhotoMaker-identity drift, not a successful identity-preserving
  recovery. Step zero remains the best checkpoint.
- E18's report and metrics are uploaded to Comet key
  `325c5d8edc844bfe8f454e8a3532814b`.
