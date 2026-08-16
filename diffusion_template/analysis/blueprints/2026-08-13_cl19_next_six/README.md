# CL19 next-six implementation blueprints

These files are **design blueprints, not runnable Hydra or MLS files**. They
exist outside `src/configs/` and `serv_run_packages/` deliberately. An
implementing agent should convert exactly one blueprint at a time into:

1. a defaults-off code path;
2. a child Hydra config inheriting
   `CL19_cosmic_true_soft_fullquery_router_24k`;
3. a focused composition/startup validator;
4. an active launcher and immutable experiment JSON;
5. a one-A100 Serv package produced from the project's current package
   template.

Do not submit a blueprint directly. Resolve every `TO_COMPUTE` field from the
live model and pin it before launch.

## Immutable shared contract

- Control: CL19, Comet `cfeda7b55c174b3c83e8d40537ebb6dd`.
- Fresh-training arms: 24,000 optimizer steps, `epoch_len=2000`, validation and
  checkpoint every 2,000 steps, fixed `manual_val` 96.
- Reward continuation arm: exact CL19 24k weights at local step 0, reset
  optimizer, 4,000 local steps, validation at local 0/2k/4k; source checkpoint
  hash must be pinned.
- Preserve prompts, references, target bboxes, seeds, RealVisXL validation
  base, DDIM50, CFG5, PhotoMaker start step 10, metric definitions, and one
  generated image per validation item.
- Always keep `use_branched_attention=true`,
  `pipeline.pose_adapt_ratio=0.0`, and
  `pipeline.ca_mixing_for_face=false`.
- Legacy branched CA remains disabled. Native PhotoMaker/text CA remains
  intact.
- Every new branch must log both its gate and
  `RMS(branch_delta) / RMS(native)` by group. A low auxiliary loss is not
  evidence that the branch affects generation.
- Promotion requires the matched full-96 subject-v2 ID delta, paired bootstrap
  interval, seven face-quality curves, prompt/text and mask-IoU checks, and a
  blinded hard-case review that rejects object deletion as a solution.

## Recommended launch order

1. `CL21_residual_identity_ca_v3.blueprint.yaml`
2. `CL22_visibility_order_router.blueprint.yaml`
3. `CL23_temporal_frequency_router.blueprint.yaml`
4. `CL24_photomaker_boundary_distillation.blueprint.yaml`
5. `CL25_low_noise_identity_reward.blueprint.yaml`
6. `CL26_anchored_highres_roi.blueprint.yaml`

CL21 retains the name already proposed in the 12 August branched-CA report.
The earlier provisional CL22 BigCelebs-transfer name is superseded: CL20 now
shows that generic BigCelebs curriculum is not a priority on the CL19 path.

