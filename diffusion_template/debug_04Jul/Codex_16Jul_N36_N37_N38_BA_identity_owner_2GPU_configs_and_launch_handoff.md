# N36/N37/N38: BA identity-owner 2-GPU experiment handoff

Date: 16 July 2026

## Objective

N34 and N35 trained successfully but remained visually dominated by PhotoMaker. The new
matrix changes identity ownership explicitly:

- PhotoMaker remains the global denoising, pose, composition, lighting, and rendering baseline.
- BA is allowed to replace PhotoMaker identity conditioning only at selected face-relevant
  cross-attention sites.
- The final BA correction is still applied only inside the hard target bbox and only once,
  after CFG. Outside the bbox the prediction remains exactly PhotoMaker.
- All new behavior is behind configuration toggles whose defaults preserve the previous code path.

All three runs use two processes and global effective batch 4:

`2 GPUs × local batch 1 × gradient accumulation 2 = 4`

Each epoch is 2,000 microbatches, or 1,000 optimizer updates. The scripts default to eight
epochs / 8,000 optimizer updates.

## Common N36/N37/N38 architecture

The common configuration is
`src/configs/one_id_ba_identity_owner_qformer_N36.yaml`.

### 1. Local identity ownership instead of competing residuals

At the six `up_blocks.1` cross-attention sites:

- the ordinary attention path receives text-only prompt embeddings;
- PhotoMaker's fused identity contribution is removed locally;
- BA identity memory becomes the only explicit identity source at those sites.

At the ten `up_blocks.0.attentions.2` sites:

- 50% of PhotoMaker identity context is retained;
- the BA residual gate starts at 0.5 rather than 1.0.

These lower-resolution sites give BA some facial geometry authority without asking it to
rebuild the complete pose. The higher-resolution `up_blocks.1` route has stronger identity
authority for facial appearance.

The remaining UNet attention sites are untouched PhotoMaker.

### 2. PhotoMaker preservation boundary

The unchanged PhotoMaker UNet prediction is computed as an external baseline. The BA branch
is converted into a post-CFG correction and hard-merged inside the target bbox:

`final = guided PhotoMaker + BA face correction inside bbox`

This is important: suppressing PM identity in selected internal BA sites does not remove the
global PhotoMaker prediction. It gives the BA branch enough contrast to learn a different
face while preserving the PM result outside the face region.

### 3. Increased but bounded BA capacity

- Patched CA sites: 16 instead of N34/N35's 6.
- `ba_face_gate_max`: 2.0.
- Initial gate: 1.0 at `up_blocks.1`.
- Initial gate: 0.5 at `up_blocks.0.attentions.2`.
- `ba_residual_scale`: 1.0, because identity-context removal already increases effective
  branch authority.

The output projection remains zero-initialized. The fixed PM identity-context attenuation,
however, means the new BA branch is intentionally not identical to PM before learning.

### 4. Stronger identity-causal objective

Relative to N34/N35:

- outer causal loss weight: 0.50 instead of 0.25;
- direct reference identity weight: 0.50 instead of 0.25;
- causal margin: 0.05 instead of 0.02;
- structure weight: 0.20 instead of 0.10;
- wrong-reference, cross-identity, and preservation terms remain active.

BA also begins at inference step 10 together with PhotoMaker, rather than waiting until step 15.

## Three matched variants

| Run | Identity memory | Main question | Expected behavior |
|---|---|---|---|
| N36 | Two frozen QFormer tokens from the full reference | Can explicit identity ownership make the already reliable N34 memory visibly control the face? | Fastest visible identity departure; possible clothing/background leakage from the full reference |
| N37 | Two canonical-aligned QFormer tokens plus eight trainable canonical face-part tokens | Do ordered eyes/nose/mouth/contour tokens improve identity detail while the QFormer pair stabilizes the new resampler? | Highest detail ceiling, but may learn more slowly because eight tokens are trainable |
| N38 | Two frozen QFormer tokens from a bbox-normalized face crop | Does removing reference background and clothing produce the best identity/artifact balance? | Recommended likely winner for clean identity dominance and pose preservation |

N38 uses 15% padding around a square reference-face crop. It changes only QFormer reference
preprocessing relative to N36, so N36 versus N38 is the cleanest comparison.

## New reversible code toggles

### `ba_pm_identity_context_scale`

Controls how much PhotoMaker fused identity remains in the standard attention path at patched
BA CA sites:

- `1.0`: legacy behavior; full PhotoMaker identity context.
- `0.0`: text-only standard context; BA owns explicit identity.
- values between 0 and 1: partial PM identity anchor.

Default: `1.0`, so existing N34/N35 and older configs retain old behavior.

### `ba_pm_identity_context_scale_overrides`

Prefix/wildcard mapping for per-layer context scale. N36/N37/N38 use:

```yaml
ba_pm_identity_context_scale: 0.0
ba_pm_identity_context_scale_overrides:
  up_blocks.0.attentions.2: 0.5
```

### `ba_face_gate_init_overrides`

Prefix/wildcard mapping for per-layer initial BA gates. This gives lower-resolution geometry
sites less authority than the primary identity route.

### `qformer_plus_canonical_parts`

New identity-memory mode used by N37. It returns ten tokens:

- first two: frozen, unpooled QFormer tokens from the canonical face;
- next eight: trainable ordered canonical face-part tokens.

Checkpoint architecture manifests and standalone inference restore all new switches. Strict
restore will reject an incompatible memory mode or token count instead of silently loading it.

## Configs and scripts

Configs:

- `src/configs/one_id_ba_identity_owner_qformer_N36.yaml`
- `src/configs/one_id_ba_identity_owner_hybrid_N37.yaml`
- `src/configs/one_id_ba_identity_owner_cropped_qformer_N38.yaml`

Scripts:

- `serv_new_runs/start_ba_identity_owner_qformer_2gpu_N36.sh`
- `serv_new_runs/start_ba_identity_owner_hybrid_2gpu_N37.sh`
- `serv_new_runs/start_ba_identity_owner_cropped_qformer_2gpu_N38.sh`

## Suggested parallel allocation

On the four-GPU machine, run the clean N36 versus N38 preprocessing comparison:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash serv_new_runs/start_ba_identity_owner_qformer_2gpu_N36.sh

CUDA_VISIBLE_DEVICES=2,3 \
bash serv_new_runs/start_ba_identity_owner_cropped_qformer_2gpu_N38.sh
```

The scripts use different default master ports, so both can run concurrently.

On the two-GPU machine:

```bash
bash serv_new_runs/start_ba_identity_owner_hybrid_2gpu_N37.sh
```

Set `PM_PATH` and `COMET_API_KEY` as for N34/N35. Each script starts detached by default and
prints its PID and log path.

To run the complete 96-image validation at step 0, add the script-only
`full_step0_val` argument:

```bash
bash serv_new_runs/start_ba_identity_owner_cropped_qformer_2gpu_N38.sh full_step0_val
```

The script removes this token before forwarding other arguments to Hydra. Without it, step 0
keeps the default smoke validation capped at 24 images. Later validations remain full
96-image evaluations in both cases.

## Scheduler correction

Accelerate advances this repository's scheduler once per process for each optimizer update.
The scripts therefore pass 400 scheduler ticks on two GPUs to obtain approximately 200 actual
optimizer updates of warmup:

`WARMUP_OPTIMIZER_STEPS=200 × NUM_PROCESSES=2`

This removes the machine-count-dependent warmup shortening observed in N34/N35.

## Reverting or making the route safer

The exact old N34/N35 behavior remains available by running their unchanged configs/scripts.
The new code defaults do not activate identity-context attenuation.

For an N36-family run, local PM suppression can be disabled without code changes:

```bash
bash serv_new_runs/start_ba_identity_owner_qformer_2gpu_N36.sh \
  model.ba_pm_identity_context_scale=1.0 \
  'model.ba_pm_identity_context_scale_overrides={}'
```

For a softer identity owner, retain 25% PM identity at the primary sites and 75% at the
geometry sites:

```bash
bash serv_new_runs/start_ba_identity_owner_qformer_2gpu_N36.sh \
  model.ba_pm_identity_context_scale=0.25 \
  'model.ba_pm_identity_context_scale_overrides={up_blocks.0.attentions.2:0.75}'
```

Memory modes and token counts must be selected before training. A strict N37 checkpoint cannot
be loaded into a two-token N36/N38 architecture.

## Validation and stopping criteria

At 1,000 optimizer updates, unlike N34/N35, the face should be visibly different from the
PhotoMaker baseline. Evaluate identity change and pose/artifact preservation separately:

1. Compare generated faces directly against the step-0/PhotoMaker images using identical seeds.
2. Check ID similarity to the BA reference, not only similarity to the training target.
3. Inspect eyes, mouth, face contour, hairline, and bbox boundary artifacts.
4. Confirm posture, head orientation, expression, lighting, and background remain PM-like.

Stop and investigate rather than spending the full budget if:

- images are still pixel-near-identical to PM at both 1k and 2k;
- the face changes but is independent of the selected reference;
- bbox seams or pose collapse worsen consistently from 1k to 2k.

Expected prioritization:

1. N38 is the best candidate for identity dominance with minimal reference contamination.
2. N36 is the most direct test that the new ownership route fixes PM dominance.
3. N37 has the highest architectural ceiling but should be judged at both 1k and 2k because
   its canonical resampler needs to learn.

## Verification completed

- Python syntax compilation passed for all modified modules.
- Hydra composition passed for N36, N37, and N38.
- Shell syntax passed for all three launch scripts.
- The scripts resolve to two processes, accumulation two, global effective batch four.
- A small attention smoke test confirmed:
  - scale `1.0` reproduces the legacy fused-PM standard attention output;
  - scale `0.0` uses the text-only standard attention output.
- A hybrid-memory smoke test confirmed N37 produces 8 canonical tokens and concatenates them
  with 2 QFormer tokens to obtain the configured 10-token memory.
- `git diff --check` passed.
