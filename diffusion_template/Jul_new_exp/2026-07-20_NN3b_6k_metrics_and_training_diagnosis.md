# NN3b at 6k: metrics and training diagnosis

Date: 20 July 2026

## Verdict

**NN3b ran as configured, but it has not improved identity by 6k. Pause the
run after preserving the epoch-3 checkpoint rather than continuing toward
20k.**

The learned-null and target-PhotoMaker attenuation paths are installed,
trainable, and receiving gradients. There is no evidence of an OOM, NaN,
detached optimizer, failed processor restore, or validation-loading bug.
Nevertheless, validation identity similarity worsens monotonically, and the
BA residual becomes increasingly cap-limited.

This export contains metrics and logs but no validation images. Therefore this
report cannot judge face artifacts, alignment, or reference-directional visual
changes. If the 6k images show an unambiguous improvement that the aggregate
metric misses, that would be the only reason to reconsider the stop decision.

## Data checked

- Comet run: `4208c99f95fd49a2adffae0c1172c448`
- Run/config: `ba_NN3b_learned_null_pm_attenuation_realvis_1gpu` /
  `one_id_ba_NN3b_learned_null_pm_attenuation`
- Complete validations: steps 0, 2k, 4k, and 6k, 96 images each
- Saved checkpoint: `checkpoint-epoch3.pth` at 6k
- Training metrics continue to approximately step 6.4k

The identical NN3a/NN3b step-zero metrics confirm that the validation panel,
seed, base model, and zero-initialized BA baseline are comparable.

## Validation trajectory

| Step | ID similarity | Change from step 0 | Text similarity |
|---:|---:|---:|---:|
| 0 | 0.523129 | — | 26.3659 |
| 2k | 0.514184 | -0.008946 | 26.3343 |
| 4k | 0.509779 | -0.013351 | 26.4676 |
| 6k | 0.507933 | **-0.015196** | 26.4372 |

Identity similarity falls by 2.9% relative to step zero. Text similarity is
effectively stable, ending only 0.27% above step zero. The result is not a
general training collapse; it is specifically a lack of useful identity gain.

At the common checkpoints:

| Step | NN3a ID | NN3b ID | NN3b minus NN3a |
|---:|---:|---:|---:|
| 2k | 0.514099 | 0.514184 | +0.000084 |
| 4k | 0.512338 | 0.509779 | **-0.002559** |

NN3b is indistinguishable from NN3a at 2k and slightly worse at 4k. The two
new mechanisms therefore have not fixed NN3a's weak identity trajectory.

## Training and architecture checks

The intended NN3b architecture is genuinely active:

- `connector_input=reference_minus_learned_null`;
- 36 selected self-attention sites and 70 frozen split cross-attention sites;
- 36 learned-null tensors, 337,920 parameters;
- 288 trainable BA tensors and 7,096,356 trainable parameters total;
- all optimizer groups, including null memory, receive nonzero gradients;
- exactly one of each physical batch of two has target PhotoMaker identity
  removed, giving the expected attenuation fraction of `0.5`;
- reference cross-attention retains full PhotoMaker conditioning;
- no invalid target/reference samples were reported.

The one transient Comet `RemoteDisconnected` during 6k validation retried
successfully. Metrics and the epoch-3 checkpoint were saved afterward.

Approximate training-window means show no useful late improvement:

| Steps | Total loss | ID loss | Weighted ID term | Total gradient norm |
|---:|---:|---:|---:|---:|
| 0–2k | 0.1854 | 0.1023 | 0.00511 | 0.00104 |
| 2–4k | 0.1797 | 0.0958 | 0.00479 | 0.00228 |
| 4–6k | 0.1830 | 0.0975 | 0.00487 | 0.00240 |

The decoded ID loss improves early and then plateaus. Its weighted
contribution remains only about 2–3% of total loss.

The reference K/V norm canary grows strongly:

```text
step 0: 0.000
step 2k: 5.353
step 4k: 8.617
step 6k: 9.573
```

Thus training is live; the problem is the learned direction, not frozen
weights.

## Residual saturation

The gate remains approximately `0.252`, while raw residual magnitude keeps
growing into the fixed `0.25` RMS cap:

| Training diagnostic | Fully capped sites | Mean cap fraction |
|---:|---:|---:|
| 2k | 7/36 | 0.194 |
| 4k | 28/36 | 0.819 |
| 6k | **32/36** | **0.889** |

At 6k, all 30 `up_blocks.0` sites and 2/6 `up_blocks.1` sites are fully
capped in the sampled diagnostic. A fully capped site's immediate residual is
at most approximately `0.25 × 0.252 = 0.063` of local base RMS before later
layers.

Continuing training is increasingly likely to grow raw values that the cap
discards. Raising the cap is not justified because the validation identity
direction is currently wrong.

## Why NN3b did not solve NN3a

There is no concise implementation bug evident in the log. The more likely
problem is architectural:

1. The learned null is optimized only through matched-reference diffusion
   examples. There is no explicit null-reference example or objective forcing
   it to represent “no person,” and no loss requiring the branch residual to
   vanish for a null reference. It can co-adapt with reference K/V and the
   connector as extra shared capacity.
2. Null-memory gradients are active but weak. Over 4–6k their mean norm is
   about `6.6e-6`, versus `3.1e-5` for reference K, `1.7e-4` for reference V,
   and `2.4e-3` for connector-up. The export does not log the null-memory
   parameter norm, so a checkpoint is needed to measure how far it moved.
3. PhotoMaker attenuation forces BA to help on half the training samples, but
   validation restores full target PhotoMaker identity. The branch receives
   no explicit arbitration signal for whether it should complement or avoid
   conflicting with the strong full-PhotoMaker identity at inference.
4. The small same-identity ID loss does not prove reference dependence:
   target PhotoMaker and the supplied reference encode the same person. A
   generic face correction can lower training loss without learning which
   identity dimensions came from the spatial reference.

## Recommended next action

1. Preserve `checkpoint-epoch3.pth` and stop/pause NN3b.
2. Run a fixed checkpoint diagnostic with branch off versus scale 1/2,
   matched versus cyclically swapped references, and identical target
   seed/prompt. Measure directional identity gain toward the swapped reference,
   face-core LPIPS/MAE, and outside-face change.
3. Download the 6k validation images if a visual decision is required; this
   metrics-only export cannot assess artifacts.
4. For the next training architecture, keep the core packed branched
   attention but add a **paired matched/null-reference objective** using the
   same target latent, noise, timestep, and prompt:
   - matched reference: require directional identity gain toward that reference;
   - null reference: penalize branch residual and identity movement;
   - optionally compare matched and null outputs/features directly.
5. Do not raise the residual cap or gate until the swap test demonstrates
   correct reference-directional behavior.

Downloading the 6k checkpoint is useful for inspecting per-site learned-null
norms and matched-versus-null candidate separation, but the existing log is
already sufficient to conclude that a longer unchanged run is unlikely to
become successful.
