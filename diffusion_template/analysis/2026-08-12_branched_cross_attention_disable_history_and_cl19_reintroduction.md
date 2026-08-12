# Branched cross-attention should return as a new CL19-based residual arm, not by re-enabling the legacy processor

**Date:** 12 August 2026

**Scope:** Historical Markdown and current-source audit of why branched U-Net
cross-attention (`attn2`) was disabled, whether the reasons still apply, and
whether it should be added to CL19 or CL20.

**Evidence cutoff:** Immutable Comet refresh at `2026-08-12T15:46:37Z`; source
and local experiment records present in the `test` worktree. CL19 and CL20 had
complete, table-sealed `20k` panels but no complete `22k`/`24k` panels at this
cutoff.

**Changes made:** Read-only analysis, a local immutable-key Comet refresh, this
Markdown report, and the required handoff update. No config, checkpoint,
running job, Comet experiment, validation input, or remote machine was changed.
No PDF or Dropbox upload was requested.

| Arm | Immutable Comet key | Comparable step | Subject-v2 ID | Text | Mask IoU |
|---|---|---:|---:|---:|---:|
| Plain PhotoMaker V2 | `74efd227d3f8488a98e83d815c77c07c` | `0` | `0.556580` | `26.0015` | `0.86515` |
| CL14 SA-only BA | `6fe0028be92242c38056b3d36665fdd6` | `20k` | `0.455268` | `26.2381` | `0.89596` |
| CL19 soft-router SA-only BA | `cfeda7b55c174b3c83e8d40537ebb6dd` | `20k` | **`0.503941`** | `26.3499` | **`0.91473`** |
| CL20 curriculum, CL14 architecture | `b05488e2cce94476acc92bcaa21d7362` | `20k` | `0.452543` | `26.5938` | `0.89226` |

The table uses the unchanged `manual_val/id_sim` curve, which is the current
mask-owned subject-v2 identity metric. `[measured]` CL19 peaks at `0.506175`
at `18k`; CL20's current peak is `0.452543` at `20k`; CL14 peaks at `0.457096`
at `22k`. `[measured]` Plain PhotoMaker is shown as an external ceiling/context
on the same validation contract, not as a matched training-step control.

## Executive conclusion

**It is worth bringing explicit branched cross-attention back, but only in a
fresh, single-delta CL19 successor. Do not modify CL19 or CL20 in flight, do
not enable CA for their existing checkpoints, and do not reactivate the legacy
`BranchedCrossAttnProcessor`.**

The historical disablement had two valid causes. First, a controlled July
intervention found that CA amplified global scene/body corruption on the
Cosmic one-ID setup: matched CA-on identity was `0.0351`, while genuine CA-off
was `0.1418`, with visibly cleaner bodies and scenes. `[measured]` Second, the
August code audit found that the legacy implementation did not implement the
intended target-query, face-local CA route: it formed the face query from the
reference hidden state, concatenated the generation result into the target
half and the identity result into the reference half, retained zero prompt
tokens in the softmax, ignored the spatial masks in its forward merge, and
used whole-lane replacement. `[code]`

Those findings reject the old processor, not the architectural principle.
The same July fixed-checkpoint matrix showed that the older leak-free one-ID
model remained coherent and scored `0.3956-0.4068` with CA on versus `0.2637`
with CA off. `[measured]` Thus CA was not universally harmful. Moreover, CL19
now provides a much stronger and mechanically audited SA substrate:
`2,240` trainable tensors / `219,217,920` parameters, correct subject
selection in validation, target-Q/reference-KV SA, exact checkpoint ownership, and a
single post-attention soft router. `[code] [measured]` At matched `20k`, CL19
is `+0.048673` ID above CL14 and `+0.051398` above CL20. `[measured]`

The recommended experiment is therefore a new
`CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r1`: inherit CL19
exactly and enable the existing corrected, bounded residual identity-token CA
only in `up_blocks.0/1`. Native PhotoMaker/text CA remains intact; target
queries consume only active PhotoMaker identity tokens through a face-local,
zero-initialized, bounded residual. `[code]` This restores an explicit CA path
without repeating the legacy routing bug or E12's destructive hard
replacement.

## 1. What was disabled, and what was not

The phrase “cross-attention is disabled” is too broad for the current code.

| Mechanism | CL19/CL20 state | Meaning |
|---|---|---|
| Native SDXL/PhotoMaker `attn2` | Active | Target latents still consume text and PhotoMaker identity conditioning. `[code]` |
| Spatial branched self-attention | Active | Target queries consume spatial-reference face K/V. `[code]` |
| Legacy `BranchedCrossAttnProcessor` | Disabled | The historical doubled-batch `attn2` replacement is not installed. `[code]` |
| `ca_mixing_for_face` | `false` | An SA K/V memory-concatenation shortcut is off; this is not U-Net `attn2`. `[code]` |
| Corrected hard identity CA v2 | Off | E12's target-Q/active-ID-token hard face replacement is not installed. `[code]` |
| Corrected residual identity CA v3 | Off | E17's native-plus-bounded-ID-residual path is available but not installed. `[code]` |

CL19 and CL20 are consequently valid **SA-only BA** experiments, not combined
SA+CA BA experiments. `[code]` Their strong results do not establish that the
original full architecture is unnecessary; they establish that a strong
spatial-reference SA route can work while native PhotoMaker CA remains active.

## 2. Historical chronology: why branched CA was turned off

### 2.1 The original architecture explicitly included branched CA

The 24 July one-ID handoff records branched SA at all selected U-Net sites,
target-face queries attending to reference-face K/V, and “branched
cross-attention remains enabled.” `[report]` The original one-ID configs used:

```yaml
disable_branched_ca: false
train_branched_ca_lora: true
```

The same handoff noticed unexpectedly broad background change and identified
global `noise_and_ref` adapters plus branched CA as paths able to influence
outside-face content. It proposed disabling CA as the first isolated
architectural intervention if face-only loss did not remove exterior drift.
`[report]`

### 2.2 Task A made the intended single-delta CA-off intervention

Task A changed only the branched-CA installation/training flags relative to
the face-only control: `disable_branched_ca=true` and all CA-trainable flags
false. Startup found `840/840` self-attention processor tensors and no
branched `attn2` trainables. `[report]`

The in-training validation was initially invalid because the temporary RealVis
validation model installed randomly initialized branched CA before receiving
the disable flag. The report correctly withdrew those trainer images and used
the standalone evaluator's audited result: `70` branched SA processors, zero
branched CA processors, and `12/12` outputs. `[report]` This matters because
the CA-off conclusion is based on the corrected replay, not on the buggy
trainer validation.

### 2.3 CA-off improved Cosmic globally, but did not solve face geometry

On the matched Cosmic one-ID endpoint:

| Route | Text | ID | Visual finding |
|---|---:|---:|---|
| CA on, legacy reproduced | `23.7565` | `0.0351` | Broad smoky/melted/displaced scene and body corruption. `[measured]` |
| CA off, validation-native | **`24.7982`** | **`0.1418`** | Cleaner scenes/bodies, but about `9/12` faces still malformed. `[measured]` |

Task B also disabled CA at inference on the same Cosmic CA-on checkpoint and
raised ID from `0.0517` to `0.1066` under the validation-native base, while
cleaning the exterior. `[measured]` This supported “CA amplifies global
corruption,” but rejected “CA is the sole face-local cause.” The dominant
malformed-face mechanism persisted in the spatial-reference SA path and was
strongly sensitive to tight versus full-scene reference formatting.
`[measured] [report]`

### 2.4 The contrary one-ID result shows CA was not intrinsically broken in every regime

For the leak-free older one-ID checkpoint, Task B found:

| Route | Text | ID | Visual finding |
|---|---:|---:|---|
| CA on, legacy | `23.2760` | **`0.4068`** | Coherent anatomy. `[measured]` |
| CA on, validation-native | `23.2930` | `0.3956` | Coherent anatomy. `[measured]` |
| CA off, validation-native | **`25.3581`** | `0.2637` | Mostly coherent but materially lower identity. `[measured]` |

This is important evidence against a blanket conclusion. CA could increase
identity on a healthy full-scene-reference regime, albeit with a text tradeoff.
It also means the July Cosmic result cannot by itself tell us whether the harm
came from CA as a concept, the tight-reference/data regime, the legacy
processor equation, or their interaction.

### 2.5 The August source audit found a decisive implementation defect

The E11/E12 design audit inspected the legacy `BranchedCrossAttnProcessor` and
found that it does not implement the advertised face-local target-query CA:

```text
query_bg  = Q_noise(target_hidden)
query_ref = Q_ref(reference_hidden)
hidden_bg = CA(query_bg, generation_prompt)
hidden_ref = CA(query_ref, identity_prompt)
return concat([hidden_bg, hidden_ref])
```

The target half therefore receives only the generation-prompt result; the
identity-prompt result is returned to the reference half. `[code]` Although
the processor stores target/reference masks, its forward path does not use
them to merge CA messages. `[code]` It also attends the full 77-token
identity-prompt tensor rather than gathering active PhotoMaker identity
tokens, so zeroed token positions remain in the softmax denominator.
`[code]` These are sufficient reasons not to reactivate it, independent of
the July metrics.

## 3. Later corrected-CA experiments: what they add to the decision

### 3.1 E12 proves hard identity-only face replacement is the wrong merge

E12 replaced native target-face CA with a corrected target-query,
active-ID-token message in `up_blocks.0/1`, at rank `256`. It used
`1,128` tensors / `134,578,176` parameters and kept the legacy processor off.
`[code] [measured]`

E12 fell from `0.26209` at step zero to `0.18305` at `12k`; at the matched
`12k` gate historical E0 beat it by `0.16770` and won `93/96` rows.
`[measured]` Visual review found colored rectangular face plates, mask seams,
missing regions, and implausible high-contrast features. `[report]` The native
text/PhotoMaker CA carried necessary structure; making an ID-only message own
the entire face was the wrong equation. More rank and training did not repair
that structural error.

### 3.2 E17 made CA safe, but did not establish a gain

E17 retained native CA everywhere and added only a zero-initialized,
face-local, bounded identity-token residual:

```text
native = CA(target_Q, full PhotoMaker/text prompt_KV)
id_msg = delta_out(CA(target_Q, active PhotoMaker ID token_KV))
output = native + face_mask * bounded_gate * rms_norm(id_msg)
```

This was visually safe, but E17 was `-0.00599` versus its E15 base, with an
interval crossing zero. `[measured]` Its gate/residual telemetry existed in
the processor but was omitted from the configured Comet writer, so the run
could not establish whether the residual was materially used. `[code]
[report]` E17 also sat on E15's persist-trained PhotoMaker-default substrate,
which was itself the main failure of that suite (`-0.07398` versus E14).
`[measured]` E17 is therefore a negative result for that exact base, not a
clean test on CL19's substantially stronger soft-router substrate.

### 3.3 What is not the cause

- Native PhotoMaker/text cross-attention was never removed from CL19/CL20;
  their success is not evidence that the U-Net works without CA. `[code]`
- The CA disablement was not caused by the later trainable-ownership,
  checkpoint, subject-selector, or CL14 mask-feather bugs; it predates those
  fixes. `[report]`
- The July evidence does not show that CA universally reduces identity; the
  leak-free one-ID checkpoint showed the opposite. `[measured]`
- E12 does not show that all corrected CA is harmful; it rejects hard
  replacement of the complete native face CA with ID tokens. `[measured]
  [code]`
- E17 does not show that CL19 plus residual CA cannot work; E17 used a weaker,
  materially different substrate and lacked branch-use telemetry. `[report]`

## 4. Why CL19, not CL20, is the right substrate

CL19 and CL20 are not two comparable strong architecture bases. CL19 changes
CL14's self-attention face-boundary routing; CL20 keeps the CL14 model exact
and changes only the training schedule. `[code]`

At the current matched `20k` gate:

| Comparison | Subject-v2 ID delta | Text delta | Mask-IoU delta |
|---|---:|---:|---:|
| CL19 minus CL14 | **`+0.048673`** | `+0.111816` | **`+0.018774`** |
| CL20 minus CL14 | `-0.002725` | `+0.355632` | `-0.003693` |
| CL19 minus CL20 | **`+0.051398`** | `-0.243815` | **`+0.022468`** |

These are matched-step aggregate comparisons on the same 96-image contract.
`[measured]` Face-quality finalization was not yet available for CL19/CL20 at
the evidence cutoff, so no TOPIQ-Face promotion claim is made. The current
CL19/CL20 images and 96-row ID tables are complete; `22k`/`24k` are not.
`[measured]`

CL19 is therefore the highest-information base for a CA question. Adding CA
to CL20 first would mix the CA hypothesis with a curriculum whose current
identity outcome is effectively CL14-level. A later CL20 transfer is useful
only after a CL19-based CA arm passes its own causal and quality gates.

## 5. Proposed experiment

### Priority 1 — `CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r1`

**Base config:** `src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml`

**New config:**
`src/configs/CL21_cosmic_true_soft_router_resididca_v3_24k.yaml`

**Single scientific change:** Enable the existing corrected residual
identity-token CA v3 in `up_blocks.0` and `up_blocks.1`, rank `64`, gate init
`0.02`, gate max `0.20`; extend writer telemetry without changing its forward
equation.

**Hypothesis:** CL19's spatial reference lane supplies morphology and
correspondence, while a bounded target-query/PhotoMaker-ID-token CA residual
can add identity evidence without disturbing native prompt structure or
CL19's face-boundary handover.

**Prediction:** Subject-v2 ID improves by at least `0.01` over CL19 at one or
more matched gates, with no loss of CL19's mask ownership, face attachment,
or prompt behavior. The gain should be larger on identity-heavy cells than on
scene/layout metrics.

**Main risk:** The residual is unused, duplicates identity already present in
native PhotoMaker CA, or reintroduces expression/accessory artifacts. A
bounded branch can also look safe simply because its gate stays near zero.

**Trainable contract:** CL19 has `2,240 / 219,217,920`. Reusing E17's 36-site
v3 contract adds `108` tensors / `5,406,756` parameters, for an expected
**`2,348 / 224,624,676`**. `[code]` This exact count must be re-derived from
the composed real model and asserted at startup; it must not be copied as an
unchecked assumption.

Suggested delta:

```yaml
defaults:
  - CL19_cosmic_true_soft_fullquery_router_24k
  - _self_

model:
  ba_residual_identity_ca_v3_enabled: true
  ba_residual_identity_ca_v3_groups: [up_blocks.0, up_blocks.1]
  ba_residual_identity_ca_v3_rank: 64
  ba_residual_identity_ca_v3_gate_init: 0.02
  ba_residual_identity_ca_v3_gate_max: 0.20

# These remain off: v3 is a separately versioned processor.
disable_branched_ca: true
train_branched_ca_lora: false
```

The following remain fixed: `use_branched_attention=true`, target-Q/reference-
K/V spatial BA, `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, CL19's soft
router, Cosmic training rows and order, optimizer and LR schedule, loss,
24k budget, batch 2, validation base, seed, prompts, references, resolved
boxes, DDIM50, CFG 5, PhotoMaker/BA onset steps, subject-v2 metric, and
step-0/every-2k 96-image validation. The legacy branched CA count must remain
zero. `[code]`

#### Decision gates

1. **Parity:** The complete step-zero 96-image panel must be pixel-identical
   to a fresh CL19 step-zero replay, or an explicitly justified numerical
   tolerance must be pre-registered before training. Zero-initialized v3 is
   expected to make the residual exactly zero. Stop on unexplained drift.
2. **Installation/ownership:** Exactly `36` residual identity-CA sites,
   `0` legacy CA sites, and the asserted trainable/optimizer/checkpoint counts.
   Save/load must reproduce the processor class, selected groups, rank, gate,
   and output tensors exactly.
3. **Causal use:** Log active ID-token count, gate, native-face RMS,
   residual-face RMS, and residual/native ratio by `up0`/`up1`. Add a detached
   matched-versus-shuffled PhotoMaker-identity diagnostic. A gain with a
   collapsed residual is not a CA result.
4. **Early safety at `2k`/`4k`:** `96/96` panel/table completion, no NaNs,
   no E12-like face plates, no new person/layout drift, and no subject-v2 ID
   loss greater than `0.01` versus CL19 at a matched step.
5. **Scientific gate at `8k`:** Continue only if ID is at least neutral within
   `0.005` and branch-use telemetry is nonzero and bounded, or if a predefined
   hard identity/occlusion slice shows a visually credible gain.
6. **Promotion:** At a matched selected checkpoint, require subject-v2 ID
   `>= CL19 + 0.01`, text regression no worse than `0.20`, mask IoU no worse
   than `0.005`, `96/96` detection, and blind review showing no increase in
   face plates, seams, duplicate goggles/glasses, hand-eye fusion, or
   expression distortion. Face-quality metrics must be complete before a
   final promotion decision.

### Priority 2 — conditional CL20 transfer, not a parallel first launch

**Proposed config:**
`src/configs/CL22_cosmic_bigcelebs_curriculum_resididca_v3_24k.yaml`

**Single scientific change:** Add the identical, already-promoted v3 CA
element to CL20; keep the sealed CL20 schedule exact.

**Hypothesis:** If CL21 proves a useful CA mechanism, the curated hard-case
rows may provide training examples where that mechanism matters.

**Prediction:** It should beat CL20 at matched gates and preferably close part
of the CL20-to-CL19 gap without losing CL20's text score.

**Risk:** A two-arm interaction study is wasteful before the CA main effect is
established; CL20 is currently not an identity improvement over CL14.

**Gate:** Do not launch unless CL21 passes causal-use, early safety, and at
least one `+0.01` matched ID gate with acceptable visuals. Apply the same
ownership and full-96 contract as CL21.

## 6. Why the current CL19/CL20 runs must not be changed

Changing either existing run now would create an uninterpretable composite
trajectory. Their `0-20k` checkpoints were trained without corrected CA and
their immutable Comet identities describe specific single-delta hypotheses.
`[code] [measured]` Their checkpoints contain no trained legacy or v3 CA state;
turning CA on at inference would invent random/untrained weights, exactly the
failure already exposed by Task A's historical temporary-validation bug.
`[report]`

A continuation that installs v3 at `20k` would answer “can a newly initialized
CA residual adapt after an already-trained CL19 model?” It would not answer
whether SA and CA co-adaptation improves the original architecture. That may
be a later efficiency diagnostic, but it is not the first clean experiment.

## 7. Implementation plan

1. Add the CL21 child config and immutable experiment JSON; do not edit CL19,
   CL20, or their records.
2. Extend `tools/validate_CL15_CL20_config.py` or create a narrowly named CL21
   validator that proves every CL19 field is equal except the v3 enablement,
   group/rank/gate fields, writer comment/metrics, expected ownership, and run
   identity.
3. Keep `disable_branched_ca=true` and `train_branched_ca_lora=false`; assert
   `36` `ResidualIdentityCrossAttnProcessorV3` instances and zero legacy/hard-v2
   CA instances.
4. Add v3 telemetry names to the writer allowlist. E17's missing telemetry
   must not recur. Preserve old writer behavior for every existing config.
5. Run focused synthetic checks for target-query source, active-token gather,
   native-CA equality outside/inside the face before residual addition,
   zero-init parity, finite first-backward gradients, mask locality, and
   doubled reference-lane preservation.
6. Verify exact schema-v2 checkpoint round-trip and validation processor-copy
   parity. Train and validation must install the same CL19 SA router and v3 CA
   processors.
7. Package as a new one-A100 job only after checking current Serv
   Running/Pending allocations and the normal six-A100 ceiling. No job is
   authorized or launched by this report.

## 8. Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Legacy branched CA was part of the original architecture | High | Original one-ID handoff/configs explicitly enable it. `[report] [code]` |
| Cosmic CA-on amplified global corruption | High | Exact fixed-checkpoint reproduction plus audited CA-off evaluation. `[measured]` |
| CA was not universally harmful | High | Leak-free one-ID matched matrix: coherent CA-on and materially higher ID. `[measured]` |
| The legacy processor routes the identity result to the wrong batch half and is non-local | High | Direct current and archived processor inspection. `[code]` |
| E12 rejects hard ID-only face CA replacement | High | Large matched ID loss plus repeated face-plate artifacts. `[measured] [report]` |
| E17 is safe but not a demonstrated positive | High | Mean delta crosses/leans negative and required telemetry was absent. `[measured] [report]` |
| CL19 is the best current base for a new CA test | High | Exact architecture comparison and matched `20k` metrics. `[code] [measured]` |
| CL19 + residual CA will improve identity | Medium-low | Plausible architectural complement; not yet tested on this substrate. `[hypothesis]` |
| CL20 should receive CA after CL21 succeeds | Low-medium | Possible data/mechanism interaction; current CL20 identity is CL14-level. `[hypothesis] [measured]` |

## 9. Not established

- It is not established that CL19 finishes above its `18k` peak; only the
  complete `20k` panel is available at this cutoff.
- It is not established that CL19 beats plain PhotoMaker; at `20k` it remains
  `0.052639` below the matched PM0 step-zero identity mean.
- It is not established that the legacy one-ID CA gain would survive the
  current 96-image, eight-identity protocol or current dataset.
- It is not established that v3's E17 null result was caused by its weak E15
  base; that is the reason to test, not a conclusion.
- It is not established that identity-token CA will fix spatial correspondence,
  small-face resolution, goggles, hands, or tears. Those failures may require
  the separate SA/ROI/ownership mechanisms already under study.
- Aggregate identity alone cannot promote CL21. Historical E12 and skiing
  examples show that an identity score can coexist with an invalid face.

## 10. Reproducing the evidence

Run from `diffusion_template/` in the existing `photomaker` environment:

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh
conda activate photomaker

# Refresh the four immutable runs. The project .env supplies the API key.
set -a
source .env
set +a
python tools/comet/export_comet_runs.py \
  --manifest tools/comet/comet_runs_12Aug_PM0_CL14_CL19_CL20.json \
  --output-dir comet_data/12Aug_PM0_CL14_CL19_CL20_refresh \
  --output-json comet_data/12Aug_PM0_CL14_CL19_CL20_refresh/comet_runs_export.json

# Historical decisions and legacy/corrected processor equations.
rg -n "branched cross|disable_branched_ca|CA on|CA off" \
  2026-07-24_test_branch_one_id_overfit_handoff.md \
  docs/experiments/2026-07-25_task_a_cosmic_faceonly_noca_4k_results.md \
  docs/experiments/2026-07-25_task_b_checkpoint_diagnostic_matrix_results.md

rg -n "class BranchedCrossAttnProcessor|class HardIdentity|class ResidualIdentity" \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/model/photomaker_branched/identity_ca_processor_v2.py \
  src/model/photomaker_branched/residual_identity_ca_processor_v3.py
```

The refresh used the per-image table as the completion seal. Every selected
panel has `96` PNGs and one `97`-line CSV (header plus 96 rows). The combined
export SHA-256 at this cutoff is
`722646e42b10744164c6644d992b861a6455df995cf9d176477add9f548b8c18`.
`[measured]`

## 11. References

1. `2026-07-24_test_branch_one_id_overfit_handoff.md` — original CA-on
   architecture and the first exterior-drift hypothesis.
2. `docs/experiments/2026-07-24_cosmic_large_abcd_implementation.md` — Task A
   single-delta design and Task B fixed-checkpoint matrix.
3. `docs/experiments/2026-07-25_task_a_cosmic_faceonly_noca_4k_results.md` —
   genuine CA-off result and validation-integrity correction.
4. `docs/experiments/2026-07-25_task_b_checkpoint_diagnostic_matrix_results.md`
   — exact reproductions, CA interventions, and the contrary leak-free one-ID
   result.
5. `docs/experiments/2026-08-04_e11_e12_large_ds_ba_capacity_plan.md` — legacy
   CA source audit and corrected hard identity-CA design.
6. `comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS.md` — E12 measured failure
   and residual-CA successor design.
7. `analysis/2026-08-06_e13_e18_results_and_next_experiments.md` — E17 result,
   missing telemetry, and substrate limitations.
8. `analysis/2026-08-11_cl19_soft_router_architecture_vs_cl14_pose_adapt_ca_mixing.md`
   — CL19 equations and distinction between `attn2` CA and SA memory mixing.
9. `src/model/photomaker_branched/attn_processor_cleanest.py` — current legacy
   branched CA plus CL19 SA router.
10. `src/model/photomaker_branched/identity_ca_processor_v2.py` and
    `residual_identity_ca_processor_v3.py` — corrected hard and residual CA
    implementations.
11. `comet_data/12Aug_PM0_CL14_CL19_CL20_refresh/comet_runs_export.json` —
    immutable-key metric histories and current complete panels.
