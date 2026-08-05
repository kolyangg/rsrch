# Clean BA32 at 32k: updated plateau analysis and architecture recommendations

**Date:** 2 August 2026  
**Primary run:** `rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1`  
**Immutable Comet ID:** `700240d8f90b48cfa2cc16f8ff2886b6`  
**Evidence cutoff:** complete fixed-96 validation at step 32,000  
**Scope:** high-priority model, attention, objective, optimization, checkpoint,
and validation issues; dataset-policy work is intentionally out of scope

This report updates and partially supersedes
[`2026-08-01_large_dataset_big_celebs_ba_architecture_recommendations.md`](2026-08-01_large_dataset_big_celebs_ba_architecture_recommendations.md).
The earlier report correctly identified the failed BA-only contract and proposed
a residual reference branch. The clean 32k run now removes the largest
correctness confound and lets us distinguish a real architectural ceiling from
the earlier trainable-state bug.

## Executive decision

The plateau is real under the clean training contract, but it is **not evidence
that the dataset is exhausted or that rank 32 by itself is the ceiling**.

The clean run passed an exact ownership contract of **840 tensors / 31,948,800
parameters**, all in the 70 branched self-attention processors. It nevertheless
reaches its best identity similarity, `0.3347`, at 18k and oscillates around
`0.31–0.33` through 32k. During the same period, text similarity stays high and
face quality rises to a much stronger level than in the old fail-open run.

That combination is the key result:

> The current BA-only model is very capable of learning a clean, structurally
> plausible **generic face adaptation**, but it has a weak and poorly controlled
> interface for converting a particular reference face into additional
> identity-specific information.

The highest-value change is therefore not a global rank increase. It is a
versioned, defaults-off **key-masked residual branched self-attention path**:

```text
frozen target path:
    y_base = SA(Q_target, K_target, V_target)

explicit reference path:
    y_ref  = SA(Q_target, K_reference, V_reference,
                valid_reference_face_keys_only)

bounded merge:
    y_target = y_base
             + target_face_mask
             * gate(layer, timestep, face_scale)
             * reference_output_adapter(y_ref)
```

This preserves the project invariant: target queries consume explicit
reference K/V. It keeps `pose_adapt_ratio=0` and
`ca_mixing_for_face=false`. It also preserves a frozen target-native attention
message instead of forcing the reference message to replace it inside the face
box.

Recommended order:

1. Run the existing 32k checkpoint once in `validation_native` mode and add
   causal zero/shuffled-spatial-reference diagnostics.
2. Fix trainable precision and checkpoint/runtime manifests for the next run.
3. Implement true reference-key masking and the residual SA-v2 processor.
4. Add semantic-layer and timestep gates; align training timestep support with
   inference.
5. Add a reference-causal objective and a full-image/boundary anchor.
6. Then compare **reference K/V + branch-output** rank 32 versus rank 64.
7. Only after that, add bbox-relative alignment, persistent reference memory,
   or a corrected target-query identity cross-attention branch.

Do not increase the existing `noise_and_ref` rank globally, full-finetune the
U-Net, reactivate the current branched CA processor, or use a nonzero
`pose_adapt_ratio` as a workaround.

## Evidence and controls

### Runs and local evidence

| Run | Comet ID | Relevant local export |
|---|---|---|
| Clean scheduled BA32 | `700240d8f90b48cfa2cc16f8ff2886b6` | `../comet_data/rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1/step_32000/` |
| Same scheduled data, historical fail-open ownership | `7c8b04738250479aac2a186ee3c96942` | `../comet_data/rhca_big_celebs_scheduled_v1_40k_full96_r1/` |
| BigCelebs historical fail-open control | `569cc685ff9144f5a9b42bf70e14e040` | `../comet_data/rhca_big_celebs_sameid_40k_full96_r1/step_32000/` |
| Large Dataset, Neb | `a99db1fb953d4511827672380e6c1645` | `../comet_data/rhca_large_dataset_sameid_40k_full96_r4/` |
| Large Dataset, two-GPU Serv | `db32f157e75a4798b2dfa530477c66d6` | `../comet_data/rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu/step_32000/` |

The canonical download index is
[`docs/experiments/2026-08-01_large_dataset_big_celebs_comet_downloads.md`](../docs/experiments/2026-08-01_large_dataset_big_celebs_comet_downloads.md).

All cited validation points use the fixed 96-image panel, one image per item,
the same prompts, seeds, references, generated/reference bboxes, RealVisXL V4
validation base, DDIM inference scheduler, 50 steps, CFG, and metric
definitions. The current report does not treat face-IQA as an identity metric.

### What Priority 0 successfully fixed

The clean run directly verified:

```text
branched SA processors                   70
branched CA processors                   0
requires-grad tensors                    840
optimizer tensors                        840
requires-grad parameters          31,948,800
generic lora_adapter trainables           0
default PhotoMaker adapter trainables     0
checkpoint format                         trainable_unet_v2
step-0 images                              96
step-0 detected faces                      94
```

It no longer suffers from the swallowed `AttnProcessor2_0.parameters()`
exception, the 171.29M-parameter fail-open optimizer, or omission of a live
trainable adapter from schema-v2 weights.

The remaining findings below are therefore about the clean branch, not a
misreported optimizer.

## 1. What the training behaviour uncovers

### 1.1 The clean branch reaches an identity ceiling, not a general quality ceiling

The complete clean trajectory is:

| Step | ID similarity | Text similarity | TOPIQ-Face mean | TOPIQ-Face p10 | Coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3063 | 26.4229 | 0.6225 | 0.5118 | 0.9063 |
| 2k | 0.1519 | 27.9289 | 0.6700 | 0.5189 | 0.7396 |
| 4k | 0.2744 | 27.7866 | 0.7146 | 0.5982 | 0.9583 |
| 6k | 0.2668 | 27.8009 | 0.7149 | 0.5997 | 0.9479 |
| 8k | 0.3192 | 27.6029 | 0.7266 | 0.6126 | 1.0000 |
| 10k | 0.3039 | 27.6916 | 0.7300 | 0.5992 | 0.9896 |
| 12k | 0.3258 | 27.5448 | 0.7353 | 0.6237 | 0.9896 |
| 14k | 0.3290 | 27.6196 | 0.7354 | 0.6012 | 0.9792 |
| 16k | 0.3330 | 27.7044 | 0.7361 | 0.6166 | 0.9896 |
| **18k** | **0.3347** | 27.7832 | 0.7392 | 0.6253 | 1.0000 |
| 20k | 0.3159 | 27.8560 | 0.7380 | 0.6211 | 0.9896 |
| **22k** | 0.3168 | 27.7897 | **0.7472** | 0.6273 | 0.9896 |
| 24k | 0.3329 | 27.9149 | 0.7414 | 0.6253 | 1.0000 |
| 26k | 0.3224 | 27.9888 | 0.7396 | 0.6281 | 0.9792 |
| 28k | 0.3230 | 27.8127 | 0.7398 | 0.6261 | 0.9792 |
| 30k | 0.3099 | 27.9300 | 0.7397 | 0.6077 | 0.9896 |
| 32k | 0.3273 | 27.8757 | 0.7353 | **0.6281** | 1.0000 |

After 8k, identity has no sustained positive slope. It oscillates while face
quality and text remain high. The branch is still changing the network and
still improving some face statistics, but those changes do not consistently
make the generated face more like the requested person.

The 2k identity collapse followed by recovery is also informative. With the
generic U-Net adapters frozen, the hard-replacement branch initially moves
away from the pretrained identity solution before it learns a useful
equilibrium. That is consistent with an intrusive merge rather than a
near-no-op residual initialization.

### 1.2 The exact scheduled dirty-versus-clean comparison isolates the trade-off

The strongest available controlled comparison uses the same pinned scheduled
dataset rows, rank, hard attention math, loss, LR, validation inputs, and
step-0 state. The historical arm trained the unintended generic/default
adapters; the clean arm trains only the 31.95M BA tensors.

| Step | Dirty ID | Clean ID | Clean − dirty | Dirty text | Clean text | Clean − dirty | Dirty face mean | Clean face mean | Clean − dirty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .3063 | .3063 | .0000 | 26.4229 | 26.4229 | .0000 | .6225 | .6225 | .0000 |
| 2k | .2403 | .1519 | -.0884 | 27.3804 | 27.9289 | +.5485 | .6318 | .6700 | +.0381 |
| 4k | .3147 | .2744 | -.0402 | 27.2809 | 27.7866 | +.5057 | .6599 | .7146 | +.0547 |
| 6k | .3203 | .2668 | -.0536 | 27.0734 | 27.8009 | +.7275 | .6640 | .7149 | +.0509 |
| 8k | .3451 | .3192 | -.0259 | 27.0820 | 27.6029 | +.5208 | .6759 | .7266 | +.0507 |
| 10k | .3595 | .3039 | -.0557 | 26.9826 | 27.6916 | +.7090 | .6711 | .7300 | +.0589 |
| 12k | .3727 | .3258 | -.0468 | 26.9440 | 27.5448 | +.6007 | .6757 | .7353 | +.0596 |
| 14k | .3684 | .3290 | -.0394 | 26.8822 | 27.6196 | +.7375 | .6688 | .7354 | +.0666 |

This comparison answers the key question raised by the earlier audit:

- the saved generic rank-32 U-Net adapter in the dirty arm contributed useful
  identity capacity;
- it did so while degrading prompt similarity and crop-based face quality;
- the clean BA path is better behaved, but less identity-discriminative;
- simply restoring broad generic trainables would likely recover some ID score
  while reintroducing the same trade-off and weakening causal BA attribution.

The dirty arm also updated the default PhotoMaker adapter without saving it,
so it is not a complete-state architecture candidate. The table is evidence
about the direction of capacity allocation, not a reason to restore the old
bug.

### 1.3 More unseen identities do not move the clean plateau

The scheduled plan gives a stronger count than an approximate dataset pass:

| Event | Rows consumed | Unique targets | Unique identities | Unique ordered pairs |
|---|---:|---:|---:|---:|
| Clean identity peak, 18k steps | 36,000 | 35,225 | 20,390 | 35,814 |
| Clean 32k validation | 64,000 | 61,402 | 25,710 | 63,166 |
| Entire 40k plan | 80,000 | 75,770 | 27,284 | 78,483 |

Between the 18k identity peak and 32k, the model receives 26,177 new targets
and 5,320 additional identities from the schedule, yet does not raise the
identity ceiling. It has still seen only 25,710 of the sealed dataset's 68,648
identities at 32k.

This does not prove every identity is equally useful, but it rules out the
simple explanation that the model stopped because it had exhausted the
available examples.

### 1.4 Optimization is still active after identity stops improving

Four-thousand-step medians from the clean Comet history show:

| Window | Median training loss | Median total grad norm |
|---:|---:|---:|
| 0–4k | .1326 | .00964 |
| 4–8k | .1262 | .00851 |
| 8–12k | .1282 | .00711 |
| 12–16k | .1326 | .01028 |
| 16–20k | .1310 | .01535 |
| 20–24k | .1286 | .01025 |
| 24–28k | .1282 | .01230 |
| 28–32k | .1310 | .01108 |

The loss is noisy and essentially flat; gradients do not vanish. The optimizer
continues making updates at a constant `1e-4` LR. The validation oscillation is
therefore more consistent with a moving equilibrium or objective mismatch than
with a dead branch.

### 1.5 Rank 32 is used, but the checkpoint does not demonstrate a rank ceiling

The schema-v2 32k weights contain 420 LoRA matrix pairs: six projections at
each of 70 SA sites. A direct small-matrix SVD of each effective `B @ A` delta
finds:

| Projection | Median entropy effective rank | Median stable rank | Median directions above 1% of largest singular value |
|---|---:|---:|---:|
| target/noise K | 16.31 | 1.26 | 32 |
| target/noise Q | 14.17 | 1.21 | 32 |
| target/noise V | 11.53 | 1.22 | 32 |
| reference K | 11.23 | 1.22 | 32 |
| reference Q | 12.52 | 1.14 | 32 |
| reference V | 7.50 | 1.03 | 32 |

All 32 directions are numerically active in the median projection, but the
spectrum is top-heavy. Effective delta norms also continue growing between 16k
and 32k. This is evidence that the adapters are not trivially dead; it is not
evidence that doubling rank will repair the attention normalization, merge,
timestep, or objective.

The correct interpretation is:

- a targeted rank-64 test is justified eventually;
- a global `noise_and_ref` rank-64 run is not the first experiment;
- rank should be spent on reference K/V and the reference output basis after
  the branch interface is corrected.

### 1.6 Visuals support a stable-structure / weak-ID diagnosis

A paired audit of 20k and 32k hard cases across the clean, BigCelebs dirty, and
two-GPU Large Dataset runs included Jisoo, Jensen, and Marion skiing, jumping,
dancing, kickboxing, crying, reading, rushing, drumming, and night-riding.

Observed visually:

- faces generally remain attached and scene/body structure stays coherent;
- clean 32k is not a catastrophic regression from clean 20k;
- later faces are often polished, with strong crop-based IQA;
- there is no consistent 20k→32k gain in identity-specific facial structure;
- several hard images change expression/detail without becoming visibly more
  like the reference identity.

This supports the metric split. It does not prove which layer causes the
ceiling, and the fixed-96 panel still contains only eight identities.

## 2. High-priority issues in the current code

### 2.1 Reference masking does not mask attention keys

Current code in
[`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py)
does this:

```python
ref_face_hidden = ref_hidden * ref_mask_flat
key_face = self._k_ref(attn, ref_face_hidden)
value_face = self._v_ref(attn, ref_face_hidden)
hidden_face = F.scaled_dot_product_attention(
    q_face, key_face, value_face,
    dropout_p=0.0,
    is_causal=False,
)
```

Tokens outside the reference bbox are zeroed before K/V projection, but they
remain in the softmax denominator. This has four consequences:

1. branch strength depends on the fraction of tokens covered by the reference
   bbox;
2. the same face with more surrounding padding produces a weaker attention
   message;
3. valid values are mixed with probability mass assigned to invalid zero
   values;
4. a rank increase cannot undo the missing normalization contract.

This is a direct code defect in the intended meaning of a reference-face mask.
The mask should be an additive or boolean **key mask**, while the target mask
should gate output locations.

There is a related dilution in `face_embed_strategy=id`: non-ID text tokens are
zeroed, but standard reference-lane cross-attention still sees all 77 token
positions. That is lower priority than spatial K/V masking, but the new
reference lane should gather actual ID tokens or explicitly mask the inactive
positions.

### 2.2 The face attention message is replaced rather than added to a target-native message

Current merge:

```python
hidden_bg = SA(q_target * (1 - mask), k_target, v_target)
hidden_face = SA(q_target * mask, k_reference_face, v_reference_face)
merged = hidden_bg * (1 - mask) + hidden_face * mask * scale
```

The transformer residual later re-adds the incoming target hidden state, so
the entire target state is not erased. However, **the target self-attention
message inside the face box is erased**. The model must choose between target
SA geometry and reference SA identity instead of learning a bounded residual
on top of target-native structure.

The only global branch control is `scale=1.0`; there is no trainable per-layer
or per-timestep gate and no branch-specific output projection. The frozen
shared `attn.to_out` must serve both messages.

This is the strongest architectural explanation for the early identity
equilibrium: making the reference path stronger risks pose/geometry and making
it weaker loses identity.

### 2.3 Half of the clean capacity adapts the target/noise path

`noise_and_ref` creates and trains:

```text
noise_to_q, noise_to_k, noise_to_v
ref_to_q,   ref_to_k,   ref_to_v
```

At rank 32 this is exactly 31,948,800 parameters. Approximately half are in
the target/noise projections. `noise_to_q` means even target queries are
trainable rather than being a stable PhotoMaker query basis.

The exact 70-site distribution from the 32k manifest is:

| U-Net group | Sites | Hidden width | Current six-projection rank-32 params |
|---|---:|---:|---:|
| `down_blocks.1` | 4 | 640 | 983,040 |
| `down_blocks.2` | 20 | 1280 | 9,830,400 |
| `mid_block` | 10 | 1280 | 4,915,200 |
| `up_blocks.0` | 30 | 1280 | 14,745,600 |
| `up_blocks.1` | 6 | 640 | 1,474,560 |

For 1024px SDXL these attention sites are expected at 64×64 and 32×32 token
resolutions; the runtime implementation should record actual sequence lengths
rather than relying on labels such as `mid_16`.

The clean metric profile is exactly what this allocation can produce: strong
generic face refinement and weak marginal identity discrimination. A stronger
BA should freeze target Q/K/V and allocate capacity to reference K/V plus a
branch-local output adapter.

### 2.4 Training applies BA at timesteps where inference does not

In [`lora2.py`](../src/model/photomaker_branched/lora2.py), the clean run uses
`train_ba_all_steps=true`, samples a uniform scalar timestep in `[0, 999]`,
repeats it across the batch, and always runs `two_branch_predict`.

Inference uses:

```text
steps 0–9:   text only / no identity
steps 10–14: PhotoMaker
steps 15–49: PhotoMaker + branched attention
```

Therefore BA is optimized on high-noise regions of the diffusion process in
which it is never called during the fixed 50-step inference protocol. The
training helper also passes `step_idx=0` unconditionally, so any future
step-index gate added without correcting this path will silently see the wrong
step.

The active training timesteps must be derived from the same scheduler and
50-step index window used by inference, and the normalized timestep/log-SNR
must be passed explicitly into every processor.

### 2.5 Face epsilon MSE does not require correct-reference dependence

The clean run applies `_masked_face_mse` on every optimizer step. The crop MSE
is correctly normalized over the crop; face-area normalization is not the
problem.

The problem is causal attribution. The loss never asks whether:

- the spatial reference is the correct identity;
- shuffling only spatial reference K/V makes the prediction worse;
- zeroing the spatial branch reduces identity;
- the model is improving because of PhotoMaker ID tokens, generic target Q/K/V
  adaptation, or explicit reference K/V.

The model can reduce average face denoising error by learning a generic face
prior. The rising face-IQA and flat identity curve show that this is not merely
a theoretical loophole.

The first correction should keep PhotoMaker prompt embeddings fixed and
shuffle **only** `reference_latents` plus `mask4_ref`; otherwise the diagnostic
changes both PhotoMaker identity conditioning and spatial BA at once.

### 2.6 Trainable parameters and Adam moments are BF16

Direct inspection of `checkpoint-epoch16.pth` at 32k found:

```text
840 trainable parameter tensors       torch.bfloat16
840 Adam exp_avg tensors              torch.bfloat16
840 Adam exp_avg_sq tensors           torch.bfloat16
840 Adam step tensors                 torch.float32
```

This follows from creating `BranchLoRALinear` in the frozen U-Net dtype and
then optimizing those tensors directly. BF16 frozen activations are
appropriate; BF16 trainable LoRA weights and BF16 optimizer moments are an
avoidable precision limitation, particularly under small late-stage
gradients.

Between 16k and 32k, the median exact-unchanged fraction is about 39.9% for
reference `lora_A` elements, while reference `lora_B` continues changing.
This does not prove BF16 caused the plateau, but it is enough to make FP32
trainables a required control before using long-run behavior to judge fine
capacity differences.

### 2.7 The canonical validation is still a hybrid processor base

The clean run intentionally uses explicit `legacy_full_copy` for comparison.
It copies complete stateful processor state from the SDXL training U-Net into
processors hosted by the RealVis validation U-Net. This transfers frozen
effective Q/K/V buffers as well as learned BA deltas, while the rest of the
validation U-Net remains RealVis.

Within-run trends are meaningful because the hybrid rule is constant. Absolute
architecture interpretation remains limited.

The existing schema-v2 checkpoint now makes a clean fixed-checkpoint
`validation_native` evaluation possible: instantiate processors from the
RealVis effective base, then load only exact trainable deltas. Run that before
training another architecture. Keep the historical legacy curve as a labeled
secondary comparison, not the primary semantics for new models.

### 2.8 The checkpoint manifest does not fully encode runtime attention semantics

Schema v2 records trainable names/shapes and several flags, but not all
behavior that changes predictions. Missing or insufficiently explicit fields
include:

- processor implementation/version;
- target-query source and merge kind;
- true-key-mask mode;
- `strict_face_routing`;
- trainable dtype;
- timestep policy and active inference window;
- gate kind and semantic layer groups;
- reference-lane conditioning policy;
- base/PhotoMaker fingerprint and load context.

The `use_attn_v2` flag is also misleading: `branched_runtime.py` currently
imports `attn_processor_cleanest` unconditionally. New architecture selection
must be one explicit version string mapped to one processor class, with an
error for unknown values.

For resume, the training-base and PhotoMaker fingerprints should match
strictly. For deliberate alternate-base validation, a separate
`validation_transfer` load context should allow the base difference while
logging it explicitly.

## 3. Architectural improvements in priority order

### Priority 0 — fixed-checkpoint semantics and causal diagnostics

This is not another long training run. Use the existing 32k schema-v2 weights.

Run four fixed-checkpoint conditions on the unchanged full-96 panel:

1. `legacy_full_copy`, correct spatial reference;
2. `validation_native`, correct spatial reference;
3. `validation_native`, spatial reference branch zeroed, PhotoMaker ID tokens
   unchanged;
4. `validation_native`, spatial references shuffled across identities,
   PhotoMaker ID tokens unchanged.

Record per-image paired ID/text/face-quality deltas and the hard-case anatomy
count. This answers two prerequisite questions:

- how much of the 32k curve is a validation-base hybrid effect;
- whether the trained spatial BA branch is causally identity-specific.

Do not train rank 64 until the correct reference materially beats both zero and
shuffled spatial references.

### Priority 1 — key-masked residual branched self-attention v2

This is the highest-priority model change.

#### Required invariants

```text
target Q source           frozen target hidden state
target base K/V           frozen target hidden state
reference branch K/V      reference hidden state only
reference key support     valid reference-face tokens only
spatial output support    target face mask only
pose_adapt_ratio          0
ca_mixing_for_face        false
branched CA               disabled
initial effect            zero or small bounded residual
```

#### New versioned processor

Add a new file such as:

```text
src/model/photomaker_branched/residual_sa_processor_v2.py
```

Do not modify historical hard-routing math in place.

Core implementation sketch:

```python
class ResidualBranchedSelfAttnProcessorV2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ref_kv_rank: int = 32,
        output_rank: int = 32,
        gate_init: float = 0.10,
        gate_max: float = 1.0,
        trainable_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ref_to_k = BranchLoRALinear(..., rank=ref_kv_rank)
        self.ref_to_v = BranchLoRALinear(..., rank=ref_kv_rank)
        self.ref_out = ZeroInitResidualLoRA(
            hidden_size, rank=output_rank, dtype=trainable_dtype
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(logit(gate_init), dtype=trainable_dtype)
        )
        self.gate_t = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        self.gate_area = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        self.gate_max = float(gate_max)

    def __call__(
        self,
        attn,
        hidden_states,
        *,
        ba_denoise_progress,
        **kwargs,
    ):
        target, reference = hidden_states.chunk(2)
        target_residual, reference_residual = target, reference
        target = apply_norms(attn, target)
        reference = apply_norms(attn, reference)

        # Frozen target-native self-attention.
        q_target = reshape_heads(attn.to_q(target), attn.heads)
        k_target = reshape_heads(attn.to_k(target), attn.heads)
        v_target = reshape_heads(attn.to_v(target), attn.heads)
        base_message = sdpa(q_target, k_target, v_target)
        base_out = apply_frozen_to_out(attn, merge_heads(base_message))

        # Explicit target-Q -> reference-K/V branch.
        k_ref = reshape_heads(self.ref_to_k(reference), attn.heads)
        v_ref = reshape_heads(self.ref_to_v(reference), attn.heads)
        ref_key_bias = self.reference_key_bias(
            self.mask_ref, target_len=k_ref.shape[-2], dtype=q_target.dtype
        )
        ref_message = sdpa(
            q_target,
            k_ref,
            v_ref,
            attn_mask=ref_key_bias,
        )
        ref_delta = self.ref_out(merge_heads(ref_message))

        target_mask = self.target_output_mask(...)
        gate = self.bounded_gate(
            ba_denoise_progress,
            target_mask.float().mean(dim=(-2, -1)),
        )
        target_out = base_out + target_mask * gate * ref_delta

        # The reference lane propagates with frozen ordinary SA. It does not
        # need a trainable ref Q merely to provide target-facing K/V.
        reference_out = frozen_reference_sa(attn, reference)

        target_out = finish_residual(attn, target_out, target_residual)
        reference_out = finish_residual(attn, reference_out, reference_residual)
        return torch.cat([target_out, reference_out], dim=0)
```

#### True key-mask implementation

Use an additive mask to avoid ambiguity between PyTorch attention APIs:

```python
def make_reference_key_bias(mask_ref, *, seq_len, batch, dtype, device):
    keep = prepare_binary_mask(mask_ref, seq_len, batch).squeeze(1).squeeze(-1)
    valid_counts = keep.sum(dim=-1)
    if torch.any(valid_counts == 0):
        raise RuntimeError("Reference face mask has zero valid attention keys")

    bias = torch.zeros(batch, 1, 1, seq_len, device=device, dtype=dtype)
    return bias.masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)
```

Do not multiply reference hidden states by the mask before projection. Project
the reference normally and exclude invalid keys in attention. This also avoids
projection-bias leakage if a future projection has bias.

#### Merge initialization

Use exactly one zero barrier:

- initialize `ref_out.B` to zero;
- initialize the sigmoid gate to a nonzero value such as 0.05–0.10;
- do not initialize both the output and the gate to exact zero.

At step 0, the target output then matches frozen target PhotoMaker SA. The
first gradients train `ref_out.B`; reference K/V and the gate begin receiving
useful gradients as the output path becomes nonzero.

#### Why this is stronger despite fewer parameters

For all 70 current sites:

| Architecture | Trainable paths | Rank | Parameters |
|---|---|---:|---:|
| Current clean | target Q/K/V + ref Q/K/V | 32 | 31,948,800 |
| Proposed v2 | ref K/V + reference output | 32/32 | 15,974,400 + gates |
| Proposed v2 | ref K/V + reference output | 64/64 | 31,948,800 + gates |

The v2 rank-64 design uses approximately the same budget as the current clean
rank-32 design, but spends it entirely on the explicit reference message and
its merge.

### Priority 2 — semantic-layer, timestep, and face-scale gates

The current model uses the same full-strength takeover at all 70 sites. Start
v2 with scalar gates per processor, not per-head gates.

Recommended first semantic set:

| Group | Sites | Expected 1024px token resolution | First v2 treatment |
|---|---:|---:|---|
| `down_blocks.1` | 4 | 64×64 | off or very small gate |
| `down_blocks.2` | 20 | 32×32 | off in first candidate; add as ablation |
| `mid_block` | 10 | 32×32 | small gate |
| `up_blocks.0` | 30 | 32×32 | primary structural identity gate |
| `up_blocks.1` | 6 | 64×64 | primary detail gate |

The initial mid+up v2 candidate has 46 sites and about **10,567,680** rank-32
ref-K/V-plus-output parameters, before gates.

Resolve groups from full processor names and observed runtime sequence lengths.
Store the exact name list and hash in the checkpoint. Do not use dictionary
position or the current fractional `top_k` selector as the semantic contract.

Gate definition:

```python
progress = 1.0 - timestep.float() / float(num_train_timesteps - 1)
log_area = torch.log(target_mask.float().mean((-2, -1)).clamp_min(1e-4))
gate = gate_max * torch.sigmoid(
    gate_logit + gate_t * normalize(progress) + gate_area * normalize(log_area)
)
```

Safeguards:

- `gate_max <= 1` in the first experiments;
- log gate mean/p10/p90 by semantic group;
- log `RMS(reference_delta) / RMS(base_out)`;
- optionally cap extreme branch/base RMS with a defaults-off setting;
- regularize against a causally dead branch, not toward an arbitrarily high
  gate;
- keep the target face mask as the final spatial output gate.

#### Thread the real timestep through training and inference

In `branched_runtime.py`:

```diff
 runtime_cross_attention_kwargs.update(
     {
+        "ba_denoise_progress": normalized_denoise_progress(
+            t_batched, pipeline.scheduler.config.num_train_timesteps
+        ),
     }
 )
```

In the processor signature:

```diff
 def __call__(
     self,
     attn,
     hidden_states,
+    ba_denoise_progress=None,
     **kwargs,
 ):
```

Use the same helper in both paths. Do not infer training gates from
`step_idx`; training currently passes zero.

### Priority 3 — reference-causal training and prompt/background anchors

This is an objective change supporting the architecture, but it is essential
to prevent the stronger branch from becoming a better generic face prior.

#### Full + face + boundary objective

Add a new defaults-off loss rather than changing `MaskedDiffusionLoss`:

```python
class BranchedReferenceLoss(nn.Module):
    def forward(
        self,
        model_pred,
        target,
        face_bbox,
        pred_wrong_spatial_ref=None,
        **batch,
    ):
        full = F.mse_loss(model_pred.float(), target.float())
        face = _masked_face_mse(model_pred, target, face_bbox)
        ring = _face_boundary_ring_mse(model_pred, target, face_bbox)

        loss = self.full_weight * full
        loss = loss + self.face_weight * face
        loss = loss + self.ring_weight * ring

        if pred_wrong_spatial_ref is not None:
            wrong = _masked_face_mse(
                pred_wrong_spatial_ref, target, face_bbox
            ).detach()
            gap = wrong - face
            causal = F.relu(self.reference_margin - gap)
            loss = loss + self.reference_weight * causal

        return {
            "loss": loss,
            "loss_full": full,
            "loss_face": face,
            "loss_ring": ring,
            "reference_error_gap": gap if pred_wrong_spatial_ref is not None else None,
        }
```

The detached wrong-reference error avoids explicitly training the network to
make wrong references arbitrarily bad. It increases correct-reference pressure
only when the correct path does not beat the current wrong-reference baseline
by the configured margin.

#### Spatial-only reference shuffle

Refactor preparation so PhotoMaker prompt conditioning and spatial BA state are
separate objects:

```python
conditioning = prepare_photomaker_conditioning(ref_images, prompts)
spatial_reference = prepare_spatial_reference(ref_images, face_bbox_ref)

pred_correct = model.forward_with_spatial_reference(
    conditioning=conditioning,
    spatial_reference=spatial_reference,
    target_noise=noise,
    timesteps=timesteps,
)

pred_wrong = model.forward_with_spatial_reference(
    conditioning=conditioning,                 # unchanged
    spatial_reference=shuffle_different_id(
        spatial_reference, identity_ids
    ),
    target_noise=noise,                        # unchanged
    timesteps=timesteps,                       # unchanged
)
```

Start by logging the gap without training it. Then enable the auxiliary forward
on 10–25% of batches if the clean branch is not causally separated. Require a
different identity; skip rather than silently using a same-ID negative.

### Priority 4 — targeted capacity: branch output first, then reference rank 64

Only after Priorities 1–3 are stable:

1. keep output rank 32 and compare reference K/V rank 32 versus 64;
2. compare branch output rank 32 versus 64;
3. consider asymmetric ranks by group;
4. add a small branch-local nonlinear output adapter if rank alone is
   insufficient.

Recommended asymmetric candidate:

```yaml
model:
  ba_self_attention:
    groups:
      mid_block:
        ref_kv_rank: 32
        output_rank: 16
      up_blocks.0:
        ref_kv_rank: 64
        output_rank: 32
      up_blocks.1:
        ref_kv_rank: 64
        output_rank: 32
```

Optional nonlinear output path:

```python
delta = up_proj(F.silu(down_proj(ref_message)))
nn.init.zeros_(up_proj.weight)
```

This remains a BA mechanism because its input is explicitly the
target-query/reference-K/V attention message. Keep it face-masked and gated.

Do not add a broad generic U-Net LoRA first. The exact scheduled comparison
already shows that generic capacity buys ID score at a cost to text/quality and
makes BA attribution weaker.

### Priority 5 — bbox-relative alignment and persistent reference memory

After the residual branch shows a positive correct-versus-shuffled reference
gap, add geometry and cross-layer identity capacity.

#### Bbox-relative positional bias

For target queries and reference keys, compute positions in each face box:

```text
p_target = ((x - target_x0) / target_width,
            (y - target_y0) / target_height)
p_ref    = ((x - ref_x0) / ref_width,
            (y - ref_y0) / ref_height)
```

Feed relative offsets through a small per-head MLP and add the result to
reference-attention logits. Zero-initialize its final layer. Do not hard-warp
the reference face in the first version; pose and expression need flexibility.

#### Persistent reference memory

Pool 4–16 tokens from valid reference ROIs at `mid_block`/`up_blocks.0` and
make them available as an additional K/V bank in later up blocks:

```text
target Q attends [local reference spatial keys ; persistent identity keys]
```

Use separate spatial and memory gates and log both causal gaps. This adds
connections and capacity while retaining explicit target-Q/reference-K/V BA.

### Priority 6 — corrected target-query identity cross-attention v2

The current branched CA remains disabled and should stay disabled. If SA-v2 is
healthy, add a separate versioned CA processor:

```python
text_out = attention(
    Q_target(target_hidden),
    K_text(prompt_tokens),
    V_text(prompt_tokens),
)

id_tokens = gather_true_identity_tokens(
    photomaker_prompt_tokens,
    class_tokens_mask,
)
id_out = attention(
    Q_target(target_hidden),
    K_id(id_tokens),
    V_id(id_tokens),
)

target_out = text_out + target_face_mask * id_gate * id_output(id_out)
```

Gather actual identity tokens; do not leave 76 zero tokens in the softmax
denominator. Start only in `up_blocks.0`/`up_blocks.1`, rank 16, with a small
gate. `pose_adapt_ratio=0` and `ca_mixing_for_face=false` remain fixed.

## 4. Supporting code changes required before the next architecture run

### 4.1 Explicit architecture version instead of the ignored `use_attn_v2`

In `branched_runtime.py`:

```diff
-from .attn_processor_cleanest import (
-    BranchedAttnProcessor,
-    BranchedCrossAttnProcessor,
-)
+version = getattr(pipeline, "ba_architecture_version", "hard_replace_v1")
+if version == "hard_replace_v1":
+    from .attn_processor_cleanest import BranchedAttnProcessor
+elif version == "residual_sa_v2":
+    from .residual_sa_processor_v2 import ResidualBranchedSelfAttnProcessorV2
+    BranchedAttnProcessor = ResidualBranchedSelfAttnProcessorV2
+else:
+    raise ValueError(f"Unknown ba_architecture_version={version!r}")
```

Keep `hard_replace_v1` as the default for historical replay.

### 4.2 Processor-owned trainable-role enumeration

The current allowlist infers ownership from parameter-name substrings. Let the
new processor declare its trainables:

```python
class ResidualBranchedSelfAttnProcessorV2(nn.Module):
    def named_ba_trainables(self):
        for name, parameter in self.ref_to_k.named_parameters():
            yield f"ref_to_k.{name}", parameter, "ref_kv"
        for name, parameter in self.ref_to_v.named_parameters():
            yield f"ref_to_v.{name}", parameter, "ref_kv"
        for name, parameter in self.ref_out.named_parameters():
            yield f"ref_out.{name}", parameter, "ref_output"
        yield "gate_logit", self.gate_logit, "gate"
        yield "gate_t", self.gate_t, "gate"
        yield "gate_area", self.gate_area, "gate"
```

Build the exact global allowlist and optimizer groups from this enumeration.
Continue to assert exact requires-grad and optimizer membership on every rank.

### 4.3 FP32 branch parameters with BF16 frozen compute

`BranchLoRALinear` must permit a BF16 input and FP32 low-rank weights:

```diff
 def forward(self, x):
     base = F.linear(x, self.base_weight, self.base_bias)
-    delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
-    return base + delta * self.scaling
+    train_dtype = self.lora_A.dtype
+    delta = F.linear(
+        F.linear(x.to(train_dtype), self.lora_A),
+        self.lora_B,
+    )
+    return base + delta.to(base.dtype) * self.scaling
```

After processor installation:

```python
if model.branched_trainable_dtype == "fp32":
    for name, parameter in model.unet.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
```

Apply the same processor/trainable dtype in validation before loading state.
Add dtype to the schema-v2 manifest and startup contract. Verify optimizer
moments are FP32 after the first update.

### 4.4 Timestep policy toggle

In `lora2.py`:

```diff
-t_scalar = torch.randint(0, num_train_timesteps, (1,), device=device)
-timesteps = t_scalar.repeat(batch_size)
+if self.ba_training_timestep_policy == "uniform_all":
+    timesteps = torch.randint(
+        0, num_train_timesteps, (batch_size,), device=device
+    )
+elif self.ba_training_timestep_policy == "inference_active":
+    active = self.get_inference_active_timesteps(
+        num_inference_steps=50,
+        branched_start_step=15,
+    ).to(device)
+    choice = torch.randint(0, len(active), (batch_size,), device=device)
+    timesteps = active[choice]
+else:
+    raise ValueError(self.ba_training_timestep_policy)
```

Use `uniform_all` as the historical default and `inference_active` for v2.

### 4.5 Separate optimizer groups and a late-stage schedule

Suggested initial v2 values:

```yaml
optimizer_groups:
  ref_kv_lr: 5.0e-5
  ref_output_lr: 1.0e-4
  gate_lr: 2.0e-4
  weight_decay: 0.0

lr_scheduler:
  kind: cosine
  warmup_steps: 500
  final_lr_ratio: 0.1

trainer:
  max_grad_norm: 1.0
```

The clean run reaches full LR after only 20 steps and stays at `1e-4` through
40k. Its nonvanishing late gradients and oscillating ID curve justify a decay
arm. Do not combine the optimizer change with the first key-mask diagnostic;
use the same v2 architecture in a matched constant-versus-cosine comparison.

### 4.6 Expanded checkpoint manifest and load contexts

Add:

```python
architecture.update({
    "ba_architecture_version": self.ba_architecture_version,
    "processor_code_version": 2,
    "merge_kind": self.ba_self_attention.merge,
    "target_query_source": "frozen_target",
    "reference_key_mask": True,
    "semantic_processor_names": resolved_names,
    "semantic_processor_names_sha256": names_hash,
    "trainable_dtype": "float32",
    "timestep_policy": self.ba_training_timestep_policy,
    "branched_start_step": self.branched_attn_start_step,
    "num_inference_steps": self.num_inference_steps,
    "pose_adapt_ratio": 0.0,
    "ca_mixing_for_face": False,
    "base_model_id": configured_base_id,
    "photomaker_sha256": photomaker_sha256,
})
```

Loader API:

```python
load_state_dict_(state, context="training_resume")
load_state_dict_(state, context="validation_transfer")
```

- `training_resume`: require exact base and PhotoMaker fingerprints;
- `validation_transfer`: allow the explicit RealVis base difference, require
  architecture/name/shape/dtype compatibility, and log both fingerprints.

## 5. Proposed defaults-off configuration

```yaml
model:
  # Historical default remains hard_replace_v1.
  ba_architecture_version: residual_sa_v2
  branched_trainable_dtype: fp32
  strict_branched_install: true
  strict_trainable_contract: true
  branched_state_dict_mode: trainable_v2

  ba_self_attention:
    merge: residual
    target_query_source: frozen_target
    target_base_path: frozen_standard_sa
    reference_kv_source: reference_only
    reference_key_mask: true
    groups: [mid_block, up_blocks.0, up_blocks.1]
    ref_kv_rank: 32
    output_rank: 32
    output_zero_init: true
    gate_kind: layer_timestep_area
    gate_init: 0.10
    gate_max: 1.0
    relative_position_bias: false
    persistent_memory_tokens: 0

  ba_training_timestep_policy: inference_active

train_ba_only: true
disable_branched_sa: false
disable_branched_ca: true
train_branched_ca_lora: false

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false

loss_kind: branched_reference
loss:
  full_weight: 1.0
  face_weight: 1.0
  boundary_weight: 0.1
  reference_shuffle_probability: 0.0  # diagnostic first
  reference_margin: 0.0

validation_processor_base_mode: validation_native
strict_validation_processor_copy: true
```

The values are starting points, not established optima. Preserve all existing
validation inputs and metrics.

## 6. Controlled implementation and experiment ladder

| Order | Arm | Primary change | Stop/continue question |
|---:|---|---|---|
| D0 | Existing 32k fixed checkpoint | Native vs legacy; correct vs zero vs shuffled spatial reference | Is spatial BA causally identity-specific, and how large is the hybrid-base effect? |
| P0 | FP32 smoke/round-trip | FP32 trainables and optimizer state, one update only | Do mixed-dtype forward, optimizer, save/load, and validation match? |
| A1 | Key-mask diagnostic | True reference key mask with fixed scale sweep, no long run | What scale range avoids an amplitude jump? |
| A2 | Residual SA-v2 rank32 | Frozen target SA + key-masked reference residual + fixed gate | Does ID exceed clean BA32 without text/structure regression? |
| A3 | Semantic/timestep gates | Learned per-layer progress/area gates and inference-active sampling | Does the useful slope continue beyond 8–12k? |
| L1 | Reference-causal loss | Spatial shuffle diagnostic, then auxiliary margin if needed | Does correct-reference separation increase? |
| K1 | Targeted rank64 | Ref K/V 32→64 only | Is reference projection capacity limiting after routing is fixed? |
| O1 | Output rank64 | Branch output 32→64 only | Is merge/output basis the remaining bottleneck? |
| G1 | Relative bbox bias | Zero-init relative-coordinate bias | Do hard poses improve without identity/text loss? |
| M1 | Persistent memory | 4–16 identity memory tokens | Does cross-layer identity retention improve? |
| CA1 | Target-ID CA-v2 | Target queries attend gathered PhotoMaker ID tokens | Does a direct identity residual help without old CA corruption? |

### Prepared implementation and run mapping — 2 August 2026

| Ladder arm | Script / configuration | Run or output label | Status and scope |
|---|---|---|---|
| D0 | `launchers/neb/run_clean_ba32_32k_d0_validation_matrix.sh` with `big_celebs_scheduled_rhca_clean_ba32_40k` | `d0_clean_ba32_32k_legacy_matched`, `d0_clean_ba32_32k_native_matched`, `d0_clean_ba32_32k_native_zero_spatial`, `d0_clean_ba32_32k_native_shuffle_spatial` | Prepared, not launched. Uses the immutable clean run's `weights-epoch16.pth` at 32k. Every arm keeps PhotoMaker identity inputs matched; only the last two intervene on the BA spatial input. |
| P0 | Local `photomaker` environment smoke/round-trip checks plus the strict startup contracts in `train.py` and schema-v2 checkpoint loader | No Comet run; this is a one-update/preflight gate | Implemented locally. Mixed BF16/frozen plus FP32/trainable forward/backward, exact step-0 residual equality, masked-key invariance, zero-mask failure, optimizer ownership, Hydra composition, and checkpoint schema checks gate the long run. |
| A1 | `ResidualBranchedSelfAttnProcessorV2` true key bias in `src/model/photomaker_branched/residual_sa_processor_v2.py` | No standalone scale-sweep run prepared yet | The key-mask mechanism is implemented. A separate fixed-scale sweep remains a diagnostic after a nonzero residual checkpoint exists; it is not silently claimed as an independent completed arm. |
| A2 + A3 | `launchers/neb/start_rhca_big_celebs_scheduled_residual_sa_v2_40k.sh` with `big_celebs_scheduled_rhca_residual_sa_v2_40k` | `rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r1` | Prepared, not launched. This first integrated candidate combines residual SA-v2 rank32 with the semantic mid/up site set, bounded layer/timestep/area gates, and inference-active timestep sampling. The old hard-replacement control remains `launchers/neb/start_rhca_big_celebs_scheduled_clean_ba32_40k.sh` / `rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1`. |
| L1 diagnostic | Same residual-SA-v2 launcher/config | Same integrated run; `reference_shuffle_applied` and `reference_error_gap` curves | Diagnostic only in this version: a wrong-spatial-reference forward is sampled on 25% of eligible batches and detached; `reference_weight=0`, so it cannot alter optimization. Enable a nonzero margin only after the measured causal gap justifies it. |
| K1, O1, G1, M1, CA1 | None yet | None | Deliberately not prepared. These are promotion-stage experiments after D0 and the rank32 residual run pass the stated gates. K1 and O1 can be created with the existing independent `ba_ref_kv_rank` and `ba_output_rank` toggles; CA remains off. |

The integrated A2+A3 run is the implementation candidate, not a claim that
A2 and A3 have already been causally separated. If it promotes, create
single-factor fixed-gate and semantic/timestep-gate children before assigning
the gain to either component.

### Efficient gates

- 2k: fail on catastrophic face routing, step-0 mismatch, dead gate, or missing
  causal reference effect.
- 4k: require structural quality and text to remain competitive.
- 8k: compare paired ID slope and correct-versus-shuffled spatial-reference
  gap.
- 12k: stop dominated arms; the clean run already exposes its broad plateau by
  this region.
- 20k: continue only if identity, causal gap, or hard-case anatomy is still
  improving.
- 32k: reserved for promoted arms, not every ablation.

### Promotion criteria

1. Exact trainable, optimizer, dtype, and checkpoint manifest contracts pass.
2. Correct spatial reference beats shuffled and zero spatial reference on
   paired identity similarity.
3. Full-96 identity exceeds the clean peak `0.3347` by a practically meaningful
   margin; use paired per-image intervals rather than only the aggregate mean.
4. Text similarity does not repeat the dirty scheduled run's decline.
5. TOPIQ-Face p10/coverage and the fixed hard-case anatomy count do not regress.
6. Background/outside-face changes remain bounded.
7. Gate and branch/base RMS logs show a live, bounded reference path.

## 7. Bottom line

### Do next

- Treat the clean 32k run as the trustworthy hard-routing BA32 baseline.
- Use its checkpoint for native-base and causal-reference diagnostics.
- Implement true reference key masking.
- Preserve frozen target SA and add reference attention as a gated residual.
- Add a branch-local output projection.
- Train branch parameters and Adam moments in FP32.
- Align BA training timesteps with the fixed inference window.
- Make correct spatial reference use measurable.
- Then test rank 64 only in reference K/V and branch output.

### Do not do next

- Do not globally double the existing `noise_and_ref` rank.
- Do not restore broad generic/default adapter training as the main model.
- Do not full-finetune the U-Net.
- Do not reactivate the current branched CA processor.
- Do not use target K/V substitution or nonzero `pose_adapt_ratio` as the fix.
- Do not treat face-IQA improvement as identity improvement.
- Do not run every ablation to 40k merely because more dataset rows remain.

The clean run shows that the branch has enough capacity to transform face
quality substantially. What it lacks is a normalized, bounded, and causally
measurable way to turn reference information into identity-specific target
updates. Correcting that interface is more likely to unlock the remaining
dataset than adding undirected parameters.

## Evidence classification

### Directly observed

- Exact 840-tensor / 31.95M-parameter clean startup contract.
- Full clean fixed-96 metric trajectory through 32k.
- Exact same-schedule dirty-versus-clean metrics through 14k.
- Scheduled row/unique-target/unique-identity counts through 32k.
- BF16 trainable and Adam-moment dtypes in the 32k optimizer checkpoint.
- LoRA delta singular-value summaries from schema-v2 weights.
- Stable face/body structure in the paired local 20k/32k hard-case review.
- Continued nonzero gradient norms after identity plateaus.

### Directly established by code inspection

- Reference features are zeroed but reference keys are not masked from SDPA.
- Target self-attention messages are replaced inside the face box.
- Target/noise Q/K/V are trainable in `noise_and_ref`.
- No branch-local output adapter or learned gate exists.
- Training applies BA uniformly across all diffusion timesteps while inference
  begins BA at step 15/50.
- Training passes `step_idx=0` to the branched helper.
- Face-only epsilon MSE has no spatial-reference causal term.
- Legacy validation copies complete processor state across different bases.
- `use_attn_v2` does not select a different active processor implementation.

### Hypotheses requiring controlled experiments

- True key masking will extend identity improvement rather than only amplify
  the branch.
- A residual merge will improve the identity/pose/text frontier.
- FP32 trainables will materially raise the late identity ceiling.
- Mid/up semantic gating is better than all 70 sites.
- Targeted rank 64 adds useful identity after routing is corrected.
- Bbox-relative bias or persistent memory repairs the remaining hard poses.
