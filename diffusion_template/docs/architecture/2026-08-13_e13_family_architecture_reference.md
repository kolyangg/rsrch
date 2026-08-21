# E13-family architecture reference

For the baseline-first history—starting from the exact 2 June `main_clean`
commit, separating correctness/data/runtime repairs, and then deriving every
supported recipe—use
[`2026-08-18_june2_to_e13_family_architecture_lineage.md`](2026-08-18_june2_to_e13_family_architecture_lineage.md).

This document is the formula-level reference for every supported experiment in
`kit/e13-family-clean`: E13, BC_E13, CL14, CL14_CA, CL18, CL19, CL20, CL23,
CL27, and CL39. It
separates model architecture, training objective, dataset policy, validation
policy, and execution-only optimizations. Those categories must not be treated
as interchangeable experimental changes.

## 1. Family map

There are ten recipes and five inference-time attention architectures.

| Recipe | Inference architecture | Training-only change | Dataset change |
|---|---|---|---|
| E13 | Shared hard-replacement branched SA | None | Large Dataset |
| BC_E13 | Exactly E13 | None | Sealed BigCelebs |
| CL14 | Exactly E13 | Two-cell feather constructed; hard route thresholds it | Corrected Cosmic Large |
| CL14_CA | CL14 plus bounded residual ID-token CA | None beyond CL14 mask construction | Corrected Cosmic Large |
| CL18 | Exactly CL14 | Cross-view consistency loss | Cosmic with two same-ID reference candidates |
| CL19 | Two-cell cosine-router branched SA | None beyond CL14 mask | Corrected Cosmic Large |
| CL20 | Exactly CL14 | Deterministic curriculum order | Cosmic/BigCelebs, then Cosmic-only |
| CL23 | CL19 plus fixed denoising-progress low/high gains | None beyond CL14 mask | Corrected Cosmic Large |
| CL27 | Exactly CL23 | Frequency-surface auxiliary loss | CL23 data plus deterministic semantic occluders |
| CL39 | CL27 plus parameter-free null-key confidence | Same surface loss on the confidence-scaled route | Exactly CL27 |

The leaf configs are:

- [E13](../../src/configs/E13_large_ds_joint_shadow_sa128_24k.yaml)
- [BC_E13](../../src/configs/BC_E13_big_celebs_joint_shadow_sa128_24k.yaml)
- [CL14](../../src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml)
- [CL14_CA](../../src/configs/CL14_CA_cosmic_residual_identity_ca_24k.yaml)
- [CL18](../../src/configs/CL18_cosmic_crossview_spatial_consistency_24k.yaml)
- [CL19](../../src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml)
- [CL20](../../src/configs/CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml)
- [CL23](../../src/configs/CL23_cosmic_temporal_frequency_router_24k.yaml)
- [CL27](../../src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml)
- [CL39](../../src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml)

The shared configuration is
[e13_family_24k.yaml](../../src/configs/e13_family_24k.yaml). The runtime rejects
architecture or ownership combinations outside these declared leaves in
[`initialise_e13_contract`](../../src/model/photomaker_branched/e13_contract.py),
[`patch_unet_attention_processors`](../../src/model/photomaker_branched/branched_runtime.py),
and the configuration validators under [`tools/`](../../tools/).

## 2. Notation

At one U-Net attention layer, let:

- \(H_t\in\mathbb{R}^{B\times L\times d}\) be target/noisy latent tokens;
- \(H_r\in\mathbb{R}^{B\times L\times d}\) be noised spatial-reference tokens;
- \(M_t,M_r\in[0,1]^{B\times L\times1}\) be target and reference face masks;
- \(P_g\) be the generation prompt tokens;
- \(P_i\) be the PhotoMaker-conditioned identity-prompt tokens;
- \(W_q,W_k,W_v,W_o\) be attention projections;
- \(h\) be the number of heads and \(d_h=d/h\).

Scaled dot-product attention is

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

An effective LoRA projection is

$$
W_{\mathrm{eff}}x
=W_0x+\frac{\alpha}{r}B(Ax),
$$

where \(r\) is the adapter rank. In this branch, the hard branched-SA
projections use rank 128, the generic effective outer adapter uses rank 32, and
the pretrained PhotoMaker `default` adapter uses rank 64. The exact cloned
effective-projection construction is in
[`_clone_effective_linear`](../../src/model/photomaker_branched/attn_processor_cleanest.py).

The U-Net is called with a doubled batch:

$$
H=[H_t;H_r],\qquad P=[P_g;P_i].
$$

[`two_branch_predict`](../../src/model/photomaker_branched/branched_runtime.py)
constructs this layout. Every supported recipe keeps
`pose_adapt_ratio=0` and `ca_mixing_for_face=false`; target-face K/V therefore
come from the explicit spatial reference, not a blend with target features.

## 3. Shared hard-replacement branched self-attention

E13, BC_E13, CL14, CL14_CA, CL18, and CL20 use the same self-attention equation.
CL19 replaces only the target merge; CL23 and CL27 extend that CL19 merge as
described later.

### 3.1 Target background message

The target/noise projections produce

$$
Q_t=W_q^tH_t,\quad K_t=W_k^tH_t,\quad V_t=W_v^tH_t.
$$

The background branch is

$$
B=\operatorname{Attn}\bigl(Q_t\odot(1-M_t),K_t,V_t\bigr).
$$

`strict_face_routing=false`, so the K/V source remains the full target tensor;
only the queries and final merge are mask-routed. This is implemented in
[`BranchedAttnProcessor.__call__`](../../src/model/photomaker_branched/attn_processor_cleanest.py).

### 3.2 Target face message from the spatial reference

The reference face source is

$$
H_r^f=H_r\odot M_r.
$$

With the required pose-adaptation ratio \(\rho=0\), the historical blend

$$
H_{\mathrm{face}}=(1-\rho)H_r^f+\rho(H_t\odot M_t)
$$

reduces exactly to \(H_r^f\). The face message is

$$
F=\operatorname{Attn}
\bigl(Q_t\odot M_t,\;W_k^rH_r^f,\;W_v^rH_r^f\bigr).
$$

The binary reference mask zeros unsupported K/V vectors but does not remove
their token positions from the softmax. These zero vectors are intentional
historical “zero sinks”; changing them to a true attention-key mask would be a
different architecture.

### 3.3 Target merge and reference lane

The target message is selected once by the target mask:

$$
Y_t=W_o\left((1-M_t)\odot B+M_t\odot F\right).
$$

The reference half preserves full reference self-attention:

$$
Y_r=W_o\operatorname{Attn}
\bigl(W_q^rH_r,W_k^rH_r,W_v^rH_r\bigr).
$$

The processor returns \([Y_t;Y_r]\), applies the layer's ordinary residual
connection and output rescale, and the U-Net ultimately consumes only the
target prediction. The exact merge is in
[`attn_processor_cleanest.py`](../../src/model/photomaker_branched/attn_processor_cleanest.py);
processor selection and mask refresh are in
[`branched_runtime.py`](../../src/model/photomaker_branched/branched_runtime.py).

### 3.4 Native cross-attention

For every recipe except CL14_CA, cross-attention is the native PhotoMaker/text
path:

$$
C_t=\operatorname{Attn}(W_qH_t,W_kP_g,W_vP_g),\qquad
C_r=\operatorname{Attn}(W_qH_r,W_kP_i,W_vP_i).
$$

`disable_branched_ca=true` and `train_branched_ca_lora=false` specifically
disable the old branched-CA processor. “CA disabled” in run descriptions means
legacy branched CA is disabled; ordinary native SDXL/PhotoMaker cross-attention
is still active.

## 4. Shared training, inference, and ownership contract

### 4.1 Diffusion objective

For latent noise \(\epsilon\), noisy latent \(z_t\), and model prediction
\(\epsilon_\theta\), the full diffusion loss is

$$
L_{\mathrm{full}}=\operatorname{mean}
\left[(\epsilon_\theta(z_t,t)-\epsilon)^2\right].
$$

The clean recipes set `loss_kind=masked_alternating`, `masked_loss_step=1`.
Therefore every batch uses the face-crop MSE:

$$
L_{\mathrm{face}}=\frac{1}{B}\sum_{b=1}^{B}
\operatorname{mean}_{c,x,y\in\mathcal{B}_b}
\left[(\epsilon_{\theta,b,c,x,y}-\epsilon_{b,c,x,y})^2\right],
$$

where \(\mathcal{B}_b\) is the target face box scaled from 1024 pixels to the
latent grid. `lambda_face=0.1` belongs to the alternative blended loss and is
not used by `masked_alternating`. The implementation is
[`MaskedDiffusionLoss`](../../src/loss/diffusion_loss.py).

CL18 and CL27 add the auxiliary terms defined in their sections. All other
recipes optimize \(L=L_{\mathrm{face}}\).

### 4.2 Learning-rate trajectory

Every recipe has 24,000 optimizer steps and base LR \(10^{-4}\). For completed
step count \(s\), the LR multiplier is

$$
f(s)=
\begin{cases}
s/20, & s<20,\\
1, & 20\le s\le14{,}000,\\
0.1+0.9\cdot\frac{1+\cos\!\left(\pi
\frac{s-14{,}000}{10{,}000}\right)}{2}, & 14{,}000<s\le24{,}000.
\end{cases}
$$

See [`WarmupHoldCosineLR`](../../src/lr_schedulers/lr_schedulers.py) and
[`warmup_hold_cosine_24k.yaml`](../../src/configs/lr_scheduler/warmup_hold_cosine_24k.yaml).

### 4.3 Training versus denoising activation

`train_ba_all_steps=true`, so training routes every sampled diffusion timestep
through branched attention. Fixed validation uses DDIM 50, CFG 5, PhotoMaker
from denoising step 10, and spatial BA from step 15. Training and inference use
the same installed processors; only their activation schedule differs.

### 4.4 Exact trainable ownership

For E13, BC_E13, CL14, CL18, CL19, CL20, CL23, CL27, and CL39:

| Role | Tensors | Parameters |
|---|---:|---:|
| Branched SA rank 128 | 840 | 127,795,200 |
| Generic effective adapter rank 32 | 700 | 30,474,240 |
| PhotoMaker default effective adapter rank 64 | 700 | 60,948,480 |
| Total | 2,240 | 219,217,920 |

CL14_CA adds 108 tensors / 5,406,756 parameters and totals 2,348 tensors /
224,624,676 parameters. Ownership, optimizer groups, checkpoint names, shapes,
dtypes, and architecture manifest are enforced by
[`e13_contract.py`](../../src/model/photomaker_branched/e13_contract.py). Only
schema-v2 trainable U-Net checkpoints are accepted by this clean family.

## 5. E13

E13 is the base architecture described in Sections 3 and 4. Its unique input
policy is `LargeDatasetTrain`:

1. choose target image \(x_t\) from identity \(i\);
2. choose reference \(x_r\) uniformly from the same identity with
   \(x_r\ne x_t\);
3. independently apply the configured target/reference transforms;
4. route the target box to \(M_t\) and the reference box to \(M_r\).

In set notation,

$$
x_r\sim\operatorname{Uniform}
\left(\mathcal{I}_i\setminus\{x_t\}\right).
$$

Singleton identities fail before training; self-reference is not silently
substituted. See
[`LargeDatasetTrain`](../../src/datasets/large_dataset.py), the
[E13 config](../../src/configs/E13_large_ds_joint_shadow_sa128_24k.yaml), and
the shared dataset wiring in
[`all_datasets.yaml`](../../src/configs/datasets/all_datasets.yaml).

## 6. BC_E13

BC_E13 is not a new model architecture. It uses exactly the E13 equations,
trainables, loss, optimizer, and validation schedule. The controlled variable
is the dataset:

$$
\theta_{\mathrm{BC\_E13}}
=\operatorname{Train}(A_{\mathrm{E13}},D_{\mathrm{BigCelebs}}).
$$

The sealed BigCelebs loader requires:

- a distinct same-identity reference;
- one and only one `img` trigger in the prompt;
- valid 1024-coordinate face boxes;
- minimum face short side 192 pixels;
- exact release-manifest fields rather than silent record filtering.

See [`BigCelebsTrain`](../../src/datasets/big_celebs.py) and the
[BC_E13 config](../../src/configs/BC_E13_big_celebs_joint_shadow_sa128_24k.yaml).

## 7. CL14

CL14 keeps the E13 attention equations but changes the Cosmic input policy and
the training target mask.

### 7.1 Corrected Cosmic reference geometry

For reference face area \(A_r\), 1024 canvas size \(S\), and sampled desired
face fraction \(u\sim U(0.06,0.30)\), the reference resize factor is

$$
s_r=\sqrt{\frac{uS^2}{A_r}}.
$$

The resized reference is cropped/padded into a 1024 target-frame canvas. Its
face center is aligned to the target face center plus a random positional
offset bounded by 0.15 of the canvas. Edge fill is used; reference flips are
disabled. This prevents hidden scale mismatch and the earlier in-place copy
shortcut. See
[`compose_target_frame_reference`](../../src/datasets/reference_frame.py) and
[`CosmicLargeAdaptedTrain`](../../src/datasets/cosmic_large_adapted.py).

Prompts are pose-first and limited to 50 words:

$$
p=\text{“class img, pose, background, appearance”}.
$$

Inherited `img` copies are removed from the appearance fragment so the trigger
contract remains unambiguous.

### 7.2 Two-cell training-mask construction

E13 uses a hard target mask. CL14 changes only the target mask constructed
during training. For feather width \(k=2\), successive inward boundary rings
receive

$$
w_j=\frac{j}{k+1},\qquad j\in\{1,2\},
$$

so the outer and inner boundary rings are constructed with weights \(1/3\) and
\(2/3\), while the interior remains 1. The shared hard-route processor then
resizes the mask and applies \(\mathbf{1}[M_t>0.5]\). Consequently, current
CL14-family hard SA sees the outer ring as background and the inner ring as
face: the effective attention boundary is hard and contracted by one latent
cell. CL14_CA applies the same \(>0.5\) rule to its CA residual. CL18 alone also
uses the pre-threshold feather values when constructing its consistency loss.
The reference mask and all inference masks stay binary. The construction is in
[`PhotomakerBranchedLora._bbox_to_mask`](../../src/model/photomaker_branched/lora2.py).
The hard-SA threshold is in
[`BranchedAttnProcessor._prepare_mask`](../../src/model/photomaker_branched/attn_processor_cleanest.py),
and the CL14_CA threshold is in
[`ResidualIdentityCrossAttnProcessorV3._prepare_spatial_mask`](../../src/model/photomaker_branched/residual_identity_ca_processor_v3.py).

The exact leaf is
[`CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`](../../src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml).

## 8. CL14_CA

CL14_CA retains all CL14 self-attention, data, loss, and scheduler behavior. It
adds a corrected residual identity-token cross-attention processor only to
`up_blocks.0` and `up_blocks.1`.

### 8.1 Native base path

The ordinary target and reference CA outputs remain

$$
N_t=W_o\operatorname{Attn}(W_qH_t,W_kP_g,W_vP_g),
$$

$$
N_r=W_o\operatorname{Attn}(W_qH_r,W_kP_i,W_vP_i).
$$

The implementation fuses these independent rows into one batch operation and
then splits them. This is execution-only batching, not cross-row attention.

### 8.2 Active identity-token residual

Let \(I(P_i)\) gather only tokens selected by PhotoMaker's class-token mask.
Target queries attend those identity tokens using the native frozen Q/K/V
projections:

$$
U=\operatorname{Attn}
\left(W_qH_t,W_kI(P_i),W_vI(P_i)\right).
$$

A rank-64 residual-only LoRA projection has no base linear term:

$$
\Delta=BAU,
$$

where \(A\in\mathbb{R}^{64\times d}\),
\(B\in\mathbb{R}^{d\times64}\), and \(B\) is initialized to zero. Its
per-token RMS normalization is

$$
\widehat{\Delta}
=\frac{\Delta}
{\sqrt{\max\left(\operatorname{mean}_d(\Delta^2),10^{-12}\right)}}.
$$

The clamp is essential: without it, the zero forward value has an undefined
`sqrt` gradient at initialization.

The scalar gate is

$$
g=0.20\,\sigma(\gamma),
\qquad g_{\mathrm{init}}=0.02.
$$

The final target and reference outputs are

$$
Y_t=N_t+M_t\odot g\widehat{\Delta},\qquad Y_r=N_r.
$$

Thus a fresh CL14_CA processor is exactly native CL14 CA because \(B=0\), and
training can only add a bounded face-local residual. Legacy branched CA remains
disabled.

The processor is
[`ResidualIdentityCrossAttnProcessorV3`](../../src/model/photomaker_branched/residual_identity_ca_processor_v3.py);
its exact block selector is in
[`branched_runtime.py`](../../src/model/photomaker_branched/branched_runtime.py),
and its leaf is
[`CL14_CA_cosmic_residual_identity_ca_24k.yaml`](../../src/configs/CL14_CA_cosmic_residual_identity_ca_24k.yaml).
The focused port and verification history is in
[the CL14_CA extension ledger](2026-08-13_cl14_ca_clean_extension.md).

## 9. CL18

CL18 has exactly the CL14 inference architecture. Its only scientific change is
a training-time same-identity cross-view objective applied with probability
\(p=0.25\).

For the same target latent, noise, timestep, prompt, ID tokens, and paired
reference noise, let

$$
T=M_t\odot\operatorname{stopgrad}
\left(\epsilon_\theta(z_t,r_1)\right)
$$

be the primary-reference teacher, and

$$
S=M_t\odot\epsilon_\theta(z_t,r_2)
$$

be the alternate-reference student. The consistency term is

$$
L_{\mathrm{cv}}
=\operatorname{SmoothL1}_{M_t}(S,T)
+0.10\left(1-\cos(\operatorname{vec}S,\operatorname{vec}T)\right).
$$

The total loss on an activated batch is

$$
L=L_{\mathrm{face}}+0.05L_{\mathrm{cv}}.
$$

On the other 75% of batches, \(L=L_{\mathrm{face}}\). The alternate view
requires at least three same-identity candidates for the target. At inference
there is one reference and no consistency pass, so CL18 generations use the
CL14 equation.

See the cross-view block in
[`PhotomakerBranchedLora.forward`](../../src/model/photomaker_branched/lora2.py),
the [CL18 leaf](../../src/configs/CL18_cosmic_crossview_spatial_consistency_24k.yaml),
and dual-reference selection in
[`CosmicLargeAdaptedTrain`](../../src/datasets/cosmic_large_adapted.py).

## 10. CL19

CL19 changes the target self-attention merge in all selected down, mid, and up
groups. It computes two complete messages from full target queries:

$$
N=\operatorname{Attn}(W_q^tH_t,W_k^tH_t,W_v^tH_t),
$$

$$
R=\operatorname{Attn}
\left(W_q^tH_t,W_k^r(H_r\odot M_r),W_v^r(H_r\odot M_r)\right).
$$

Unlike the hard route, queries are not pre-multiplied by face/background
masks. The binary reference zero-sink behavior remains unchanged.

Let \(M^{(0)}=M_t\). Repeated 3×3 binary erosion gives

$$
M^{(j)}=\operatorname{Erode}_{3\times3}(M^{(j-1)}).
$$

For two transition cells, ring \(j\) is

$$
G_j=M^{(j-1)}-M^{(j)},\qquad j\in\{1,2\},
$$

and its cosine weight is

$$
c_j=\frac{1-\cos\left(\pi j/3\right)}{2}.
$$

Therefore \(c_1=0.25\), \(c_2=0.75\), the face interior is 1, and the exterior
is 0. Calling the resulting router \(C\), CL19 merges once:

$$
Y_t=W_o\left((1-C)\odot N+C\odot R\right).
$$

It does not multiply partial queries and then blend a second time. This “full
messages, one router, one merge” invariant is enforced in
[`BranchedAttnProcessor._call_soft_router`](../../src/model/photomaker_branched/attn_processor_cleanest.py).
The exact groups and two-cell width are in
[`CL19_cosmic_true_soft_fullquery_router_24k.yaml`](../../src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml).

## 11. CL20

CL20 has exactly the CL14 model and loss. Its scientific variable is a sealed,
sequential 48,000-row curriculum with batch size two, hence 24,000 optimizer
steps. For schedule row \(n\), optimizer step is

$$
s(n)=\left\lfloor n/2\right\rfloor.
$$

The source rule is

$$
D(n)=
\begin{cases}
\mathrm{BigCelebs}, & s(n)<20{,}000\ \land\ n\bmod5=0,\\
\mathrm{Cosmic}, & \text{otherwise}.
\end{cases}
$$

This produces an 80/20 Cosmic/BigCelebs row mixture through step 19,999 and
Cosmic-only re-anchoring for steps 20,000–23,999. BigCelebs rows rotate evenly
through:

1. synthetic-small-face examples;
2. occlusion-caption examples;
3. action-caption examples.

The schedule owns row order, target/reference pairing, target scale, reference
face fraction/position, and flip decisions. The loader requires sequential
sampling, verifies the schedule and both source manifests by SHA-256, rejects
self-reference, and verifies resume row \(=2\times\) completed optimizer steps.

See
[`build_cl20_hardcase_schedule.py`](../../tools/datasets/build_cl20_hardcase_schedule.py),
[`CL20HardcaseCurriculumTrain`](../../src/datasets/cl20_hardcase_curriculum.py),
and the [CL20 leaf](../../src/configs/CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml).

## 12. CL23

CL23 keeps CL19's complete native message \(N\), reference message \(R\), and
two-cell cosine router \(C\). It changes only the routed
reference-minus-native message

$$
D=R-N.
$$

A fixed separable 5x5 Gaussian kernel
\([1,4,6,4,1]/16\) splits this message into

$$
D_{\mathrm{low}}=G*D,\qquad
D_{\mathrm{high}}=D-D_{\mathrm{low}}.
$$

For scheduler timestep \(t\) and training scheduler length \(T\), real
denoising progress is

$$
p=1-\frac{t}{T-1}.
$$

The fixed gains are

$$
a_{\mathrm{low}}(p)=0.50+0.35p,\qquad
a_{\mathrm{high}}(p)=0.75+0.50p.
$$

The target message is therefore

$$
Y_t=N+C\odot\left(
a_{\mathrm{low}}D_{\mathrm{low}}+
a_{\mathrm{high}}D_{\mathrm{high}}
\right).
$$

There are no new parameters. Training uses progress derived from the sampled
diffusion timestep; validation uses the matching live scheduler timestep. The
equation is implemented in
[`BranchedAttnProcessor._call_temporal_frequency`](../../src/model/photomaker_branched/attn_processor_cleanest.py),
the runtime passes progress in
[`two_branch_predict`](../../src/model/photomaker_branched/branched_runtime.py),
and the exact leaf is
[`CL23_cosmic_temporal_frequency_router_24k.yaml`](../../src/configs/CL23_cosmic_temporal_frequency_router_24k.yaml).

## 13. CL27

CL27 has exactly the CL23 inference equation and the same 2,240 trainable
tensors. Its only model-training change is an auxiliary loss in
`up_blocks.0/1`; its only data change is a deterministic synthetic top-object
mask on 25% of Cosmic samples, using seed 150017.

Let \(O\) be that top-object mask, \(M\) the target face mask, and

$$
M_{\mathrm{top}}=O\odot M,\qquad
M_{\mathrm{visible}}=\max(M-M_{\mathrm{top}},0).
$$

Only samples with non-empty top and visible regions are eligible. For the
already routed low/high components \(L=C\odot a_{\mathrm{low}}D_{\mathrm{low}}\)
and \(H=C\odot a_{\mathrm{high}}D_{\mathrm{high}}\), the top-object penalty is

$$
E_{\mathrm{top}}
=\operatorname{mean}_{M_{\mathrm{top}}}(H^2)
+0.25\operatorname{mean}_{M_{\mathrm{top}}}(L^2).
$$

For routed delta \(\Delta=L+H\), visible-face preservation uses

$$
r=\frac{\operatorname{RMS}_{M_{\mathrm{visible}}}(\Delta)}
{\operatorname{stopgrad}\left(
\operatorname{RMS}_{M_{\mathrm{visible}}}(N)\right)},
$$

and the auxiliary objective is

$$
L_{\mathrm{surface}}
=0.02E_{\mathrm{top}}
+0.005\max(0,0.35-r)^2.
$$

The complete objective is
\(L=L_{\mathrm{face}}+L_{\mathrm{surface}}\). The mask is supervision only:
it is never passed to validation or inference routing. The implementation is
in
[`BranchedAttnProcessor._frequency_surface_loss`](../../src/model/photomaker_branched/attn_processor_cleanest.py),
[`collect_frequency_surface_aux_loss`](../../src/model/photomaker_branched/lora2_helpers.py),
and
[`CosmicLargeAdaptedTrain`](../../src/datasets/cosmic_large_adapted.py). The
exact leaf is
[`CL27_cosmic_frequency_surface_energy_24k.yaml`](../../src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml).

## 14. CL39

CL39 keeps CL27's data, objective, masks, temporal-frequency schedule, and
2,240 trainable tensors. Its only change is a parameter-free confidence on the
already-routed CL27 reference delta in `up_blocks.0/1`.

For target query (q_{hi}), masked reference key (k_{hj}), head width (d),
and (L) reference tokens:

$$
p_{hij}=\operatorname{softmax}_j\left(
\frac{q_{hi}^{\mathsf T}k_{hj}}{\sqrt d}\right),
\qquad
e_i=\frac{1}{H}\sum_h
\frac{-\sum_j p_{hij}\log(p_{hij}+10^{-8})}{\log L}.
$$

The detached virtual-null mass and retained reference fraction are

$$
n_i=\sigma\left(\frac{e_i-0.75}{0.08}\right),
\qquad
c_i=\operatorname{clip}(1-0.75n_i,0.25,1).
$$

If Δ_CL27 is CL27's routed low/high reference-minus-native delta, selected
blocks output

$$
Y_{CL39}=N+c\odot\Delta_{CL27}.
$$

Thus ambiguous reference matches retain more native target self-attention;
they do not emit a zero value. Confidence is detached, so CL39 adds neither a
predictor nor a new gradient path. Other blocks remain exact CL27. The code is
[`BranchedAttnProcessor._null_key_confidence`](../../src/model/photomaker_branched/attn_processor_cleanest.py),
and the leaf is
[`CL39_cosmic_null_key_confidence_router_24k.yaml`](../../src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml).

## 15. Validation contract

All recipes preserve the fixed 96-image `manual_val` panel with one image per
item at step 0 and every 2,000 optimizer steps, RealVisXL V4.0, DDIM 50, CFG 5,
batch 12, PhotoMaker start 10, and BA start 15. E13/BC_E13 use their sealed bbox
cache; CL14-family runs use the sealed CL14 cache.

CL14_CA, CL18, CL19, CL20, CL23, CL27, and CL39 use the isolated corrected
subject-v2 validation wrapper:

- `bbox_overlap_v2` chooses the reference face owned by the declared box;
- `id_sim_subject_v2` measures the generated face owned by the exact BA box;
- historical best-over-any-face identity remains as an audit metric.

This is a validation identity-selection repair, not an attention-architecture
change. See
[`photomaker_branched_cl18_cl20.py`](../../src/pipelines/photomaker_branched_cl18_cl20.py),
[`face_subject_selector.py`](../../src/face_subject_selector.py), and
[`all_metrics_subject_v2.yaml`](../../src/configs/metrics/all_metrics_subject_v2.yaml).

## 16. Execution-only optimizations

The following changes are intended to preserve mathematical output:

- batch preparation encodes conditioning together instead of one sample at a
  time;
- unused text-only conditioning is skipped because BA is active at every
  training timestep;
- resized attention masks are cached only on the current mask tensor;
- CL14_CA computes native target/reference CA rows in one batched call;
- CL14_CA builds active identity-token indices once per U-Net call;
- training diagnostics are stacked for one synchronization, and one-GPU runs
  bypass the unnecessary distributed gather;
- face-quality scoring is deferred until training completes;
- Diffusers' recursive `unet.attn_processors` property is resolved once per
  collector rather than once per selected attention layer;
- disabled auxiliary collectors return before processor-map resolution, and
  CL27 eligibility remains on-device rather than synchronizing a Python bool;
- full-activation temporal-frequency telemetry is disabled for the clean CL23
  CL27, and CL39 launches because none consumes it.

These optimizations must not change seeds, sampled timesteps, reference noise,
loss terms, processor state, or generated pixels. Their controls are in
[`e13_family_24k.yaml`](../../src/configs/e13_family_24k.yaml),
[`lora2.py`](../../src/model/photomaker_branched/lora2.py),
[`lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py),
[`residual_identity_ca_processor_v3.py`](../../src/model/photomaker_branched/residual_identity_ca_processor_v3.py),
and [`sdxl_trainers.py`](../../src/trainer/sdxl_trainers.py).

## 17. Fail-closed checks and launch references

The principal static gates are:

- [`validate_e13_family_config.py`](../../tools/validate_e13_family_config.py)
- [`verify_cl14_generation_parity.py`](../../tools/verify_cl14_generation_parity.py)
- [`validate_cl14_ca_config.py`](../../tools/validate_cl14_ca_config.py)
- [`validate_cl18_cl20_config.py`](../../tools/validate_cl18_cl20_config.py)
- [`validate_cl23_cl27_config.py`](../../tools/validate_cl23_cl27_config.py)

The shared launcher is
[`run_e13_family_24k_1gpu.sh`](../../launchers/active/run_e13_family_24k_1gpu.sh).
Exact Serv YAMLs and environment requirements are indexed in
[`serv_run_packages/README.md`](../../serv_run_packages/README.md). These files
prepare jobs but do not authorize submission.

The non-negotiable architectural invariants are:

```text
target Q receives explicit spatial-reference information through branched SA
pose_adapt_ratio = 0
ca_mixing_for_face = false
legacy branched CA = disabled
training and validation install the checkpoint's exact processors
checkpoint trainable names/shapes/dtypes = exact manifest contract
fixed validation inputs and metric definitions do not drift silently
```
