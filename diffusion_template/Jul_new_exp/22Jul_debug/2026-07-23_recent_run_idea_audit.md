# Recent-run architecture idea audit

## Scope

This audit maps useful findings from both `Jul_new_exp` and the older
`diffusion_template/debug_04Jul` experiment record into the current step-zero
search. It is intentionally selective: mechanisms already falsified are not
re-run under new names.

## Strong recurring findings

### 1. Step-zero activity is a legitimate architecture criterion

The July 4 record repeatedly found that untrained N3a was visually stronger
than trained checkpoints and that denoising MSE could degrade identity. For the
current pre-training comparison, broad face validity, visible branch activity,
alignment, and spatial containment therefore take priority over matched-ID
metrics. This supports retaining `n3a_fullgrid_up_core_ring_anchor` after its
96-case run: face MAE `0.09124`, face detection 96/96, landmarks `0.02872`, bbox
IoU `0.93418`, and exact protected output outside the trusted core.

### 2. Disable branched cross-attention

N11 and later analyses identified broad branched CA as a drift/melt pathway;
SA-only training improved steadily. Every current promising configuration keeps
branched CA disabled and explicitly disables trainable CA processors.

### 3. Target-coordinate queries preserve pose and attachment

N29/N31/N32/N33 showed that target queries attending compact reference memory,
with PhotoMaker retained as the global generator, solved the older copied-pose,
detached-head, hand, hair, and scene failures. NN7 clean-patch attention keeps
the target query coordinates and the PM epsilon baseline.

### 4. High-resolution up1 is safe but may be weak; staged up0 can add shape

The older record found that late/high-resolution sites preserve geometry but
often provide only local expression/texture changes, while broad/all-site
routes damage chroma and pose. Multiple proposals recommend low authority in a
small up0 route plus stronger up1 detail authority. The new staged NN7 screens
test effective gates `0.02/0.065` and `0.03/0.075` for up0/up1 respectively.

### 5. Canonical/part-aware memory is preferable to raw crop coordinates

N30's square bbox crop was not true alignment. N32's richer face patches stayed
safe but entangled pose, expression, illumination, and crop. Later N35/N40
proposals recommended ordered canonical eye/nose/mouth evidence. The current
landmark-local mapping and semantic-radius masks are a lightweight step-zero
version: they keep target pose, map reference neighborhoods by five-point
correspondence, and allow exact target fallback outside eligible parts.

### 6. Bounded residual/ownership and independent PM output are consistently useful

N29 and later configurations established a strong preservation contract:
PhotoMaker is the global baseline, BA is localized, and output outside the face
region is restored. Current NN7 and repaired N3a screens keep this contract.
For N3a, `core_ring` is the target/reference ownership arbitration; for NN7,
the direct candidate has an explicit gate, cap, and BA0 causal control.

## Mechanisms now being tested

| Historical idea | Current experiment |
|---|---|
| N3a visible full spatial ownership, but protected | canonical full-grid core `0.68`, all 96 |
| smaller ownership for tighter alignment | N3a core `0.35/0.50/0.60` |
| delay reference takeover after PM pose | N3a starts 7/20 and 8/20; NN7 starts 8/20 |
| tighter trusted final write region | N3a erosion `0.15/0.20`; NN7 erosion `0.22` |
| target fallback / N24-style arbitration | full-grid dual attention `0.25/0.35/0.50` |
| confidence-weighted target fallback | full-grid confidence residual `0.25/0.50/0.75` |
| clean patch memory | NN7a_init-v2 PMv2 context patches |
| geometric correspondence | landmark-IDW local 3x3 attention |
| semantic face-part eligibility | landmark radii `0.18/0.25` |
| layer specialization | staged up0/up1 gates |
| causality | identical-architecture BA0 plus cyclic wrong-reference mapping |

## Deferred ideas

- Clean PMv2 identity-token fallback is well motivated by NN5/NN6, but its
  connector is exact-zero before training. A native step-zero run is inactive
  and cannot rank the idea. It should be evaluated as a trained independent
  lane or after a principled warm initialization is designed.
- Canonical learned face-part resamplers (N35/N40) require training. Landmark
  neighborhoods are used here only as a non-trained topology screen.
- PhotoMaker identity attenuation is explicitly deferred. Older reports warn
  that it can create apparent BA authority by weakening a strong baseline.
- Broad/all-site CA residuals, absolute reference output replacement, pose
  adaptation, and CA face mixing are excluded because they repeatedly caused
  drift, pose copying, or PhotoMaker collapse.
- Raising scale/caps without improving routing is excluded; older PPR results
  showed it amplifies generic expression and rendering edits.

## Current ranking after the diverse-eight hybrid screen

1. `n3a_fullgrid_up_dual35_div8` — strongest validated balance on all 96:
   face MAE `0.07723`, landmark `0.01134`, bbox `0.95526`, 96/96 faces. On 24:
   face MAE `0.08121` (near canonical activity), landmark shift `0.01223`, bbox
   IoU `0.96120`, and 24/24 coherent faces. This is the leading training init.
2. `n3a_fullgrid_up_dual25_div8` — safest visibly strong N3a hybrid on all 96:
   face `0.05787`, landmark `0.00732`, bbox `0.96643`, 96/96 faces. On 24:
   face MAE `0.05830`, landmark shift `0.00902`, bbox IoU `0.96873`, and 24/24
   faces. This is the conservative training fallback.
3. `n3a_fullgrid_up_core_ring_anchor` — still the most strongly validated
   active initialization (96/96 faces) and the reference training candidate,
   but the fixed target fallback is substantially cleaner on the diverse eight.
4. `n3a_core68_plus_zero_refpooled_div8` — canonical-strength face change
   (`0.08389`) but only a small 24-case landmark improvement (`0.02147`) and no
   bbox improvement (`0.94362`). It is dominated by both dual variants.
5. `nn7v2_lmkidw3_staged_up003_up1075_ba0` — safest modern visible route:
   face MAE `0.01597` against BA0, landmark shift `0.00148`, bbox IoU
   `0.99235`; coherent but much gentler than the N3a family.

Dual `0.50` is a useful high-activity boundary (`0.10386` face MAE) but loses
some of the fixed-fallback alignment advantage (`0.01572` landmarks, `0.95603`
bbox IoU). Adjacent-column review of the 24-case promotions confirms that
dual `0.35` preserves broad coherence while providing more visible change than
dual `0.25`; the latter remains preferable if training later amplifies drift.
