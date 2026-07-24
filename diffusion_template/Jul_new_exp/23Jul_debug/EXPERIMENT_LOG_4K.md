# NN3a 4k experiment log

Protocol: 4,000 optimizer steps; checkpoints and four-prompt canonical
validation every 500 steps.

Step-zero is a required validation stage for every arm, not an optional
baseline. Each run must contain four canonical step-zero images, four
PhotoMaker controls, step-zero similarity/text/alignment metrics, and their
uploads under the original training Comet experiment key. For a run already
training, derive the zero-functional NN3a state from its first stable
checkpoint and backfill step zero without interrupting the trainer. A run is
not valid and the queue must not advance if the final Comet-unity audit cannot
verify step zero together with steps 500 through 4000.

| arm | dataset | recipe | state | run / result |
|---|---|---|---|---|
| `L4_O1_oneid_projection_alt` | OneID distinct pairs | projection split + alternating | complete | nine stages, report upload, and Comet-unity audit PASS |
| `L4_C1_large_projection_alt` | CosmicLarge single ID | projection split + alternating | complete | nine stages, report upload, and Comet-unity audit PASS |
| `L4_O2_oneid_projection_blend20` | OneID distinct pairs | projection split + 80/20 anchored | interrupted by user | stopped after step-1000 checkpoint; artifacts preserved |
| `L4_C2_large_projection_blend20` | CosmicLarge single ID | projection split + 80/20 anchored | running | started 2026-07-24 08:30 UTC |
| `L4_O3_oneid_ref_value_blend20` | OneID distinct pairs | reference V only + 80/20 anchored | queued | pending |
| `L4_C3_large_ref_value_blend20` | CosmicLarge single ID | reference V only + 80/20 anchored | queued | pending |
| `L4_OF1_oneid_full18_projection_alt` | OneID full set minus held-out `51.jpg` | priority-1 projection split + alternating | running | started 2026-07-24 08:58 UTC as replacement for subset arm |

Every result in this file is invalid unless its run-local pairing audit and
Comet-unity audit both pass.

The full-OneID priority insertion uses all 18 available training images after
excluding `51.jpg`, the sole validation reference. Its preflight sampled eight
references per target and passed with zero same-image violations.

## Full continuation

The live six-run queue is followed by ten more paired waves, in this order:
projection-split schedule matching; active-up plus blended loss; active-up
schedule matching; all-scope noise damping; all-scope blended loss; matched
PhotoMaker teacher; exact control; active-up control; up1-only control; and
staged up1/up0. Each recipe runs once with leak-free OneID distinct pairing
and once with CosmicLarge's separately sampled reference. See
`4K_EXPERIMENT_PLAN_AND_SCHEDULE.md` for exact run IDs and priorities.

E15/E16 are excluded because their 600-step runs used a pixel-identical target
as reference. Their mechanisms are represented by corrected, distinct-pair
arms; leakage itself is not a useful 4k training condition.

## First live inspection: projection-split OneID step 500

The four step-500 images are much sharper than step zero, but the result is
not yet promotable. Reading develops a mismatched/deformed eye behind the
glasses; Rushing is broadly coherent but still asymmetric; Skiing retains a
difficult double-goggle/glasses composition; and Drumming has a visibly
misaligned eye. Outer-scene preservation is strong. This is an early
checkpoint only; the full identity, geometry, and CLIP trajectory will decide
whether the branch stabilizes or drifts through steps 1000–4000.

Metrics agree that sharper/aligned does not mean better identity: median
reference similarity falls `0.3434 → 0.2647`, and gain over PhotoMaker falls
`+0.0689 → -0.0453`. Landmark displacement improves `0.0323 → 0.0183` and
CLIP prompt cosine rises slightly `0.2481 → 0.2526`. Thus the first 500 steps
trade away held-out identity while improving geometry and prompt adherence.

The paired CosmicLarge step-500 images are substantially worse: all four
prompts exhibit severe horizontally displaced or duplicated eyes, nose, and
mouth. The failure is clearest in Reading and Rushing and remains obvious in
Kickboxing and Dancing. This reproduces the 600-step Cosmic-loader failure
well before the long-run endpoint, so later checkpoints are being retained to
test recovery rather than treating step 500 as a candidate.

CosmicLarge's scalar trajectory is another warning against metric-only
selection. Reference similarity increases `0.2549 → 0.3501` and landmark
displacement improves `0.0318 → 0.0195`, even though every step-500 face is
visibly malformed. Reference gain remains strongly below PhotoMaker
(`-0.2673 → -0.1852`), while CLIP prompt cosine falls `0.2464 → 0.2349`.
Visual anatomy therefore remains the primary promotion gate.

OneID step 1000 shows a real geometric recovery relative to step 500:
Reading's mismatched eye is largely corrected, Rushing is coherent, Skiing's
inner face is aligned inside the expected goggles-plus-glasses composition,
and Drumming no longer has the displaced eye. The Drumming face still looks
locally composited and the expressions are somewhat rigid, so this is not yet
a promotion. Identity metrics are being backfilled before deciding whether
the recovery preserves the step-zero branched-attention identity or simply
moves toward PhotoMaker.

The step-1000 metrics indicate mostly PhotoMaker-directed recovery, not
identity recovery. Reference similarity is only `0.2665` versus `0.2647` at
step 500 and remains below the `0.3434` step-zero value; gain versus
PhotoMaker remains negative at `-0.0494`. Landmark displacement improves
further to `0.0132`, but face similarity to the PhotoMaker output rises to
`0.6257`. CLIP prompt cosine reaches `0.2564`. This checkpoint is visually
cleaner but less identity-distinct than step zero.

At OneID step 1500, Reading and Drumming regress visually with renewed eye
asymmetry/warping; Rushing and Skiing remain broadly coherent. Scalars improve
slightly over step 1000 (`ref sim 0.2717`, gain vs PM `-0.0142`, landmark
`0.0131`, CLIP `0.2570`) and similarity to PhotoMaker falls from `0.6257` to
`0.5556`, but step zero still has the strongest held-out identity and step
1000 is cleaner anatomically. No trained checkpoint is promotable yet.

CosmicLarge step 1000 remains severely malformed in all four prompts despite
reference similarity increasing to `0.3579`, gain improving only to
`-0.1760`, and CLIP recovering to `0.2501`. Reading and Dancing retain strong
horizontal face smearing; Rushing and Kickboxing retain displaced eyes and
mouth. The scalar/visual mismatch persists.
