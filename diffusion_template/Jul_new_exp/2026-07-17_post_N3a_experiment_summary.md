# Post-N3a experiment summary and architecture diversion

Date: 17 July 2026

## Executive summary

The post-N3a work produced two valuable findings:

1. The original full spatial branch is strong enough to change identity, but
   ordinary denoising supervision can train it toward reference appearance,
   color, pose, hair, props, and dataset-average faces rather than identity.
2. The later target-only residual family is much safer for pose and rendering,
   but progressively ceased to be the original branched-attention mechanism and
   eventually became too weak to overcome PhotoMaker identity.

The later code is preserved in `main_clean_exp`. It is not discarded; it is
separated so future NN experiments can begin from an explicit full-BA contract.

## Phase 1: N3a and early full-spatial stabilization

N3a retained both processor classes at all SDXL attention sites:

- 70 `BranchedAttnProcessor` self-attention processors;
- 70 `BranchedCrossAttnProcessor` cross-attention processors;
- one doubled `[target, reference]` U-Net call at active BA denoising steps.

N3a added optimizer hygiene and target/noise LR damping, but its identity score
fell sharply after training. Step zero was substantially better than trained
checkpoints. The key lesson was that denoising MSE does not guarantee movement
toward reference identity.

N4-N9 explored LR, weight decay, crop jitter, blended masking, and short
checkpoint cadence. These runs kept the full spatial BA architecture but did
not fully solve identity direction or reference-content transfer.

## Phase 2: N10-N24 full-BA training lessons

This phase still used the doubled spatial mechanism.

- N10 showed that adding more generally trainable capacity did not solve drift.
- N11 froze branched cross-attention weights while keeping
  `BranchedCrossAttnProcessor` active in the forward path. This produced clean,
  monotonic gains and removed much of the cross-attention drift.
- N13 added direct decoded identity supervision and improved hard poses.
- N14/N17 combined frozen CA weights with decoded identity loss. N17 became the
  best aggregate result in that family, although the gain was uneven and later
  checkpoints over-strengthened some faces.
- N15 confirmed that self-attention-only training was a strong independent
  lever.
- N16 confirmed that broadly training branched CA together with identity loss
  was worse.
- N18-N24 explored loss strength, runtime knobs, CA modes, and interpolation.

The most transferable lessons are:

- keep both processor mechanisms in the forward path;
- train spatial self-attention first;
- keep branched CA frozen until there is evidence that changing it is required;
- use a direct identity objective only at timesteps where decoded identity is
  meaningful;
- save frequently and select checkpoints visually, not only by final metrics;
- broad face boxes cause collisions with goggles, hair, hands, hats, and props.

## Phase 3: N25-N33 compact target-residual architecture

N25-N30 introduced staged and residual alternatives. The decisive diversion was
N28:

- spatial reference latents were disabled;
- `ba_sa_mode: standard` retained original `attn1` processors, so
  `BranchedAttnProcessor` was no longer active;
- `BranchedCrossAttnProcessor` changed from split target/reference CA into an
  additive target-face residual driven by compact identity tokens;
- a separate PhotoMaker prediction became the protected absolute baseline.

N29/N33 were stable but close to PhotoMaker. N31 made the residual visibly
important but learned desaturation/expression shortcuts. N32 produced a clean,
active residual with eight face-patch tokens, but identity gains were not
monotonic.

These runs are useful evidence for safe residual arbitration, but they are not
implementations of the original full branched-attention plan.

## Phase 4: N34-N38 restricted identity-owner residuals

N34 and N35 were problematic and are retained as historical attempts rather
than recommended baselines.

N36-N38 restricted the compact residual to 16 of 70 cross-attention sites,
equivalent to approximately 11 unit-gate sites. PhotoMaker remained the
external absolute prediction owner and retained most identity-conditioned
context. The branch weights trained and validation loaded them, but face
movement remained about half of N32 and metrics stayed effectively flat.

The main lesson is that richer identity memory cannot help if the downstream
route has too little authority. The label “identity owner” did not match the
implemented arbitration.

## Why the code moved to `main_clean_exp`

The compact residual line changed all of the following simultaneously over
time:

- identity representation;
- presence or absence of the spatial reference stream;
- self-attention processor type;
- cross-attention semantics;
- layer count and gate initialization;
- PhotoMaker context;
- CFG/epsilon composition;
- loss and negative-reference objectives;
- checkpoint and mixed-precision plumbing.

Continuing to call every version “branched attention” obscured which mechanism
was actually under test. `main_clean` now means full spatial BA; the later
residual line remains available in `main_clean_exp`.

## What should carry into NN1

Carry:

- the N3a doubled target/reference topology;
- both processor classes at all 70 SA/CA sites;
- N11’s finding that CA can remain active but frozen;
- N13/N14’s identity-supervision result;
- frequent validation and fixed same-seed PhotoMaker comparisons;
- hard runtime assertions for processor counts, masks, and checkpoint loading.

Do not carry initially:

- `ba_sa_mode: standard`;
- target-only compact residuals;
- N34-N38 layer allowlists;
- PhotoMaker-context attenuation;
- post-CFG residual composition;
- simultaneous memory, gate, layer, and objective changes;
- N36-N38 checkpoints.

