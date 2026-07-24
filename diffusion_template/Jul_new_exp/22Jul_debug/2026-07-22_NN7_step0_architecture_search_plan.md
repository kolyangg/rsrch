# NN7 step-zero architecture search: execution plan

**Status:** awaiting approval; no GPU experiments have been started  
**Environment:** `photomaker_NS`  
**GPU:** one NVIDIA H100 80 GB  
**Allowed write scope:** `Jul_new_exp/22Jul_debug/` only

## Instruction summary

The task is a step-zero, untrained architecture search for a branched-attention
configuration between two known extremes:

- N3a has enough reference ownership to change identity, but its unregistered
  full spatial takeover breaks pose, anatomy, boundaries, and occluders.
- NN5/NN6/normal NN7 variants preserve PhotoMaker geometry but are too weak or
  move along generic appearance directions rather than reference identity.

The architectural invariant to preserve is:

```text
target-coordinate face Q -> spatial reference-face K/V
```

The search should repair N3a one axis at a time: disable branched CA, normalize
the reference ROI, restrict ownership to up blocks, restore target fallback,
protect the face boundary, anchor PhotoMaker epsilon outside the core, and vary
authority and start time.

The first screen uses the first four fixed `manual_val` RealVis records, the
same prompts/seeds/bboxes, BF16, 20 denoising steps, PhotoMaker from step 4, and
BA from step 6 by default. Every experiment must save its exact recipe,
architecture signature, per-sample and aggregate metrics, images, face crops,
difference maps, and contact sheet. Failed experiments remain in the registry.

Screening requires all three kinds of evidence:

1. visible face activity: median face MAE versus PM at least `0.012`;
2. provisional safety: 4/4 detected faces, outside-face MAE at most `0.015`,
   landmark displacement at most `0.08`, bbox IoU at least `0.60`, and face MAE
   at most `0.12`;
3. identity direction: median reference-similarity gain at least `0.003`,
   positive on at least 3/4 samples, plus mandatory visual inspection.

At most three candidates may be promoted. Promotion proceeds through a 50-step
repeat, a fixed wrong-reference test, and then the deterministic 24-case
RealVis causal matrix. No training configuration should be created during this
search.

## Notebook preflight finding

The sample notebook has valid Python syntax but calls `json.dumps` in the
validation-data cell without importing `json`. The first approved change will
add that import or place the same correction in a local runner. No source,
training entry point, or repository config will be modified.

## Plan of attack

### 1. Make the harness reliable

- Apply the minimal `json` import fix inside this directory.
- Add an in-directory command-line runner if needed so experiments can be
  selected, sharded, resumed, and logged without manually editing notebook
  state.
- Preserve the notebook's core-code SHA-256 guard and unique output folders.
- Run a small harness smoke test, verify deterministic PM0 hashes, validate the
  installed processor signature, and inspect one generated bundle before the
  larger search.
- Make registry/leaderboard updates concurrency-safe or merge worker-local
  registries after each batch.

### 2. Establish controls and the initial frontier

Run the quick ladder on all four samples:

- exact N3a and N3a without branched CA;
- full-grid versus normalized ROI;
- reference-only up-block ownership with protected output;
- repaired core-ring, dual-75, and confidence-residual candidates;
- NN7a_init v1, NN7a_init-v2 default, and strong-v2 controls.

Start with two parallel GPU workers. Record peak memory and utilization, then
increase to three workers only if memory headroom and throughput are healthy.
Each worker receives disjoint experiment IDs and unique artifact directories.

### 3. Review and choose one-axis ablations

After each batch:

- rebuild the leaderboard from completed and failed runs;
- inspect every contact sheet and the four face crops, not only the score;
- record identity gain, face MAE, landmark movement, bbox IoU, outside/ring
  MAE, detection count, and visible failure types;
- write the decision and next single-axis change to a progress log in this
  directory;
- report the key result to the user.

The next batch will depend on the observed failure mode:

- identity-positive but unsafe: reduce core/dual ownership, start BA later, or
  restrict the layer scope;
- safe but too PM-like: increase only one of ownership, core size, or BA
  duration;
- active but not identity-directed: compare ROI resolution/full-grid and
  target fallback modes rather than merely increasing scale;
- boundary or attachment damage: increase the target-owned ring/erosion;
- inconsistent or invalid metrics: repair the harness before interpreting the
  architecture.

Primary sweep axes are core ratio `0.50/0.60/0.68/0.75/0.82`, dual reference
weight `0.35/0.50/0.65/0.75/0.85`, ROI grid `6/8/12/16`, and BA start at
`20/30/40/50%` of denoising. These will be staged adaptively rather than run as
an undifferentiated Cartesian grid.

### 4. Promote no more than three candidates

For candidates that are visibly active, reference-improving, and visually
safe:

1. rerun at the exact 50-step schedule on the same four records;
2. add a fixed wrong-reference A-to-B intervention with target latent, prompt,
   seed, bbox, and PhotoMaker identity held fixed;
3. require movement toward B rather than only movement away from PM/A;
4. run the deterministic 24-case RealVis matrix;
5. inspect difficult pose, glasses, hats, hair, hands, occlusion, seams, neck
   attachment, body, and background behavior.

### 5. Deliver a reproducible result

Maintain under `22Jul_debug/`:

- `experiments/registry.jsonl` and `experiments/leaderboard.csv`;
- immutable per-run recipes, metrics, images, and contact sheets;
- a chronological research/progress log;
- a final Markdown report naming the top candidates, rejected alternatives,
  causal evidence, visual failures, and recommended next architecture.

## Stop conditions

Stop the step-zero search and report when one of these occurs:

- one to three candidates pass the four-case screen and promotion tests;
- the staged search finds no stable positive reference direction, indicating
  that registration or explicit identity/spatial factorization is required;
- a systematic harness or metric problem prevents valid comparisons.

Training, production YAML creation, edits outside `22Jul_debug/`, and Git pushes
remain out of scope unless separately authorized.
