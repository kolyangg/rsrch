# BigCelebs dataset usage improvement plan

## Decision

Keep `BigCelebsTrain` and its current launcher unchanged as the control. Add an
opt-in, deterministic schedule for the sealed v2 release so target coverage,
identity weighting, reference selection, scale curriculum, and horizontal
flips are explicit and reproducible.

The current trainer already preserves one shuffled DataLoader iterator across
the 2,000-step validation boundaries. An uninterrupted batch-size-2 run sees
40,000 distinct targets by step 20,000. The problems to address are therefore
image-proportional identity weighting, weak two-view identities, uniformly
random references, and directionally incorrect captions after flips.

## Policy v1

Build an 80,000-row plan for 40,000 optimizer steps with these fixed rules:

- require at least four reference-eligible images with face side at least
  256px per identity;
- sample identities with weight `sqrt(min(target_count, 16))`;
- rotate targets within each identity/scale pool before repeating them;
- use only >=256px targets through step 6,000, then sample 80% >=256px and
  20% 192--255px targets;
- rank distinct >=256px references by ArcFace similarity to their normalized
  identity centroid, breaking ties by face size;
- draw from the best three references with temperature `0.05`, with no
  self-reference or silent fallback;
- never flip a target whose caption contains standalone `left` or `right`;
  otherwise encode a deterministic 50% flip in the plan;
- retain raw reference formatting in the first run. Test face-focused
  crop+resize separately after the sampling policy is evaluated.

The plan builder must consume the sealed manifest plus the existing curation
asset index and all three provenance-specific, read-only ArcFace embedding
caches. It routes legacy, EQR6, and Neb-incremental records through the
authoritative provenance in `final_assets.jsonl`; it never resolves overlapping
cache keys by arbitrary precedence. It writes a scored-image
sidecar, the schedule JSONL, a summary/manifest, and SHA-256 hashes. It must
fail on incomplete joins or embeddings, invalid pairs, insufficient reference
candidates, and source/hash mismatches. It never edits the sealed dataset.

Generate the offline artifacts on Neb under
`/home/niko/rsrch/dataset_publish/sampling_policies/big_celebs_v2_policy_v1`:

```bash
export POLICY_ROOT=/home/niko/rsrch/dataset_publish/sampling_policies/big_celebs_v2_policy_v1
mkdir -p "$POLICY_ROOT"

python tools/datasets/build_big_celebs_sampling_plan.py scores \
  --manifest /home/niko/rsrch/dataset_publish/releases/v2/filtered_ids3_adj.json \
  --expected-manifest-sha256 f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa \
  --asset-index /home/niko/rsrch/curation/combined_1024_final/v1/final_assets.jsonl \
  --legacy-embedding-db /home/niko/rsrch/curation/combined_1024_final/v1/legacy_original_metadata/identity_embeddings.sqlite3 \
  --eqr6-embedding-db /home/niko/rsrch/curation/combined_1024_final/v1/eqr6_identity_stage/identity_embeddings.sqlite3 \
  --neb-incremental-embedding-db /home/niko/rsrch/curation/combined_1024_final/v1/neb_incremental_identity_stage/identity_embeddings.sqlite3 \
  --output "$POLICY_ROOT/reference_scores.jsonl" \
  --output-manifest "$POLICY_ROOT/reference_scores_manifest.json"

python tools/datasets/build_big_celebs_sampling_plan.py schedule \
  --manifest /home/niko/rsrch/dataset_publish/releases/v2/filtered_ids3_adj.json \
  --expected-manifest-sha256 f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa \
  --scores "$POLICY_ROOT/reference_scores.jsonl" \
  --scores-manifest "$POLICY_ROOT/reference_scores_manifest.json" \
  --output "$POLICY_ROOT/train_40k_bs2.jsonl" \
  --output-manifest "$POLICY_ROOT/train_40k_bs2_manifest.json"
```

## Code shape and rollback

- Add an offline `build_big_celebs_sampling_plan.py` tool with separate
  `scores` and `schedule` commands so expensive score extraction is reusable.
- Add `BigCelebsScheduledTrain`, which retains all strict BigCelebs manifest
  checks and model-facing fields but reads target, reference, and flip choices
  from the pinned plan.
- Register it as `big_celebs_scheduled` and add a new config/launcher inheriting
  the current model, optimizer, eligible BA flags, and fixed full-96 validation.
- Make training shuffle an explicit opt-in config value; old configurations
  continue to default to `true`, while the scheduled dataset requires `false`.
- Support an explicit schedule-row offset. On recovery it must equal completed
  optimizer steps multiplied by global batch size, preventing early samples
  from being replayed after a restart.

Returning to the old policy requires only selecting the existing `big_celebs`
dataset/config/launcher; no compatibility switch or data migration is needed.

## Acceptance and experiment gates

Before training, require config composition, shell syntax, import/compile,
deterministic plan regeneration, exact plan/source hashes, a two-worker loader
smoke test, and 64 decoded scheduled target/reference pairs. The 80,000 rows
must contain zero self/cross-identity pairs, all references must be >=256px,
and scale/flip/coverage summaries must match the declared policy.

Use the existing run as the uniform-sampling control. Evaluate the new policy
on the unchanged full-96 panel every 2,000 steps, with decision gates at 8k,
12k, and 20k. Compare matched visuals, identity similarity, text similarity,
TOPIQ-Face p10, and coverage. Do not continue merely because loss declines if
identity or visual quality has already plateaued.

Implementation and smoke verification do not authorize a commit or a new
training launch; those remain separate user decisions.

## Prepared batch-4 variant

The opt-in batch-4 variant uses a separately generated 160,000-row plan, so
the 6,000-step scale warmup and 40,000 optimizer-step budget retain their
original meaning. The Neb artifact is `train_40k_bs4.jsonl` with SHA-256
`ff373204841cec5d06014faa7d3932442bfc256adf7fa02c63ca1e010ed2cbb8`.
It has zero self/cross-identity rows, 141,685 unique targets, and 151,869
unique ordered pairs. Byte-identical regeneration, the 64-pair decode
preflight, resolved Hydra invariants, and a real two-worker batch of four all
pass.

Use config `big_celebs_scheduled_rhca_40k_bs4` and Neb launcher
`launchers/neb/start_rhca_big_celebs_scheduled_sameid_40k_bs4.sh`. Its default
Comet run name is `rhca_big_celebs_scheduled_v1_40k_bs4_full96_r1`. It is
prepared but not launched; a future launch must verify peak GPU allocation
over several optimizer batches before being left unattended.
