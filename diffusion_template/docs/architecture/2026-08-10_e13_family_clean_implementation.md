# Clean E13-family implementation and Serv runbook

- **Date:** 10 August 2026
- **Branch:** `clean`
- **Clean base:** `2157eada14824d14019e80f9416e6d736c837306`
  (`code clean-up - restore 1 ref only`, 2 June 2026 21:58 BST)
- **Supported recipes:** E13, BC_E13, and CL14
- **State:** implemented, locally audited, committed in the branch history, and
  deliberately not launched

## Outcome

The branch keeps the June single-reference model as its base and adds one
small, fail-closed E13-family contract. E13 and BC_E13 share exactly the same
model, optimizer, pipeline, schedule, and validation behavior; their only
difference is the training dataset. CL14 uses the same contract with two named
training-side changes: the inherited CL9 Cosmic dataset policy and a two-latent-
cell target-mask feather. CL14 does not change the inference mask or denoising
equation.

The implementation intentionally separates four concerns:

```text
thin leaf config
├── shared hard-v1 E13 model/checkpoint contract
├── sealed CL14 inference/validation path
├── one dataset policy (Large, BigCelebs, or Cosmic)
└── explicit training-efficiency profile
```

Later E14-E24 mechanisms were not copied. In particular, there is no residual,
anchored, or query-adaptive BA; no identity cross-attention; no ArcFace
auxiliary; no reference dropout; no ROI warp; and no branch-output adapter.

## Provenance

The implementation was reconstructed path-by-path rather than by merging
`test`. The source hierarchy was:

| Purpose | Immutable source |
|---|---|
| June clean base | `2157eada14824d14019e80f9416e6d736c837306` |
| strict E13 substrate | `e860f9ed4d021226575845ae24a9fda1e5a3fa58` |
| effective adapter scopes/rank | `dd65dec312271610e46dd507ee010a5927b8cbda` |
| joint groups/validation shadow | `8b8b9abd726df111ce725b6c283869c3dd19e6a0` |
| successful E13 runtime | `ebf1ac8295f363adb0055cd74db1a96c2ff03a35` |
| BC_E13 runtime | `ad194a026ab701dd979712d415c487dd536a4645` |
| exact CL14 overlay | `c04970f342a186d1092f07f9a08d7d8a797383e8+cl12-cl14-snapshot-v1-20260809` |

Sealed CL14 evidence:

- source-manifest SHA-256:
  `d43fa65815aa4fc4c106f6ed3e939b5dee690f2a2927b43a022a8e6025ccc294`;
- resolved saved-config SHA-256:
  `642cdcb4acd2b4fcf0ef9fd5fadaa5fb5a117c092b1e07394b8e9c2fd406b2c4`;
- 24k weights SHA-256:
  `0de10ec611c8a5e55e0b362ea90fa348fe686ca5d949f762752bc1add7992ed9`;
- immutable Comet experiment key:
  `6fe0028be92242c38056b3d36665fdd6`.

Other historical records:

| Run | Comet key | Known 24k checkpoint SHA-256 |
|---|---|---|
| E13 r4 | `1cc0a02371094b24a6a02a4cc649f10c` | `4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab` |
| BC_E13 r1 | `c138db7c41ae435c8a7560f40cf5f58d` | `99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e` |
| CL14 r1 | `6fe0028be92242c38056b3d36665fdd6` | `0de10ec611c8a5e55e0b362ea90fa348fe686ca5d949f762752bc1add7992ed9` |

BigCelebs sealed-v2 manifest SHA-256 is
`f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa`.
The historical E13 Large Dataset manifest SHA-256 is
`0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50`.

## Supported recipe matrix

| Field | E13 | BC_E13 | CL14 |
|---|---|---|---|
| leaf config | `E13_large_ds_joint_shadow_sa128_24k` | `BC_E13_big_celebs_joint_shadow_sa128_24k` | `CL14_cosmic_joint_shadow_sa128_softmask_24k` |
| dataset | `LargeDatasetTrain` | `BigCelebsTrain` | `CosmicLargeAdaptedTrain` |
| model architecture | hard-replace-v1 | identical | identical |
| target training-mask feather | 0 | 0 | 2 |
| inference mask | hard historical mask | identical | identical |
| fixed-96 active bbox cache | E13/BC `4db6344d...` | identical | CL10-CL14 `b33cf026...` |
| optimizer ownership | 840/700/700 | identical | identical |
| schedule/validation/runtime | shared | shared | shared |

All leaves resolve to 24,000 optimizer steps: batch size 2, 2,000-step
epochs, 12 epochs, checkpoint and fixed-96 validation at step 0/every 2,000
steps, and `WarmupHoldCosineLR(20, 14000, 24000, 0.1)`. Validation uses DDIM
50, CFG 5, PhotoMaker from step 10, and BA from step 15. Eligible BA invariants
remain `pose_adapt_ratio=0` and `ca_mixing_for_face=false`.

## Change ledger

Every new logical code block is adjacent to a dated `E13C-*` marker. Existing
`AICODE-NOTE:` anchors were retained where they document critical cache,
routing, dataset, or deferred-scoring invariants.

| Marker | Main files | What changed | Why it exists |
|---|---|---|---|
| `E13C-CORE-01` | `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` | strict hard-v1 installation and route validation | prevents silent fallback to a plain processor or wrong architecture |
| `E13C-CORE-02` | `e13_contract.py`, `branched_runtime.py` | rank-128 hard branch while preserving generic rank 32 in the historical manifest | reproduces E13 capacity and old checkpoint schema |
| `E13C-CORE-03` | `e13_contract.py`, `lora2.py`, `train.py` | one 840/700/700 allowlist and optimizer grouping | proves every trained tensor has one owner and no base U-Net tensor trains |
| `E13C-CORE-04` | `e13_contract.py`, `lora2.py` | complete schema-v2 state, trainable shapes/dtypes, semantic processor hash, strict load | saves all three trained paths and rejects incompatible checkpoints |
| `E13C-CORE-05` | `base_trainer.py` | restore 700 pretrained PhotoMaker-default tensors in validation and full-copy 70 processor states | reproduces the E13 shadow-coadapter validation mechanism without changing saved training weights |
| `E13C-CORE-06` | `lora2.py` | CL14-only `1/3`, `2/3`, `1` inward target training-mask ramp | teaches a gradual training handover while leaving reference and inference masks unchanged |
| `E13C-PIPE-01` | `photomaker_branched_clean.py`, `br_pipeline_helpers.py` | one spatial reference for a 12-prompt identity batch | prevents accidental repeated/multiple spatial lanes |
| `E13C-PIPE-02` | pipeline files, `branched_runtime.py` | sealed reference latent/mask/noise preparation and denoising schedule | preserves CL14 RNG consumption and generation equations |
| `E13C-PIPE-03` | leaf configs, fixed-96 protocol files, validators | pin the canonical manual map and run-family-specific automatic caches | prevents face-detector or bbox-map drift from changing validation generations |
| `E13C-DATA-01` | `large_dataset.py`, dataset registry | distinct same-ID target/reference loader | exact E13 data relationship |
| `E13C-DATA-02` | `big_celebs.py`, dataset registry | sealed field/bbox/trigger/min-face validation | makes BC_E13 a dataset-only transfer and fails before training on a bad release |
| `E13C-DATA-03` | `cosmic_large_adapted.py`, `reference_policy.py` | no reference flip, pose-first prompt, one trigger, 50-word cap | avoids independent identity mirroring and losing useful pose/background tokens |
| `E13C-DATA-04` | `reference_frame.py`, Cosmic loader | 1024 target-frame reference with 6%-30% face-area and 0.15 position jitter | fixes the 256-pixel tight-crop scale mismatch without the fixed-position copy shortcut |
| `E13C-PERF-01` | `lora2.py`, `lora2_helpers.py` | batched frozen text/PhotoMaker/VAE conditioning and cache-off diverse-pair policy | removes repeated per-sample frozen encoder work |
| `E13C-PERF-02` | model/runtime/trainer | skip unreachable text-only encoding/host sync, per-forward mask cache, no debug outputs, no zero-touch, sparse grad telemetry | removes output-irrelevant training work while preserving June defaults outside the named family |
| `E13C-PERF-03` | active/Serv launchers, model preflight | async CUDA and exact ORT-GPU 1.20.1 fail-closed checks | prevents the prior silent 5-7 s/step CPU/blocking path |
| `E13C-PERF-04` | trainer, face-quality module, Comet tools | stage generated images and score only after training succeeds | PyIQA cannot delay, perturb, or invalidate optimizer/checkpoint work |
| `E13C-CFG-01` | shared config, scheduler | one architecture/schedule/efficiency contract | makes cross-dataset equivalence reviewable |
| `E13C-CFG-02` | three leaves, launcher, validators | only dataset and CL14 mask delta live in leaf configs; manifest hashes required | rejects ad-hoc drift before GPU startup |
| `E13C-DOC-01` | this file, handoff, tools index | provenance, evidence, limitations, and runbook | lets another agent audit and run the branch without reconstructing history |

## 1. Core model and checkpoint implementation

`e13_contract.py` is the only E13-specific ownership switchboard. It freezes
everything, then enables exactly:

| Optimizer group | Tensors | Parameters | LR |
|---|---:|---:|---:|
| rank-128 hard spatial BA | 840 | 127,795,200 | `1e-4` |
| rank-32 generic effective adapter | 700 | 30,474,240 | `1e-4` |
| rank-64 PhotoMaker-default effective adapter | 700 | 60,948,480 | `1e-4` |
| **total** | **2,240** | **219,217,920** | — |

The branch rank is stored as `hard_v1_extensions.lora_rank=128`; the legacy
top-level `branched_attn_lora_rank=32` remains the generic rank exactly as in
the sealed schema. The loader compares the output-affecting E13 projection,
semantic processor-name hash, exact trainable names, shapes, and dtypes. It
allows unrelated later manifest fields so genuine E13/BC_E13/CL14 schema-v2
checkpoints remain loadable, but rejects route, scope, timing, rank, processor,
name, or shape drift.

The current full construction produced semantic processor hash
`1c34710dd22f0108d015d1ac0d62c1d4a23ae271853fdfdb26c58d05ac717e3d`.

## 2. Pipeline and CL14 generation parity

The two pipeline files are the sealed CL14 files plus only a leading dated
audit marker. After removing that marker and trailing blank lines, SHA-256 is:

| File | Sealed canonical SHA-256 |
|---|---|
| `src/pipelines/br_pipeline_helpers.py` | `4c1516d3536a85c028580f601b61773df55c49d6b16dfac9d93c997102be5c95` |
| `src/pipelines/photomaker_branched_clean.py` | `85e1b3a2da90ba4a007f8bda895c722c7a21c8e5a519b86881626ed665e9071c` |

The complete AST source for `two_branch_predict`, canonicalized only for
historical trailing spaces, hashes to
`9145856534abe92a6f48e9328dad5a1692ff65f27a51ac3554f9b4db82ad3689`.
`tools/verify_cl14_generation_parity.py` checks all three values before every
server launch.

Generation-box routing is also pinned because it is an input to the denoising
mask. Both families use the 96-entry canonical manual protocol
`a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614`,
whose `Reading pa_jensen.png` entry is intentionally `force_manual`. E13 and
BC_E13 load the historical 96-entry automatic cache
`4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099`;
CL14 loads the later CL10-CL14 cache
`b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`.
Thus Jensen uses its manual record and the other 95 routes use the selected
cache. The files live under `dataset_full/val_dataset/protocols/`, and both
validators reject a missing, renamed, reserialized, or regenerated map before
model loading. The parity gate also seals the prompt file, class map, reference
boxes, legacy identity embeddings, and all 12 reference images byte-for-byte.

The attention processor is intentionally smaller than the sealed post-CL14
switchboard. Under CL14's selected flags, a deterministic fixture with nonzero
branch-LoRA weights produced `torch.equal=True` against the sealed processor
source. The only active addition is per-forward mask memoization; the cached
tensor is the same resized mask object and cannot cross samples or timesteps.

These facts establish source-level pipeline parity and bit-exact selected-
processor parity. A fresh historical-checkpoint/full-96 RGB replay was **not**
run locally: the available machine is not the pinned A100/ORT-GPU runtime, the
historical weight artifact is not present locally, and the user explicitly
asked not to launch the Serv job yet. Consequently this document does not claim
a newly measured pixel replay. The first Serv run must retain the generated
step-0 panel and compare it with the immutable CL14 export before promotion.

CL14's mask feather cannot affect inference: it is read only by training
`_bbox_to_mask`; the reference mask and the pipeline's inference mask are
unchanged.

## 3. Dataset improvements

Dataset changes finish before model input and do not appear in model or
pipeline conditionals.

### E13 Large Dataset

The loader samples a distinct image from the same identity, propagates target
horizontal flips to its bbox, never independently flips the reference, and
emits the final reference image, bbox, identity, path, and cache key.

### BC_E13 BigCelebs

BC_E13 inherits E13 and changes only `train_dataset_name`. The loader accepts
the sealed manifest schema (`new_face_crop`, `text`), requires a minimum
192-pixel face side, exactly one `img` trigger, and at least two images per
identity. The launcher hard-pins the sealed-v2 manifest hash.

### CL14 Cosmic

The fixes are separate and explicit:

1. Reference geometry: the 256-pixel crop is composed into a 1024 target frame
   before the VAE; face area is sampled from 6%-30% and position jitter is 0.15.
2. Reference hygiene: independent reference mirroring is disabled.
3. Prompt hygiene: pose-first captions preserve one lowercase `img` trigger and
   cap at 50 words so pose/background content normally survives tokenization.
4. Cache correctness: target-dependent scale/position framing is encoded in
   the reference cache key.

The launcher requires the exact Cosmic manifest hash to be recorded in `.env`.
There is no known hash hard-coded because this repository snapshot did not
contain the data artifact; the operator must take it from the sealed CL14 data
package rather than substituting a similarly named manifest.

## 4. Training-efficiency improvements

Efficiency is an explicit profile, not an architecture or dataset claim:

| Switch | Selected value | Work avoided | Numerical status |
|---|---:|---|---|
| `conditioning_cache_enabled` | false | ineffective diverse-pair cache bookkeeping | output-neutral |
| `batched_conditioning_preparation` | true | per-sample frozen text/ID/VAE calls | historical E13-family path; semantically equivalent, not bit-identical to unbatched bf16 GEMMs |
| `skip_unused_text_conditioning` | true | unreachable text-only encode and `timestep.item()` sync | output-neutral because BA trains at every timestep |
| `cache_prepared_masks` | true | repeated mask resize in one doubled forward | exact tensor reuse |
| `compute_branch_debug_outputs` | false | unused post-merge diagnostic tensors | output-neutral |
| `post_backward_parameter_touch` | false | full zero-valued trainable scan | cannot affect optimizer values |
| `grad_norm_log_only` | true | norm reductions outside logging steps | telemetry-only |
| train workers | 2 | DataLoader stalls | same historical setting; persistent workers remain off |

Historical evidence measured roughly 5 s/step falling to roughly 0.9 s/step
after batched conditioning. One bf16 step differed by about 0.074% in loss from
the scalar implementation, so it is deliberately described as the exact run
setting—not as bit-identical to the slower path.

Production also requires unset/zero `CUDA_LAUNCH_BLOCKING`, ONNX Runtime GPU
1.20.1, an available and actually loaded CUDA provider in every InsightFace
session, and single-process Accelerate. Failure is fatal before training.

Face-quality generation remains step 0/every 2,000; only PyIQA scoring moves
after successful training. The finalizer checks the immutable Comet key, all 13
expected steps, 96 images per step, and per-image assets.

## 5. Concision review

The model was not copied wholesale from the 2,335-line sealed experimental
switchboard:

| File | Clean branch lines | Sealed CL14 lines |
|---|---:|---:|
| `lora2.py` | 953 | 2,335 |
| `lora2_helpers.py` | 505 | 949 |
| `branched_runtime.py` | 680 | 1,214 |
| `attn_processor_cleanest.py` | 779 | 931 |
| new focused `e13_contract.py` | 311 | not separate |

The pipeline is deliberately not shortened because exact CL14 source parity is
more important than cosmetic cleanup: its two files are sealed content plus
one audit comment. Rich dataset and Comet/face-quality tools remain outside the
model and do not enlarge its architecture switchboard.

Retained standalone tooling includes immutable-key Comet retrieval/export,
deferred and historical face-quality scoring, the three dataset preflights,
face/body alignment measurement, the research-report skill and renderer, and
the hash-verifying Dropbox uploader. `TOOLS.md` lists only utilities actually
present on this branch; later architecture evaluators and Serv runtime debris
were intentionally not copied.

## 6. Verification performed

| Gate | Result |
|---|---|
| branch/base | clean worktree created directly from exact 2 June SHA; no merge from `test` |
| changed Python compilation/import | passed |
| launcher shell syntax | passed |
| Hydra composition | all three recipes passed `validate_e13_family_config.py` |
| ownership | full SDXL CPU construction passed exact 840/700/700 and 2,240/219,217,920 counts |
| manifest | generic rank 32, hard rank 128, 2,240 names, semantic hash present |
| schema-v2 | save/load round trip passed; wrong semantic processor hash rejected |
| pipeline | both canonical sealed file hashes passed |
| denoising | sealed `two_branch_predict` AST hash passed |
| fixed-96 inputs | prompt/classes/reference-box/embedding/12-image hashes passed; canonical manual and separate active-cache hashes passed; 96 records and Jensen override verified |
| processor | deterministic nonzero-LoRA hard-v1 fixture was bit-exact to sealed CL14 source |
| mask | E13/BC binary; CL14 rings exactly `1/3`, `2/3`, `1` |
| datasets | distinct Large/BigCelebs refs, strict BigCelebs, Cosmic prompt and target-frame scale ratio 1.0 passed |
| deferred scoring | staged two images and immutable experiment key without importing/running PyIQA |
| Comet contract | logger writes `saved/<run>/comet_experiment.json`; launcher requires it after training |
| diff hygiene | focused `git diff --check` passed before commit |

A repository-wide `compileall` is not a valid clean signal on this June base:
the pre-existing tracked file `src/model/attn_procs/attn_processor.py` starts
with invalid `phaimport torch` in both the base commit and this branch. It was
not changed because it is outside this implementation and is not the selected
processor. Focused compilation covers every changed Python file.

Not performed yet:

- no A100 benchmark or training job;
- no historical weight download/checkpoint load;
- no fixed-96 RGB replay or metric recomputation;
- no push to a remote branch.

## 7. Serv preparation and launch instructions

Do not run these commands until the branch is intentionally transferred to the
Serv checkout and a job launch is approved. Never use Neb; it is unavailable.

### 7.1 Pin the source

After this local branch is pushed or otherwise transferred, use a dedicated
Serv checkout/worktree and pin the final commit shown by local `git log`:

```bash
git fetch origin clean
git switch clean
git pull --ff-only
git rev-parse HEAD
git status --short
```

The status must be empty. Record that commit in the run notes. Do not train from
an uncommitted NFS copy and do not add Hydra overrides.

### 7.2 Configure machine-local inputs

Run from `diffusion_template/` and keep secrets/paths in the ignored `.env`:

```bash
cp .env.example .env
chmod 600 .env
sha256sum /path/to/large_manifest.json
sha256sum /path/to/cosmic_manifest.json
```

Fill only the recipe-relevant dataset variables plus:

- `COMET_API_KEY`;
- `FACE_QUALITY_SCORER_PYTHON`, pointing to the existing PyIQA 0.1.15
  interpreter;
- `PM_PATH` if the pinned PhotoMaker v2 file is not at the cached default;
- the exact dataset manifest path, image/root path, and manifest SHA-256.

For Large Dataset and BigCelebs, do not change the supplied historical/sealed
hashes. For Cosmic, copy the hash from the exact historical CL14 data package
or immutable run record and record it in the run notes. Do not commit `.env`.

### 7.3 Run read-only preflights

Use the existing Serv `photomaker_NS` environment:

```bash
cd /absolute/path/to/diffusion_template
conda activate /absolute/path/to/conda_env/photomaker_NS
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py
bash -n launchers/active/run_e13_family_24k_1gpu.sh
bash -n launchers/serv/start_e13_family_1gpu.sh
```

The active launcher repeats the config/parity checks and then requires ORT-GPU
1.20.1/CUDA provider activation before model construction.

### 7.4 Prepare one MLS request

Use the exact E13, BC_E13 or CL14 clean YAML linked from
`serv_run_packages/README.md`. The generic template remains available only for
an intentionally new run identity; replace every `REPLACE_*` value if using it:

- project root is the absolute `diffusion_template` checkout;
- Conda path is the existing Serv `photomaker_NS` environment;
- log directory and run name are unique;
- config is exactly one supported leaf name from the recipe matrix.

Before submission, inspect both queues:

```bash
mls job list --status Running --limit 100 --output json
mls job list --status Pending --limit 100 --output json
```

Count this project's one- and two-GPU Running/Pending requests. The normal
ceiling is six requested A100s. Do not submit if the new one-GPU request would
exceed it. The current task authorizes preparation only, not submission.

When separately approved, the submission command is:

```bash
mls job submit --config /absolute/path/to/run_<run_name>_1gpu.yaml
```

### 7.5 Startup and completion checks

Monitor the run-specific stdout/stderr paths from the YAML. Startup is valid
only after:

- the manifest SHA check passes;
- the run-specific dataset decode/policy JSON exists under
  `preflight_records/<RUN_NAME>/`;
- config/parity validators pass;
- the fixed-96 manual and recipe-specific automatic bbox hashes pass, so no
  detector pass can replace the historical generation-mask inputs;
- ORT reports 1.20.1 with `CUDAExecutionProvider`;
- ownership logs 840/700/700 and the total 2,240/219,217,920;
- `saved/<RUN_NAME>/comet_experiment.json` exists and contains the immutable
  experiment key.

Expected validation/checkpoint boundaries are
`0,2000,...,24000`. After successful training, the same launcher invokes the
deferred face-quality finalizer. Scoring failure is nonfatal to completed model
artifacts but must be repaired before the run is called fully reported.

For CL14, preserve the step-0 full-96 images before continuing promotion work
and compare them against the immutable historical export. Any mismatch should
be investigated at the first divergent tensor/image; do not replace pixel
parity with aggregate metric similarity.

## Related documents

- [Implementation plan](2026-08-10_e13_bc_e13_cl14_clean_port_plan.md)
- [Current handoff](../handoffs/LATEST.md)
- [Validation protocol](../validation_protocol.md)
- [Tool index](../../TOOLS.md)
