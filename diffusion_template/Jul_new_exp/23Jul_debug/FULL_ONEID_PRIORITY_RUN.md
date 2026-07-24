# Priority-1 architecture on the full OneID training set

## Run

- run ID: `L4_OF1_oneid_full18_projection_alt`
- base architecture: `L4_O1_oneid_projection_alt`
- protocol: 4,000 optimizer steps; validation at step 0 and every 500 steps
- Comet: training and all validation stages use one experiment key

## Dataset

The source OneID JSON contains 19 images. `51.jpg` is byte-identical to the
only validation reference and is excluded from training. The effective
training dataset therefore contains the other 18 images:

`1, 11, 31, 36, 37, 38, 40, 46, 57, 58, 62, 80, 83, 103, 104, 109, 116, 117`.

`train_on_separate_image=true` remains mandatory. A preflight instantiated
the real loader, sampled eight references for every target, and found zero
target/reference identity violations. The validation reference is held out.

The exact filtered JSON and provenance are in
`data/one_id_nm0005092/full18_no_validation_train.json` and
`data/one_id_nm0005092/full18_heldout_manifest.json`.

## Queue placement and start

At the user's request, the subset-OneID priority-2 arm was stopped after its
step-1000 checkpoint. Its artifacts remain under `experiments_4k/`; its
manifest is marked `interrupted`.

The full18 run started at 2026-07-24 08:58:38 UTC alongside the retained
CosmicLarge priority-2 arm. Once both finish validation, reporting, and their
Comet-unity audits, the replacement continuation retires the old pair-level
queue and resumes at priority pair 3. The stopped subset arm is explicitly
excluded from relaunch.
