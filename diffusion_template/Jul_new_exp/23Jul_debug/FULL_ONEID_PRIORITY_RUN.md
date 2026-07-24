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

## Queue placement

The active priority-2 pair continues without interruption. The main queue is
paused at its current `wait`, so it cannot advance to pair 3. When both active
runs finish training, validation, reporting, and Comet-unity audits, the
full-OneID priority run starts alone. After it completes and passes the same
report/audit path, the main queue resumes with priority pair 3.
