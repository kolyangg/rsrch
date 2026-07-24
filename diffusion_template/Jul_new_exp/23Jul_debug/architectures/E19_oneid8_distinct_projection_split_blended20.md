# E19 corrected distinct-pair OneIDTrain with anchored loss

Purpose: test a cheap late-drift control on top of the corrected E18 pairing.

- Dataset, target/reference guarantee, identity, subset, seed, and validation:
  exact E18.
- Architecture and optimizer groups: exact E18/E04 projection split.
- Loss: `0.80 * full_image_MSE + 0.20 * face_MSE` on every step.
- No PhotoMaker teacher pass.

Re-run:

```bash
./run_architecture.sh E19_oneid8_distinct_projection_split_blended20
```

Compare first against E18, then against the invalid same-image E15 only to
measure how much apparent quality came from leakage.
