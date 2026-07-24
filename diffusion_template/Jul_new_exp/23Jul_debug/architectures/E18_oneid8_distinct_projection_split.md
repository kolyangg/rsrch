# E18 corrected OneIDTrain projection split

Purpose: repeat E15 without target/reference leakage.

- Architecture, optimizer, loss, seed, subset size, and validation: exact E15.
- Dataset class: `src.datasets.cosmic.OneIDTrain`.
- `train_on_separate_image=true`.
- For every sample, the reference index is selected from
  `all_indices - {target_index}`; therefore target and reference are different
  filenames but remain the same identity (`nm0005092`).
- Fixed subset: the same eight images used by E15/E16; held-out validation
  reference `51.jpg` remains excluded.

Re-run:

```bash
./run_architecture.sh E18_oneid8_distinct_projection_split
```

This is the valid dataset-type ablation. E15/E16 are retained only as leakage
audits and must not be used for architecture promotion.

Result: rejected. Step 200 develops displaced/duplicated landmarks. Geometry
largely recovers by step 600, but median reference similarity drops from
`0.3434` at step zero to `0.2931`, and gain versus PhotoMaker changes from
`+0.0689` to `-0.0175`. The late clean-up is PhotoMaker-identity drift.
