# E16 E14 PhotoMaker-teacher arm on OneIDTrain — leakage audit only

> **Invalid for promotion:** like E15, every target is its own pixel-identical
> reference. Retained only as a within-leakage teacher-loss comparison. Use
> E18/E19 for corrected distinct-image experiments.

Purpose: test the OneIDTrain preprocessing/self-reference hypothesis with the
stronger preservation objective, using the same fixed subset as E15.

- Architecture/optimizer: exact E14.
- Added loss: `0.20 * MSE(epsilon_BA, epsilon_PhotoMaker_teacher)`.
- Uses E14's audited vanilla-attention teacher pass and restores the exact
  branched processor instances before backpropagation.
- Dataset, identity, subset, seed, checkpoints, and validation: exact E15.

Re-run:

```bash
./run_architecture.sh E16_oneid8_pm_teacher20
```

E15 versus E16 isolates the matched-teacher anchor inside the one-id dataset
type; E04/E14 provide the corresponding CosmicLarge-style comparison.

## Result

Completed at steps 200/400/600. Step 200 is nearly indistinguishable from E15,
and the same eye/glasses distortion appears at steps 400/600. Reference/noise
LoRA-B norms end at `3.4865/0.2204`, versus E15's `3.5069/0.2205`, so the
teacher only weakly damps reference drift and does not change the visual
failure mode. Do not promote this arm over E15: it adds a second denoiser pass
during training without a useful visual gain.
