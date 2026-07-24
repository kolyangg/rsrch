# E15 E04 projection split on OneIDTrain — leakage audit only

> **Invalid for promotion:** `train_on_separate_image=false` makes every
> target image its own pixel-identical reference. Retained only to quantify
> the optimistic effect of target leakage. Use E18 for the corrected run.

Purpose: isolate the user's dataset-type hypothesis without changing E04's
architecture, optimizer groups, learning rates, loss, step budget, or seed.

- Architecture/optimizer: exact E04 projection split.
- Dataset class: `src.datasets.cosmic.OneIDTrain`.
- Identity: native one-id subject `nm0005092`.
- Training subset: eight fixed 1024×1024 records, excluding validation
  reference `51.jpg`.
- `train_on_separate_image=false`, matching the native one-id configuration:
  each target is also its training reference.
- Protocol: 600 steps; checkpoints at 200/400/600.
- Validation: native `one_id_val` reference and four fixed prompts, with masks
  derived from its PhotoMaker controls.

Re-run:

```bash
./run_architecture.sh E15_oneid8_projection_split
```

Compare to E04 qualitatively, since the identity and validation cases differ.
The causal question is whether OneIDTrain avoids the duplicated/smeared
landmarks seen in E04 under the same optimizer design.

## Result

Completed at steps 200/400/600. Step 200 is the promotion checkpoint: all four
faces remain aligned, Reading/Rushing are sharp and coherent, and the gross
horizontal landmark collapse from E04 is absent. Steps 400/600 begin to show
eye/glasses distortion, so this protocol should early-stop at step 200.

The initially strong result is explained by target/reference leakage and is
not evidence that the loader generalizes identity. It must not be promoted.
