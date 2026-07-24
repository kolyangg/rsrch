# E14 projection split plus matched PhotoMaker teacher

Purpose: directly penalize the malformed-face trajectory seen in E00–E04 by
anchoring the BA epsilon prediction to a frozen PhotoMaker prediction, while
retaining the normal target diffusion objective.

- Forward/inference architecture: exact NN3a_new1.
- Optimizer routing and LR multipliers: exact E04 projection split.
- Primary diffusion loss: E04/E00 `masked_alternating`.
- Added loss: `0.20 * MSE(epsilon_BA, epsilon_PhotoMaker_teacher)`.
- The teacher and BA passes use the exact same noisy target latent, sampled
  timestep, prompt, reference augmentation, and random state.
- Teacher pass follows the normal pre-BA text/PhotoMaker schedule, is executed
  under `torch.no_grad()`, and consumes no additional experiment RNG state.
- Because NN3a installs processors that require doubled
  `[target, reference]` batches, the one-row teacher pass temporarily swaps
  only those branched processors for vanilla `AttnProcessor2_0` objects. The
  exact original processor instances are restored before the differentiable
  BA pass, preserving all trainable parameters and optimizer references.
- No production-repository code is changed; the implementation is confined to
  `training_lab/nn3a_lab_model.py`.

Re-run:

```bash
./run_architecture.sh E14_projection_split_pm_teacher20
```

This is a bounded-correction training objective, not an identity objective.
It should be promoted only if BA remains visibly distinct from PhotoMaker
without developing duplicated/displaced landmarks.

Implementation audit: the first launch at `20260723T204604Z` exposed the
required processor swap before taking an optimizer step (`total=1,
reference=0`). It is retained as a failed preflight artifact. The corrected
implementation then completed a console-only forward/backward/optimizer smoke
step with finite teacher loss (`0.0017985`), weighted contribution
(`0.0003597`), total loss (`0.044944`), and gradients in all six optimizer
groups. That implementation was used for the clean 600-step run below.

## Result

The corrected clean retry completed and is rejected. Step 200 already has the
same severe horizontal landmark displacement as E04; steps 400/600 remain
malformed. Metrics differ only slightly from E04 (for example step-600
reference similarity `0.3562` versus `0.3533`). A teacher weight of `0.20`
does not preserve usable geometry in the valid distinct-reference Cosmic
setup.
