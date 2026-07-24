# E20 corrected distinct-pair reference-value-only training

Purpose: test whether NN3a can learn reference identity content without
moving the attention coordinates that fail in E18.

- Dataset and target/reference exclusion guarantee: exact E18/E19.
- Forward and inference: exact NN3a_new1; no branched-attention logic changes.
- Trainable tensors: only `ref_to_v` LoRA tensors in the active up blocks.
- Frozen tensors: reference Q/K and all noise-clone Q/K/V tensors.
- LR: up1 reference V at `5e-5`; up0 reference V at `1.25e-5`.
- Loss: E19's always-anchored `0.80 * full + 0.20 * face` loss.

This is a causal response to the corrected E18 step-200 failure. Reference V
can change the identity-bearing content delivered at the fixed attention
locations, while frozen Q/K prevents training from changing which spatial
locations attend to one another. Freezing the noise clone also protects the
base target-coordinate route.

Re-run:

```bash
./run_architecture.sh E20_oneid8_distinct_ref_value_only
```

Promote only if its held-out validation faces retain coherent eyes, nose, and
mouth while becoming meaningfully different from both step zero and ordinary
PhotoMaker.
