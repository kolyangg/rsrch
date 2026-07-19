# PPR validation fix and checkpoint revalidation

## Conclusion

The 2k plots show that PPR **was training**: `connector_up` has gradients immediately, then `connector_down`, gate, reference K/V and `ba_norm/sa_ref` become nonzero. Do not retrain yet. Fix validation and rerun the saved 2k checkpoint.

## Code fix

`two_branch_predict()` caches `_ba_packed_branch_exactly_off=True` during zero-weight validation. Reset this cache once at the start of **every new pipeline generation**, before the denoising loop (preferably at the start of `run_branched_setup()`):

```python
for attr in (
    "_ba_packed_branch_exactly_off",
    "_ba_output_anchor_logged",
):
    if hasattr(pipeline, attr):
        delattr(pipeline, attr)
```

Keep the cache during the 50 denoising steps; only invalidate it between pipeline calls.

Add a regression test that:

1. Runs step-zero inference under `torch.no_grad()` and confirms exact base parity.
2. Makes `connector_up` nonzero.
3. Starts a new generation/invalidate caches.
4. Runs again under `torch.no_grad()` and confirms a nonzero, face-local output change.

## Rerun validation from the 2k checkpoint

Yes—no retraining is required if the checkpoint contains `state_dict.attn_processors`.

1. Load the **exact saved Hydra configuration** and instantiate the RealVis validation model/pipeline.
2. Call `prepare_for_training()` before loading so all PPR processors exist.
3. Load `checkpoint["state_dict"]` with `model.load_state_dict_()`; keep strict processor restoration enabled.
4. Assert that saved and live `connector_up.weight` tensors are nonzero and identical after loading.
5. Run the unchanged fixed 96-image RealVis validation set with the original seeds, prompts and masks under `model.eval()` and `torch.no_grad()`.
6. Do not restore optimizer state or execute a training step. Prefer a dedicated validation-only script over resuming `train.py`.

Checkpoint preflight:

```python
ckpt = torch.load(PATH, map_location="cpu")
procs = ckpt["state_dict"]["attn_processors"]
up = [v["connector_up.weight"].float() for v in procs.values()]
assert up and sum(torch.count_nonzero(x).item() for x in up) > 0
```

## Required comparison

For step zero versus revalidated 2k, record:

- image SHA-256 and face-region pixel MAE;
- ID similarity and text similarity;
- `connector_up` L2/nonzero count and actual gate values;
- pre/post-cap residual ratios;
- final face-core `RMS(epsilon_ppr - epsilon_base) / RMS(epsilon_base)`.

If outputs are numerically different but still weak, first reduce future warmup from 2,000 to 500 steps. Do not strengthen both gate and RMS cap until the residual and identity direction have been measured.
