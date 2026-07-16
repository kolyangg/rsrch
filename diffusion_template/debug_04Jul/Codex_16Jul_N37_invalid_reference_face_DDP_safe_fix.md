# N37 invalid-reference-face failure and DDP-safe fix

## Failure

N37 stopped at training microbatch 1569 with:

```text
ValueError: Reference face detection failed for identity-conditioned training
```

This was not an OOM, optimizer, canonical-resampler, or NCCL root cause.
InsightFace failed on one randomly selected Cosmic Large reference image.
`ba_require_reference_face: true`, introduced in N34 and inherited by N35,
N36, N37, and N38, converted that single detection miss into a fatal exception.
The later NCCL warning was only a consequence of rank 0 exiting.

N37 is especially dependent on reliable detection because its canonical memory
uses face landmarks, but the same fatal reference path existed in all N34-N38
runs. A difficult target face could also fail the strict causal-landmark check.

## Fix

1. Face detection first runs on the original image as before.
2. If it fails or lacks the required embedding/landmarks, it retries on a
   padded crop of the dataset-provided face bbox.
3. Landmarks detected in the crop are translated back into original-image
   coordinates before canonical alignment.
4. If both attempts fail, the invalid status is synchronized before any later
   global-negative collectives. Every DDP rank therefore takes the same path.
5. The trainer skips the complete gradient-accumulation window. It does not
   train on a zero identity embedding, perform a partial effective-batch update,
   or terminate the distributed job.

The behavior is controlled by:

```yaml
model:
  ba_reference_face_bbox_fallback: true
  ba_skip_invalid_identity_samples: true
```

Both toggles are enabled in N34 and inherited by N35-N38. Setting the second
toggle to `false` restores fatal strict behavior after both detection attempts.
The toggles do not change tensor shapes or the strict checkpoint architecture
manifest, so existing checkpoints remain loadable.

## Expected diagnostics

Successful bbox recovery, limited to the first three messages:

```text
[BA Reference Face] recovered detection from bbox crop count=1
```

An image that remains unusable is identified on its local rank and skipped:

```text
[BA INVALID IDENTITY] rank=... reference_face_missing ... identity=... bbox=... sha1=...
[INVALID_IDENTITY_SKIP] ... batch_idx=...
```

## Verification

- Python compilation passes for the changed model, helper, trainer, and tests.
- Shell syntax passes for the N37 launch script.
- YAML inheritance resolves both safety toggles to `true` for N34-N38.
- The flags were intentionally excluded from the strict tensor-architecture
  manifest to preserve checkpoint compatibility.
- The local system Python lacks PyTorch, so the GPU/runtime unit test suite
  must be run in the PhotoMaker training environment.

