# E00 control

Purpose: exact `NN3a_new1` training control under the fixed 600-step one-ID
protocol.

- Forward: unchanged 70-processor NN3a_new1 branched self-attention.
- Trainable scope: all down/mid/up BA projection clones.
- Trainable size: 840 tensors, 31.95M parameters.
- Loss: `masked_alternating`, `lambda_face=0.1`.
- LR: reference `5e-5`; noise `1.25e-5`.

Re-run:

```bash
./run_architecture.sh E00_control
```

Observed: severe inner-core blur at step 0 and non-rigid face-coordinate
warping by step 200; no checkpoint through step 600 is visually acceptable.
