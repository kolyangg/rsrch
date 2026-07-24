# E02 up1-only detail route

Purpose: prevent coarse face-coordinate movement while allowing the
highest-resolution active route to learn eyes, mouth, skin, and local identity
detail.

- Forward: identical to E00/E01.
- Trainable scope: the six `up_blocks.1` BA processors only.
- Trainable size: 72 tensors (exact parameter count is recorded at launch).
- Loss/LR: unchanged from E00.
- Step-zero parity: exact; non-up1 processors are frozen, not removed.

Re-run:

```bash
./run_architecture.sh E02_up1_detail
```

Promotion signal: sharper held-out faces with materially lower landmark/bbox
displacement than E00 and E01. Weak identity change is preferable to a face
melt in this isolated screen.
