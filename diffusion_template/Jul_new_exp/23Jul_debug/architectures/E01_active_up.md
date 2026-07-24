# E01 active-up optimizer scope

Purpose: test whether updates in down/mid processors cause NN3a_new1's
training-time alignment drift.

- Forward: identical to E00; all 70 processors remain installed.
- Trainable scope: 36 processors in `up_blocks.0` and `up_blocks.1`.
- Trainable size: 432 tensors, 16.22M parameters.
- Loss/LR: unchanged from E00.
- Step-zero parity: exact by construction because only optimizer membership
  changes.

Re-run:

```bash
./run_architecture.sh E01_active_up
```

Resume after a checkpoint-boundary interruption:

```bash
./run_architecture.sh E01_active_up \
  --resume-run-dir experiments/<E01-run-folder>
```

Status: 600-step training complete; checkpoint validation pending.
