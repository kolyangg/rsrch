# Anchored mix SA-v3 E5-L40 overnight launcher

**Date:** 2 August 2026  
**Status:** running on Neb; startup contract and step-zero validation start
verified  
**Run:**
`rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1`

## Purpose

E5-L40 is a fresh, exact long-horizon repetition of E4. It changes only
`trainer.n_epochs` from 1 to 20, giving a maximum of 40,000 optimizer steps.
Its first scientific decision point remains 8k. The additional ceiling allows
an unattended process to retain the complete 10–20k likely-peak window and a
later plateau curve.

No architecture, loss, rank, optimizer, dataset order, validation input, or
inference setting changes. In particular:

- explicit target-Q/reference-KV branched self-attention remains the model;
- architecture is `anchored_mix_sa_v3`, rank 32, at 46 mid/up0/up1 sites;
- the E4 differentiable rank objective remains weight `.10`, relative margin
  `.02`, and 50% shuffled-reference sampling;
- `pose_adapt_ratio=0` and `ca_mixing_for_face=false`;
- full and weights-only checkpoints plus fixed-96 validation are produced
  every 2,000 steps.

The resolved Hydra diff against E4 is exactly:

```text
trainer.n_epochs: 1 -> 20
```

## Live run

The canonical run was launched on Neb at `2026-08-02T20:59:39Z`:

- launcher PID/PGID: `3468188` / `3468188`;
- accelerate PID: `3468397`;
- training PID/PGID: `3468434` / `3468434`;
- GPU: `0`;
- immutable Comet key: `f5b5a7054e854137abe53c47f34ebae0`;
- Comet project: `jul-comet-large-testing-tr`;
- log:
  `/home/niko/rsrch/diffusion_template/logs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1.log`.

Both dataset preflights, ONNX CUDA, face-quality scorer, audited runtime, and
immutable-record creation passed. Anchored-v3 processor construction took
5.785 seconds. Exact ownership was `414 tensors / 10,567,818 parameters`, and
optimizer membership was `414/414`. Step-zero `validation_native` fixed-96
generation started on RealVisXL and reached its first image marker. No
traceback, OOM, non-finite value, or integrity error was present at handoff.

## Why this is one run rather than chained experiments

One continuous run preserves optimizer state and deterministic dataset
position and records one directly comparable Comet trajectory. Separate
chained runs would duplicate model construction and step-zero validation,
consume another Comet identity, and prevent a clean continuous learning curve.

Training to a later step cannot erase an earlier candidate: checkpoints and
fixed-96 results are retained independently at every 2k boundary. Select the
best validated intermediate checkpoint rather than automatically selecting
step 40k.

## Artifacts

- Config:
  `src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_40k.yaml`
- Neb launcher:
  `launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_40k.sh`
- Prepared immutable record:
  `experiments/big_celebs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1.json`

## Launch on Neb

After synchronizing the three artifacts and confirming the current GPU process
group is intentionally stopped:

```bash
cd /home/niko/rsrch/diffusion_template
RUN_NAME=rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1
nohup setsid bash \
  launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_40k.sh \
  > "logs/${RUN_NAME}.log" 2>&1 < /dev/null &
echo $!
```

The launcher fails closed if Neb already has an active GPU compute process. It
reuses the existing sealed-dataset, schedule, CUDA ONNX, runtime-integrity,
strict trainable-contract, fixed-96, and immutable Comet-record gates.

During startup, confirm:

1. `saved/<run_name>/comet_experiment.json` contains a non-null immutable Comet
   key in `jul-comet-large-testing-tr`;
2. step-zero validation logs 96 images, identity/text, and all seven compact
   face-quality curves;
3. trainable ownership is exactly `414 tensors / 10,567,818 parameters` and
   optimizer membership is `414/414`;
4. the first three training batches and role gradients are finite.

## Interpretation

- Step 2k is a reproducibility/safety point, not a promotion decision.
- Step 8k is the first meaningful decision point. Look for identity recovery
  from the 2k trough, p10 stabilization, coherent faces, and persistent
  matched-reference causality.
- Steps 10–20k are the likely checkpoint-selection window based on historical
  BA curves, which peaked around 18k.
- Steps 22–40k are useful only if improvement continues; otherwise they
  document plateau or overtraining. They do not invalidate a stronger earlier
  checkpoint.

E4 observed roughly `2.0–2.6 s/it`, and each fixed-96 validation took about 12
minutes before scoring overhead. The complete 40k ceiling will therefore take
approximately 27–34 hours, not one conventional night. A 9–12 hour unattended
window should normally reach roughly the first 8–14k decision region, after
which the same process can continue if left running.
