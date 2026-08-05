# Anchored mix SA-v3 E4 differentiable-ranking implementation

**Date:** 2 August 2026  
**Status:** implemented, synchronized to Neb, and launched; step-zero
validation and the first training-batch gate passed

## Outcome

E4 keeps the completed E3 architecture, rank, 46 sites, mix bounds, data
schedule, optimizer, timestep policy, and fixed-96 validation unchanged. It
changes only the spatial-reference counterfactual objective:

```yaml
model:
  ba_reference_loss_mode: differentiable_rank
  ba_spatial_reference_shuffle_probability: 0.50

loss_function:
  reference_mode: differentiable_rank
  reference_weight: 0.10
  reference_relative_margin: 0.02
```

The old E3 detached diagnostic remains available through
`big_celebs_scheduled_rhca_anchored_mix_sa_v3_2k`; v1/v2 selectors and shared
defaults were not changed.

## Implemented files

- E4 config:
  `src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_2k.yaml`;
- Neb launcher:
  `launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_2k.sh`;
- canonical r2 experiment record:
  `experiments/big_celebs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2.json`;
- conditional counterfactual logging:
  `src/trainer/sdxl_trainers.py`;
- reference/native cosine and post-merge RMS telemetry:
  `src/model/photomaker_branched/anchored_mix_sa_processor_v3.py`;
- v3 checkpoint-evaluator recognition and audited `--ba-mix-override`:
  `tools/inference/evaluate_rhca_checkpoint.py`;
- audited runtime hash gate:
  `launchers/active/run_rhca_apr2026_one_id_1gpu.sh`.

Unconditional E3-compatible reference curves remain unchanged. New
`*_conditional` curves average only ranks/batches where the shuffled forward
actually ran, so E3's 25% and E4's 50% sampling rates are directly comparable.
Historical PhotoMaker configs without `reference_shuffle_applied` retain their
original logger path.

## Verification

Local static checks and the remote `photomaker_NS` environment passed:

- Python compilation, JSON parsing, and shell syntax;
- Hydra composition with exact v3/rank-32/46-site controls;
- resolved E4 objective mode/weight/margin/shuffle values;
- `pose_adapt_ratio=0` and `ca_mixing_for_face=false`;
- telemetry smoke for finite cosine and merged/native RMS;
- conditional logger smoke for active shuffles and the historical
  non-reference fallback;
- evaluator CLI exposure of `--ba-mix-override`;
- local/remote SHA-256 equality for synchronized files.

Remote originals were preserved under:

```text
/home/niko/rsrch/runtime_backups/anchored_mix_sa_v3_e4_20260802_180639
```

## Neb launch

The first submission identity (`r1`) stopped before model construction, GPU
use, or Comet experiment creation because the integrity gate still classified
the intentionally changed trainer as historical. Its null-key saved record was
preserved; it was not reused.

Fresh run:

```text
rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2
```

- launcher/process group: `3454285`;
- training PID/PGID at startup: `3454498` / `3454498`;
- log:
  `/home/niko/rsrch/diffusion_template/logs/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2.log`;
- saved directory:
  `/home/niko/rsrch/diffusion_template/saved/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2`;
- immutable Comet key: `f72ea55eb0af44828cd6511a15ba5933`;
- Comet project: `jul-comet-large-testing-tr`;
- Comet URL:
  `https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/f72ea55eb0af44828cd6511a15ba5933`.

Startup passed the sealed dataset/schedule preflights, ONNX CUDA provider,
runtime-integrity gate, GPU-side processor construction in 5.75 seconds, exact
414 / 10,567,818 trainable contract, and immutable Comet record creation.
Step-zero `validation_native` generated all 96 images on RealVisXL in 12m12s;
face quality detected all 96 faces. Comet API verification found the identity,
text, and seven face-quality metrics plus exactly 96 image assets. All 96
saved E4 step-zero PNG SHA-256 values are identical to E3 r2 step zero, proving
the training-only objective toggle did not change inference initialization.

Training then passed the required first three batches and continued beyond
batch 21 with finite loss. Observed iteration time alternated around
`2.0-2.6 s/it`, as expected when a differentiable wrong-reference forward is
sampled on 50% of batches. Training GPU memory was about 43.4 GiB, below the
79.4 GiB fixed-96 validation peak. New cosine/merged-RMS series and shuffle
fraction were visible through the Comet API at the first training log. At step
50 the effective shuffle fraction was `.54`, conditional relative error gap
`1.89%`, conditional prediction delta `10.88%`, reference/native cosine `.806`,
and merged/native RMS `.950`.
