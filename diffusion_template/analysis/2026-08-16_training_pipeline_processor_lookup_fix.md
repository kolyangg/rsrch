# Training pipeline fix: cache Diffusers processor maps once per collector

**Date:** 16 August 2026  
**Scope:** CL14 and CL29 training throughput; one A100, batch 2  
**Primary metric:** median displayed seconds/iteration over optimizer steps
`21–99` in bounded `ex_` runs

## Executive conclusion

The current pipeline slowdown was caused by repeated evaluation of Diffusers'
`unet.attn_processors` property inside per-layer collection loops. That property
recursively walks the U-Net and rebuilds the complete processor dictionary on
every access. A disabled CL14 auxiliary therefore performed about `70` complete
U-Net traversals per optimizer step. CL29 repeated the same pattern in multiple
enabled and cleanup collectors. **[code]**

The code-level fix skips collectors whose owning modes are disabled and caches
the processor map once before every per-layer loop. It does not change Q/K/V
routing, activations, losses, gradients, optimizer ownership, data, or
validation. **[code]** Current-source CL14 improved from **`3.56` to `2.06
s/it`**, recovering the immutable historical control's `2.15 s/it`. The first
fixed CL29 qualification improved from **`6.21` to `3.21 s/it`**. **[measured]**

## 1. Controlled results

| Arm | Immutable Comet key | Source | Warm window | Median s/it |
|---|---|---|---:|---:|
| Historical CL14 production | `6fe0028be92242c38056b3d36665fdd6` | `c04970f...` + sealed CL14 overlay | `21–100` | `2.21` |
| Historical CL14 replay | `92b86b61d701479d85a66733a14c0262` | exact sealed historical source | `21–99` | `2.15` |
| Current CL14 before lookup fix | `e17cbf2df1e245f1a4685acd34db072d` | `0aa7abb` | `21–100` | `3.56` |
| Current CL14 after lookup fix | `02adf5c00410448898240da572a3ba25` | `65ba4a9` | `21–99` | **`2.06`** |
| CL29 speed pipeline before lookup fix | `2c5d2e18558249138e5edf7b6be0b01f` | `8dec793` | `21–80` | `6.21` |
| CL29 after lookup fix, first confirmation | `1d2766f1a95648bbb55ce9822ee953cb` | `f40ecb2` | `21–99` | **`3.21`** |

The CL14 intervention removes `1.50 s/it` (`42.1%`) and is `4.2%` faster than
the historical-source replay on the same current infrastructure. The first
CL29 confirmation removes `3.00 s/it` (`48.3%`, `1.93x` throughput). Its larger
gain is consistent with CL29 exercising more than one affected processor
collector per step. **[measured] [code]**

## 2. Exact fix and required defaults

The implementation is in
`src/model/photomaker_branched/lora2_helpers.py` and
`src/datasets/cosmic_large_adapted.py`:

1. Return immediately from hard-case auxiliary collection unless semantic
   ownership or visibility-order supervision owns that loss.
2. Resolve `model.unet.attn_processors` once in branched telemetry, hard-case,
   frequency-surface, schedule-anchor, low-band, and low-band-clear collectors.
3. Do not allocate a `1024×1024` fp32 semantic-occlusion mask when semantic
   occlusion is disabled.
4. For new experiments, keep `trainer.active_grad_norm_mode=requested_only` and
   disable full-activation BA telemetry unless the run explicitly consumes it.
5. For CL29-derived experiments, keep CPU-side low-band gate sampling. This
   removes an otherwise unnecessary synchronization of the active CUDA stream.

These are the required pipeline defaults for new experiments. Historical
replays retain their sealed behavior. Any proposed deviation must be presented
to the user and explicitly approved before configuration or launch.

## 3. What is not the cause

- **[measured]** The current Serv/A100 runtime is not the CL14 regression: the
  immutable historical source reproduced `2.15 s/it` on current infrastructure.
- **[code]** The active-gradient scan is removable dead work when unrequested,
  but it exists in the exact sealed historical CL14 overlay and therefore does
  not explain the CL14 source-to-source gap.
- **[measured]** Validation is outside every reported timing window. CL29
  qualifications intentionally omitted step zero; CL14 used the same bounded
  12-image startup validation before its compared training window.

## 4. Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Repeated processor-map reconstruction caused the current CL14 regression | High | Direct control intervention recovers historical speed; exact source inspection identifies the per-step recursive lookup |
| The fix preserves scientific computation | High | Only disabled-path guards, dictionary lookup reuse, and disabled-mask allocation changed |
| The fix materially accelerates CL29 | High | Same CL29 contract and bounded warm window improve `6.21 → 3.21 s/it` |
| Every remaining CL19–CL29 cost has been eliminated | Not established | Later arms still contain real frequency routing, diagnostic, and auxiliary compute |

## 5. Remaining high-impact work

Priority candidates remain separately gated optimizations: avoid CL29's second
branched forward while its ramp weight is exactly zero; keep CL26's discarded
legacy result removed; retain tensorized CL27 eligibility; cadence optional
full-activation diagnostics; and profile data transfer/alternate-reference
decoding before introducing caching. Each change must preserve numerical
loss/gradient behavior or be treated as a user-approved experiment change.

## 6. Reproducing the checks

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template

python tools/validate_CL14_speedcheck_config.py \
  --config-name CL14_cosmic_joint_shadow_sa128_softmask_24k_speedcheck

python tools/validate_CL29_speedcheck_config.py \
  --config-name CL29_cosmic_lowband_causal_contrastive_24k_speedcheck

rg -n 'unet\.attn_processors\.get' \
  src/model/photomaker_branched/lora2_helpers.py
```

Expected: no `unet.attn_processors.get(...)` call remains inside a per-layer
runtime collector loop. Initialization and checkpoint-load lookups are outside
the training hot path.

## 7. References

- `analysis/2026-08-16_cl14_cl29_training_throughput_optimization_plan.md`
- `docs/handoffs/LATEST.md`
- `experiments/cosmic_large/ex_CL14_current_auxlookup_fix_r1.json`
- `experiments/cosmic_large/ex_CL29_auxlookup_fix_speedcheck_r3.json`
