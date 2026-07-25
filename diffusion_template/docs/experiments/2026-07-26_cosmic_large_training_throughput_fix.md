# Full-Cosmic training throughput fix

Date: 26 July 2026
Branch: `test`
Base commit during investigation: `6c8fd6790265cdbd8a24d512fd1a37884f44ae01`

## Observed evidence

- The completed Serv run `rhca_cosmic_full_crop20_legacy_4k` trained at
  approximately 4.8-5.2 seconds per step after startup. Occasional early
  progress estimates reached 6-7 seconds per step.
- The Neb one-ID run
  `rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k` settled at approximately
  1.25-1.30 seconds per step.
- Both resolved configurations used batch size 2 and 12 DataLoader workers.
  Data loading worker count was therefore not the cause.
- The full-Cosmic loader accepted 22,140 records at the current
  `min_face_res=192` setting. Its target/reference pairs are effectively
  unique. The current 512-entry conditioning LRU therefore does not warm
  during a 4k run.
- The old conditioning cache key includes the target prompt and target face
  box. A different target invalidates the whole entry, including frozen
  reference-only work: text encoding, PhotoMaker ID encoding, InsightFace
  embedding extraction, reference VAE encoding, and masks.
- A one-ID run pays the expensive miss at startup and then repeatedly hits the
  cache. Full Cosmic paid the same miss on almost every step.

## Implemented change

The training model now has an opt-in
`model.batched_conditioning_preparation` switch. When enabled for the
full-Cosmic configuration, it performs the frozen text, PhotoMaker ID, and
reference-VAE work once per batch instead of once per sample.

Legacy behavior remains the default:

- `one_id_rhca_apr2026_replay`: conditioning cache enabled; batched
  preparation defaults to false.
- `cosmic_large_adapted_rhca`: conditioning cache disabled; batched
  preparation enabled.

The new path preserves each sample's prompt, reference image, InsightFace
embedding, target/reference face boxes, masks, and reference latent. It changes
only how independent frozen encoder calls are grouped.

Changed files:

- `src/model/photomaker_branched/lora2.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/configs/cosmic_large_adapted_rhca.yaml`
- `launchers/active/run_rhca_apr2026_one_id_1gpu.sh`

## Verification

Focused checks completed:

- Python compilation passed for both modified model files.
- Shell syntax passed for the active one-GPU launchers.
- Hydra composition on Neb resolved:
  - full Cosmic: cache false, batched preparation true;
  - one-ID replay: cache true, batched preparation absent/default false.
- A 20-step full-Cosmic training smoke test completed without OOM.
- A 100-step full-Cosmic training benchmark completed without OOM or runtime
  failure:
  - run: `profile_cosmic_batched_conditioning_100step`;
  - location:
    `/home/niko/rsrch/diffusion_template/saved/profile_cosmic_batched_conditioning_100step`;
  - measured at step 50: 1.1466 steps/second (0.872 seconds/step);
  - complete tqdm average including the slow first step: approximately
    1.07 steps/second (0.93 seconds/step).
- A same-seed, same-sample one-step comparison produced:
  - batched loss: `0.06430425494909286`;
  - legacy loss: `0.06425689160823822`;
  - relative difference: approximately 0.074%;
  - both logged the same bf16 gradient norm: `0.031982421875`.

The small loss difference is consistent with batching frozen bf16 inference
through different GEMM execution shapes. No input, conditioning source, loss
definition, or trainable parameter set changed.

## Conclusion

The slowdown was a conditioning-cache locality problem, not a slow
full-Cosmic DataLoader. Batching the frozen conditioning path reduced measured
full-Cosmic training from roughly 5 seconds/step to roughly 0.9 seconds/step,
slightly faster than the prior warmed one-ID runs.

A persistent disk cache is not needed for the current batch-size-2 workload.
It would add substantial storage, invalidation, and preprocessing complexity
without improving the measured batched throughput.

The benchmark runs disabled validation and used the console writer. Their
weights are throughput artifacts only and must not be treated as experiment
endpoints.
