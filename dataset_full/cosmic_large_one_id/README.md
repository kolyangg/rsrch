# Cosmic Large one-ID dataset

This is the self-contained single-identity dataset used by the 23 July
experiments (`id_00081_1017318003459`, class `woman`).

- `target/target.jpg` is the only diffusion training target.
- `train_refs/` contains eight distinct reference images sampled by the
  training dataset class. None is the target image.
- `validation_refs/holdout_A.jpg` is excluded from training and is used for
  recurring validation.
- `holdouts/holdout_B.jpg` is also excluded from training and is reserved for
  final-only checks.
- `validation_prompts.txt` is an exact copy of the 12-prompt `one_id`
  validation prompt set.
- The templates are shared, but the resolved class token is intentionally
  different: this dataset generates `woman img`, whereas `one_id` generates
  `man img`. The corrected seed path therefore locks the latent scene and
  broad composition, but PhotoMaker may also change hair, clothing details,
  and body contour along with the face.
- `photomaker_validation/` contains the seed-0 PhotoMaker baseline used to
  define generated-face masks. It exactly follows the RHCA validation
  stochastic path: RealVisXL DDIM, one CUDA generator per sample, and all
  12 prompts generated in one batch.
- `photomaker_generated_bboxes.json` is indexed in validation order as
  `00.png` through `11.png`, matching `ManualPhotoMakerValDataset`.
- `cosmic_large_one_id_photomaker_bboxes_rhca_seed0.pdf` overlays the detected
  boxes for visual inspection.

The source-to-local mapping and the reference/target separation guarantee are
recorded in `split_manifest.json`.
