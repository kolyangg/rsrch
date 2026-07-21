# PPR 8k reference-content versus reference-noise conclusion

## Automatic result

- Classification from face-core pixel effects: **primarily reference-content-driven**
- Mean reference-image effect: `0.01254256`
- Mean reference-noise effect: `0.00902592`
- First tensor stage with mean swap sensitivity below `1e-3`: `none detected automatically`
- LPIPS status: available
- Reference-half CA mode: `zero`

This classification is provisional. Review `contact_sheets/`, `face_crops/`,
and `difference_heatmaps/`, then compare identity-to-original and
identity-to-swapped columns in `metrics_per_image.csv`.

The reported seam score is a bbox-boundary gradient-discrepancy proxy, not a
learned perceptual artifact detector. Tensor differences use exact tensor
SHA-256/RMS plus the same deterministic 512-value sketch at each paired stage.
