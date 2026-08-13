# E13-family clean source archive

`rsrch-kit-e13-family-clean-source-0695f34.zip` is a source-only snapshot of
commit `0695f345f149d691d43a4b1ba541a010ffebcd71` on
`kit/e13-family-clean`.

- Size: 694,482 bytes (680 KiB)
- Tracked files: 251
- SHA-256: `8f3030c0d95a38e05b8bde4adcc79b96c5e8dff04b27a1217380ff3ef8932676`

The archive contains every tracked file at that commit except:

- `dataset_full/` — dataset records, analysis notebooks, and validation media;
- `compare/` — comparison/reference media;
- `diffusion_template/bbox_utils/yolov8n-face.pt` — third-party detector weights.

Git archives never contain `.git/` or untracked/ignored files, so local `.env`
credentials, caches, generated checkpoints, and other machine artifacts are
also absent. The archive is made from the preceding source commit and is not
self-referential, so it does not contain this `release/` directory.

Verify after downloading:

```bash
sha256sum rsrch-kit-e13-family-clean-source-0695f34.zip
unzip -t rsrch-kit-e13-family-clean-source-0695f34.zip
```

## Expanded archive

`rsrch-kit-e13-family-clean-expanded-f8bc4d2.zip` is the broader snapshot of
commit `f8bc4d2bcf966ceab0d495344716d722d32f0741` requested with a 20 MB limit.

- Size: 18,145,085 bytes (17.3 MiB)
- Tracked files: 389
- SHA-256: `623a8aa87fba70edee9d11523aaa62b1a4ca1e4bc75dec0344c627b26cf30631`
- Added versus the small archive: `compare/`, the YOLO face-detector weight,
  and both sealed fixed-96 validation protocol directories.

Only the bulk `dataset_full/` content outside those protocol directories and
the self-referential `release/` directory are excluded. Dataset manifests,
notebooks, and media alone are hundreds of megabytes, so they cannot fit the
requested limit.
