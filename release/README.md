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
