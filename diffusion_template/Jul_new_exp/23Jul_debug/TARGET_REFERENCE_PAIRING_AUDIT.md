# Target/reference pairing audit — 23Jul experiments

Date: 2026-07-23 UTC

## Verdict

| Runs | Pairing | Validity |
|---|---|---|
| E00–E04 | Target `image_path`; reference sampled from a different `face_paths` file | Valid |
| E14 clean retry | Same loader and pairing as E04 | Valid |
| E15–E16 | Target copied as its own pixel-identical reference | **Invalid leakage audit only** |
| E17 | Same-image follow-up | Canceled before launch |
| E18–E19 | Reference sampled from every subset index except the target index | Valid; both training runs completed with zero preflight violations |

## Evidence

E15/E16 resolved to `src.datasets.cosmic.OneIDTrain` with
`train_on_separate_image=false`. The executed branch is:

```python
instance_data["face_bbox_ref"] = deepcopy(bbox)
ref_images = [deepcopy(img)]
```

An actual Hydra-instantiated audit sampled all eight records. Every target and
reference had the same filename, exact pixels (`max_abs=0`), and the same bbox:

```text
83→83, 109→109, 38→38, 57→57,
104→104, 36→36, 1→1, 116→116
```

The E00–E04 saved configs instantiate the newer `CosmicLargeTrain`. Its
constructor discards the legacy `train_on_separate_image` option, and
`__getitem__` independently loads:

```python
img = self._load_train_image(img_data["image_path"], img_data)
ref_image, ref_bbox = self.get_ref_image(img_data)
```

`get_ref_image` samples a path from `img_data["face_paths"]`. Seeded live
samples confirmed different source paths:

```text
target .../1017318003459.jpg → reference .../6.jpg
target .../1017318003459.jpg → reference .../5.jpg
target .../1017318003459.jpg → reference .../2.jpg
```

## Corrected guardrail

E18/E19 use dataset profile
`one_id_nm0005092_subset8_distinct`, which resolves to
`train_on_separate_image=true`. `OneIDTrain` then constructs the reference
candidate set as every index except the target.

The reusable CPU check is:

```bash
/home/niko/miniconda3/envs/photomaker_NS/bin/python \
  audit_one_id_pairing.py --seeds-per-target 16
```

The 128-pair preflight passed with zero violations. `launch_training.py` now
runs an eight-seed-per-target version automatically for every distinct-profile
launch, stores the result in `run_manifest.json`, and refuses to start if any
same-image pair is observed.

## Interpretation correction

E15/E16 visual and metric artifacts remain useful only to quantify the
optimistic effect of leakage. Their prior promotion conclusion is withdrawn.
Corrected E18 shows that the apparent improvement does not survive distinct
target/reference pairing: at step 200 it develops the same duplicated landmark
failure as the valid Cosmic-loader arms. By step 600 geometry largely recovers,
but median reference similarity falls from `0.3434` at step zero to `0.2931`,
and reference gain versus PhotoMaker falls from `+0.0689` to `-0.0175`.
Therefore there is currently no evidence that OneIDTrain or its preprocessing
improves identity generalization. E19 is the loss-only follow-up.
