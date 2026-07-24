# External model baselines

This directory contains comparison implementations and their shared evaluation
harness. The primary research code is in `../diffusion_template`.

## Contents

- `PhotoMaker/` — standalone PhotoMaker checkout and historical experiments.
- `PuLID/` — standalone PuLID checkout.
- `persongen/` — PersonaGen code and legacy metric scripts.
- `compare/` — shared prompts, references, masks, and comparison scripts.
- `pm_requirements.txt` — extra PhotoMaker environment requirements.
- `pl_requirements.txt` — extra PuLID environment requirements.
- `setup_pulid_NS3.sh` — historical PhotoMaker/PuLID integration helper.

## Environment setup

From this directory:

```bash
pip install -r pm_requirements.txt
pip install -r pl_requirements.txt
bash setup_pulid_NS3.sh
```

The setup script now resolves this directory automatically, so it can also be
called from the repository root:

```bash
bash _other_models/setup_pulid_NS3.sh
```

The older command collection is retained in
[`README_legacy.md`](README_legacy.md). Prefer the paths documented here.
