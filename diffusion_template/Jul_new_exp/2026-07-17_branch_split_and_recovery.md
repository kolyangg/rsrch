# Branch split and recovery contract

Date: 17 July 2026

## Why the repository was split

The experiments after N3a accumulated increasingly large changes to the active
attention contract. The most important transition was the N28 family:

- `ba_sa_mode: standard` stopped installing `BranchedAttnProcessor`;
- the full spatial reference U-Net half was removed;
- `BranchedCrossAttnProcessor` became a compact target-face residual;
- later runs added layer allowlists, gates, alternative identity memory,
  PhotoMaker-context attenuation, and new epsilon-composition rules.

Those experiments produced useful evidence, but they no longer represented the
original branched-attention proposal: a doubled `[target, reference]` U-Net with
branched self-attention and split cross-attention. The complete later code was
therefore preserved on `main_clean_exp`, while `main_clean` was restored to the
runnable N3a behavior.

## Recovery points

| Reference | Commit | Meaning |
|---|---|---|
| `main_clean_exp` | `99f72cb80e0d7f9f1c5b72b1d92fd6294cfc7be6` | Complete repository immediately before the reset |
| `main_clean_before_n3a_restore_2026-07-17` | `99f72cb80e0d7f9f1c5b72b1d92fd6294cfc7be6` | Immutable annotated safety tag |
| `2157eada...` | `2157eada14824d14019e80f9416e6d736c837306` | Original spatial processor/runtime topology reference |
| `e42c966...` | `e42c96604ee73b8b073b3def268beead8c8af684` | Runnable N3a baseline used by `main_clean` |

`2157eada` is not the complete N3a run snapshot. It predates the N3a launcher,
per-branch optimizer grouping, unconditional face-prompt fix, and the critical
post-validation processor reattachment. The core processor topology is the same,
but `e42c966` is the correct runnable baseline.

## Behavioral restore manifest

The following active paths on `main_clean` are restored exactly from
`e42c966`:

```text
diffusion_template/src/**
diffusion_template/train.py
diffusion_template/infer.py
diffusion_template/serv_new_runs/**
diffusion_template/tests/**
```

This broad manifest is intentional. Post-N3a behavior had spread beyond
`src/model` into pipeline composition, trainer/loss plumbing, datasets, Hydra
configs, checkpoint handling, and the training entry point. Restoring only the
processor files would create a hybrid state that never produced N3a.

The following categories remain from the newer repository because they do not
define active model behavior:

- `debug_04Jul` reports, images, and the HTML architecture explorer;
- Comet full-validation download tools and JSON manifests;
- PDF/report-generation utilities;
- downloaded result metadata and validation artifacts;
- selected post-N3a launcher/config examples archived under `Jul_new_exp`.

## How to return to the pre-reset code

The simplest recovery is:

```bash
cd /home/kolyangg/rsrch
git switch main_clean_exp
```

To create another branch from the immutable snapshot:

```bash
git switch -c restore_pre_n3a_reset \
  main_clean_before_n3a_restore_2026-07-17
```

To inspect exactly what the reset changed:

```bash
git diff main_clean_before_n3a_restore_2026-07-17..main_clean
```

No history was rewritten and no force-push is required.

## Baseline verification command

On the restoration commit, this command should produce no diff:

```bash
git diff --exit-code e42c966 -- \
  diffusion_template/src \
  diffusion_template/train.py \
  diffusion_template/infer.py \
  diffusion_template/serv_new_runs \
  diffusion_template/tests
```

## Branch rules going forward

- `main_clean`: preserve the full spatial branched-attention contract. Any new
  architecture behavior must be opt-in, minimal, documented, and approved.
- `main_clean_exp`: retain and, if desired, continue the compact-residual and
  identity-owner research line.
- Do not copy post-N3a Hydra configs back into active `src/configs` on
  `main_clean`.
- Do not treat archived scripts as runnable. Their exact compatible code is on
  `main_clean_exp`.
- Before NN1 implementation, record the intended change in a dated file under
  `Jul_new_exp` and prove that BA-off or the default toggle reproduces N3a.

