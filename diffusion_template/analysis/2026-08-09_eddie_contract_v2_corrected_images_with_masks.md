---
title: "Eddie contract-v2 corrected validation images with masks"
subtitle: "E13 24k, BC_E13 24k, and CL11 20k - 36 corrected outputs"
date: "9 August 2026"
status: "FINAL: guarded Serv replay passed 36/36 historical pixel gates"
---

This visual appendix contains every corrected Eddie output from the guarded
contract-v2 Serv validation. Red is the immutable target-face mask. Cyan is the
detected generated face selected by maximum overlap with that mask. Each tile
shows the complete generated image, a larger face crop, intended-Eddie ArcFace
similarity, mask IoU, and detected-face count.

Before these corrected arms ran, E13, BC_E13 and CL11 each reproduced their 12
historical Eddie images RGB pixel-exact. The correction changes the ArcFace
vector fused into global PhotoMaker identity tokens; it does not change the BA
spatial bbox, prompt, seed, RealVis base, scheduler, 50 steps, CFG 5, fixed
generation mask, or checkpoint.

\newpage

# E13 24k - prompts 1-6

![](assets/problematic_validation_20260809/corrected_eddie_masks_e13_24k_part1.png)

\newpage

# E13 24k - prompts 7-12

![](assets/problematic_validation_20260809/corrected_eddie_masks_e13_24k_part2.png)

\newpage

# BC_E13 24k - prompts 1-6

![](assets/problematic_validation_20260809/corrected_eddie_masks_bc_e13_24k_part1.png)

\newpage

# BC_E13 24k - prompts 7-12

![](assets/problematic_validation_20260809/corrected_eddie_masks_bc_e13_24k_part2.png)

\newpage

# CL11 20k - prompts 1-6

![](assets/problematic_validation_20260809/corrected_eddie_masks_cl11_20k_part1.png)

\newpage

# CL11 20k - prompts 7-12

![](assets/problematic_validation_20260809/corrected_eddie_masks_cl11_20k_part2.png)

# Provenance

- Serv job: `lm-mpi-job-baea4903-7f8d-4785-a67d-f153df3299da`.
- Images and manifests: `analysis/assets/problematic_validation_20260809/`
  `final_checkpoint_sidecar_contract_v2/`.
- E13 checkpoint SHA-256:
  `4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab`.
- BC_E13 checkpoint SHA-256:
  `99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e`.
- CL11 checkpoint SHA-256:
  `e65972c8c14b5031f879e1ee8b1e11a707823e0cfccdb80553219fc8069dbb83`.
