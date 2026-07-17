# Archived post-N3a launcher and config examples

Date: 17 July 2026

The `archived_post_n3a_examples` directory contains representative launchers and
leaf Hydra configs for N31-N38.

These files are documentation only:

- they are intentionally outside `serv_new_runs` and `src/configs`;
- they are not compatible with the N3a active code on `main_clean`;
- parent configs and shared implementation are not duplicated here;
- the exact runnable implementation is preserved on `main_clean_exp`.

Use them to inspect run-level overrides or to support the HTML architecture
explorer. Do not launch them from `main_clean`.

To inspect the compatible shared code:

```bash
git show main_clean_exp:diffusion_template/src/model/photomaker_branched/branched_runtime.py
git show main_clean_exp:diffusion_template/src/model/photomaker_branched/attn_processor_cleanest.py
git show main_clean_exp:diffusion_template/src/model/photomaker_branched/lora2_helpers.py
```

