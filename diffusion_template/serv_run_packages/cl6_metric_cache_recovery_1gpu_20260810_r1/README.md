# CL6 shared-CLIP-cache recovery

The eight-GPU recovery job's CL6 worker generated the corrected 22k images but
failed during metric scoring when OpenAI CLIP downloaded a truncated worker-local
891 MB model. This one-GPU sidecar reuses the ten complete checkpoint manifests,
preserves and regenerates the incomplete 22k stage, and continues through 24k.

The shared model is
`metric_cache/clip/ViT-L-14-336px.pt`, 934,088,680 bytes, SHA-256
`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
The backfill passes that directory explicitly to `clip.load` through
`CLIP_CACHE_DIR`; no per-worker model download is allowed. The shared worker
also fails startup unless both `python` and `CONDA_PREFIX` resolve exactly to
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS`.
