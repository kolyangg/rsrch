# Dynamic no-idle subject-v2 recovery

This wave replaces the fixed-chain eight-GPU job after three of its allocated
workers became idle while work remained. It reuses the shared transactional
staging root and presents all unfinished runs as one NFS-claimed queue:

1. CL6, CL12, CL13, CL14, E14, E15;
2. E16, E17, E18, E19, E20, E21, E22.

Eight workers claim one run at a time. As soon as a worker completes and
hash-verifies a run in its original Comet ID, it claims the next unclaimed run.
Already verified runs return immediately. Complete checkpoint manifests are
hash-validated and reused; interrupted steps are preserved under
`incomplete_recovery/` and regenerated.

Every worker fails startup unless it resolves the Nasilaev `photomaker_NS`
interpreter and `CONDA_PREFIX`. CLIP scoring uses the checksum-verified shared
cache rather than worker-local downloads.
