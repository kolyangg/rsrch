Snapshot contents:
- conda_explicit.txt        # exact conda package URLs/builds (best for same OS/arch)
- environment_nobuilds.yml  # portable conda spec
- pip_freeze.txt            # pip packages pinned exactly

Recommended restore command:
  ./setup/recreate_env_from_snapshot.sh <snapshot_dir> <new_env_name>
