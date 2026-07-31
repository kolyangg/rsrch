# Cosmic Large experiment summaries and launch scripts

**Date:** 25 July 2026

**Branch:** `test`

**Status:** code-ready locally; not committed, pushed, deployed, or launched.

For analysis and rationale, see [Cosmic Large training analysis, adaptation
plan, and prepared
experiments](2026-07-25_cosmic_large_training_recommendations_and_experiments.md).

## Experiment list

| Order | Run name | Summary | Machine | Launch selector |
|---:|---|---|---|---|
| 1 | `rhca_cosmic_oneid_margin40_4k` | Widen Task D crop context from 20% to 40% per side at fixed 256px output | Neb | `one_id margin40` |
| 2 | `rhca_cosmic_full_crop20_legacy_4k` | Clean full-manifest CA-off baseline with Task D's winning 20%/256px reference policy | Serv 1 GPU | prepared YAML |
| 3 | `rhca_cosmic_full_crop20_posefirst_4k` | Isolate pose-first, 55-word captions at the same reference policy | Serv 1 GPU | prepared YAML |
| 4 | `rhca_cosmic_oneid_canvas1024_4k` | Put exact 256px Task D crop on a neutral 1024px canvas to isolate spatial occupancy | Neb, conditional | `one_id canvas1024` |
| 5 | `rhca_cosmic_full_canvas1024_posefirst_4k` | Transfer a successful canvas intervention to full Cosmic | Serv 1 GPU, conditional | prepared YAML |
| 6 | `rhca_cosmic_full_crop20_posefirst_20k` | Long-run stability of a winning 4k policy | Serv 1 GPU, gated | prepared YAML |

The JSON plan for each run is under
[`experiments/cosmic_large_adaptation/`](../experiments/cosmic_large_adaptation/).
At startup it becomes the canonical live record:

```text
saved/<run_name>/comet_experiment.json
```

The Comet writer automatically fills the immutable experiment ID and URL.

## Deployment prerequisite

The launch scripts validate historical runtime hashes, so the new code must be
committed on `test` and the same commit synced to the target machine before a
run starts.

Confirm locally:

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template
git branch --show-current
git status --short
git diff --check
```

Do not copy `.env`, credentials, or machine-local bbox files through Git.

## Neb: inspect before every launch

Connect and confirm that no GPU process group is active:

```bash
ssh neb
cd /home/niko/rsrch/diffusion_template
git branch --show-current
git rev-parse HEAD
nvidia-smi
ps -eo pid,ppid,pgid,lstart,etime,args \
  | grep -E '[t]rain.py|[a]ccelerate|[t]orchrun'
```

Neb is a one-job machine for these experiments. Do not infer safety from a
temporary dip in memory or utilization; validation can reach about 79.3GB.

### Neb run 1: wider one-ID context

```bash
cd /home/niko/rsrch/diffusion_template
test ! -e saved/rhca_cosmic_oneid_margin40_4k
mkdir -p logs

nohup setsid bash launchers/neb/start_rhca_cosmic_experiment.sh \
  one_id margin40 \
  > logs/rhca_cosmic_oneid_margin40_4k.log 2>&1 < /dev/null &

pid=$!
ps -o pid,pgid,etime,args -p "$pid"
```

Startup checks:

```bash
sleep 10
tail -n 80 logs/rhca_cosmic_oneid_margin40_4k.log
test -s saved/rhca_cosmic_oneid_margin40_4k/comet_experiment.json
python tools/comet/comet_experiment.py show \
  saved/rhca_cosmic_oneid_margin40_4k/comet_experiment.json
nvidia-smi
```

Inspect step 500 and 1,000 against Task D `multi_cosref`. Continue unless both
gates show at least 10/12 catastrophic faces and no visible improvement.

### Neb run 2: one-ID canvas, conditional

Run only if wider-context or fixed-checkpoint evidence keeps spatial occupancy
as a live hypothesis:

```bash
cd /home/niko/rsrch/diffusion_template
test ! -e saved/rhca_cosmic_oneid_canvas1024_4k
mkdir -p logs

nohup setsid bash launchers/neb/start_rhca_cosmic_experiment.sh \
  one_id canvas1024 \
  > logs/rhca_cosmic_oneid_canvas1024_4k.log 2>&1 < /dev/null &

pid=$!
ps -o pid,pgid,etime,args -p "$pid"
```

## Serv: deploy prepared packages

The following local packages already contain a start script, one-GPU MLS
YAML, and the exact source experiment JSON:

```text
diffusion_template/serv_run_packages/rhca_cosmic_full_crop20_legacy_4k/
diffusion_template/serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k/
diffusion_template/serv_run_packages/rhca_cosmic_full_canvas1024_posefirst_4k/
diffusion_template/serv_run_packages/rhca_cosmic_full_crop20_posefirst_20k/
```

After the code commit is available in Serv's `rsrch_test` checkout, deploy a
package with the run builder or copy the exact package using its documented
`--deploy` workflow. Verify:

```bash
python3 local_scripts/serv_job.py check
```

### Serv run 1: clean full-Cosmic baseline

This run may start immediately in parallel with the Neb one-ID control:

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/rhca_cosmic_full_crop20_legacy_4k/run_rhca_cosmic_full_crop20_legacy_4k_1gpu.yaml \
  --comment "Clean full Cosmic 4k CA-off baseline with 20% crop references"
```

Require `64/64` decoded preflight samples, 22,140 accepted records for the
current manifest, a live Comet key, no fatal error, and valid step-500 images.

### Serv run 2: pose-first 4k

Submit after the full legacy baseline passes its startup and step-500 gates:

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k/run_rhca_cosmic_full_crop20_posefirst_4k_1gpu.yaml \
  --comment "Full Cosmic 4k pose-first caption control at fixed 20% crop policy"
```

Record the returned MLS job name. It is required for inspection or stopping:

```bash
python3 local_scripts/serv_job.py status <job_name>
python3 local_scripts/serv_job.py inspect <job_name> --lines 80
```

This Serv run can overlap the Neb baseline after the baseline's step-500 gate;
they are on different machines.

### Serv run 3: full canvas 4k, conditional

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/rhca_cosmic_full_canvas1024_posefirst_4k/run_rhca_cosmic_full_canvas1024_posefirst_4k_1gpu.yaml \
  --comment "Conditional full Cosmic 4k reference-canvas occupancy control"
```

Submit only if the one-ID canvas or fixed-checkpoint matrix improves anatomy.

### Serv run 4: 20k, gated

```bash
python3 local_scripts/serv_job.py submit \
  /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/rhca_cosmic_full_crop20_posefirst_20k/run_rhca_cosmic_full_crop20_posefirst_20k_1gpu.yaml \
  --comment "Gated 20k stability run for the winning full Cosmic policy"
```

Do not submit this job until a 4k arm passes the multi-identity anatomy gate.
If a different 4k policy wins, rebuild the long-run JSON/package instead of
using this name with different settings.

## Comet verification and retrieval

For any running machine, registration is part of startup:

```bash
test -s saved/<run_name>/comet_experiment.json
python tools/comet/comet_experiment.py show \
  saved/<run_name>/comet_experiment.json
```

Retrieve Neb results by the recorded immutable ID:

```bash
python tools/comet/comet_experiment.py fetch \
  --host neb \
  --run-name <run_name> \
  --step-number 4000
```

Retrieve Serv results:

```bash
python tools/comet/comet_experiment.py fetch \
  --host serv \
  --remote-project /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template \
  --remote-python /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS/bin/python \
  --run-name <run_name> \
  --step-number 4000
```

Accept an export only when its requested step, image count, warnings, and
per-asset records are correct. Do not locate experiments by display name.

## Stop rules

On Neb, identify the exact process group and terminate the group, not an
unverified child PID:

```bash
ps -eo pid,ppid,pgid,etime,args \
  | grep -E '[t]rain.py|[a]ccelerate|[t]orchrun'
kill -TERM -- -<verified_pgid>
```

On Serv:

```bash
python3 local_scripts/serv_job.py kill <job_name>
```

Stop and diagnose before scheduling a successor on OOM, fatal error, invalid
preflight, missing Comet registration, corrupted bbox/mask output, or a failed
reproduction/control gate.

## Expected order and parallelism

```text
Neb:
  margin40 one-ID
    -> canvas one-ID only if justified

Serv:
  crop20 legacy full baseline
    -> crop20 pose-first 4k after the baseline step-500 gate
    -> canvas full 4k only if justified
    -> 20k only after the final 4k promotion gate
```

Only cross-machine overlap is approved. No same-GPU overlap is safe for the
current validation path.
