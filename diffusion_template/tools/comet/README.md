# Comet experiment record and retrieval tool

Use `comet_experiment.py` for every new Comet-tracked training run. It makes
the immutable Comet experiment key the link between a saved run and its
metrics/images.

## Automatic record

`CometMLWriter` writes this file immediately after Comet creates or resumes an
online experiment:

```text
saved/<run_name>/comet_experiment.json
```

The record contains the run name, Comet experiment key, project/workspace,
URL when available, host, PID, Git branch/commit, and save directory. It never
contains the Comet API key.

After starting a new experiment, treat registration as part of the startup
check:

```bash
test -s saved/<run_name>/comet_experiment.json
python tools/comet/comet_experiment.py show \
  saved/<run_name>/comet_experiment.json
```

Do not identify a run later by name alone. Names can be reused; the experiment
key cannot.

For planned Cosmic experiments, the launcher first copies the matching
non-secret JSON from `experiments/cosmic_large_adaptation/` to this canonical
path. The writer preserves its `plan` object and replaces the placeholder
`comet` object with the live immutable key and URL. This gives each run one
JSON containing both intent and observed registration; do not redirect the
record to a second tracked location.

## Retrieve from a remote run

The easiest Neb workflow pulls and validates the remote record, caches it
under the ignored local `comet_records/` directory, and exports the requested
Comet step:

```bash
python tools/comet/comet_experiment.py fetch \
  --host neb \
  --run-name <run_name> \
  --step-number 4000
```

For another machine or checkout:

```bash
python tools/comet/comet_experiment.py fetch \
  --host serv \
  --remote-project /absolute/remote/path/to/diffusion_template \
  --remote-python /absolute/path/to/photomaker_NS/bin/python \
  --run-name <run_name> \
  --step-number 4000
```

The `serv` host profile activates the required absolute Conda environment and
selects `rsrch_test` for every pull/export/copy/cleanup SSH hop.

For a local record:

```bash
python tools/comet/comet_experiment.py fetch \
  --record saved/<run_name>/comet_experiment.json \
  --step-number 4000
```

The command uses `COMET_API_KEY` without printing it. If the local
`diffusion_template/.env` contains the key, export happens locally. If it is
empty and `--host` is supplied, export runs through that host's
`<remote-project>/.env` and only the non-secret results are streamed back.
It then uses `export_comet_runs.py` to download:

- experiment metadata and parameters;
- complete scalar metric histories and summaries;
- image assets at the exact requested step, or the exporter's documented
  nearest-lower fallback;
- `comet_runs_export.json`, including warnings and per-asset records.

Default output is:

```text
comet_data/<run_name>/
```

Both `comet_data/` and the cached `comet_records/` are excluded from Git.

## Backfill an older active run

Runs started before automatic records existed need a one-time exact-key
backfill. Confirm the key through Comet's API/UI and record it explicitly:

```bash
python tools/comet/comet_experiment.py backfill \
  --run-name <run_name> \
  --experiment-key <32-character-key> \
  --project-name <project> \
  --workspace <workspace> \
  --runtime-host <host> \
  --save-dir /absolute/remote/saved/<run_name> \
  --git-branch <branch> \
  --git-commit <commit>
```

Backfill is a migration path, not the normal workflow. Future runs must obtain
the key directly from `CometMLWriter`.

## Existing reporting utilities

- `export_comet_runs.py` exports manifest-listed runs.
- `build_comet_report_pdf.py` builds image/metric comparison reports from an
  export.
- `comet_runs_template.json` documents the lower-level export manifest.
