#!/usr/bin/env bash

# Prepare the canonical per-run JSON before Comet fills its immutable key.
prepare_comet_record() {
  if [[ "$#" -ne 3 ]]; then
    echo "prepare_comet_record requires: <project-root> <run-name> <spec-path>" >&2
    return 2
  fi

  local project_root="$1"
  local run_name="$2"
  local spec_path="$3"
  local run_dir="${project_root}/saved/${run_name}"
  local record_path="${run_dir}/comet_experiment.json"

  if [[ ! -f "${spec_path}" ]]; then
    echo "Experiment spec not found: ${spec_path}" >&2
    return 2
  fi

  python3 - "${spec_path}" "${run_name}" <<'PY'
import json
import sys

path, expected_name = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    spec = json.load(handle)
actual_name = spec.get("run_name")
if actual_name != expected_name:
    raise SystemExit(
        f"Experiment spec run_name mismatch: expected {expected_name!r}, "
        f"found {actual_name!r}"
    )
PY

  if [[ -d "${run_dir}" ]]; then
    local unexpected
    unexpected="$(
      find "${run_dir}" -mindepth 1 -maxdepth 1 \
        ! -name comet_experiment.json -print -quit
    )"
    if [[ -n "${unexpected}" ]]; then
      echo "Refusing to reuse non-empty output directory: ${run_dir}" >&2
      return 3
    fi
    if [[ -f "${record_path}" ]]; then
      python3 - "${record_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    record = json.load(handle)
key = (record.get("comet") or {}).get("experiment_key")
if key:
    raise SystemExit(
        "Refusing to reuse a run whose Comet experiment key is already registered"
    )
PY
    fi
  else
    mkdir -p "${run_dir}"
  fi

  # 25 Jul 2026 - Seed the canonical record rather than a second tracking
  # location. CometMLWriter preserves `plan` and atomically fills `comet`.
  install -m 600 "${spec_path}" "${record_path}"
  printf 'Prepared Comet record: %s\n' "${record_path}"
}
