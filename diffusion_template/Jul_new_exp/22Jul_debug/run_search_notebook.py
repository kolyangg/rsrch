#!/usr/bin/env python3
"""Execute the architecture-search notebook as an unattended parameterized job.

All persistent outputs remain below 22Jul_debug. Environment overrides make it
possible to smoke-test and shard the notebook without rewriting its cells.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "NN7_step0_architecture_search.ipynb"


def csv_strings(name: str, default: list[str] | None = None) -> list[str] | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or None


def csv_ints(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


def apply_overrides(namespace: dict) -> None:
    worker = os.environ.get("NN7_WORKER", "worker")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    namespace["RUN_ID"] = os.environ.get("NN7_RUN_ID", f"{timestamp}__{worker}")
    namespace["SAMPLE_INDICES"] = csv_ints(
        "NN7_SAMPLE_INDICES", namespace["SAMPLE_INDICES"]
    )
    requested_experiment_ids = csv_strings(
        "NN7_EXPERIMENT_IDS", namespace["SELECTED_EXPERIMENT_IDS"]
    )
    # Delay filtering until after cell 9 so locally defined custom specs can be
    # included without modifying the production config tree or notebook ladder.
    namespace["_RUNNER_SELECTED_EXPERIMENT_IDS"] = requested_experiment_ids
    namespace["SELECTED_EXPERIMENT_IDS"] = None
    namespace["RUN_PROFILE"] = os.environ.get(
        "NN7_RUN_PROFILE", namespace["RUN_PROFILE"]
    )
    namespace["NUM_INFERENCE_STEPS"] = int(
        os.environ.get("NN7_NUM_INFERENCE_STEPS", namespace["NUM_INFERENCE_STEPS"])
    )
    namespace["PHOTOMAKER_START_STEP"] = int(
        os.environ.get("NN7_PHOTOMAKER_START_STEP", namespace["PHOTOMAKER_START_STEP"])
    )
    namespace["DEFAULT_BA_START_STEP"] = int(
        os.environ.get("NN7_BA_START_STEP", namespace["DEFAULT_BA_START_STEP"])
    )
    namespace["COLLECT_LIGHT_PROCESSOR_DIAGNOSTICS"] = env_bool(
        "NN7_COLLECT_DIAGNOSTICS",
        namespace["COLLECT_LIGHT_PROCESSOR_DIAGNOSTICS"],
    )
    namespace["SHOW_PROGRESS_BARS"] = env_bool(
        "NN7_SHOW_PROGRESS", namespace["SHOW_PROGRESS_BARS"]
    )
    print("Runner overrides:")
    for key in (
        "RUN_ID",
        "SAMPLE_INDICES",
        "_RUNNER_SELECTED_EXPERIMENT_IDS",
        "RUN_PROFILE",
        "NUM_INFERENCE_STEPS",
        "PHOTOMAKER_START_STEP",
        "DEFAULT_BA_START_STEP",
        "COLLECT_LIGHT_PROCESSOR_DIAGNOSTICS",
    ):
        print(f"  {key}={namespace[key]!r}")


def apply_spec_overrides(namespace: dict) -> None:
    custom_path = Path(
        os.environ.get("NN7_CUSTOM_SPECS", HERE / "adaptive_specs.json")
    )
    custom_specs = []
    if custom_path.exists():
        payloads = json.loads(custom_path.read_text(encoding="utf-8"))
        if not isinstance(payloads, list):
            raise TypeError(f"Custom spec file must contain a list: {custom_path}")
        for payload in payloads:
            values = dict(payload)
            if "tags" in values:
                values["tags"] = tuple(values["tags"])
            custom_specs.append(namespace["spec"](**values))

    combined = list(namespace["EXPERIMENT_SPECS"])
    known = {item.experiment_id for item in combined}
    for item in custom_specs:
        if item.experiment_id in known:
            raise KeyError(f"Duplicate custom experiment ID: {item.experiment_id}")
        combined.append(item)
        known.add(item.experiment_id)

    requested = namespace.get("_RUNNER_SELECTED_EXPERIMENT_IDS")
    if requested is not None:
        requested_set = set(requested)
        missing = requested_set - known
        if missing:
            raise KeyError(f"Unknown experiment IDs: {sorted(missing)}")
        combined = [item for item in combined if item.experiment_id in requested_set]
    namespace["EXPERIMENT_SPECS"] = combined
    print(
        "Runner experiment set:",
        [item.experiment_id for item in namespace["EXPERIMENT_SPECS"]],
    )


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
    }
    override_applied = False
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"\n[runner] executing notebook cell {index}", flush=True)
        try:
            exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)
            if index == 1:
                apply_overrides(namespace)
                override_applied = True
            elif index == 9:
                apply_spec_overrides(namespace)
        except BaseException:
            print(f"[runner] cell {index} failed", file=sys.stderr)
            traceback.print_exc()
            return 1
    if not override_applied:
        raise RuntimeError("Notebook settings cell was not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
