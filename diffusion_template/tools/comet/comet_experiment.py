#!/usr/bin/env python3
# 10 Aug 2026 - E13C-DOC-01: Retained the immutable-key Comet fetcher as
# standalone tooling; it is not imported by training or generation code.
"""Manage per-run Comet records and retrieve metrics/images by experiment key."""

from __future__ import annotations

import argparse
import base64
import json
import os
import posixpath
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXPORT_SCRIPT = SCRIPT_DIR / "export_comet_runs.py"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "comet_records"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_REMOTE_PROJECT = "/home/niko/rsrch/diffusion_template"
SERV_REMOTE_ENV = (
    "/mnt/virtual_ai0001053-01309_SR006-nfs1/"
    "nasilaev/conda_env/photomaker_NS"
)
SERV_REMOTE_REPO = (
    "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test"
)
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")
EXPERIMENT_KEY_RE = re.compile(r"^[A-Za-z0-9]{32}$")


class RecordError(ValueError):
    """Raised when a Comet experiment record is missing or invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_name(run_name: str) -> str:
    if not RUN_NAME_RE.fullmatch(run_name):
        raise RecordError(
            "run name may contain only letters, digits, periods, underscores, and hyphens"
        )
    return run_name


def validate_host(host: str) -> str:
    if not HOST_RE.fullmatch(host):
        raise RecordError(
            "SSH host may contain only letters, digits, periods, underscores, and hyphens"
        )
    return host


def remote_shell_command(host: str, script: str) -> str:
    """Wrap a remote command with the host's required connection prelude."""
    if host == "serv":
        script = f"""\
conda activate {shlex.quote(SERV_REMOTE_ENV)}
cd {shlex.quote(SERV_REMOTE_REPO)}
if [[ "${{CONDA_PREFIX:-}}" != {shlex.quote(SERV_REMOTE_ENV)} ]]; then
  echo "ERROR: wrong Serv Conda environment" >&2
  exit 70
fi
{script}
"""
        # AICODE-NOTE: Serv exposes `conda activate` through its interactive
        # shell initialization; every SSH hop must also select rsrch_test.
        return f"bash -ic {shlex.quote(script)}"
    return f"bash -lc {shlex.quote(script)}"


def default_cache_path(run_name: str) -> Path:
    return DEFAULT_CACHE_DIR / f"{validate_run_name(run_name)}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".comet-record-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def validate_record(value: Any, *, expected_run_name: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecordError("record must be a JSON object")
    if value.get("schema_version") != 1:
        raise RecordError("record schema_version must be 1")

    run_name = str(value.get("run_name", "")).strip()
    validate_run_name(run_name)
    if expected_run_name and run_name != expected_run_name:
        raise RecordError(
            f"record run_name {run_name!r} does not match {expected_run_name!r}"
        )

    comet = value.get("comet")
    if not isinstance(comet, dict):
        raise RecordError("record is missing the 'comet' object")
    experiment_key = str(comet.get("experiment_key", "")).strip()
    if not EXPERIMENT_KEY_RE.fullmatch(experiment_key):
        raise RecordError("record has an invalid Comet experiment_key")
    project_name = str(comet.get("project_name", "")).strip()
    if not project_name:
        raise RecordError("record has no Comet project_name")
    return value


def load_record(path: Path, *, expected_run_name: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RecordError(f"record does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordError(f"record is not valid JSON: {path}") from exc
    return validate_record(value, expected_run_name=expected_run_name)


def load_env_file(path: Path, environment: dict[str, str]) -> None:
    if not path.is_file():
        raise RecordError(f"environment file does not exist: {path}")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise RecordError(f"invalid assignment in {path}:{line_number}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        environment.setdefault(key, value)


def remote_record_path(remote_project: str, run_name: str) -> str:
    normalized_project = posixpath.normpath(remote_project)
    if not posixpath.isabs(normalized_project):
        raise RecordError("remote project directory must be absolute")
    return posixpath.join(
        normalized_project,
        "saved",
        validate_run_name(run_name),
        "comet_experiment.json",
    )


def pull_remote_record(
    *,
    host: str,
    remote_project: str,
    run_name: str,
    destination: Path,
) -> tuple[Path, dict[str, Any]]:
    host = validate_host(host)
    path = remote_record_path(remote_project, run_name)
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        host,
        remote_shell_command(
            host,
            f"set -euo pipefail\ncat -- {shlex.quote(path)}",
        ),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"ssh exited {result.returncode}"
        raise RecordError(f"failed to read {host}:{path}: {detail}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecordError(f"{host}:{path} did not contain valid JSON") from exc
    record = validate_record(value, expected_run_name=run_name)
    write_json_atomic(destination, record)
    return destination, record


def make_backfill_record(args: argparse.Namespace) -> int:
    run_name = validate_run_name(args.run_name)
    experiment_key = args.experiment_key.strip()
    if not EXPERIMENT_KEY_RE.fullmatch(experiment_key):
        raise RecordError("experiment key must be a 32-character Comet key")
    now = utc_now()
    workspace = args.workspace or None
    url = (
        f"https://www.comet.com/{workspace}/{args.project_name}/{experiment_key}"
        if workspace
        else None
    )
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at_utc": now,
        "updated_at_utc": now,
        "source": "manual_backfill",
        "runtime": {
            "hostname": args.runtime_host or socket.gethostname(),
            "pid": None,
            "save_dir": args.save_dir,
        },
        "git": {
            "branch": args.git_branch,
            "commit": args.git_commit,
        },
        "comet": {
            "experiment_key": experiment_key,
            "project_name": args.project_name,
            "workspace": workspace,
            "url": url,
            "mode": "online",
        },
    }
    destination = (args.record or default_cache_path(run_name)).resolve()
    write_json_atomic(destination, payload)
    print(f"Wrote Comet experiment record: {destination}")
    print(f"Comet experiment key: {experiment_key}")
    return 0


def pull_record(args: argparse.Namespace) -> int:
    run_name = validate_run_name(args.run_name)
    destination = (args.record or default_cache_path(run_name)).resolve()
    path, record = pull_remote_record(
        host=args.host,
        remote_project=args.remote_project,
        run_name=run_name,
        destination=destination,
    )
    print(f"Cached Comet experiment record: {path}")
    print(f"Comet experiment key: {record['comet']['experiment_key']}")
    return 0


def resolve_fetch_record(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.record:
        path = args.record.resolve()
        return path, load_record(path, expected_run_name=args.run_name)
    if not args.run_name:
        raise RecordError("fetch requires --record or --run-name")

    run_name = validate_run_name(args.run_name)
    if args.host:
        destination = default_cache_path(run_name).resolve()
        return pull_remote_record(
            host=args.host,
            remote_project=args.remote_project,
            run_name=run_name,
            destination=destination,
        )

    candidates = [
        PROJECT_ROOT / "saved" / run_name / "comet_experiment.json",
        default_cache_path(run_name),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), load_record(
                candidate, expected_run_name=run_name
            )
    raise RecordError(
        f"no record found for {run_name}; use --host for a remote run or --record"
    )


def fetch_experiment(args: argparse.Namespace) -> int:
    record_path, record = resolve_fetch_record(args)
    run_name = record["run_name"]
    output_dir = (
        args.output_dir or (PROJECT_ROOT / "comet_data" / run_name)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    if args.env_file.is_file():
        load_env_file(args.env_file.resolve(), environment)
    elif not args.host:
        raise RecordError(f"environment file does not exist: {args.env_file}")
    if environment.get("COMET_API_KEY"):
        return fetch_with_local_api(
            args=args,
            record_path=record_path,
            record=record,
            output_dir=output_dir,
            environment=environment,
        )
    if args.host:
        return fetch_with_remote_api(
            args=args,
            record_path=record_path,
            record=record,
            output_dir=output_dir,
        )
    raise RecordError(
        f"COMET_API_KEY is absent from the environment and {args.env_file}; "
        "pass --host to use that machine's .env without copying its credential"
    )


def export_manifest(record: dict[str, Any], step_number: int) -> dict[str, Any]:
    return {
        "runs": [
            {
                "run_id": record["comet"]["experiment_key"],
                "run_name": record["run_name"],
                "step_number": step_number,
            }
        ]
    }


def fetch_with_local_api(
    *,
    args: argparse.Namespace,
    record_path: Path,
    record: dict[str, Any],
    output_dir: Path,
    environment: dict[str, str],
) -> int:
    manifest = export_manifest(record, args.step_number)
    with tempfile.TemporaryDirectory(prefix="comet-fetch-") as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--step-number",
            str(args.step_number),
        ]
        if args.keep_run_dir:
            command.append("--keep-run-dir")
        result = subprocess.run(command, env=environment, check=False)

    print(f"Record: {record_path}")
    print(f"Comet experiment key: {record['comet']['experiment_key']}")
    print(f"Export JSON: {output_dir / 'comet_runs_export.json'}")
    return result.returncode


def fetch_with_remote_api(
    *,
    args: argparse.Namespace,
    record_path: Path,
    record: dict[str, Any],
    output_dir: Path,
) -> int:
    host = validate_host(args.host)
    remote_project = posixpath.normpath(args.remote_project)
    if not posixpath.isabs(remote_project):
        raise RecordError("remote project directory must be absolute")
    remote_env_file = args.remote_env_file or posixpath.join(
        remote_project, ".env"
    )
    if not posixpath.isabs(remote_env_file):
        raise RecordError("remote environment file must be absolute")

    manifest_bytes = (
        json.dumps(export_manifest(record, args.step_number), indent=2) + "\n"
    ).encode("utf-8")
    manifest_b64 = base64.b64encode(manifest_bytes).decode("ascii")
    keep_flag = " --keep-run-dir" if args.keep_run_dir else ""
    remote_script = f"""\
set -euo pipefail
cd {shlex.quote(remote_project)}
set -a
source {shlex.quote(remote_env_file)}
set +a
if [[ -z "${{COMET_API_KEY:-}}" ]]; then
  echo "ERROR: COMET_API_KEY is missing from {remote_env_file}" >&2
  exit 81
fi
remote_temp="$(mktemp -d /tmp/comet-fetch-XXXXXXXX)"
printf 'COMET_REMOTE_EXPORT_DIR=%s\\n' "$remote_temp"
printf %s {shlex.quote(manifest_b64)} | base64 -d > "$remote_temp/manifest.json"
{shlex.quote(args.remote_python)} {shlex.quote(posixpath.join(remote_project, "tools/comet/export_comet_runs.py"))} \
  --manifest "$remote_temp/manifest.json" \
  --output-dir "$remote_temp/output" \
  --step-number {args.step_number}{keep_flag}
"""
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        host,
        remote_shell_command(host, remote_script),
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(
            result.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    marker = re.search(
        r"^COMET_REMOTE_EXPORT_DIR=(/tmp/comet-fetch-[A-Za-z0-9]+)$",
        result.stdout,
        flags=re.MULTILINE,
    )
    if marker is None:
        raise RecordError(
            f"remote Comet export on {host} did not report its temporary directory"
        )

    remote_temp = marker.group(1)
    cleanup_command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        host,
        remote_shell_command(
            host,
            f"set -euo pipefail\nrm -rf -- {shlex.quote(remote_temp)}",
        ),
    ]
    if result.returncode != 0:
        subprocess.run(cleanup_command, check=False)
        raise RecordError(
            f"remote Comet export failed on {host} with exit code {result.returncode}"
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        producer = subprocess.Popen(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                host,
                remote_shell_command(
                    host,
                    f"tar -C {shlex.quote(f'{remote_temp}/output')} "
                    "-cf - .",
                ),
            ],
            stdout=subprocess.PIPE,
        )
        if producer.stdout is None:
            producer.kill()
            raise RecordError("failed to open the remote export stream")
        copy_result = subprocess.run(
            ["tar", "-C", str(output_dir), "-xf", "-"],
            stdin=producer.stdout,
            check=False,
        )
        producer.stdout.close()
        producer_return_code = producer.wait()
        if copy_result.returncode != 0 or producer_return_code != 0:
            raise RecordError(
                f"failed to copy Comet export from {host} "
                f"(ssh={producer_return_code}, tar={copy_result.returncode})"
            )
    finally:
        # The target is accepted only from the strict /tmp marker above.
        subprocess.run(cleanup_command, check=False)

    print(f"Record: {record_path}")
    print(f"Comet experiment key: {record['comet']['experiment_key']}")
    print(f"Export JSON: {output_dir / 'comet_runs_export.json'}")
    return 0


def show_record(args: argparse.Namespace) -> int:
    record = load_record(args.record.resolve())
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage the JSON record written by CometMLWriter and export a run's "
            "metrics/images using its immutable Comet experiment key."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser(
        "backfill",
        help="create a record for a run started before automatic records existed",
    )
    backfill.add_argument("--run-name", required=True)
    backfill.add_argument("--experiment-key", required=True)
    backfill.add_argument("--project-name", required=True)
    backfill.add_argument("--workspace")
    backfill.add_argument("--runtime-host")
    backfill.add_argument("--save-dir")
    backfill.add_argument("--git-branch")
    backfill.add_argument("--git-commit")
    backfill.add_argument("--record", type=Path)
    backfill.set_defaults(func=make_backfill_record)

    pull = subparsers.add_parser(
        "pull",
        help="copy and validate a run's record from an SSH host",
    )
    pull.add_argument("--run-name", required=True)
    pull.add_argument("--host", default="neb")
    pull.add_argument("--remote-project", default=DEFAULT_REMOTE_PROJECT)
    pull.add_argument("--record", type=Path)
    pull.set_defaults(func=pull_record)

    fetch = subparsers.add_parser(
        "fetch",
        help="retrieve Comet metrics and images using a local or remote run record",
    )
    fetch.add_argument("--record", type=Path)
    fetch.add_argument("--run-name")
    fetch.add_argument("--host")
    fetch.add_argument("--remote-project", default=DEFAULT_REMOTE_PROJECT)
    fetch.add_argument("--step-number", type=int, required=True)
    fetch.add_argument("--output-dir", type=Path)
    fetch.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    fetch.add_argument(
        "--remote-env-file",
        help="absolute remote .env path; defaults to <remote-project>/.env",
    )
    fetch.add_argument(
        "--remote-python",
        default="python3",
        help="Python executable on --host. Default: python3",
    )
    fetch.add_argument("--keep-run-dir", action="store_true")
    fetch.set_defaults(func=fetch_experiment)

    show = subparsers.add_parser("show", help="validate and print a record")
    show.add_argument("record", type=Path)
    show.set_defaults(func=show_record)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "step_number", 1) < 0:
        parser.error("--step-number must be non-negative")
    try:
        return int(args.func(args))
    except (OSError, RecordError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
