#!/usr/bin/env python3
"""Selectively repair subject-v2 checkpoint validation in an existing Comet run.

The workflow is fail-closed and transactional at the job level:

1. resolve one checkpoint or every complete saved checkpoint;
2. download the exact historical fixed-96 panel from the immutable Comet key;
3. replay the original validation batch and require RGB pixel equality;
4. regenerate only rows whose reference identity selection changed;
5. merge those rows into the other 96 historical images and rescore the whole
   panel, including mask-owned ID similarity; and
6. after every selected checkpoint is staged, optionally replace the exact
   images/tables and reconstruct affected Comet metric histories.

Dry-run staging is the default.  Comet mutation requires ``--write``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import hydra
from hydra.utils import instantiate
import pandas as pd
from PIL import Image
import requests
import torch


# 09 Aug 2026 - Allow an immutable historical run tree to supply its exact
# model/config implementation while this backfill tool is delivered as an overlay.
PROJECT_ROOT = Path(
    os.environ.get("PM_BACKFILL_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_subject_selector import bbox_iou
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim
from tools.comet.comet_experiment import load_env_file, load_record
from tools.comet.export_comet_runs import (
    CometAPIError,
    CometRestClient,
    normalized_export_image_file_name,
)
from tools.inference.evaluate_rhca_checkpoint import (
    checkpoint_state,
    load_config,
    output_filename,
    resolve_generation_bboxes,
    sha256_file,
)


CORE_METRICS = (
    "manual_val/id_sim",
    "manual_val/text_sim",
    "face_quality/face_detection_rate",
    "face_quality/topiq_face_mean",
    "face_quality/topiq_face_p10",
    "face_quality/topiq_face_coverage",
    "face_quality/topiq_mean",
    "face_quality/musiq_mean",
    "face_quality/maniqa_mean",
)
SUBJECT_V2_METRICS = (
    "manual_val/id_sim_legacy_best",
    "manual_val/id_sim_mask_iou",
    "manual_val/id_sim_face_count",
    "manual_val/id_sim_no_face",
    "manual_val/id_sim_unowned",
    "manual_val/id_sim_ambiguous",
)
ALL_REPLACED_METRICS = CORE_METRICS + SUBJECT_V2_METRICS
CHECKPOINT_RE = re.compile(r"^(?:weights|checkpoint)-epoch(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selective subject-v2 validation backfill with exact replay gates."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--staging-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", type=Path, action="append")
    selection.add_argument("--all-safe-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        action="append",
        help="Required only when a selected checkpoint filename has no epoch number.",
    )
    parser.add_argument(
        "--affected-identities",
        default="auto",
        help="Comma-separated IDs, or auto from the subject-v2 embedding manifest.",
    )
    parser.add_argument(
        "--affected-prompts",
        default="",
        help="Optional comma-separated exact prompts or prompt prefixes.",
    )
    parser.add_argument(
        "--subject-manifest",
        type=Path,
        default=Path("../dataset_full/val_dataset/id_embeds_manual_val_subject_v2.json"),
    )
    parser.add_argument(
        "--legacy-id-embeddings",
        type=Path,
        default=Path("../dataset_full/val_dataset/id_embeds_manual_val.pth"),
    )
    parser.add_argument(
        "--subject-v2-id-embeddings",
        type=Path,
        default=Path("../dataset_full/val_dataset/id_embeds_manual_val_subject_v2.pth"),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("tools/inference/evaluate_rhca_checkpoint.py"),
    )
    parser.add_argument("--validation-dataset", default="manual_val")
    parser.add_argument(
        "--generation-bbox-map",
        type=Path,
        help=(
            "Exact active generation-bbox JSON for the historical run. The "
            "pixel replay gate still verifies that it is the right protocol."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--face-quality-device", default="cuda")
    parser.add_argument("--face-quality-batch-size", type=int, default=8)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default=os.getenv("COMET_API_KEY"))
    parser.add_argument("--base-url", default="https://www.comet.com")
    parser.add_argument("--initial-seconds-per-checkpoint", type=float, default=1500.0)
    parser.add_argument("--reuse-staging", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify-attempts", type=int, default=30)
    parser.add_argument("--verify-delay", type=float, default=10.0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_job_eta(
    *,
    step: int,
    index: int,
    total: int,
    current_remaining: float,
    seconds_per_checkpoint: float,
) -> None:
    whole_remaining = current_remaining + max(0, total - index - 1) * seconds_per_checkpoint
    print(
        "BACKFILL_ETA "
        f"checkpoint_step={step} checkpoint={index + 1}/{total} "
        f"current_remaining={format_eta(current_remaining)} "
        f"job_remaining={format_eta(whole_remaining)}",
        flush=True,
    )


class BackfillEtaReporter:
    """Emit rolling current-checkpoint and whole-job ETAs at each batch."""

    def __init__(self, total: int, initial_seconds: float) -> None:
        self.total = int(total)
        self.initial_seconds = float(initial_seconds)
        self.completed_durations: list[float] = []
        self.step = 0
        self.index = 0
        self.started = 0.0

    def start_checkpoint(self, *, step: int, index: int) -> None:
        self.step = int(step)
        self.index = int(index)
        self.started = time.monotonic()
        self.update(0.0)

    def update(self, fraction: float) -> None:
        elapsed = max(0.0, time.monotonic() - self.started)
        fraction = min(max(float(fraction), 0.0), 1.0)
        historical = (
            sum(self.completed_durations) / len(self.completed_durations)
            if self.completed_durations
            else self.initial_seconds
        )
        if fraction > 0.0:
            projected = elapsed / fraction
            estimate = 0.5 * historical + 0.5 * projected
            current_remaining = max(0.0, projected - elapsed)
        else:
            estimate = historical
            current_remaining = historical
        print_job_eta(
            step=self.step,
            index=self.index,
            total=self.total,
            current_remaining=current_remaining,
            seconds_per_checkpoint=estimate,
        )

    def finish_checkpoint(self, *, record_duration: bool = True) -> None:
        duration = max(0.0, time.monotonic() - self.started)
        if record_duration:
            self.completed_durations.append(duration)
        seconds_per_checkpoint = (
            sum(self.completed_durations) / len(self.completed_durations)
            if self.completed_durations
            else self.initial_seconds
        )
        print_job_eta(
            step=self.step,
            index=self.index,
            total=self.total,
            current_remaining=0.0,
            seconds_per_checkpoint=seconds_per_checkpoint,
        )


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return str(args.api_key)
    environment = dict(os.environ)
    load_env_file(args.env_file.resolve(), environment)
    api_key = environment.get("COMET_API_KEY")
    if not api_key:
        raise ValueError("COMET_API_KEY is missing")
    return api_key


def discover_checkpoints(args: argparse.Namespace, config) -> list[tuple[int, Path]]:
    epoch_len = int(config.trainer.epoch_len)
    selected: list[Path]
    if args.all_safe_checkpoints:
        by_epoch: dict[int, Path] = {}
        # Weights-only files are faster to load and contain the same schema-v2
        # trainable state; full checkpoints are the fallback.
        for prefix in ("checkpoint", "weights"):
            for path in sorted(args.run_dir.glob(f"{prefix}-epoch*.pth")):
                match = CHECKPOINT_RE.fullmatch(path.name)
                if match:
                    by_epoch[int(match.group(1))] = path
        selected = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    else:
        selected = [path.resolve() for path in args.checkpoint or []]
    if not selected:
        raise ValueError("No checkpoint files were selected")

    explicit_steps = list(args.checkpoint_step or [])
    if explicit_steps and len(explicit_steps) != len(selected):
        raise ValueError("--checkpoint-step count must match --checkpoint count")
    resolved = []
    for position, path in enumerate(selected):
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"Missing or truncated checkpoint: {path}")
        match = CHECKPOINT_RE.fullmatch(path.name)
        step = (
            explicit_steps[position]
            if explicit_steps
            else int(match.group(1)) * epoch_len if match else None
        )
        if step is None or step <= 0:
            raise ValueError(f"Cannot derive a positive validation step from {path}")
        # A full torch deserialization is the safe-checkpoint gate.  Do not
        # publish from a filename/size-only candidate.
        _state, metadata = checkpoint_state(path)
        if metadata["schema_version"] not in {1, 2}:
            raise ValueError(f"Unsupported checkpoint schema: {path}")
        del _state
        resolved.append((int(step), path.resolve()))
    if [step for step, _ in resolved] != sorted(set(step for step, _ in resolved)):
        raise ValueError("Checkpoint steps must be unique and increasing")
    return resolved


def affected_identity_set(args: argparse.Namespace) -> set[str]:
    manifest = json.loads(args.subject_manifest.read_text(encoding="utf-8"))
    expected_subject_sha = manifest.get("embedding_sha256")
    expected_legacy_sha = manifest.get("legacy_embedding_sha256")
    if expected_subject_sha != sha256_file(args.subject_v2_id_embeddings):
        raise RuntimeError(
            "Subject-v2 embedding file does not match its selector manifest"
        )
    if expected_legacy_sha != sha256_file(args.legacy_id_embeddings):
        raise RuntimeError("Legacy embedding file does not match the selector manifest")
    comparison = manifest.get("legacy_comparison", {})
    changed = {
        str(identity).lower()
        for identity, record in comparison.items()
        if float(record.get("legacy_cosine", 1.0)) < 0.999999
    }
    if not changed:
        raise RuntimeError("Subject manifest identifies no changed identities")
    if args.affected_identities != "auto":
        result = {
            value.strip().lower()
            for value in args.affected_identities.split(",")
            if value.strip()
        }
        if not result:
            raise ValueError("--affected-identities resolved to an empty set")
        unexpected = result - changed
        if unexpected:
            raise ValueError(
                "Requested identities are unchanged in the subject manifest: "
                f"{sorted(unexpected)}"
            )
        return result
    return changed


def prompt_matches(prompt: str, filters: list[str]) -> bool:
    normalized = prompt.strip().lower()
    return not filters or any(
        normalized == value or normalized.startswith(value) for value in filters
    )


def panel_contract(
    config,
    dataset_name: str,
    identity_ids: set[str],
    prompt_filters: list[str],
    *,
    generation_bbox_map: Path | None,
):
    dataset = instantiate(config.datasets.val[dataset_name])
    if len(dataset) != 96:
        raise ValueError(f"Subject-v2 backfill requires fixed-96, found {len(dataset)}")
    samples = [dict(dataset[index]) for index in range(len(dataset))]
    keys = [
        output_filename(str(sample["prompt"]), str(sample["id"]))
        for sample in samples
    ]
    prompts = [str(sample["prompt"]) for sample in samples]
    person_ids = [str(sample["id"]) for sample in samples]
    generation_bboxes, bbox_audit = resolve_generation_bboxes(
        config,
        samples,
        prompts,
        person_ids,
        validation_dataset=dataset_name,
        active_map_override=generation_bbox_map,
    )
    if any(bbox is None for bbox in generation_bboxes):
        raise RuntimeError("Subject-v2 scoring requires all 96 resolved generation boxes")
    for sample, bbox in zip(samples, generation_bboxes):
        sample["face_bbox_gen"] = bbox
    if len(keys) != len(set(keys)):
        raise RuntimeError("Fixed-panel output filenames are not unique")
    affected = [
        index
        for index, sample in enumerate(samples)
        if str(sample["id"]).lower() in identity_ids
        and prompt_matches(str(sample["prompt"]), prompt_filters)
    ]
    if not affected:
        raise ValueError("No fixed-panel rows match the affected identity/prompt filters")
    dataloader_config = config.dataloaders.get(dataset_name) or config.dataloaders.val_default
    batch_size = int(dataloader_config.batch_size)
    expanded = sorted(
        {
            candidate
            for index in affected
            for candidate in range(
                (index // batch_size) * batch_size,
                min(((index // batch_size) + 1) * batch_size, len(samples)),
            )
        }
    )
    return dataset, samples, keys, affected, expanded, batch_size, bbox_audit


def retry_comet_read(label: str, operation):
    """Retry bounded transient failures for idempotent Comet reads/downloads."""
    attempts = int(os.environ.get("PM_COMET_READ_ATTEMPTS", "8"))
    initial_delay = float(os.environ.get("PM_COMET_READ_RETRY_SECONDS", "5"))
    transient_tokens = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "bad gateway",
        "connection",
        "temporarily unavailable",
        "timed out",
        "timeout",
    )
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (requests.RequestException, CometAPIError) as exc:
            message = str(exc).lower()
            transient = isinstance(
                exc, (requests.Timeout, requests.ConnectionError)
            ) or any(token in message for token in transient_tokens)
            if not transient or attempt == attempts:
                raise
            delay = min(60.0, initial_delay * (2 ** (attempt - 1)))
            print(
                "COMET_READ_RETRY "
                f"label={label} attempt={attempt}/{attempts} "
                f"delay_seconds={delay:g} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def live_assets(client: CometRestClient, key: str) -> list[dict[str, Any]]:
    payload = retry_comet_read(
        f"asset-list:{key}",
        lambda: client.get_json(
            "/experiment/asset/list", experimentKey=key, type="all"
        ),
    )
    return payload.get("assets", [])


def download_asset_atomic(
    client: CometRestClient,
    experiment_key: str,
    asset_id: str,
    destination: Path,
) -> dict[str, Any]:
    temporary = destination.with_name(destination.name + ".download")

    def operation():
        temporary.unlink(missing_ok=True)
        result = client.download_asset(experiment_key, asset_id, temporary)
        temporary.replace(destination)
        return result

    return retry_comet_read(f"asset:{experiment_key}:{asset_id}", operation)


def assets_for_panel(
    assets: list[dict[str, Any]], step: int, expected_names: list[str]
) -> dict[str, dict[str, Any]]:
    expected = set(expected_names)
    result = {}
    for asset in assets:
        if str(asset.get("type")) != "image" or asset.get("step") is None:
            continue
        if int(asset["step"]) != step:
            continue
        # 09 Aug 2026 - Comet drops the PNG suffix and appends nondeterministic
        # duplicate counters such as " (8)". Match the trainer's stable output
        # key through the same normalization used by the export/report tools.
        name = normalized_export_image_file_name(
            str(asset.get("fileName") or "")
        )
        if name not in expected:
            continue
        if name in result:
            raise RuntimeError(f"Duplicate Comet image {name} at step {step}")
        result[name] = asset
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise RuntimeError(
            f"Comet step {step} has {len(result)}/96 canonical images; missing={missing[:5]}"
        )
    return result


def retain_safe_comet_panels(
    checkpoints: list[tuple[int, Path]],
    assets: list[dict[str, Any]],
    expected_names: list[str],
) -> list[tuple[int, Path]]:
    """Keep deserializable checkpoints that also own one complete Comet panel."""
    expected = set(expected_names)
    names_by_step: dict[int, list[str]] = defaultdict(list)
    for asset in assets:
        if str(asset.get("type")) != "image" or asset.get("step") is None:
            continue
        name = normalized_export_image_file_name(
            str(asset.get("fileName") or "")
        )
        if name in expected:
            names_by_step[int(asset["step"])].append(name)
    safe = []
    for step, checkpoint in checkpoints:
        names = names_by_step.get(step, [])
        if len(names) == len(expected) and set(names) == expected:
            safe.append((step, checkpoint))
        else:
            print(
                "SUBJECT_V2_SKIP_UNSAFE_CHECKPOINT "
                f"step={step} checkpoint={checkpoint} "
                f"canonical_comet_images={len(names)}/96",
                flush=True,
            )
    if not safe:
        raise RuntimeError("No safe checkpoint has one complete fixed-96 Comet panel")
    return safe


def download_panel(
    client: CometRestClient,
    experiment_key: str,
    assets: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, asset in sorted(assets.items()):
        path = destination / name
        if not path.is_file():
            download_asset_atomic(
                client, experiment_key, str(asset["assetId"]), path
            )
        with Image.open(path) as image:
            image.verify()


def evaluator_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    step: int,
    output_dir: Path,
    indices: list[int],
    embedding_policy: str,
) -> list[str]:
    command = [
        sys.executable,
        str(args.evaluator.resolve()),
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--validation-dataset",
        args.validation_dataset,
        "--sample-indices",
        ",".join(map(str, indices)),
        "--checkpoint-step",
        str(step),
        "--reference-id-embedding-policy",
        embedding_policy,
        "--device",
        args.device,
        "--skip-metrics",
    ]
    if args.generation_bbox_map is not None:
        command.extend(
            ["--generation-bbox-map", str(args.generation_bbox_map.resolve())]
        )
    return command


def run_evaluator(
    command: list[str],
    log_path: Path,
    *,
    progress=None,
    progress_base: float = 0.0,
    progress_span: float = 0.0,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            match = re.search(r"VALIDATION_ETA completed=(\d+)/(\d+)", line)
            if match and progress is not None:
                completed, total = (int(value) for value in match.groups())
                progress(
                    progress_base
                    + progress_span * (completed / max(total, 1))
                )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Evaluator failed with exit code {return_code}: {log_path}")


def rows_by_dataset_index(output_dir: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads((output_dir / "per_image.json").read_text(encoding="utf-8"))
    result = {int(row["dataset_index"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate evaluator dataset indices in {output_dir}")
    return result


def require_exact_replay(
    replay_dir: Path,
    original_dir: Path,
    expected_keys: list[str],
    expanded_indices: list[int],
) -> dict[str, Any]:
    rows = rows_by_dataset_index(replay_dir)
    mismatches = []
    for index in expanded_indices:
        replay_path = replay_dir / "images" / rows[index]["filename"]
        original_path = original_dir / expected_keys[index]
        with Image.open(replay_path) as opened:
            replay = opened.convert("RGB")
        with Image.open(original_path) as opened:
            original = opened.convert("RGB")
        if replay.size != original.size or replay.tobytes() != original.tobytes():
            mismatches.append(index)
    audit = {
        "expected_count": len(expanded_indices),
        "exact_count": len(expanded_indices) - len(mismatches),
        "mismatch_indices": mismatches,
    }
    if mismatches:
        raise RuntimeError(f"Historical replay gate failed for indices {mismatches}")
    return audit


def merge_corrected_panel(
    *,
    corrected_dir: Path,
    original_dir: Path,
    merged_dir: Path,
    expected_keys: list[str],
    affected_indices: list[int],
) -> dict[str, Any]:
    corrected_rows = rows_by_dataset_index(corrected_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    affected_set = set(affected_indices)
    changed = []
    for index, name in enumerate(expected_keys):
        source = original_dir / name
        if index in affected_set:
            source = corrected_dir / "images" / corrected_rows[index]["filename"]
            with Image.open(source) as opened:
                corrected = opened.convert("RGB")
            with Image.open(original_dir / name) as opened:
                original = opened.convert("RGB")
            if corrected.size != original.size or corrected.tobytes() != original.tobytes():
                changed.append(index)
        shutil.copy2(source, merged_dir / name)
    if not changed:
        raise RuntimeError("Subject-v2 regeneration changed no affected image pixels")
    return {
        "affected_indices": affected_indices,
        "pixel_changed_indices": changed,
        "replaced_output_keys": [expected_keys[index] for index in affected_indices],
    }


def score_identity_and_text(
    *,
    samples: list[dict],
    keys: list[str],
    panel_dir: Path,
    legacy_embeddings_path: Path,
    subject_embeddings_path: Path,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    import clip

    legacy_embeddings = torch.load(legacy_embeddings_path, map_location="cpu")
    subject_embeddings = torch.load(subject_embeddings_path, map_location="cpu")
    aligner = Aligner()
    # 10 Aug 2026 - AICODE-NOTE: MLS worker-local HOME caches are ephemeral and
    # can retain a truncated 891 MB CLIP download. Production backfills pin a
    # checksum-verified shared cache so metric scoring never downloads per node.
    text_model, text_preprocess = clip.load(
        "ViT-L/14@336px",
        device=device,
        download_root=os.environ.get("CLIP_CACHE_DIR"),
    )
    text_model.eval()
    rows = []
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        images = []
        for key in keys[start:end]:
            with Image.open(panel_dir / key) as opened:
                images.append(opened.convert("RGB"))
        boxes_batch, embeds_batch = aligner(images)
        image_tensor = torch.stack(
            [text_preprocess(image) for image in images]
        ).to(device)
        prompt_tokens = clip.tokenize(
            [str(sample["prompt"]) for sample in samples[start:end]],
            truncate=True,
        ).to(device)
        with torch.no_grad():
            logits_per_image, _ = text_model(image_tensor, prompt_tokens)
            text_scores = logits_per_image.diagonal().detach().float().cpu().tolist()

        for local_index, (sample, key, boxes_raw, embeds_raw, text_score) in enumerate(
            zip(
                samples[start:end],
                keys[start:end],
                boxes_batch,
                embeds_batch,
                text_scores,
            )
        ):
            index = start + local_index
            boxes = boxes_raw or []
            embeds = embeds_raw or []
            identity = str(sample["id"])
            legacy_score = 0.0
            if embeds:
                legacy_score = max(
                    float(cos_sim(embed, legacy_embeddings[identity]))
                    for embed in embeds
                )
            target_bbox = sample.get("face_bbox_gen")
            if not isinstance(target_bbox, (list, tuple)) or len(target_bbox) != 4:
                raise RuntimeError(f"Missing fixed generation bbox at index {index}")
            ranked = sorted(
                (
                    (bbox_iou(box, target_bbox), face_index, embed)
                    for face_index, (box, embed) in enumerate(zip(boxes, embeds))
                ),
                key=lambda item: (-item[0], item[1]),
            )
            best_iou = float(ranked[0][0]) if ranked else 0.0
            unowned = not ranked or best_iou < 0.05
            ambiguous = bool(
                len(ranked) > 1
                and ranked[1][0] >= 0.05
                and abs(ranked[0][0] - ranked[1][0]) <= 0.02
            )
            subject_score = (
                0.0
                if unowned
                else float(cos_sim(ranked[0][2], subject_embeddings[identity]))
            )
            rows.append(
                {
                    "validation_step": None,
                    "partition": "manual_val",
                    "image_index": index,
                    "output_key": key,
                    "identity": identity,
                    "prompt": str(sample["prompt"]),
                    "seed": int(sample.get("seed", 0)),
                    "generated_image_count": 1,
                    "id_sim": subject_score,
                    "id_sim_legacy_best": legacy_score,
                    "id_sim_mask_iou": best_iou,
                    "id_sim_face_count": len(boxes),
                    "id_sim_no_face": int(not boxes),
                    "id_sim_unowned": int(unowned),
                    "id_sim_ambiguous": int(ambiguous),
                    "text_sim": float(text_score),
                }
            )
    text_model.to("cpu")
    aggregate = {
        "manual_val/id_sim": float(pd.Series([row["id_sim"] for row in rows]).mean()),
        "manual_val/id_sim_legacy_best": float(
            pd.Series([row["id_sim_legacy_best"] for row in rows]).mean()
        ),
        "manual_val/id_sim_mask_iou": float(
            pd.Series([row["id_sim_mask_iou"] for row in rows]).mean()
        ),
        "manual_val/id_sim_face_count": float(
            pd.Series([row["id_sim_face_count"] for row in rows]).mean()
        ),
        "manual_val/id_sim_no_face": float(
            pd.Series([row["id_sim_no_face"] for row in rows]).mean()
        ),
        "manual_val/id_sim_unowned": float(
            pd.Series([row["id_sim_unowned"] for row in rows]).mean()
        ),
        "manual_val/id_sim_ambiguous": float(
            pd.Series([row["id_sim_ambiguous"] for row in rows]).mean()
        ),
        "manual_val/text_sim": float(pd.Series([row["text_sim"] for row in rows]).mean()),
    }
    return rows, aggregate


def run_face_quality(
    args: argparse.Namespace,
    *,
    step: int,
    keys: list[str],
    panel_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, float]]:
    manifest_path = output_dir / "face_quality_input_manifest.json"
    output_json = output_dir / "face_quality_metrics.json"
    output_csv = output_dir / "face_quality_per_image.csv"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "subject_v2_merged_fixed96",
            "experiment_key": None,
            "project_name": "subject_v2_backfill_staging",
            "steps": {
                str(step): [
                    {
                        "asset_id": f"local-{index:03d}",
                        "file_name": key,
                        "local_path": str((panel_dir / key).resolve()),
                    }
                    for index, key in enumerate(keys)
                ]
            },
        },
    )
    command = [
        sys.executable,
        "tools/inference/calculate_face_quality_metrics.py",
        "--manifest",
        str(manifest_path),
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--metrics",
        "topiq_nr-face,topiq_nr,musiq,maniqa-pipal",
        "--device",
        args.face_quality_device,
        "--batch-size",
        str(args.face_quality_batch_size),
        "--crop-padding",
        "0.25",
        "--crop-size",
        "512",
    ]
    subprocess.run(command, check=True)
    payload = json.loads(output_json.read_text(encoding="utf-8"))["steps"][str(step)]
    metrics = payload["metrics"]
    scalars = {
        "face_quality/face_detection_rate": float(payload["face_detection_rate"]),
        "face_quality/topiq_face_mean": float(metrics["topiq_nr_face"]["mean"]),
        "face_quality/topiq_face_p10": float(metrics["topiq_nr_face"]["p10"]),
        "face_quality/topiq_face_coverage": float(metrics["topiq_nr_face"]["count"]) / 96.0,
        "face_quality/topiq_mean": float(metrics["topiq_nr"]["mean"]),
        "face_quality/musiq_mean": float(metrics["musiq"]["mean"]),
        "face_quality/maniqa_mean": float(metrics["maniqa_pipal"]["mean"]),
    }
    return output_json, output_csv, scalars


def stage_checkpoint(
    args: argparse.Namespace,
    *,
    client: CometRestClient,
    experiment_key: str,
    config,
    samples: list[dict],
    keys: list[str],
    affected_indices: list[int],
    expanded_indices: list[int],
    checkpoint: Path,
    step: int,
    step_root: Path,
    progress=None,
) -> dict[str, Any]:
    manifest_path = step_root / "step_manifest.json"
    if args.reuse_staging and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "staged":
            raise RuntimeError(f"Reusable staging is not complete: {manifest_path}")
        if int(payload.get("checkpoint_step", -1)) != step:
            raise RuntimeError(f"Reusable staging step mismatch: {manifest_path}")
        if payload.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError(f"Reusable staging checkpoint changed: {checkpoint}")
        if payload.get("expanded_generation_indices") != expanded_indices:
            raise RuntimeError("Reusable staging generation indices changed")
        if payload.get("merge", {}).get("affected_indices") != affected_indices:
            raise RuntimeError("Reusable staging affected indices changed")
        staged_hashes = [
            (Path(payload["id_table"]), payload["id_table_sha256"]),
            (Path(payload["face_quality_csv"]), payload["face_quality_csv_sha256"]),
            *[
                (Path(image["path"]), image["sha256"])
                for image in payload.get("changed_images", [])
            ],
        ]
        for staged_path, expected_sha in staged_hashes:
            if not staged_path.is_file() or sha256_file(staged_path) != expected_sha:
                raise RuntimeError(
                    f"Reusable staging artifact changed: {staged_path}"
                )
        return payload
    if step_root.exists() and any(step_root.iterdir()):
        if not args.reuse_staging:
            raise FileExistsError(
                f"Refusing partial staging without --reuse-staging: {step_root}"
            )
        # 10 Aug 2026 - AICODE-NOTE: a transient Comet failure may leave a
        # partially downloaded step. Preserve it for audit, but never mix its
        # files into a resumed exact-replay stage.
        quarantine_root = args.staging_root / "incomplete_recovery"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = quarantine_root / f"{step_root.name}_{suffix}_{os.getpid()}"
        step_root.replace(quarantine)
        print(
            f"BACKFILL_PARTIAL_STAGE_QUARANTINED source={step_root} "
            f"destination={quarantine}",
            flush=True,
        )
    step_root.mkdir(parents=True, exist_ok=True)
    assets = assets_for_panel(live_assets(client, experiment_key), step, keys)
    if progress is not None:
        progress(0.02)
    original_dir = step_root / "original_panel"
    download_panel(client, experiment_key, assets, original_dir)
    if progress is not None:
        progress(0.08)

    replay_dir = step_root / "historical_replay"
    run_evaluator(
        evaluator_command(
            args,
            checkpoint=checkpoint,
            step=step,
            output_dir=replay_dir,
            indices=expanded_indices,
            embedding_policy="legacy_first",
        ),
        step_root / "historical_replay.log",
        progress=progress,
        progress_base=0.08,
        progress_span=0.34,
    )
    replay_audit = require_exact_replay(
        replay_dir, original_dir, keys, expanded_indices
    )
    if progress is not None:
        progress(0.43)

    corrected_dir = step_root / "subject_v2_generation"
    run_evaluator(
        evaluator_command(
            args,
            checkpoint=checkpoint,
            step=step,
            output_dir=corrected_dir,
            indices=expanded_indices,
            embedding_policy="bbox_overlap_v2",
        ),
        step_root / "subject_v2_generation.log",
        progress=progress,
        progress_base=0.43,
        progress_span=0.34,
    )
    merged_dir = step_root / "merged_panel"
    merge_audit = merge_corrected_panel(
        corrected_dir=corrected_dir,
        original_dir=original_dir,
        merged_dir=merged_dir,
        expected_keys=keys,
        affected_indices=affected_indices,
    )
    if progress is not None:
        progress(0.79)

    rows, aggregate = score_identity_and_text(
        samples=samples,
        keys=keys,
        panel_dir=merged_dir,
        legacy_embeddings_path=args.legacy_id_embeddings,
        subject_embeddings_path=args.subject_v2_id_embeddings,
        device=args.device,
        batch_size=args.metric_batch_size,
    )
    for row in rows:
        row["validation_step"] = step
    if progress is not None:
        progress(0.87)
    id_table = step_root / f"id_sim__manual_val__step_{step:06d}.csv"
    pd.DataFrame(rows).drop(columns=["text_sim"]).to_csv(id_table, index=False)
    quality_json, quality_csv, quality_scalars = run_face_quality(
        args,
        step=step,
        keys=keys,
        panel_dir=merged_dir,
        output_dir=step_root,
    )
    aggregate.update(quality_scalars)
    if progress is not None:
        progress(0.99)
    if set(aggregate) != set(ALL_REPLACED_METRICS):
        raise RuntimeError(f"Unexpected aggregate metric set: {sorted(aggregate)}")
    write_json(step_root / "aggregate_metrics.json", aggregate)

    payload = {
        "schema_version": 2,
        "kind": "subject_v2_selective_checkpoint_validation",
        "status": "staged",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "historical_replay": replay_audit,
        "merge": merge_audit,
        "panel_image_count": len(keys),
        "expanded_generation_indices": expanded_indices,
        "metrics": aggregate,
        "id_table": str(id_table.resolve()),
        "id_table_sha256": sha256_file(id_table),
        "face_quality_json": str(quality_json.resolve()),
        "face_quality_csv": str(quality_csv.resolve()),
        "face_quality_csv_sha256": sha256_file(quality_csv),
        "merged_panel": str(merged_dir.resolve()),
        "changed_images": [
            {
                "file_name": keys[index],
                "path": str((merged_dir / keys[index]).resolve()),
                "sha256": sha256_file(merged_dir / keys[index]),
            }
            for index in affected_indices
        ],
    }
    write_json(manifest_path, payload)
    return payload


class MutationClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = api_key

    def delete_metric(self, experiment_key: str, metric_name: str) -> None:
        response = self.session.post(
            f"{self.base_url}/api/rest/v2/write/experiment/metric/delete",
            json={"experimentKey": experiment_key, "metricName": metric_name},
            timeout=120,
        )
        response.raise_for_status()


def metric_history(
    client: CometRestClient, experiment_key: str, metric_name: str
) -> list[dict[str, Any]]:
    payload = retry_comet_read(
        f"metric:{experiment_key}:{metric_name}",
        lambda: client.get_json(
            "/experiment/metrics/get-metric",
            experimentKey=experiment_key,
            metricName=metric_name,
        ),
    )
    return [entry for entry in payload.get("metrics", []) if entry.get("step") is not None]


def history_value_map(history: list[dict[str, Any]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for entry in history:
        grouped[int(entry["step"])].append(float(entry["metricValue"]))
    duplicates = {step: values for step, values in grouped.items() if len(values) != 1}
    if duplicates:
        raise RuntimeError(f"Metric history has duplicate steps: {list(duplicates)[:5]}")
    return {step: values[0] for step, values in grouped.items()}


def exact_assets_for_replacement(
    assets: list[dict[str, Any]], staged_steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    image_targets = {
        (int(stage["checkpoint_step"]), image["file_name"])
        for stage in staged_steps
        for image in stage["changed_images"]
    }
    other_names = {
        Path(stage["id_table"]).name for stage in staged_steps
    } | {
        f"face_quality_details__manual_val__step_{int(stage['checkpoint_step']):08d}.csv"
        for stage in staged_steps
    }
    selected = []
    for asset in assets:
        filename = str(asset.get("fileName") or "")
        raw_step = asset.get("step")
        step = None if raw_step is None else int(raw_step)
        if str(asset.get("type")) == "image" and (
            step,
            normalized_export_image_file_name(filename),
        ) in image_targets:
            selected.append(asset)
        elif filename in other_names:
            selected.append(asset)
    return selected


def expected_uploaded_assets(
    staged_steps: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, str]]:
    expected: dict[tuple[int, str], dict[str, str]] = {}
    for stage in staged_steps:
        step = int(stage["checkpoint_step"])
        for image in stage["changed_images"]:
            expected[(step, image["file_name"])] = {
                "path": image["path"],
                "sha256": image["sha256"],
                "kind": "image",
            }
        expected[(step, Path(stage["id_table"]).name)] = {
            "path": stage["id_table"],
            "sha256": stage["id_table_sha256"],
            "kind": "table",
        }
        quality_name = f"face_quality_details__manual_val__step_{step:08d}.csv"
        expected[(step, quality_name)] = {
            "path": stage["face_quality_csv"],
            "sha256": stage["face_quality_csv_sha256"],
            "kind": "table",
        }
    return expected


def publish(
    args: argparse.Namespace,
    *,
    api_key: str,
    client: CometRestClient,
    record: dict[str, Any],
    staged_steps: list[dict[str, Any]],
) -> None:
    from comet_ml import API

    experiment_key = record["comet"]["experiment_key"]
    live_metadata = client.get_json(
        "/experiment/metadata", experimentKey=experiment_key
    )
    if str(live_metadata.get("experimentName") or "") != record["run_name"]:
        raise RuntimeError("Live Comet experiment name does not match the immutable record")
    if str(live_metadata.get("projectName") or "") != record["comet"]["project_name"]:
        raise RuntimeError("Live Comet project does not match the immutable record")

    steps = [int(stage["checkpoint_step"]) for stage in staged_steps]
    existing_histories = {
        name: metric_history(client, experiment_key, name)
        for name in ALL_REPLACED_METRICS
    }
    replacement_histories = {}
    for name, history in existing_histories.items():
        if name == "manual_val/id_sim_legacy_best" and not history:
            # The original canonical ID series is exactly the legacy
            # max-over-detections definition. Seed the audit curve from it,
            # then rescore corrected panels at the selected steps.
            values = history_value_map(existing_histories["manual_val/id_sim"])
        else:
            values = history_value_map(history)
        for stage in staged_steps:
            values[int(stage["checkpoint_step"])] = float(stage["metrics"][name])
        replacement_histories[name] = values

    current_assets = live_assets(client, experiment_key)
    target_assets = exact_assets_for_replacement(current_assets, staged_steps)
    expected_image_deletes = sum(len(stage["changed_images"]) for stage in staged_steps)
    actual_image_deletes = sum(str(asset.get("type")) == "image" for asset in target_assets)
    if actual_image_deletes != expected_image_deletes:
        raise RuntimeError(
            f"Expected {expected_image_deletes} changed Comet images, found {actual_image_deletes}"
        )

    backup_root = args.staging_root / "comet_backup"
    backup_root.mkdir(parents=True, exist_ok=True)
    for asset in target_assets:
        destination = backup_root / str(asset["assetId"])
        if not destination.is_file():
            download_asset_atomic(
                client, experiment_key, str(asset["assetId"]), destination
            )
    backup = {
        "schema_version": 1,
        "kind": "subject_v2_comet_before_replacement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_key": experiment_key,
        "metadata": live_metadata,
        "metric_histories": existing_histories,
        "target_assets": target_assets,
        "download_root": str(backup_root.resolve()),
    }
    backup_path = args.staging_root / "comet_before_replacement.json"
    if backup_path.is_file():
        previous = json.loads(backup_path.read_text(encoding="utf-8"))
        if previous.get("experiment_key") != experiment_key:
            raise RuntimeError("Existing backup belongs to another Comet experiment")
        previous_ids = {
            str(asset["assetId"]) for asset in previous.get("target_assets", [])
        }
        current_ids = {str(asset["assetId"]) for asset in target_assets}
        if previous_ids != current_ids:
            raise RuntimeError(
                "Target Comet assets changed after the staging backup was created"
            )
        previous_histories = previous.get("metric_histories", {})
        for name, history in existing_histories.items():
            if history_value_map(previous_histories.get(name, [])) != history_value_map(history):
                raise RuntimeError(
                    f"Comet metric history changed after staging: {name}"
                )
    else:
        write_json(backup_path, backup)

    print(
        "SUBJECT_V2_COMET_PREFLIGHT_OK "
        f"key={experiment_key} steps={','.join(map(str, steps))} "
        f"changed_images={expected_image_deletes}",
        flush=True,
    )
    if not args.write:
        print("SUBJECT_V2_COMET_DRY_RUN_COMPLETE")
        return

    api = API(api_key=api_key, cache=False)
    experiment = api.get_experiment_by_key(experiment_key)
    if experiment is None:
        raise RuntimeError("Comet APIExperiment lookup failed")
    for asset in target_assets:
        experiment.delete_asset(str(asset["assetId"]))
    mutator = MutationClient(api_key, args.base_url)
    deleted_metric_names = []
    for name in ALL_REPLACED_METRICS:
        if existing_histories[name]:
            mutator.delete_metric(experiment_key, name)
            deleted_metric_names.append(name)

    # Comet deletes are eventually consistent. Logging replacements before the
    # old points/assets disappear can create duplicate steps or duplicate names.
    deleted_asset_ids = {str(asset["assetId"]) for asset in target_assets}
    for attempt in range(args.verify_attempts):
        remaining_ids = {
            str(asset["assetId"])
            for asset in live_assets(client, experiment_key)
            if str(asset.get("assetId")) in deleted_asset_ids
        }
        remaining_metric_points = sum(
            len(metric_history(client, experiment_key, name))
            for name in deleted_metric_names
        )
        if not remaining_ids and remaining_metric_points == 0:
            break
        if attempt + 1 < args.verify_attempts:
            time.sleep(args.verify_delay)
    else:
        raise RuntimeError(
            "Comet deletion did not converge before replacement: "
            f"assets={len(remaining_ids)} metric_points={remaining_metric_points}"
        )

    all_metric_steps = sorted(
        {
            step
            for values in replacement_histories.values()
            for step in values
        }
    )
    for metric_step in all_metric_steps:
        values = {
            name: history[metric_step]
            for name, history in replacement_histories.items()
            if metric_step in history
        }
        experiment.log_metrics(values, step=metric_step)
    for stage in staged_steps:
        step = int(stage["checkpoint_step"])
        for image in stage["changed_images"]:
            if experiment.log_image(
                image["path"],
                image_name=image["file_name"],
                step=step,
                overwrite=False,
                metadata={
                    "schema_version": 2,
                    "kind": "subject_v2_revalidated_image",
                    "sha256": image["sha256"],
                    "validation_step": step,
                },
            ) is None:
                raise RuntimeError(f"Comet rejected image {image['file_name']}@{step}")
        if experiment.log_asset(
            stage["id_table"],
            name=Path(stage["id_table"]).name,
            step=step,
            overwrite=False,
            ftype="dataframe",
            metadata={
                "kind": "subject_v2_per_image_identity",
                "sha256": stage["id_table_sha256"],
                "row_count": 96,
                "validation_step": step,
            },
        ) is None:
            raise RuntimeError(f"Comet rejected subject-v2 ID table at {step}")
        quality_name = f"face_quality_details__manual_val__step_{step:08d}.csv"
        if experiment.log_asset(
            stage["face_quality_csv"],
            name=quality_name,
            step=step,
            overwrite=False,
            metadata={
                "kind": "subject_v2_face_quality_per_image",
                "sha256": stage["face_quality_csv_sha256"],
                "row_count": 96,
                "validation_step": step,
            },
        ) is None:
            raise RuntimeError(f"Comet rejected face-quality table at {step}")

    experiment.log_other(
        "subject_v2_validation_replacement",
        (
            "Reference identity detections were bound to declared reference boxes; "
            "only affected conditioning rows were regenerated after an exact-pixel "
            "historical replay gate. The full fixed-96 panel was rescored with "
            "mask-owned ID similarity; legacy max-over-any-face ID remains in "
            "manual_val/id_sim_legacy_best."
        ),
    )

    expected_assets = expected_uploaded_assets(staged_steps)
    matched_assets: dict[tuple[int, str], dict[str, Any]] = {}
    # Verify exact metric values and exactly one current asset per expected key.
    for attempt in range(args.verify_attempts):
        failures = []
        for name, expected in replacement_histories.items():
            actual = history_value_map(metric_history(client, experiment_key, name))
            if set(actual) != set(expected) or any(
                not math.isclose(actual[step], value, rel_tol=1e-9, abs_tol=1e-12)
                for step, value in expected.items()
            ):
                failures.append(name)
        assets = live_assets(client, experiment_key)
        matched_assets = {}
        for (step, file_name), spec in expected_assets.items():
            matches = [
                asset
                for asset in assets
                if (
                    (
                        spec["kind"] == "image"
                        and normalized_export_image_file_name(
                            str(asset.get("fileName") or "")
                        )
                        == file_name
                    )
                    or (
                        spec["kind"] == "table"
                        and str(asset.get("fileName") or "") == file_name
                    )
                )
                and (
                    (
                        spec["kind"] == "image"
                        and str(asset.get("type")) == "image"
                        and asset.get("step") is not None
                        and int(asset["step"]) == step
                    )
                    or (
                        spec["kind"] == "table"
                        and (
                            asset.get("step") is None
                            or int(asset["step"]) == step
                        )
                    )
                )
            ]
            if len(matches) != 1:
                failures.append(f"{file_name}@{step}")
            else:
                matched_assets[(step, file_name)] = matches[0]
        if not failures:
            break
        if attempt + 1 < args.verify_attempts:
            time.sleep(args.verify_delay)
    else:
        raise RuntimeError(f"Comet post-write verification failed: {failures[:20]}")

    verification_root = args.staging_root / "comet_after_replacement"
    verification_root.mkdir(parents=True, exist_ok=True)
    for asset_key, spec in expected_assets.items():
        asset = matched_assets[asset_key]
        destination = verification_root / str(asset["assetId"])
        download_asset_atomic(
            client, experiment_key, str(asset["assetId"]), destination
        )
        actual_sha = sha256_file(destination)
        if actual_sha != spec["sha256"]:
            raise RuntimeError(
                f"Comet content hash mismatch for {asset_key}: "
                f"expected={spec['sha256']} actual={actual_sha}"
            )

    job_manifest_path = args.staging_root / "job_manifest.json"
    job_manifest = json.loads(job_manifest_path.read_text(encoding="utf-8"))
    job_manifest["status"] = "verified_on_comet"
    job_manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(job_manifest_path, job_manifest)
    audit = {
        "schema_version": 2,
        "kind": "subject_v2_validation_replacement_audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_key": experiment_key,
        "replacement_steps": steps,
        "changed_images": expected_image_deletes,
        "verified_asset_count": len(expected_assets),
        "staging_manifest": str(job_manifest_path.resolve()),
    }
    audit_path = args.staging_root / "replacement_verified.json"
    write_json(audit_path, audit)
    if experiment.log_asset(
        str(audit_path),
        name=f"subject_v2_validation_replacement__{min(steps)}_{max(steps)}.json",
        overwrite=False,
        metadata={"kind": audit["kind"], "sha256": sha256_file(audit_path)},
    ) is None:
        raise RuntimeError("Comet rejected the replacement audit asset")
    print(
        "SUBJECT_V2_COMET_REPLACEMENT_VERIFIED "
        f"key={experiment_key} steps={','.join(map(str, steps))} "
        f"verified_assets={len(expected_assets)}"
    )


def main() -> int:
    args = parse_args()
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    if args.face_quality_batch_size <= 0:
        raise ValueError("--face-quality-batch-size must be positive")
    if args.initial_seconds_per_checkpoint <= 0:
        raise ValueError("--initial-seconds-per-checkpoint must be positive")
    if args.verify_attempts <= 0 or args.verify_delay < 0:
        raise ValueError("Comet verification attempts/delay are invalid")
    args.run_dir = args.run_dir.resolve()
    args.staging_root = args.staging_root.resolve()
    args.subject_manifest = args.subject_manifest.resolve()
    args.legacy_id_embeddings = args.legacy_id_embeddings.resolve()
    args.subject_v2_id_embeddings = args.subject_v2_id_embeddings.resolve()
    if args.generation_bbox_map is not None:
        args.generation_bbox_map = args.generation_bbox_map.resolve()
    if args.experiment_record is None:
        args.experiment_record = args.run_dir / "comet_experiment.json"
    record = load_record(args.experiment_record.resolve())
    if args.run_dir.name != record["run_name"]:
        raise ValueError("Run directory name and immutable Comet record differ")
    api_key = resolve_api_key(args)
    client = CometRestClient(api_key, args.base_url, timeout=120)
    config, config_source = load_config(args.config)
    checkpoints = discover_checkpoints(args, config)
    identity_ids = affected_identity_set(args)
    prompt_filters = [
        value.strip().lower()
        for value in args.affected_prompts.split(",")
        if value.strip()
    ]
    _dataset, samples, keys, affected, expanded, batch_size, bbox_audit = panel_contract(
        config,
        args.validation_dataset,
        identity_ids,
        prompt_filters,
        generation_bbox_map=args.generation_bbox_map,
    )
    if args.all_safe_checkpoints:
        checkpoints = retain_safe_comet_panels(
            checkpoints,
            live_assets(client, record["comet"]["experiment_key"]),
            keys,
        )
    print(
        "SUBJECT_V2_BACKFILL_PLAN "
        f"run={record['run_name']} checkpoints={len(checkpoints)} "
        f"affected_rows={len(affected)} generated_rows={len(expanded)} "
        f"validation_batch_size={batch_size} identities={sorted(identity_ids)}",
        flush=True,
    )

    if args.staging_root.exists() and any(args.staging_root.iterdir()) and not args.reuse_staging:
        raise FileExistsError(
            f"Staging root is non-empty; pass --reuse-staging only after inspection: {args.staging_root}"
        )
    args.staging_root.mkdir(parents=True, exist_ok=True)
    staged = []
    eta = BackfillEtaReporter(
        total=len(checkpoints),
        initial_seconds=args.initial_seconds_per_checkpoint,
    )
    for position, (step, checkpoint) in enumerate(checkpoints):
        reusable_manifest = (
            args.reuse_staging
            and (args.staging_root / f"step_{step:06d}" / "step_manifest.json").is_file()
        )
        eta.start_checkpoint(step=step, index=position)
        staged.append(
            stage_checkpoint(
                args,
                client=client,
                experiment_key=record["comet"]["experiment_key"],
                config=config,
                samples=samples,
                keys=keys,
                affected_indices=affected,
                expanded_indices=expanded,
                checkpoint=checkpoint,
                step=step,
                step_root=args.staging_root / f"step_{step:06d}",
                progress=eta.update,
            )
        )
        eta.finish_checkpoint(record_duration=not reusable_manifest)

    job_manifest = {
        "schema_version": 2,
        "kind": "subject_v2_validation_backfill_job",
        "status": "staged",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": record["run_name"],
        "experiment_key": record["comet"]["experiment_key"],
        "config_source": config_source,
        "affected_identities": sorted(identity_ids),
        "affected_prompts": prompt_filters,
        "affected_indices": affected,
        "expanded_generation_indices": expanded,
        "validation_batch_size": batch_size,
        "generation_bbox_protocol": bbox_audit,
        "steps": [stage["checkpoint_step"] for stage in staged],
        "step_manifests": [
            str((args.staging_root / f"step_{stage['checkpoint_step']:06d}" / "step_manifest.json").resolve())
            for stage in staged
        ],
    }
    write_json(args.staging_root / "job_manifest.json", job_manifest)
    publish(
        args,
        api_key=api_key,
        client=client,
        record=record,
        staged_steps=staged,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
