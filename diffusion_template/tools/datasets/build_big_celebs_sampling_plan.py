#!/usr/bin/env python3
"""Build reproducible BigCelebs reference scores and training schedules."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import sys
from typing import Iterable

import numpy as np


SCHEMA_VERSION = 1
DIRECTION_RE = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> str:
    expected = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Expected SHA-256 must contain 64 lowercase hex digits")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, found {actual}"
        )
    return actual


def load_manifest(path: Path) -> dict[str, dict[str, dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, dict) or not records:
        raise ValueError(f"Invalid or empty BigCelebs manifest: {path}")
    for identity, images in records.items():
        if not isinstance(images, dict) or not images:
            raise ValueError(f"Invalid image mapping for identity {identity!r}")
    return records


def face_side(metadata: dict) -> float:
    bbox = metadata["new_face_crop"]
    return min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def _selected_pairs(records: dict[str, dict[str, dict]]) -> dict[tuple[str, str], float]:
    return {
        (str(identity), str(image_id)): face_side(metadata)
        for identity, images in records.items()
        for image_id, metadata in images.items()
    }


def _embedding_partition(record: dict, asset_index: Path, line_number: int) -> str:
    if record.get("legacy_reference"):
        return "legacy_original_metadata"
    provenance = str(record.get("host_provenance") or "")
    if provenance == "eqr6":
        return "eqr6_identity_stage"
    if provenance == "neb":
        return "neb_incremental_identity_stage"
    raise ValueError(
        f"Cannot route {record.get('key')!r} to an embedding cache at "
        f"{asset_index}:{line_number}; host_provenance={provenance!r}"
    )


def _join_source_keys(
    asset_index: Path,
    selected: dict[tuple[str, str], float],
) -> dict[str, tuple[tuple[str, str], str]]:
    source_records: dict[str, tuple[tuple[str, str], str]] = {}
    found_pairs: dict[tuple[str, str], str] = {}
    with asset_index.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number % 200000 == 0:
                progress(
                    f"asset-index rows={line_number} matched={len(found_pairs)}"
                )
            if not line.strip():
                continue
            record = json.loads(line)
            pair = (
                str(record.get("published_identity_id")),
                str(record.get("published_item_id")),
            )
            if pair not in selected:
                continue
            source_key = str(record.get("key") or "")
            if not source_key:
                raise ValueError(
                    f"Missing source key in {asset_index}:{line_number}"
                )
            previous = found_pairs.get(pair)
            if previous is not None and previous != source_key:
                raise RuntimeError(
                    f"Published image {pair!r} maps to multiple source keys"
                )
            partition = _embedding_partition(record, asset_index, line_number)
            other_record = source_records.get(source_key)
            if other_record is not None and other_record[0] != pair:
                raise RuntimeError(
                    f"Source key {source_key!r} maps to multiple published images"
                )
            found_pairs[pair] = source_key
            source_records[source_key] = (pair, partition)

    missing = set(selected) - set(found_pairs)
    if missing:
        raise RuntimeError(
            f"Asset index is missing {len(missing)} selected images; "
            f"examples={sorted(missing)[:5]}"
        )
    return source_records


def _embedding_rows(connection: sqlite3.Connection):
    return connection.execute(
        "SELECT image_key, embedding, insightface_detection_count, "
        "matched_iou, detection_size FROM identity_embeddings"
    )


def build_scores(args: argparse.Namespace) -> None:
    manifest_digest = require_sha256(args.manifest, args.expected_manifest_sha256)
    records = load_manifest(args.manifest)
    selected = _selected_pairs(records)
    progress(
        f"selected images={len(selected)} identities={len(records)}; "
        "joining curation keys"
    )
    source_records = _join_source_keys(args.asset_index, selected)
    embedding_dbs = {
        "legacy_original_metadata": args.legacy_embedding_db,
        "eqr6_identity_stage": args.eqr6_embedding_db,
        "neb_incremental_identity_stage": args.neb_incremental_embedding_db,
    }
    expected_partition_counts = Counter(
        partition for _, partition in source_records.values()
    )
    progress(
        f"joined source keys={len(source_records)} partitions="
        f"{dict(sorted(expected_partition_counts.items()))}; "
        "scanning embeddings pass 1/2"
    )

    sums: dict[str, np.ndarray] = {}
    embedding_meta: dict[tuple[str, str], tuple[int, float, int, str]] = {}
    found_sources: set[str] = set()
    for partition, embedding_db in embedding_dbs.items():
        connection = sqlite3.connect(
            f"file:{embedding_db}?mode=ro",
            uri=True,
        )
        scanned_embeddings = 0
        try:
            for (
                source_key,
                blob,
                detection_count,
                matched_iou,
                detection_size,
            ) in _embedding_rows(connection):
                scanned_embeddings += 1
                if scanned_embeddings % 100000 == 0:
                    progress(
                        f"embedding pass 1/2 partition={partition} "
                        f"rows={scanned_embeddings} matched={len(found_sources)}"
                    )
                source_record = source_records.get(str(source_key))
                if source_record is None or source_record[1] != partition:
                    continue
                pair = source_record[0]
                if str(source_key) in found_sources:
                    raise RuntimeError(
                        f"Duplicate authoritative embedding for {source_key!r}"
                    )
                embedding = np.frombuffer(blob, dtype=np.float32)
                if embedding.shape != (512,) or not np.isfinite(embedding).all():
                    raise ValueError(f"Invalid ArcFace embedding for {source_key!r}")
                norm = float(np.linalg.norm(embedding))
                if norm <= 0:
                    raise ValueError(f"Zero ArcFace embedding for {source_key!r}")
                normalized = embedding / norm
                identity = pair[0]
                if identity not in sums:
                    sums[identity] = np.zeros(512, dtype=np.float64)
                sums[identity] += normalized
                embedding_meta[pair] = (
                    int(detection_count),
                    float(matched_iou),
                    int(detection_size),
                    partition,
                )
                found_sources.add(str(source_key))
        finally:
            connection.close()

    missing_sources = set(source_records) - found_sources
    if missing_sources:
        missing_partitions = Counter(
            source_records[source][1] for source in missing_sources
        )
        raise RuntimeError(
            f"Embedding caches are missing {len(missing_sources)} selected images; "
            f"partitions={dict(sorted(missing_partitions.items()))}, "
            f"examples={sorted(missing_sources)[:5]}"
        )

    centroids: dict[str, np.ndarray] = {}
    for identity, total in sums.items():
        norm = float(np.linalg.norm(total))
        if norm <= 0:
            raise ValueError(f"Identity {identity!r} has a zero centroid")
        centroids[identity] = total / norm

    similarities: dict[tuple[str, str], float] = {}
    progress("scanning embeddings pass 2/2")
    for partition, embedding_db in embedding_dbs.items():
        connection = sqlite3.connect(
            f"file:{embedding_db}?mode=ro",
            uri=True,
        )
        scanned_embeddings = 0
        try:
            for source_key, blob, _, _, _ in _embedding_rows(connection):
                scanned_embeddings += 1
                if scanned_embeddings % 100000 == 0:
                    progress(
                        f"embedding pass 2/2 partition={partition} "
                        f"rows={scanned_embeddings} scored={len(similarities)}"
                    )
                source_record = source_records.get(str(source_key))
                if source_record is None or source_record[1] != partition:
                    continue
                pair = source_record[0]
                embedding = np.frombuffer(blob, dtype=np.float32)
                normalized = embedding / float(np.linalg.norm(embedding))
                similarities[pair] = float(
                    np.dot(normalized, centroids[pair[0]])
                )
        finally:
            connection.close()

    if set(similarities) != set(selected):
        raise RuntimeError("Not every selected image received a centroid score")

    source_by_pair = {
        pair: (source, partition)
        for source, (pair, partition) in source_records.items()
    }
    output_rows = []
    for pair in sorted(selected):
        detection_count, matched_iou, detection_size, partition = embedding_meta[pair]
        output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "identity_id": pair[0],
                "image_id": pair[1],
                "source_key": source_by_pair[pair][0],
                "embedding_partition": partition,
                "face_side": round(float(selected[pair]), 6),
                "centroid_similarity": round(float(similarities[pair]), 8),
                "insightface_detection_count": detection_count,
                "matched_iou": round(matched_iou, 8),
                "detection_size": detection_size,
            }
        )
    atomic_jsonl(args.output, output_rows)
    output_digest = sha256(args.output)
    progress(f"wrote reference scores={len(output_rows)} sha256={output_digest}")

    if args.skip_large_input_hashes:
        asset_index_digest = None
        embedding_db_digests = {
            partition: None for partition in embedding_dbs
        }
    else:
        asset_index_digest = sha256(args.asset_index)
        embedding_db_digests = {
            partition: sha256(path)
            for partition, path in embedding_dbs.items()
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "big_celebs_reference_scores",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_digest,
        "asset_index": str(args.asset_index),
        "asset_index_sha256": asset_index_digest,
        "embedding_dbs": {
            partition: {
                "path": str(path),
                "sha256": embedding_db_digests[partition],
                "selected_images": expected_partition_counts[partition],
            }
            for partition, path in embedding_dbs.items()
        },
        "score_file": str(args.output),
        "score_file_sha256": output_digest,
        "images": len(output_rows),
        "identities": len(records),
        "score": "cosine_to_normalized_identity_centroid",
    }
    atomic_json(args.output_manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def load_scores(path: Path) -> dict[tuple[str, str], dict]:
    scores: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pair = (str(record["identity_id"]), str(record["image_id"]))
            if pair in scores:
                raise ValueError(f"Duplicate score row for {pair!r} at line {line_number}")
            scores[pair] = record
    if not scores:
        raise ValueError(f"Empty score file: {path}")
    return scores


def _weighted_choice(
    rng: random.Random,
    identities: list[str],
    cumulative_weights: list[float],
) -> str:
    value = rng.random() * cumulative_weights[-1]
    return identities[min(bisect_left(cumulative_weights, value), len(identities) - 1)]


def _weighted_tables(
    identity_pools: dict[str, list[dict]],
    target_counts: dict[str, int],
    cap: int,
) -> tuple[list[str], list[float]]:
    identities = sorted(identity_pools)
    cumulative = []
    total = 0.0
    for identity in identities:
        total += math.sqrt(min(target_counts[identity], cap))
        cumulative.append(total)
    if not cumulative:
        raise ValueError("Sampling pool has no eligible identities")
    return identities, cumulative


def _draw_softmax(
    rng: random.Random,
    candidates: list[dict],
    temperature: float,
) -> tuple[dict, int]:
    maximum = max(float(candidate["centroid_similarity"]) for candidate in candidates)
    weights = [
        math.exp((float(candidate["centroid_similarity"]) - maximum) / temperature)
        for candidate in candidates
    ]
    chosen = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
    return candidates[chosen], chosen + 1


def _exposure_summary(exposures: Counter[str]) -> dict:
    values = sorted(exposures.values())
    return {
        "sampled_identities": len(values),
        "identities_once": sum(value == 1 for value in values),
        "identities_at_least_twice": sum(value >= 2 for value in values),
        "minimum": values[0],
        "median": values[len(values) // 2],
        "maximum": values[-1],
    }


def build_schedule(args: argparse.Namespace) -> None:
    if args.optimizer_steps < 1 or args.batch_size < 1:
        raise ValueError("Optimizer steps and batch size must be positive")
    if not 0 <= args.small_target_fraction <= 1:
        raise ValueError("--small-target-fraction must be within [0, 1]")
    if args.warmup_steps < 0 or args.warmup_steps > args.optimizer_steps:
        raise ValueError("--warmup-steps must fit within the schedule")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.min_reference_views < 4:
        raise ValueError("Policy v1 requires at least four reference views")

    manifest_digest = require_sha256(args.manifest, args.expected_manifest_sha256)
    score_manifest = json.loads(args.scores_manifest.read_text(encoding="utf-8"))
    if score_manifest.get("manifest_sha256") != manifest_digest:
        raise RuntimeError("Score manifest belongs to a different source manifest")
    score_digest = sha256(args.scores)
    if score_manifest.get("score_file_sha256") != score_digest:
        raise RuntimeError("Score file does not match its score manifest")

    records = load_manifest(args.manifest)
    scores = load_scores(args.scores)
    progress(
        f"building schedule from images={len(scores)} "
        f"optimizer_steps={args.optimizer_steps} batch_size={args.batch_size}"
    )
    expected_pairs = {
        (str(identity), str(image_id))
        for identity, images in records.items()
        for image_id in images
    }
    if set(scores) != expected_pairs:
        missing = expected_pairs - set(scores)
        extra = set(scores) - expected_pairs
        raise RuntimeError(
            f"Score coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    high_targets: dict[str, list[dict]] = {}
    low_targets: dict[str, list[dict]] = {}
    reference_pools: dict[str, list[dict]] = {}
    target_counts: dict[str, int] = {}
    for identity, images in records.items():
        enriched = []
        for image_id, metadata in images.items():
            score = scores[(str(identity), str(image_id))]
            enriched.append(
                {
                    "image_id": str(image_id),
                    "face_side": face_side(metadata),
                    "text": str(metadata["text"]),
                    "centroid_similarity": float(score["centroid_similarity"]),
                }
            )
        references = [
            image for image in enriched
            if image["face_side"] >= args.min_reference_face
        ]
        if len(references) < args.min_reference_views:
            continue
        identity = str(identity)
        reference_pools[identity] = references
        high_targets[identity] = [
            image for image in enriched
            if image["face_side"] >= args.min_reference_face
        ]
        low = [
            image for image in enriched
            if args.min_target_face <= image["face_side"] < args.min_reference_face
        ]
        if low:
            low_targets[identity] = low
        target_counts[identity] = len(high_targets[identity]) + len(low)

    high_identities, high_cumulative = _weighted_tables(
        high_targets, target_counts, args.identity_count_cap
    )
    low_identities, low_cumulative = _weighted_tables(
        low_targets, target_counts, args.identity_count_cap
    ) if low_targets else ([], [])

    rng = random.Random(args.seed)
    cycles: dict[tuple[str, str], list[dict]] = {}

    def target_for(identity: str, target_bin: str) -> dict:
        key = (identity, target_bin)
        if not cycles.get(key):
            source = high_targets[identity] if target_bin == "ge256" else low_targets[identity]
            cycle = list(source)
            rng.shuffle(cycle)
            cycles[key] = cycle
        return cycles[key].pop()

    rows = []
    target_exposures: Counter[str] = Counter()
    pair_exposures: Counter[tuple[str, str, str]] = Counter()
    target_paths: set[tuple[str, str]] = set()
    bin_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    flip_counts: Counter[str] = Counter()
    total_rows = args.optimizer_steps * args.batch_size
    for row_number in range(total_rows):
        optimizer_step = row_number // args.batch_size
        use_low = (
            optimizer_step >= args.warmup_steps
            and bool(low_identities)
            and rng.random() < args.small_target_fraction
        )
        if use_low:
            target_bin = "192_255"
            identity = _weighted_choice(rng, low_identities, low_cumulative)
        else:
            target_bin = "ge256"
            identity = _weighted_choice(rng, high_identities, high_cumulative)
        target = target_for(identity, target_bin)

        ranked = sorted(
            (
                candidate for candidate in reference_pools[identity]
                if candidate["image_id"] != target["image_id"]
            ),
            key=lambda candidate: (
                -candidate["centroid_similarity"],
                -candidate["face_side"],
                candidate["image_id"],
            ),
        )
        if len(ranked) < 3:
            raise RuntimeError(
                f"Identity {identity!r} has fewer than three distinct references"
            )
        reference, reference_rank = _draw_softmax(
            rng, ranked[:3], args.temperature
        )
        directional = bool(DIRECTION_RE.search(target["text"]))
        flip_target = False if directional else rng.random() < 0.5

        row = {
            "schema_version": SCHEMA_VERSION,
            "row": row_number,
            "optimizer_step": optimizer_step,
            "identity_id": identity,
            "target_image_id": target["image_id"],
            "reference_image_id": reference["image_id"],
            "target_face_bin": target_bin,
            "reference_rank": reference_rank,
            "reference_centroid_similarity": round(
                reference["centroid_similarity"], 8
            ),
            "flip_target": flip_target,
        }
        rows.append(row)
        target_exposures[identity] += 1
        pair_exposures[(identity, target["image_id"], reference["image_id"])] += 1
        target_paths.add((identity, target["image_id"]))
        bin_counts[target_bin] += 1
        rank_counts[reference_rank] += 1
        flip_counts["directional_not_flipped" if directional else "nondirectional"] += 1
        if flip_target:
            flip_counts["flipped"] += 1

    atomic_jsonl(args.output, rows)
    output_digest = sha256(args.output)
    progress(f"wrote schedule rows={len(rows)} sha256={output_digest}")
    repeated_pair_rows = sum(value - 1 for value in pair_exposures.values() if value > 1)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "big_celebs_sampling_plan",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": manifest_digest,
        "score_file": str(args.scores),
        "score_file_sha256": score_digest,
        "plan_file": str(args.output),
        "plan_file_sha256": output_digest,
        "rows": total_rows,
        "optimizer_steps": args.optimizer_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "policy": {
            "minimum_target_face": args.min_target_face,
            "minimum_reference_face": args.min_reference_face,
            "minimum_reference_views": args.min_reference_views,
            "identity_weight": f"sqrt(min(target_count,{args.identity_count_cap}))",
            "warmup_steps": args.warmup_steps,
            "small_target_fraction_after_warmup": args.small_target_fraction,
            "reference_mode": "top3_softmax_identity_centroid",
            "temperature": args.temperature,
            "directional_caption_flip": False,
            "nondirectional_flip_probability": 0.5,
            "reference_format": "raw",
        },
        "audit": {
            "eligible_identities": len(reference_pools),
            "eligible_high_targets": sum(map(len, high_targets.values())),
            "eligible_low_targets": sum(map(len, low_targets.values())),
            "unique_targets": len(target_paths),
            "unique_ordered_pairs": len(pair_exposures),
            "repeated_pair_rows": repeated_pair_rows,
            "target_bins": dict(sorted(bin_counts.items())),
            "reference_ranks": {str(key): value for key, value in sorted(rank_counts.items())},
            "flips": dict(sorted(flip_counts.items())),
            "identity_exposures": _exposure_summary(target_exposures),
            "self_reference_rows": 0,
            "cross_identity_rows": 0,
        },
    }
    atomic_json(args.output_manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scores = subparsers.add_parser("scores", help="join release images to ArcFace scores")
    scores.add_argument("--manifest", type=Path, required=True)
    scores.add_argument("--expected-manifest-sha256", required=True)
    scores.add_argument("--asset-index", type=Path, required=True)
    scores.add_argument("--legacy-embedding-db", type=Path, required=True)
    scores.add_argument("--eqr6-embedding-db", type=Path, required=True)
    scores.add_argument("--neb-incremental-embedding-db", type=Path, required=True)
    scores.add_argument("--output", type=Path, required=True)
    scores.add_argument("--output-manifest", type=Path, required=True)
    scores.add_argument("--skip-large-input-hashes", action="store_true")
    scores.set_defaults(function=build_scores)

    schedule = subparsers.add_parser("schedule", help="build a deterministic pair schedule")
    schedule.add_argument("--manifest", type=Path, required=True)
    schedule.add_argument("--expected-manifest-sha256", required=True)
    schedule.add_argument("--scores", type=Path, required=True)
    schedule.add_argument("--scores-manifest", type=Path, required=True)
    schedule.add_argument("--output", type=Path, required=True)
    schedule.add_argument("--output-manifest", type=Path, required=True)
    schedule.add_argument("--optimizer-steps", type=int, default=40000)
    schedule.add_argument("--batch-size", type=int, default=2)
    schedule.add_argument("--warmup-steps", type=int, default=6000)
    schedule.add_argument("--small-target-fraction", type=float, default=0.2)
    schedule.add_argument("--min-target-face", type=int, default=192)
    schedule.add_argument("--min-reference-face", type=int, default=256)
    schedule.add_argument("--min-reference-views", type=int, default=4)
    schedule.add_argument("--identity-count-cap", type=int, default=16)
    schedule.add_argument("--temperature", type=float, default=0.05)
    schedule.add_argument("--seed", type=int, default=20260801)
    schedule.set_defaults(function=build_schedule)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
