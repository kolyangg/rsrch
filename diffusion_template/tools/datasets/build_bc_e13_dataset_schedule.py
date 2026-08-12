#!/usr/bin/env python3
"""Build deterministic, sealed 48k schedules for BC_E13_ds1/ds2/ds3."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from src.datasets.bc_e13_schedule_policy import (
    POLICY_VERSION,
    SCHEMA_VERSION,
    face_area,
    face_side,
    is_canonical_reference,
    is_directional,
    iter_manifest_paths,
    load_identity_manifest,
    sha256_file,
    stable_digest,
)


DEFAULT_BIG_SHA256 = "f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
DEFAULT_LARGE_SHA256 = "0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50"
DEFAULT_SCENE_AREA_MAX = 0.17154455184936523
EXPECTED_COHORT_IMAGES = 60087
EXPECTED_COHORT_SCENE_TARGETS = 27165
EXPECTED_COHORT_CANONICAL_REFS = 23094


def require_sha256(path: Path, expected: str) -> str:
    expected = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Expected SHA-256 must be 64 lowercase hex digits")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Manifest SHA-256 mismatch for {path}: expected={expected}, found={actual}"
        )
    return actual


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def shuffled_paths(paths: Iterable[str], *salt: object) -> list[str]:
    return sorted(paths, key=lambda path: stable_digest(POLICY_VERSION, *salt, path))


def deterministic_flip(seed: int, row_index: int, path: str, prompt: str) -> bool:
    if is_directional(prompt):
        return False
    return int(stable_digest(POLICY_VERSION, seed, "flip", row_index, path), 16) % 2 == 0


def path_metadata(records: dict[str, dict[str, dict]], relative_path: str) -> dict:
    identity, filename = relative_path.split("/", 1)
    if not filename.endswith(".jpg"):
        raise ValueError(f"Unexpected manifest path: {relative_path}")
    return records[identity][filename[:-4]]


def identity_paths(records: dict[str, dict[str, dict]], identity: str) -> list[str]:
    return [f"{identity}/{image_id}.jpg" for image_id in records[identity]]


def eligible_big_cohort(
    records: dict[str, dict[str, dict]],
    *,
    cohort_size: int,
    seed: int,
    scene_area_max: float,
    canonical_min_side: float,
) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    pools: dict[str, dict[str, list[str]]] = {}
    eligible: list[str] = []
    for identity in records:
        all_paths = identity_paths(records, identity)
        scene_paths = [
            path
            for path in all_paths
            if face_area(path_metadata(records, path)) <= scene_area_max
        ]
        canonical_paths = [
            path
            for path in all_paths
            if is_canonical_reference(
                path_metadata(records, path), min_side=canonical_min_side
            )
        ]
        if len(scene_paths) < 2 or len(canonical_paths) < 2:
            continue
        eligible.append(identity)
        pools[identity] = {
            "all": shuffled_paths(all_paths, seed, "all", identity),
            "scene": shuffled_paths(scene_paths, seed, "scene", identity),
            "canonical": sorted(
                canonical_paths,
                key=lambda path: (
                    -face_side(path_metadata(records, path)),
                    stable_digest(POLICY_VERSION, "canonical", identity, path),
                ),
            ),
        }
    ranked = sorted(
        eligible,
        key=lambda identity: (
            -len(pools[identity]["all"]),
            # Match the audit's exact stable_rank implementation.
            hashlib.sha256(
                f"bc-e13-policy-v1:{identity}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    if len(ranked) < cohort_size:
        raise RuntimeError(
            f"Only {len(ranked)} identities satisfy strict eligibility; "
            f"need {cohort_size}"
        )
    selected = ranked[:cohort_size]
    # Identity order is independent of dict insertion order and gives every
    # selected identity one visit before any receives its next visit.
    round_robin = sorted(
        selected,
        key=lambda identity: stable_digest(POLICY_VERSION, "round_robin", identity),
    )
    return round_robin, {identity: pools[identity] for identity in selected}


def choose_distinct(
    candidates: list[str],
    target_path: str,
    *,
    seed: int,
    row_index: int,
    salt: str,
) -> tuple[str, int]:
    distinct = [path for path in candidates if path != target_path]
    if not distinct:
        raise RuntimeError(f"No distinct reference candidate for {target_path}")
    choice = int(
        stable_digest(POLICY_VERSION, seed, salt, row_index, target_path), 16
    ) % len(distinct)
    return distinct[choice], choice + 1


def make_row(
    *,
    row_index: int,
    source: str,
    phase: str,
    identity: str,
    target_path: str,
    reference_path: str,
    target_role: str,
    reference_tier: str,
    flip_target: bool,
    manifest_sha256: str,
    records: dict[str, dict[str, dict]],
) -> dict:
    target = path_metadata(records, target_path)
    reference = path_metadata(records, reference_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "schedule_index": row_index,
        "optimizer_step": row_index // 2,
        "source": source,
        "phase": phase,
        "identity_id": identity,
        "target_path": target_path,
        "reference_path": reference_path,
        "target_role": target_role,
        "reference_tier": reference_tier,
        "flip_target": bool(flip_target),
        "source_manifest_sha256": manifest_sha256,
        "target_bbox": target["new_face_crop"],
        "reference_bbox": reference["new_face_crop"],
        "prompt": target["text"],
    }


def big_row(
    *,
    row_index: int,
    big_index: int,
    target_visit: int,
    mode: str,
    seed: int,
    identities: list[str],
    pools: dict[str, dict[str, list[str]]],
    records: dict[str, dict[str, dict]],
    manifest_sha256: str,
) -> dict:
    identity = identities[big_index % len(identities)]
    if mode == "ds1":
        target_pool = pools[identity]["all"]
        target_path = target_pool[target_visit % len(target_pool)]
        reference_path, _ = choose_distinct(
            pools[identity]["all"],
            target_path,
            seed=seed,
            row_index=row_index,
            salt="ds1-unrestricted-ref",
        )
        target_role = "unrestricted"
        reference_tier = "unrestricted_distinct"
        phase = "repeat_depth_balanced"
    else:
        scene_quota = big_index % 3 != 2
        target_pool = pools[identity]["scene" if scene_quota else "all"]
        target_path = target_pool[target_visit % len(target_pool)]
        ranked_refs = [
            path for path in pools[identity]["canonical"] if path != target_path
        ][:3]
        reference_path, rank = choose_distinct(
            ranked_refs,
            target_path,
            seed=seed,
            row_index=row_index,
            salt="ds2-canonical-top3-ref",
        )
        target_role = "scene" if scene_quota else "unrestricted"
        reference_tier = f"canonical_top{rank}"
        phase = "scene_target_canonical_ref"
    prompt = str(path_metadata(records, target_path)["text"])
    return make_row(
        row_index=row_index,
        source="big_celebs",
        phase=phase,
        identity=identity,
        target_path=target_path,
        reference_path=reference_path,
        target_role=target_role,
        reference_tier=reference_tier,
        flip_target=deterministic_flip(seed, row_index, target_path, prompt),
        manifest_sha256=manifest_sha256,
        records=records,
    )


def build_rows(
    *,
    mode: str,
    rows: int,
    seed: int,
    big_records: dict[str, dict[str, dict]],
    big_sha256: str,
    identities: list[str],
    pools: dict[str, dict[str, list[str]]],
    large_records: dict[str, dict[str, dict]] | None,
    large_sha256: str | None,
) -> list[dict]:
    if mode in {"ds1", "ds2"}:
        visits: Counter[tuple[str, str]] = Counter()
        output = []
        for row_index in range(rows):
            identity = identities[row_index % len(identities)]
            role = "all" if mode == "ds1" else (
                "scene" if row_index % 3 != 2 else "all"
            )
            output.append(big_row(
                row_index=row_index,
                big_index=row_index,
                target_visit=visits[(identity, role)],
                mode=mode,
                seed=seed,
                identities=identities,
                pools=pools,
                records=big_records,
                manifest_sha256=big_sha256,
            ))
            visits[(identity, role)] += 1
        return output

    if large_records is None or large_sha256 is None:
        raise ValueError("ds3 requires Large Dataset records and SHA-256")
    large_items = [
        (identity, path)
        for identity, path, _ in iter_manifest_paths(large_records)
    ]
    large_items.sort(
        key=lambda item: stable_digest(
            POLICY_VERSION, seed, "large-image-permutation", item[0], item[1]
        )
    )
    large_needed = rows * 2 // 3
    if len(large_items) < large_needed:
        raise RuntimeError(
            f"Large Dataset has only {len(large_items)} images; need {large_needed}"
        )
    output: list[dict] = []
    large_index = 0
    big_index = 0
    big_visits: Counter[tuple[str, str]] = Counter()
    for row_index in range(rows):
        if row_index % 3 == 2:
            identity = identities[big_index % len(identities)]
            role = "scene" if big_index % 3 != 2 else "all"
            output.append(
                big_row(
                    row_index=row_index,
                    big_index=big_index,
                    target_visit=big_visits[(identity, role)],
                    mode="ds2",
                    seed=seed,
                    identities=identities,
                    pools=pools,
                    records=big_records,
                    manifest_sha256=big_sha256,
                )
            )
            output[-1]["phase"] = "large_anchor_big_supplement"
            big_visits[(identity, role)] += 1
            big_index += 1
            continue
        identity, target_path = large_items[large_index]
        target_metadata = path_metadata(large_records, target_path)
        candidates = shuffled_paths(
            identity_paths(large_records, identity),
            seed,
            "large-reference",
            identity,
        )
        reference_path, _ = choose_distinct(
            candidates,
            target_path,
            seed=seed,
            row_index=row_index,
            salt="ds3-large-distinct-ref",
        )
        prompt = str(target_metadata["text"])
        output.append(
            make_row(
                row_index=row_index,
                source="large_dataset",
                phase="large_anchor",
                identity=identity,
                target_path=target_path,
                reference_path=reference_path,
                target_role="unrestricted",
                reference_tier="unrestricted_distinct",
                flip_target=deterministic_flip(seed, row_index, target_path, prompt),
                manifest_sha256=large_sha256,
                records=large_records,
            )
        )
        large_index += 1
    if large_index != large_needed or big_index != rows // 3:
        raise AssertionError("ds3 source interleave count drift")
    return output


def count_rows(rows: list[dict]) -> dict:
    counts = {
        "source": Counter(),
        "phase": Counter(),
        "target_role": Counter(),
        "reference_tier": Counter(),
        "flip_target": Counter(),
        "identity": Counter(),
    }
    directional_rows = 0
    nondirectional_rows = 0
    nondirectional_flips = 0
    for row in rows:
        for field in ("source", "phase", "target_role", "reference_tier"):
            counts[field][str(row[field])] += 1
        counts["flip_target"][str(bool(row["flip_target"])).lower()] += 1
        counts["identity"][(row["source"], row["identity_id"])] += 1
        if is_directional(str(row["prompt"])):
            directional_rows += 1
        else:
            nondirectional_rows += 1
            nondirectional_flips += int(bool(row["flip_target"]))
    identity_values = sorted(counts["identity"].values())
    return {
        "source": dict(sorted(counts["source"].items())),
        "phase": dict(sorted(counts["phase"].items())),
        "target_role": dict(sorted(counts["target_role"].items())),
        "reference_tier": dict(sorted(counts["reference_tier"].items())),
        "flip_target": dict(sorted(counts["flip_target"].items())),
        "identity_exposure": {
            "identities": len(identity_values),
            "minimum": min(identity_values),
            "maximum": max(identity_values),
            "mean": sum(identity_values) / len(identity_values),
            "histogram": dict(sorted(Counter(identity_values).items())),
        },
        "fallback_reasons": {},
        "directional_caption_audit": {
            "directional_rows": directional_rows,
            "directional_flips": 0,
            "nondirectional_rows": nondirectional_rows,
            "nondirectional_flips": nondirectional_flips,
            "nondirectional_flip_rate": (
                nondirectional_flips / nondirectional_rows
                if nondirectional_rows
                else 0.0
            ),
        },
    }


def window_counts(rows: list[dict], *, optimizer_steps_per_window: int = 2000) -> list[dict]:
    rows_per_window = optimizer_steps_per_window * 2
    output = []
    for start in range(0, len(rows), rows_per_window):
        selected = rows[start : start + rows_per_window]
        output.append(
            {
                "start_step": start // 2,
                "end_step_exclusive": (start + len(selected)) // 2,
                "rows": len(selected),
                "sources": dict(sorted(Counter(row["source"] for row in selected).items())),
                "target_roles": dict(
                    sorted(Counter(row["target_role"] for row in selected).items())
                ),
                "unique_identities_by_source": {
                    source: len(
                        {
                            row["identity_id"]
                            for row in selected
                            if row["source"] == source
                        }
                    )
                    for source in sorted({row["source"] for row in selected})
                },
            }
        )
    return output


def validate_complete_schedule(
    rows: list[dict],
    *,
    mode: str,
    big_records: dict[str, dict[str, dict]],
    source_roots: dict[str, Path],
    scene_area_max: float,
    canonical_min_side: float,
) -> None:
    if len(rows) != 48000:
        raise RuntimeError(f"Expected 48,000 schedule rows, found {len(rows)}")
    for expected_index, row in enumerate(rows):
        if row["schedule_index"] != expected_index:
            raise RuntimeError(f"Non-contiguous row {expected_index}")
        if row["target_path"] == row["reference_path"]:
            raise RuntimeError(f"Self-reference at row {expected_index}")
        root = source_roots[row["source"]]
        target_identity = row["target_path"].split("/", 1)[0]
        reference_identity = row["reference_path"].split("/", 1)[0]
        if target_identity != row["identity_id"] or reference_identity != row["identity_id"]:
            raise RuntimeError(f"Cross-identity pair at row {expected_index}")
        if not (root / row["target_path"]).is_file():
            raise FileNotFoundError(
                f"Missing scheduled target at row {expected_index}: "
                f"{root / row['target_path']}"
            )
        if not (root / row["reference_path"]).is_file():
            raise FileNotFoundError(
                f"Missing scheduled reference at row {expected_index}: "
                f"{root / row['reference_path']}"
            )
        if row["flip_target"] and is_directional(row["prompt"]):
            raise RuntimeError(f"Directional target flipped at row {expected_index}")
        if row["source"] == "big_celebs" and mode in {"ds2", "ds3"}:
            target = path_metadata(big_records, row["target_path"])
            reference = path_metadata(big_records, row["reference_path"])
            if row["target_bbox"] != target["new_face_crop"] or row["prompt"] != target["text"]:
                raise RuntimeError(f"Target metadata drift at row {expected_index}")
            if row["reference_bbox"] != reference["new_face_crop"]:
                raise RuntimeError(f"Reference metadata drift at row {expected_index}")
            if row["target_role"] == "scene" and face_area(target) > scene_area_max:
                raise RuntimeError(f"Scene target violates area gate at row {expected_index}")
            if not is_canonical_reference(reference, min_side=canonical_min_side):
                raise RuntimeError(f"Reference violates canonical gate at row {expected_index}")
            canonical = sorted(
                (
                    path
                    for path in identity_paths(big_records, row["identity_id"])
                    if path != row["target_path"]
                    and is_canonical_reference(
                        path_metadata(big_records, path), min_side=canonical_min_side
                    )
                ),
                key=lambda path: (
                    -face_side(path_metadata(big_records, path)),
                    stable_digest(
                        POLICY_VERSION, "canonical", row["identity_id"], path
                    ),
                ),
            )[:3]
            if row["reference_path"] not in canonical:
                raise RuntimeError(f"Reference is outside canonical top three at row {expected_index}")
            expected_tier = f"canonical_top{canonical.index(row['reference_path']) + 1}"
            if row["reference_tier"] != expected_tier:
                raise RuntimeError(f"Canonical reference-rank drift at row {expected_index}")
    sources = Counter(row["source"] for row in rows)
    roles = Counter(
        row["target_role"] for row in rows if row["source"] == "big_celebs"
    )
    if mode in {"ds1", "ds2"} and sources != {"big_celebs": 48000}:
        raise RuntimeError(f"Unexpected {mode} source counts: {sources}")
    if mode == "ds3" and sources != {"large_dataset": 32000, "big_celebs": 16000}:
        raise RuntimeError(f"Unexpected ds3 source counts: {sources}")
    expected_big = sources["big_celebs"]
    if mode in {"ds2", "ds3"}:
        expected_scene = (expected_big // 3) * 2 + min(expected_big % 3, 2)
        if roles["scene"] != expected_scene:
            raise RuntimeError(f"Scene quota drift: expected={expected_scene}, found={roles}")
    big_exposures = Counter(
        row["identity_id"] for row in rows if row["source"] == "big_celebs"
    )
    expected_range = (18, 19) if mode in {"ds1", "ds2"} else (6, 7)
    if len(big_exposures) != 2561 or (
        min(big_exposures.values()), max(big_exposures.values())
    ) != expected_range:
        raise RuntimeError(
            f"BigCelebs round-robin exposure drift: identities={len(big_exposures)}, "
            f"range={(min(big_exposures.values()), max(big_exposures.values()))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ds1", "ds2", "ds3"], required=True)
    parser.add_argument("--big-manifest", type=Path, required=True)
    parser.add_argument("--big-images-root", type=Path, required=True)
    parser.add_argument("--big-manifest-sha256", default=DEFAULT_BIG_SHA256)
    parser.add_argument("--large-manifest", type=Path)
    parser.add_argument("--large-images-root", type=Path)
    parser.add_argument("--large-manifest-sha256", default=DEFAULT_LARGE_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=48000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cohort-size", type=int, default=2561)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--scene-area-max", type=float, default=DEFAULT_SCENE_AREA_MAX)
    parser.add_argument("--canonical-min-side", type=float, default=384.0)
    parser.add_argument("--expected-cohort-images", type=int, default=EXPECTED_COHORT_IMAGES)
    parser.add_argument("--expected-cohort-min-images", type=int, default=14)
    parser.add_argument("--expected-cohort-max-images", type=int, default=186)
    parser.add_argument(
        "--expected-cohort-scene-targets",
        type=int,
        default=EXPECTED_COHORT_SCENE_TARGETS,
    )
    parser.add_argument(
        "--expected-cohort-canonical-refs",
        type=int,
        default=EXPECTED_COHORT_CANONICAL_REFS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows != 48000 or args.batch_size != 2:
        raise ValueError("BC_E13 ds1-ds3 are sealed to 48,000 rows and batch size 2")
    if args.cohort_size != 2561:
        raise ValueError("BC_E13 ds1-ds3 require the shared 2,561-ID cohort")
    big_sha256 = require_sha256(args.big_manifest, args.big_manifest_sha256)
    if not args.big_images_root.is_dir():
        raise FileNotFoundError(f"BigCelebs image root is unavailable: {args.big_images_root}")
    big_records = load_identity_manifest(args.big_manifest)
    identities, pools = eligible_big_cohort(
        big_records,
        cohort_size=args.cohort_size,
        seed=args.seed,
        scene_area_max=args.scene_area_max,
        canonical_min_side=args.canonical_min_side,
    )
    cohort_images = sum(len(pools[identity]["all"]) for identity in identities)
    cohort_scene = sum(len(pools[identity]["scene"]) for identity in identities)
    cohort_canonical = sum(len(pools[identity]["canonical"]) for identity in identities)
    expected = {
        "images": args.expected_cohort_images,
        "scene_targets": args.expected_cohort_scene_targets,
        "canonical_references": args.expected_cohort_canonical_refs,
    }
    actual = {
        "images": cohort_images,
        "scene_targets": cohort_scene,
        "canonical_references": cohort_canonical,
    }
    if actual != expected:
        raise RuntimeError(f"Shared strict cohort count mismatch: expected={expected}, found={actual}")
    cohort_sizes = sorted(len(pools[identity]["all"]) for identity in identities)
    if (cohort_sizes[0], cohort_sizes[-1]) != (
        args.expected_cohort_min_images,
        args.expected_cohort_max_images,
    ):
        raise RuntimeError(
            "Shared strict cohort depth-range mismatch: "
            f"expected={(args.expected_cohort_min_images, args.expected_cohort_max_images)}, "
            f"found={(cohort_sizes[0], cohort_sizes[-1])}"
        )

    large_records = None
    large_sha256 = None
    if args.mode == "ds3":
        if args.large_manifest is None or args.large_images_root is None:
            raise ValueError("ds3 requires --large-manifest and --large-images-root")
        if not args.large_images_root.is_dir():
            raise FileNotFoundError(
                f"Large Dataset image root is unavailable: {args.large_images_root}"
            )
        large_sha256 = require_sha256(
            args.large_manifest, args.large_manifest_sha256
        )
        large_records = load_identity_manifest(args.large_manifest)

    rows = build_rows(
        mode=args.mode,
        rows=args.rows,
        seed=args.seed,
        big_records=big_records,
        big_sha256=big_sha256,
        identities=identities,
        pools=pools,
        large_records=large_records,
        large_sha256=large_sha256,
    )
    validate_complete_schedule(
        rows,
        mode=args.mode,
        big_records=big_records,
        source_roots={
            "big_celebs": args.big_images_root,
            **(
                {"large_dataset": args.large_images_root}
                if args.mode == "ds3"
                else {}
            ),
        },
        scene_area_max=args.scene_area_max,
        canonical_min_side=args.canonical_min_side,
    )
    atomic_jsonl(args.output, rows)
    schedule_sha256 = sha256_file(args.output)
    base_name_groups: dict[str, list[str]] = {}
    for identity in identities:
        base_name_groups.setdefault(identity.rsplit("__", 1)[0], []).append(identity)
    repeated_name_groups = {
        base: sorted(group)
        for base, group in sorted(base_name_groups.items())
        if len(group) > 1
    }
    first_last = {
        "first": stable_digest(json.dumps(rows[0], sort_keys=True, separators=(",", ":"))),
        "last": stable_digest(json.dumps(rows[-1], sort_keys=True, separators=(",", ":"))),
    }
    sources = {
        "big_celebs": {
            "path": str(args.big_manifest),
            "images_root": str(args.big_images_root),
            "sha256": big_sha256,
            "images": sum(len(images) for images in big_records.values()),
            "identities": len(big_records),
        }
    }
    if args.mode == "ds3":
        sources["large_dataset"] = {
            "path": str(args.large_manifest),
            "images_root": str(args.large_images_root),
            "sha256": large_sha256,
            "images": sum(len(images) for images in large_records.values()),
            "identities": len(large_records),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bc_e13_dataset_schedule",
        "policy_version": POLICY_VERSION,
        "mode": args.mode,
        "arguments": {
            "mode": args.mode,
            "big_manifest": str(args.big_manifest),
            "big_images_root": str(args.big_images_root),
            "big_manifest_sha256": args.big_manifest_sha256,
            "large_manifest": (
                str(args.large_manifest) if args.large_manifest is not None else None
            ),
            "large_images_root": (
                str(args.large_images_root)
                if args.large_images_root is not None
                else None
            ),
            "large_manifest_sha256": (
                args.large_manifest_sha256 if args.mode == "ds3" else None
            ),
            "output": str(args.output),
            "summary_output": str(args.summary_output),
            "rows": args.rows,
            "batch_size": args.batch_size,
            "cohort_size": args.cohort_size,
            "seed": args.seed,
            "scene_area_max": args.scene_area_max,
            "canonical_min_side": args.canonical_min_side,
            "expected_cohort_images": args.expected_cohort_images,
            "expected_cohort_min_images": args.expected_cohort_min_images,
            "expected_cohort_max_images": args.expected_cohort_max_images,
            "expected_cohort_scene_targets": args.expected_cohort_scene_targets,
            "expected_cohort_canonical_refs": args.expected_cohort_canonical_refs,
        },
        "sources": sources,
        "schedule": {
            "path": str(args.output),
            "sha256": schedule_sha256,
            "rows": len(rows),
            "optimizer_steps": len(rows) // args.batch_size,
            "first_last_fingerprints": first_last,
        },
        "cohort": {
            "selection": "strict-role-eligible then descending usable image count with stable SHA-256 tie-break",
            "identity_order_sha256": stable_digest(*identities),
            "identities": len(identities),
            "images": cohort_images,
            "scene_targets": cohort_scene,
            "canonical_references": cohort_canonical,
            "minimum_images": min(cohort_sizes),
            "maximum_images": max(cohort_sizes),
            "mean_images": sum(cohort_sizes) / len(cohort_sizes),
            "repeated_name_audit": {
                "candidate_base_names": len(repeated_name_groups),
                "candidate_identity_groups": sum(
                    len(group) for group in repeated_name_groups.values()
                ),
                "groups": repeated_name_groups,
                "interpretation": "name-only candidates; no identity merge or filter applied",
            },
        },
        "counts": count_rows(rows),
        "windows_2000_steps": window_counts(rows),
        "audit": {
            "target_reference_distinct": True,
            "same_identity": True,
            "directional_targets_flipped": 0,
            "fallbacks": 0,
            "complete_schedule_scan": True,
        },
    }
    atomic_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
