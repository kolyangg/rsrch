#!/usr/bin/env python3
"""Offline Cosmic Large identity assets for the CL suite.

Two subcommands, both read-only by default (`--write` is explicit):

  groups       CL1 prerequisite. Embed every accepted target face with the pinned
               Buffalo-L recogniser and join targets into identity components via
               mutual nearest neighbours, so the loader can sample a distinct
               same-identity 1024px scene reference.

  accept-list  Shared CL control. Keep only the 256px reference crops whose
               detected face overlaps the supplied box, so the loader never has
               to call InsightFace in a DataLoader worker and never substitutes a
               silent zero identity embedding.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.reference_policy import valid_bbox  # noqa: E402
from src.model.photomaker_branched.insightface_package import (  # noqa: E402
    analyze_faces,
    create_face_analyzer,
)


ARCFACE_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
DEFAULT_ARCFACE = "~/.insightface/models/buffalo_l/w600k_r50.onnx"


def embed_targets_torch(items, root: Path, *, model_path: str, batch_size: int,
                        workers: int, device: str, stats):
    """Embed target faces on the GPU with E22's PyTorch ArcFace graph.

    # 07 Aug 2026 - The local onnxruntime is a CPU-only build, so InsightFace
    # cannot reach the GPU. `FrozenOnnxArcFace` executes the identical
    # w600k_r50 graph with torch operators (E22 verified cosine=1.0 and max
    # absolute error 3.7e-6 against ONNX Runtime), so this is the same identity
    # signal at GPU speed. Like E22 it embeds the supplied face box directly
    # rather than a landmark-aligned crop.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.model.photomaker_branched.arcface_identity_aux import FrozenOnnxArcFace

    model = FrozenOnnxArcFace(
        model_path=str(Path(model_path).expanduser()),
        expected_sha256=ARCFACE_SHA256,
    ).to(device).eval()

    def crop(item):
        target_path, record = item
        try:
            image = load_target(root, target_path, record)
        except Exception:
            return target_path, None
        x0, y0, x1, y1 = [int(round(float(v))) for v in record["face_crop_new"]]
        face = image.crop((x0, y0, x1, y1)).resize((112, 112), Image.BILINEAR)
        array = np.asarray(face, dtype=np.float32)
        return target_path, (array - 127.5) / 127.5

    paths, embeddings = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch_paths, batch_arrays = [], []
        for target_path, array in pool.map(crop, items, chunksize=8):
            if array is None:
                stats["unreadable"] += 1
                continue
            batch_paths.append(target_path)
            batch_arrays.append(array)
            if len(batch_arrays) < batch_size:
                continue
            paths.extend(batch_paths)
            embeddings.append(_run_batch(model, batch_arrays, device, torch))
            batch_paths, batch_arrays = [], []
        if batch_arrays:
            paths.extend(batch_paths)
            embeddings.append(_run_batch(model, batch_arrays, device, torch))

    stats["embedded"] += len(paths)
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 512), np.float32)
    return paths, matrix


def _run_batch(model, arrays, device, torch):
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        out = model(tensor).float()
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.cpu().numpy()


def build_analyzer(use_cuda: bool):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    )
    return create_face_analyzer(
        providers=providers,
        provider_options=[{"device_id": 0}, {}] if use_cuda else None,
        allowed_modules=["detection", "recognition"],
        ctx_id=0 if use_cuda else -1,
        det_size=(640, 640),
        fallback_ctx_id=-1,
        quiet=True,
    )


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def best_face(faces, bbox):
    """Select the detection overlapping the supplied box, not simply faces[0]."""
    best, best_iou = None, 0.0
    for face in faces:
        score = iou(face["bbox"], bbox)
        if score > best_iou:
            best, best_iou = face, score
    return best, best_iou


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, dict) or not records:
        raise ValueError(f"Invalid or empty manifest: {path}")
    return records


def open_image(root: Path, relative: str) -> Image.Image:
    candidate = Path(relative)
    resolved = candidate if candidate.is_absolute() else root / candidate
    return Image.open(resolved).convert("RGB")


def load_target(root: Path, relative: str, record: dict) -> Image.Image:
    image = open_image(root, relative)
    if image.size != (1024, 1024):
        body_crop = record.get("body_crop")
        if body_crop is None or len(body_crop) != 4:
            raise ValueError(f"{relative} is {image.size} with no body_crop")
        left, top, right, bottom = [int(v) for v in body_crop]
        array = np.asarray(image)[top:bottom, left:right]
        if array.shape[:2] != (1024, 1024):
            raise ValueError(f"body_crop for {relative} produced {array.shape[:2]}")
        image = Image.fromarray(array)
    return image


def seal(payload: dict, output: Path, write: bool) -> str:
    blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    if write:
        output.write_bytes(blob)
    return digest


def cmd_groups(args) -> None:
    records = load_manifest(args.manifest)
    root = Path(args.dataset_root)
    stats = Counter()

    accepted = []
    for target_path, record in records.items():
        if not isinstance(record, dict):
            continue
        bbox = record.get("face_crop_new")
        if not valid_bbox(bbox, (1024, 1024)):
            stats["filtered_target_bbox"] += 1
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if min(x1 - x0, y1 - y0) < args.min_face_res:
            stats["filtered_target_face"] += 1
            continue
        accepted.append((str(target_path), record))
        if args.limit and len(accepted) >= args.limit:
            break

    if args.backend == "torch":
        paths, matrix = embed_targets_torch(
            accepted, root, model_path=args.arcface_onnx, batch_size=args.batch_size,
            workers=args.workers, device=args.device, stats=stats,
        )
    else:
        analyzer = build_analyzer(not args.cpu)
        paths, embeddings = [], []
        for target_path, record in accepted:
            try:
                image = load_target(root, target_path, record)
            except Exception:
                stats["unreadable"] += 1
                continue
            bbox = record["face_crop_new"]
            faces = analyze_faces(analyzer, np.array(image)[:, :, ::-1])
            face, overlap = best_face(faces, bbox)
            if face is None or overlap < args.min_iou:
                stats["no_matching_detection"] += 1
                continue
            vector = np.asarray(face["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                stats["degenerate_embedding"] += 1
                continue
            paths.append(target_path)
            embeddings.append(vector / norm)
            stats["embedded"] += 1
        matrix = np.stack(embeddings) if embeddings else np.zeros((0, 512), np.float32)

    if len(paths) < 2:
        raise RuntimeError(f"Too few embedded targets to group: {len(paths)}")

    if args.save_embeddings:
        np.savez_compressed(args.save_embeddings, paths=np.array(paths), embeddings=matrix)
        print(f"[saved embeddings] {args.save_embeddings}", file=sys.stderr)

    # Mutual nearest neighbour above the threshold, then connected components.
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.argmax(axis=1)
    best_similarity = similarity[np.arange(len(paths)), nearest]

    def components_at(threshold: float):
        parent = list(range(len(paths)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        edges = 0
        for i, j in enumerate(nearest):
            j = int(j)
            if int(nearest[j]) == i and similarity[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    edges += 1
        groups: dict[int, list[str]] = {}
        for i, path in enumerate(paths):
            groups.setdefault(find(i), []).append(path)
        return groups, edges

    # Report the launch gate across thresholds from one embedding pass.
    sweep = {}
    for candidate in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        found, _ = components_at(candidate)
        kept = [m for m in found.values() if len(m) >= 2]
        sweep[f"{candidate:.2f}"] = {
            "identities": len(kept),
            "targets_in_groups_ge_2": sum(len(m) for m in kept),
        }

    components, edges = components_at(args.threshold)
    groups = {
        f"cosmic_id_{index:06d}": sorted(members)
        for index, members in enumerate(
            sorted((m for m in components.values() if len(m) >= 2), key=lambda m: m[0])
        )
    }
    grouped = sum(len(m) for m in groups.values())
    histogram = Counter(len(m) for m in groups.values())

    payload = {
        "schema_version": 1,
        "groups": groups,
        "audit": {
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "embedding_model": "buffalo_l/w600k_r50",
            "threshold": args.threshold,
            "min_face_res": args.min_face_res,
            "min_iou": args.min_iou,
            "backend": args.backend,
            "mutual_nn_edges": edges,
            "identities": len(groups),
            "targets_in_groups_ge_2": grouped,
            "group_size_histogram": dict(sorted(histogram.items())),
            "threshold_sweep": sweep,
            "mutual_nn_similarity": {
                "p10": float(np.percentile(best_similarity, 10)),
                "median": float(np.median(best_similarity)),
                "p90": float(np.percentile(best_similarity, 90)),
                "max": float(best_similarity.max()),
            },
            **dict(stats),
        },
    }
    output = Path(args.output)
    digest = seal(payload, output, args.write)
    print(json.dumps({**payload["audit"], "sha256": digest, "written": bool(args.write),
                      "output": str(output)}, indent=2, sort_keys=True))
    if grouped < args.required_targets:
        raise SystemExit(
            f"LAUNCH GATE FAILED: {grouped} targets in groups of >=2, "
            f"need >= {args.required_targets}. Do not launch CL1."
        )


def cmd_accept_list(args) -> None:
    records = load_manifest(args.manifest)
    root = Path(args.dataset_root)
    analyzer = build_analyzer(not args.cpu)

    accepted, stats = [], Counter()
    seen: set[str] = set()
    for record in records.values():
        if not isinstance(record, dict):
            continue
        face_bboxes = record.get("face_bboxes") or {}
        for reference_path in record.get("face_paths") or []:
            key = str(reference_path)
            if key in seen:
                continue
            seen.add(key)
            bbox = face_bboxes.get(key) or face_bboxes.get(key.lstrip("/"))
            if not valid_bbox(bbox, (256, 256)):
                stats["invalid_bbox"] += 1
                continue
            try:
                image = open_image(root, key)
            except Exception:
                stats["unreadable"] += 1
                continue
            faces = analyze_faces(analyzer, np.array(image)[:, :, ::-1])
            if not faces:
                stats["no_detection"] += 1
                continue
            face, overlap = best_face(faces, bbox)
            if face is None or overlap < args.min_iou:
                stats["bbox_mismatch"] += 1
                continue
            accepted.append(key)
            stats["accepted"] += 1
            if args.limit and len(seen) >= args.limit:
                break
        if args.limit and len(seen) >= args.limit:
            break

    examined = sum(stats.values())
    payload = {
        "schema_version": 1,
        "accepted": sorted(accepted),
        "audit": {
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "min_iou": args.min_iou,
            "examined": examined,
            "rejection_rate": (examined - stats["accepted"]) / examined if examined else 0.0,
            **dict(stats),
        },
    }
    output = Path(args.output)
    digest = seal(payload, output, args.write)
    print(json.dumps({**payload["audit"], "sha256": digest, "written": bool(args.write),
                      "output": str(output)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("groups", cmd_groups), ("accept-list", cmd_accept_list)):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--dataset-root", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--min-iou", type=float, default=0.3)
        p.add_argument("--limit", type=int, default=0, help="debug: stop after N items")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--write", action="store_true", help="required to write the file")
        p.set_defaults(func=handler)

    groups = sub.choices["groups"]
    groups.add_argument("--threshold", type=float, default=0.75)
    groups.add_argument("--min-face-res", type=int, default=192)
    groups.add_argument(
        "--backend", choices=("torch", "insightface"), default="torch",
        help="torch: E22's PyTorch w600k_r50 graph on the GPU (same weights, no "
             "landmark alignment). insightface: ONNX detect+align, CPU-only in this env.",
    )
    groups.add_argument("--arcface-onnx", default=DEFAULT_ARCFACE)
    groups.add_argument("--device", default="cuda")
    groups.add_argument("--batch-size", type=int, default=64)
    groups.add_argument("--workers", type=int, default=8)
    groups.add_argument("--save-embeddings", default=None,
                        help="write paths+embeddings .npz for offline linkage studies")
    groups.add_argument(
        "--required-targets",
        type=int,
        default=3000,
        help="CL1 launch gate: minimum targets in groups of >=2",
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
