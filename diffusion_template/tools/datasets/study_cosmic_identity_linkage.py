#!/usr/bin/env python3
"""Decide whether Cosmic Large can supply multi-view identities for CL1.

Mutual-nearest-neighbour linkage can only ever emit pairs, so it cannot answer
"does this package contain identities with 3+ distinct targets?". This builds the
full similarity graph on the GPU and compares linkages, and — the number that
actually matters — counts components built only from *genuine* same-identity
edges, excluding near-duplicate edges that would be self-reference leakage.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def components(edges, count):
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return [m for m in groups.values() if len(m) >= 2]


def summarize(members, count):
    sizes = [len(m) for m in members]
    return {
        "identities": len(members),
        "targets_in_components_ge_2": int(sum(sizes)),
        "targets_in_components_ge_3": int(sum(s for s in sizes if s >= 3)),
        "identities_ge_3": int(sum(1 for s in sizes if s >= 3)),
        "max_component": int(max(sizes)) if sizes else 0,
        "coverage_pct": round(100.0 * sum(sizes) / count, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duplicate-cosine", type=float, default=0.95,
                        help="edges at or above this are treated as the same photo")
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()

    import torch

    payload = np.load(args.embeddings, allow_pickle=True)
    matrix = torch.from_numpy(payload["embeddings"]).float().to(args.device)
    count = matrix.shape[0]

    report = {"targets": count, "duplicate_cosine": args.duplicate_cosine, "linkage": {}}
    for threshold in (0.65, 0.70, 0.75, 0.80):
        all_edges, genuine_edges, duplicate_edges = [], [], 0
        for start in range(0, count, args.chunk):
            block = matrix[start : start + args.chunk] @ matrix.T
            rows = torch.arange(block.shape[0], device=block.device)[:, None] + start
            cols = torch.arange(count, device=block.device)[None, :]
            keep = (block >= threshold) & (cols > rows)      # upper triangle only
            idx = keep.nonzero(as_tuple=False)
            if idx.numel():
                scores = block[idx[:, 0], idx[:, 1]]
                pairs = torch.stack([idx[:, 0] + start, idx[:, 1]], dim=1).cpu().numpy()
                scores = scores.cpu().numpy()
                all_edges.extend(pairs.tolist())
                genuine = pairs[scores < args.duplicate_cosine]
                duplicate_edges += int((scores >= args.duplicate_cosine).sum())
                genuine_edges.extend(genuine.tolist())
            del block

        report["linkage"][f"{threshold:.2f}"] = {
            "edges_total": len(all_edges),
            "edges_near_duplicate": duplicate_edges,
            "edges_genuine": len(genuine_edges),
            "threshold_graph_all_edges": summarize(components(all_edges, count), count),
            "threshold_graph_genuine_only": summarize(components(genuine_edges, count), count),
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
