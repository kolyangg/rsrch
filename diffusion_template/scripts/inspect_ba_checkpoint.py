#!/usr/bin/env python
"""Probe branched-attention LoRA deltas in a saved checkpoint.

Reports per-group (attn1/attn2 x noise/ref x q/k/v) Frobenius-norm stats of the
BranchLoRALinear deltas (B @ A, scaling = alpha/rank = 1.0) plus the top sites.
Works on CPU with either weights-only or full checkpoints.

Usage:
    python scripts/inspect_ba_checkpoint.py saved/<run>/weights-epoch1.pth [more.pth ...]
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys

import torch


def load_attn_processors(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and "attn_processors" not in sd:
        sd = sd["state_dict"]
    ap = sd.get("attn_processors") if isinstance(sd, dict) else None
    if not ap:
        raise KeyError(f"No 'attn_processors' entry found in {path}")
    return ap


def probe(path: str, top_k: int = 10) -> None:
    ap = load_attn_processors(path)
    rows = []
    for site, tensors in ap.items():
        kind = "attn1" if ".attn1." in site else "attn2"
        for key in tensors:
            if not key.endswith(".lora_A"):
                continue
            stem = key[: -len(".lora_A")]  # e.g. ref_to_q
            branch, proj = stem.rsplit("_to_", 1)
            A = tensors[f"{stem}.lora_A"].float()
            B = tensors[f"{stem}.lora_B"].float()
            delta = B @ A  # BranchLoRALinear scaling = alpha/rank = 1.0
            rows.append((site, kind, branch, proj, delta.norm().item(), delta.abs().max().item()))

    if not rows:
        print(f"{path}: no LoRA tensors found (kind=full checkpoint?)")
        return

    groups = collections.defaultdict(list)
    for _site, kind, branch, proj, fro, _mx in rows:
        groups[(kind, branch, proj)].append(fro)

    print(f"\n=== {path} ===")
    print(f"{'group':30s} {'n':>4s} {'mean_fro':>9s} {'max_fro':>9s}")
    for k in sorted(groups):
        v = groups[k]
        print(f"{str(k):30s} {len(v):4d} {statistics.mean(v):9.4f} {max(v):9.4f}")

    rows.sort(key=lambda r: -r[4])
    print(f"\nTop-{top_k} sites by delta Frobenius norm:")
    for site, _kind, branch, proj, fro, mx in rows[:top_k]:
        print(f"  {fro:8.3f} absmax={mx:8.4f}  {branch}_to_{proj:<2s} {site}")

    zero = sum(1 for r in rows if r[4] < 1e-8)
    print(f"\ntensors total={len(rows)}, exactly-zero deltas={zero}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="checkpoint .pth paths")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    for path in args.checkpoints:
        probe(path, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
