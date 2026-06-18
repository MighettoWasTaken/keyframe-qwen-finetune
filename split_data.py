"""
Combine the source dataset (data/keyframes_all.jsonl) and the Claude-generated
dataset (data/generated/keyframes_generated.jsonl) into one pool, then split it
80/20 into train/val with a *representative* validation set.

Representativeness is enforced with iterative (multilabel-aware) stratification
(Sechidis et al., 2011 — the same method scikit-multilearn uses). Each example
carries several @keyframes technique buckets at once, so a single "stratify by
column" doesn't apply; instead every stratification signal is expressed as a
label and balanced jointly:

  - animation-type buckets   (rotate, translate, ... — multilabel)
  - source                   (fluent-dev vs claude)
  - prompt length            (quantile bin of the user-message length)
  - content length           (quantile bin of the assistant-code length)
  - complexity               (how many buckets the example spans)

The script prints a per-label comparison of train vs val vs overall so the split
can be eyeballed for drift, and is deterministic given --seed.

Usage:
  uv run python split_data.py                 # 80/20, default paths
  uv run python split_data.py --val-frac 0.1 --seed 7
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from dataset_utils import ALL_BUCKETS, DATA_DIR


# ---------------------------------------------------------------------------
# Load + de-duplicate
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, source: str) -> list[dict]:
    """Read a dataset file, stamping `source` from the origin file (the source
    field is missing in the older keyframes_all.jsonl, so we don't trust it)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["source"] = source
        rows.append(r)
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Drop exact-duplicate (prompt, code) pairs so the same example can't land
    in both train and val (the one real leakage risk for an SFT split)."""
    seen, out = set(), []
    for r in rows:
        key = (r["messages"][0]["content"], r["messages"][1]["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Stratification labels
# ---------------------------------------------------------------------------

def quantile_edges(values: list[int], bins: int) -> np.ndarray:
    """Inner cut points splitting `values` into `bins` roughly-equal quantiles.
    Duplicates are removed, so degenerate distributions just yield fewer bins."""
    qs = np.linspace(0, 100, bins + 1)[1:-1]
    return np.unique(np.percentile(values, qs)) if len(values) else np.array([])


def bin_index(value: int, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def make_labeler(rows: list[dict], length_bins: int):
    plen = [len(r["messages"][0]["content"]) for r in rows]
    clen = [len(r["messages"][1]["content"]) for r in rows]
    plen_edges = quantile_edges(plen, length_bins)
    clen_edges = quantile_edges(clen, length_bins)

    def labels(r: dict) -> set[str]:
        ls = {f"bucket:{b}" for b in r.get("buckets", [])} or {"bucket:none"}
        ls.add(f"source:{r['source']}")
        ls.add(f"plen:{bin_index(len(r['messages'][0]['content']), plen_edges)}")
        ls.add(f"clen:{bin_index(len(r['messages'][1]['content']), clen_edges)}")
        ls.add(f"complexity:{min(len(r.get('buckets', [])), 5)}")
        return ls

    return labels


# ---------------------------------------------------------------------------
# Iterative stratification
# ---------------------------------------------------------------------------

def iterative_stratify(label_sets: list[set[str]], val_frac: float,
                       seed: int) -> list[bool]:
    """Return a mask where True == validation. Greedy iterative stratification:
    repeatedly take the rarest still-unassigned label and place its examples into
    whichever split most needs that label, keeping every label's train/val ratio
    close to `val_frac` simultaneously."""
    rng = random.Random(seed)
    n = len(label_sets)
    ratios = {"val": val_frac, "train": 1.0 - val_frac}

    # Desired remaining counts, overall and per-label-per-split.
    desired = {s: ratios[s] * n for s in ratios}
    label_total = Counter(l for ls in label_sets for l in ls)
    desired_l = {l: {s: ratios[s] * c for s in ratios} for l, c in label_total.items()}

    assignment: dict[int, str] = {}
    remaining = set(range(n))

    while remaining:
        rem_count = Counter()
        for i in remaining:
            for l in label_sets[i]:
                rem_count[l] += 1
        if not rem_count:                      # examples with no labels (rare)
            for i in list(remaining):
                s = max(ratios, key=lambda s: (desired[s], rng.random()))
                assignment[i] = s
                desired[s] -= 1
                remaining.discard(i)
            break

        # Rarest remaining label (ties broken randomly), assign all its examples.
        fewest = min(rem_count.values())
        label = rng.choice([l for l, c in rem_count.items() if c == fewest])
        members = [i for i in remaining if label in label_sets[i]]
        rng.shuffle(members)
        for i in members:
            s = max(ratios, key=lambda s: (desired_l[label][s], desired[s], rng.random()))
            assignment[i] = s
            desired[s] -= 1
            for l in label_sets[i]:
                desired_l[l][s] -= 1
            remaining.discard(i)

    return [assignment[i] == "val" for i in range(n)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(rows: list[dict], is_val: list[bool], labeler):
    train = [r for r, v in zip(rows, is_val) if not v]
    val = [r for r, v in zip(rows, is_val) if v]
    print(f"\nSplit: {len(train)} train / {len(val)} val "
          f"({len(val) / len(rows):.1%} val) from {len(rows)} total\n")

    # Per-label share landing in val — want each close to the target val fraction.
    target = len(val) / len(rows)
    overall = Counter(l for r in rows for l in labeler(r))
    val_cnt = Counter(l for r in val for l in labeler(r))

    def show(group: str, keys: list[str]):
        print(f"  {group:<12} {'total':>6} {'val':>5} {'val%':>7}  (target {target:.0%})")
        for k in keys:
            if overall.get(k):
                frac = val_cnt.get(k, 0) / overall[k]
                flag = "  <-- skew" if abs(frac - target) > 0.12 else ""
                print(f"    {k:<22} {overall[k]:>6} {val_cnt.get(k, 0):>5} {frac:>7.1%}{flag}")

    show("buckets", [f"bucket:{b}" for b in ALL_BUCKETS])
    show("source", sorted(k for k in overall if k.startswith("source:")))
    show("prompt-len", sorted((k for k in overall if k.startswith("plen:")), key=lambda k: int(k.split(":")[1])))
    show("content-len", sorted((k for k in overall if k.startswith("clen:")), key=lambda k: int(k.split(":")[1])))
    show("complexity", sorted((k for k in overall if k.startswith("complexity:")), key=lambda k: int(k.split(":")[1])))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-bins", type=int, default=4,
                        help="Quantile bins for prompt/content length stratification")
    parser.add_argument("--all", default=str(DATA_DIR / "keyframes_all.jsonl"))
    parser.add_argument("--generated", default=str(DATA_DIR / "generated" / "keyframes_generated.jsonl"))
    parser.add_argument("--train-out", default=str(DATA_DIR / "keyframes_train.jsonl"))
    parser.add_argument("--val-out", default=str(DATA_DIR / "keyframes_val.jsonl"))
    args = parser.parse_args()

    rows = (load_jsonl(Path(args.all), "fluent-dev")
            + load_jsonl(Path(args.generated), "claude"))
    if not rows:
        raise SystemExit("No input rows found — run prepare_data.py / generate_data.py first.")

    before = len(rows)
    rows = dedupe(rows)
    if len(rows) < before:
        print(f"Dropped {before - len(rows)} exact-duplicate example(s).")

    labeler = make_labeler(rows, args.length_bins)
    label_sets = [labeler(r) for r in rows]
    is_val = iterative_stratify(label_sets, args.val_frac, args.seed)

    train = [r for r, v in zip(rows, is_val) if not v]
    val = [r for r, v in zip(rows, is_val) if v]

    for path, split in [(args.train_out, train), (args.val_out, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    report(rows, is_val, labeler)
    print(f"\nWrote {len(train)} -> {args.train_out}")
    print(f"Wrote {len(val)} -> {args.val_out}")


if __name__ == "__main__":
    main()
