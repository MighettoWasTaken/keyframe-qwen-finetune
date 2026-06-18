"""
Load justmalhar/fluent-dev, combine all splits, filter for @keyframes,
run a headless render check, classify by animation type, then save:
  - data/keyframes_all.jsonl     all passing samples with bucket metadata
  - data/preview.html            gallery with bucket tags

Usage:
  uv run python prepare_data.py              # full pipeline with render check
  uv run python prepare_data.py --no-render-check
"""

import argparse
import json

from datasets import load_dataset

from dataset_utils import (
    DATA_DIR,
    KEYFRAMES_RE,
    build_preview,
    classify,
    print_bucket_summary,
    render_filter,
)

DATASET_NAME = "justmalhar/fluent-dev"


def to_example(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["code"]},
        ],
        "buckets": row["_buckets"],
        "category": row.get("category", ""),
        "source": "fluent-dev",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render-check", action="store_true",
                        help="Skip headless render filtering (faster, less clean)")
    args = parser.parse_args()

    ds = load_dataset(DATASET_NAME)
    DATA_DIR.mkdir(exist_ok=True)

    # Combine all splits — we make our own split later from the full pool
    all_rows = list(ds["train"]) + list(ds["validation"])
    print(f"Combined: {len(all_rows)} total rows")

    kf_rows = [r for r in all_rows if KEYFRAMES_RE.search(r.get("code", ""))]
    print(f"With @keyframes: {len(kf_rows)}")

    if not args.no_render_check:
        mask = render_filter([r["code"] for r in kf_rows])
        passed = [r for r, ok in zip(kf_rows, mask) if ok]
        print(f"Dropped {len(kf_rows) - len(passed)} blank renders, {len(passed)} kept")
    else:
        passed = kf_rows

    for row in passed:
        row["_buckets"] = classify(row["code"])

    print_bucket_summary(passed)

    out_path = DATA_DIR / "keyframes_all.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in passed:
            f.write(json.dumps(to_example(row)) + "\n")
    print(f"Saved {len(passed)} examples -> {out_path}")

    build_preview(passed, DATA_DIR / "preview.html", title="@keyframes dataset (fluent-dev)")


if __name__ == "__main__":
    main()
