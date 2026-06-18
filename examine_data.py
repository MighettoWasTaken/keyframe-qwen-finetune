# count_fluentdev_keyframes.py
# pip install datasets

import re

from datasets import load_dataset

DATASET_NAME = "justmalhar/fluent-dev"

KEYFRAMES_RE = re.compile(r"@keyframes\b", re.IGNORECASE)


def row_text(row: dict) -> str:
    parts = []
    for value in row.values():
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def extract_keyframes(text: str):
    """
    Not a perfect CSS parser, but good enough to grab @keyframes blocks.
    """
    blocks = []

    idx = 0
    while True:
        start = text.lower().find("@keyframes", idx)
        if start == -1:
            break

        brace = text.find("{", start)
        if brace == -1:
            break

        depth = 1
        end = brace + 1

        while end < len(text) and depth > 0:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1

        blocks.append(text[start:end])
        idx = end

    return blocks


def main():
    ds = load_dataset(DATASET_NAME)

    total = 0
    with_keyframes = 0

    samples = []

    print(f"Loaded splits: {list(ds.keys())}\n")

    for split_name, split in ds.items():
        split_total = len(split)
        split_keyframes = 0

        for row in split:
            text = row_text(row)

            if KEYFRAMES_RE.search(text):
                split_keyframes += 1

                if len(samples) < 20:
                    blocks = extract_keyframes(text)
                    if blocks:
                        samples.append(blocks[0])

        total += split_total
        with_keyframes += split_keyframes

        print(f"{split_name}:")
        print(f"  total rows: {split_total}")
        print(f"  rows with @keyframes: {split_keyframes}")

    print("\nTOTAL:")
    print(f"  total rows: {total}")
    print(f"  rows with @keyframes: {with_keyframes}")
    print(f"  percent with @keyframes: {100 * with_keyframes / total:.2f}%")

    print("\n" + "=" * 80)
    print("FIRST 20 KEYFRAME SAMPLES")
    print("=" * 80)

    for i, sample in enumerate(samples, start=1):
        print(f"\n--- SAMPLE {i} ---")
        print(sample[:3000])  # prevent giant outputs


if __name__ == "__main__":
    main()
