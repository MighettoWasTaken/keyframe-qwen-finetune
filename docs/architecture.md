# Architecture

## Goal

Fine-tune a small, local code model to turn a one-sentence motion brief into a
minimal, self-contained CSS `@keyframes` animation. The hard part isn't the
training loop — it's assembling a **clean, well-balanced dataset** of animations
that actually render and move. Most of the code here is about data quality.

## Pipeline overview

```
                         justmalhar/fluent-dev  (HF dataset)
                                    │
            examine_data.py ────────┤  (exploratory only; not in the pipeline)
                                    │
                                    ▼
                            prepare_data.py
              ┌── filter @keyframes ──► render_filter ──► classify ──┐
              │                                                       ▼
              │                                        data/keyframes_all.jsonl
              │                                        data/preview.html
              │
              ▼
        generate_data.py  (optional supplementation, Anthropic API)
          Stage A: motion briefs ──► dedup ──► data/generated/prompts.jsonl
          Stage B: animations  ──► render_filter + judge ──► cascade
                                                       │
                                                       ▼
                                    data/generated/keyframes_generated.jsonl
                                    data/generated/preview_generated.html
              │
              ▼
        (manual / TODO) merge + split  ──►  data/keyframes_train.jsonl
                                            data/keyframes_val.jsonl
              │
              ▼
            train.py  (QLoRA SFT, TRL)  ──►  outputs/qwen-keyframes-lora
              │
              ▼
           generate.py  (load base + LoRA adapter, run inference)
```

`dataset_utils.py` is the shared spine: `prepare_data.py` and `generate_data.py`
both import its classifier, render filter, and preview builder so sourced and
generated data are bucketed and filtered **identically**.

## Components

| File | Role | Entry point |
|------|------|-------------|
| `examine_data.py` | One-off exploration: counts how many fluent-dev rows contain `@keyframes` and prints samples. Not part of the build. | `uv run python examine_data.py` |
| `prepare_data.py` | Builds the base dataset from `justmalhar/fluent-dev`. | `uv run python prepare_data.py` |
| `dataset_utils.py` | Shared library: keyframe extraction, technique classification, headless render filter, bucket summary, HTML preview. No `__main__`. | imported |
| `generate_data.py` | Supplements thin buckets with Claude-generated animations (two-stage, cascading, cached). | `uv run python generate_data.py` |
| `train.py` | 4-bit QLoRA SFT of Qwen2.5-Coder-1.5B via TRL's `SFTTrainer`. | `uv run python train.py` |
| `generate.py` | Loads the base model + trained LoRA adapter and runs inference. | `uv run python generate.py` |
| `tests/test_cascade.py` | Unit tests for the generation engine using fakes (no API/browser). | `uv run pytest -q` |

## The training-example format

Every example — sourced or generated — is a chat-style record:

```json
{
  "messages": [
    {"role": "user", "content": "<motion description / instruction>"},
    {"role": "assistant", "content": "<HTML + inline <style> with @keyframes>"}
  ],
  "buckets": ["rotate", "opacity", "multi_keyframe"],
  "category": "generated",
  "source": "fluent-dev" | "claude"
}
```

`buckets` are the animation techniques detected by the classifier (see
[data-pipeline.md](data-pipeline.md)); `source` records where the row came from
so the two pools can be told apart downstream.

## Design principles

- **One quality bar, applied everywhere.** The same `render_filter` gates both
  the HF data and the Claude generations — nothing static or blank gets in.
- **Balance by construction.** The classifier buckets every row, and the
  generator is steered toward the *under-represented* buckets rather than
  generating blindly.
- **Cheapest-first generation.** Supplementation starts on the cheapest model and
  only escalates rows that fail, keeping API spend low (see
  [data-generation.md](data-generation.md)).
- **Resumable, append-only data.** Generation flushes to disk incrementally and
  dedupes against what already exists, so runs can be stopped and resumed.
- **Testable core.** The generation cascade, validator composition, dedup, and
  cost accounting are pure functions exercised by `pytest` with fakes — no
  network or browser in the test path.
