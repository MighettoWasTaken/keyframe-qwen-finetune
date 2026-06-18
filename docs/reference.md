# Reference

## File & directory layout

```
24hr/
├── prepare_data.py        # build base dataset from justmalhar/fluent-dev
├── generate_data.py       # supplement with Claude-generated animations
├── dataset_utils.py       # shared: classify, render filter, preview (no __main__)
├── examine_data.py        # one-off exploration of fluent-dev (not in pipeline)
├── train.py               # QLoRA SFT (Qwen2.5-Coder-1.5B)
├── generate.py            # inference with base + LoRA adapter
├── conftest.py            # adds project root to sys.path for tests
├── pyproject.toml         # deps, Python 3.12, CUDA-12.4 torch index
├── uv.lock
├── .env.example           # template (.env is gitignored)
├── tests/
│   └── test_cascade.py    # unit tests for the generation engine (fakes only)
├── data/                  # gitignored — datasets + previews
│   ├── keyframes_all.jsonl          # base dataset (prepare_data.py)
│   ├── keyframes_train.jsonl        # training split (see Known gaps)
│   ├── keyframes_val.jsonl          # validation split (see Known gaps)
│   ├── preview.html
│   └── generated/
│       ├── prompts.jsonl            # Stage A motion briefs
│       ├── keyframes_generated.jsonl# Stage B kept animations
│       └── preview_generated.html
├── outputs/               # gitignored — model adapters / checkpoints
└── docs/                  # this documentation
```

`data/` and `outputs/` are **gitignored** (datasets and model artifacts are too
large for git); only `.gitkeep` placeholders are tracked.

## JSONL schemas

**Training records** (`keyframes_all.jsonl`, `keyframes_train/val.jsonl`,
`keyframes_generated.jsonl`):

```json
{
  "messages": [
    {"role": "user", "content": "<motion description>"},
    {"role": "assistant", "content": "<HTML + inline <style> with @keyframes>"}
  ],
  "buckets": ["rotate", "opacity", "multi_keyframe"],
  "category": "<source-defined>",
  "source": "fluent-dev" | "claude"
}
```

**Motion briefs** (`prompts.jsonl`, Stage A output):

```json
{"description": "<one-sentence motion brief>", "techniques": ["skew", "color"]}
```

## Environment & config

- **Python:** 3.12 (`>=3.12,<3.13`).
- **Dependency manager:** [uv](https://docs.astral.sh/uv/). `uv sync` installs
  everything, including the CUDA-12.4 torch wheel (routed via
  `[tool.uv.sources]` + the `pytorch-cu124` index).
- **Browser for the render check:** `uv run playwright install chromium` once.
- **`.env`** (copy from `.env.example`):
  - `ANTHROPIC_API_KEY` — required for `generate_data.py` (loaded via
    `python-dotenv`).
  - `HF_TOKEN` — optional; raises HF rate limits / download speed for
    `prepare_data.py`.
  - Weights & Biases: `train.py` logs to W&B; `wandb login` or set `WANDB_*`, or
    change `report_to` to `"none"`.

## Buckets

The 16 technique buckets (`ALL_BUCKETS`), in order:

```
rotate, translate, scale, skew, opacity, color, size, stroke, shadow, clip,
background_position, border_radius, staggered, hover, multi_keyframe, other
```

`FOCUS_BUCKETS` (the thin ones `generate_data.py` actively pushes for):

```
stroke, border_radius, clip, skew, color, shadow, size, background_position, staggered
```

Rule definitions and detection patterns: `_BUCKET_RULES` in `dataset_utils.py`
(see [data-pipeline.md](data-pipeline.md#technique-classification--classifycode---liststr)).

## Tunable constants

**Render filter** (`dataset_utils.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `BLANK_THRESHOLD` | 0.01 | min non-white pixel fraction for a frame to count as "has content" |
| `RENDER_SAMPLES` | 6 | frames sampled across the timeline |
| `RENDER_SAMPLE_GAP` | 400 ms | gap between sampled frames |
| `RENDER_MIN_VISIBLE` | 0.6 | fraction of frames that must have content |
| `MOTION_PIXEL_DELTA` | 20 | grayscale delta for a pixel to count as "changed" |
| `MOTION_MIN_FRACTION` | 0.005 | fraction of pixels that must change between some frame pair |
| `RENDER_CONCURRENCY` | 12 | parallel browser pages |
| `PREVIEW_LIMIT` | 40 | max cards in an HTML preview |

**Generation** (`generate_data.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MODEL` | `claude-haiku-4-5` | default Stage A model |
| `MODEL_CASCADE` | haiku → sonnet → opus | Stage B escalation order |
| `PROMPT_GEN_CHUNK` | 25 | briefs requested per Stage A call |
| `MAX_STALLED_CHUNKS` | 6 | consecutive all-duplicate chunks before Stage A stops |
| `DEDUP_THRESHOLD` | 0.82 | `SequenceMatcher` ratio above which two briefs are "the same" |
| `MOTIF_POOL` | ~38 | subject motifs sampled per chunk for diversity |
| `PRICING` | per-model | $/1M tokens, for cost reporting only |

## Tests

```bash
uv run pytest -q
```

`tests/test_cascade.py` (13 tests) covers the generation engine with **fakes** —
no API calls, no browser:

- cascade escalation (parse failures, validation failures, partial-batch
  escalation, last-tier drop, leftover partial batch),
- validator composition (judge runs only on render-passers; judge can reject a
  match; render-only when no judge),
- dedup (exact + near-duplicate, within-batch),
- usage accounting (per-tier tracking, per-model pricing).

`conftest.py` puts the project root on `sys.path` so `import generate_data`
works.

## Known gaps

- **No train/val split script.** `train.py` reads `data/keyframes_train.jsonl`,
  but nothing in the repo produces `keyframes_train.jsonl` /
  `keyframes_val.jsonl` from `keyframes_all.jsonl` (+ the generated rows). The
  split is currently manual/ad-hoc. A small `split_data.py` that merges
  `keyframes_all.jsonl` + `data/generated/keyframes_generated.jsonl`, shuffles,
  and writes a stratified split would close this.
- **Generated data isn't auto-merged into training.** `keyframes_generated.jsonl`
  must be folded into the training split by hand before it influences `train.py`.
- **`train.py` config is hardcoded.** No CLI; edit the `Args` dataclass to change
  the model, data path, or hyperparameters.
- **Prompt-cache engagement is model-dependent.** The shared system prefix
  (~5.3k chars) may fall below a tier's minimum cacheable prefix; watch the
  per-model `cache_read` line in the usage report to confirm caching is active.
