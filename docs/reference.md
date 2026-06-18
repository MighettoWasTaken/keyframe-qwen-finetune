# Reference

## File & directory layout

```
24hr/
├── prepare_data.py        build base dataset from justmalhar/fluent-dev
├── generate_data.py       supplement with Claude-generated animations
├── split_data.py          merge sources + iterative multilabel stratification
├── dataset_utils.py       shared: render filter, classifier, preview (no __main__)
├── prompting.py           single source of truth for system prompt + chat template
├── train.py               bf16 LoRA SFT (Qwen2.5-Coder-3B-Instruct, TRL)
├── eval.py                knockout evaluation harness
├── verify_model.py        GPU smoke test
├── generate.py            simple inference script
├── make_plots.py          training curves + metrics comparison plots
├── make_preview.py        side-by-side HTML preview of base vs fine-tuned
├── examine_data.py        one-off exploration (not in pipeline)
├── conftest.py            adds project root to sys.path for tests
├── pyproject.toml         deps, Python 3.12, CUDA-12.4 torch index
├── uv.lock
├── .env.example           template (.env is gitignored)
├── tests/
│   └── test_cascade.py    unit tests for generation engine (fakes only)
├── data/
│   ├── keyframes_all.jsonl          base dataset (prepare_data.py)
│   ├── keyframes_train.jsonl        training split (494 examples)
│   ├── keyframes_val.jsonl          val split (130 examples)
│   ├── preview.html                 gallery of base dataset
│   └── generated/
│       ├── prompts.jsonl            Stage A motion briefs
│       ├── keyframes_generated.jsonl Stage B kept animations
│       └── preview_generated.html
├── outputs/
│   ├── eval/
│   │   ├── base.jsonl               per-example eval cache (base model)
│   │   ├── base_metrics.json        aggregated metrics (base model)
│   │   ├── ft.jsonl                 per-example eval cache (fine-tuned)
│   │   └── ft_metrics.json          aggregated metrics (fine-tuned)
│   ├── plots/
│   │   ├── training_curves.png
│   │   └── metrics_comparison.png
│   ├── preview.html                 base vs fine-tuned animation preview
│   └── qwen3b-keyframes-lora/       LoRA adapter (gitignored, >1GB)
│       ├── train_log.jsonl          per-step loss log for plotting
│       └── checkpoint-*/
└── docs/
```

## JSONL schemas

**Training records** (`keyframes_all.jsonl`, `keyframes_train/val.jsonl`, `keyframes_generated.jsonl`):

```json
{
  "messages": [
    {"role": "user",      "content": "<motion description>"},
    {"role": "assistant", "content": "<HTML + inline <style> with @keyframes>"}
  ],
  "buckets": ["rotate", "opacity", "multi_keyframe"],
  "source": "fluent-dev" | "claude"
}
```

**Motion briefs** (`prompts.jsonl`, Stage A output):

```json
{"description": "<one-sentence motion brief>", "techniques": ["skew", "color"]}
```

**Train log** (`train_log.jsonl`):

```json
{"step": 10, "epoch": 0.32, "loss": 1.23, "learning_rate": 1.8e-4, "grad_norm": 0.9}
{"step": 10, "epoch": 0.32, "eval_loss": 1.15}
```

**Eval cache** (`outputs/eval/<tag>.jsonl`), one record per val example:

```json
{
  "description": "...", "reference": "...",
  "output": "...", "code_ext": "...",
  "render_raw": true, "render_ext": true,
  "match_raw": true,  "match_ext": true,
  "nll": 4.2,         "n_tokens": 312
}
```

## Environment & config

- **Python:** 3.12 (`>=3.12,<3.13`).
- **Dependency manager:** [uv](https://docs.astral.sh/uv/). `uv sync` installs everything including the CUDA 12.4 torch wheel.
- **Browser:** `uv run playwright install chromium` once, required for the render filter.
- **`.env`** (copy from `.env.example`):
  - `ANTHROPIC_API_KEY` — required for `generate_data.py` and `eval.py` judge phase.
  - `HF_TOKEN` — optional, raises HF rate limits / download speed.
  - `WANDB_API_KEY` — optional; pass `--report-to none` to skip W&B entirely.

## Animation technique buckets

The 16 technique buckets (`ALL_BUCKETS`):

```
rotate, translate, scale, skew, opacity, color, size, stroke, shadow, clip,
background_position, border_radius, staggered, hover, multi_keyframe, other
```

`FOCUS_BUCKETS` (thin buckets `generate_data.py` actively pushes for):

```
stroke, border_radius, clip, skew, color, shadow, size, background_position, staggered
```

## Tunable constants

**Render filter** (`dataset_utils.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `BLANK_THRESHOLD` | 0.01 | min non-white pixel fraction for "has content" |
| `RENDER_SAMPLES` | 6 | frames sampled per snippet |
| `RENDER_SAMPLE_GAP` | 400 ms | gap between frames |
| `RENDER_MIN_VISIBLE` | 0.6 | fraction of frames that must have content |
| `MOTION_PIXEL_DELTA` | 20 | grayscale delta to count a pixel as "changed" |
| `MOTION_MIN_FRACTION` | 0.005 | fraction of pixels that must change across some frame pair |
| `RENDER_CONCURRENCY` | 12 | parallel browser pages |

**Generation** (`generate_data.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MODEL_CASCADE` | haiku → sonnet → opus | Stage B escalation order |
| `PROMPT_GEN_CHUNK` | 25 | briefs per Stage A API call |
| `MAX_STALLED_CHUNKS` | 6 | consecutive all-duplicate chunks before Stage A stops |
| `DEDUP_THRESHOLD` | 0.82 | SequenceMatcher ratio for near-duplicate detection |
| `AVOID_SAMPLE` | 60 | max existing descriptions shown to model as avoid-list |
| `MOTIF_POOL` | ~38 | subject motifs rotated per chunk for diversity |

## Tests

```bash
uv run pytest -q
```

`tests/test_cascade.py` covers the generation engine with fakes — no API calls, no browser:

- Cascade escalation (parse failures, validation failures, partial-batch escalation, last-tier drop)
- Validator composition (judge runs only on render-passers; render-only mode)
- Dedup (exact + near-duplicate, within-batch)
- Usage accounting (per-tier tracking, per-model pricing)
