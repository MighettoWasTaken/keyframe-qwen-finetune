# Data pipeline — `prepare_data.py` + `dataset_utils.py`

This stage turns the public `justmalhar/fluent-dev` instruction dataset into a clean, bucketed pool of CSS `@keyframes` animations that provably render and move.

## `prepare_data.py`

```bash
uv run python prepare_data.py              # full pipeline (with render check)
uv run python prepare_data.py --no-render-check   # faster, skips motion check
```

Steps:

1. **Load** `justmalhar/fluent-dev` and combine its `train` + `validation` splits.
2. **Filter** to rows whose `code` contains `@keyframes` (`KEYFRAMES_RE`).
3. **Render-check** every candidate with `render_filter` (skippable via `--no-render-check`). Blank or static rows are dropped.
4. **Classify** each survivor into technique buckets via `classify`.
5. **Write** `data/keyframes_all.jsonl` (chat-format records, `source: "fluent-dev"`) and print a bucket-coverage summary.
6. **Build** `data/preview.html`, a visual gallery of the kept animations.

## `split_data.py`

```bash
uv run python split_data.py         # 80/20 split, seed 42
uv run python split_data.py --val-frac 0.15 --seed 7
```

Merges `data/keyframes_all.jsonl` + `data/generated/keyframes_generated.jsonl` (if it exists), dedupes exact `(prompt, code)` pairs, then splits using **iterative multilabel stratification** (Sechidis et al. 2011). Labels used: `bucket:X` for each animation technique, `source:X`, `plen:N` (prompt length bucket), `clen:N` (code length bucket), `complexity:N` (bucket count).

Produces `data/keyframes_train.jsonl` (494 examples) and `data/keyframes_val.jsonl` (130 examples).

## `dataset_utils.py` — the shared spine

Imported by `prepare_data.py`, `generate_data.py`, and `eval.py` so all three filter and label identically.

### Keyframe extraction — `extract_keyframes_blocks(code)`

A small brace-matching scanner that pulls each `@keyframes { ... }` block out of a code string. Used by the classifier and the preview builder.

### Technique classification — `classify(code) -> list[str]`

Runs a table of regex rules (`_BUCKET_RULES`) and returns every matching bucket. Most rules only look **inside** the `@keyframes` blocks (so a static `color:` on the element doesn't count as a color *animation*); structural rules scan the full code.

| Bucket | Detects | Searched in |
|--------|---------|-------------|
| `rotate` | `rotate(` | keyframes only |
| `translate` | `translate[XYZ/3d](` | keyframes only |
| `scale` | `scale(` / `scale[XYZ](` | keyframes only |
| `skew` | `skew[XY](` | keyframes only |
| `opacity` | `opacity:` | keyframes only |
| `color` | `background-color` / `border-color` / `color:` | keyframes only |
| `size` | `width:` / `height:` | keyframes only |
| `stroke` | `stroke-dasharray` / `stroke-dashoffset` | keyframes only |
| `shadow` | `box-shadow:` | keyframes only |
| `clip` | `clip-path:` | keyframes only |
| `background_position` | `background-position:` | keyframes only |
| `border_radius` | `border-radius:` | keyframes only |
| `staggered` | `animation-delay:` / `nth-child(` | full code |
| `hover` | `:hover {` | full code |
| `multi_keyframe` | more than one `@keyframes` block | structural |
| `other` | fallback when nothing matched | — |

### Render filter — `render_filter(codes) -> list[bool]`

The quality backbone. Runs Chromium headless (Playwright) over many snippets concurrently. For each snippet it:

1. Wraps bare fragments in a centered 300×300 white page. Full HTML documents (`<html>`) are passed through untouched.
2. Loads the page, nudges the mouse to center to trigger `:hover` states.
3. Samples `RENDER_SAMPLES` (6) screenshots `RENDER_SAMPLE_GAP` (400ms) apart.
4. Requires **content** — at least `RENDER_MIN_VISIBLE` (0.6) of frames must have a non-white pixel fraction above `BLANK_THRESHOLD` (0.01).
5. Requires **motion** — between some adjacent frame pair, at least `MOTION_MIN_FRACTION` (0.005) of pixels must change (grayscale delta > `MOTION_PIXEL_DELTA`).

A snippet must pass **both** checks. Any exception (timeout, bad markup) is treated as a failure. Concurrency capped at `RENDER_CONCURRENCY` (12) pages.
