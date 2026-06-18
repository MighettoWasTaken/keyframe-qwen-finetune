# Data pipeline — `prepare_data.py` + `dataset_utils.py`

This stage turns the public `justmalhar/fluent-dev` instruction dataset into a
clean, bucketed pool of CSS `@keyframes` animations that provably render and
move.

## `prepare_data.py`

```bash
uv run python prepare_data.py              # full pipeline (with render check)
uv run python prepare_data.py --no-render-check   # faster, less clean
```

Steps:

1. **Load** `justmalhar/fluent-dev` and combine its `train` + `validation`
   splits into one pool (the project makes its own split later).
2. **Filter** to rows whose `code` contains `@keyframes` (`KEYFRAMES_RE`).
3. **Render-check** every candidate with `render_filter` (skippable via
   `--no-render-check`). Blank or static rows are dropped.
4. **Classify** each survivor into technique buckets via `classify`.
5. **Write** `data/keyframes_all.jsonl` (chat-format records, `source:
   "fluent-dev"`) and print a bucket-coverage summary.
6. **Build** `data/preview.html`, a visual gallery of the kept animations.

Output record shape is the shared training format documented in
[architecture.md](architecture.md#the-training-example-format).

> The **train/val split** (`data/keyframes_train.jsonl`,
> `data/keyframes_val.jsonl`) that `train.py` consumes is **not** produced here.
> See [reference.md → Known gaps](reference.md#known-gaps).

## `dataset_utils.py` — the shared spine

Imported by both `prepare_data.py` and `generate_data.py` so the two data
sources are filtered and labelled identically.

### Keyframe extraction — `extract_keyframes_blocks(code)`

A small brace-matching scanner that pulls each `@keyframes { ... }` block out of
a code string (handles nesting). Used by both the classifier and the preview.

### Technique classification — `classify(code) -> list[str]`

Runs a table of regex rules (`_BUCKET_RULES`) and returns every bucket that
matches. Most rules only look **inside** the `@keyframes` blocks (so a static
`color:` on the element doesn't count as a color *animation*); the structural
rules scan the full code.

| Bucket | Detects (abridged) | Searched in |
|--------|--------------------|-------------|
| `rotate` | `rotate(` | keyframes only |
| `translate` | `translate[XYZ/3d](` | keyframes only |
| `scale` | `scale(` / `scale[XYZ](` | keyframes only |
| `skew` | `skew[XY](` / `scale3d(` | keyframes only |
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
| `multi_keyframe` | more than one `@keyframes` block | (structural) |
| `other` | fallback when nothing else matched | — |

`ALL_BUCKETS` is this list in order. A row with no detected technique is tagged
`["other"]` rather than left empty.

### Render filter — `render_filter(codes) -> list[bool]`

The quality backbone. Runs Chromium headless (Playwright) over many snippets
concurrently and returns a keep/drop mask. For each snippet it:

1. Wraps bare snippets in a centered 300×300 white page (`wrap_html`). **Full
   HTML documents are passed through untouched** — double-wrapping a complete
   `<html>` doc produces invalid nested markup that renders unpredictably.
2. Loads the page and nudges the mouse to the center to trigger `:hover` states.
3. Samples `RENDER_SAMPLES` (6) screenshots `RENDER_SAMPLE_GAP` (400 ms) apart.
4. Requires **content in most frames** — at least `RENDER_MIN_VISIBLE` (0.6) of
   frames must have a non-white fraction above `BLANK_THRESHOLD` (0.01). This
   rejects blank renders.
5. Requires **motion** — between some adjacent frame pair, at least
   `MOTION_MIN_FRACTION` (0.005) of pixels must change (grayscale delta beyond
   `MOTION_PIXEL_DELTA`, via `ImageChops.difference`). This rejects static
   "animations" that load but never move.

A snippet must pass **both** the visibility and motion checks. Any exception
(timeout, bad markup) is treated as a failure. Concurrency is capped at
`RENDER_CONCURRENCY` (12) browser pages.

Tunable constants live at the top of `dataset_utils.py`.

### Summaries & preview

- `bucket_counts` / `print_bucket_summary` — coverage counts + a text bar chart,
  plus average buckets-per-sample.
- `build_preview(rows, out_path, title)` — writes a dark-themed HTML gallery
  (up to `PREVIEW_LIMIT` = 40 cards). Each card shows the category, colored
  bucket tags, the description, a sandboxed `<iframe>` live render, and the
  extracted `@keyframes` source side by side.
