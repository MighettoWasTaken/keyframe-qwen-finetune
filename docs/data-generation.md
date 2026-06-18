# Data generation — `generate_data.py`

Supplements the dataset with Claude-generated CSS `@keyframes` animations,
deliberately skewed toward the **under-represented buckets** and toward higher
complexity. It runs in two stages and is built to be cheap, resumable, and
self-balancing.

```bash
uv run python generate_data.py --n 4 --dry-run            # wiring check, no API calls
uv run python generate_data.py --n 4                       # small real run
uv run python generate_data.py --n 200 --batch-size 5 --workers 6
uv run python generate_data.py --n 50 --prompts-only       # Stage A only
```

Requires `ANTHROPIC_API_KEY` (via `.env`). `--dry-run` makes **no** API calls and
prints the assembled prompts, so it works without a key.

## Stage A — motion descriptions

Generates one-sentence motion briefs as **structured JSON** (`json_schema`
output, `PROMPT_SCHEMA`). Each brief is `{description, techniques}`.

Key behaviours:

- **Chunked, not all-in-one.** Briefs are requested `PROMPT_GEN_CHUNK` (25) at a
  time, each chunk a separate API call, so `max_tokens` (8000) stays safe
  regardless of `--n`.
- **Truncation-safe.** If a response stops at `max_tokens`, the chunk size is
  halved and retried rather than feeding `json.loads` a half-written string. If a
  chunk is still unparseable, it's skipped with a warning instead of crashing the
  run.
- **Bucket-aware.** The prompt shows current dataset coverage and tells the model
  to lean on the `FOCUS_BUCKETS` (the thin ones: `stroke`, `border_radius`,
  `clip`, `skew`, `color`, `shadow`, `size`, `background_position`, `staggered`).
- **Per-chunk diversification.** Each chunk rotates which `FOCUS_BUCKET` it
  emphasizes and samples 8 random subjects from `MOTIF_POOL` (~38 motifs:
  orbiting particles, DNA helix, gears, radar sweep, …). This is the main lever
  against duplicates — without it, repeated identical instructions make the model
  converge on the same handful of ideas and the duplicate rate climbs toward 100%
  as the avoid-list grows.
- **Deduplicated.** `dedupe_prompts` drops near-duplicates within the batch and
  against everything already generated, using `difflib.SequenceMatcher` with a
  ratio threshold of `DEDUP_THRESHOLD` (0.82).
- **Incremental + resumable.** Each accepted brief is flushed to
  `data/generated/prompts.jsonl` immediately, so a crash or Ctrl-C never loses
  collected work, and the next run dedupes against the file.
- **Stall guard.** After `MAX_STALLED_CHUNKS` (6) consecutive all-duplicate
  chunks, Stage A stops ("motion space looks exhausted") rather than burning API
  calls forever.

`--prompts-only` stops here. Each brief is `{"description": ..., "techniques":
[...]}` in `prompts.jsonl`.

## Stage B — animations (cascade)

Turns each brief into a minimal HTML+CSS animation, validates it, and keeps the
good ones.

### Model cascade — `run_cascade(...)`

The cost-control core. Defined in `MODEL_CASCADE`:

```
claude-haiku-4-5  →  claude-sonnet-4-6  →  claude-opus-4-8
   (cheapest)                                 (most capable)
```

- Each tier processes **only** the briefs that failed above it (a parse failure
  *or* a validation failure), re-batched at `--batch-size`.
- Within a tier, the **first batch runs alone to warm the prompt cache**, then
  the rest fan out across `--workers` threads.
- Briefs that survive validation are kept; failures escalate to the next tier.
  The last tier drops whatever still fails instead of crashing.
- Returns `(results, per-tier stats)`; stats report attempted / kept /
  escalated-or-dropped per tier.

### Validation — `make_validator(render_fn, judge_fn)`

A kept row must pass **both**:

1. **Render check** — the same `render_filter` from `dataset_utils.py` (renders
   *and* moves). See [data-pipeline.md](data-pipeline.md#render-filter--render_filtercodes---listbool).
2. **Match judge** (optional, on by default) — `make_judge` asks a model
   (default `claude-haiku-4-5`) whether the animation actually implements the
   brief, flagging ones that are unrelated, static, or ignore the core motion.
   The judge only runs on render-passers (no point judging a broken render) and
   judges in chunks of 10; missing indices default to "match" (benefit of the
   doubt). Disable with `--no-judge`.

### Output — incremental + cross-run caching

- Before generating, `load_generated_descriptions()` reads
  `data/generated/keyframes_generated.jsonl` and **skips briefs that already
  produced a kept animation** in a prior run.
- Each kept row is flushed to `keyframes_generated.jsonl` **immediately** via an
  `on_keep` callback inside `run_cascade`, so a mid-run failure resumes instead
  of restarting.
- After the run, `data/generated/preview_generated.html` is rebuilt from the full
  accumulated file, and a bucket summary is printed.

Generated rows use the shared training format with `source: "claude"`.

## Prompt construction

- `PROMPT_GEN_SYSTEM` / `build_prompt_gen_user` — Stage A. Pushes "pure visual
  animations, NOT UI components" (no buttons/forms/cards/menus), compound
  multi-phase motion, and the focus buckets + per-chunk emphasis/motifs.
- `ANIM_INSTRUCTIONS` + `build_anim_system` — Stage B system prompt. Demands a
  **minimal** snippet (just the animated element(s) + one inline `<style>`, no
  page chrome), dropped into a pre-centered 300×300 white canvas. Includes up to
  2 real high-complexity few-shot examples selected from the source data (UI
  components filtered out via `_UI_MARKERS`). The whole system prefix is marked
  with `cache_control` so it's cached across batches.
- `parse_animations` — extracts the per-brief `<animation index="N">…</animation>`
  blocks and keeps only those that actually contain a `@keyframes` rule.

## Cost control (FrugalGPT-inspired subset)

| Strategy | Where |
|----------|-------|
| **Model cascade** — start cheap, escalate only failures | `run_cascade` / `MODEL_CASCADE` |
| **Query concatenation** — many animations per call | `--batch-size`, `build_anim_user` |
| **Prompt selection** — a *small* curated few-shot set | `few_shot_block` (2 examples) |
| **Prompt caching** — identical system prefix cached ~90% off | `build_anim_system` (`cache_control`), first-batch cache warm-up |

Per-model token usage and an estimated cost (from the `PRICING` table) are
tracked by the `Usage` class and printed at the end of every run. Costs are
attributed **per tier**, so the cascade's spend is visible.

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--n` | 4 | Number of animations to generate |
| `--batch-size` | 5 | Animations per API call in Stage B (query concatenation) |
| `--workers` | 4 | Parallel API calls within a tier |
| `--model` | `claude-haiku-4-5` | Stage A model (Stage B uses `MODEL_CASCADE`) |
| `--judge-model` | `claude-haiku-4-5` | Model for the match judge |
| `--no-judge` | off | Skip the match judge (render check only) |
| `--prompts-only` | off | Stage A only; write briefs and stop |
| `--dry-run` | off | Print the plan + assembled prompts; make **no** API calls |

> Note: `--batch-size` controls **Stage B only**. Stage A's chunk size is the
> separate `PROMPT_GEN_CHUNK` constant (25) — they were decoupled on purpose:
> large chunks for cheap text, small batches for the expensive animation calls.
