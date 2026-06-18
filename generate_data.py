"""
Supplement the keyframes dataset with Claude-generated CSS animations that
target the emptier buckets and skew toward higher complexity.

Cost-control strategies (FrugalGPT, cheapest-to-implement subset):
  1. Query concatenation  - many animations per API call (one shared prompt
     amortized over a whole batch instead of one call per animation).
  2. Prompt selection     - a small curated few-shot set, not a large one.
  3. Prompt caching        - the system + few-shot prefix is identical across
     every batch, marked with cache_control for ~90% off those input tokens.
     The first batch runs alone to warm the cache before the rest fan out.

Outputs (kept separate from the HF data on purpose):
  data/generated/prompts.jsonl              generated motion descriptions
  data/generated/keyframes_generated.jsonl  filtered, bucketed training rows
  data/generated/preview_generated.html     gallery

Usage:
  uv run python generate_data.py --n 4 --dry-run   # wiring check, no API calls
  uv run python generate_data.py --n 4             # small real run
  uv run python generate_data.py --n 200 --batch-size 5 --workers 6
"""

import argparse
import difflib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from dataset_utils import (
    ALL_BUCKETS,
    DATA_DIR,
    KEYFRAMES_RE,
    bucket_counts,
    build_preview,
    classify,
    print_bucket_summary,
    render_filter,
)

load_dotenv()

# --- Defaults (override via CLI) -------------------------------------------
# Start with cheapest model; failed batches cascade up through the tiers.
MODEL = "claude-haiku-4-5"
MODEL_CASCADE = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
SOURCE_DATA = DATA_DIR / "keyframes_all.jsonl"
OUT_DIR = DATA_DIR / "generated"

# Pricing per 1M tokens (input, output) — used for cost reporting only.
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Emptier buckets to actively push for. rotate/translate/scale show up plenty
# on their own, so we don't ask for them explicitly.
FOCUS_BUCKETS = [
    "stroke", "border_radius", "clip", "skew", "color",
    "shadow", "size", "background_position", "staggered",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_existing_counts() -> dict[str, int]:
    if not SOURCE_DATA.exists():
        return {}
    rows = [json.loads(line) for line in SOURCE_DATA.read_text(encoding="utf-8").splitlines()]
    return dict(bucket_counts([{"_buckets": r.get("buckets", [])} for r in rows]))


PROMPT_GEN_SYSTEM = """\
You write concise plain-English motion descriptions for CSS animations, in the \
style of a designer briefing an engineer. Each description is ONE rich sentence \
describing multi-step, visually complex motion (holds, sequenced phases, \
staggered elements, easing changes), e.g.:

"A loader where three dots scale up in sequence then collapse inward while the \
ring around them rotates and its stroke dashes chase the rotation."

Favor compound, multi-phase motion over single transforms.

IMPORTANT: Describe pure visual animations and motion graphics only — NOT UI \
components. No buttons, forms, inputs, cards, menus, modals, or any interactive \
widget. Think abstract shapes, geometric motion, loaders, particles, SVG paths, \
text reveal — things that animate, not things that you click."""


# Subject motifs sampled per call so each Stage A request explores a different
# region of the space instead of converging on the model's few default ideas.
# This is the main lever against the climbing-duplicate problem.
MOTIF_POOL = [
    "orbiting particles", "a liquid blob morphing", "concentric pulsing rings",
    "a typographic text reveal", "a DNA double helix", "audio-waveform bars",
    "an origami fold/unfold", "a kaleidoscope", "interlocking gears",
    "a water ripple", "a comet with a trailing tail", "a morphing polygon",
    "a segmented loading spinner", "a flickering flame", "falling confetti",
    "a neon sign flicker", "a swinging pendulum", "a spirograph curve",
    "a glitch/datamosh effect", "matrix-style falling glyphs", "a breathing grid",
    "a bouncing ball with squash and stretch", "an unfurling spiral",
    "a starfield warp", "a clock with sweeping hands", "a heartbeat pulse line",
    "stacked cards fanning out", "a wave of dominoes", "a blooming flower",
    "a lightning arc", "a soap bubble wobble", "a radar sweep",
    "a checkerboard flip cascade", "a metaball merge", "a ferris wheel rotation",
    "an hourglass with falling sand", "a vinyl record spin", "a paper plane flight",
]


def build_prompt_gen_user(n: int, counts: dict[str, int],
                          avoid: list[str] | None = None,
                          emphasis: str | None = None,
                          motifs: list[str] | None = None) -> str:
    underrep = sorted(FOCUS_BUCKETS, key=lambda b: counts.get(b, 0))
    counts_line = ", ".join(f"{b}={counts.get(b, 0)}" for b in ALL_BUCKETS)
    parts = [
        f"Generate {n} distinct motion descriptions for CSS @keyframes animations.\n",
        f"Current dataset coverage (lower = needs more): {counts_line}\n",
        f"Skew the set toward these under-represented techniques: "
        f"{', '.join(underrep)}. It is fine for common ones (rotate, translate, "
        f"scale) to also appear, but every description should lean on at least "
        f"one under-represented technique. Each must be higher-complexity than a "
        f"single transform: combine motion phases, timing, or multiple elements.\n",
        "Keep every description to ONE sentence under 280 characters — be vivid "
        "but concise, not a paragraph.\n",
    ]
    if emphasis:
        parts.append(
            f"For THIS batch, make the technique `{emphasis}` the dominant motion "
            f"in most descriptions.\n"
        )
    if motifs:
        listed = ", ".join(motifs)
        parts.append(
            "Draw subject matter from a VARIETY of these visual motifs (don't reuse "
            f"the same one twice in this batch): {listed}. Invent others freely — the "
            "goal is breadth.\n"
        )
    if avoid:
        # Show what already exists so Claude doesn't repeat or paraphrase it.
        listed = "\n".join(f"- {d}" for d in avoid)
        parts.append(
            "These descriptions ALREADY EXIST. Do not repeat or paraphrase any of "
            f"them; produce clearly distinct motions:\n{listed}\n"
        )
    parts.append(f"List `techniques` using only these tags: {', '.join(ALL_BUCKETS)}.")
    return "\n".join(parts)


# --- Near-duplicate detection for generated prompts (concern: repeated prompts) ---
DEDUP_THRESHOLD = 0.82  # SequenceMatcher ratio above which two briefs are "the same"


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def is_duplicate(desc: str, existing_norms: list[str],
                 threshold: float = DEDUP_THRESHOLD) -> bool:
    n = _normalize(desc)
    if not n:
        return True
    for e in existing_norms:
        if n == e or difflib.SequenceMatcher(None, n, e).ratio() >= threshold:
            return True
    return False


def dedupe_prompts(new: list[dict], existing: list[str],
                   threshold: float = DEDUP_THRESHOLD) -> list[dict]:
    """Drop new prompts that duplicate each other or anything already generated."""
    seen = [_normalize(d) for d in existing]
    unique = []
    for p in new:
        if is_duplicate(p["description"], seen, threshold):
            continue
        unique.append(p)
        seen.append(_normalize(p["description"]))
    return unique


def load_existing_prompts() -> list[str]:
    path = OUT_DIR / "prompts.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line)["description"])
    return out


def load_generated_descriptions() -> set[str]:
    """Briefs that already have a kept animation, so Stage B can skip them on rerun."""
    path = OUT_DIR / "keyframes_generated.jsonl"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.add(row["messages"][0]["content"])
    return out

PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Bound the length so output can't balloon as the model strains
                    # to look "distinct" from a growing avoid list — that runaway
                    # verbosity is what blows past max_tokens late in a run.
                    "description": {"type": "string", "maxLength": 280},
                    "techniques": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description", "techniques"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["prompts"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Animation generation
# ---------------------------------------------------------------------------

ANIM_INSTRUCTIONS = """\
You generate minimal, self-contained CSS @keyframes animations from motion descriptions.

Output a MINIMAL snippet — just the animated element(s) and one inline <style>:
- NO <!DOCTYPE>, <html>, <head>, <body>, <meta>, or <title>. Emit only the
  element markup followed by a single <style> block. The element is dropped into
  a pre-centered 300x300 white canvas, so do not add your own page wrapper,
  background-color on body, canvas/card container, padding, border-radius,
  box-shadow, or centering layout around the animation.
- Style only the animated element(s) themselves and their @keyframes. Every
  visual property you write should belong to the animation, not to page chrome.
- It must include at least one @keyframes rule and visibly animate.
- No external assets, no JS, no network requests.
- Match the brief precisely and prefer higher complexity (multiple keyframe
  stops, multiple animated properties, staggered timing where it fits).
- Use only the techniques the brief calls for plus whatever the motion needs.
- Generate pure visual animations only — NOT UI components. No buttons, forms,
  inputs, cards, menus, or interactive widgets. Think abstract shapes, geometric
  motion, loaders, particles, SVG paths, text effects — things that animate.

Example of the expected minimal shape:
<animation index="1">
<div class="orbit"></div>
<style>
.orbit { width: 40px; height: 40px; background: #3a7; border-radius: 50%;
         animation: orbit 2s linear infinite; }
@keyframes orbit { from { transform: rotate(0) translateX(60px); }
                   to { transform: rotate(360deg) translateX(60px); } }
</style>
</animation>

Output format: for each numbered brief, emit exactly one block:
<animation index="N">
...the minimal HTML+CSS snippet...
</animation>
Emit nothing else between or around the blocks."""

_ANIM_RE = re.compile(r'<animation\s+index="(\d+)"\s*>(.*?)</animation>', re.DOTALL | re.IGNORECASE)


# Markers of UI-component examples we don't want to show as exemplars.
_UI_MARKERS = re.compile(
    r"<button|<input|<form|<textarea|<select\b|<nav\b|<a\s|class=[\"'][^\"']*\b"
    r"(?:btn|button|card|menu|modal|navbar|tooltip|dropdown|toggle|switch)\b",
    re.IGNORECASE,
)


def few_shot_block() -> str:
    """A couple of complex real examples = prompt selection (Strategy 1)."""
    if not SOURCE_DATA.exists():
        return ""
    rows = [json.loads(line) for line in SOURCE_DATA.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if "multi_keyframe" in r.get("buckets", [])]
    # Skip UI-component examples — they contradict the "pure animation" brief.
    rows = [r for r in rows if not _UI_MARKERS.search(r["messages"][1]["content"])]
    rows.sort(key=lambda r: len(r.get("buckets", [])), reverse=True)
    examples = []
    for r in rows[:2]:
        code = r["messages"][1]["content"]
        examples.append(f"<animation index=\"1\">\n{code}\n</animation>")
    if not examples:
        return ""
    return "\n\nReference examples of the expected output quality:\n" + "\n\n".join(examples)


def build_anim_system() -> list[dict]:
    """System blocks with a cache_control breakpoint on the shared prefix."""
    text = ANIM_INSTRUCTIONS + few_shot_block()
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_anim_user(batch: list[dict]) -> str:
    lines = ["Generate one animation per brief below.\n"]
    for i, p in enumerate(batch, start=1):
        lines.append(f"{i}. {p['description']}")
    return "\n".join(lines)


def parse_animations(text: str, batch: list[dict]) -> list[dict]:
    by_index = {int(m.group(1)): m.group(2).strip() for m in _ANIM_RE.finditer(text)}
    out = []
    for i, p in enumerate(batch, start=1):
        code = by_index.get(i)
        if code and KEYFRAMES_RE.search(code):
            out.append({"description": p["description"], "code": code})
    return out


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class Usage:
    def __init__(self):
        # per-model token buckets so the cost estimate is accurate across tiers
        self._by_model: dict[str, dict] = {}

    def add(self, u, model: str):
        m = self._by_model.setdefault(model, {"in": 0, "out": 0, "cw": 0, "cr": 0})
        m["in"] += u.input_tokens
        m["out"] += u.output_tokens
        m["cw"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        m["cr"] += getattr(u, "cache_read_input_tokens", 0) or 0

    def cost(self) -> float:
        total = 0.0
        for model, m in self._by_model.items():
            ip, op = PRICING.get(model, (5.0, 25.0))
            total += (
                m["in"] * ip
                + m["cw"] * ip * 1.25
                + m["cr"] * ip * 0.1
                + m["out"] * op
            ) / 1_000_000
        return total

    def report(self):
        for model, m in self._by_model.items():
            print(
                f"  [{model}] in={m['in']} cache_write={m['cw']} "
                f"cache_read={m['cr']} out={m['out']}"
            )
        print(f"  estimated cost: ${self.cost():.4f}")


# ---------------------------------------------------------------------------
# Match judge (concern: animation generated but doesn't match the prompt)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a strict reviewer. For each (brief, animation) pair, decide whether the \
animation plausibly implements the motion the brief describes. Judge intent and \
the main motions, not pixel-perfect fidelity. Mark matches=false if the animation \
is unrelated, static, or ignores the brief's core motion."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "matches": {"type": "boolean"},
                },
                "required": ["index", "matches"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_judge_user(rows: list[dict]) -> str:
    parts = ["Review each pair. Return matches=true/false for every index.\n"]
    for i, r in enumerate(rows, start=1):
        parts.append(f"--- index {i} ---\nBRIEF: {r['description']}\nANIMATION:\n{r['code']}\n")
    return "\n".join(parts)


def make_judge(client, model: str, usage: "Usage | None" = None, batch_size: int = 10):
    """Return judge(rows)->list[bool], batched so no single call is huge.

    Rows are chunked into groups of batch_size. Each chunk is judged independently
    with indices reset to 1..N per chunk, then results are stitched back together.
    Missing indices default to True (benefit of the doubt).
    """
    def judge_chunk(chunk: list[dict]) -> list[bool]:
        resp = client.messages.create(
            model=model,
            max_tokens=500 + 50 * len(chunk),
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": build_judge_user(chunk)}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        if usage is not None:
            usage.add(resp.usage, model)
        text = next(b.text for b in resp.content if b.type == "text")
        by_idx = {r["index"]: r["matches"] for r in json.loads(text)["results"]}
        return [bool(by_idx.get(i + 1, True)) for i in range(len(chunk))]

    def judge(rows: list[dict]) -> list[bool]:
        if not rows:
            return []
        results = []
        for chunk in chunked(rows, batch_size):
            results.extend(judge_chunk(chunk))
        return results
    return judge


# ---------------------------------------------------------------------------
# Generation engine (extracted so it can be unit-tested with fakes)
# ---------------------------------------------------------------------------

def make_generator(client, system, usage: "Usage | None" = None, max_tokens: int = 16000):
    """Return generate(tier, batch)->(text, usage_obj) backed by the real API."""
    import time
    import anthropic as _anthropic

    def generate(tier: str, batch: list[dict]):
        delay = 5
        for attempt in range(8):
            try:
                with client.messages.stream(
                    model=tier,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": build_anim_user(batch)}],
                ) as stream:
                    msg = stream.get_final_message()
                text = next((b.text for b in msg.content if b.type == "text"), "")
                return text, msg.usage
            except _anthropic.RateLimitError:
                if attempt == 7:
                    raise
                print(f"  rate limited; retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
    return generate


def make_validator(render_fn, judge_fn=None):
    """Compose render check + optional match judge into validate(rows)->list[bool].

    The judge only runs on rows that pass the render check (no point judging a
    broken render), and a row must pass BOTH to be kept.
    """
    def validate(rows: list[dict]) -> list[bool]:
        mask = list(render_fn([r["code"] for r in rows]))
        if judge_fn:
            keep_idx = [i for i, ok in enumerate(mask) if ok]
            if keep_idx:
                jmask = judge_fn([rows[i] for i in keep_idx])
                for i, jok in zip(keep_idx, jmask):
                    mask[i] = mask[i] and jok
        return mask
    return validate


def run_cascade(prompts, *, tiers, batch_size, workers, generate, validate,
                usage: "Usage | None" = None, log=print, on_keep=None):
    """Tier-by-tier cascade. Each tier runs only the prompts that failed above it;
    failures (parse OR validation) are re-batched for the next tier. The last tier
    keeps whatever validates and drops the rest.

    generate(tier, batch) -> (text, usage_obj)
    validate(rows)        -> list[bool] aligned to rows
    Returns (results, stats) where stats is a per-tier list of dicts.
    """
    results: list[dict] = []
    stats: list[dict] = []
    pending = list(prompts)

    for tier_idx, tier in enumerate(tiers):
        if not pending:
            break
        is_last = tier_idx == len(tiers) - 1
        batches = list(chunked(pending, batch_size))
        log(f"\n  [{tier}] {len(pending)} prompt(s) -> {len(batches)} batch(es)")

        tier_kept = 0
        tier_failed: list[dict] = []
        batch_num = 0

        def handle(text, usage_obj, batch):
            nonlocal batch_num, tier_kept
            batch_num += 1
            if usage is not None and usage_obj is not None:
                usage.add(usage_obj, tier)
            parsed = parse_animations(text, batch)
            parse_failed = [p for p in batch if p["description"] not in {r["description"] for r in parsed}]

            # Validate immediately so each batch reports before the next fires.
            kept_rows = []
            if parsed:
                mask = validate(parsed)
                for row, ok in zip(parsed, mask):
                    if ok:
                        kept_rows.append(row)
                        results.append(row)
                        tier_kept += 1
                        if on_keep is not None:
                            on_keep(row)
                    else:
                        tier_failed.append(next((p for p in batch if p["description"] == row["description"]), None) or {"description": row["description"]})
            tier_failed.extend(parse_failed)

            status = (f"batch {batch_num}/{len(batches)}  "
                      f"generated {len(parsed)}/{len(batch)}  "
                      f"kept {len(kept_rows)}/{len(parsed) or len(batch)}  "
                      f"| total kept {len(results)}")
            log(f"  [{tier}] {status}")

        # First batch alone warms the prompt cache for this model, then fan out.
        text, u = generate(tier, batches[0])
        handle(text, u, batches[0])
        rest = batches[1:]
        if rest:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for (text, u), batch in zip(pool.map(lambda b: generate(tier, b), rest), rest):
                    handle(text, u, batch)

        failed = [f for f in tier_failed if f is not None]
        kept = tier_kept
        stats.append({"tier": tier, "attempted": len(pending), "kept": kept,
                      "failed": len(failed)})
        if failed:
            if is_last:
                log(f"  [{tier}] {len(failed)} prompt(s) failed on all tiers, dropping")
            else:
                log(f"  [{tier}] escalating {len(failed)} failure(s) (parse+validation) to next tier")
        pending = failed

    return results, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="Number of animations to generate")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Animations per API call (query concatenation)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel API calls")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompts-only", action="store_true",
                        help="Generate motion descriptions, skip animation generation")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM match-check that filters animations not matching their brief")
    parser.add_argument("--judge-model", default="claude-haiku-4-5",
                        help="Model used for the match-check (default: cheapest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and assembled prompts; make NO API calls")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = load_existing_counts()
    existing_prompts = load_existing_prompts()

    # --- Plan ---
    n_batches = (args.n + args.batch_size - 1) // args.batch_size
    print(f"Model: {args.model}")
    print(f"Plan: {args.n} animations, {n_batches} batch(es) of <= {args.batch_size}, "
          f"{args.workers} workers")
    print(f"Match judge: {'off' if args.no_judge else args.judge_model}")
    print(f"Already generated (deduped against): {len(existing_prompts)} prompt(s)")
    print(f"Focus buckets (current counts): "
          f"{', '.join(f'{b}={counts.get(b, 0)}' for b in FOCUS_BUCKETS)}")

    if args.dry_run:
        print("\n--- DRY RUN: no API calls ---")
        print("\n[Prompt-gen system]\n" + PROMPT_GEN_SYSTEM)
        print("\n[Prompt-gen user]\n" + build_prompt_gen_user(
            args.n, counts, existing_prompts,
            emphasis=FOCUS_BUCKETS[0],
            motifs=random.sample(MOTIF_POOL, k=min(8, len(MOTIF_POOL)))))
        sys_text = build_anim_system()[0]["text"]
        print(f"\n[Animation system block] ({len(sys_text)} chars, cache_control set)")
        print(sys_text[:600] + ("..." if len(sys_text) > 600 else ""))
        return

    import anthropic  # imported here so --dry-run works without the key
    client = anthropic.Anthropic()
    usage = Usage()

    # --- Stage A: generate motion descriptions (chunked to avoid truncation) ---
    # Each chunk is a separate API call so max_tokens stays manageable regardless of --n.
    # ~25 descriptions comfortably fit in 8k output tokens; if a chunk is still
    # truncated (stop_reason == "max_tokens"), we halve it and retry rather than
    # feed json.loads a half-written string.
    PROMPT_GEN_CHUNK = 25
    MAX_STALLED_CHUNKS = 6  # consecutive all-duplicate chunks before we give up
    # --n is a TARGET TOTAL for prompts.jsonl, not "this many more". Reruns top up
    # to n rather than appending n each time, so the file can't run past the goal.
    shortfall = max(0, args.n - len(existing_prompts))
    print(f"\nStage A: {len(existing_prompts)} description(s) on disk, target {args.n} "
          f"-> generating {shortfall} more ({PROMPT_GEN_CHUNK} per call)...")
    # Only the most recent AVOID_SAMPLE descriptions are shown to the model as
    # soft "don't repeat" guidance. The real dedup guard is dedupe_prompts()
    # below, which uses the FULL list locally for free — so we don't need to
    # re-send hundreds of sentences on every call (that grows O(n) per call,
    # O(n^2) over the run, bloating input cost/latency for no extra protection).
    AVOID_SAMPLE = 60
    prompts: list[dict] = []
    avoid = list(existing_prompts)  # full local dedup set; grows as we generate
    total_dropped = 0
    stalled = 0
    chunk_i = 0
    # Each new description is flushed to disk as it's accepted, so a crash or a
    # Ctrl-C never loses collected work and the next run dedupes against it.
    prompts_path = OUT_DIR / "prompts.jsonl"
    prompts_f = prompts_path.open("a", encoding="utf-8")

    while len(prompts) < shortfall:
        needed = min(PROMPT_GEN_CHUNK, shortfall - len(prompts))
        # Steer each chunk to a different region: rotate the emphasized technique
        # and sample a fresh set of subject motifs. This is what keeps the
        # duplicate rate from climbing as `avoid` grows.
        emphasis = FOCUS_BUCKETS[chunk_i % len(FOCUS_BUCKETS)]
        motifs = random.sample(MOTIF_POOL, k=min(8, len(MOTIF_POOL)))
        avoid_hint = avoid[-AVOID_SAMPLE:]  # bounded; full list still used for dedup
        chunk_i += 1
        while True:
            resp = client.messages.create(
                model=args.model,
                max_tokens=12000,
                system=PROMPT_GEN_SYSTEM,
                messages=[{"role": "user",
                           "content": build_prompt_gen_user(needed, counts, avoid_hint,
                                                            emphasis=emphasis, motifs=motifs)}],
                output_config={"format": {"type": "json_schema", "schema": PROMPT_SCHEMA}},
            )
            usage.add(resp.usage, args.model)
            if resp.stop_reason == "max_tokens" and needed > 1:
                needed = max(1, needed // 2)
                print(f"  output truncated at max_tokens; retrying with {needed} per call")
                continue
            break
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            raw = json.loads(text)["prompts"]
        except (json.JSONDecodeError, KeyError) as e:
            # Truncated even at the smallest chunk, no text block, or malformed
            # output: skip this chunk instead of crashing the whole run.
            print(f"  skipping unparseable chunk (stop_reason={resp.stop_reason}): {e}")
            stalled += 1
            if stalled >= MAX_STALLED_CHUNKS:
                print(f"  too many unproductive chunks; stopping at {len(prompts)}.")
                break
            continue
        unique = dedupe_prompts(raw, avoid)
        total_dropped += len(raw) - len(unique)
        for p in unique:
            prompts.append(p)
            avoid.append(p["description"])
            prompts_f.write(json.dumps(p) + "\n")
        prompts_f.flush()
        print(f"  {len(prompts)}/{shortfall} collected ({total_dropped} duplicates dropped so far)")

        if unique:
            stalled = 0
        else:
            stalled += 1
            if stalled >= MAX_STALLED_CHUNKS:
                print(f"  no new descriptions in {MAX_STALLED_CHUNKS} chunks; "
                      f"the motion space looks exhausted — stopping at {len(prompts)}.")
                break

    prompts_f.close()
    print(f"  {len(prompts)} new descriptions -> {prompts_path}")

    if args.prompts_only:
        usage.report()
        return

    # --- Stage B: generate animations (cascade: cheap -> expensive on failure) ---
    print("\nStage B: generating animations...")
    # Operate on the WHOLE prompt pool (up to n), not just this run's new briefs,
    # so descriptions left over from earlier runs finally get animations too.
    # Skip any brief that already produced a kept animation (resumable across runs).
    pool = load_existing_prompts()[:args.n]
    already_done = load_generated_descriptions()
    pending = [{"description": d} for d in pool if d not in already_done]
    print(f"  {len(pool)} brief(s) in pool, {len(pool) - len(pending)} already have an "
          f"animation; {len(pending)} to generate")

    out_path = OUT_DIR / "keyframes_generated.jsonl"
    out_f = out_path.open("a", encoding="utf-8")

    def flush_row(row):
        row["_buckets"] = classify(row["code"])
        row["category"] = "generated"
        out_f.write(json.dumps({
            "messages": [
                {"role": "user", "content": row["description"]},
                {"role": "assistant", "content": row["code"]},
            ],
            "buckets": row["_buckets"],
            "category": "generated",
            "source": "claude",
        }) + "\n")
        out_f.flush()

    system = build_anim_system()
    generate = make_generator(client, system, usage)
    judge_fn = None if args.no_judge else make_judge(client, args.judge_model, usage)
    validate = make_validator(render_filter, judge_fn)

    results, stats = run_cascade(
        pending, tiers=MODEL_CASCADE, batch_size=args.batch_size,
        workers=args.workers, generate=generate, validate=validate, usage=usage,
        on_keep=flush_row,
    )
    out_f.close()

    print(f"\n  total kept this run: {len(results)}/{len(pending)}")
    print("  cascade breakdown:")
    for s in stats:
        print(f"    {s['tier']}: attempted {s['attempted']}, kept {s['kept']}, escalated/dropped {s['failed']}")
    # Rows were already flushed to keyframes_generated.jsonl by flush_row as they
    # were kept; now rebuild the preview from the full accumulated file.
    print(f"Saved {len(results)} new examples -> {out_path}")

    all_rows = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            all_rows.append({
                "code": r["messages"][1]["content"],
                "_buckets": r.get("buckets", []),
                "description": r["messages"][0]["content"],
                "category": r.get("category", "generated"),
            })
    print_bucket_summary(all_rows)
    print(f"  total generated dataset: {len(all_rows)} examples")
    build_preview(all_rows, OUT_DIR / "preview_generated.html",
                  title="@keyframes (Claude-generated)")

    print("\nUsage:")
    usage.report()


if __name__ == "__main__":
    main()
