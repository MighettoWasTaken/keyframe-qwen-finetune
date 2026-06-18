"""
Shared dataset utilities: keyframe extraction, animation-type classification,
headless render filtering, and HTML preview generation.

Used by both prepare_data.py (HF dataset) and generate_data.py (Claude
generations) so that both are filtered and bucketed identically.
"""

import asyncio
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm as atqdm

DATA_DIR = Path("data")
KEYFRAMES_RE = re.compile(r"@keyframes\b", re.IGNORECASE)
BLANK_THRESHOLD = 0.01      # fraction of non-white pixels required to pass
RENDER_CONCURRENCY = 12     # parallel browser pages
PREVIEW_LIMIT = 40
RENDER_SAMPLES = 6          # frames sampled across the animation timeline
RENDER_SAMPLE_GAP = 400     # ms between sampled frames
RENDER_MIN_VISIBLE = 0.6    # fraction of frames that must show content to pass
MOTION_PIXEL_DELTA = 20     # grayscale delta for a pixel to count as "changed"
MOTION_MIN_FRACTION = 0.005 # fraction of pixels that must change between frames


# ---------------------------------------------------------------------------
# Keyframe block extraction
# ---------------------------------------------------------------------------

def extract_keyframes_blocks(code: str) -> list[str]:
    blocks, idx = [], 0
    while True:
        start = code.lower().find("@keyframes", idx)
        if start == -1:
            break
        brace = code.find("{", start)
        if brace == -1:
            break
        depth, end = 1, brace + 1
        while end < len(code) and depth > 0:
            if code[end] == "{":
                depth += 1
            elif code[end] == "}":
                depth -= 1
            end += 1
        blocks.append(code[start:end])
        idx = end
    return blocks


# ---------------------------------------------------------------------------
# Animation type classification
# ---------------------------------------------------------------------------

# (bucket_name, pattern, search_in_keyframes_only)
_BUCKET_RULES: list[tuple[str, re.Pattern, bool]] = [
    ("rotate",              re.compile(r"rotate\s*\(",                          re.I), True),
    ("translate",           re.compile(r"translate(?:X|Y|Z|3d)?\s*\(",         re.I), True),
    ("scale",               re.compile(r"\bscale(?!3d)\s*\(|scale[XYZ]\s*\(", re.I), True),
    ("skew",                re.compile(r"skew(?:X|Y)?\s*\(|scale3d\s*\(",      re.I), True),
    ("opacity",             re.compile(r"\bopacity\s*:",                        re.I), True),
    ("color",               re.compile(r"(?:background-color|border-color|(?<![a-z-])color)\s*:", re.I), True),
    ("size",                re.compile(r"\b(?:width|height)\s*:",               re.I), True),
    ("stroke",              re.compile(r"stroke-(?:dasharray|dashoffset)\s*:",  re.I), True),
    ("shadow",              re.compile(r"box-shadow\s*:",                       re.I), True),
    ("clip",                re.compile(r"clip-path\s*:",                        re.I), True),
    ("background_position", re.compile(r"background-position\s*:",             re.I), True),
    ("border_radius",       re.compile(r"border-radius\s*:",                   re.I), True),
    # Structural — search full code
    ("staggered",           re.compile(r"animation-delay\s*:|nth-child\s*\(",  re.I), False),
    ("hover",               re.compile(r":hover\s*\{",                         re.I), False),
]

ALL_BUCKETS = [name for name, _, _ in _BUCKET_RULES] + ["multi_keyframe", "other"]


def classify(code: str) -> list[str]:
    kf_blocks = extract_keyframes_blocks(code)
    kf_text = " ".join(kf_blocks)

    buckets = []
    for name, pattern, kf_only in _BUCKET_RULES:
        haystack = kf_text if kf_only else code
        if pattern.search(haystack):
            buckets.append(name)

    if len(kf_blocks) > 1:
        buckets.append("multi_keyframe")

    return buckets or ["other"]


# ---------------------------------------------------------------------------
# Render check
# ---------------------------------------------------------------------------

def wrap_html(code: str) -> str:
    # Claude often returns a complete HTML document; wrapping that in another
    # document produces invalid nested <html> that renders unpredictably. Only
    # wrap bare snippets, and pass full documents through untouched.
    if re.search(r"<!doctype|<html[\s>]", code, re.IGNORECASE):
        return code
    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
        "<style>body{display:flex;align-items:center;justify-content:center;"
        "min-height:100vh;margin:0;background:#fff;}</style>"
        f"</head><body>{code}</body></html>"
    )


def _nonwhite_fraction(img: Image.Image) -> float:
    """Fraction of pixels darker than near-white, from the grayscale histogram."""
    hist = img.histogram()        # 'L' mode -> 256 bins
    return sum(hist[:245]) / sum(hist)


async def _check_one(sem: asyncio.Semaphore, browser, code: str) -> bool:
    async with sem:
        page = await browser.new_page(viewport={"width": 300, "height": 300})
        try:
            await page.set_content(wrap_html(code), wait_until="load", timeout=5000)
            await page.mouse.move(150, 150)  # trigger :hover states
            # Sample several frames across the timeline so we can check two things:
            # (1) content is visible in most frames, not just one lucky instant;
            # (2) the frames actually differ, i.e. it's animating, not a still.
            frames: list[Image.Image] = []
            for _ in range(RENDER_SAMPLES):
                await page.wait_for_timeout(RENDER_SAMPLE_GAP)
                shot = await page.screenshot()
                frames.append(Image.open(io.BytesIO(shot)).convert("L"))

            visible = sum(1 for f in frames if _nonwhite_fraction(f) > BLANK_THRESHOLD)
            if visible / len(frames) < RENDER_MIN_VISIBLE:
                return False

            n_pixels = frames[0].size[0] * frames[0].size[1]
            max_motion = 0.0
            for a, b in zip(frames, frames[1:]):
                diff = ImageChops.difference(a, b)
                changed = sum(diff.histogram()[MOTION_PIXEL_DELTA + 1:])
                max_motion = max(max_motion, changed / n_pixels)
            return max_motion >= MOTION_MIN_FRACTION
        except Exception:
            return False
        finally:
            await page.close()


async def _render_filter_async(codes: list[str]) -> list[bool]:
    sem = asyncio.Semaphore(RENDER_CONCURRENCY)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        tasks = [_check_one(sem, browser, code) for code in codes]
        results = await atqdm.gather(*tasks, desc="Render check", unit="snippet")
        await browser.close()
    return list(results)


def render_filter(codes: list[str]) -> list[bool]:
    """Return a boolean mask — True means the render has visible content."""
    return asyncio.run(_render_filter_async(codes))


# ---------------------------------------------------------------------------
# Bucket summary
# ---------------------------------------------------------------------------

def bucket_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for b in row["_buckets"]:
            counts[b] += 1
    return counts


def print_bucket_summary(rows: list[dict]):
    counts = bucket_counts(rows)
    print(f"\n{'Bucket':<22} {'Count':>5}  {'Bar'}")
    print("-" * 50)
    for b in ALL_BUCKETS:
        n = counts.get(b, 0)
        print(f"  {b:<20} {n:>5}  {'#' * (n // 2)}")
    if rows:
        print(f"\n  Total samples: {len(rows)}")
        print(f"  Avg buckets/sample: {sum(len(r['_buckets']) for r in rows) / len(rows):.1f}\n")


# ---------------------------------------------------------------------------
# HTML preview
# ---------------------------------------------------------------------------

_BUCKET_COLORS = {
    "rotate": "#5c4a8a", "translate": "#2a5a8a", "scale": "#2a7a5a",
    "skew": "#7a5a2a", "opacity": "#5a7a2a", "color": "#7a2a5a",
    "size": "#2a6a7a", "stroke": "#7a3a2a", "shadow": "#4a4a7a",
    "clip": "#7a4a2a", "background_position": "#2a4a6a", "border_radius": "#6a2a4a",
    "staggered": "#4a6a2a", "hover": "#6a4a2a", "multi_keyframe": "#2a2a6a",
    "other": "#4a4a4a",
}

_CARD = """\
<div class="card">
  <div class="meta">
    <span class="cat">{category}</span>
    {bucket_spans}
    <p class="desc">{description}</p>
  </div>
  <div class="cols">
    <iframe srcdoc="{srcdoc}" sandbox="allow-scripts" loading="lazy"></iframe>
    <pre><code>{keyframes_esc}</code></pre>
  </div>
</div>"""

_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body{{font-family:monospace;background:#0e0e0e;color:#ccc;padding:1rem;}}
  h1{{color:#fff;}}
  .summary{{background:#161616;border:1px solid #333;border-radius:6px;padding:1rem;margin-bottom:2rem;}}
  .summary td{{padding:2px 12px;}}
  .bar{{height:10px;display:inline-block;vertical-align:middle;border-radius:2px;}}
  .card{{border:1px solid #333;border-radius:6px;padding:1rem;margin-bottom:2rem;background:#161616;}}
  .cat{{background:#2a2a6e;padding:2px 8px;border-radius:4px;font-size:.75rem;margin-right:4px;}}
  .bucket{{padding:2px 8px;border-radius:4px;font-size:.72rem;margin-right:4px;}}
  .desc{{color:#aaa;font-size:.82rem;margin:.4rem 0;}}
  .cols{{display:flex;gap:1rem;align-items:flex-start;}}
  iframe{{flex:0 0 300px;width:300px;height:300px;border:1px solid #333;border-radius:4px;background:#fff;}}
  pre{{flex:1;margin:0;font-size:.78rem;background:#111;padding:.6rem;border-radius:4px;
       max-height:300px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;}}
</style>
</head>
<body>
<h1>{title} &mdash; {count} samples</h1>
{summary_html}
{cards}
</body>
</html>"""


def _bucket_summary_html(rows: list[dict]) -> str:
    counts = bucket_counts(rows)
    rows_html = ""
    for b in ALL_BUCKETS:
        n = counts.get(b, 0)
        color = _BUCKET_COLORS.get(b, "#444")
        rows_html += (
            f"<tr><td style='color:#aaa'>{b}</td>"
            f"<td style='text-align:right'>{n}</td>"
            f"<td><span class='bar' style='width:{n * 2}px;background:{color}'></span></td></tr>\n"
        )
    return f"<div class='summary'><table>{rows_html}</table></div>"


def _srcdoc(code: str) -> str:
    return wrap_html(code).replace("&", "&amp;").replace('"', "&quot;")


def build_preview(rows: list[dict], out_path: Path, title: str = "@keyframes dataset"):
    cards = []
    for row in rows[:PREVIEW_LIMIT]:
        kf = extract_keyframes_blocks(row["code"])
        kf_text = "\n\n".join(kf) if kf else "(none extracted)"
        bucket_spans = " ".join(
            "<span class='bucket' style='background:{}'>{}</span>".format(
                _BUCKET_COLORS.get(b, "#444"), b
            )
            for b in row["_buckets"]
        )
        tags_raw = row.get("tags", [])
        tags = tags_raw if isinstance(tags_raw, list) else []
        desc = row.get("description") or " / ".join(tags) or ""
        cards.append(_CARD.format(
            category=row.get("category", ""),
            bucket_spans=bucket_spans,
            description=desc,
            srcdoc=_srcdoc(row["code"]),
            keyframes_esc=kf_text.replace("&", "&amp;").replace("<", "&lt;"),
        ))
    out_path.write_text(
        _PAGE.format(
            title=title,
            count=len(rows),
            summary_html=_bucket_summary_html(rows),
            cards="\n".join(cards),
        ),
        encoding="utf-8",
    )
    print(f"Preview -> {out_path}  ({min(len(rows), PREVIEW_LIMIT)} cards shown)")
