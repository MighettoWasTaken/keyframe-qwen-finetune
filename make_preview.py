"""
Generate a side-by-side HTML preview of base vs fine-tuned model outputs.
Samples are stratified: FT-wins, both-pass, base-wins, both-fail so the
interesting cases (FT improvements + regressions) appear first.

Usage:
  uv run python make_preview.py
  uv run python make_preview.py --ftw 10 --bop 5 --bof 5 --seed 7 --out outputs/preview.html
"""

import argparse
import html
import json
import random
import re
from pathlib import Path


def load_cache(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["description"]] = r
    return out


def outcome(rec: dict) -> bool:
    return bool(rec.get("render_ext")) and bool(rec.get("match_ext"))


def badge(ok: bool | None, label: str) -> str:
    if ok is None:
        cls, sym = "unknown", "?"
    elif ok:
        cls, sym = "pass", "✓"
    else:
        cls, sym = "fail", "✗"
    return f'<span class="badge {cls}">{sym} {label}</span>'


_HEAD_RE = re.compile(r"<head[^>]*>(.*?)</head>", re.S | re.I)
_BODY_RE = re.compile(r"<body[^>]*>(.*?)</body>", re.S | re.I)


def make_iframe(code: str) -> str:
    # Outputs come in two shapes: bare fragments (<div>+<style>) and COMPLETE html
    # documents (<!doctype><html><head>..<body>..). To render every pane identically
    # (and centered), normalize both into ONE template: pull the snippet's own
    # <head> styles + <body> content out of full docs, then re-wrap. This avoids
    # nesting a document inside <body> (which leaked <head>/<title> text and broke
    # layout) AND fixes the base-vs-FT centering asymmetry.
    code = code.strip()
    is_full_doc = "<html" in code[:256].lower() or code[:64].lstrip().lower().startswith("<!doctype")
    if is_full_doc:
        hm, bm = _HEAD_RE.search(code), _BODY_RE.search(code)
        head_inner = hm.group(1) if hm else ""
        body_inner = bm.group(1) if bm else code
    else:
        head_inner, body_inner = "", code

    # Our centering style goes AFTER the snippet's own head styles and uses
    # !important so it wins regardless of what the snippet sets on <body>.
    # NOTE: deliberately NO stacking-context fix here. A component that needs one
    # to render (z-index:-1 pseudo-elements behind a non-isolating host) renders
    # as a black box -- that's a fair "failed generation" signal, not something to
    # paper over in the preview.
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"{head_inner}"
        "<style>html,body{height:100%!important;margin:0!important}"
        "body{background:#fff;overflow:hidden;"
        "display:grid!important;place-items:center!important}</style></head>"
        f"<body>{body_inner}</body></html>"
    )
    return (
        f'<iframe srcdoc="{html.escape(doc, quote=True)}" '
        'sandbox="allow-scripts" scrolling="no"></iframe>'
    )


def make_card(idx: int, desc: str, base: dict, ft: dict) -> str:
    def half(rec: dict, title: str) -> str:
        code = rec.get("code_ext") or rec.get("output") or ""
        render = rec.get("render_ext")
        match  = rec.get("match_ext")
        badges = badge(render, "Render") + " " + badge(match, "Match")
        preview = make_iframe(code) if code.strip() else '<div class="empty">no output</div>'
        return f"""<div class="half">
          <div class="model-label">{title}</div>
          <div class="badges">{badges}</div>
          {preview}
          <details><summary>Code</summary><pre><code>{html.escape(code)}</code></pre></details>
        </div>"""

    return f"""<div class="card">
  <div class="card-header">
    <span class="card-num">#{idx}</span>
    <span class="desc">{html.escape(desc)}</span>
  </div>
  <div class="comparison">
    {half(base, "Base")}
    {half(ft, "Fine-tuned")}
  </div>
</div>"""


PAGE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f0f13; color: #e0e0e0;
       padding: 28px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.4rem; color: #fff; margin-bottom: 8px; }
.summary { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 8px;
           padding: 14px 18px; margin-bottom: 28px; font-size: 0.88rem; line-height: 1.8;
           color: #aaa; }
.summary b { color: #ddd; }
.legend { display: flex; gap: 20px; margin-top: 6px; font-size: 0.82rem; }
.legend span { display: flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; }
.dot.ftw { background: #3a7d3a; }
.dot.bop { background: #3a5a7d; }
.dot.baw { background: #7d4a3a; }
.dot.bof { background: #444; }
.card { background: #16161e; border: 1px solid #2a2a3a; border-radius: 10px;
        margin-bottom: 24px; overflow: hidden; }
.card.ftw  { border-left: 3px solid #3dba3d; }
.card.bop  { border-left: 3px solid #5b9bd5; }
.card.baw  { border-left: 3px solid #cc6633; }
.card.bof  { border-left: 3px solid #444; }
.card-header { padding: 10px 16px; background: #1e1e2c; border-bottom: 1px solid #2a2a3a;
               display: flex; gap: 12px; align-items: baseline; }
.card-num { font-size: 0.72rem; color: #666; flex-shrink: 0; }
.desc { font-size: 0.88rem; color: #c0c0d0; font-style: italic; }
.comparison { display: grid; grid-template-columns: 1fr 1fr; }
.half { padding: 14px 16px; }
.half + .half { border-left: 1px solid #2a2a3a; }
.model-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: .07em; color: #666; margin-bottom: 8px; }
.badges { display: flex; gap: 6px; margin-bottom: 10px; }
.badge { font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.badge.pass { background: #152815; color: #5dcc5d; border: 1px solid #1f4f1f; }
.badge.fail { background: #2a1515; color: #cc5d5d; border: 1px solid #4f1f1f; }
.badge.unknown { background: #222; color: #888; border: 1px solid #333; }
iframe { width: 100%; height: 220px; border: 1px solid #2a2a3a; border-radius: 6px;
         background: #fff; display: block; margin-bottom: 8px; }
.empty { height: 220px; display: flex; align-items: center; justify-content: center;
         color: #444; font-size: 0.82rem; border: 1px dashed #282828; border-radius: 6px;
         margin-bottom: 8px; }
details { font-size: 0.78rem; }
summary { cursor: pointer; color: #555; padding: 2px 0; user-select: none; }
summary:hover { color: #999; }
pre { background: #0a0a10; border: 1px solid #1e1e2a; border-radius: 4px;
      padding: 10px; overflow-x: auto; font-size: 0.72rem;
      max-height: 220px; overflow-y: auto; margin-top: 4px; color: #bbb; }
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="outputs/eval/base.jsonl")
    p.add_argument("--ft",   default="outputs/eval/ft.jsonl")
    p.add_argument("--ftw",  type=int, default=10, help="# FT-wins to show")
    p.add_argument("--bop",  type=int, default=5,  help="# both-pass to show")
    p.add_argument("--baw",  type=int, default=0,  help="# base-wins to show")
    p.add_argument("--bof",  type=int, default=5,  help="# both-fail to show")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",  default="outputs/preview.html")
    args = p.parse_args()

    base_cache = load_cache(args.base)
    ft_cache   = load_cache(args.ft)
    descs = [d for d in base_cache if d in ft_cache]

    buckets = {
        "ftw": [d for d in descs if outcome(ft_cache[d]) and not outcome(base_cache[d])],
        "bop": [d for d in descs if outcome(ft_cache[d]) and     outcome(base_cache[d])],
        "baw": [d for d in descs if outcome(base_cache[d]) and not outcome(ft_cache[d])],
        "bof": [d for d in descs if not outcome(ft_cache[d]) and not outcome(base_cache[d])],
    }

    alloc = {"ftw": args.ftw, "bop": args.bop, "baw": args.baw, "bof": args.bof}

    rng = random.Random(args.seed)
    selected_tagged: list[tuple[str, str]] = []  # (desc, bucket_key)
    for key, bucket in buckets.items():
        k = min(alloc[key], len(bucket))
        selected_tagged.extend((d, key) for d in rng.sample(bucket, k))

    # sort: ftw → bop → baw → bof
    order = {"ftw": 0, "bop": 1, "baw": 2, "bof": 3}
    selected_tagged.sort(key=lambda x: order[x[1]])

    cards = "".join(
        make_card(i + 1, d, base_cache[d], ft_cache[d]).replace('<div class="card">', f'<div class="card {bk}">', 1)
        for i, (d, bk) in enumerate(selected_tagged)
    )

    counts = {k: len(v) for k, v in buckets.items()}
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Keyframes Preview: Base vs Fine-tuned</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<h1>Keyframes Fine-tune Preview</h1>
<div class="summary">
  <b>{len(descs)}</b> val examples &nbsp;|&nbsp;
  FT wins: <b>{counts['ftw']}</b> &nbsp;|&nbsp;
  Both pass: <b>{counts['bop']}</b> &nbsp;|&nbsp;
  Base wins: <b>{counts['baw']}</b> &nbsp;|&nbsp;
  Both fail: <b>{counts['bof']}</b>
  &nbsp;&nbsp;·&nbsp;&nbsp; showing <b>{len(selected_tagged)}</b> samples (10 FT-wins · 5 both-pass · 5 both-fail)
  <div class="legend">
    <span><span class="dot ftw"></span> FT wins (FT pass, base fail)</span>
    <span><span class="dot bop"></span> Both pass</span>
    <span><span class="dot baw"></span> Base wins (base pass, FT fail)</span>
    <span><span class="dot bof"></span> Both fail</span>
  </div>
</div>
{cards}
</body>
</html>"""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"Written {len(selected_tagged)} samples -> {args.out}")
    for k, label in [("ftw", "FT-wins"), ("bop", "both-pass"), ("baw", "base-wins"), ("bof", "both-fail")]:
        print(f"  {label}: {counts[k]} total, sampled {sum(1 for _, bk in selected_tagged if bk == k)}")


if __name__ == "__main__":
    main()
