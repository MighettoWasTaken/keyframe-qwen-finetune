"""
Evaluate a model (base or base+LoRA adapter) on the held-out val descriptions and
report the metrics we care about: knockout = render-pass AND brief-match, with the
two numbers kept separate so they can be plotted independently. Also reports the
"raw vs extracted" split (output as-is vs code pulled out of any markdown fence)
so format adherence can be told apart from animation correctness, plus val
perplexity (the standard loss/sanity metric).

Training stays standard SFT; nothing here touches the gradient. This is pass@1
greedy, the animation analog of HumanEval-style execute-and-check.

EVERYTHING IS CACHED per description in outputs/eval/<tag>.jsonl, phase by phase
(generation, perplexity, render, judge). Re-running resumes — it only does the
work that's missing, so an interrupted long run loses nothing.

Usage (run in your own terminal):
  uv run python eval.py --tag base
  uv run python eval.py --tag ft --adapter outputs/qwen3b-keyframes-lora
  uv run python eval.py --tag ft --adapter ... --compare base   # print A/B table
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: render_filter is imported lazily inside run_render_phase so the GPU
# phases never load Playwright (keeps generation startup light and avoids a
# torch+playwright import clash).
from prompting import build_messages, extract_code

load_dotenv()

EVAL_DIR = Path("outputs/eval")
_KEYFRAMES = re.compile(r"@keyframes\b", re.IGNORECASE)


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------------------
# Cache (one JSON record per description, rewritten after each phase)
# ---------------------------------------------------------------------------

def load_val(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"description": r["messages"][0]["content"],
                         "reference": r["messages"][1]["content"]})
    return rows


def load_cache(tag: str, val: list[dict]) -> dict:
    path = EVAL_DIR / f"{tag}.jsonl"
    cache = {r["description"]: dict(r) for r in val}  # seed with description+reference
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["description"] in cache:
                    cache[rec["description"]].update(rec)
    return cache


def save_cache(tag: str, cache: dict):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{tag}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in cache.values():
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# GPU phases: generation + perplexity (share one model load)
# ---------------------------------------------------------------------------

def load_model(model_id: str, adapter: str | None):
    src = adapter or model_id
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def run_gpu_phases(tag, cache, model_id, adapter, max_new_tokens, gen_batch):
    descs = list(cache)
    need_gen = [d for d in descs if cache[d].get("output") is None]
    need_ppl = [d for d in descs if cache[d].get("nll") is None]
    if not need_gen and not need_ppl:
        print("  (generation + perplexity fully cached)")
        return

    print(f"  loading model ({'adapter: ' + adapter if adapter else 'base'}) ...")
    model, tok = load_model(model_id, adapter)

    # --- generation (greedy, pass@1) ---
    if need_gen:
        tok.padding_side = "left"
        print(f"  generating {len(need_gen)} completion(s)...")
        done = 0
        for batch in chunked(need_gen, gen_batch):
            prompts = [tok.apply_chat_template(build_messages(d), tokenize=False,
                                               add_generation_prompt=True) for d in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for i, d in enumerate(batch):
                text = tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                cache[d]["output"] = text
                cache[d]["code_ext"] = extract_code(text)
            done += len(batch)
            save_cache(tag, cache)
            print(f"    {done}/{len(need_gen)} generated")

    # --- perplexity (teacher-forced, completion-only mask) ---
    if need_ppl:
        tok.padding_side = "right"
        print(f"  scoring perplexity on {len(need_ppl)} example(s)...")
        for i, d in enumerate(need_ppl, 1):
            ref = cache[d]["reference"]
            full = tok.apply_chat_template(build_messages(d, ref), tokenize=True,
                                           return_tensors="pt", return_dict=True).to("cuda")
            plen = tok.apply_chat_template(build_messages(d), tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)["input_ids"].shape[1]
            input_ids = full["input_ids"]
            labels = input_ids.clone()
            labels[:, :plen] = -100
            with torch.inference_mode():
                loss = model(input_ids=input_ids, attention_mask=full["attention_mask"],
                             labels=labels).loss
            n = int((labels != -100).sum().item())
            cache[d]["nll"] = float(loss.item()) * n
            cache[d]["n_tokens"] = n
            if i % 20 == 0 or i == len(need_ppl):
                save_cache(tag, cache)
                print(f"    {i}/{len(need_ppl)} scored")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CPU phase: render check (raw + extracted)
# ---------------------------------------------------------------------------

def run_render_phase(tag, cache):
    need = [d for d in cache if cache[d].get("render_raw") is None]
    if not need:
        print("  (render fully cached)")
        return
    from dataset_utils import render_filter
    print(f"  render-checking {len(need)} output(s) (raw + extracted)...")
    raw_mask = render_filter([cache[d]["output"] for d in need])
    ext_mask = render_filter([cache[d]["code_ext"] for d in need])
    for d, r, e in zip(need, raw_mask, ext_mask):
        cache[d]["render_raw"] = bool(r)
        cache[d]["render_ext"] = bool(e)
    save_cache(tag, cache)


# ---------------------------------------------------------------------------
# API phase: LLM brief-match judge (only on render-passers)
# ---------------------------------------------------------------------------

def run_judge_phase(tag, cache, judge_model):
    from generate_data import make_judge, Usage
    import anthropic

    client = anthropic.Anthropic()
    usage = Usage()
    judge = make_judge(client, judge_model, usage)

    for render_key, code_key, match_key in [("render_raw", "output", "match_raw"),
                                            ("render_ext", "code_ext", "match_ext")]:
        need = [d for d in cache
                if cache[d].get(match_key) is None and cache[d].get(render_key)]
        if not need:
            continue
        print(f"  judging {len(need)} render-passer(s) [{match_key}]...")
        rows = [{"description": d, "code": cache[d][code_key]} for d in need]
        verdicts = judge(rows)
        for d, ok in zip(need, verdicts):
            cache[d][match_key] = bool(ok)
        save_cache(tag, cache)
    usage.report()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(tag, cache, adapter, judged) -> dict:
    descs = list(cache)
    n = len(descs)
    nll = sum(cache[d].get("nll", 0.0) for d in descs)
    ntok = sum(cache[d].get("n_tokens", 0) for d in descs)
    ppl = math.exp(nll / ntok) if ntok else None

    def variant(render_key, match_key):
        rendered = [d for d in descs if cache[d].get(render_key)]
        matched = [d for d in rendered if cache[d].get(match_key)] if judged else []
        return {
            "render_pass": len(rendered),
            "render_rate": len(rendered) / n,
            "match_of_rendered": (len(matched) / len(rendered)) if (judged and rendered) else None,
            # knockout-inclusive: passes render AND brief-match
            "valid_and_match": len(matched) if judged else None,
            "knockout_rate": (1 - len(matched) / n) if judged else (1 - len(rendered) / n),
        }

    fmt_valid = sum(1 for d in descs
                    if _KEYFRAMES.search(cache[d].get("output", ""))
                    and "```" not in cache[d].get("output", "")) / n

    return {
        "tag": tag, "adapter": adapter, "n": n, "judged": judged,
        "perplexity": ppl, "mean_loss": (nll / ntok) if ntok else None,
        "format_valid_rate": fmt_valid,
        "raw": variant("render_raw", "match_raw"),
        "extracted": variant("render_ext", "match_ext"),
    }


def print_metrics(m: dict):
    print(f"\n=== {m['tag']} (n={m['n']}, {'judged' if m['judged'] else 'render-only'}) ===")
    print(f"  perplexity:        {m['perplexity']:.3f}" if m["perplexity"] else "  perplexity: n/a")
    print(f"  format-valid rate: {m['format_valid_rate']:.1%}")
    for v in ("raw", "extracted"):
        d = m[v]
        match = f"{d['match_of_rendered']:.1%}" if d["match_of_rendered"] is not None else "n/a"
        print(f"  [{v:9}] render-pass {d['render_rate']:.1%}  "
              f"match-of-rendered {match}  knockout {d['knockout_rate']:.1%}")


def print_comparison(base: dict, ft: dict):
    print("\n========== A/B: base vs fine-tuned ==========")
    def row(label, b, f, pct=True):
        fmt = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
        bs = fmt(b) if b is not None else "n/a"
        fs = fmt(f) if f is not None else "n/a"
        print(f"  {label:30} {bs:>10} -> {fs:>10}")
    row("perplexity", base["perplexity"], ft["perplexity"], pct=False)
    row("format-valid rate", base["format_valid_rate"], ft["format_valid_rate"])
    for v in ("raw", "extracted"):
        row(f"[{v}] render-pass", base[v]["render_rate"], ft[v]["render_rate"])
        row(f"[{v}] knockout (incl. match)", base[v]["knockout_rate"], ft[v]["knockout_rate"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Label for this run's cache/metrics (e.g. base, ft)")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the base model")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--val", default="data/keyframes_val.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--judge-model", default="claude-haiku-4-5")
    p.add_argument("--no-judge", action="store_true", help="Skip the brief-match judge (render only)")
    p.add_argument("--compare", default=None, help="Another tag to print an A/B table against")
    args = p.parse_args()

    val = load_val(args.val)
    cache = load_cache(args.tag, val)
    print(f"Evaluating '{args.tag}' on {len(val)} val example(s)")

    run_gpu_phases(args.tag, cache, args.model_id, args.adapter,
                   args.max_new_tokens, args.gen_batch_size)
    run_render_phase(args.tag, cache)
    if not args.no_judge:
        run_judge_phase(args.tag, cache, args.judge_model)

    metrics = compute_metrics(args.tag, cache, args.adapter, judged=not args.no_judge)
    (EVAL_DIR / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print_metrics(metrics)

    if args.compare:
        other = EVAL_DIR / f"{args.compare}_metrics.json"
        if other.exists():
            print_comparison(json.loads(other.read_text()), metrics)
        else:
            print(f"\n(no metrics for '{args.compare}' yet — run eval on it first)")


if __name__ == "__main__":
    main()
