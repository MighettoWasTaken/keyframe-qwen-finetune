# Architecture

## Goal

Fine-tune a small, local code model to turn a one-sentence motion brief into a minimal, self-contained CSS `@keyframes` animation. The hard part isn't the training loop — it's assembling a **clean, well-balanced dataset** of animations that actually render and move. Most of the code is about data quality.

## Pipeline overview

```
justmalhar/fluent-dev  (HuggingFace dataset)
        │
        ▼
prepare_data.py
  filter @keyframes → render_filter → classify → data/keyframes_all.jsonl
        │
        ▼
generate_data.py  (Anthropic API, optional supplementation)
  Stage A: motion briefs → dedup → data/generated/prompts.jsonl
  Stage B: animations → render_filter + judge → model cascade
                                          ↓
                         data/generated/keyframes_generated.jsonl
        │
        ▼
split_data.py
  merge both sources → iterative multilabel stratification
                                          ↓
                         data/keyframes_train.jsonl  (494 examples)
                         data/keyframes_val.jsonl    (130 examples)
        │
        ▼
train.py  (bf16 LoRA SFT, TRL SFTTrainer)
                                          ↓
                         outputs/qwen3b-keyframes-lora/
        │
        ▼
eval.py  (pass@1 greedy, render check, LLM judge)
                                          ↓
                         outputs/eval/<tag>.jsonl
                         outputs/eval/<tag>_metrics.json
        │
        ▼
make_plots.py + make_preview.py
                                          ↓
                         outputs/plots/
                         outputs/preview.html
```

`dataset_utils.py` is the shared spine: `prepare_data.py`, `generate_data.py`, and `eval.py` all import its render filter and classifier so sourced and generated data are filtered **identically**.

## Components

| File | Role |
|------|------|
| `examine_data.py` | One-off exploration — not in the pipeline |
| `prepare_data.py` | Builds the base dataset from `justmalhar/fluent-dev` |
| `dataset_utils.py` | Shared library: render filter, bucket classifier, preview builder |
| `generate_data.py` | Supplements thin buckets with Claude-generated animations |
| `split_data.py` | Merges both sources, iterative multilabel stratification → train/val |
| `prompting.py` | Single source of truth for system prompt and chat template |
| `train.py` | bf16 LoRA SFT with TRL SFTTrainer |
| `eval.py` | Knockout evaluation harness (generation → render → judge → metrics) |
| `verify_model.py` | GPU smoke test — loads model, reports VRAM and tok/s |
| `generate.py` | Simple inference script (load base + adapter, run one prompt) |
| `make_plots.py` | Training curves + before/after metrics bar chart |
| `make_preview.py` | Side-by-side HTML preview of base vs fine-tuned outputs |

## Training example format

Every example — sourced or generated — is a two-turn chat record:

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

`prompting.py` prepends the shared system prompt at training time (`train.py:add_system`) and at eval time so both see byte-identical templates. `buckets` records animation techniques for stratified splitting; `source` lets the two pools be told apart downstream.

## Design principles

- **One quality bar, applied everywhere.** The same `render_filter` gates the HF data, Claude generations, and eval outputs — nothing static or blank gets in.
- **Cheapest-first generation.** Supplementation starts on `claude-haiku-4-5` and escalates only failures to sonnet → opus, keeping API spend low.
- **Everything is resumable.** Prompts flush per chunk, animations flush per kept row, training checkpoints every 10 steps, eval caches each phase. Nothing repeats on interruption.
- **Template consistency.** `prompting.py` is the single import for system prompt + message builder. Training, eval, and inference all use it — the #1 cause of fine-tune A/B failures is a train/eval template mismatch.
- **Knockout eval, not just loss.** Perplexity measures format learning; render-pass + LLM brief-match measures functional correctness. Both are reported and plotted separately.
