# Training & evaluation

## `train.py` — bf16 LoRA SFT

Fine-tunes Qwen2.5-Coder-3B-Instruct on the keyframe examples using TRL's `SFTTrainer` with a bf16 LoRA adapter. Standard supervised fine-tuning: cross-entropy on assistant tokens only. No exotic objectives — the render/knockout metric lives in `eval.py` and never touches training.

```bash
uv run python train.py
uv run python train.py --epochs 2 --use-4bit    # QLoRA fallback if OOM
uv run python train.py --report-to none         # skip W&B
```

| Flag | Default | Notes |
|------|---------|-------|
| `--model-id` | `Qwen/Qwen2.5-Coder-3B-Instruct` | base model |
| `--train-path` | `data/keyframes_train.jsonl` | |
| `--val-path` | `data/keyframes_val.jsonl` | |
| `--output-dir` | `outputs/qwen3b-keyframes-lora` | adapter + checkpoints |
| `--max-length` | 2048 | |
| `--epochs` | 3 | |
| `--batch-size` | 4 | per-device |
| `--grad-accum` | 4 | effective batch = 16 |
| `--lr` | 2e-4 | |
| `--save-steps` | 10 | checkpoint + eval every N optimizer steps |
| `--use-4bit` | off | NF4 QLoRA fallback for <16GB cards |
| `--report-to` | `wandb` | `none` to disable |

Key choices:

- **bf16 LoRA, not QLoRA by default** — Qwen 3B in bf16 (~6GB) fits a 16GB card with room and trains faster/cleaner than 4-bit. Pass `--use-4bit` to fall back.
- **Completion-only loss** via `assistant_only_loss=True` — Qwen's chat template carries `{% generation %}` tags; TRL masks prompt tokens natively without a custom DataCollator.
- **Step-based checkpointing** — `save_strategy="steps"` / `eval_strategy="steps"` every `--save-steps` (default 10). With ~93 total optimizer steps over 3 epochs, this loses at most 10 steps on a crash vs. a whole epoch with epoch-level saves. Resumable via `get_last_checkpoint()` — re-running picks up exactly where it stopped.
- **`JsonlLossLogger`** callback — appends every log event (step, epoch, loss, lr, grad_norm, eval_loss) to `outputs/qwen3b-keyframes-lora/train_log.jsonl` for offline plotting independent of W&B.

## `prompting.py` — template consistency

Single import for the system prompt and message builder used identically by `train.py`, `eval.py`, and `generate.py`. A train/eval template mismatch is the #1 cause of silent fine-tune A/B failures.

```python
from prompting import SYSTEM_PROMPT, build_messages, extract_code

build_messages(description)               # inference prompt (no response)
build_messages(description, response)     # full training turn
extract_code(text)                        # strips markdown fences from model output
```

## `eval.py` — knockout evaluation

Pass@1 greedy eval, the HumanEval-style execute-and-check analog for animations. Knockout = render-pass AND brief-match, reported as separate numbers.

```bash
uv run python eval.py --tag base
uv run python eval.py --tag ft --adapter outputs/qwen3b-keyframes-lora --compare base
uv run python eval.py --tag ft --adapter ... --no-judge   # render-only, no API cost
```

Three phases run in sequence, each cached so re-runs resume:

1. **GPU — generation + perplexity.** Greedy decoding (pass@1). Perplexity is completion-only (prompt tokens masked) to match the training objective. Both phases share one model load.
2. **CPU — render check.** `render_filter` from `dataset_utils.py`, run on both the raw output and `extract_code(output)` (fence-stripped). Produces `render_raw` / `render_ext` per example.
3. **API — LLM judge.** `claude-haiku-4-5` judges whether each render-passing animation actually implements the brief. Only runs on render-passers. Produces `match_raw` / `match_ext`.

Results are cached per description in `outputs/eval/<tag>.jsonl` and rewritten after each phase. Final metrics written to `outputs/eval/<tag>_metrics.json`.

**Raw vs extracted split** — separates format adherence from animation correctness. The base model wraps output in markdown fences (raw = broken render, extracted = correct render); the fine-tuned model outputs bare HTML/CSS (raw ≈ extracted).

**Metrics reported:**

| Metric | What it measures |
|--------|-----------------|
| `perplexity` | Model confidence on the val distribution |
| `format_valid_rate` | Output contains `@keyframes` and no ` ``` ` fences |
| `render_rate` | Fraction that render + visibly move |
| `match_of_rendered` | Fraction of renders where LLM judge says it matches the brief |
| `knockout_rate` | Fraction that fail (1 − animation pass rate); lower is better |

## `verify_model.py` — GPU smoke test

```bash
uv run python verify_model.py
```

Loads the base model in bf16, runs a single generation, and reports VRAM usage and throughput (tok/s). Useful to confirm the environment is working before a training run. Expected output on RTX 4080: ~6.3GB VRAM, ~28 tok/s.

## Hardware

Developed on an NVIDIA RTX 4080 (16GB VRAM), CUDA 12.4. Training: ~93 optimizer steps over 3 epochs, ~20 minutes. `pyproject.toml` pins the CUDA 12.4 torch wheel via `[tool.uv.sources]` so `uv sync` pulls the GPU build automatically.
