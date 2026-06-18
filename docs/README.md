# fluentdev-probe — documentation

LoRA fine-tuning of a small code model to generate **CSS `@keyframes` animations**
from plain-English motion descriptions. Built as a 24-hour challenge.

The project takes a public instruction dataset, keeps only the rows that contain
real, *visibly animating* `@keyframes` CSS, classifies them by animation
technique, supplements the thin buckets with Claude-generated examples, and
fine-tunes [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
on the result with a 4-bit QLoRA.

## Documentation map

| Doc | What's in it |
|-----|--------------|
| [architecture.md](architecture.md) | End-to-end pipeline, data flow, and where each file fits |
| [data-pipeline.md](data-pipeline.md) | `prepare_data.py` + `dataset_utils.py`: bucket classification, headless render filter, HTML preview |
| [data-generation.md](data-generation.md) | `generate_data.py`: Claude generation, the model cascade, dedup, and cross-run caching |
| [training-and-inference.md](training-and-inference.md) | `train.py` (QLoRA SFT) and `generate.py` (inference) |
| [reference.md](reference.md) | CLI flags, file layout, JSONL schemas, env/config, tests, and known gaps |

## Quickstart

```bash
# 0. Install (uv manages the venv + CUDA-12.4 torch wheel)
uv sync
uv run playwright install chromium      # needed for the render check

# 1. Build the base dataset from justmalhar/fluent-dev
uv run python prepare_data.py           # -> data/keyframes_all.jsonl + data/preview.html

# 2. (Optional) Supplement thin buckets with Claude-generated animations
cp .env.example .env                     # add ANTHROPIC_API_KEY
uv run python generate_data.py --n 4 --dry-run   # wiring check, no API calls
uv run python generate_data.py --n 200 --batch-size 5 --workers 6

# 3. Train + run inference (CUDA GPU)
uv run python train.py                  # -> outputs/qwen-keyframes-lora
uv run python generate.py

# Tests (no API/browser needed)
uv run pytest -q
```

> **Heads up:** step 3 expects `data/keyframes_train.jsonl`. The repo does not
> currently ship a script that produces the train/val split from
> `keyframes_all.jsonl` + the generated data — see
> [reference.md → Known gaps](reference.md#known-gaps).

## At a glance

- **Language / tooling:** Python 3.12, [uv](https://docs.astral.sh/uv/) for deps, `pytest` for tests.
- **Hardware target:** single NVIDIA GPU (developed on an RTX 4080, CUDA 12.4).
- **External services:** Hugging Face (source dataset), Anthropic API (data supplementation), Weights & Biases (training logs).
- **The render filter is the quality backbone:** every example — sourced or
  generated — must pass a headless-Chromium check proving it renders *and*
  actually moves, so the model never trains on blank or static "animations".
