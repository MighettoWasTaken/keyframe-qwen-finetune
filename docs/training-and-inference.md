# Training & inference

## `train.py` — QLoRA SFT

Fine-tunes a small instruct-tuned code model on the keyframe examples using
TRL's `SFTTrainer` with a 4-bit base and a LoRA adapter.

```bash
uv run python train.py
```

Configuration is a hardcoded `Args` dataclass (no CLI parsing) — edit the file to
change it:

| Field | Default | Notes |
|-------|---------|-------|
| `model_id` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | base model |
| `data_path` | `data/keyframes_train.jsonl` | training split (chat-format) |
| `output_dir` | `outputs/qwen-keyframes-lora` | adapter output |
| `max_seq_length` | 2048 | |
| `num_train_epochs` | 3 | |
| `per_device_train_batch_size` | 4 | |
| `gradient_accumulation_steps` | 4 | effective batch = 16 |
| `learning_rate` | 2e-4 | |
| `use_4bit` | `True` | NF4 quantization |

Details:

- **4-bit base** via `BitsAndBytesConfig` (NF4, bf16 compute) — QLoRA-style, fits
  a small model on a single consumer GPU.
- **LoRA** via `peft.LoraConfig`: `r=16`, `lora_alpha=32`,
  `target_modules="all-linear"`, `lora_dropout=0.05`, causal-LM task.
- **Data** is loaded with `datasets.load_dataset("json", ...)`. The records are
  the shared chat format (`messages` = user brief + assistant code); `SFTTrainer`
  applies the chat template.
- **Logging** goes to **Weights & Biases** (`report_to="wandb"`). Set `WANDB_*`
  env vars or `wandb login` first, or change `report_to` to `"none"` to disable.
- Checkpoints save per epoch; the final adapter is written to `output_dir`.

> `train.py` reads **only** `data/keyframes_train.jsonl`. To include the
> Claude-generated rows, merge `data/generated/keyframes_generated.jsonl` into
> the training split first. There is no script that does this yet — see
> [reference.md → Known gaps](reference.md#known-gaps).

## `generate.py` — inference

Loads the base model plus the trained LoRA adapter and runs a prompt through it.

```bash
uv run python generate.py
```

- `BASE_MODEL` = `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
  `ADAPTER_PATH` = `outputs/qwen-keyframes-lora`.
- `load_model(adapter_path)` loads the base in bf16 and wraps it with the PEFT
  adapter (`PeftModel.from_pretrained`).
- `generate(prompt, model, tokenizer, max_new_tokens=512)` applies the chat
  template, runs **greedy** decoding (`do_sample=False`), and returns only the
  newly generated tokens.
- `main()` runs a single hardcoded demo prompt; import `load_model` / `generate`
  to use it programmatically.

## Hardware notes

- Developed on an **NVIDIA RTX 4080** (Ada Lovelace). `pyproject.toml` pins the
  torch **CUDA 12.4** wheel via a `[tool.uv.sources]` index, so `uv sync` pulls
  the GPU build automatically.
- The data stages (`prepare_data.py`, `generate_data.py`) are CPU + network
  only; the GPU is needed for `train.py` / `generate.py`.
