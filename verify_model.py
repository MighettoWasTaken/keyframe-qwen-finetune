"""
Smoke test: load the base model onto the GPU and run a single CSS @keyframes
request, to confirm the model + CUDA + tokenizer chat template all work before
committing to a fine-tune.

No LoRA, no quantization — full bf16 (a 3B model is ~6GB, fits a 16GB card).

Usage:
  uv run python verify_model.py
  uv run python verify_model.py --prompt "Animate a pulsing neon ring."

A Hugging Face token (HF_TOKEN in .env) is optional here — Qwen2.5-Coder is
public — but speeds up the first download.
"""

import argparse
import os
import time

# Enable HF's high-performance Xet transfer for any (re)download. Set before the
# hub is imported. (hf_transfer / HF_HUB_ENABLE_HF_TRANSFER is now deprecated.)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_PROMPT = ("Write a CSS @keyframes animation: three dots that scale up in "
                  "sequence then collapse inward while a ring around them rotates.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — torch sees no GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    token = os.getenv("HF_TOKEN")
    print(f"HF token: {'set' if token else 'not set (public download)'}")

    print(f"\nLoading {args.model} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, token=token,
    ).to("cuda")
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s  "
          f"| weights on {next(model.parameters()).device} "
          f"| VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print(f"\nPrompt: {args.prompt}\n{'-' * 70}")
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    dt = time.time() - t0

    print(tokenizer.decode(gen, skip_special_tokens=True))
    print("-" * 70)
    print(f"Generated {gen.shape[0]} tokens in {dt:.1f}s ({gen.shape[0] / dt:.1f} tok/s) "
          f"| peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
