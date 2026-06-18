"""
LoRA fine-tune Qwen2.5-Coder-3B-Instruct on the keyframe examples with TRL's
SFTTrainer. Standard supervised fine-tuning: cross-entropy on the assistant
tokens only (completion-only loss). No exotic objectives — the render/knockout
metric lives in eval.py and never touches training.

Key choices:
  - bf16 LoRA, NOT 4-bit: a 3B in bf16 (~6GB) + LoRA fits a 16GB card with room,
    and trains faster/cleaner than QLoRA. Pass --use-4bit to fall back if OOM.
  - completion-only loss via assistant_only_loss (Qwen's chat template carries
    the {% generation %} tags TRL needs).
  - same system prompt + chat template as eval.py / generate.py (prompting.py).
  - evaluates on the val split each epoch so the base-vs-FT loss comparison is
    apples-to-apples, and keeps the best checkpoint by eval loss.
  - resumable: re-running picks up from the latest checkpoint in output_dir.

Usage (run in your own terminal so you can watch):
  uv run python train.py
  uv run python train.py --epochs 2 --use-4bit
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from prompting import SYSTEM_PROMPT

# Speed up any (re)download of the base model.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--train-path", default="data/keyframes_train.jsonl")
    p.add_argument("--val-path", default="data/keyframes_val.jsonl")
    p.add_argument("--output-dir", default="outputs/qwen3b-keyframes-lora")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=10,
                   help="Checkpoint + eval every N optimizer steps (resume granularity)")
    p.add_argument("--use-4bit", action="store_true",
                   help="QLoRA fallback if bf16 LoRA OOMs")
    p.add_argument("--report-to", default="wandb",
                   help="'wandb' or 'none' (set WANDB_MODE=offline to log locally)")
    return p.parse_args()


class JsonlLossLogger(TrainerCallback):
    """Append every Trainer log line (train loss, eval_loss, lr, grad_norm, ...)
    to a plain JSONL file so a later plotting script doesn't depend on W&B. Each
    record carries step + epoch. Survives resume (just dedupe by step when plotting)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        rec = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def add_system(example: dict) -> dict:
    """Prepend the shared system prompt; keep ONLY the chat messages so SFTTrainer
    sees a clean conversational example."""
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + example["messages"]}


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    quant = None
    if args.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=quant, dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )

    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    val_ds = load_dataset("json", data_files=args.val_path, split="train")
    train_ds = train_ds.map(add_system, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(add_system, remove_columns=val_ds.column_names)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        packing=False,                 # required for completion-only masking
        assistant_only_loss=True,      # loss on assistant tokens only
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        # Step-based checkpoint + eval so an interrupted run resumes with at most
        # --save-steps steps lost (epoch-based would lose a whole epoch mid-run).
        # eval_strategy must match save_strategy for load_best_model_at_end.
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=3,            # keeps best + most-recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[JsonlLossLogger(os.path.join(args.output_dir, "train_log.jsonl"))],
    )

    # Resume from the latest checkpoint if this output_dir already has one.
    last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last:
        print(f"Resuming from checkpoint: {last}")
    trainer.train(resume_from_checkpoint=last)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
