"""
Run inference with the fine-tuned model — local PEFT adapter or via API.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "outputs/qwen-keyframes-lora"


def load_model(adapter_path: str = ADAPTER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(prompt: str, model, tokenizer, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    model, tokenizer = load_model()

    prompt = "Write a CSS @keyframes animation for a smooth fade-in from left."
    result = generate(prompt, model, tokenizer)
    print(result)


if __name__ == "__main__":
    main()
