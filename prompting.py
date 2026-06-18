"""
Single source of truth for how the model is prompted, so training, evaluation,
and inference are byte-for-byte consistent (the #1 way fine-tune A/Bs go wrong
is a train/eval template mismatch). Deliberately dependency-free so train.py can
import it without pulling in Playwright/PIL.

The user message is always the raw motion description, verbatim from the dataset.
A short fixed system prompt states the task — used identically everywhere so the
base model gets a fair shot in the A/B and inference matches training.
"""

import re

SYSTEM_PROMPT = (
    "You generate minimal, self-contained CSS @keyframes animations from a motion "
    "description. Output only the element markup and one inline <style> block — no "
    "prose, no markdown fences, no explanation."
)


def build_messages(description: str, response: str | None = None,
                   system: str = SYSTEM_PROMPT) -> list[dict]:
    """Chat messages for one example. With `response`, it's a full training turn;
    without, it's an inference prompt (caller adds the generation prompt)."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": description}]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the code out of a model response: if it's wrapped in a markdown fence
    (as the base model tends to do), return the fence contents; otherwise return
    the text as-is. Lets us separate 'animation correctness' from 'output format'
    in the eval (extracted vs raw knockout)."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()
