"""Synthetic workloads with controlled token length (needle-in-haystack for long-context)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt "
    "ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
    "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
    "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
).split()


def needle_in_haystack(target_tokens: int, tokenizer, seed: int = 0) -> dict:
    """Build a single prompt where a unique needle is buried in lorem filler.

    `tokenizer` is a HF AutoTokenizer; `target_tokens` is approximate (we trim to within ~5%).
    """
    rng = random.Random(seed)
    needle_id = f"NDL-{rng.randint(100_000, 999_999)}"
    needle = f"The magic code is {needle_id}."
    question = f"What is the magic code mentioned in the passage above? Answer just the code."

    # Grow filler until we exceed target, then trim.
    words: list[str] = []
    while True:
        words.extend(rng.sample(LOREM, k=len(LOREM)))
        text = " ".join(words)
        if len(tokenizer.encode(text)) > target_tokens * 1.1:
            break
    # Insert needle near the middle.
    parts = text.split()
    mid = len(parts) // 2
    parts = parts[:mid] + needle.split() + parts[mid:]
    text = " ".join(parts)

    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        text = tokenizer.decode(ids)

    return {
        "prompt": f"{text}\n\nQuestion: {question}",
        "answer": needle_id,
        "n_tokens": len(ids),
    }


def gen_long_context_set(
    target_tokens: int, n: int, tokenizer, out: Path
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in range(n):
            row = needle_in_haystack(target_tokens, tokenizer, seed=i)
            row["target_tokens"] = target_tokens
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)
