#!/usr/bin/env bash
# Print raw model output for a handful of prompts, so the text itself can be read.
# =============================================================================
# WHY: the gsm8k harness scores responses and throws the text away -- it only
#   saves accuracy / invalid_rate. To check whether a build emits garbage (the
#   repeated-"!" failure mode), the actual characters have to be visible.
#
# Greedy, /v1/completions, same endpoint and stop sequences the gsm8k harness
# uses, so what prints here is what that harness would have scored.
# =============================================================================
set -euo pipefail

MODEL="${1:?served model name required}"
PORT="${2:-8000}"

python3 - "$MODEL" "$PORT" <<'PY'
import json
import sys
import urllib.request

model, port = sys.argv[1], sys.argv[2]
url = f"http://localhost:{port}/v1/completions"

FEWSHOT = (
    "Question: Natalia sold clips to 48 friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether?\n"
    "Answer: In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. "
    "The answer is 72.\n\n"
)

PROMPTS = [
    ("gsm8k-style 1-shot",
     FEWSHOT
     + "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
       "minutes of babysitting. How much did she earn?\nAnswer:"),
    ("gsm8k-style 1-shot",
     FEWSHOT
     + "Question: Betty is saving money for a new wallet which costs $100. Betty has "
       "only half of the money she needs. Her parents decided to give her $15, and her "
       "grandparents twice as much as her parents. How much more money does Betty need?\n"
       "Answer:"),
    ("plain continuation", "The capital of France is"),
    ("plain continuation", "Count from one to ten: one, two,"),
]

for label, prompt in PROMPTS:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.0,
        "stop": ["Question", "Assistant:", "<|separator|>"],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    text = out["choices"][0]["text"]
    n_bang = text.count("!")
    print("=" * 78)
    print(f"[{label}] prompt tail: ...{prompt[-70:]!r}")
    print(f"[{label}] finish_reason={out['choices'][0].get('finish_reason')} "
          f"len={len(text)} chars, '!' count={n_bang}")
    print("--- raw output ---")
    print(repr(text))
    print("--- as text ---")
    print(text)
print("=" * 78)
print("=== dump-sample-completions: done ===")
PY
