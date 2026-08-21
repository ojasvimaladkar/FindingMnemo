"""
explore_groq.py — Phase 1: throwaway exploration script.

This is NOT part of the final app. It has exactly one job: call Groq for
real, with a real key, and show us the RAW, unparsed response. We need to
see this before writing any "real" code, because everything downstream
(how we parse JSON, how we handle edge cases) depends on what the model
actually sends back — not what we assume it sends back.

Two versions of the call are included on purpose:
  1. get_synonyms_plain()  -> no special JSON enforcement
  2. get_synonyms_json_mode() -> uses Groq's response_format=json_object

Comparing their output teaches you why "JSON mode" exists: without it,
models sometimes add a sentence before the JSON, or wrap it in ```json
fences, even when you ask nicely in the prompt. JSON mode forces the
model to emit ONLY valid JSON, which is one less failure case for us
to defend against later.

Setup:
    pip install requests
    export GROQ_API_KEY=your_key_here      (get one free at console.groq.com)

Run:
    python3 explore_groq.py
"""

import json
import os

import requests

GROQ_API_KEY = os.environ["GROQ_API_KEY"]  # crashes with a clear message if you forgot to set it
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You suggest single-word synonyms for a given concept, useful for "
    "building an acronym. Respond with ONLY raw JSON in exactly this shape: "
    '{"synonyms": ["word1", "word2", "word3", "word4"]}. '
    "Provide exactly 4 synonyms, each a single word, ordered from most to "
    "least common. No commentary, no markdown fences, just the JSON object."
)


def _call_groq(concept: str, json_mode: bool) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Concept: {concept}"},
        ],
        "temperature": 0.7,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def get_synonyms_plain(concept: str) -> str:
    return _call_groq(concept, json_mode=False)


def get_synonyms_json_mode(concept: str) -> str:
    return _call_groq(concept, json_mode=True)


if __name__ == "__main__":
    test_concepts = ["Limitations", "Advantages", "Data Rate", "Risks"]

    for concept in test_concepts:
        print(f"\n=== {concept} ===")

        print("-- plain (no JSON mode) --")
        raw_plain = get_synonyms_plain(concept)
        print(repr(raw_plain))  # repr() shows hidden whitespace/newlines/fences clearly

        print("-- json_mode=True --")
        raw_json = get_synonyms_json_mode(concept)
        print(repr(raw_json))

        # try parsing the json_mode version - this is the one we plan to rely on
        try:
            parsed = json.loads(raw_json)
            print("Parsed OK:", parsed)
        except json.JSONDecodeError as e:
            print("Parse FAILED:", e)
