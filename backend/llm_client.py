"""
llm_client.py — Phase 3: the ONLY file that knows how to talk to an LLM provider.

Everything else in the app (backronym_logic.py, the FastAPI routes later)
only ever calls methods on the LLMClient interface below. They never
import `requests` or know the word "Groq" exists. That's the whole point
of the abstraction we discussed: swapping providers later means writing
one new class here, not touching the search logic or the API routes.

LLMClient is an "abstract base class" (ABC) - Python's way of saying
"this defines a contract; anyone implementing it MUST provide these
methods." If you forget to implement get_synonyms() in a new provider
class, Python will refuse to let you instantiate it, catching the mistake
immediately instead of failing mysteriously later.
"""

import json
import os
import re
from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    @abstractmethod
    def get_synonyms(self, concept: str, n: int = 4) -> list[str]:
        """Return up to n single-word synonyms for `concept`."""
        raise NotImplementedError


SYNONYM_SYSTEM_PROMPT_TEMPLATE = (
    "You suggest single-word synonyms for a given concept, useful for "
    "building an acronym. Respond with ONLY raw JSON in exactly this shape: "
    '{{"synonyms": ["word1", "word2", ...]}}. '
    "Provide exactly {n} synonyms, each a single word, ordered from most to "
    "least common. Do not include any word that already appears in the "
    "original concept. No commentary, no markdown fences, just the JSON object."
)


class GroqClient(LLMClient):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]

    def get_synonyms(self, concept: str, n: int = 4) -> list[str]:
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": SYNONYM_SYSTEM_PROMPT_TEMPLATE.format(n=n)},
                {"role": "user", "content": f"Concept: {concept}"},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
        return self._parse_synonyms(raw_text)

    @staticmethod
    def _parse_synonyms(raw_text: str) -> list[str]:
        """
        Defensive parsing: json_mode should always give clean JSON (we
        confirmed this in Phase 1's exploration), but strip markdown fences
        anyway in case a future model/provider swap doesn't guarantee it.
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        return [str(w).strip() for w in data.get("synonyms", []) if str(w).strip()]
