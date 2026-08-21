"""
concept_builder.py

Same job as build_concepts() in the Phase 3 CLI script: turn raw concept
text into ConceptOptions with real, live-fetched synonyms attached. This
version has no print() calls, since there's no terminal to print to when
this runs inside a web request - the FastAPI route will decide what (if
anything) to log.

Deliberately NOT changing backronym_logic.py or llm_client.py to add this -
those are tested and working. This is a new, thin, separate piece.
"""

from backronym_logic import ConceptOption, filter_valid_synonyms
from llm_client import LLMClient


def build_concepts(terms: list[tuple[str, bool]], client: LLMClient) -> list[ConceptOption]:
    """
    terms: list of (text, locked) tuples, one per concept, in order.
    """
    concepts = []
    for text, locked in terms:
        text = text.strip()
        if locked:
            concepts.append(ConceptOption(original=text, locked=True))
            continue
        synonyms_raw = client.get_synonyms(text)
        synonyms = filter_valid_synonyms(text, synonyms_raw)
        concepts.append(ConceptOption(original=text, synonyms=synonyms))
    return concepts
