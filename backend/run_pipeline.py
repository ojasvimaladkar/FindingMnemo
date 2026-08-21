"""
run_pipeline.py — Phase 3: the full brain, running for real, end to end.

Still no FastAPI, no React - just a terminal script. This is the LAST
step before those, because it's the cheapest way to see the whole
pipeline work: type concepts in, get a real acronym out, powered by a
live Groq call. If something's wrong here, it's much easier to debug in
a 10-line script than through a browser calling an API calling a script.

What this does, matching backronym_logic.search()'s job description:
    1. Take your comma-separated concepts
    2. For each one, call Groq for real synonyms (Phase 3's new part)
    3. Filter out fake "synonyms" that just reuse the original words
       (the bug we caught and fixed in Phase 2)
    4. Hand everything to search() (the part we already proved correct
       in Phase 2, completely unchanged here)
    5. Print the result

Run:
    export GROQ_API_KEY=your_key_here      (Windows PowerShell: $env:GROQ_API_KEY = "...")
    python3 run_pipeline.py
"""

from backronym_logic import ConceptOption, filter_valid_synonyms, search
from llm_client import GroqClient


def build_concepts(raw_terms: list[str], client, locked_terms: set[str] | None = None) -> list[ConceptOption]:
    """
    Turns plain text concepts into ConceptOptions with real, live-fetched
    synonyms. `locked_terms` is a set of concept names (case-insensitive)
    that should skip synonym fetching entirely - the per-concept "keep my
    exact word" lock we designed earlier.
    """
    locked_terms = {t.lower() for t in (locked_terms or set())}
    concepts = []
    for term in raw_terms:
        term = term.strip()
        is_locked = term.lower() in locked_terms
        if is_locked:
            concepts.append(ConceptOption(original=term, locked=True))
            print(f"  {term}: locked, skipping synonym lookup")
            continue

        synonyms_raw = client.get_synonyms(term)
        synonyms = filter_valid_synonyms(term, synonyms_raw)
        print(f"  {term}: got synonyms {synonyms}")
        concepts.append(ConceptOption(original=term, synonyms=synonyms))
    return concepts


def run(raw_terms: list[str], allow_reorder: bool, require_real_word: bool,
        locked_terms: set[str] | None = None):
    client = GroqClient()

    print("Fetching synonyms...")
    concepts = build_concepts(raw_terms, client, locked_terms)

    print("\nSearching for a valid acronym...")
    result = search(concepts, allow_reorder=allow_reorder, require_real_word=require_real_word)

    print()
    if result.success:
        best = result.best
        print(f"SUCCESS: {best.word}")
        for concept_name, word in zip(best.concept_order, best.words_used):
            marker = "" if word.lower() == concept_name.lower() else f"  (synonym of '{concept_name}')"
            print(f"  {word[0].upper()} — {word}{marker}")
        print(f"\n({best.substitutions} substitution(s), reordered: {best.reordered})")
    else:
        print("No valid acronym found. Closest attempts:")
        for nm in result.near_misses:
            print(f"  {nm.word}  <-  {nm.words_used}")


if __name__ == "__main__":
    # Edit these to try different inputs while you get a feel for it.
    terms = ["Limitations", "Advantages", "Data Rate", "Risks"]
    run(terms, allow_reorder=False, require_real_word=True)
