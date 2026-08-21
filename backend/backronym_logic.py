"""
backronym_logic.py — Phase 2: the deterministic core, with ZERO calls to any LLM.

This file answers: "given a list of concepts, and for each concept a set of
candidate words (the original + synonyms), which combination's first letters
form a real word?" That's a pure search problem. No API involved. This is
worth keeping separate and testing on its own BEFORE plugging in a real LLM
call, so that if something's wrong, we know for certain it's not a prompt
or parsing issue - it's a logic issue we can debug with plain Python.

Core building blocks:
    - ConceptOption:  one concept + its usable word choices (locked or not)
    - letter_of():    first letter of a word/phrase
    - filter_valid_synonyms(): drops "synonyms" that just reuse a word
      already in the original phrase (the "Data Rate" -> "Rate" bug we
      caught in the real Groq output)
    - is_real_word() / pronounceability_score(): the two word-type checks
    - search(): the actual combinatorial search, with reorder on/off,
      substitution-count minimization, and a graceful "closest attempts"
      fallback when nothing fully succeeds
"""

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

VOWELS = set("aeiou")

# Two DIFFERENT word lists, used for two DIFFERENT jobs - this distinction
# matters, so don't collapse them into one:
#
# COMMON_WORDS (~10k words, web-frequency based): used for is_real_word(),
# the strict "must be a RECOGNIZABLE word" check. Small and common on
# purpose - a good backronym should be a word people actually know.
#
# DICTIONARY_WORDS (~370k words, a real curated dictionary): used ONLY to
# learn phonotactic patterns (what letter pairs plausibly start/end/appear
# in real English) for pronounceability scoring. We need a MUCH bigger,
# cleaner source for this - the small web-frequency list turned out to
# contain non-English artifacts (like "msgstr", a software term) that
# accidentally validated nonsense endings. A real dictionary doesn't have
# that problem, and more data means better-calibrated bigram statistics.
_WORD_LIST_PATH = Path(__file__).parent / "data" / "common_words.txt"
COMMON_WORDS: set[str] = {
    w.strip().lower() for w in _WORD_LIST_PATH.read_text().splitlines() if w.strip()
}

_DICTIONARY_PATH = Path(__file__).parent / "data" / "dictionary_words.txt"
DICTIONARY_WORDS: set[str] = {
    w.strip().lower() for w in _DICTIONARY_PATH.read_text().splitlines() if w.strip()
}

# Two SEPARATE, opt-in word lists - deliberately not merged into
# COMMON_WORDS, because accepting slang or acronyms is a real choice
# about tone/audience, not something that should turn on silently.
# Both are plain text files: edit them freely to add or remove entries -
# curating slang in particular is subjective, so treat the starter list
# as a baseline, not a final answer.
_SLANG_PATH = Path(__file__).parent / "data" / "slang_words.txt"
SLANG_WORDS: set[str] = {
    w.strip().lower() for w in _SLANG_PATH.read_text().splitlines() if w.strip()
}

_ACRONYMS_PATH = Path(__file__).parent / "data" / "acronyms.txt"
KNOWN_ACRONYMS: set[str] = {
    w.strip().upper() for w in _ACRONYMS_PATH.read_text().splitlines() if w.strip()
}

# Built from DICTIONARY_WORDS at import time, once. These aren't hand-picked
# rules - they're just "what letter pairs actually show up at the start/
# end of real English words, and adjacent to each other anywhere in one."
# This is what lets us catch something like "LADR": no real word ends in
# "dr", so its final bigram fails the check, even though a naive vowel-
# count heuristic would have let it through.
#
# IMPORTANT: even a 370k-word "dictionary" turns out to contain technical
# abbreviations (addr, bldr, cmdr, comdr...) that aren't real English
# spelling patterns - they just happen to be in the file. A single
# occurrence shouldn't be enough to validate a pattern. So instead of raw
# set membership, we count how many DIFFERENT words show each bigram, and
# only accept patterns above MIN_BIGRAM_COUNT. Real patterns like "rd" as
# a word-ending appear in 1,000+ words; noise like "dr" as a word-ending
# appears in about 6 - all abbreviations. The gap between real patterns
# and noise is enormous, so the exact threshold isn't sensitive.
from collections import Counter as _Counter

MIN_BIGRAM_COUNT = 15

_initial_counts: _Counter = _Counter()
_final_counts: _Counter = _Counter()
_internal_counts: _Counter = _Counter()
for _w in DICTIONARY_WORDS:
    if len(_w) >= 4:
        _initial_counts[_w[:2]] += 1
        _final_counts[_w[-2:]] += 1
        for _bigram in {_w[_i:_i + 2] for _i in range(len(_w) - 1)}:  # set: count each word once
            _internal_counts[_bigram] += 1

_INITIAL_BIGRAMS: set[str] = {bg for bg, count in _initial_counts.items() if count >= MIN_BIGRAM_COUNT}
_FINAL_BIGRAMS: set[str] = {bg for bg, count in _final_counts.items() if count >= MIN_BIGRAM_COUNT}
_INTERNAL_BIGRAMS: set[str] = {bg for bg, count in _internal_counts.items() if count >= MIN_BIGRAM_COUNT}


# ---------------------------------------------------------------------------
# Basic word-level helpers
# ---------------------------------------------------------------------------

def letter_of(word_or_phrase: str) -> str:
    """First alphabetic character, uppercase. 'Data Rate' -> 'D'."""
    match = re.search(r"[A-Za-z]", word_or_phrase)
    return match.group(0).upper() if match else ""


def filter_valid_synonyms(original_concept: str, synonyms: list[str]) -> list[str]:
    """
    Drop any 'synonym' that's actually just a word already present in the
    original phrase (case-insensitive). Real bug we saw: Groq suggested
    "Rate" as a synonym for "Data Rate" - that's not a substitute, it's
    just reusing half the original phrase.
    """
    original_words = {w.lower() for w in re.findall(r"[A-Za-z]+", original_concept)}
    seen = set()
    cleaned = []
    for syn in synonyms:
        syn_clean = syn.strip()
        key = syn_clean.lower()
        if not syn_clean or key in original_words or key in seen:
            continue
        seen.add(key)
        cleaned.append(syn_clean)
    return cleaned


def is_real_word(word: str) -> bool:
    """Strict mode: is this in our common-English-word list?"""
    return word.lower() in COMMON_WORDS


def is_slang(word: str) -> bool:
    return word.lower() in SLANG_WORDS


def is_known_acronym(word: str) -> bool:
    return word.upper() in KNOWN_ACRONYMS


def pronounceability_score(word: str) -> float:
    """
    Loose mode / fallback ranking: how plausible is this AS a word, even if
    it isn't a real one? Higher = more plausible (0 = no violations found).
    Used two ways:
      1. As the pass/fail check itself in "pronounceable" mode (score >= 0)
      2. As a ranking signal for "closest attempts" when strict mode fails
         entirely - lets us show the user our best near-misses instead of
         nothing.

    Checks two kinds of things, both derived from real English words rather
    than hand-guessed rules:
      - vowel presence / consonant run length (cheap sanity checks)
      - does this word's START, END, and every adjacent letter-pair
        actually occur in real English words? This is what catches
        something like "LADR": "dr" essentially never ends a real word,
        even though it isn't a crazy consonant cluster on its own.
    """
    w = word.lower()
    if not w:
        return -99.0

    violations = 0
    if not any(c in VOWELS for c in w):
        violations += 3  # no vowels at all is a hard "doesn't sound like a word"

    run_len = 1
    for i in range(1, len(w)):
        same_type = (w[i] in VOWELS) == (w[i - 1] in VOWELS)
        run_len = run_len + 1 if same_type else 1
        if run_len > 3:
            violations += 1

    if len(w) >= 2:
        if w[:2] not in _INITIAL_BIGRAMS:
            violations += 1
        if w[-2:] not in _FINAL_BIGRAMS:
            violations += 1
        for i in range(len(w) - 1):
            if w[i:i + 2] not in _INTERNAL_BIGRAMS:
                violations += 1

    return -float(violations)


def is_pronounceable(word: str) -> bool:
    return pronounceability_score(word) >= 0


# ---------------------------------------------------------------------------
# Concept model
# ---------------------------------------------------------------------------

@dataclass
class ConceptOption:
    original: str
    locked: bool = False
    synonyms: list[str] = field(default_factory=list)  # already filtered, ready to use

    def choices(self) -> list[str]:
        """All usable words for this concept: original always first, then synonyms - unless locked."""
        if self.locked:
            return [self.original]
        return [self.original] + self.synonyms


@dataclass
class CandidateResult:
    word: str                     # the resulting acronym, e.g. "CARD"
    concept_order: list[str]      # concepts in the order used to build it
    words_used: list[str]         # which word/synonym was picked per concept, same order
    substitutions: int            # how many concepts used a synonym instead of their original word
    reordered: bool               # did this candidate use a different order than input?
    passes_word_check: bool
    score: float                  # pronounceability_score of the resulting word, for ranking near-misses


@dataclass
class SearchResult:
    success: bool
    results: list[CandidateResult]      # up to max_results distinct passing words, best first
    near_misses: list[CandidateResult]  # top 2-3 closest attempts, populated only when results is empty


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------

def _build_candidate(concepts: list[ConceptOption], order: list[int], picks: list[int],
                      word_check) -> CandidateResult:
    ordered_concepts = [concepts[i] for i in order]
    words_used = [ordered_concepts[j].choices()[picks[order[j]]] for j in range(len(order))]
    # picks is indexed by ORIGINAL concept index, not order position - see note in search()
    word = "".join(letter_of(w) for w in words_used)
    substitutions = sum(
        1 for i, w in zip(order, words_used) if w.lower() != concepts[i].original.lower()
    )
    return CandidateResult(
        word=word,
        concept_order=[ordered_concepts[j].original for j in range(len(order))],
        words_used=words_used,
        substitutions=substitutions,
        reordered=(order != list(range(len(concepts)))),
        passes_word_check=word_check(word),
        score=pronounceability_score(word),
    )


def search(
    concepts: list[ConceptOption],
    allow_reorder: bool = False,
    require_real_word: bool = True,
    allow_slang_and_acronyms: bool = False,
    max_concepts: int = 8,
    max_results: int = 3,
) -> SearchResult:
    """
    The main entry point. Tries increasing numbers of synonym substitutions
    (0, 1, 2, ...) before trying more, and only explores reordering if
    allow_reorder=True. Collects up to max_results DISTINCT passing words
    (deduped, so we never show the same word 3 times) and stops as soon as
    it has enough - since we search in order of "fewest substitutions
    first," the results list is naturally ranked best-first: closer to
    what the user actually typed comes earlier in the list.

    allow_slang_and_acronyms is independent of require_real_word: it's an
    OR, not a replacement. With it on, a word passes if it clears the
    normal bar (real word or pronounceable, per require_real_word) OR it's
    in the curated slang/acronym lists - so turning it on only ever makes
    the search MORE permissive, never less.

    max_concepts is a deliberate, documented limit: with reordering on, we
    try every permutation of concept order, which is n! - factorial growth.
    8! is ~40,000, still instant; go much higher and it stops being
    instant. If you need more than 8 concepts with reordering, the search
    strategy here would need to become smarter (e.g. only trying promising
    orderings) rather than brute-force.
    """
    n = len(concepts)
    if n > max_concepts:
        raise ValueError(f"search() supports up to {max_concepts} concepts (got {n}).")

    base_check = is_real_word if require_real_word else is_pronounceable
    if allow_slang_and_acronyms:
        def word_check(word: str) -> bool:
            return base_check(word) or is_slang(word) or is_known_acronym(word)
    else:
        word_check = base_check

    orders = [list(range(n))]
    if allow_reorder:
        orders = [list(p) for p in itertools.permutations(range(n))]

    all_candidates: list[CandidateResult] = []
    found: list[CandidateResult] = []
    seen_words: set[str] = set()
    unlocked_indices = [i for i, c in enumerate(concepts) if not c.locked]

    # Increasing substitution levels: 0 = everyone uses their original word,
    # 1 = exactly one unlocked concept swapped, etc.
    for level in range(len(unlocked_indices) + 1):
        for substituted_set in itertools.combinations(unlocked_indices, level):
            # For concepts NOT in substituted_set, pick index 0 (their original word).
            # For concepts IN substituted_set, try every synonym (index >= 1).
            alt_index_options = []
            for i in substituted_set:
                num_choices = len(concepts[i].choices())
                alt_index_options.append(range(1, num_choices) if num_choices > 1 else range(0, 0))

            for alt_combo in itertools.product(*alt_index_options) if substituted_set else [()]:
                picks = [0] * n
                for i, alt_idx in zip(substituted_set, alt_combo):
                    picks[i] = alt_idx

                for order in orders:
                    candidate = _build_candidate(concepts, order, picks, word_check)
                    all_candidates.append(candidate)
                    if candidate.passes_word_check and candidate.word not in seen_words:
                        seen_words.add(candidate.word)
                        found.append(candidate)
                        if len(found) >= max_results:
                            return SearchResult(success=True, results=found, near_misses=[])

    if found:
        return SearchResult(success=True, results=found, near_misses=[])

    # Nothing passed - return the closest attempts instead of a flat failure.
    near_misses = sorted(all_candidates, key=lambda c: c.score, reverse=True)[:3]
    return SearchResult(success=False, results=[], near_misses=near_misses)