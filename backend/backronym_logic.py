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


_SLANG_PATH = Path(__file__).parent / "data" / "slang_words.txt"
SLANG_WORDS: set[str] = {
    w.strip().lower() for w in _SLANG_PATH.read_text().splitlines() if w.strip()
}

_ACRONYMS_PATH = Path(__file__).parent / "data" / "acronyms.txt"
KNOWN_ACRONYMS: set[str] = {
    w.strip().upper() for w in _ACRONYMS_PATH.read_text().splitlines() if w.strip()
}


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


def _find_multiword_candidates(
    concepts: list[ConceptOption],
    allow_reorder: bool,
    vocab_sources: list,
    word_check,
    max_needed: int = 3,
) -> list[CandidateResult]:
    n = len(concepts)
    if n < 4:
        return []

    concept_orig_letter = [letter_of(c.original).lower() for c in concepts]
    concept_syn_letters = []
    for c in concepts:
        syn_map = {}
        if not c.locked:
            for s in c.synonyms:
                let = letter_of(s).lower()
                if let not in syn_map:
                    syn_map[let] = s
        concept_syn_letters.append(syn_map)

    all_allowed = set(concept_orig_letter)
    for sm in concept_syn_letters:
        all_allowed.update(sm.keys())

    words_by_len: dict[int, list[str]] = {}
    for vocab in vocab_sources:
        for w in vocab:
            l = len(w)
            if 2 <= l <= 7 and not any(ch not in all_allowed for ch in w.lower()):
                words_by_len.setdefault(l, []).append(w.lower())

    splits = []
    for l1 in range(max(2, n - 7), min(8, n - 1)):
        l2 = n - l1
        if 2 <= l2 <= 7:
            splits.append([l1, l2])
    if n >= 6:
        for l1 in range(max(2, n - 14), min(7, n - 3)):
            for l2 in range(max(2, n - l1 - 7), min(7, n - l1 - 1)):
                l3 = n - l1 - l2
                if 2 <= l3 <= 7:
                    splits.append([l1, l2, l3])
    splits.sort(key=lambda s: (len(s), max(s) - min(s)))

    results: list[CandidateResult] = []
    seen_phrases: set[str] = set()

    if not allow_reorder:
        for split in splits:
            idx = 0
            seg_matches = []
            possible = True
            for seg_len in split:
                seg_c_orig = concept_orig_letter[idx:idx + seg_len]
                seg_c_syn = concept_syn_letters[idx:idx + seg_len]
                idx += seg_len
                valid_words = []
                for w in words_by_len.get(seg_len, []):
                    m_possible = True
                    used = []
                    subs = 0
                    for pos, ch in enumerate(w):
                        c_idx = idx - seg_len + pos
                        if seg_c_orig[pos] == ch:
                            used.append(concepts[c_idx].original)
                        elif ch in seg_c_syn[pos]:
                            used.append(seg_c_syn[pos][ch])
                            subs += 1
                        else:
                            m_possible = False
                            break
                    if m_possible:
                        valid_words.append((w.upper(), subs, used))
                if not valid_words:
                    possible = False
                    break
                valid_words.sort(key=lambda x: x[1])
                seg_matches.append(valid_words[:2])
            if possible:
                for combo in itertools.product(*seg_matches):
                    phrase = " ".join(c[0] for c in combo)
                    if phrase not in seen_phrases:
                        seen_phrases.add(phrase)
                        total_subs = sum(c[1] for c in combo)
                        all_used = [w for c in combo for w in c[2]]
                        results.append(CandidateResult(
                            word=phrase,
                            concept_order=[c.original for c in concepts],
                            words_used=all_used,
                            substitutions=total_subs,
                            reordered=False,
                            passes_word_check=True,
                            score=sum(pronounceability_score(c[0]) for c in combo),
                        ))
                        if len(results) >= max_needed:
                            return results
    else:
        def match_target(target: str, allowed_indices: list[int]):
            t_len = len(target)
            best = None
            min_s = 999
            def dfs(pos, used_mask, cur_subs):
                nonlocal best, min_s
                if cur_subs >= min_s:
                    return
                if pos == t_len:
                    best = (cur_subs, list(order), list(words))
                    min_s = cur_subs
                    return
                ch = target[pos]
                for idx in allowed_indices:
                    if not (used_mask & (1 << idx)):
                        if concept_orig_letter[idx] == ch:
                            order.append(idx)
                            words.append(concepts[idx].original)
                            dfs(pos + 1, used_mask | (1 << idx), cur_subs)
                            order.pop()
                            words.pop()
                for idx in allowed_indices:
                    if not (used_mask & (1 << idx)):
                        if ch in concept_syn_letters[idx]:
                            order.append(idx)
                            words.append(concept_syn_letters[idx][ch])
                            dfs(pos + 1, used_mask | (1 << idx), cur_subs + 1)
                            order.pop()
                            words.pop()
            order = []
            words = []
            dfs(0, 0, 0)
            return best

        for split in splits:
            if len(split) == 2:
                l1, l2 = split
                w1_candidates = words_by_len.get(l1, [])
                for w1 in w1_candidates[:60]:
                    m1 = match_target(w1, list(range(n)))
                    if m1 and m1[0] <= 4:
                        used_indices = set(m1[1])
                        rem_indices = [i for i in range(n) if i not in used_indices]
                        for w2 in words_by_len.get(l2, []):
                            m2 = match_target(w2, rem_indices)
                            if m2 and m2[0] <= 4:
                                phrase = f"{w1.upper()} {w2.upper()}"
                                if phrase not in seen_phrases:
                                    seen_phrases.add(phrase)
                                    order = m1[1] + m2[1]
                                    words_used = m1[2] + m2[2]
                                    results.append(CandidateResult(
                                        word=phrase,
                                        concept_order=[concepts[i].original for i in order],
                                        words_used=words_used,
                                        substitutions=m1[0] + m2[0],
                                        reordered=True,
                                        passes_word_check=True,
                                        score=pronounceability_score(w1) + pronounceability_score(w2),
                                    ))
                                    if len(results) >= max_needed:
                                        return results
                                    break
            elif len(split) == 3 and len(results) < max_needed:
                l1, l2, l3 = split
                for w1 in words_by_len.get(l1, [])[:30]:
                    m1 = match_target(w1, list(range(n)))
                    if m1 and m1[0] <= 3:
                        u1 = set(m1[1])
                        rem1 = [i for i in range(n) if i not in u1]
                        for w2 in words_by_len.get(l2, [])[:30]:
                            m2 = match_target(w2, rem1)
                            if m2 and m2[0] <= 3:
                                u2 = u1 | set(m2[1])
                                rem2 = [i for i in range(n) if i not in u2]
                                for w3 in words_by_len.get(l3, []):
                                    m3 = match_target(w3, rem2)
                                    if m3 and m3[0] <= 3:
                                        phrase = f"{w1.upper()} {w2.upper()} {w3.upper()}"
                                        if phrase not in seen_phrases:
                                            seen_phrases.add(phrase)
                                            order = m1[1] + m2[1] + m3[1]
                                            words_used = m1[2] + m2[2] + m3[2]
                                            results.append(CandidateResult(
                                                word=phrase,
                                                concept_order=[concepts[i].original for i in order],
                                                words_used=words_used,
                                                substitutions=m1[0] + m2[0] + m3[0],
                                                reordered=True,
                                                passes_word_check=True,
                                                score=pronounceability_score(w1) + pronounceability_score(w2) + pronounceability_score(w3),
                                            ))
                                            if len(results) >= max_needed:
                                                return results
                                            break
                                if len(results) >= max_needed:
                                    return results
    return results


def search(
    concepts: list[ConceptOption],
    allow_reorder: bool = False,
    require_real_word: bool = True,
    allow_slang_and_acronyms: bool = False,
    max_concepts: int = 15,
    max_results: int = 3,
) -> SearchResult:
    
    n = len(concepts)
    if n > max_concepts:
        raise ValueError(f"search() supports up to {max_concepts} concepts (got {n}).")

    base_check = is_real_word if require_real_word else is_pronounceable
    if allow_slang_and_acronyms:
        def word_check(word: str) -> bool:
            return base_check(word) or is_slang(word) or is_known_acronym(word)
    else:
        word_check = base_check

    # For small concept counts (n <= 5), the permutation & combinatorial substitution search works well.
    if (not allow_reorder and n <= 5) or (allow_reorder and n <= 5):
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

        # If single-word search found nothing and n >= 4, attempt multi-word phrases
        vocab_sources = [COMMON_WORDS]
        if allow_slang_and_acronyms:
            vocab_sources.extend([SLANG_WORDS, KNOWN_ACRONYMS])
        if not require_real_word:
            vocab_sources.append(DICTIONARY_WORDS)
        multiword_cands = _find_multiword_candidates(
            concepts,
            allow_reorder=allow_reorder,
            vocab_sources=vocab_sources,
            word_check=word_check,
            max_needed=max_results,
        )
        if multiword_cands:
            return SearchResult(success=True, results=multiword_cands[:max_results], near_misses=[])

        near_misses = sorted(all_candidates, key=lambda c: c.score, reverse=True)[:3]
        return SearchResult(success=False, results=[], near_misses=near_misses)

    # For larger concept counts (up to 15), dictionary-guided branch & bound search
    concept_orig_letter = [letter_of(c.original).lower() for c in concepts]
    concept_syn_letters = []
    for c in concepts:
        syn_map = {}
        if not c.locked:
            for s in c.synonyms:
                let = letter_of(s).lower()
                if let not in syn_map:
                    syn_map[let] = s
        concept_syn_letters.append(syn_map)

    vocab_sources = [COMMON_WORDS]
    if allow_slang_and_acronyms:
        vocab_sources.extend([SLANG_WORDS, KNOWN_ACRONYMS])
    if not require_real_word:
        vocab_sources.append(DICTIONARY_WORDS)

    all_allowed_letters = set(concept_orig_letter)
    for s_map in concept_syn_letters:
        all_allowed_letters.update(s_map.keys())

    matched_candidates: list[CandidateResult] = []
    seen_words: set[str] = set()

    for vocab in vocab_sources:
        for raw_word in vocab:
            if len(raw_word) != n:
                continue
            w_lower = raw_word.lower()
            if any(ch not in all_allowed_letters for ch in w_lower):
                continue
            target_letters = list(w_lower)

            if not allow_reorder:
                subs = 0
                words_used = []
                possible = True
                for i in range(n):
                    t_let = target_letters[i]
                    if concept_orig_letter[i] == t_let:
                        words_used.append(concepts[i].original)
                    elif t_let in concept_syn_letters[i]:
                        words_used.append(concept_syn_letters[i][t_let])
                        subs += 1
                    else:
                        possible = False
                        break
                if possible:
                    cand_word = w_lower.upper()
                    if cand_word not in seen_words:
                        seen_words.add(cand_word)
                        matched_candidates.append(CandidateResult(
                            word=cand_word,
                            concept_order=[c.original for c in concepts],
                            words_used=words_used,
                            substitutions=subs,
                            reordered=False,
                            passes_word_check=word_check(cand_word),
                            score=pronounceability_score(cand_word)
                        ))
            else:
                best_match = None
                min_subs = 999
                def dfs(pos, used_mask, cur_subs):
                    nonlocal best_match, min_subs
                    if cur_subs >= min_subs:
                        return
                    if pos == n:
                        best_match = (cur_subs, list(cur_order), list(cur_words))
                        min_subs = cur_subs
                        return
                    t_let = target_letters[pos]
                    for i in range(n):
                        if not (used_mask & (1 << i)):
                            if concept_orig_letter[i] == t_let:
                                cur_order.append(i)
                                cur_words.append(concepts[i].original)
                                dfs(pos + 1, used_mask | (1 << i), cur_subs)
                                cur_order.pop()
                                cur_words.pop()
                    for i in range(n):
                        if not (used_mask & (1 << i)):
                            if t_let in concept_syn_letters[i]:
                                cur_order.append(i)
                                cur_words.append(concept_syn_letters[i][t_let])
                                dfs(pos + 1, used_mask | (1 << i), cur_subs + 1)
                                cur_order.pop()
                                cur_words.pop()
                cur_order = []
                cur_words = []
                dfs(0, 0, 0)
                if best_match:
                    cand_word = w_lower.upper()
                    if cand_word not in seen_words:
                        seen_words.add(cand_word)
                        subs, order, words_used = best_match
                        matched_candidates.append(CandidateResult(
                            word=cand_word,
                            concept_order=[concepts[idx].original for idx in order],
                            words_used=words_used,
                            substitutions=subs,
                            reordered=True,
                            passes_word_check=word_check(cand_word),
                            score=pronounceability_score(cand_word)
                        ))
            if len(matched_candidates) >= 50:
                break
        if len(matched_candidates) >= max_results:
            break

    passing = [c for c in matched_candidates if c.passes_word_check]
    if len(passing) < max_results:
        # Also explore multi-word phrases (e.g. 2-word or 3-word combinations)
        multiword_cands = _find_multiword_candidates(
            concepts,
            allow_reorder=allow_reorder,
            vocab_sources=vocab_sources,
            word_check=word_check,
            max_needed=max_results - len(passing),
        )
        for mwc in multiword_cands:
            if mwc.word not in seen_words:
                seen_words.add(mwc.word)
                passing.append(mwc)

    if passing:
        passing.sort(key=lambda c: (c.substitutions, c.reordered))
        return SearchResult(success=True, results=passing[:max_results], near_misses=[])

    base_words = [c.original for c in concepts]
    base_acronym = "".join(letter_of(w) for w in base_words)
    near_misses = [CandidateResult(
        word=base_acronym,
        concept_order=[c.original for c in concepts],
        words_used=base_words,
        substitutions=0,
        reordered=False,
        passes_word_check=word_check(base_acronym),
        score=pronounceability_score(base_acronym)
    )]
    return SearchResult(success=False, results=[], near_misses=near_misses)