# 🧠 MnemoBuddy

Turn a list of concepts into a real, memorable acronym.

Give it `Limitations, Advantages, Data Rate, Risks` and it searches for a
genuine English word hiding in the first letters — swapping in synonyms
where needed, and optionally reordering the concepts — instead of just
handing you the first four letters and calling it a day.

```
Limitations, Advantages, Data Rate, Risks   →   BRAD, CARD, HARD
```

---

## Why this exists

Most "mnemonic generator" demos just concatenate first letters and call
it done. MnemoBuddy actually **searches** for a valid word: it tries
substituting synonyms for concepts, tries reordering them, checks the
result against real English spelling patterns, and — if nothing works —
tells you *why*, instead of returning garbage or a flat failure.

## How it works

| Step | What happens |
|---|---|
| **1. Input** | Comma-separated concepts. Prefix any concept with `!` to lock its exact wording — e.g. `Limitations, !Risks, Advantages`. |
| **2. Synonyms** | For every unlocked concept, an LLM (Groq) suggests ~4 single-word synonyms. Fake "synonyms" that just reuse a word already in the phrase (e.g. "Rate" for "Data Rate") are filtered out. |
| **3. Search** | Tries increasing numbers of substitutions — 0, then 1, then 2 — optionally trying every reordering too, until it finds up to 3 *distinct* arrangements whose first letters spell a valid word. |
| **4. Validation** | "Valid" means either a real dictionary word, or just phonetically plausible — checked against real English spelling patterns (letter-pair frequency from a 370k-word dictionary), not hand-picked rules. |
| **5. Fallback** | If nothing passes, you get the closest attempts plus a hint about what's likely blocking a match (usually a lock or fixed ordering). |

The entire search in steps 3–4 is **pure Python with zero LLM calls** —
which is what makes it possible to test exhaustively with fixture data,
get consistent results, and debug logic issues separately from prompt or
API issues.

## Architecture

```
mnemobuddy/
├── backend/                     FastAPI app + the search engine
│   ├── main.py                       API route: POST /generate
│   ├── schemas.py                    Request/response shapes
│   ├── concept_builder.py            Concepts → synonym-equipped ConceptOptions
│   ├── backronym_logic.py            The core search — pure logic, fully unit-testable
│   ├── llm_client.py                 Swappable LLM interface (Groq implementation included)
│   ├── data/
│   │   ├── common_words.txt          ~10k common words — the "must be recognizable" check
│   │   └── dictionary_words.txt      ~370k curated words — phonotactic pattern statistics
│   └── requirements.txt
└── frontend-streamlit/
    ├── app.py                        UI — talks to the backend over plain HTTP
    └── requirements.txt
```

**Frontend and backend are separate processes on purpose.** The Streamlit
UI talks to FastAPI over plain HTTP, the same way a React frontend would.
That means the frontend is swappable without touching a single line of
backend code — the backend is a real, independently useful API, not
glue code welded to one UI.

**The LLM provider is swappable too.** `llm_client.py` defines an
abstract `LLMClient` interface; `GroqClient` is the only implementation
right now, but nothing else in the codebase imports Groq directly —
adding OpenAI, Claude, or a local model later is a one-file change.

## Getting started

### Prerequisites
- Python 3.10+
- A free Groq API key → [console.groq.com](https://console.groq.com)

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env
```
Open `.env` and paste in your real `GROQ_API_KEY`.

```bash
uvicorn main:app --reload
```
Confirm it's running at **http://localhost:8000/docs**.

### 2. Frontend

In a second terminal:

```bash
cd frontend-streamlit
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Opens automatically at **http://localhost:8501**.

## Example

**Input:** `Limitations, Advantages, Data Rate, Risks`, reordering allowed, real word required

**Output:**

| Word | Built from | Substitutions |
|---|---|---|
| 🏆 **BRAD** | Barriers · Risks · Advantages · Data Rate | 1 |
| **CARD** | Constraints · Advantages · Risks · Data Rate | 1 |
| **HARD** | Handicaps · Advantages · Risks · Data Rate | 1 |

## Known limitations

- The "just pronounceable" (non-strict) mode is looser than the
  real-dictionary check, so it can let a few weak permutation artifacts
  through alongside genuinely good options.
- No thematic-relevance scoring yet — a word only has to be *valid*, not
  meaningfully connected to the concepts it came from (a future NASA/RADAR
  -style "does this actually make sense" pass is on the roadmap).
- `search()` supports up to 8 concepts with reordering enabled; beyond
  that the brute-force permutation search would need to become smarter
  rather than exhaustive.

## Testing approach

`backronym_logic.py` has no network calls in it at all, so it was
developed and verified with plain `assert`-based tests against fixture
data — no API key, no cost, no flakiness from LLM randomness. The one
file that *does* talk to a real provider (`llm_client.py`) sits behind an
interface specifically so the rest of the app never has to care.
