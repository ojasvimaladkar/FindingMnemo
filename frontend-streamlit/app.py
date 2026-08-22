"""
app.py — MnemoBuddy's Streamlit frontend.

This talks to your FastAPI backend over plain HTTP, exactly like before -
same /generate endpoint, same request/response shapes from schemas.py.
Nothing about the backend changes.

Concepts go in ONE comma-separated field. To lock a concept (keep its
exact wording, never substitute a synonym), prefix it with "!":

    Limitations, !Risks, Advantages, Data Rate

There's also a "Lock all concepts" checkbox for locking everything at
once without prefixing each concept individually.

THEME: a single, permanent "hacker terminal" look - near-black background,
JetBrains Mono throughout, phosphor-green accent, sharp corners, a subtle
scanline texture, and terminal-style prefixes ($ on the button, > and #
on headers). No light/dark toggle - one theme, always on.
Streamlit's default widgets (text input, checkbox, button) are re-skinned
via CSS targeting their stable data-testid attributes, since those are
far less likely to break across Streamlit versions than the generated
emotion-cache class names.

LOGO: drop a file at assets/logo.png (or .svg/.jpg - update LOGO_PATH
below to match) and it'll appear automatically in the header and as the
browser tab icon. Until then, a text monogram fallback is shown so the
app doesn't break or look unfinished in the meantime.

Run:
    pip install streamlit requests
    streamlit run app.py
Make sure your FastAPI backend (uvicorn main:app --reload) is running too -
this app is just a client for it.
"""

from pathlib import Path

import requests
import streamlit as st

API_URL = "http://localhost:8000/generate"
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="FindingMnemo",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🧠",
    layout="centered",
)


def parse_concepts(raw_text: str, lock_all: bool = False) -> list[dict]:
    """
    'Limitations, !Risks, Advantages' ->
    [{"text": "Limitations", "locked": False},
     {"text": "Risks", "locked": True},
     {"text": "Advantages", "locked": False}]

    lock_all=True marks every concept as locked, regardless of whether
    it was individually prefixed with "!".
    """
    concepts = []
    for chunk in raw_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        locked = lock_all or chunk.startswith("!")
        text = chunk[1:].strip() if chunk.startswith("!") else chunk
        if text:
            concepts.append({"text": text, "locked": locked})
    return concepts


# ---------------------------------------------------------------------------
# Theme CSS - single dark "terminal" theme
# ---------------------------------------------------------------------------

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #0a0d0a;
    --bg-raised: #0f140f;
    --text: #baffc9;
    --text-muted: #5d8a6a;
    --accent: #39ff8f;
    --accent-text: #06120a;
    --border: #1f3324;
    --border-strong: #2c4a34;
    --shadow: 0 0 18px rgba(57, 255, 143, 0.08);
    --font-body: 'JetBrains Mono', monospace;
    --font-display: 'JetBrains Mono', monospace;
}

.stApp {
    background:
        repeating-linear-gradient(0deg, rgba(57,255,143,0.015) 0px, rgba(57,255,143,0.015) 1px, transparent 1px, transparent 3px),
        var(--bg);
    color: var(--text);
    font-family: var(--font-body);
}

.mb-title {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 0 12px rgba(57, 255, 143, 0.35);
}
.mb-title::before { content: "> "; opacity: 0.6; }
.mb-subtitle {
    font-family: var(--font-body);
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-top: 4px;
}
.mb-logo-fallback {
    width: 48px; height: 48px;
    border-radius: 4px;
    background: var(--bg-raised);
    border: 1.5px solid var(--accent);
    color: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-display);
    font-weight: 700; font-size: 1.05rem;
    box-shadow: var(--shadow);
}

.mb-section-label {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
    margin: 1.4rem 0 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.mb-section-label::before { content: "# "; opacity: 0.5; }

div[data-testid="stTextInput"] input {
    background: var(--bg-raised) !important;
    border: 1.5px solid var(--border-strong) !important;
    border-radius: 4px !important;
    color: var(--accent) !important;
    font-family: var(--font-body) !important;
    padding: 0.6rem 0.8rem !important;
    caret-color: var(--accent) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(57, 255, 143, 0.25) !important;
}
div[data-testid="stTextInput"] label {
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
}

div[data-testid="stCheckbox"] label p,
div[data-testid="stToggle"] label p {
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}
div[data-testid="stCheckbox"] span[data-checked="true"],
div[data-testid="stToggle"] span[data-checked="true"] {
    background-color: var(--accent) !important;
}

div[data-testid="stButton"] button {
    background: var(--bg-raised) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.08s ease;
}
div[data-testid="stButton"] button::before { content: "$ "; }
div[data-testid="stButton"] button:hover {
    background: rgba(57, 255, 143, 0.08) !important;
}
div[data-testid="stButton"] button:active {
    transform: scale(0.98);
}

.mb-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow);
}
.mb-card-label {
    font-family: var(--font-body);
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.15rem;
}
.mb-card-word {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0.1rem 0 0.7rem;
    letter-spacing: 0.04em;
    text-shadow: 0 0 10px rgba(57, 255, 143, 0.3);
}
.mb-trace-row {
    font-family: var(--font-body);
    font-size: 0.88rem;
    color: var(--text);
    padding: 0.15rem 0;
}
.mb-trace-letter {
    display: inline-block;
    width: 22px; height: 22px;
    line-height: 20px;
    text-align: center;
    border-radius: 3px;
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-weight: 700;
    font-size: 0.76rem;
    margin-right: 8px;
}
.mb-trace-syn {
    color: var(--text-muted);
    font-style: italic;
}
.mb-card-caption {
    font-family: var(--font-body);
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-top: 0.6rem;
    border-top: 1px solid var(--border);
    padding-top: 0.5rem;
}

div[data-testid="stAlert"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
}

hr, div[data-testid="stDivider"] { border-color: var(--border) !important; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_logo, col_title = st.columns([1, 6])

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=48)
    else:
        st.markdown('<div class="mb-logo-fallback">MB</div>', unsafe_allow_html=True)

with col_title:
    st.markdown(
        '<p class="mb-title">FindingMnemo</p>'
        '<p class="mb-subtitle">Turn a list of concepts into a real, memorable acronym.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

st.markdown('<p class="mb-section-label">Your concepts</p>', unsafe_allow_html=True)
raw_input = st.text_input(
    "Concepts, comma-separated. Prefix any concept with ! to lock its exact wording.",
    value="Limitations, Advantages, Data Rate, !Risks",
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

st.markdown('<p class="mb-section-label">Options</p>', unsafe_allow_html=True)
allow_reorder = st.checkbox("Allow reordering concepts", value=False)
require_real_word = st.checkbox("Require a real dictionary word", value=True)
lock_all = st.checkbox(
    "Lock all concepts (keep every word exactly as typed, no synonyms)",
    value=False,
    help="Overrides the per-concept '!' prefix - locks everything at once, "
         "so you don't have to prefix each concept individually.",
)

parsed_concepts = parse_concepts(raw_input, lock_all=lock_all)
if parsed_concepts:
    preview = ", ".join(
        f"**{c['text']}**" + (" 🔒" if c["locked"] else "")
        for c in parsed_concepts
    )
    st.markdown(preview)

if lock_all and allow_reorder:
    st.info(
        "All concepts are locked and reordering is on - reordering has no "
        "effect here, since locking fixes each concept's word but not its "
        "position. That's fine, just flagging it."
    )

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

if st.button("Generate", type="primary"):
    if not parsed_concepts:
        st.warning("Add at least one concept first.")
    else:
        with st.spinner("Fetching synonyms and searching for a valid acronym..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "concepts": parsed_concepts,
                        "allow_reorder": allow_reorder,
                        "require_real_word": require_real_word,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the backend. Is `uvicorn main:app --reload` running "
                    "in another terminal, in your backend folder?"
                )
                result = None
            except requests.exceptions.Timeout:
                st.error("The backend took too long to respond. Please try again.")
                result = None
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                result = None
            except ValueError:
                st.error("The backend returned an invalid response.")
                result = None

        # --- Results view -------------------------------------------
        if result:
            st.divider()

            if result.get("success"):
                results = result.get("results", [])
                st.markdown(
                    f'<p class="mb-section-label">Found {len(results)} option(s)</p>',
                    unsafe_allow_html=True,
                )

                for i, candidate in enumerate(results):
                    label = "Best match" if i == 0 else f"Option {i + 1}"

                    trace_html = ""
                    for concept_name, word_used in zip(
                        candidate["concept_order"], candidate["words_used"]
                    ):
                        letter = word_used[0].upper()
                        if word_used.lower() == concept_name.lower():
                            trace_html += (
                                f'<div class="mb-trace-row">'
                                f'<span class="mb-trace-letter">{letter}</span>{word_used}'
                                f'</div>'
                            )
                        else:
                            trace_html += (
                                f'<div class="mb-trace-row">'
                                f'<span class="mb-trace-letter">{letter}</span>{word_used} '
                                f'<span class="mb-trace-syn">(synonym of "{concept_name}")</span>'
                                f'</div>'
                            )

                    caption_bits = [f"{candidate['substitutions']} substitution(s)"]
                    if candidate["reordered"]:
                        caption_bits.append("reordered")

                    card_html = (
                        f'<div class="mb-card">'
                        f'<div class="mb-card-label">{label}</div>'
                        f'<div class="mb-card-word">{candidate["word"]}</div>'
                        f'{trace_html}'
                        f'<div class="mb-card-caption">{" · ".join(caption_bits)}</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

            else:
                st.markdown('<p class="mb-section-label">No valid acronym found</p>', unsafe_allow_html=True)
                st.write("Here are the closest attempts:")

                for nm in result.get("near_misses", []):
                    card_html = (
                        f'<div class="mb-card">'
                        f'<div class="mb-card-word">{nm["word"]}</div>'
                        f'<div class="mb-card-caption">{" — ".join(nm["words_used"])}</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)