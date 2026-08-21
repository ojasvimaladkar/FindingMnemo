"""
app.py — MnemoBuddy's Streamlit frontend.

This talks to your FastAPI backend over plain HTTP, exactly like before -
same /generate endpoint, same request/response shapes from schemas.py.
Nothing about the backend changes.

Concepts go in ONE comma-separated field. To lock a concept (keep its
exact wording, never substitute a synonym), prefix it with "!":

    Limitations, !Risks, Advantages, Data Rate

Here "Risks" is locked, everything else is free to be substituted. This
trades a little discoverability (you have to know the "!" convention -
that's why it's spelled out in the UI's help text) for a much simpler
input: one field instead of a growing list of rows.

Run:
    pip install streamlit requests
    streamlit run app.py
Make sure your FastAPI backend (uvicorn main:app --reload) is running too -
this app is just a client for it.
"""

import requests
import streamlit as st

API_URL = "https://findingmnemo.onrender.com"

st.set_page_config(page_title="MnemoBuddy", page_icon="frontend-streamlit/assets/logo.png"", layout="centered")

def parse_concepts(raw_text: str) -> list[dict]:
    """
    'Limitations, !Risks, Advantages' ->
    [{"text": "Limitations", "locked": False},
     {"text": "Risks", "locked": True},
     {"text": "Advantages", "locked": False}]
    """
    concepts = []
    for chunk in raw_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        locked = chunk.startswith("!")
        text = chunk[1:].strip() if locked else chunk
        if text:
            concepts.append({"text": text, "locked": locked})
    return concepts


st.image("frontend-streamlit/assets/logo.png", width=80)
st.title("MnemoBuddy")
st.caption("Turn a list of concepts into a real, memorable acronym.")

# --- Input ---------------------------------------------------------
st.subheader("Your concepts")
raw_input = st.text_input(
    "Concepts, comma-separated. Prefix any concept with ! to lock its exact wording.",
    value="Limitations, Advantages, Data Rate, !Risks",
)

parsed_concepts = parse_concepts(raw_input)
if parsed_concepts:
    locked_names = [c["text"] for c in parsed_concepts if c["locked"]]
    preview = ", ".join(
        f"**{c['text']}**" + (" 🔒" if c["locked"] else "") for c in parsed_concepts
    )
    st.markdown(preview)

# --- Options ---------------------------------------------------------
st.subheader("Options")
allow_reorder = st.checkbox("Allow reordering concepts", value=False)
require_real_word = st.checkbox("Require a real dictionary word", value=True)

# --- Generate ---------------------------------------------------------
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
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                result = None

        # --- Results view -----------------------------------------------
        if result:
            st.divider()

            if result["success"]:
                st.subheader(f"Found {len(result['results'])} option(s)")
                for i, candidate in enumerate(result["results"]):
                    label = "🏆 Best match" if i == 0 else f"Option {i + 1}"
                    with st.container(border=True):
                        st.markdown(f"**{label}**")
                        st.markdown(f"## {candidate['word']}")

                        # Show each letter traced back to which concept/word
                        # produced it, flagging synonym substitutions clearly.
                        for concept_name, word_used in zip(candidate["concept_order"], candidate["words_used"]):
                            letter = word_used[0].upper()
                            if word_used.lower() == concept_name.lower():
                                st.markdown(f"**{letter}** — {word_used}")
                            else:
                                st.markdown(f"**{letter}** — {word_used}  *(synonym of \"{concept_name}\")*")

                        caption_bits = [f"{candidate['substitutions']} substitution(s)"]
                        if candidate["reordered"]:
                            caption_bits.append("reordered")
                        st.caption(" · ".join(caption_bits))
            else:
                st.subheader("No valid acronym found")
                locked_count = sum(1 for c in parsed_concepts if c["locked"])
                hints = []
                if not allow_reorder:
                    hints.append("try checking **Allow reordering concepts** — fixed order can rule out otherwise-valid words")
                if locked_count > 0:
                    hints.append(f"try unlocking one of your {locked_count} locked concept(s) to allow a synonym there")
                if hints:
                    st.info("This exact combination of settings may be structurally impossible. Before assuming the concepts don't work: " + ", or ".join(hints) + ".")
                st.write("Here are the closest attempts anyway:")
                for nm in result["near_misses"]:
                    with st.container(border=True):
                        st.markdown(f"### {nm['word']}")
                        st.caption(" — ".join(nm["words_used"]))