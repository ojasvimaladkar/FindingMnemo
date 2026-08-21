"""
main.py — Phase 4: the plumbing layer.

This file deliberately does almost nothing "smart." Its whole job:
    1. Receive a request, validated by schemas.py
    2. Call build_concepts() + search() — the exact same functions
       run_pipeline.py calls from the terminal
    3. Convert the result into the response shape and send it back

If you understand run_pipeline.py, you already understand this file -
it's the same three steps, triggered by an HTTP request instead of
running as a script.

Run locally:
    uvicorn main:app --reload
Then visit http://localhost:8000/docs to try it without any frontend yet.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backronym_logic import search
from concept_builder import build_concepts
from llm_client import GroqClient
from schemas import CandidateOut, GenerateRequest, GenerateResponse

load_dotenv()

app = FastAPI(title="MnemoBuddy API")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="Server is missing GROQ_API_KEY.")

    client = GroqClient()
    terms = [(c.text, c.locked) for c in request.concepts]

    try:
        concepts = build_concepts(terms, client)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch synonyms: {e}")

    result = search(
        concepts,
        allow_reorder=request.allow_reorder,
        require_real_word=request.require_real_word,
        allow_slang_and_acronyms=True,  # always on - no toggle needed, purely additive anyway
    )

    def to_out(candidate) -> CandidateOut:
        return CandidateOut(
            word=candidate.word,
            concept_order=candidate.concept_order,
            words_used=candidate.words_used,
            substitutions=candidate.substitutions,
            reordered=candidate.reordered,
        )

    return GenerateResponse(
        success=result.success,
        results=[to_out(c) for c in result.results],
        near_misses=[to_out(nm) for nm in result.near_misses],
    )