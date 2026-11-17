"""
schemas.py

The request/response shapes for the /generate endpoint. FastAPI uses these
to validate incoming requests automatically and to generate the docs at
/docs.
"""

from pydantic import BaseModel, Field


class ConceptInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=100)
    locked: bool = False


class GenerateRequest(BaseModel):
    concepts: list[ConceptInput] = Field(..., min_length=1, max_length=15)
    allow_reorder: bool = False       # matches the "keep input order" default we locked in
    require_real_word: bool = True    # matches the "real dictionary word" default we locked in


class CandidateOut(BaseModel):
    word: str
    concept_order: list[str]
    words_used: list[str]
    substitutions: int
    reordered: bool


class GenerateResponse(BaseModel):
    success: bool
    results: list[CandidateOut]      # up to 3 distinct valid words, best first
    near_misses: list[CandidateOut]  # only populated when results is empty