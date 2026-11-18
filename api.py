from motor import buscar_tfidf, buscar_bm25, N
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="NFCorpus - TF-IDF y BM25")

class SearchRequest(BaseModel):
    query: str
    model: Literal["bm25", "tfidf"]
    top_k: int = Field(default=10, ge=1, strict=True)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value):
        if not value.strip():
            raise ValueError("query no puede estar vacía")
        return value.strip()

@app.get("/health")
def health():
    return {"status": "ok", "corpus": "nfcorpus", "documents": N}

@app.post("/search")
def search(request: SearchRequest):
    function = buscar_bm25 if request.model == "bm25" else buscar_tfidf
    return function(request.query, request.top_k)
