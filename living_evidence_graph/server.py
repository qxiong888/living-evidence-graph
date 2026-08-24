"""Cloud Run HTTP entry. Cloud Scheduler POSTs /scheduler daily."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from living_evidence_graph.agent import daily_refresh, ingest_goal
from living_evidence_graph.config import DEMO_GOAL, GEMINI_MODEL
from living_evidence_graph.rag import DISCLAIMER, rag_compare

app = FastAPI(
    title="Living Evidence Graph",
    description=(
        "Text-first living evidence graph for LLM use. "
        "Default demo: pembrolizumab / Keytruda (NSCLC). "
        "Public data only — not a medical product. "
        "POST /rag = retrieval-augmented answer (graph edges only; no fine-tuning)."
    ),
    version="0.1.0",
)


class RunBody(BaseModel):
    goal: str = Field(default=DEMO_GOAL, description="Evidence-graph goal string")


class RagBody(BaseModel):
    question: str = Field(
        ...,
        description="User question answered bare vs grounded on retrieved graph edges",
    )
    k: int = Field(default=8, ge=1, le=25, description="Top-k edges to retrieve")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": GEMINI_MODEL,
        "vertex_default": False,
        "demo_vertical": "pembrolizumab / Keytruda / NSCLC",
        "phi": False,
        "rag": True,
    }


@app.post("/run")
@app.get("/run")
def run(goal: str = DEMO_GOAL) -> JSONResponse:
    try:
        result = daily_refresh(goal)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"run failed: {e}") from e
    # Do not dump entire graph in default response if huge — include summary + path
    return JSONResponse(
        {
            "status": result.get("status"),
            "goal": result.get("goal"),
            "goal_slug": result.get("goal_slug"),
            "node_count": result.get("node_count"),
            "edge_count": result.get("edge_count"),
            "path": result.get("path"),
            "sources": result.get("sources"),
            "parsed": ingest_goal(goal),
        }
    )


@app.post("/scheduler")
def scheduler(body: RunBody | None = None) -> JSONResponse:
    """Cloud Scheduler daily hook."""
    goal = (body.goal if body else None) or DEMO_GOAL
    return run(goal=goal)


@app.post("/rag")
def rag(body: RagBody) -> JSONResponse:
    """Retrieve high-trust graph edges and compare bare vs grounded Gemini answers."""
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    try:
        result = rag_compare(q, k=body.k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"rag failed: {e}") from e
    return JSONResponse(
        {
            "question": result["question"],
            "retrieved_edges": result["retrieved_edges"],
            "bare": result["bare"],
            "grounded": result["grounded"],
            "disclaimer": result.get("disclaimer") or DISCLAIMER,
            "k": result.get("k"),
            "graph_path": result.get("graph_path"),
            "gemini_used": result.get("gemini_used"),
        }
    )
