"""Cloud Run HTTP entry. Cloud Scheduler POSTs /scheduler daily."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from living_evidence_graph.agent import daily_refresh, ingest_goal
from living_evidence_graph.changes import diff, digest_document, load_digest, load_snapshot
from living_evidence_graph.config import DEMO_GOAL, GEMINI_MODEL
from living_evidence_graph.graph_store import load_graph
from living_evidence_graph.private_ingest import ingest_directory, library_status
from living_evidence_graph.rag import DISCLAIMER, rag_compare

app = FastAPI(
    title="Living Evidence Graph",
    description=(
        "Text-first living evidence graph for LLM use. "
        "Default demo: pembrolizumab / Keytruda (NSCLC). "
        "Public data only — not a medical product. "
        "POST /rag = retrieval-augmented answer (graph edges only; no fine-tuning). "
        "GET /changes = human-readable change digest (what / why / sources). "
        "POST /library/ingest = personal/enterprise private folder → private graph "
        "(never mixed with the public Keytruda demo)."
    ),
    version="0.1.0",
)


class RunBody(BaseModel):
    goal: str = Field(default=DEMO_GOAL, description="Evidence-graph goal string")


class RagBody(BaseModel):
    question: str = Field(
        ...,
        description="User question answered bare vs grounded (and optional strict) on retrieved graph edges",
    )
    k: int = Field(default=8, ge=1, le=25, description="Top-k edges to retrieve")
    strict: bool = Field(
        default=False,
        description=(
            "If true, also return library-only strict answer "
            "(abstain when no edges retrieved; no bare-model freestyle)"
        ),
    )
    graph_slug: str | None = Field(
        default=None,
        description="Optional graph slug (public demo or private_* library). Default = public Keytruda demo.",
    )
    library_slug: str | None = Field(
        default=None,
        description="Alias for graph_slug when targeting a personal/enterprise private library.",
    )


class LibraryIngestBody(BaseModel):
    path: str = Field(..., description="Local absolute (or relative) directory to scan")
    slug: str | None = Field(
        default=None,
        description="Library id/slug (stored as private_<slug>); default = hash of path",
    )
    mode: Literal["personal", "enterprise"] = Field(
        default="personal",
        description="Product boundary: personal or enterprise private graph",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": GEMINI_MODEL,
        "vertex_default": False,
        "demo_vertical": "pembrolizumab / Keytruda / NSCLC",
        "phi": False,
        "rag": True,
        "changes": True,
        "private_library": True,
    }


@app.post("/run")
@app.get("/run")
def run(goal: str = DEMO_GOAL) -> JSONResponse:
    try:
        result = daily_refresh(goal)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"run failed: {e}") from e
    graph = result.get("graph") or {}
    change_meta = (graph.get("meta") or {}).get("change_digest") or {}
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
            "change_digest": change_meta,
        }
    )


@app.post("/scheduler")
def scheduler(body: RunBody | None = None) -> JSONResponse:
    """Cloud Scheduler daily hook."""
    goal = (body.goal if body else None) or DEMO_GOAL
    return run(goal=goal)


@app.get("/changes")
def changes(
    goal_slug: str | None = Query(
        default=None,
        description="Graph slug; default resolves from DEMO_GOAL ingest",
    ),
    recompute: bool = Query(
        default=False,
        description="If true, re-diff prev snapshot vs current graph instead of cached digest",
    ),
) -> JSONResponse:
    """Human-readable change digest: what / why / sources (+ evidence URLs)."""
    slug = (goal_slug or "").strip() or ingest_goal(DEMO_GOAL)["goal_slug"]
    if recompute:
        prev = load_snapshot(slug)
        nxt = load_graph(slug)
        if not (nxt.get("nodes") or nxt.get("edges")):
            raise HTTPException(404, f"no graph for slug={slug}")
        events = diff(prev, nxt)
        doc = digest_document(
            events,
            goal=str(nxt.get("goal") or slug),
            goal_slug=slug,
        )
        return JSONResponse(doc)
    cached = load_digest(slug)
    if cached is None:
        prev = load_snapshot(slug)
        nxt = load_graph(slug)
        if not (nxt.get("nodes") or nxt.get("edges")):
            raise HTTPException(404, f"no change digest or graph for slug={slug}")
        events = diff(prev, nxt)
        doc = digest_document(
            events,
            goal=str(nxt.get("goal") or slug),
            goal_slug=slug,
        )
        return JSONResponse(doc)
    return JSONResponse(cached)


@app.post("/library/ingest")
def library_ingest(body: LibraryIngestBody) -> JSONResponse:
    """Scan a local directory into a personal/enterprise private living graph.

    Sync scan for contest/demo. Large directories should use the CLI:
    `python -m living_evidence_graph.private_ingest --dir PATH --slug SLUG --mode personal`.
    Never mixes private folder docs into the public Keytruda graph.
    """
    path = (body.path or "").strip()
    if not path:
        raise HTTPException(400, "path is required")
    try:
        result = ingest_directory(path, slug=body.slug, mode=body.mode)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"library ingest failed: {e}") from e
    return JSONResponse(result)


@app.get("/library/{slug}")
def get_library(slug: str) -> JSONResponse:
    """Private library status: path, file count, node/edge counts, last_updated."""
    status = library_status(slug)
    if status is None:
        raise HTTPException(404, f"no private library for slug={slug}")
    return JSONResponse(status)


@app.post("/rag")
def rag(body: RagBody) -> JSONResponse:
    """Retrieve high-trust graph edges; bare vs grounded (+ optional strict) Gemini."""
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    slug = (body.library_slug or body.graph_slug or "").strip() or None
    try:
        result = rag_compare(q, k=body.k, strict=bool(body.strict), graph_slug=slug)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"rag failed: {e}") from e
    payload = {
        "question": result["question"],
        "retrieved_edges": result["retrieved_edges"],
        "bare": result["bare"],
        "grounded": result["grounded"],
        "disclaimer": result.get("disclaimer") or DISCLAIMER,
        "k": result.get("k"),
        "graph_path": result.get("graph_path"),
        "graph_slug": result.get("graph_slug") or slug,
        "gemini_used": result.get("gemini_used"),
        "strict_requested": bool(body.strict),
    }
    if body.strict:
        payload["strict"] = result.get("strict")
    return JSONResponse(payload)
