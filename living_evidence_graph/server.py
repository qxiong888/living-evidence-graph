"""Cloud Run HTTP entry. Cloud Scheduler POSTs /scheduler daily."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from living_evidence_graph.agent import daily_refresh, ingest_goal
from living_evidence_graph.changes import diff, digest_document, load_digest, load_snapshot
from living_evidence_graph.config import DEMO_GOAL, DEMO_GRAPH_SLUG, DEMO_RAG_QUESTION, GEMINI_MODEL
from living_evidence_graph.graph_store import load_graph
from living_evidence_graph.private_ingest import ingest_directory, library_status
from living_evidence_graph.rag import DISCLAIMER, rag_compare
from living_evidence_graph.seed import seed_demo_graph_if_missing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Copy baked fixtures/demo_graph into GRAPH_DIR when the slug file is missing."""
    seed_demo_graph_if_missing()
    yield


app = FastAPI(
    title="Living Evidence Graph",
    description=(
        "Text-first living evidence graph for LLM use. "
        "Default demo: pembrolizumab / Keytruda (NSCLC). "
        "Public data only — not a medical product. "
        "GET /graph = demo graph node/edge counts (seeded 14/10 on cold start). "
        "GET /rag = same JSON in a browser (default demo question, strict=true). "
        "POST /rag = retrieval-augmented answer (question required; graph edges only). "
        "GET /changes = human-readable change digest (what / why / sources). "
        "POST /library/ingest = personal/enterprise private folder → private graph "
        "(never mixed with the public Keytruda demo)."
    ),
    version="0.1.0",
    lifespan=lifespan,
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
    seed = seed_demo_graph_if_missing()
    slug = DEMO_GRAPH_SLUG
    doc = load_graph(slug)
    return {
        "ok": True,
        "model": GEMINI_MODEL,
        "vertex_default": False,
        "demo_vertical": "pembrolizumab / Keytruda / NSCLC",
        "phi": False,
        "rag": True,
        "changes": True,
        "private_library": True,
        "graph_slug": slug,
        "node_count": len(doc.get("nodes") or []),
        "edge_count": len(doc.get("edges") or []),
        "graph_seeded": bool(seed.get("seeded") or (doc.get("nodes") or doc.get("edges"))),
    }


@app.get("/graph")
def get_graph(
    goal_slug: str | None = Query(
        default=None,
        description="Graph slug; default is the baked Keytruda/NSCLC demo",
    ),
) -> JSONResponse:
    """Public demo graph: node/edge counts plus the document (seeded on cold start)."""
    seed_demo_graph_if_missing()
    slug = (goal_slug or "").strip() or DEMO_GRAPH_SLUG
    doc = load_graph(slug)
    nodes = list(doc.get("nodes") or [])
    edges = list(doc.get("edges") or [])
    if not nodes and not edges:
        raise HTTPException(404, f"no graph for slug={slug}")
    return JSONResponse(
        {
            "goal": doc.get("goal"),
            "goal_slug": slug,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
            "meta": doc.get("meta") or {},
            "disclaimer": DISCLAIMER,
        }
    )


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


def _rag_payload(
    *,
    question: str,
    k: int,
    strict: bool,
    graph_slug: str | None,
    library_slug: str | None,
) -> dict[str, Any]:
    """Shared GET/POST /rag body. POST still requires a non-empty question."""
    q = (question or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    slug = (library_slug or graph_slug or "").strip() or None
    try:
        result = rag_compare(q, k=k, strict=bool(strict), graph_slug=slug)
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
        "strict_requested": bool(strict),
    }
    if strict:
        payload["strict"] = result.get("strict")
    return payload


@app.post("/rag")
def rag_post(body: RagBody) -> JSONResponse:
    """Retrieve high-trust graph edges; bare vs grounded (+ optional strict) Gemini."""
    return JSONResponse(
        _rag_payload(
            question=body.question,
            k=body.k,
            strict=bool(body.strict),
            graph_slug=body.graph_slug,
            library_slug=body.library_slug,
        )
    )


@app.get("/rag")
def rag_get(
    question: str | None = Query(
        default=None,
        description="Optional; defaults to DEMO_RAG_QUESTION (compare-page mixed question)",
    ),
    k: int = Query(default=8, ge=1, le=25, description="Top-k edges to retrieve"),
    strict: bool = Query(
        default=True,
        description="Default true so a browser shows graph-backed clauses + KEYNOTE-888 abstain",
    ),
    graph_slug: str | None = Query(
        default=None,
        description="Optional graph slug (public demo or private_* library)",
    ),
    library_slug: str | None = Query(
        default=None,
        description="Alias for graph_slug when targeting a private library",
    ),
) -> JSONResponse:
    """Same JSON as POST /rag. Open in a browser; question defaults to DEMO_RAG_QUESTION."""
    q = (question or "").strip() or DEMO_RAG_QUESTION
    return JSONResponse(
        _rag_payload(
            question=q,
            k=k,
            strict=bool(strict),
            graph_slug=graph_slug,
            library_slug=library_slug,
        )
    )
