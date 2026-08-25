"""Cloud Run HTTP entry. Cloud Scheduler POSTs /scheduler daily."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from living_evidence_graph.agent import daily_refresh, ingest_goal
from living_evidence_graph.changes import diff, digest_document, load_digest, load_snapshot
from living_evidence_graph.config import DEMO_GOAL, DEMO_GRAPH_SLUG, DEMO_RAG_QUESTION, GEMINI_MODEL
from living_evidence_graph.graph_store import load_graph
from living_evidence_graph.library_watch import (
    start_watcher,
    start_watchers_from_manifests,
    stop_all_watchers,
)
from living_evidence_graph.private_ingest import ingest_directory, library_status
from living_evidence_graph.rag import DISCLAIMER, rag_compare
from living_evidence_graph.seed import seed_demo_graph_if_missing
from living_evidence_graph.session_store import create_session, get_session

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE = "session_id"
_HTML_PAGES = {
    "index.html",
    "compare.html",
    "update.html",
    "push.html",
}
_ASSET_TYPES = {
    "style.css": "text/css",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Seed the public demo graph, then watch existing private folders."""
    seed_demo_graph_if_missing()
    start_watchers_from_manifests()
    try:
        yield
    finally:
        stop_all_watchers()


app = FastAPI(
    title="Living Evidence Graph",
    description=(
        "Text-first living evidence graph for LLM use. "
        "Default demo: pembrolizumab / Keytruda (NSCLC). "
        "Public data only — not a medical product. "
        "GET / = demo hub (compare / update / push). "
        "GET /graph = demo graph node/edge counts (seeded 14/10 on cold start). "
        "GET /rag = same JSON in a browser (default demo question, strict=true). "
        "POST /rag = retrieval-augmented answer (question required; graph edges only). "
        "POST /session/push = bind this demo session to the latest public graph. "
        "GET /changes = human-readable change digest (what / why / sources). "
        "POST /library/ingest = personal/enterprise private folder → private graph "
        "(never mixed with the public Keytruda demo; first ingest starts a folder watcher)."
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
    session_id: str | None = Field(
        default=None,
        description="Optional demo session id (also accepted as cookie or query). Binds retrieval slug.",
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


class SessionPushBody(BaseModel):
    graph_slug: str | None = Field(
        default=None,
        description="Optional slug; demo one-click binds the public Keytruda graph when omitted or missing.",
    )
    mode: Literal["grounded", "strict"] = Field(
        default="grounded",
        description="RAG mode this session will use after push: grounded or strict.",
    )


def _html_page(name: str) -> FileResponse:
    if name not in _HTML_PAGES:
        raise HTTPException(404, "unknown page")
    path = STATIC_DIR / name
    if not path.is_file():
        raise HTTPException(404, f"missing static page {name}")
    return FileResponse(path, media_type="text/html; charset=utf-8")


def _session_id_from(request: Request, session_id: str | None = None) -> str | None:
    raw = (session_id or "").strip()
    if raw:
        return raw
    cookie = (request.cookies.get(SESSION_COOKIE) or "").strip()
    return cookie or None


def _bind_slug(requested: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve a push slug. Missing/empty/unknown → public demo slug."""
    seed_demo_graph_if_missing()
    want = (requested or "").strip()
    if want:
        doc = load_graph(want)
        if doc.get("nodes") or doc.get("edges"):
            return want, doc
    doc = load_graph(DEMO_GRAPH_SLUG)
    return DEMO_GRAPH_SLUG, doc


def _resolve_rag_slug(
    request: Request,
    *,
    graph_slug: str | None,
    library_slug: str | None,
    session_id: str | None,
) -> tuple[str | None, str | None]:
    """Explicit slug wins; else bound session slug. Returns (slug, session_id)."""
    explicit = (library_slug or graph_slug or "").strip() or None
    sid = _session_id_from(request, session_id)
    if explicit:
        return explicit, sid
    sess = get_session(sid)
    if sess:
        return sess.graph_slug, sid
    return None, sid


@app.get("/", include_in_schema=False)
def hub_page() -> FileResponse:
    return _html_page("index.html")


@app.get("/compare", include_in_schema=False)
def compare_page() -> FileResponse:
    return _html_page("compare.html")


@app.get("/update", include_in_schema=False)
def update_page() -> FileResponse:
    return _html_page("update.html")


@app.get("/push", include_in_schema=False)
def push_page() -> FileResponse:
    return _html_page("push.html")


@app.get("/assets/{filename}", include_in_schema=False)
def static_asset(filename: str) -> FileResponse:
    media = _ASSET_TYPES.get(filename)
    if not media:
        raise HTTPException(404, "unknown asset")
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(404, f"missing asset {filename}")
    return FileResponse(path, media_type=media)


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
        "session_push": True,
        "pages": True,
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
    watching = start_watcher(result["library_slug"], result.get("watched_path"))
    result["auto_refresh"] = watching
    result["watching"] = watching
    return JSONResponse(result)


@app.get("/library/{slug}")
def get_library(slug: str) -> JSONResponse:
    """Private library status: path, file count, node/edge counts, last_updated."""
    status = library_status(slug)
    if status is None:
        raise HTTPException(404, f"no private library for slug={slug}")
    return JSONResponse(status)


@app.post("/session/push")
def session_push(body: SessionPushBody) -> JSONResponse:
    """Bind this demo session to the latest public graph (no file download)."""
    slug, doc = _bind_slug(body.graph_slug)
    nodes = list(doc.get("nodes") or [])
    edges = list(doc.get("edges") or [])
    sess = create_session(
        graph_slug=slug,
        mode=body.mode,
        node_count=len(nodes),
        edge_count=len(edges),
    )
    payload = {
        "bound": True,
        "session_id": sess.session_id,
        "graph_slug": sess.graph_slug,
        "mode": sess.mode,
        "created_at": sess.created_at,
        "node_count": sess.node_count,
        "edge_count": sess.edge_count,
        "demo_slug": DEMO_GRAPH_SLUG,
        "demo_question": DEMO_RAG_QUESTION,
        "disclaimer": DISCLAIMER,
        "download": False,
    }
    out = JSONResponse(payload)
    out.set_cookie(
        SESSION_COOKIE,
        sess.session_id,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=86400,
    )
    return out


@app.get("/session")
def session_get(
    request: Request,
    session_id: str | None = Query(
        default=None,
        description="Optional session id; otherwise the session_id cookie is used",
    ),
) -> JSONResponse:
    """Current demo session (bound slug/mode) plus the default RAG question."""
    sid = _session_id_from(request, session_id)
    sess = get_session(sid)
    if sess is None:
        return JSONResponse(
            {
                "bound": False,
                "session_id": None,
                "graph_slug": None,
                "mode": None,
                "demo_slug": DEMO_GRAPH_SLUG,
                "demo_question": DEMO_RAG_QUESTION,
                "disclaimer": DISCLAIMER,
            }
        )
    return JSONResponse(
        {
            "bound": True,
            "session_id": sess.session_id,
            "graph_slug": sess.graph_slug,
            "mode": sess.mode,
            "created_at": sess.created_at,
            "node_count": sess.node_count,
            "edge_count": sess.edge_count,
            "demo_slug": DEMO_GRAPH_SLUG,
            "demo_question": DEMO_RAG_QUESTION,
            "disclaimer": DISCLAIMER,
        }
    )


def _rag_payload(
    *,
    question: str,
    k: int,
    strict: bool,
    graph_slug: str | None,
    library_slug: str | None,
    session_id: str | None = None,
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
        "session_id": session_id,
        "session_bound": bool(session_id and get_session(session_id)),
    }
    if strict:
        payload["strict"] = result.get("strict")
    return payload


@app.post("/rag")
def rag_post(
    body: RagBody,
    request: Request,
    session_id: str | None = Query(
        default=None,
        description="Optional demo session id (cookie session_id also honored)",
    ),
) -> JSONResponse:
    """Retrieve high-trust graph edges; bare vs grounded (+ optional strict) Gemini."""
    slug, sid = _resolve_rag_slug(
        request,
        graph_slug=body.graph_slug,
        library_slug=body.library_slug,
        session_id=body.session_id or session_id,
    )
    return JSONResponse(
        _rag_payload(
            question=body.question,
            k=body.k,
            strict=bool(body.strict),
            graph_slug=slug,
            library_slug=None,
            session_id=sid,
        )
    )


@app.get("/rag")
def rag_get(
    request: Request,
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
    session_id: str | None = Query(
        default=None,
        description="Optional demo session id (cookie session_id also honored)",
    ),
) -> JSONResponse:
    """Same JSON as POST /rag. Open in a browser; question defaults to DEMO_RAG_QUESTION."""
    q = (question or "").strip() or DEMO_RAG_QUESTION
    slug, sid = _resolve_rag_slug(
        request,
        graph_slug=graph_slug,
        library_slug=library_slug,
        session_id=session_id,
    )
    return JSONResponse(
        _rag_payload(
            question=q,
            k=k,
            strict=bool(strict),
            graph_slug=slug,
            library_slug=None,
            session_id=sid,
        )
    )
