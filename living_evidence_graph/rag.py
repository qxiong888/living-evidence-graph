"""Retrieval-augmented answering over the living evidence graph.

Contest demo path: question → high-trust graph edges → Gemini answer grounded
only in that context, compared to bare Gemini. Retrieval-only — no fine-tuning,
no dumping PubMed abstracts into training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from living_evidence_graph.config import (
    DEMO_DIR,
    DEMO_GOAL,
    GEMINI_MODEL,
    GRAPH_DIR,
    gemini_api_key,
    has_gemini_key,
)
from living_evidence_graph.schema import TRIANGLE_EDGE_TYPES, display_source_labels

DISCLAIMER = (
    "Public evidence graph demo only — not a medical product. "
    "openFDA FAERS values are voluntary reports, not incidence rates, and not causation. "
    "Not endorsed by FDA, NLM, NIH, NCBI, Open Targets, or ChEMBL. "
    "LLM path is retrieval-only over the graph (no abstract/full-text training dumps)."
)

SYSTEM_GROUNDED = (
    "You answer using ONLY the provided living-evidence-graph edges. "
    "Cite edges by their edge id or type + endpoints. "
    "Do NOT invent NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts. "
    "Do NOT claim causation or incidence rates. "
    "openFDA FAERS figures are voluntary reports only. "
    "If the graph lacks evidence for the question, say so clearly. "
    "This is not medical advice and not an endorsement by FDA/NLM/NIH."
)

# Library-only / strict: stronger than grounded — no outside knowledge, abstain if missing.
SYSTEM_STRICT = (
    "You answer ONLY from the provided living-evidence-graph / library edges. "
    "Use nothing from your pretrained knowledge or the open web. "
    "Cite edges by their edge id or type + endpoints. "
    "Do NOT invent NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts. "
    "Do NOT claim causation, incidence rates, or medical certainty. "
    "openFDA FAERS figures are voluntary reports only. "
    "If retrieved edges were provided, answer every clause those edges support "
    "and cite the supporting edge ids. "
    "For any clause with no supporting edge, say that no related information "
    "was found / that clause is unsupported — do not invent it. "
    "Do NOT reply with the single-line global abstain if any edges were provided. "
    "Only if the provided edges are empty, reply with exactly: "
    "No related information was found in the evidence graph for this question. "
    "Never invent an answer. This is not medical advice and not an endorsement by FDA/NLM/NIH."
)

STRICT_ABSTAIN_MESSAGE = (
    "No related information was found in the evidence graph for this question."
)

SYSTEM_BARE = (
    "You are a careful oncology research assistant for a contest demo. "
    "Do NOT invent specific NCT IDs, PMIDs, setids, ChEMBL IDs, or FDA counts. "
    "Do NOT claim causation or incidence rates from FAERS-style reports. "
    "Say when you are uncertain. Not medical advice."
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Synonyms so “Keytruda / NSCLC” questions hit the demo graph entities.
_ALIASES: dict[str, tuple[str, ...]] = {
    "keytruda": ("pembrolizumab", "keytruda", "mk3475", "mk-3475"),
    "pembrolizumab": ("pembrolizumab", "keytruda"),
    "nsclc": ("nsclc", "non-small", "lung", "non_small_cell_lung_cancer"),
    "lung": ("lung", "nsclc", "non-small"),
    "pd1": ("pdcd1", "pd-1", "pd1", "cd279"),
    "pd-1": ("pdcd1", "pd-1", "pd1"),
    "pdcd1": ("pdcd1", "pd-1", "pd1"),
    "faers": ("reports_ae", "openfda", "adverse"),
    "adverse": ("reports_ae", "warns_ae", "adverse", "openfda"),
    "warning": ("warns_ae", "boxed", "warning", "dailymed"),
    "trial": ("studied_in", "clinicaltrials", "nct"),
    "target": ("drug_targets_gene", "gene", "chembl", "opentargets"),
    "indication": ("drug_indicated_for_disease", "treats_indication", "indication"),
}


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def _expand_query_tokens(question: str) -> set[str]:
    toks = _tokenize(question)
    expanded = set(toks)
    low = (question or "").lower()
    for key, aliases in _ALIASES.items():
        if key in low or key in toks:
            expanded.update(aliases)
            expanded.update(_tokenize(" ".join(aliases)))
    return expanded


def _is_graph_json(path: Path) -> bool:
    """Exclude sidecar files (prev/changes/manifest) from graph discovery."""
    name = path.name.lower()
    if not name.endswith(".json"):
        return False
    if name.endswith(".prev.json") or name.endswith(".changes.json"):
        return False
    if name.endswith(".manifest.json"):
        return False
    return True


def resolve_graph_slug(slug: str | None) -> Path | None:
    """Resolve library_slug / graph_slug to a graph JSON path under GRAPH_DIR."""
    raw = (slug or "").strip()
    if not raw:
        return None
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [raw]
    if not raw.startswith("private_"):
        candidates.append(f"private_{raw}")
    for s in candidates:
        path = GRAPH_DIR / f"{s}.json"
        if path.is_file():
            return path
    return None


def discover_graph_path(
    explicit: str | Path | None = None,
    *,
    graph_slug: str | None = None,
) -> Path | None:
    """Pick a graph JSON. Prefer explicit path or slug; else public Keytruda demo.

    Private library graphs are never mixed into the default public demo selection
    unless graph_slug / library_slug points at them.
    """
    if graph_slug:
        by_slug = resolve_graph_slug(graph_slug)
        if by_slug is not None:
            return by_slug
        # Explicit slug requested but missing → empty (caller gets abstain / empty)
        return None
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    if GRAPH_DIR.is_dir():
        candidates.extend(sorted(c for c in GRAPH_DIR.glob("*.json") if _is_graph_json(c)))
    # Prefer Keytruda / pembrolizumab public demo slug when present.
    # Exclude private_* libraries from the default public pick.
    public = [c for c in candidates if not c.name.startswith("private_")]
    preferred = [
        c
        for c in public
        if "pembrolizumab" in c.name.lower() or "keytruda" in c.name.lower()
    ]
    pool = preferred or public
    if not pool:
        # Fall back to any JSON under out/demo that looks like a graph
        for p in DEMO_DIR.glob("*.json"):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(doc, dict) and "edges" in doc and "nodes" in doc:
                pool.append(p)
    if not pool:
        return None
    return max(pool, key=lambda p: p.stat().st_mtime)


def load_rag_graph(
    path: str | Path | None = None,
    *,
    graph_slug: str | None = None,
) -> dict[str, Any]:
    """Load graph document for RAG. Empty doc if nothing on disk."""
    resolved = discover_graph_path(path, graph_slug=graph_slug)
    if resolved is None:
        return {
            "goal": DEMO_GOAL if not graph_slug else f"library:{graph_slug}",
            "nodes": [],
            "edges": [],
            "meta": {
                "rag_graph_path": None,
                "empty": True,
                "graph_slug": graph_slug,
            },
        }
    with resolved.open(encoding="utf-8") as f:
        doc = json.load(f)
    meta = dict(doc.get("meta") or {})
    meta["rag_graph_path"] = str(resolved)
    if graph_slug:
        meta["graph_slug"] = graph_slug
    doc["meta"] = meta
    return doc


def _node_index(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {n["id"]: n for n in nodes if n.get("id")}


def _edge_haystack(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = [
        str(edge.get("id") or ""),
        str(edge.get("type") or ""),
        str(edge.get("source") or ""),
        str(edge.get("target") or ""),
        " ".join(str(s) for s in (edge.get("sources") or [])),
        " ".join(str(u) for u in (edge.get("evidence_urls") or [])),
    ]
    props = edge.get("props") or {}
    if isinstance(props, dict):
        parts.append(" ".join(str(v) for v in props.values() if isinstance(v, (str, int, float))))
    for endpoint in (edge.get("source"), edge.get("target")):
        node = nodes_by_id.get(str(endpoint) if endpoint else "")
        if node:
            parts.append(str(node.get("label") or ""))
            parts.append(str(node.get("type") or ""))
            nprops = node.get("props") or {}
            if isinstance(nprops, dict):
                parts.append(
                    " ".join(str(v) for v in nprops.values() if isinstance(v, (str, int, float)))
                )
    return " ".join(parts).lower()


# Intent tokens → edge types that should surface even vs high-trust triangle defaults.
_INTENT_EDGE_TYPES: dict[str, frozenset[str]] = {
    "faers": frozenset({"reports_ae"}),
    "openfda": frozenset({"reports_ae"}),
    "adverse": frozenset({"reports_ae", "warns_ae"}),
    "reports": frozenset({"reports_ae"}),
    "warning": frozenset({"warns_ae"}),
    "warnings": frozenset({"warns_ae"}),
    "boxed": frozenset({"warns_ae"}),
    "labelled": frozenset({"warns_ae"}),
    "labeled": frozenset({"warns_ae"}),
    "trial": frozenset({"studied_in"}),
    "trials": frozenset({"studied_in"}),
    "nct": frozenset({"studied_in"}),
    "target": frozenset({"drug_targets_gene"}),
    "targets": frozenset({"drug_targets_gene"}),
    "indication": frozenset({"drug_indicated_for_disease", "treats_indication"}),
    "indicated": frozenset({"drug_indicated_for_disease", "treats_indication"}),
    "spine": frozenset(TRIANGLE_EDGE_TYPES),
    "triangle": frozenset(TRIANGLE_EDGE_TYPES),

}


def score_edge_for_question(
    edge: dict[str, Any],
    query_tokens: set[str],
    nodes_by_id: dict[str, dict[str, Any]],
) -> float:
    """Rank score: trust + keyword overlap, with triangle / evidence URL boosts."""
    trust = float(edge.get("trust_score") or 0.0)
    hay = _edge_haystack(edge, nodes_by_id)
    hay_tokens = _tokenize(hay)
    if not query_tokens:
        overlap = 0.0
    else:
        hits = query_tokens & hay_tokens
        # Also count substring hits for multi-part ids like non_small_cell_lung_cancer
        substr = 0
        for q in query_tokens:
            if len(q) >= 3 and q in hay:
                substr += 1
        overlap = (len(hits) + 0.25 * substr) / max(len(query_tokens), 1)

    etype = str(edge.get("type") or "")
    # Prefer triangle spine, but do not drown out intent-matched corroboration edges.
    intent_types: set[str] = set()
    for tok in query_tokens:
        intent_types |= set(_INTENT_EDGE_TYPES.get(tok, ()))
    intent_hit = etype in intent_types
    triangle_bonus = 0.35 if etype in TRIANGLE_EDGE_TYPES else 0.0
    if intent_types and not intent_hit and etype in TRIANGLE_EDGE_TYPES:
        triangle_bonus = 0.1  # still slight preference, not a lock
    intent_bonus = 0.55 if intent_hit else 0.0
    urls = edge.get("evidence_urls") or []
    url_bonus = 0.15 if urls else 0.0
    # Soft preference for multi-source corroboration
    sources = edge.get("sources") or []
    source_bonus = min(0.1, 0.03 * len(set(sources)))
    retracted_penalty = 0.5 if edge.get("retracted") else 0.0

    return (
        0.55 * trust
        + 0.45 * min(1.0, overlap)
        + triangle_bonus
        + intent_bonus
        + url_bonus
        + source_bonus
        - retracted_penalty
    )


def retrieve_edges(
    question: str,
    *,
    graph: dict[str, Any] | None = None,
    k: int | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return ranked edges for the question (compact dicts, no invented IDs).

    Default (k is None / omitted / 0): every edge in the loaded graph, ranked
    only. Trust and retrieval score never drop an edge on that path, and the
    per-type diversity cap is not applied. An explicit k is an optional cap
    used only when k < edge_count (bounded, e.g. le=200, for tests).
    """
    doc = graph if graph is not None else load_rag_graph(path)
    nodes_by_id = _node_index(list(doc.get("nodes") or []))
    query_tokens = _expand_query_tokens(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for edge in doc.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        s = score_edge_for_question(edge, query_tokens, nodes_by_id)
        compact = _compact_edge(edge, nodes_by_id, retrieval_score=s)
        scored.append((s, compact))
    scored.sort(key=lambda t: (-t[0], -(t[1].get("trust_score") or 0.0), t[1].get("id") or ""))
    edge_count = len(scored)
    if edge_count == 0:
        return []

    # Default / omitted / 0 → whole graph. Explicit k is a cap only when smaller.
    want_all = k is None
    if not want_all:
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = 0
        if k_int <= 0:
            want_all = True
        else:
            k = min(k_int, 200)
            if k >= edge_count:
                want_all = True
    if want_all:
        return [compact for _, compact in scored]

    intent_types: set[str] = set()
    for tok in query_tokens:
        intent_types |= set(_INTENT_EDGE_TYPES.get(tok, ()))

    # Soft per-type diversity so many warns_ae / reports_ae do not crowd out spine edges.
    max_per_type = max(2, (k + 2) // 3)
    picked: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    seen_ids: set[str] = set()

    def _try_add(compact: dict[str, Any], *, ignore_cap: bool = False) -> bool:
        eid = str(compact.get("id") or "")
        if eid and eid in seen_ids:
            return False
        et = str(compact.get("type") or "")
        if not ignore_cap and type_counts.get(et, 0) >= max_per_type:
            return False
        picked.append(compact)
        if eid:
            seen_ids.add(eid)
        type_counts[et] = type_counts.get(et, 0) + 1
        return True

    # Coverage pass: at least one edge per intent-matched type (when available).
    for want in sorted(intent_types):
        if len(picked) >= k:
            break
        for _, compact in scored:
            if compact.get("type") == want and _try_add(compact, ignore_cap=True):
                break

    # Score-ordered fill with diversity cap.
    for _, compact in scored:
        if len(picked) >= k:
            break
        _try_add(compact)

    # If still short, ignore cap.
    if len(picked) < k:
        for _, compact in scored:
            if len(picked) >= k:
                break
            _try_add(compact, ignore_cap=True)

    return picked


def _compact_edge(
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    *,
    retrieval_score: float,
) -> dict[str, Any]:
    src = str(edge.get("source") or "")
    tgt = str(edge.get("target") or "")
    src_node = nodes_by_id.get(src) or {}
    tgt_node = nodes_by_id.get(tgt) or {}
    return {
        "id": edge.get("id"),
        "type": edge.get("type"),
        "source": src,
        "source_label": src_node.get("label") or src,
        "target": tgt,
        "target_label": tgt_node.get("label") or tgt,
        "trust_score": edge.get("trust_score"),
        "sources": list(edge.get("sources") or [])
        or display_source_labels(None, list(edge.get("evidence_urls") or [])),
        "evidence_urls": list(edge.get("evidence_urls") or [])[:8],
        "triangle_spine": str(edge.get("type") or "") in TRIANGLE_EDGE_TYPES,
        "retrieval_score": round(float(retrieval_score), 4),
        "retracted": bool(edge.get("retracted")),
    }


def format_context(edges: list[dict[str, Any]]) -> str:
    """Compact context block for the grounded Gemini prompt."""
    if not edges:
        return (
            "GRAPH CONTEXT: (empty — no edges retrieved. "
            "Say that the living evidence graph lacks evidence for this question.)"
        )
    lines = [
        "GRAPH CONTEXT (cite only these edges; do not invent IDs or rates):",
        "NOTE: openFDA / reports_ae = voluntary reports, not incidence rates or causation.",
        "",
    ]
    for i, e in enumerate(edges, 1):
        urls = e.get("evidence_urls") or []
        url_s = "; ".join(urls[:5]) if urls else "(none)"
        srcs = ", ".join(display_source_labels(e.get("sources"), e.get("evidence_urls"))) or "(none)"
        spine = " [triangle spine]" if e.get("triangle_spine") else ""
        lines.append(
            f"{i}. [{e.get('type')}]{spine} "
            f"{e.get('source_label')} ({e.get('source')}) → "
            f"{e.get('target_label')} ({e.get('target')})"
        )
        lines.append(
            f"   edge_id={e.get('id')} | trust={e.get('trust_score')} | "
            f"sources=[{srcs}]"
        )
        lines.append(f"   evidence_urls: {url_s}")
    return "\n".join(lines)


def _call_gemini(*, system: str, user: str) -> dict[str, Any]:
    """Call Gemini via Gemini API. Prefer GEMINI_MODEL; on 429 try gemini-3.6-flash.

    Never print the API key.
    """
    if not has_gemini_key():
        return {
            "status": "gemini_skipped",
            "model": GEMINI_MODEL,
            "text": (
                "[gemini_skipped] No GEMINI_API_KEY / GOOGLE_API_KEY in environment. "
                "Retrieval context was still built; answers require an AI Studio key."
            ),
            "used": False,
        }
    key = gemini_api_key()
    models = [GEMINI_MODEL]
    if GEMINI_MODEL != "gemini-3.6-flash":
        models.append("gemini-3.6-flash")
    last_err: Exception | None = None
    last_model = models[0]
    try:
        from google import genai
        from google.genai import types
    except Exception as e:  # noqa: BLE001
        return {
            "status": "gemini_error",
            "model": GEMINI_MODEL,
            "text": f"[gemini_error] {type(e).__name__}: {e}",
            "used": False,
            "error": str(e),
        }

    client = genai.Client(api_key=key)
    for model in models:
        last_model = model
        try:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(system_instruction=system),
                )
            except (TypeError, AttributeError):
                # Older google-genai may not support system_instruction the same way
                resp = client.models.generate_content(
                    model=model,
                    contents=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
                )
            text = (getattr(resp, "text", None) or str(resp)).strip()
            return {
                "status": "ok",
                "model": model,
                "text": text[:8000],
                "used": True,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            err = str(e)
            if "429" not in err and "RESOURCE_EXHAUSTED" not in err:
                break
    e = last_err
    return {
        "status": "gemini_error",
        "model": last_model,
        "text": f"[gemini_error] {type(e).__name__}: {e}" if e else "[gemini_error]",
        "used": False,
        "error": str(e) if e else "",
    }


def answer_bare(question: str) -> dict[str, Any]:
    """Gemini answer without graph context."""
    user = f"Question: {question}\n\nProvide a brief, careful answer."
    result = _call_gemini(system=SYSTEM_BARE, user=user)
    result["mode"] = "bare"
    result["question"] = question
    return result


def answer_with_graph(
    question: str,
    *,
    k: int | None = None,
    graph: dict[str, Any] | None = None,
    path: str | Path | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retrieve ranked graph edges (all by default) and answer with Gemini from that context."""
    doc = graph if graph is not None else load_rag_graph(path)
    retrieved = edges if edges is not None else retrieve_edges(question, graph=doc, k=k)
    context = format_context(retrieved)
    user = (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the graph context above. "
        "Refuse causation and rate claims. If evidence is missing, say so."
    )
    result = _call_gemini(system=SYSTEM_GROUNDED, user=user)
    result["mode"] = "grounded"
    result["question"] = question
    result["context"] = context
    result["retrieved_edges"] = retrieved
    result["graph_path"] = (doc.get("meta") or {}).get("rag_graph_path")
    return result


def answer_strict(
    question: str,
    *,
    k: int | None = None,
    graph: dict[str, Any] | None = None,
    path: str | Path | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Library-only answer: only from retrieved edges; abstain if none retrieved.

    If retrieval returns no edges, do not call Gemini — return a fixed abstain
    message (no bare-model freestyle). If edges exist, call Gemini with
    SYSTEM_STRICT so supported clauses are answered and only unsupported
    clauses are left empty — never a global abstain on a mixed question.
    """
    doc = graph if graph is not None else load_rag_graph(path)
    retrieved = edges if edges is not None else retrieve_edges(question, graph=doc, k=k)
    graph_path = (doc.get("meta") or {}).get("rag_graph_path")
    if len(retrieved) == 0:
        return {
            "status": "abstained",
            "model": GEMINI_MODEL,
            "text": STRICT_ABSTAIN_MESSAGE,
            "used": False,
            "abstained": True,
            "mode": "strict",
            "question": question,
            "context": format_context([]),
            "retrieved_edges": [],
            "graph_path": graph_path,
        }
    context = format_context(retrieved)
    user = (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer ONLY from the graph / library context above. "
        "Answer each clause the retrieved edges support and cite those edges. "
        "If a clause has no supporting edge, say no related information was found "
        "for that clause only (unsupported). "
        "Do not use the global abstain sentence when any edges were retrieved. "
        "Never invent IDs or use outside knowledge."
    )
    result = _call_gemini(system=SYSTEM_STRICT, user=user)
    result["mode"] = "strict"
    result["question"] = question
    result["context"] = context
    result["retrieved_edges"] = retrieved
    result["graph_path"] = graph_path
    result["abstained"] = False
    return result


def rag_compare(
    question: str,
    *,
    k: int | None = None,
    path: str | Path | None = None,
    strict: bool = False,
    graph_slug: str | None = None,
) -> dict[str, Any]:
    """Side-by-side bare vs grounded (and optional strict) for /rag and demo_rag.py.

    graph_slug / library_slug selects a private library graph; default remains the
    public demo graph (never mixed with private folder docs).
    """
    doc = load_rag_graph(path, graph_slug=graph_slug)
    retrieved = retrieve_edges(question, graph=doc, k=k)
    grounded = answer_with_graph(question, k=k, graph=doc, edges=retrieved)
    bare = answer_bare(question)
    out: dict[str, Any] = {
        "question": question,
        "k": k,
        "graph_path": (doc.get("meta") or {}).get("rag_graph_path"),
        "graph_slug": graph_slug or (doc.get("meta") or {}).get("library_slug"),
        "goal": doc.get("goal"),
        "retrieved_edges": retrieved,
        "context": format_context(retrieved),
        "bare": bare,
        "grounded": grounded,
        "disclaimer": DISCLAIMER,
        "gemini_used": bool(bare.get("used") or grounded.get("used")),
        "strict_requested": bool(strict),
    }
    if strict:
        strict_ans = answer_strict(question, k=k, graph=doc, edges=retrieved)
        out["strict"] = strict_ans
        out["gemini_used"] = bool(
            out["gemini_used"] or strict_ans.get("used")
        )
    return out
