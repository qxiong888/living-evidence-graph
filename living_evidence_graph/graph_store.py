"""Local JSON graph store under out/graph/ + optional Firestore adapter stub."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from living_evidence_graph.config import FIRESTORE_COLLECTION, GRAPH_DIR, USE_FIRESTORE
from living_evidence_graph.credibility import recompute_edges


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def graph_path(goal_slug: str = "default") -> Path:
    _ensure_dir(GRAPH_DIR)
    return GRAPH_DIR / f"{goal_slug}.json"


def load_graph(goal_slug: str = "default") -> dict[str, Any]:
    path = graph_path(goal_slug)
    if not path.exists():
        return {"goal": goal_slug, "nodes": [], "edges": [], "meta": {}}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_graph(doc: dict[str, Any], goal_slug: str = "default") -> Path:
    path = graph_path(goal_slug)
    _ensure_dir(path.parent)
    doc = dict(doc)
    meta = dict(doc.get("meta") or {})
    meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    doc["meta"] = meta
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    if USE_FIRESTORE:
        try:
            firestore_upsert(doc, goal_slug=goal_slug)
        except Exception as e:  # noqa: BLE001 — local demo must not fail on stub
            meta["firestore_error"] = str(e)
            doc["meta"] = meta
            with path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
    return path


def upsert_graph(
    *,
    goal: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    goal_slug: str = "default",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge nodes/edges by id, recompute trust, persist."""
    existing = load_graph(goal_slug)
    node_map = {n["id"]: n for n in existing.get("nodes") or [] if n.get("id")}
    for n in nodes:
        if n.get("id"):
            node_map[n["id"]] = {**node_map.get(n["id"], {}), **n}
    edge_map = {e["id"]: e for e in existing.get("edges") or [] if e.get("id")}
    for e in edges:
        if e.get("id"):
            prev = edge_map.get(e["id"], {})
            merged = {**prev, **e}
            if prev.get("first_seen") and not e.get("first_seen"):
                merged["first_seen"] = prev["first_seen"]
            edge_map[e["id"]] = merged

    scored = recompute_edges(edge_map.values())
    doc: dict[str, Any] = {
        "goal": goal or existing.get("goal") or goal_slug,
        "nodes": list(node_map.values()),
        "edges": scored,
        "meta": {**(existing.get("meta") or {}), **(meta or {})},
    }
    path = save_graph(doc, goal_slug=goal_slug)
    doc["meta"]["path"] = str(path)
    return doc


def recompute_trust(goal_slug: str = "default") -> dict[str, Any]:
    doc = load_graph(goal_slug)
    doc["edges"] = recompute_edges(doc.get("edges") or [])
    save_graph(doc, goal_slug=goal_slug)
    return doc


def firestore_upsert(doc: dict[str, Any], *, goal_slug: str) -> str | None:
    """Optional Firestore Native adapter. Requires GCP credentials when enabled."""
    if not USE_FIRESTORE:
        return None
    from google.cloud import firestore  # type: ignore

    client = firestore.Client()
    ref = client.collection(FIRESTORE_COLLECTION).document(goal_slug)
    # Store a compact summary + full JSON blob path reference
    ref.set(
        {
            "goal": doc.get("goal"),
            "node_count": len(doc.get("nodes") or []),
            "edge_count": len(doc.get("edges") or []),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "graph": doc,
        },
        merge=True,
    )
    return ref.path
