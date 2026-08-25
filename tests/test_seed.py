"""Seed-on-startup: baked 14-node / 10-edge Keytruda NSCLC demo graph."""

from __future__ import annotations

from living_evidence_graph.config import DEMO_GRAPH_SLUG
from living_evidence_graph.graph_store import load_graph
from living_evidence_graph.rag import discover_graph_path, load_rag_graph
from living_evidence_graph.seed import seed_demo_graph_if_missing


def _point_graph_dir(monkeypatch, tmp_path) -> None:
    import living_evidence_graph.config as config
    import living_evidence_graph.graph_store as graph_store
    import living_evidence_graph.rag as rag
    import living_evidence_graph.seed as seed

    monkeypatch.setattr(config, "GRAPH_DIR", tmp_path)
    monkeypatch.setattr(graph_store, "GRAPH_DIR", tmp_path)
    monkeypatch.setattr(rag, "GRAPH_DIR", tmp_path)
    monkeypatch.setattr(seed, "GRAPH_DIR", tmp_path)


def test_seed_then_load_graph_and_discover_finds_14_10(tmp_path, monkeypatch):
    _point_graph_dir(monkeypatch, tmp_path)
    assert not (tmp_path / f"{DEMO_GRAPH_SLUG}.json").exists()

    info = seed_demo_graph_if_missing(graph_dir=tmp_path)
    assert info["seeded"] is True
    assert info["slug"] == DEMO_GRAPH_SLUG
    assert f"{DEMO_GRAPH_SLUG}.json" in info["copied"]

    doc = load_graph(DEMO_GRAPH_SLUG)
    assert len(doc.get("nodes") or []) == 14
    assert len(doc.get("edges") or []) == 10
    assert "pembrolizumab" in (doc.get("goal") or "").lower()
    assert "Keytruda" in (doc.get("goal") or "")

    path = discover_graph_path()
    assert path is not None
    assert path.name == f"{DEMO_GRAPH_SLUG}.json"

    rag_doc = load_rag_graph()
    assert len(rag_doc.get("nodes") or []) == 14
    assert len(rag_doc.get("edges") or []) == 10
    assert rag_doc.get("meta", {}).get("empty") is not True

    again = seed_demo_graph_if_missing(graph_dir=tmp_path)
    assert again["seeded"] is False
    assert again["reason"] == "already_present"
    assert len(load_graph(DEMO_GRAPH_SLUG).get("nodes") or []) == 14
