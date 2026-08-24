"""Private directory → living graph ingest + refresh + strict abstain."""

from __future__ import annotations

from pathlib import Path

import living_evidence_graph.changes as changes_mod
import living_evidence_graph.graph_store as store_mod
import living_evidence_graph.private_ingest as pi
from living_evidence_graph.rag import STRICT_ABSTAIN_MESSAGE, answer_strict, rag_compare


def _patch_graph_dirs(tmp_path: Path, monkeypatch) -> Path:
    gdir = tmp_path / "graph"
    ddir = tmp_path / "demo"
    gdir.mkdir()
    ddir.mkdir()
    monkeypatch.setattr(pi, "GRAPH_DIR", gdir)
    monkeypatch.setattr(store_mod, "GRAPH_DIR", gdir)
    monkeypatch.setattr(changes_mod, "GRAPH_DIR", gdir)
    monkeypatch.setattr(changes_mod, "DEMO_DIR", ddir)
    return gdir


def test_ingest_two_md_builds_nodes_and_edges(tmp_path, monkeypatch):
    _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "alpha.md").write_text(
        "# Alpha Protocol\n\nAlpha notes mention pembrolizumab and PDCD1.\n",
        encoding="utf-8",
    )
    (lib / "beta.md").write_text(
        "# Beta Findings\n\nSee alpha.md for protocol. Beta reviews PDCD1.\n",
        encoding="utf-8",
    )

    result = pi.ingest_directory(lib, slug="my-lib", mode="personal")
    assert result["status"] == "success"
    assert result["library_slug"] == "private_my_lib"
    assert result["file_count"] == 2
    assert result["node_count"] >= 2
    assert result["edge_count"] >= 1

    graph = store_mod.load_graph("private_my_lib")
    assert len(graph["nodes"]) == result["node_count"]
    assert len(graph["edges"]) == result["edge_count"]
    assert (graph.get("meta") or {}).get("public_demo_mixed") is False
    types = {n["type"] for n in graph["nodes"]}
    assert "SourceDoc" in types
    # File-path provenance only
    for e in graph["edges"]:
        assert "private_library" in (e.get("sources") or []) or "personal" in (
            e.get("sources") or []
        )
        assert e.get("evidence_urls"), "edges must cite file paths"
        joined = " ".join(e.get("evidence_urls") or [])
        assert "alpha.md" in joined or "beta.md" in joined or "file://" in joined

    status = pi.library_status("my-lib")
    assert status is not None
    assert status["file_count"] == 2
    assert status["node_count"] == result["node_count"]
    assert status["edge_count"] == result["edge_count"]


def test_refresh_after_delete_changes_digest_or_edges(tmp_path, monkeypatch):
    gdir = _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib2"
    lib.mkdir()
    a = lib / "keep.md"
    b = lib / "drop.md"
    a.write_text("# Keep Doc\n\nStable content about protocol.\n", encoding="utf-8")
    b.write_text("# Drop Doc\n\nTemporary note mentioning keep.md.\n", encoding="utf-8")

    first = pi.ingest_directory(lib, slug="refresh-demo", mode="enterprise")
    assert first["file_count"] == 2
    edges_before = first["edge_count"]
    nodes_before = first["node_count"]

    b.unlink()
    second = pi.ingest_directory(lib, slug="refresh-demo", mode="enterprise")
    assert second["file_count"] == 1
    assert second["library_slug"] == "private_refresh_demo"

    # Edge and/or node count should drop after delete + full replace
    assert second["edge_count"] < edges_before or second["node_count"] < nodes_before

    digest_path = gdir / "private_refresh_demo.changes.json"
    assert digest_path.exists()
    change_meta = second.get("change_digest") or {}
    # After refresh with a real prior graph, digest should record removals/updates
    assert change_meta.get("change_count", 0) >= 1 or second["edge_count"] != edges_before

    # Public demo digest must not be overwritten by private ingest
    assert not (tmp_path / "demo" / "change_digest.json").exists()


def test_strict_rag_empty_private_graph_abstains(tmp_path, monkeypatch):
    gdir = _patch_graph_dirs(tmp_path, monkeypatch)
    # Empty private graph on disk
    empty = {
        "goal": "empty private lib",
        "nodes": [],
        "edges": [],
        "meta": {"library": True, "mode": "personal", "library_slug": "private_empty"},
    }
    store_mod.save_graph(empty, goal_slug="private_empty")
    monkeypatch.setattr("living_evidence_graph.rag.GRAPH_DIR", gdir)

    calls: list[str] = []

    def _fake(*, system: str = "", user: str = ""):
        calls.append(system[:40])
        return {
            "status": "ok",
            "model": "test",
            "text": "should-not-appear-in-strict-empty",
            "used": True,
        }

    monkeypatch.setattr("living_evidence_graph.rag._call_gemini", _fake)

    strict = answer_strict(
        "What does the private library say about zebra therapy?",
        graph=empty,
        k=5,
    )
    assert strict["abstained"] is True
    assert strict["status"] == "abstained"
    assert STRICT_ABSTAIN_MESSAGE in strict["text"]
    assert strict["used"] is False
    # Empty strict path must not call Gemini
    assert calls == []

    # Via rag_compare + graph_slug (empty library): bare/grounded may call Gemini,
    # but strict must abstain without an extra SYSTEM_STRICT call.
    out = rag_compare(
        "Anything about zebra?",
        k=5,
        strict=True,
        graph_slug="private_empty",
    )
    assert out["strict"]["abstained"] is True
    assert out["retrieved_edges"] == []
    assert STRICT_ABSTAIN_MESSAGE in out["strict"]["text"]
    assert not any("ONLY from the provided" in c or "library edges" in c for c in calls)


def test_public_default_rag_ignores_private_slug_files(tmp_path, monkeypatch):
    """Default discover must prefer public demo, not private_* libraries."""
    gdir = _patch_graph_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr("living_evidence_graph.rag.GRAPH_DIR", gdir)
    monkeypatch.setattr("living_evidence_graph.rag.DEMO_DIR", tmp_path / "demo")

    # Only a private graph present — default discover should not treat it as public demo
    store_mod.save_graph(
        {
            "goal": "private only",
            "nodes": [{"id": "sourcedoc:x", "type": "SourceDoc", "label": "x"}],
            "edges": [],
            "meta": {"library": True},
        },
        goal_slug="private_only",
    )
    from living_evidence_graph.rag import discover_graph_path, load_rag_graph

    assert discover_graph_path() is None
    # Explicit slug still works
    doc = load_rag_graph(graph_slug="private_only")
    assert len(doc["nodes"]) == 1
    assert doc["meta"].get("empty") is not True
