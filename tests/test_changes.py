"""Tiny synthetic before/after tests for the change digest."""

from __future__ import annotations

from living_evidence_graph.changes import (
    build_and_persist_digest,
    diff,
    digest_document,
)


def _edge(
    eid: str,
    *,
    etype: str = "studied_in",
    sources: list[str] | None = None,
    urls: list[str] | None = None,
    trust: float = 0.6,
    retracted: bool = False,
    props: dict | None = None,
) -> dict:
    return {
        "id": eid,
        "type": etype,
        "source": "drug:pembrolizumab",
        "target": "condition:nsclc",
        "sources": sources or ["clinicaltrials_registry"],
        "evidence_urls": urls or ["https://clinicaltrials.gov/study/NCT01295827"],
        "trust_score": trust,
        "retracted": retracted,
        "props": props or {"nct_id": "NCT01295827"},
    }


def test_diff_added_updated_retracted_trust_shift():
    prev = {
        "goal": "demo",
        "nodes": [
            {"id": "drug:pembrolizumab", "type": "Drug", "label": "pembrolizumab"},
            {
                "id": "pub:27216199",
                "type": "Publication",
                "label": "old title",
                "props": {"pmid": "27216199"},
                "sources": ["pubmed_peer_reviewed"],
                "evidence_urls": ["https://pubmed.ncbi.nlm.nih.gov/27216199/"],
            },
        ],
        "edges": [
            _edge("edge:studied:a", trust=0.70),
            _edge(
                "edge:support:b",
                etype="supports",
                sources=["pubmed_peer_reviewed"],
                urls=["https://pubmed.ncbi.nlm.nih.gov/27216199/"],
                trust=0.80,
                props={"pmid": "27216199"},
            ),
            _edge(
                "edge:gone",
                etype="reports_ae",
                sources=["openfda_faers"],
                urls=[],
                trust=0.40,
                props={},
            ),
        ],
    }
    nxt = {
        "goal": "demo",
        "nodes": [
            {"id": "drug:pembrolizumab", "type": "Drug", "label": "pembrolizumab"},
            {
                "id": "pub:27216199",
                "type": "Publication",
                "label": "updated title",
                "props": {"pmid": "27216199"},
                "sources": ["pubmed_peer_reviewed"],
                "evidence_urls": ["https://pubmed.ncbi.nlm.nih.gov/27216199/"],
            },
            {
                "id": "trial:NCT02142738",
                "type": "Trial",
                "label": "KEYNOTE-024",
                "props": {"nct_id": "NCT02142738"},
                "sources": ["clinicaltrials_registry"],
                "evidence_urls": ["https://clinicaltrials.gov/study/NCT02142738"],
            },
        ],
        "edges": [
            # trust_shift only (no payload change)
            _edge("edge:studied:a", trust=0.85),
            # corroboration increased → updated
            _edge(
                "edge:support:b",
                etype="supports",
                sources=["pubmed_peer_reviewed", "clinicaltrials_registry"],
                urls=[
                    "https://pubmed.ncbi.nlm.nih.gov/27216199/",
                    "https://clinicaltrials.gov/study/NCT01295827",
                ],
                trust=0.90,
                props={"pmid": "27216199"},
            ),
            # newly retracted
            _edge(
                "edge:retract:c",
                etype="supports",
                sources=["europepmc", "pubmed_peer_reviewed"],
                urls=["https://pubmed.ncbi.nlm.nih.gov/27216199/"],
                trust=0.20,
                retracted=True,
                props={"pmid": "27216199"},
            ),
        ],
    }

    events = diff(prev, nxt, at="2026-08-24T00:00:00+00:00")
    by_what = {e["what"] for e in events}
    assert "added" in by_what
    assert "updated" in by_what
    assert "retracted_or_downgraded" in by_what
    assert "trust_shift" in by_what

    added = [e for e in events if e["what"] == "added"]
    assert any(e["edge_or_node_ref"] == "trial:NCT02142738" for e in added)
    assert any("new trial" in e["why"] for e in added)

    updated = [e for e in events if e["what"] == "updated"]
    assert any("corroboration" in e["why"] for e in updated)

    retracted = [e for e in events if e["what"] == "retracted_or_downgraded"]
    assert any(e["edge_or_node_ref"] == "edge:gone" for e in retracted)
    assert any("retract" in e["why"] or "downgrad" in e["why"] for e in retracted)

    shifts = [e for e in events if e["what"] == "trust_shift"]
    assert any(e["edge_or_node_ref"] == "edge:studied:a" for e in shifts)
    assert shifts[0]["trust_before"] == 0.70
    assert shifts[0]["trust_after"] == 0.85

    # Provenance must come from payload — NCT from fixture URL/props, never invented
    sample = next(e for e in added if e["edge_or_node_ref"] == "trial:NCT02142738")
    assert "clinicaltrials" in sample["sources"] or any(
        "NCT02142738" in s for s in sample["sources"]
    )
    assert any("NCT02142738" in u for u in sample["evidence_urls"])


def test_diff_empty_when_identical():
    g = {
        "nodes": [{"id": "n1", "type": "Drug", "label": "x"}],
        "edges": [_edge("e1")],
    }
    assert diff(g, g) == []


def test_digest_document_and_persist(tmp_path, monkeypatch):
    from living_evidence_graph import changes as ch

    monkeypatch.setattr(ch, "GRAPH_DIR", tmp_path / "graph")
    monkeypatch.setattr(ch, "DEMO_DIR", tmp_path / "demo")
    (tmp_path / "graph").mkdir()
    (tmp_path / "demo").mkdir()

    prev = {"goal": "g", "nodes": [], "edges": []}
    nxt = {
        "goal": "g",
        "nodes": [
            {
                "id": "trial:NCT01295827",
                "type": "Trial",
                "label": "KEYNOTE-001",
                "props": {"nct_id": "NCT01295827"},
                "sources": ["clinicaltrials_registry"],
                "evidence_urls": ["https://clinicaltrials.gov/study/NCT01295827"],
            }
        ],
        "edges": [],
    }
    digest = build_and_persist_digest(prev, nxt, goal_slug="demo_slug", also_demo=True)
    assert digest["change_count"] >= 1
    assert (tmp_path / "demo" / "change_digest.json").exists()
    assert (tmp_path / "demo" / "change_digest.md").exists()
    assert (tmp_path / "graph" / "demo_slug.changes.json").exists()

    wrapped = digest_document(digest["changes"], goal="g", goal_slug="demo_slug")
    assert "what" in (wrapped["changes"][0])
    assert "why" in (wrapped["changes"][0])
    assert "sources" in (wrapped["changes"][0])


def test_never_invents_ids():
    prev = {"nodes": [], "edges": []}
    nxt = {
        "nodes": [{"id": "drug:x", "type": "Drug", "label": "x", "props": {}}],
        "edges": [
            {
                "id": "edge:1",
                "type": "supports",
                "source": "drug:x",
                "target": "condition:y",
                "sources": ["pubmed_peer_reviewed"],
                "evidence_urls": [],
                "props": {},
            }
        ],
    }
    events = diff(prev, nxt)
    joined = " ".join(
        " ".join(e.get("sources") or []) + " " + " ".join(e.get("evidence_urls") or [])
        for e in events
    )
    assert "NCT" not in joined
    assert "PMID" not in joined.upper() or "pmid" not in joined.lower()
