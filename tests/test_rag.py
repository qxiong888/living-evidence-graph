"""Unit tests for RAG retriever — no live Gemini required."""

from __future__ import annotations

from living_evidence_graph.rag import (
    DISCLAIMER,
    format_context,
    retrieve_edges,
    score_edge_for_question,
)
from living_evidence_graph.schema import TRIANGLE_EDGE_TYPES


def _mini_graph() -> dict:
    return {
        "goal": "pembrolizumab / Keytruda NSCLC",
        "nodes": [
            {"id": "drug:pembrolizumab", "type": "Drug", "label": "Keytruda (pembrolizumab)"},
            {"id": "gene:PDCD1", "type": "Gene", "label": "PDCD1"},
            {
                "id": "condition:non_small_cell_lung_cancer",
                "type": "Condition",
                "label": "non-small cell lung cancer",
            },
            {"id": "ae:pneumonitis", "type": "AdverseEventConcept", "label": "pneumonitis"},
        ],
        "edges": [
            {
                "id": "edge:targets:pembrolizumab:PDCD1",
                "type": "drug_targets_gene",
                "source": "drug:pembrolizumab",
                "target": "gene:PDCD1",
                "trust_score": 0.82,
                "sources": ["opentargets_kb", "chembl"],
                "evidence_urls": ["https://platform.opentargets.org/"],
            },
            {
                "id": "edge:indicated:pembrolizumab:nsclc",
                "type": "drug_indicated_for_disease",
                "source": "drug:pembrolizumab",
                "target": "condition:non_small_cell_lung_cancer",
                "trust_score": 0.9,
                "sources": ["opentargets_kb", "dailymed_label"],
                "evidence_urls": [
                    "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=demo"
                ],
            },
            {
                "id": "edge:gene_disease:PDCD1:nsclc",
                "type": "gene_associated_with_disease",
                "source": "gene:PDCD1",
                "target": "condition:non_small_cell_lung_cancer",
                "trust_score": 0.7,
                "sources": ["opentargets_kb"],
                "evidence_urls": ["https://platform.opentargets.org/"],
            },
            {
                "id": "edge:reports:pembrolizumab:pneumonitis",
                "type": "reports_ae",
                "source": "drug:pembrolizumab",
                "target": "ae:pneumonitis",
                "trust_score": 0.4,
                "sources": ["openfda_faers"],
                "evidence_urls": [],
            },
            {
                "id": "edge:unrelated:low",
                "type": "cites",
                "source": "drug:pembrolizumab",
                "target": "ae:pneumonitis",
                "trust_score": 0.2,
                "sources": ["preprint"],
                "evidence_urls": [],
                "props": {"note": "unrelated filler about zebra"},
            },
        ],
        "meta": {},
    }


def test_retrieve_prefers_triangle_and_trust():
    g = _mini_graph()
    q = "Does Keytruda (pembrolizumab) target a gene linked to NSCLC?"
    edges = retrieve_edges(q, graph=g, k=3)
    assert len(edges) == 3
    types = [e["type"] for e in edges]
    assert "edge:unrelated:low" not in [e["id"] for e in edges]
    assert any(t in TRIANGLE_EDGE_TYPES for t in types)
    assert edges[0]["triangle_spine"] is True or edges[0]["trust_score"] >= 0.7
    assert all("retrieval_score" in e for e in edges)
    assert all(e.get("id") for e in edges)


def test_retrieve_keyword_overlap_boosts_ae_question():
    g = _mini_graph()
    q = "What openFDA FAERS adverse event reports are linked to pembrolizumab?"
    edges = retrieve_edges(q, graph=g, k=2)
    ids = [e["id"] for e in edges]
    assert "edge:reports:pembrolizumab:pneumonitis" in ids


def test_format_context_includes_urls_and_trust():
    g = _mini_graph()
    edges = retrieve_edges("Keytruda NSCLC indication", graph=g, k=2)
    ctx = format_context(edges)
    assert "GRAPH CONTEXT" in ctx
    assert "trust=" in ctx
    assert "evidence_urls" in ctx
    assert "voluntary reports" in ctx.lower() or "not incidence" in ctx.lower()


def test_score_retracted_penalty():
    nodes = {
        "drug:pembrolizumab": {"id": "drug:pembrolizumab", "label": "pembrolizumab"},
        "gene:PDCD1": {"id": "gene:PDCD1", "label": "PDCD1"},
    }
    edge = {
        "type": "drug_targets_gene",
        "source": "drug:pembrolizumab",
        "target": "gene:PDCD1",
        "trust_score": 0.9,
        "sources": ["opentargets_kb"],
        "evidence_urls": ["https://example.org"],
        "retracted": True,
    }
    q = {"pembrolizumab", "pdcd1", "target"}
    s = score_edge_for_question(edge, q, nodes)
    edge2 = dict(edge)
    edge2["retracted"] = False
    s2 = score_edge_for_question(edge2, q, nodes)
    assert s < s2


def test_disclaimer_elena_posture():
    assert "not a medical product" in DISCLAIMER.lower()
    assert "not causation" in DISCLAIMER.lower() or "not rates" in DISCLAIMER.lower()
    assert "retrieval-only" in DISCLAIMER.lower()
