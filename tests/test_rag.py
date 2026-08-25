"""Unit tests for RAG retriever — no live Gemini required."""

from __future__ import annotations

from living_evidence_graph.rag import (
    DISCLAIMER,
    K_UNSET,
    STRICT_ABSTAIN_MESSAGE,
    answer_strict,
    format_context,
    graph_mode,
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



def _ten_edge_graph() -> dict:
    """Generic 10-edge graph: ranking must not hide the lowest-trust edge."""
    nodes = [
        {"id": "drug:pembrolizumab", "type": "Drug", "label": "Keytruda (pembrolizumab)"},
        {"id": "gene:PDCD1", "type": "Gene", "label": "PDCD1"},
        {
            "id": "condition:non_small_cell_lung_cancer",
            "type": "Condition",
            "label": "non-small cell lung cancer",
        },
        {"id": "ae:hepatitis", "type": "AdverseEventConcept", "label": "hepatitis"},
        {"id": "ae:pneumonitis", "type": "AdverseEventConcept", "label": "pneumonitis"},
    ]
    edges = []
    for i in range(1, 9):
        edges.append(
            {
                "id": f"edge_{i}",
                "type": "warns_ae" if i % 2 == 0 else "studied_in",
                "source": "drug:pembrolizumab",
                "target": "ae:pneumonitis" if i % 2 == 0 else "condition:non_small_cell_lung_cancer",
                "trust_score": round(0.95 - (i * 0.04), 4),
                "sources": ["dailymed_label"] if i % 2 == 0 else ["clinicaltrials_registry"],
                "evidence_urls": ["https://dailymed.nlm.nih.gov/"] if i % 2 == 0 else [],
            }
        )
    edges.append(
        {
            "id": "edge_9",
            "type": "warns_ae",
            "source": "drug:pembrolizumab",
            "target": "ae:hepatitis",
            "trust_score": 0.11,
            "sources": ["dailymed_label"],
            "evidence_urls": ["https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=demo-hep"],
            "props": {"note": "DailyMed hepatitis warning"},
        }
    )
    edges.append(
        {
            "id": "edge_10",
            "type": "cites",
            "source": "drug:pembrolizumab",
            "target": "ae:hepatitis",
            "trust_score": 0.05,
            "sources": ["preprint"],
            "evidence_urls": [],
            "props": {"note": "lowest-trust filler"},
        }
    )
    return {
        "goal": "pembrolizumab / Keytruda NSCLC",
        "nodes": nodes,
        "edges": edges,
        "meta": {},
    }



def _n_edge_graph(n: int, *, mode: str = "public") -> dict:
    nodes = [
        {"id": "drug:pembrolizumab", "type": "Drug", "label": "Keytruda (pembrolizumab)"},
        {"id": "gene:PDCD1", "type": "Gene", "label": "PDCD1"},
    ]
    edges = []
    for i in range(1, n + 1):
        edges.append(
            {
                "id": f"edge_{i}",
                "type": "cites",
                "source": "drug:pembrolizumab",
                "target": "gene:PDCD1",
                "trust_score": round(max(0.01, 0.99 - i * 0.005), 4),
                "sources": ["private_library", mode] if mode != "public" else ["opentargets_kb"],
                "evidence_urls": [],
            }
        )
    meta: dict = {}
    if mode != "public":
        meta = {"mode": mode, "library": True, "source_boundary": "private"}
    return {"goal": f"{mode} graph", "nodes": nodes, "edges": edges, "meta": meta}


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



def test_default_retrieve_returns_every_edge_including_lowest_trust():
    """Default retrieve is the whole graph, ranked — trust must not drop edges."""
    g = _ten_edge_graph()
    assert len(g["edges"]) == 10
    q = "What does DailyMed warn about hepatitis for Keytruda / pembrolizumab?"
    defaulted = retrieve_edges(q, graph=g)
    omitted = retrieve_edges(q, graph=g, k=None)
    zero = retrieve_edges(q, graph=g, k=0)
    for edges in (defaulted, omitted, zero):
        ids = [e["id"] for e in edges]
        assert len(edges) == 10
        assert "edge_9" in ids
        assert "edge_10" in ids
        trusts = [float(e["trust_score"] or 0) for e in edges]
        assert min(trusts) == 0.05
        assert all("retrieval_score" in e for e in edges)


def test_explicit_k_caps_when_smaller_than_edge_count():
    g = _ten_edge_graph()
    edges = retrieve_edges("Keytruda NSCLC DailyMed hepatitis", graph=g, k=3)
    assert len(edges) == 3
    assert all(e.get("id") for e in edges)



def test_public_default_is_all_ranked_edges():
    g = _ten_edge_graph()
    assert graph_mode(g) == "public"
    q = "Keytruda DailyMed hepatitis"
    assert len(retrieve_edges(q, graph=g)) == 10
    assert len(retrieve_edges(q, graph=g, k=K_UNSET)) == 10


def test_personal_default_k_is_32():
    g = _n_edge_graph(40, mode="personal")
    assert graph_mode(g) == "personal"
    q = "What does my library say about pembrolizumab?"
    defaulted = retrieve_edges(q, graph=g)
    assert len(defaulted) == 32
    assert len(retrieve_edges(q, graph=g, k=None)) == 40
    assert len(retrieve_edges(q, graph=g, k=0)) == 40
    assert len(retrieve_edges(q, graph=g, k="all")) == 40


def test_enterprise_default_k_is_128():
    g = _n_edge_graph(140, mode="enterprise")
    assert graph_mode(g) == "enterprise"
    q = "What does the vault say about pembrolizumab?"
    defaulted = retrieve_edges(q, graph=g)
    assert len(defaulted) == 128
    assert len(retrieve_edges(q, graph=g, k=0)) == 140
    assert len(retrieve_edges(q, graph=g, k="all")) == 140


def test_graph_smaller_than_k_injects_all_never_invents():
    personal = _n_edge_graph(10, mode="personal")
    enterprise = _n_edge_graph(10, mode="enterprise")
    public = _n_edge_graph(10, mode="public")
    q = "pembrolizumab"
    for g in (personal, enterprise, public):
        edges = retrieve_edges(q, graph=g, k=32)
        assert len(edges) == 10
        assert all(e.get("id") for e in edges)
    assert len(retrieve_edges(q, graph=personal)) == 10  # default 32 > 10
    assert len(retrieve_edges(q, graph=enterprise)) == 10  # default 128 > 10


def test_public_opt_in_without_number_is_32():
    g = _n_edge_graph(40, mode="public")
    q = "pembrolizumab"
    assert len(retrieve_edges(q, graph=g, k=True)) == 32
    assert len(retrieve_edges(q, graph=g, k="")) == 32
    assert len(retrieve_edges(q, graph=g, k="default")) == 32


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


def test_disclaimer_safety_posture():
    assert "not a medical product" in DISCLAIMER.lower()
    assert "not causation" in DISCLAIMER.lower() or "not rates" in DISCLAIMER.lower()
    assert "retrieval-only" in DISCLAIMER.lower()


def test_answer_strict_empty_retrieval_abstains_without_gemini(monkeypatch):
    """Empty graph → fixed abstain message; Gemini must not be called."""
    calls = {"n": 0}

    def _boom(*, system: str, user: str):
        calls["n"] += 1
        raise AssertionError("Gemini must not be called when retrieval is empty")

    monkeypatch.setattr("living_evidence_graph.rag._call_gemini", _boom)
    empty = {"goal": "empty", "nodes": [], "edges": [], "meta": {}}
    out = answer_strict("What about zebra oncology trials?", graph=empty, k=5)
    assert calls["n"] == 0
    assert out["mode"] == "strict"
    assert out["abstained"] is True
    assert out["used"] is False
    assert out["status"] == "abstained"
    assert out["text"] == STRICT_ABSTAIN_MESSAGE
    assert out["retrieved_edges"] == []
    assert "no related information" in out["text"].lower()


def test_answer_strict_nonempty_sets_mode_strict(monkeypatch):
    """Non-empty retrieval → Gemini called with SYSTEM_STRICT; mode=strict."""
    seen = {}

    def _fake(*, system: str, user: str):
        seen["system"] = system
        seen["user"] = user
        return {
            "status": "ok",
            "model": "fake-model",
            "text": "Cited edge:edge:indicated:pembrolizumab:nsclc only.",
            "used": True,
        }

    monkeypatch.setattr("living_evidence_graph.rag._call_gemini", _fake)
    g = _mini_graph()
    out = answer_strict(
        "What indication edges link pembrolizumab to NSCLC?",
        graph=g,
        k=3,
    )
    assert out["mode"] == "strict"
    assert out["abstained"] is False
    assert out["used"] is True
    assert len(out["retrieved_edges"]) > 0
    assert "ONLY from the provided living-evidence-graph" in seen["system"] or (
        "ONLY from the provided" in seen["system"]
    )
    assert "outside" in seen["system"].lower() or "pretrained" in seen["system"].lower()

def test_system_strict_answers_supported_clauses():
    """Mixed questions: answer graph-backed clauses; abstain only on missing slice."""
    from living_evidence_graph.rag import SYSTEM_STRICT

    assert STRICT_ABSTAIN_MESSAGE in SYSTEM_STRICT
    low = SYSTEM_STRICT.lower()
    assert "every clause" in low or "supported" in low
    assert "global abstain" in low
    assert "insufficient to answer" not in low


def test_answer_strict_nonempty_user_prompt_is_partial(monkeypatch):
    """When edges exist, user prompt must not push a full-question abstain."""
    seen = {}

    def _fake(*, system: str, user: str):
        seen["system"] = system
        seen["user"] = user
        return {
            "status": "ok",
            "model": "fake-model",
            "text": "Cited edges for supported clauses; KEYNOTE-888 unsupported.",
            "used": True,
        }

    monkeypatch.setattr("living_evidence_graph.rag._call_gemini", _fake)
    g = _mini_graph()
    out = answer_strict(
        "What NSCLC indication does the graph list? What is the OS HR for KEYNOTE-888?",
        graph=g,
        k=3,
    )
    assert out["abstained"] is False
    assert len(out["retrieved_edges"]) > 0
    user = seen["user"].lower()
    assert "that clause only" in user or "unsupported" in user
    assert "do not use the global abstain" in user


def test_demo_rag_question_is_mixed_graph_and_keynote_888():
    """Default /rag + compare question: graph-backed clauses + KEYNOTE-888 trap."""
    from living_evidence_graph.config import DEMO_RAG_QUESTION

    src = DEMO_RAG_QUESTION
    assert "What NSCLC indication and PDCD1 target" in src
    assert "DailyMed" in src
    assert "pneumonitis" in src and "hepatitis" in src
    assert "KEYNOTE-799" in src
    assert "NCT03631784" in src
    assert "KEYNOTE-888" in src
    assert "hazard ratio" in src.lower()
    assert "Do not invent IDs or claim causation" not in src


def test_get_rag_uses_default_question(monkeypatch):
    """GET /rag (no query) uses DEMO_RAG_QUESTION, omitted k (public=all), strict=true."""
    from fastapi.testclient import TestClient

    from living_evidence_graph.config import DEMO_RAG_QUESTION
    from living_evidence_graph.server import app

    captured: dict = {}

    def _fake(question, k=None, strict=False, graph_slug=None):
        captured["question"] = question
        captured["k"] = k
        captured["strict"] = strict
        captured["graph_slug"] = graph_slug
        if k is K_UNSET:
            k = None
        return {
            "question": question,
            "retrieved_edges": [{"id": "edge:indicated:pembrolizumab:nsclc", "type": "drug_indicated_for_disease"}],
            "bare": {"status": "ok", "text": "bare-stub", "used": False},
            "grounded": {"status": "ok", "text": "grounded-stub", "used": False},
            "strict": {
                "status": "ok",
                "text": "graph-backed clauses; KEYNOTE-888 unsupported",
                "used": False,
            },
            "disclaimer": "not a medical product",
            "k": k,
            "graph_path": "/tmp/leg-out/graph/pembrolizumab_non_small_cell_lung_cancer.json",
            "graph_slug": "pembrolizumab_non_small_cell_lung_cancer",
            "gemini_used": False,
        }

    monkeypatch.setattr("living_evidence_graph.server.rag_compare", _fake)
    client = TestClient(app)
    r = client.get("/rag")
    assert r.status_code == 200
    assert captured["question"] == DEMO_RAG_QUESTION
    assert captured["k"] is K_UNSET
    assert captured["k"] != 8
    assert captured["strict"] is True
    body = r.json()
    assert body["question"] == DEMO_RAG_QUESTION
    assert body["retrieved_edges"]
    assert "grounded" in body
    assert "strict" in body
    assert body["strict_requested"] is True
    assert "KEYNOTE-799" in body["question"]
    assert "NCT03631784" in body["question"]
    # Handler itself must not invent extra NCT/PMID strings.
    assert "NCT04875416" not in str(body)
