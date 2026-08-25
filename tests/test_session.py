"""Session bind + HTML pages. No live Gemini."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from living_evidence_graph.config import DEMO_GRAPH_SLUG, DEMO_RAG_QUESTION
from living_evidence_graph.server import app
from living_evidence_graph.session_store import reset_sessions


@pytest.fixture(autouse=True)
def _clean_sessions():
    reset_sessions()
    yield
    reset_sessions()


def _fake_rag(question, k=8, strict=False, graph_slug=None):
    return {
        "question": question,
        "retrieved_edges": [
            {"id": "edge_3", "type": "drug_indicated_for_disease", "source": "CHEMBL3137343"}
        ],
        "bare": {"status": "ok", "text": "bare-stub", "used": False},
        "grounded": {"status": "ok", "text": "grounded cites edge_3", "used": False},
        "strict": {"status": "ok", "text": "strict cites edge_3", "used": False},
        "disclaimer": "not a medical product",
        "k": k,
        "graph_path": f"/tmp/leg-out/graph/{DEMO_GRAPH_SLUG}.json",
        "graph_slug": graph_slug or DEMO_GRAPH_SLUG,
        "gemini_used": False,
    }


def test_get_root_and_pages_serve_html():
    client = TestClient(app)
    for path, needle in (
        ("/", b"/compare"),
        ("/compare", b"Bare vs Grounded"),
        ("/update", b"/graph"),
        ("/push", b"Import / Push"),
    ):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers.get("content-type", "")
        assert needle in r.content
        body = r.text
        assert "53/51" not in body
        assert "Elena" not in body and "Reed" not in body and "Quinn" not in body
        assert "yellow-light" not in body
        assert "NCT04875416" not in body


def test_assets_style_css():
    client = TestClient(app)
    r = client.get("/assets/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers.get("content-type", "")


def test_session_push_binds_demo_slug_and_cookie():
    client = TestClient(app)
    r = client.post("/session/push", json={"mode": "grounded"})
    assert r.status_code == 200
    body = r.json()
    assert body["bound"] is True
    assert body["graph_slug"] == DEMO_GRAPH_SLUG
    assert body["mode"] == "grounded"
    assert body["download"] is False
    assert body["demo_question"] == DEMO_RAG_QUESTION
    assert body["edge_count"] == 10
    assert body["node_count"] == 14
    sid = body["session_id"]
    assert sid
    assert r.cookies.get("session_id") == sid

    got = client.get("/session")
    assert got.status_code == 200
    s = got.json()
    assert s["bound"] is True
    assert s["session_id"] == sid
    assert s["graph_slug"] == DEMO_GRAPH_SLUG
    assert s["mode"] == "grounded"


def test_session_push_unknown_slug_falls_back_to_demo():
    client = TestClient(app)
    r = client.post("/session/push", json={"graph_slug": "does_not_exist", "mode": "strict"})
    assert r.status_code == 200
    body = r.json()
    assert body["graph_slug"] == DEMO_GRAPH_SLUG
    assert body["mode"] == "strict"
    assert body["edge_count"] == 10


def test_get_session_unbound():
    client = TestClient(app)
    r = client.get("/session")
    assert r.status_code == 200
    body = r.json()
    assert body["bound"] is False
    assert body["session_id"] is None
    assert body["demo_question"] == DEMO_RAG_QUESTION
    assert body["demo_slug"] == DEMO_GRAPH_SLUG


def test_rag_uses_bound_session_slug_from_cookie(monkeypatch):
    captured: dict = {}

    def _fake(question, k=8, strict=False, graph_slug=None):
        captured["question"] = question
        captured["k"] = k
        captured["strict"] = strict
        captured["graph_slug"] = graph_slug
        return _fake_rag(question, k=k, strict=strict, graph_slug=graph_slug)

    monkeypatch.setattr("living_evidence_graph.server.rag_compare", _fake)
    client = TestClient(app)
    pushed = client.post("/session/push", json={"mode": "strict"})
    sid = pushed.json()["session_id"]

    r = client.get("/rag")
    assert r.status_code == 200
    assert captured["graph_slug"] == DEMO_GRAPH_SLUG
    assert captured["question"] == DEMO_RAG_QUESTION
    body = r.json()
    assert body["graph_slug"] == DEMO_GRAPH_SLUG
    assert body["session_id"] == sid
    assert body["session_bound"] is True


def test_rag_honors_query_session_id(monkeypatch):
    captured: dict = {}

    def _fake(question, k=8, strict=False, graph_slug=None):
        captured["graph_slug"] = graph_slug
        return _fake_rag(question, k=k, strict=strict, graph_slug=graph_slug)

    monkeypatch.setattr("living_evidence_graph.server.rag_compare", _fake)
    binder = TestClient(app)
    pushed = binder.post("/session/push", json={"mode": "grounded"})
    sid = pushed.json()["session_id"]

    # Fresh client: no cookie, only query param.
    other = TestClient(app)
    r = other.get("/rag", params={"session_id": sid, "strict": False})
    assert r.status_code == 200
    assert captured["graph_slug"] == DEMO_GRAPH_SLUG
    assert r.json()["session_bound"] is True
    assert r.json()["session_id"] == sid


def test_post_rag_body_session_id_binds_slug(monkeypatch):
    captured: dict = {}

    def _fake(question, k=8, strict=False, graph_slug=None):
        captured["graph_slug"] = graph_slug
        captured["question"] = question
        return _fake_rag(question, k=k, strict=strict, graph_slug=graph_slug)

    monkeypatch.setattr("living_evidence_graph.server.rag_compare", _fake)
    client = TestClient(app)
    sid = client.post("/session/push", json={"mode": "grounded"}).json()["session_id"]
    r = client.post(
        "/rag",
        json={
            "question": DEMO_RAG_QUESTION,
            "k": 8,
            "strict": False,
            "session_id": sid,
        },
    )
    assert r.status_code == 200
    assert captured["graph_slug"] == DEMO_GRAPH_SLUG
    assert captured["question"] == DEMO_RAG_QUESTION
    assert r.json()["session_bound"] is True


def test_explicit_graph_slug_wins_over_session(monkeypatch):
    captured: dict = {}

    def _fake(question, k=8, strict=False, graph_slug=None):
        captured["graph_slug"] = graph_slug
        return _fake_rag(question, k=k, strict=strict, graph_slug=graph_slug)

    monkeypatch.setattr("living_evidence_graph.server.rag_compare", _fake)
    client = TestClient(app)
    client.post("/session/push", json={"mode": "grounded"})
    r = client.get("/rag", params={"graph_slug": "private_my-lib", "strict": True})
    assert r.status_code == 200
    assert captured["graph_slug"] == "private_my-lib"
