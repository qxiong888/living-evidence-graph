"""Unit tests for extract path — Gemini structure vs rules fallback (no live key required)."""

from __future__ import annotations

import json
from pathlib import Path

import living_evidence_graph.extract as extract_mod
from living_evidence_graph.extract import (
    _allowed_ids,
    _extract_rules,
    _find_invented_ids,
    _source_digest,
    extract_from_sources,
)

FIXTURE = json.loads(
    (Path(__file__).resolve().parent.parent / "fixtures" / "keytruda_nsclc.json").read_text(
        encoding="utf-8"
    )
)


def _base_kwargs(**extra):
    kw = dict(
        drug_brand="Keytruda",
        drug_ingredient="pembrolizumab",
        condition="non-small cell lung cancer",
        clinicaltrials=FIXTURE["clinicaltrials"],
        pubmed=FIXTURE["pubmed"],
        openfda=FIXTURE["openfda"],
        dailymed={},
        europepmc={},
        opentargets={},
        chembl={},
    )
    kw.update(extra)
    return kw


def test_rules_extract_maps_fixture_ids_only():
    result = _extract_rules(**_base_kwargs())
    assert result["gemini"]["mode"] == "rules"
    ncts = {n["props"].get("nct_id") for n in result["nodes"] if n["type"] == "Trial"}
    assert ncts == {"NCT01295827", "NCT02142738"}
    pmids = {n["props"].get("pmid") for n in result["nodes"] if n["type"] == "Publication"}
    assert pmids == {"27216199", "26712084"}
    ae_edges = [e for e in result["edges"] if e["type"] == "reports_ae"]
    assert ae_edges
    assert all("openfda_faers" in (e.get("sources") or []) for e in ae_edges)
    # FAERS posture: reports, not rates/causation in node disclaimer
    ae_nodes = [n for n in result["nodes"] if n["type"] == "AdverseEventConcept"]
    assert ae_nodes
    assert all("not causation" in (n.get("props") or {}).get("disclaimer", "").lower() for n in ae_nodes)


def test_extract_without_key_uses_rules(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = extract_from_sources(**_base_kwargs(), use_gemini=True)
    assert result["gemini"]["used"] is False
    assert result["gemini"]["mode"] == "rules"
    assert any(e["type"] == "studied_in" for e in result["edges"])


def test_extract_use_gemini_false_skips_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    called = {"n": 0}

    def boom(_prompt: str):
        called["n"] += 1
        raise AssertionError("should not call Gemini when use_gemini=False")

    monkeypatch.setattr(extract_mod, "call_gemini_extract_json", boom)
    result = extract_from_sources(**_base_kwargs(), use_gemini=False)
    assert called["n"] == 0
    assert result["gemini"]["mode"] == "rules"


def test_gemini_structure_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    def fake_call(_prompt: str):
        return {
            "ok": True,
            "used": True,
            "model": "gemini-3.5-flash",
            "data": {
                "nodes": [
                    {
                        "id": "drug:pembrolizumab",
                        "type": "Drug",
                        "label": "Keytruda (pembrolizumab)",
                        "props": {"brand": "Keytruda", "ingredient": "pembrolizumab"},
                    },
                    {
                        "id": "trial:NCT01295827",
                        "type": "Trial",
                        "label": "KEYNOTE-001",
                        "props": {"nct_id": "NCT01295827"},
                    },
                    {
                        "id": "ae:fatigue",
                        "type": "AdverseEventConcept",
                        "label": "Fatigue",
                        "props": {
                            "count_in_sample": 3,
                            "disclaimer": "FAERS voluntary reports — not rates, not causation.",
                        },
                    },
                ],
                "edges": [
                    {
                        "id": "edge:studied:pembrolizumab:NCT01295827",
                        "type": "studied_in",
                        "source": "drug:pembrolizumab",
                        "target": "trial:NCT01295827",
                        "evidence_urls": ["https://clinicaltrials.gov/study/NCT01295827"],
                        "sources": ["clinicaltrials_registry"],
                        "props": {"note": "Registry link — not causation."},
                    },
                    {
                        "id": "edge:reports_ae:pembrolizumab:fatigue",
                        "type": "reports_ae",
                        "source": "drug:pembrolizumab",
                        "target": "ae:fatigue",
                        "evidence_urls": ["https://api.fda.gov/drug/event.json?search=FIXTURE"],
                        "sources": ["openfda_faers"],
                        "props": {"count_in_sample": 3},
                    },
                ],
                "summary": "Use retrieved edges only; FAERS values are reports not rates.",
            },
        }

    monkeypatch.setattr(extract_mod, "call_gemini_extract_json", fake_call)
    result = extract_from_sources(**_base_kwargs(), use_gemini=True)
    assert result["gemini"]["used"] is True
    assert result["gemini"]["mode"] == "gemini_structure"
    assert any(e["type"] == "studied_in" for e in result["edges"])
    assert any(e["type"] == "reports_ae" for e in result["edges"])
    assert "not causation" in (result.get("disclaimer") or "").lower()


def test_gemini_invented_nct_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    def fake_call(_prompt: str):
        return {
            "ok": True,
            "used": True,
            "model": "gemini-3.5-flash",
            "data": {
                "nodes": [
                    {
                        "id": "trial:NCT99999999",
                        "type": "Trial",
                        "label": "Invented",
                        "props": {"nct_id": "NCT99999999"},
                    }
                ],
                "edges": [
                    {
                        "id": "edge:studied:x:NCT99999999",
                        "type": "studied_in",
                        "source": "drug:pembrolizumab",
                        "target": "trial:NCT99999999",
                        "evidence_urls": ["https://clinicaltrials.gov/study/NCT99999999"],
                        "sources": ["clinicaltrials_registry"],
                    }
                ],
            },
        }

    monkeypatch.setattr(extract_mod, "call_gemini_extract_json", fake_call)
    result = extract_from_sources(**_base_kwargs(), use_gemini=True)
    assert result["gemini"]["mode"] == "rules_fallback"
    assert result["gemini"]["gemini_structure_error"] == "invented_ids_refused"
    assert "invented_nct:NCT99999999" in (result["gemini"].get("invented") or [])
    # Rules fallback still has real fixture NCTs only
    ncts = {n["props"].get("nct_id") for n in result["nodes"] if n["type"] == "Trial"}
    assert "NCT99999999" not in ncts
    assert "NCT01295827" in ncts


def test_gemini_api_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    def fake_call(_prompt: str):
        return {"ok": False, "used": False, "error": "TimeoutError: boom", "model": "gemini-3.5-flash"}

    monkeypatch.setattr(extract_mod, "call_gemini_extract_json", fake_call)
    result = extract_from_sources(**_base_kwargs(), use_gemini=True)
    assert result["gemini"]["mode"] == "rules_fallback"
    assert "TimeoutError" in (result["gemini"].get("gemini_structure_error") or "")
    assert any(e["type"] == "reports_ae" for e in result["edges"])


def test_find_invented_ids_detects_bad_count():
    digest = _source_digest(**_base_kwargs())
    allowed = _allowed_ids(digest)
    problems = _find_invented_ids(
        {"nodes": [{"id": "ae:x", "props": {"count_in_sample": 99999}}]},
        allowed,
    )
    assert any(p.startswith("invented_count:") for p in problems)


def test_source_digest_has_no_abstract_field():
    digest = _source_digest(**_base_kwargs())
    blob = json.dumps(digest)
    assert "abstract" not in blob.lower()
    assert "NCT01295827" in blob
