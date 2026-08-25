"""Minimal tests for the locked credibility formula."""

from __future__ import annotations

import math

from living_evidence_graph.credibility import (
    WEIGHT_CONSISTENCY,
    WEIGHT_CORROBORATION,
    WEIGHT_RECENCY,
    WEIGHT_SOURCE,
    corroboration,
    recompute_edges,
    recency,
    score_edge,
    source_tier,
)
from living_evidence_graph.schema import (
    ACTIVE_SOURCE_FAMILIES,
    SOURCE_FAMILY,
    SOURCE_TIERS,
    TRIANGLE_EDGE_TYPES,
)


def test_source_tier_registry():
    assert source_tier(["clinicaltrials_registry"]) == 0.9
    assert source_tier(["pubmed_peer_reviewed"]) == 0.8
    assert source_tier(["openfda_faers"]) == 0.55
    assert source_tier(["preprint"]) == 0.4
    assert source_tier(["clinicaltrials_registry", "openfda_faers"]) == 0.9


def test_source_tier_new_families():
    assert SOURCE_TIERS["dailymed_label"] == 0.95
    assert SOURCE_TIERS["europepmc"] == 0.75
    assert SOURCE_TIERS["opentargets_kb"] == 0.7
    assert SOURCE_TIERS["chembl"] == 0.65
    assert source_tier(["dailymed_label"]) == 0.95
    assert source_tier(["opentargets_kb", "chembl"]) == 0.7
    assert source_tier(["chembl"]) == 0.65
    assert source_tier(["europepmc"]) == 0.75
    assert SOURCE_FAMILY["dailymed_label"] == "dailymed"
    assert SOURCE_FAMILY["opentargets_kb"] == "opentargets"
    assert SOURCE_FAMILY["chembl"] == "chembl"
    assert SOURCE_FAMILY["europepmc"] == "europepmc"
    assert len(ACTIVE_SOURCE_FAMILIES) == 7
    assert set(ACTIVE_SOURCE_FAMILIES) == {
        "clinicaltrials",
        "pubmed",
        "openfda",
        "dailymed",
        "europepmc",
        "opentargets",
        "chembl",
    }


def test_triangle_edge_types_locked():
    assert TRIANGLE_EDGE_TYPES == (
        "drug_targets_gene",
        "gene_associated_with_disease",
        "drug_indicated_for_disease",
    )


def test_corroboration_families():
    assert corroboration([]) == 0.0
    assert corroboration(["pubmed_peer_reviewed"]) == 1 / 3
    assert math.isclose(
        corroboration(
            ["clinicaltrials_registry", "pubmed_peer_reviewed", "openfda_faers"]
        ),
        1.0,
    )
    # New families also count toward corroboration
    assert math.isclose(
        corroboration(["opentargets_kb", "chembl", "dailymed_label"]),
        1.0,
    )


def test_recency_decay():
    assert math.isclose(recency(0), 1.0)
    assert recency(365) < 0.4
    assert recency(None) == 0.5


def test_score_edge_example():
    # Documented example: CT + PubMed, 120 days, no contradict, not retracted
    out = score_edge(
        sources=["clinicaltrials_registry", "pubmed_peer_reviewed"],
        age_days=120,
        has_contradict=False,
        retracted=False,
    )
    st = 0.9
    corr = 2 / 3
    rec = math.exp(-120 / 365)
    cons = 1.0
    raw = (
        WEIGHT_SOURCE * st
        + WEIGHT_CORROBORATION * corr
        + WEIGHT_RECENCY * rec
        + WEIGHT_CONSISTENCY * cons
    )
    assert math.isclose(out["trust_score"], max(0.0, min(1.0, raw)), abs_tol=1e-3)
    assert out["trust_breakdown"]["retraction_penalty"] == 0.0


def test_score_edge_dailymed_high_tier():
    out = score_edge(sources=["dailymed_label"], age_days=0, has_contradict=False)
    assert out["trust_breakdown"]["source_tier"] == 0.95
    assert out["trust_score"] > 0.5


def test_retraction_and_contradict_penalties():
    base = score_edge(sources=["pubmed_peer_reviewed"], age_days=0, has_contradict=False)
    bad = score_edge(
        sources=["pubmed_peer_reviewed", "europepmc"],
        age_days=0,
        has_contradict=True,
        retracted=True,
    )
    assert bad["trust_score"] < base["trust_score"]
    assert bad["trust_breakdown"]["consistency"] == 0.4
    assert bad["trust_breakdown"]["retraction_penalty"] == 0.5


def test_recompute_edges_sets_scores():
    edges = [
        {
            "id": "e1",
            "type": "drug_targets_gene",
            "source": "a",
            "target": "b",
            "sources": ["opentargets_kb", "chembl"],
            "age_days": 10,
        }
    ]
    out = recompute_edges(edges)
    assert "trust_score" in out[0]
    assert 0.0 <= out[0]["trust_score"] <= 1.0
    assert out[0]["trust_breakdown"]["source_tier"] == 0.7



def test_url_maps_to_source_tier_tags():
    from living_evidence_graph.schema import (
        display_source_labels,
        source_tag_from_url,
        sources_from_evidence_urls,
    )

    assert source_tag_from_url("https://platform.opentargets.org/drug/CHEMBL3137343") == "opentargets_kb"
    assert source_tag_from_url(
        "https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL3137343/"
    ) == "chembl"
    assert source_tag_from_url("https://clinicaltrials.gov/study/NCT03631784") == "clinicaltrials_registry"
    assert source_tag_from_url("https://pubmed.ncbi.nlm.nih.gov/42628840/") == "pubmed_peer_reviewed"
    assert source_tag_from_url("https://europepmc.org/article/MED/42628840") == "europepmc"
    assert source_tag_from_url(
        "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=9333c79b-d487-4538-a9f0-71b91a02b287"
    ) == "dailymed_label"
    assert source_tag_from_url(
        "https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct%3A%22Keytruda%22&limit=5"
    ) == "openfda_faers"
    assert source_tag_from_url("NCT03631784") is None
    assert source_tag_from_url("PMID:42628840") is None

    tags = sources_from_evidence_urls(
        [
            "https://platform.opentargets.org/drug/CHEMBL3137343",
            "https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL3137343/",
            "https://platform.opentargets.org/drug/CHEMBL3137343",
        ]
    )
    assert tags == ["opentargets_kb", "chembl"]

    # Fallback host labels only when sources are empty.
    hosts = display_source_labels(
        [],
        ["https://platform.opentargets.org/drug/CHEMBL3137343"],
    )
    assert hosts == ["opentargets"]
    assert display_source_labels(["opentargets_kb"], ["https://example.org"]) == ["opentargets_kb"]


def test_recompute_backfills_sources_from_urls():
    edges = [
        {
            "id": "e1",
            "type": "drug_targets_gene",
            "source": "a",
            "target": "b",
            "sources": [],
            "evidence_urls": [
                "https://platform.opentargets.org/drug/CHEMBL3137343",
                "https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL3137343/",
            ],
            "age_days": 90.0,
        }
    ]
    out = recompute_edges(edges)
    assert out[0]["sources"] == ["opentargets_kb", "chembl"]
    assert out[0]["trust_score"] != 0.4113
    assert out[0]["trust_breakdown"]["source_tier"] == 0.7
    assert out[0]["trust_breakdown"]["corroboration"] > 0
