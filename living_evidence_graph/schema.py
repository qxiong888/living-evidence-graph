"""Locked entity / edge schema for the living evidence graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

EntityType = Literal[
    "Drug",
    "Condition",
    "Gene",
    "Trial",
    "Publication",
    "AdverseEventConcept",
    "SourceDoc",
]

# Core motif for LLM retrieval: Drug ↔ Gene ↔ Condition (Open Targets + ChEMBL spine).
# Corroboration layers: ClinicalTrials, PubMed, openFDA, DailyMed, Europe PMC.
EdgeType = Literal[
    "drug_targets_gene",
    "gene_associated_with_disease",
    "drug_indicated_for_disease",
    "treats_indication",  # label/registry corroboration of indication (not causation)
    "studied_in",
    "reports_ae",
    "warns_ae",  # FDA SPL / DailyMed warning text (not causation, not rates)
    "supports",
    "contradicts",
    "cites",
]

# First-class triangle edge types (surface prominently in demo).
TRIANGLE_EDGE_TYPES: tuple[str, ...] = (
    "drug_targets_gene",
    "gene_associated_with_disease",
    "drug_indicated_for_disease",
)

SOURCE_TIERS: dict[str, float] = {
    "dailymed_label": 0.95,  # FDA SPL labeled indication / boxed warning text (DailyMed/NLM)
    "clinicaltrials_registry": 0.9,
    "pubmed_peer_reviewed": 0.8,
    "europepmc": 0.75,  # literature metadata + retraction/erratum/correction signals
    "opentargets_kb": 0.7,  # structured drug–target–disease KB
    "chembl": 0.65,  # molecule / mechanism links
    "openfda_faers": 0.55,  # reports only — not rates
    "preprint": 0.4,
}

SOURCE_FAMILY: dict[str, str] = {
    "dailymed_label": "dailymed",
    "clinicaltrials_registry": "clinicaltrials",
    "pubmed_peer_reviewed": "pubmed",
    "europepmc": "europepmc",
    "opentargets_kb": "opentargets",
    "chembl": "chembl",
    "openfda_faers": "openfda",
    "preprint": "preprint",
}

# Seven active public source families used in the Keytruda / NSCLC demo.
ACTIVE_SOURCE_FAMILIES: tuple[str, ...] = (
    "clinicaltrials",
    "pubmed",
    "openfda",
    "dailymed",
    "europepmc",
    "opentargets",
    "chembl",
)


class Node(TypedDict, total=False):
    id: str
    type: EntityType
    label: str
    props: dict[str, Any]


class Edge(TypedDict, total=False):
    id: str
    type: EdgeType
    source: str
    target: str
    evidence_urls: list[str]
    sources: list[str]
    first_seen: str
    last_seen: str
    trust_score: float
    trust_breakdown: dict[str, Any]
    retracted: bool
    age_days: float
    props: dict[str, Any]


class GraphDoc(TypedDict, total=False):
    goal: str
    nodes: list[Node]
    edges: list[Edge]
    meta: dict[str, Any]
