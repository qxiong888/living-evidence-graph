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
    "private_library": 0.5,  # personal/enterprise folder docs (file-path provenance only)
    "personal": 0.5,
    "enterprise": 0.5,
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
    "private_library": "private_library",
    "personal": "private_library",
    "enterprise": "private_library",
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



# Host / path fragments → SOURCE_TIERS tags. Map only from real evidence URLs.
_URL_SOURCE_RULES: tuple[tuple[str, str], ...] = (
    ("opentargets.org", "opentargets_kb"),
    ("ebi.ac.uk/chembl", "chembl"),
    ("/chembl/", "chembl"),
    ("clinicaltrials.gov", "clinicaltrials_registry"),
    ("pubmed.ncbi.nlm.nih.gov", "pubmed_peer_reviewed"),
    ("europepmc.org", "europepmc"),
    ("dailymed", "dailymed_label"),
    ("api.fda.gov", "openfda_faers"),
    ("openfda", "openfda_faers"),
    ("fda.gov", "openfda_faers"),
)


def source_tag_from_url(url: str) -> str | None:
    """Map one evidence URL to a SOURCE_TIERS tag. None if unrecognized."""
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return None
    for needle, tag in _URL_SOURCE_RULES:
        if needle in u and tag in SOURCE_TIERS:
            return tag
    return None


def sources_from_evidence_urls(urls: list[str] | None) -> list[str]:
    """Deduped SOURCE_TIERS tags inferred from real evidence URLs only."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls or []:
        tag = source_tag_from_url(str(raw))
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _short_host(url: str) -> str | None:
    raw = (url or "").strip()
    if "://" not in raw:
        return None
    host = raw.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
    host = host.lower().removeprefix("www.")
    return host or None


def display_source_labels(
    sources: list[str] | None,
    evidence_urls: list[str] | None = None,
) -> list[str]:
    """Sources-column labels. Prefer stored tags; else short host from URLs."""
    tags = [str(s) for s in (sources or []) if s]
    if tags:
        # preserve order, drop dupes
        seen: set[str] = set()
        out: list[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
    labels: list[str] = []
    seen_l: set[str] = set()
    for raw in evidence_urls or []:
        tag = source_tag_from_url(str(raw))
        label = SOURCE_FAMILY.get(tag, tag) if tag else _short_host(str(raw))
        if label and label not in seen_l:
            seen_l.add(label)
            labels.append(label)
    return labels


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
