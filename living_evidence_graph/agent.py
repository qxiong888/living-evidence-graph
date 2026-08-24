"""Google ADK root agent — Living Evidence Graph (Taskmaster).

Tools fetch public sources, extract edges, upsert the graph, and refresh trust.
The model must not invent NCT IDs, PMIDs, setids, ChEMBL IDs, or FDA counts.

LLM path: retrieval-only over the graph — do not dump PubMed abstracts or
non-OA Europe PMC full text into training corpora.
"""

from __future__ import annotations

import json
import re
from typing import Any

from living_evidence_graph.config import (
    DEMO_CONDITION,
    DEMO_DRUG_BRAND,
    DEMO_DRUG_INGREDIENT,
    DEMO_GOAL,
    GEMINI_MODEL,
    HTTP_TIMEOUT,
)
from living_evidence_graph.credibility import recompute_edges
from living_evidence_graph.extract import extract_from_sources
from living_evidence_graph.graph_store import load_graph, recompute_trust, upsert_graph
from living_evidence_graph.tools.fetch_chembl import fetch_chembl
from living_evidence_graph.tools.fetch_clinicaltrials import fetch_clinicaltrials
from living_evidence_graph.tools.fetch_dailymed import fetch_dailymed
from living_evidence_graph.tools.fetch_europepmc import fetch_europepmc_status
from living_evidence_graph.tools.fetch_openfda import fetch_openfda_events
from living_evidence_graph.tools.fetch_opentargets import fetch_opentargets
from living_evidence_graph.tools.fetch_pubmed import fetch_pubmed

try:
    from google.adk.agents import Agent
except ImportError:  # local demo / tests can run without the wheel
    Agent = None  # type: ignore


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "default"


def ingest_goal(goal: str) -> dict[str, Any]:
    """Parse a free-text goal into drug/condition slots for the cancer demo vertical.

    Args:
        goal: e.g. "pembrolizumab / Keytruda NSCLC solid tumor evidence graph"

    Returns:
        Structured goal fields. Does not invent registry IDs.
    """
    g = (goal or DEMO_GOAL).strip()
    brand = DEMO_DRUG_BRAND
    ingredient = DEMO_DRUG_INGREDIENT
    condition = DEMO_CONDITION
    low = g.lower()
    if "keytruda" in low or "pembrolizumab" in low:
        brand, ingredient = "Keytruda", "pembrolizumab"
    if "nsclc" in low or "non-small" in low or "lung" in low:
        condition = "non-small cell lung cancer"
    elif "solid tumor" in low or "solid tumours" in low:
        condition = "solid tumor"
    elif "melanoma" in low:
        condition = "melanoma"
    return {
        "status": "success",
        "goal": g,
        "goal_slug": _slug(f"{ingredient}_{condition}"),
        "drug_brand": brand,
        "drug_ingredient": ingredient,
        "condition": condition,
        "reminder": (
            "Public data only. No PHI. No causation claims. "
            "openFDA FAERS = voluntary reports, not rates. "
            "No FDA/NLM endorsement. LLM = retrieval-only over the graph."
        ),
    }


def _source_mode(payload: dict[str, Any], nonempty_keys: list[str]) -> str:
    """Classify a source payload as live | skipped | error."""
    if not payload.get("ok"):
        return "error"
    for k in nonempty_keys:
        if payload.get(k):
            return "live"
    if payload.get("empty"):
        return "skipped"
    return "live"


def fetch_sources(
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
) -> dict[str, Any]:
    """Call all seven public APIs with timeouts. On failure, structured error (no invented IDs)."""
    t = HTTP_TIMEOUT
    q_ct = f"{drug_ingredient} OR {drug_brand} {condition}"
    q_pm = f"{drug_ingredient} AND {condition}"
    ct = fetch_clinicaltrials(q_ct, page_size=5, timeout=t)
    pm = fetch_pubmed(q_pm, retmax=5, timeout=t)
    fda = fetch_openfda_events(drug_brand, limit=5, timeout=t)
    if not fda.get("ok") or fda.get("empty"):
        fda_ing = fetch_openfda_events(drug_ingredient, limit=5, timeout=t)
        if fda_ing.get("ok") and not fda_ing.get("empty"):
            fda = fda_ing

    dm = fetch_dailymed(drug_brand, pagesize=5, timeout=t)
    if not dm.get("ok") or dm.get("empty"):
        dm_ing = fetch_dailymed(drug_ingredient, pagesize=5, timeout=t)
        if dm_ing.get("ok") and not dm_ing.get("empty"):
            dm = dm_ing

    pmids = [p.get("pmid") for p in (pm.get("publications") or []) if p.get("pmid")]
    epmc = fetch_europepmc_status(pmids, timeout=t)

    ot = fetch_opentargets(drug_ingredient, condition_hint=condition, timeout=t)
    ch = fetch_chembl(drug_ingredient, timeout=t)

    source_mode = {
        "clinicaltrials": _source_mode(ct, ["studies"]),
        "pubmed": _source_mode(pm, ["publications"]),
        "openfda": _source_mode(fda, ["reactions"]),
        "dailymed": _source_mode(dm, ["labels", "preferred"]),
        "europepmc": _source_mode(epmc, ["publications"]),
        "opentargets": _source_mode(ot, ["targets", "indications", "gene_disease"]),
        "chembl": _source_mode(ch, ["mechanisms", "molecule_chembl_id"]),
    }

    return {
        "status": "success",
        "clinicaltrials": ct,
        "pubmed": pm,
        "openfda": fda,
        "dailymed": dm,
        "europepmc": epmc,
        "opentargets": ot,
        "chembl": ch,
        "source_mode": source_mode,
        "disclaimer": (
            "openFDA values are report counts from the API response, not incidence rates. "
            "No causation. No FDA/NLM endorsement. Prefer APIs (this pipeline) over scraping."
        ),
    }


def extract_edges(
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    sources_json: str,
) -> dict[str, Any]:
    """Turn source JSON into candidate nodes/edges. No invented IDs."""
    try:
        payload = json.loads(sources_json) if isinstance(sources_json, str) else sources_json
    except json.JSONDecodeError as e:
        return {"status": "error", "error": str(e)}
    result = extract_from_sources(
        drug_brand=drug_brand,
        drug_ingredient=drug_ingredient,
        condition=condition,
        clinicaltrials=payload.get("clinicaltrials") or {},
        pubmed=payload.get("pubmed") or {},
        openfda=payload.get("openfda") or {},
        dailymed=payload.get("dailymed") or {},
        europepmc=payload.get("europepmc") or {},
        opentargets=payload.get("opentargets") or {},
        chembl=payload.get("chembl") or {},
        use_gemini=True,
    )
    return {"status": "success", **result}


def upsert_graph_tool(
    goal: str,
    goal_slug: str,
    nodes_json: str,
    edges_json: str,
) -> dict[str, Any]:
    """Persist merged graph to out/graph/ (and Firestore if enabled)."""
    try:
        nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
        edges = json.loads(edges_json) if isinstance(edges_json, str) else edges_json
    except json.JSONDecodeError as e:
        return {"status": "error", "error": str(e)}
    doc = upsert_graph(
        goal=goal,
        nodes=nodes,
        edges=edges,
        goal_slug=goal_slug or "default",
    )
    return {
        "status": "success",
        "goal_slug": goal_slug,
        "node_count": len(doc.get("nodes") or []),
        "edge_count": len(doc.get("edges") or []),
        "path": (doc.get("meta") or {}).get("path"),
    }


def recompute_trust_tool(goal_slug: str) -> dict[str, Any]:
    """Re-score all edges with the locked credibility formula."""
    doc = recompute_trust(goal_slug or "default")
    return {
        "status": "success",
        "goal_slug": goal_slug,
        "edge_count": len(doc.get("edges") or []),
        "sample_scores": [
            {"id": e.get("id"), "trust_score": e.get("trust_score")}
            for e in (doc.get("edges") or [])[:5]
        ],
    }


def daily_refresh(goal: str = DEMO_GOAL) -> dict[str, Any]:
    """End-to-end refresh for Cloud Scheduler: ingest → fetch → extract → upsert → trust."""
    g = ingest_goal(goal)
    sources = fetch_sources(g["drug_brand"], g["drug_ingredient"], g["condition"])
    extracted = extract_from_sources(
        drug_brand=g["drug_brand"],
        drug_ingredient=g["drug_ingredient"],
        condition=g["condition"],
        clinicaltrials=sources.get("clinicaltrials") or {},
        pubmed=sources.get("pubmed") or {},
        openfda=sources.get("openfda") or {},
        dailymed=sources.get("dailymed") or {},
        europepmc=sources.get("europepmc") or {},
        opentargets=sources.get("opentargets") or {},
        chembl=sources.get("chembl") or {},
        use_gemini=True,
    )
    edges = recompute_edges(extracted.get("edges") or [])
    source_mode = sources.get("source_mode") or {
        k: ("live" if (sources.get(k) or {}).get("ok") else "error")
        for k in (
            "clinicaltrials",
            "pubmed",
            "openfda",
            "dailymed",
            "europepmc",
            "opentargets",
            "chembl",
        )
    }
    source_ok = {k: source_mode.get(k) == "live" for k in source_mode}
    doc = upsert_graph(
        goal=g["goal"],
        nodes=extracted.get("nodes") or [],
        edges=edges,
        goal_slug=g["goal_slug"],
        meta={
            "refresh": "daily",
            "source_ok": source_ok,
            "source_mode": source_mode,
            "gemini": extracted.get("gemini"),
            "attribution_block": extracted.get("attribution_block"),
        },
    )
    return {
        "status": "success",
        "goal": g["goal"],
        "goal_slug": g["goal_slug"],
        "node_count": len(doc.get("nodes") or []),
        "edge_count": len(doc.get("edges") or []),
        "path": (doc.get("meta") or {}).get("path"),
        "source_mode": source_mode,
        "sources": {
            **{f"{k}_ok": source_ok[k] for k in source_ok},
            "source_mode": source_mode,
            "openfda_total_reports": (sources.get("openfda") or {}).get("total_reports"),
        },
        "graph": doc,
    }


INSTRUCTION = """You are the Living Evidence Graph agent (All Things Agentic / Taskmaster).

Default vertical: pembrolizumab / Keytruda for NSCLC / solid tumor — PUBLIC evidence only.

Core graph motif for LLM retrieval (spine):
  drug_targets_gene · gene_associated_with_disease · drug_indicated_for_disease
  (Open Targets + ChEMBL)
Corroboration layers: ClinicalTrials.gov, PubMed, openFDA, DailyMed, Europe PMC.

Workflow:
1) ingest_goal
2) fetch_sources
3) extract_edges (pass sources as JSON string)
4) upsert_graph_tool
5) recompute_trust_tool
Or call daily_refresh for a full pass.

Rules:
- Never invent NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts — only use tool outputs.
- openFDA FAERS = voluntary reports, not rates, not causation, not a safety signal.
- No PHI. No clinical advice. No FDA/NLM endorsement.
- LLM path: retrieval-only over the graph; do not dump PubMed abstracts or non-OA full text into training.
- Prefer APIs (these tools) — never scrape HTML as a primary source.
- Prefer English for contest judges.
"""

if Agent is not None:
    root_agent = Agent(
        name="living_evidence_graph",
        model=GEMINI_MODEL,
        description=(
            "Background Taskmaster: builds a text-first living evidence graph "
            "with an Open Targets/ChEMBL drug–target–disease spine and "
            "ClinicalTrials.gov, PubMed, openFDA, DailyMed, Europe PMC corroboration "
            "(default: pembrolizumab / Keytruda)."
        ),
        instruction=INSTRUCTION,
        tools=[
            ingest_goal,
            fetch_sources,
            extract_edges,
            upsert_graph_tool,
            recompute_trust_tool,
            daily_refresh,
        ],
    )
else:
    root_agent = None
