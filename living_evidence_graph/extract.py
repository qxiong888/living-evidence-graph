"""Extract candidate graph triples from source payloads.

When GEMINI_API_KEY is set, prefer Gemini 3.5 Flash to structure/normalize
triples from fetched public records (strict JSON schema). Falls back to
deterministic rule-based extraction when no key or Gemini fails.

Never invents NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts —
only maps fields already present in tool outputs. Invented IDs reject the
Gemini result and trigger the rules fallback.

Core motif (Open Targets + ChEMBL spine):
  drug_targets_gene · gene_associated_with_disease · drug_indicated_for_disease
Corroboration layers: ClinicalTrials, PubMed, openFDA, DailyMed, Europe PMC.

Elena red lines: public data only; no causation; FAERS = reports not rates;
LLM path is retrieval-oriented over the graph (no abstract/full-text dumps).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from living_evidence_graph.config import GEMINI_MODEL, gemini_api_key, has_gemini_key
from living_evidence_graph.schema import Edge, Node


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:80]


def _extract_rules(
    *,
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    clinicaltrials: dict[str, Any],
    pubmed: dict[str, Any],
    openfda: dict[str, Any],
    dailymed: dict[str, Any] | None = None,
    europepmc: dict[str, Any] | None = None,
    opentargets: dict[str, Any] | None = None,
    chembl: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic rule-based extract from public source payloads (no LLM)."""
    dailymed = dailymed or {}
    europepmc = europepmc or {}
    opentargets = opentargets or {}
    chembl = chembl or {}

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    now = _now_iso()

    drug_id = f"drug:{_slug(drug_ingredient or drug_brand)}"
    cond_id = f"condition:{_slug(condition)}"
    nodes[drug_id] = {
        "id": drug_id,
        "type": "Drug",
        "label": f"{drug_brand} ({drug_ingredient})",
        "props": {"brand": drug_brand, "ingredient": drug_ingredient},
    }
    nodes[cond_id] = {
        "id": cond_id,
        "type": "Condition",
        "label": condition,
        "props": {},
    }

    def _ensure_gene(symbol: str | None, ensembl: str | None = None, **extra: Any) -> str | None:
        if ensembl and str(ensembl).startswith("ENSG"):
            gid = f"gene:{ensembl}"
            label = symbol or ensembl
        elif symbol:
            gid = f"gene:{_slug(symbol)}"
            label = symbol
        else:
            return None
        prev = nodes.get(gid) or {
            "id": gid,
            "type": "Gene",
            "label": label,
            "props": {},
        }
        props = dict(prev.get("props") or {})
        if symbol:
            props["symbol"] = symbol
        if ensembl:
            props["ensembl_id"] = ensembl
        props.update({k: v for k, v in extra.items() if v is not None})
        prev["props"] = props
        prev["label"] = label
        nodes[gid] = prev
        return gid

    def _condition_node(name: str, ontology_id: str | None = None) -> str:
        if ontology_id:
            cid = f"condition:{_slug(ontology_id)}"
        else:
            cid = f"condition:{_slug(name)}"
        if cid not in nodes:
            nodes[cid] = {
                "id": cid,
                "type": "Condition",
                "label": name,
                "props": {"ontology_id": ontology_id} if ontology_id else {},
            }
        return cid

    edges.append(
        {
            "id": f"edge:treats:{_slug(drug_ingredient)}:{_slug(condition)}",
            "type": "treats_indication",
            "source": drug_id,
            "target": cond_id,
            "evidence_urls": [],
            "sources": [],
            "first_seen": now,
            "last_seen": now,
            "trust_score": 0.0,
            "trust_breakdown": {},
            "retracted": False,
            "age_days": 90.0,
            "props": {
                "note": "Public-record indication / study context — not a causation claim."
            },
        }
    )
    treats = edges[-1]

    # Open Targets spine
    if opentargets.get("ok"):
        chembl_id = opentargets.get("chembl_id")
        if chembl_id:
            nodes[drug_id]["props"] = {
                **(nodes[drug_id].get("props") or {}),
                "chembl_id": chembl_id,
            }
        ot_url = opentargets.get("url")
        for tgt in opentargets.get("targets") or []:
            gid = _ensure_gene(
                tgt.get("symbol"),
                tgt.get("ensembl_id"),
                mechanism=tgt.get("mechanism"),
                target_name=tgt.get("target_name"),
            )
            if not gid:
                continue
            evidence = [ot_url] if ot_url else []
            edges.append(
                {
                    "id": f"edge:drug_targets_gene:ot:{_slug(drug_ingredient)}:{_slug(gid)}",
                    "type": "drug_targets_gene",
                    "source": drug_id,
                    "target": gid,
                    "evidence_urls": evidence,
                    "sources": ["opentargets_kb"],
                    "first_seen": now,
                    "last_seen": now,
                    "trust_score": 0.0,
                    "trust_breakdown": {},
                    "retracted": False,
                    "age_days": 30.0,
                    "props": {
                        "mechanism": tgt.get("mechanism"),
                        "attribution": opentargets.get("attribution"),
                        "license": opentargets.get("license"),
                        "motif": "triangle_spine",
                        "note": "Structured KB association — not causation.",
                    },
                }
            )
        for ind in (opentargets.get("indications") or [])[:8]:
            cid = _condition_node(ind.get("disease_name") or "", ind.get("disease_id"))
            evidence = [ot_url] if ot_url else []
            edges.append(
                {
                    "id": (
                        f"edge:drug_indicated_for_disease:ot:"
                        f"{_slug(drug_ingredient)}:{_slug(cid)}"
                    ),
                    "type": "drug_indicated_for_disease",
                    "source": drug_id,
                    "target": cid,
                    "evidence_urls": evidence,
                    "sources": ["opentargets_kb"],
                    "first_seen": now,
                    "last_seen": now,
                    "trust_score": 0.0,
                    "trust_breakdown": {},
                    "retracted": False,
                    "age_days": 30.0,
                    "props": {
                        "disease_id": ind.get("disease_id"),
                        "attribution": opentargets.get("attribution"),
                        "license": opentargets.get("license"),
                        "motif": "triangle_spine",
                        "note": "Open Targets clinical indication row — not causation.",
                    },
                }
            )
            dname = str(ind.get("disease_name") or "").lower()
            if any(k in dname for k in ("non-small cell lung", "nsclc", "lung carcinoma")):
                if ot_url and ot_url not in treats["evidence_urls"]:
                    treats["evidence_urls"].append(ot_url)
                if "opentargets_kb" not in treats["sources"]:
                    treats["sources"].append("opentargets_kb")
        for gd in opentargets.get("gene_disease") or []:
            gid = _ensure_gene(gd.get("symbol"), gd.get("ensembl_id"))
            if not gid:
                continue
            cid = _condition_node(gd.get("disease_name") or "", gd.get("disease_id"))
            evidence = [ot_url] if ot_url else []
            edges.append(
                {
                    "id": (
                        f"edge:gene_associated_with_disease:ot:"
                        f"{_slug(gid)}:{_slug(cid)}"
                    ),
                    "type": "gene_associated_with_disease",
                    "source": gid,
                    "target": cid,
                    "evidence_urls": evidence,
                    "sources": ["opentargets_kb"],
                    "first_seen": now,
                    "last_seen": now,
                    "trust_score": 0.0,
                    "trust_breakdown": {},
                    "retracted": False,
                    "age_days": 30.0,
                    "props": {
                        "score": gd.get("score"),
                        "disease_id": gd.get("disease_id"),
                        "attribution": opentargets.get("attribution"),
                        "license": opentargets.get("license"),
                        "motif": "triangle_spine",
                        "note": "Target–disease association score — not causation.",
                    },
                }
            )

    # ChEMBL mechanism spine
    if chembl.get("ok"):
        mol_id = chembl.get("molecule_chembl_id")
        if mol_id:
            nodes[drug_id]["props"] = {
                **(nodes[drug_id].get("props") or {}),
                "chembl_id": mol_id,
                "chembl_db_version": chembl.get("chembl_db_version"),
            }
        ch_url = chembl.get("url")
        for mech in chembl.get("mechanisms") or []:
            gid = _ensure_gene(
                mech.get("gene_symbol"),
                None,
                target_chembl_id=mech.get("target_chembl_id"),
                target_name=mech.get("target_name"),
            )
            if not gid and mech.get("target_chembl_id"):
                tid = mech["target_chembl_id"]
                gid = f"gene:chembl:{tid}"
                nodes[gid] = {
                    "id": gid,
                    "type": "Gene",
                    "label": mech.get("target_name") or tid,
                    "props": {
                        "target_chembl_id": tid,
                        "gene_symbol": mech.get("gene_symbol"),
                    },
                }
            if not gid:
                continue
            evidence = [ch_url] if ch_url else []
            edges.append(
                {
                    "id": (
                        f"edge:drug_targets_gene:chembl:"
                        f"{_slug(drug_ingredient)}:{_slug(gid)}"
                    ),
                    "type": "drug_targets_gene",
                    "source": drug_id,
                    "target": gid,
                    "evidence_urls": evidence,
                    "sources": ["chembl"],
                    "first_seen": now,
                    "last_seen": now,
                    "trust_score": 0.0,
                    "trust_breakdown": {},
                    "retracted": False,
                    "age_days": 60.0,
                    "props": {
                        "action_type": mech.get("action_type"),
                        "mechanism_of_action": mech.get("mechanism_of_action"),
                        "max_phase": mech.get("max_phase"),
                        "attribution": chembl.get("attribution"),
                        "license": chembl.get("license"),
                        "citation": chembl.get("citation"),
                        "chembl_db_version": chembl.get("chembl_db_version"),
                        "sharealike_note": chembl.get("sharealike_note"),
                        "motif": "triangle_spine",
                        "note": "ChEMBL mechanism link — not causation.",
                    },
                }
            )

    # ClinicalTrials
    for st in clinicaltrials.get("studies") or []:
        nct = st.get("nct_id")
        if not nct:
            continue
        tid = f"trial:{nct}"
        url = st.get("url") or f"https://clinicaltrials.gov/study/{nct}"
        nodes[tid] = {
            "id": tid,
            "type": "Trial",
            "label": st.get("title") or nct,
            "props": {
                "nct_id": nct,
                "overall_status": st.get("overall_status"),
                "phases": st.get("phases") or [],
                "conditions": st.get("conditions") or [],
            },
        }
        src_doc = f"sourcedoc:ctgov:{nct}"
        nodes[src_doc] = {
            "id": src_doc,
            "type": "SourceDoc",
            "label": f"ClinicalTrials.gov {nct}",
            "props": {
                "url": url,
                "family": "clinicaltrials",
                "courtesy": "Courtesy of the U.S. National Library of Medicine",
            },
        }
        edges.append(
            {
                "id": f"edge:studied:{_slug(drug_ingredient)}:{nct}",
                "type": "studied_in",
                "source": drug_id,
                "target": tid,
                "evidence_urls": [url],
                "sources": ["clinicaltrials_registry"],
                "first_seen": now,
                "last_seen": now,
                "trust_score": 0.0,
                "trust_breakdown": {},
                "retracted": False,
                "age_days": 180.0,
                "props": {},
            }
        )
        if url not in treats["evidence_urls"]:
            treats["evidence_urls"].append(url)
        if "clinicaltrials_registry" not in treats["sources"]:
            treats["sources"].append("clinicaltrials_registry")

    retracted_pmids: set[str] = set()
    for pub in europepmc.get("publications") or []:
        if pub.get("retracted") and pub.get("pmid"):
            retracted_pmids.add(str(pub["pmid"]))

    for pub in pubmed.get("publications") or []:
        pmid = pub.get("pmid")
        if not pmid:
            continue
        pid = f"publication:{pmid}"
        url = pub.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        is_retracted = str(pmid) in retracted_pmids
        nodes[pid] = {
            "id": pid,
            "type": "Publication",
            "label": pub.get("title") or f"PMID {pmid}",
            "props": {
                "pmid": str(pmid),
                "pubdate": pub.get("pubdate"),
                "journal": pub.get("source"),
                "retracted": is_retracted,
                "ncbi_disclaimer": (
                    "NCBI / NLM disclaimer: content from NCBI sites is subject to "
                    "NCBI terms; this project is not endorsed by NCBI/NLM/NIH."
                ),
            },
        }
        edges.append(
            {
                "id": f"edge:supports:{pmid}:{_slug(condition)}",
                "type": "supports",
                "source": pid,
                "target": cond_id,
                "evidence_urls": [url],
                "sources": ["pubmed_peer_reviewed"],
                "first_seen": now,
                "last_seen": now,
                "trust_score": 0.0,
                "trust_breakdown": {},
                "retracted": is_retracted,
                "age_days": 200.0,
                "props": {
                    "drug_context": drug_ingredient,
                    "retrieval_only": True,
                    "note": "Graph stores citation metadata/IDs — not full abstracts for training.",
                },
            }
        )
        edges.append(
            {
                "id": f"edge:cites:{pmid}:{_slug(drug_ingredient)}",
                "type": "cites",
                "source": pid,
                "target": drug_id,
                "evidence_urls": [url],
                "sources": ["pubmed_peer_reviewed"],
                "first_seen": now,
                "last_seen": now,
                "trust_score": 0.0,
                "trust_breakdown": {},
                "retracted": is_retracted,
                "age_days": 200.0,
                "props": {"retrieval_only": True},
            }
        )
        if url not in treats["evidence_urls"]:
            treats["evidence_urls"].append(url)
        if "pubmed_peer_reviewed" not in treats["sources"]:
            treats["sources"].append("pubmed_peer_reviewed")

    for pub in europepmc.get("publications") or []:
        pmid = pub.get("pmid")
        if not pmid:
            continue
        pid = f"publication:{pmid}"
        if pid not in nodes:
            nodes[pid] = {
                "id": pid,
                "type": "Publication",
                "label": pub.get("title") or f"PMID {pmid}",
                "props": {"pmid": str(pmid)},
            }
        props = dict(nodes[pid].get("props") or {})
        props.update(
            {
                "europepmc_retracted": bool(pub.get("retracted")),
                "europepmc_erratum": bool(pub.get("erratum")),
                "europepmc_correction": bool(pub.get("correction")),
                "europepmc_url": pub.get("url"),
                "license_note": (
                    "Europe PMC: check per-article license before reuse of full text; "
                    "this graph stores status metadata only (no non-OA full text)."
                ),
            }
        )
        nodes[pid]["props"] = props
        if pub.get("retracted"):
            for e in edges:
                if e.get("source") == pid or (
                    e.get("type") in {"supports", "cites"} and str(pmid) in str(e.get("id"))
                ):
                    e["retracted"] = True
                    srcs = list(e.get("sources") or [])
                    if "europepmc" not in srcs:
                        srcs.append("europepmc")
                    e["sources"] = srcs

    if dailymed.get("ok"):
        preferred = dailymed.get("preferred") or {}
        labels = dailymed.get("labels") or []
        for lab in labels[:5]:
            setid = lab.get("setid")
            if not setid:
                continue
            sid = f"sourcedoc:dailymed:{setid}"
            nodes[sid] = {
                "id": sid,
                "type": "SourceDoc",
                "label": lab.get("title") or setid,
                "props": {
                    "setid": setid,
                    "url": lab.get("url"),
                    "family": "dailymed",
                    "attribution": dailymed.get("attribution"),
                    "courtesy": "Courtesy of the U.S. National Library of Medicine",
                },
            }
        if preferred.get("setid"):
            url = preferred.get("url")
            if url and url not in treats["evidence_urls"]:
                treats["evidence_urls"].append(url)
            if "dailymed_label" not in treats["sources"]:
                treats["sources"].append("dailymed_label")
            edges.append(
                {
                    "id": (
                        f"edge:drug_indicated_for_disease:dailymed:"
                        f"{_slug(drug_ingredient)}:{_slug(condition)}"
                    ),
                    "type": "drug_indicated_for_disease",
                    "source": drug_id,
                    "target": cond_id,
                    "evidence_urls": [url] if url else [],
                    "sources": ["dailymed_label"],
                    "first_seen": now,
                    "last_seen": now,
                    "trust_score": 0.0,
                    "trust_breakdown": {},
                    "retracted": False,
                    "age_days": 14.0,
                    "props": {
                        "setid": preferred.get("setid"),
                        "indications_snippet": (preferred.get("indications_snippet") or "")[:400]
                        or None,
                        "attribution": dailymed.get("attribution"),
                        "note": (
                            "FDA SPL label text via DailyMed — labeled indication context, "
                            "not a causation claim. No FDA/NLM endorsement."
                        ),
                        "motif": "label_corroboration",
                    },
                }
            )
            for term in preferred.get("warning_terms") or []:
                ae_id = f"ae:{_slug(term)}"
                nodes[ae_id] = {
                    "id": ae_id,
                    "type": "AdverseEventConcept",
                    "label": term,
                    "props": {
                        "from": "dailymed_label_warning",
                        "disclaimer": (
                            "Label warning concept from FDA SPL — not rates, not causation."
                        ),
                    },
                }
                edges.append(
                    {
                        "id": f"edge:warns_ae:{_slug(drug_ingredient)}:{_slug(term)}",
                        "type": "warns_ae",
                        "source": drug_id,
                        "target": ae_id,
                        "evidence_urls": [url] if url else [],
                        "sources": ["dailymed_label"],
                        "first_seen": now,
                        "last_seen": now,
                        "trust_score": 0.0,
                        "trust_breakdown": {},
                        "retracted": False,
                        "age_days": 14.0,
                        "props": {
                            "warnings_snippet": (preferred.get("warnings_snippet") or "")[:400]
                            or None,
                            "attribution": dailymed.get("attribution"),
                            "note": "Label warning text — not incidence rates, not causation.",
                        },
                    }
                )

    for rx in openfda.get("reactions") or []:
        term = rx.get("term")
        if not term:
            continue
        ae_id = f"ae:{_slug(term)}"
        nodes[ae_id] = {
            "id": ae_id,
            "type": "AdverseEventConcept",
            "label": term,
            "props": {
                "count_in_sample": rx.get("count_in_sample"),
                "disclaimer": (
                    "FAERS voluntary reports — not rates, not causation, not a safety signal."
                ),
            },
        }
        evidence = []
        if openfda.get("request_url"):
            evidence.append(str(openfda["request_url"]))
        edges.append(
            {
                "id": f"edge:reports_ae:{_slug(drug_ingredient)}:{_slug(term)}",
                "type": "reports_ae",
                "source": drug_id,
                "target": ae_id,
                "evidence_urls": evidence,
                "sources": ["openfda_faers"],
                "first_seen": now,
                "last_seen": now,
                "trust_score": 0.0,
                "trust_breakdown": {},
                "retracted": False,
                "age_days": 60.0,
                "props": {
                    "label": openfda.get("label"),
                    "total_reports_meta": openfda.get("total_reports"),
                },
            }
        )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "gemini": {"used": False, "mode": "rules"},
        "disclaimer": (
            "Public data only. No PHI. Edges are evidence links for LLM retrieval — "
            "not clinical advice, not causation, not incidence rates. "
            "No FDA/NLM/Open Targets/ChEMBL endorsement. "
            "LLM path is retrieval-only over the graph (no abstract/full-text training dumps)."
        ),
        "attribution_block": {
            "open_targets": opentargets.get("attribution"),
            "chembl": chembl.get("attribution"),
            "dailymed": dailymed.get("attribution"),
            "europepmc": europepmc.get("attribution"),
            "openfda": (
                "openFDA FAERS voluntary reports (FDA) — not rates, not causation; "
                "not an FDA endorsement."
            ),
            "nlm_courtesy": (
                "ClinicalTrials.gov, DailyMed, and PubMed: Courtesy of the "
                "U.S. National Library of Medicine. Not endorsed by NLM/NIH/NCBI."
            ),
            "ncbi": (
                "NCBI disclaimer: use of NCBI information does not imply endorsement; "
                "see https://www.ncbi.nlm.nih.gov/home/about/policies/"
            ),
        },
    }



ALLOWED_NODE_TYPES = (
    "Drug",
    "Condition",
    "Gene",
    "Trial",
    "Publication",
    "AdverseEventConcept",
    "SourceDoc",
)

ALLOWED_EDGE_TYPES = (
    "drug_targets_gene",
    "gene_associated_with_disease",
    "drug_indicated_for_disease",
    "treats_indication",
    "studied_in",
    "reports_ae",
    "warns_ae",
    "supports",
    "contradicts",
    "cites",
)

# Strict JSON Schema for Gemini structured extract (response_json_schema).
GEMINI_EXTRACT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "description": "Graph nodes. IDs and registry props must come from SOURCE DIGEST only.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": list(ALLOWED_NODE_TYPES)},
                    "label": {"type": "string"},
                    "props": {
                        "type": "object",
                        "description": "Only copy IDs/counts present in the digest.",
                    },
                },
                "required": ["id", "type", "label"],
            },
        },
        "edges": {
            "type": "array",
            "description": "Evidence edges for retrieval. No causation claims.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": list(ALLOWED_EDGE_TYPES)},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "evidence_urls": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "retracted": {"type": "boolean"},
                    "props": {"type": "object"},
                },
                "required": ["id", "type", "source", "target"],
            },
        },
        "summary": {
            "type": "string",
            "description": "2-3 sentences on retrieval-only use of this graph. No invented IDs.",
        },
    },
    "required": ["nodes", "edges"],
}

_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.I)
_PMID_RE = re.compile(r"\bPMID[:\s]*(\d+)\b", re.I)
_PMID_BARE_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|publication:)(\d{5,})")
_SETID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
_CHEMBL_RE = re.compile(r"\bCHEMBL\d+\b", re.I)
_ENSG_RE = re.compile(r"\bENSG\d+\b", re.I)


def _attribution_block(
    *,
    opentargets: dict[str, Any],
    chembl: dict[str, Any],
    dailymed: dict[str, Any],
    europepmc: dict[str, Any],
) -> dict[str, Any]:
    return {
        "open_targets": opentargets.get("attribution"),
        "chembl": chembl.get("attribution"),
        "dailymed": dailymed.get("attribution"),
        "europepmc": europepmc.get("attribution"),
        "openfda": (
            "openFDA FAERS voluntary reports (FDA) — not rates, not causation; "
            "not an FDA endorsement."
        ),
        "nlm_courtesy": (
            "ClinicalTrials.gov, DailyMed, and PubMed: Courtesy of the "
            "U.S. National Library of Medicine. Not endorsed by NLM/NIH/NCBI."
        ),
        "ncbi": (
            "NCBI disclaimer: use of NCBI information does not imply endorsement; "
            "see https://www.ncbi.nlm.nih.gov/home/about/policies/"
        ),
    }


def _disclaimer() -> str:
    return (
        "Public data only. No PHI. Edges are evidence links for LLM retrieval — "
        "not clinical advice, not causation, not incidence rates. "
        "No FDA/NLM/Open Targets/ChEMBL endorsement. "
        "LLM path is retrieval-only over the graph (no abstract/full-text training dumps)."
    )


def _source_digest(
    *,
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    clinicaltrials: dict[str, Any],
    pubmed: dict[str, Any],
    openfda: dict[str, Any],
    dailymed: dict[str, Any],
    europepmc: dict[str, Any],
    opentargets: dict[str, Any],
    chembl: dict[str, Any],
) -> dict[str, Any]:
    """Compact public-record digest for Gemini — IDs/titles/counts only, no abstracts."""
    studies = []
    for st in clinicaltrials.get("studies") or []:
        if not st.get("nct_id"):
            continue
        studies.append(
            {
                "nct_id": st.get("nct_id"),
                "title": st.get("title"),
                "overall_status": st.get("overall_status"),
                "phases": st.get("phases") or [],
                "conditions": st.get("conditions") or [],
                "url": st.get("url"),
            }
        )
    pubs = []
    for pub in pubmed.get("publications") or []:
        if not pub.get("pmid"):
            continue
        pubs.append(
            {
                "pmid": str(pub.get("pmid")),
                "title": pub.get("title"),
                "pubdate": pub.get("pubdate"),
                "journal": pub.get("source"),
                "url": pub.get("url"),
            }
        )
    epmc = []
    for pub in europepmc.get("publications") or []:
        if not pub.get("pmid"):
            continue
        epmc.append(
            {
                "pmid": str(pub.get("pmid")),
                "retracted": bool(pub.get("retracted")),
                "erratum": bool(pub.get("erratum")),
                "correction": bool(pub.get("correction")),
                "url": pub.get("url"),
            }
        )
    preferred = dailymed.get("preferred") or {}
    labels = []
    for lab in (dailymed.get("labels") or [])[:5]:
        if not lab.get("setid"):
            continue
        labels.append(
            {"setid": lab.get("setid"), "title": lab.get("title"), "url": lab.get("url")}
        )
    reactions = []
    for rx in openfda.get("reactions") or []:
        if not rx.get("term"):
            continue
        reactions.append(
            {"term": rx.get("term"), "count_in_sample": rx.get("count_in_sample")}
        )
    targets = []
    for tgt in opentargets.get("targets") or []:
        targets.append(
            {
                "symbol": tgt.get("symbol"),
                "ensembl_id": tgt.get("ensembl_id"),
                "mechanism": tgt.get("mechanism"),
                "target_name": tgt.get("target_name"),
            }
        )
    indications = []
    for ind in (opentargets.get("indications") or [])[:8]:
        indications.append(
            {
                "disease_name": ind.get("disease_name"),
                "disease_id": ind.get("disease_id"),
            }
        )
    gene_disease = []
    for gd in opentargets.get("gene_disease") or []:
        gene_disease.append(
            {
                "symbol": gd.get("symbol"),
                "ensembl_id": gd.get("ensembl_id"),
                "disease_name": gd.get("disease_name"),
                "disease_id": gd.get("disease_id"),
                "score": gd.get("score"),
            }
        )
    mechanisms = []
    for mech in chembl.get("mechanisms") or []:
        mechanisms.append(
            {
                "gene_symbol": mech.get("gene_symbol"),
                "target_chembl_id": mech.get("target_chembl_id"),
                "target_name": mech.get("target_name"),
                "action_type": mech.get("action_type"),
                "mechanism_of_action": mech.get("mechanism_of_action"),
                "max_phase": mech.get("max_phase"),
            }
        )
    return {
        "drug_brand": drug_brand,
        "drug_ingredient": drug_ingredient,
        "condition": condition,
        "clinicaltrials": {"ok": bool(clinicaltrials.get("ok")), "studies": studies},
        "pubmed": {"ok": bool(pubmed.get("ok")), "publications": pubs},
        "openfda": {
            "ok": bool(openfda.get("ok")),
            "total_reports": openfda.get("total_reports"),
            "reactions": reactions,
            "request_url": openfda.get("request_url"),
            "label": openfda.get("label"),
        },
        "dailymed": {
            "ok": bool(dailymed.get("ok")),
            "preferred": {
                "setid": preferred.get("setid"),
                "url": preferred.get("url"),
                "warning_terms": preferred.get("warning_terms") or [],
                "indications_snippet": (preferred.get("indications_snippet") or "")[:400],
                "warnings_snippet": (preferred.get("warnings_snippet") or "")[:400],
            },
            "labels": labels,
            "attribution": dailymed.get("attribution"),
        },
        "europepmc": {"ok": bool(europepmc.get("ok")), "publications": epmc},
        "opentargets": {
            "ok": bool(opentargets.get("ok")),
            "chembl_id": opentargets.get("chembl_id"),
            "url": opentargets.get("url"),
            "targets": targets,
            "indications": indications,
            "gene_disease": gene_disease,
            "attribution": opentargets.get("attribution"),
            "license": opentargets.get("license"),
        },
        "chembl": {
            "ok": bool(chembl.get("ok")),
            "molecule_chembl_id": chembl.get("molecule_chembl_id"),
            "chembl_db_version": chembl.get("chembl_db_version"),
            "url": chembl.get("url"),
            "mechanisms": mechanisms,
            "attribution": chembl.get("attribution"),
            "license": chembl.get("license"),
        },
        "rules": [
            "PUBLIC data only — structure what is in this digest.",
            "Do NOT invent NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts.",
            "openFDA FAERS counts are voluntary REPORTS — not rates, not causation.",
            "No causation claims; edges are retrieval-oriented evidence links.",
            "Prefer triangle spine: drug_targets_gene, gene_associated_with_disease, "
            "drug_indicated_for_disease when Open Targets/ChEMBL rows exist.",
            "Use reports_ae for FAERS; warns_ae for DailyMed label warnings.",
            "Copy evidence_urls only from URLs present in this digest.",
        ],
    }


def _allowed_ids(digest: dict[str, Any]) -> dict[str, set[str]]:
    ncts: set[str] = set()
    pmids: set[str] = set()
    setids: set[str] = set()
    chembls: set[str] = set()
    ensembls: set[str] = set()
    urls: set[str] = set()
    counts: set[str] = set()  # stringified allowed numeric counts

    for st in (digest.get("clinicaltrials") or {}).get("studies") or []:
        if st.get("nct_id"):
            ncts.add(str(st["nct_id"]).upper())
        if st.get("url"):
            urls.add(str(st["url"]))
    for pub in (digest.get("pubmed") or {}).get("publications") or []:
        if pub.get("pmid"):
            pmids.add(str(pub["pmid"]))
        if pub.get("url"):
            urls.add(str(pub["url"]))
    for pub in (digest.get("europepmc") or {}).get("publications") or []:
        if pub.get("pmid"):
            pmids.add(str(pub["pmid"]))
        if pub.get("url"):
            urls.add(str(pub["url"]))
    dm = digest.get("dailymed") or {}
    pref = dm.get("preferred") or {}
    if pref.get("setid"):
        setids.add(str(pref["setid"]).lower())
    if pref.get("url"):
        urls.add(str(pref["url"]))
    for lab in dm.get("labels") or []:
        if lab.get("setid"):
            setids.add(str(lab["setid"]).lower())
        if lab.get("url"):
            urls.add(str(lab["url"]))
    ot = digest.get("opentargets") or {}
    if ot.get("chembl_id"):
        chembls.add(str(ot["chembl_id"]).upper())
    if ot.get("url"):
        urls.add(str(ot["url"]))
    for tgt in ot.get("targets") or []:
        if tgt.get("ensembl_id"):
            ensembls.add(str(tgt["ensembl_id"]).upper())
    for gd in ot.get("gene_disease") or []:
        if gd.get("ensembl_id"):
            ensembls.add(str(gd["ensembl_id"]).upper())
    ch = digest.get("chembl") or {}
    if ch.get("molecule_chembl_id"):
        chembls.add(str(ch["molecule_chembl_id"]).upper())
    if ch.get("url"):
        urls.add(str(ch["url"]))
    for mech in ch.get("mechanisms") or []:
        if mech.get("target_chembl_id"):
            chembls.add(str(mech["target_chembl_id"]).upper())
    fda = digest.get("openfda") or {}
    if fda.get("request_url"):
        urls.add(str(fda["request_url"]))
    if fda.get("total_reports") is not None:
        counts.add(str(fda["total_reports"]))
    for rx in fda.get("reactions") or []:
        if rx.get("count_in_sample") is not None:
            counts.add(str(rx["count_in_sample"]))

    return {
        "nct": ncts,
        "pmid": pmids,
        "setid": setids,
        "chembl": chembls,
        "ensembl": ensembls,
        "url": urls,
        "count": counts,
    }


def _walk_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk_strings(v))
    return out


def _find_invented_ids(payload: dict[str, Any], allowed: dict[str, set[str]]) -> list[str]:
    """Return human-readable problems if Gemini invented registry IDs or counts."""
    problems: list[str] = []
    blob = "\n".join(_walk_strings(payload))

    for m in _NCT_RE.findall(blob):
        if m.upper() not in allowed["nct"]:
            problems.append(f"invented_nct:{m.upper()}")
    for m in _PMID_RE.findall(blob):
        if str(m) not in allowed["pmid"]:
            problems.append(f"invented_pmid:{m}")
    for m in _PMID_BARE_RE.findall(blob):
        if str(m) not in allowed["pmid"]:
            problems.append(f"invented_pmid:{m}")
    for m in _SETID_RE.findall(blob):
        if m.lower() not in allowed["setid"]:
            problems.append(f"invented_setid:{m.lower()}")
    for m in _CHEMBL_RE.findall(blob):
        if m.upper() not in allowed["chembl"]:
            problems.append(f"invented_chembl:{m.upper()}")
    for m in _ENSG_RE.findall(blob):
        if m.upper() not in allowed["ensembl"]:
            problems.append(f"invented_ensembl:{m.upper()}")

    # Counts: any numeric count_in_sample / total_reports props must be allowlisted
    def _check_counts(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in {"count_in_sample", "total_reports", "total_reports_meta"} and v is not None:
                    if str(v) not in allowed["count"]:
                        problems.append(f"invented_count:{path}.{k}={v}")
                else:
                    _check_counts(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_counts(v, f"{path}[{i}]")

    _check_counts(payload)

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in problems:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _normalize_gemini_graph(
    raw: dict[str, Any],
    *,
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
) -> dict[str, Any]:
    """Fill edge defaults and ensure Drug/Condition scaffold nodes exist."""
    now = _now_iso()
    nodes_in = list(raw.get("nodes") or [])
    edges_in = list(raw.get("edges") or [])
    nodes: dict[str, Node] = {}
    for n in nodes_in:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        ntype = n.get("type")
        if ntype not in ALLOWED_NODE_TYPES:
            continue
        nodes[str(n["id"])] = {
            "id": str(n["id"]),
            "type": ntype,  # type: ignore[typeddict-item]
            "label": str(n.get("label") or n["id"]),
            "props": dict(n.get("props") or {}),
        }

    drug_id = f"drug:{_slug(drug_ingredient or drug_brand)}"
    cond_id = f"condition:{_slug(condition)}"
    if drug_id not in nodes:
        nodes[drug_id] = {
            "id": drug_id,
            "type": "Drug",
            "label": f"{drug_brand} ({drug_ingredient})",
            "props": {"brand": drug_brand, "ingredient": drug_ingredient},
        }
    if cond_id not in nodes:
        nodes[cond_id] = {
            "id": cond_id,
            "type": "Condition",
            "label": condition,
            "props": {},
        }

    edges: list[Edge] = []
    for e in edges_in:
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        if etype not in ALLOWED_EDGE_TYPES:
            continue
        src, tgt = e.get("source"), e.get("target")
        if not src or not tgt or not e.get("id"):
            continue
        # Ensure endpoints exist as stubs if Gemini omitted them
        if str(src) not in nodes:
            nodes[str(src)] = {
                "id": str(src),
                "type": "SourceDoc",
                "label": str(src),
                "props": {"stub": True},
            }
        if str(tgt) not in nodes:
            nodes[str(tgt)] = {
                "id": str(tgt),
                "type": "SourceDoc",
                "label": str(tgt),
                "props": {"stub": True},
            }
        props = dict(e.get("props") or {})
        if etype == "reports_ae":
            props.setdefault(
                "label",
                "FAERS voluntary reports — not rates, not causation, not a safety signal.",
            )
            props.setdefault(
                "note",
                "openFDA FAERS = voluntary reports only — not incidence rates, not causation.",
            )
        elif "note" not in props:
            props["note"] = "Public-record evidence link for retrieval — not a causation claim."
        edges.append(
            {
                "id": str(e["id"]),
                "type": etype,  # type: ignore[typeddict-item]
                "source": str(src),
                "target": str(tgt),
                "evidence_urls": [str(u) for u in (e.get("evidence_urls") or []) if u][:12],
                "sources": [str(s) for s in (e.get("sources") or []) if s],
                "first_seen": now,
                "last_seen": now,
                "trust_score": 0.0,
                "trust_breakdown": {},
                "retracted": bool(e.get("retracted")),
                "age_days": float(e.get("age_days") or 90.0),
                "props": props,
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges, "summary": raw.get("summary")}


def call_gemini_extract_json(prompt: str) -> dict[str, Any]:
    """Call Gemini with strict JSON schema. Never prints the API key.

    Separated for unit-test monkeypatching.
    """
    if not has_gemini_key():
        return {"ok": False, "error": "no_gemini_key", "used": False}
    key = gemini_api_key()
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=GEMINI_EXTRACT_JSON_SCHEMA,
                    temperature=0.1,
                ),
            )
        except (TypeError, AttributeError):
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GEMINI_EXTRACT_JSON_SCHEMA,
                    temperature=0.1,
                ),
            )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            return {"ok": False, "error": "empty_gemini_response", "used": True, "model": GEMINI_MODEL}
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"ok": False, "error": "gemini_json_not_object", "used": True, "model": GEMINI_MODEL}
        return {"ok": True, "data": data, "used": True, "model": GEMINI_MODEL}
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "used": False,
            "model": GEMINI_MODEL,
        }


def _gemini_structure_from_sources(
    *,
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    clinicaltrials: dict[str, Any],
    pubmed: dict[str, Any],
    openfda: dict[str, Any],
    dailymed: dict[str, Any],
    europepmc: dict[str, Any],
    opentargets: dict[str, Any],
    chembl: dict[str, Any],
) -> dict[str, Any]:
    """Gemini structured extract with allowlist validation. On any invent → fail."""
    digest = _source_digest(
        drug_brand=drug_brand,
        drug_ingredient=drug_ingredient,
        condition=condition,
        clinicaltrials=clinicaltrials,
        pubmed=pubmed,
        openfda=openfda,
        dailymed=dailymed,
        europepmc=europepmc,
        opentargets=opentargets,
        chembl=chembl,
    )
    allowed = _allowed_ids(digest)
    prompt = (
        "You structure a living evidence graph from the SOURCE DIGEST JSON below.\n"
        "Return ONLY JSON matching the schema (nodes + edges + optional summary).\n"
        "Elena red lines:\n"
        "- Public records only (already in the digest)\n"
        "- Do NOT invent NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts\n"
        "- Do NOT claim causation or incidence rates\n"
        "- openFDA FAERS = voluntary reports only (edge type reports_ae)\n"
        "- Retrieval-oriented evidence links for an LLM — not clinical advice\n"
        "- Prefer drug–gene–disease triangle when Open Targets / ChEMBL rows exist\n\n"
        f"SOURCE DIGEST:\n{json.dumps(digest, ensure_ascii=False)}"
    )
    raw = call_gemini_extract_json(prompt)
    if not raw.get("ok"):
        return {
            "ok": False,
            "used": bool(raw.get("used")),
            "mode": "gemini_structure",
            "error": raw.get("error") or "gemini_failed",
            "model": raw.get("model") or GEMINI_MODEL,
        }
    data = raw["data"]
    invented = _find_invented_ids(data, allowed)
    if invented:
        return {
            "ok": False,
            "used": True,
            "mode": "gemini_structure",
            "error": "invented_ids_refused",
            "invented": invented[:20],
            "model": GEMINI_MODEL,
        }
    normalized = _normalize_gemini_graph(
        data,
        drug_brand=drug_brand,
        drug_ingredient=drug_ingredient,
        condition=condition,
    )
    if not normalized["edges"]:
        return {
            "ok": False,
            "used": True,
            "mode": "gemini_structure",
            "error": "empty_edges_after_normalize",
            "model": GEMINI_MODEL,
        }
    # Re-check after normalization (scaffold ids are fine; re-scan props/urls)
    invented2 = _find_invented_ids(
        {"nodes": normalized["nodes"], "edges": normalized["edges"]},
        allowed,
    )
    if invented2:
        return {
            "ok": False,
            "used": True,
            "mode": "gemini_structure",
            "error": "invented_ids_refused_after_normalize",
            "invented": invented2[:20],
            "model": GEMINI_MODEL,
        }
    return {
        "ok": True,
        "used": True,
        "mode": "gemini_structure",
        "model": GEMINI_MODEL,
        "summary": (normalized.get("summary") or "")[:2000],
        "nodes": normalized["nodes"],
        "edges": normalized["edges"],
    }


def extract_from_sources(
    *,
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    clinicaltrials: dict[str, Any],
    pubmed: dict[str, Any],
    openfda: dict[str, Any],
    dailymed: dict[str, Any] | None = None,
    europepmc: dict[str, Any] | None = None,
    opentargets: dict[str, Any] | None = None,
    chembl: dict[str, Any] | None = None,
    use_gemini: bool = True,
) -> dict[str, Any]:
    """Build candidate nodes/edges from live or fixture source payloads.

    Preference order:
    1) Gemini 3.5 Flash structured JSON extract when GEMINI_API_KEY is set
       and use_gemini=True (strict schema; refuse invented IDs/counts).
    2) Deterministic rule-based extract on missing key, API failure, or refuse.
    """
    dailymed = dailymed or {}
    europepmc = europepmc or {}
    opentargets = opentargets or {}
    chembl = chembl or {}

    gemini_meta: dict[str, Any] = {"used": False, "mode": "rules"}

    if use_gemini and has_gemini_key():
        gem = _gemini_structure_from_sources(
            drug_brand=drug_brand,
            drug_ingredient=drug_ingredient,
            condition=condition,
            clinicaltrials=clinicaltrials,
            pubmed=pubmed,
            openfda=openfda,
            dailymed=dailymed,
            europepmc=europepmc,
            opentargets=opentargets,
            chembl=chembl,
        )
        if gem.get("ok"):
            return {
                "nodes": gem["nodes"],
                "edges": gem["edges"],
                "gemini": {
                    "used": True,
                    "mode": "gemini_structure",
                    "model": gem.get("model") or GEMINI_MODEL,
                    "summary": gem.get("summary") or "",
                },
                "disclaimer": _disclaimer(),
                "attribution_block": _attribution_block(
                    opentargets=opentargets,
                    chembl=chembl,
                    dailymed=dailymed,
                    europepmc=europepmc,
                ),
            }
        gemini_meta = {
            "used": bool(gem.get("used")),
            "mode": "rules_fallback",
            "gemini_structure_error": gem.get("error"),
            "invented": gem.get("invented"),
            "model": gem.get("model") or GEMINI_MODEL,
        }

    rules = _extract_rules(
        drug_brand=drug_brand,
        drug_ingredient=drug_ingredient,
        condition=condition,
        clinicaltrials=clinicaltrials,
        pubmed=pubmed,
        openfda=openfda,
        dailymed=dailymed,
        europepmc=europepmc,
        opentargets=opentargets,
        chembl=chembl,
    )
    rules["gemini"] = gemini_meta
    return rules



def extract_edges_tool(
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    sources_json: str,
) -> dict[str, Any]:
    """ADK-facing tool: sources_json holds clinicaltrials/pubmed/openfda/+new keys."""
    try:
        payload = json.loads(sources_json) if isinstance(sources_json, str) else sources_json
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"invalid sources_json: {e}"}
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
    return {"ok": True, **result}
