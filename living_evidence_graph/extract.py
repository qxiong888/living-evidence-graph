"""Extract candidate graph triples from source payloads (Gemini optional).

Falls back to deterministic rule-based extraction when no API key / ADK.
Never invents NCT IDs, PMIDs, setids, ChEMBL/Ensembl IDs, or FDA counts —
only maps fields already present in tool outputs.

Core motif (Open Targets + ChEMBL spine):
  drug_targets_gene · gene_associated_with_disease · drug_indicated_for_disease
Corroboration layers: ClinicalTrials, PubMed, openFDA, DailyMed, Europe PMC.

LLM path: retrieval-only over the graph — do not dump PubMed abstracts or
non-OA Europe PMC full text into training corpora.
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
    """Build candidate nodes/edges from live or fixture source payloads."""
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

    gemini_notes: dict[str, Any] = {"used": False}
    if use_gemini and has_gemini_key():
        gemini_notes = _gemini_enrich(
            drug_brand=drug_brand,
            drug_ingredient=drug_ingredient,
            condition=condition,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "gemini": gemini_notes,
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


def _gemini_enrich(
    *,
    drug_brand: str,
    drug_ingredient: str,
    condition: str,
    node_count: int,
    edge_count: int,
) -> dict[str, Any]:
    """Ask Gemini for a short narrative summary only — no new IDs/counts."""
    key = gemini_api_key()
    prompt = (
        "You are summarizing a PUBLIC evidence graph for cancer research reuse. "
        "Do NOT invent NCT IDs, PMIDs, setids, ChEMBL IDs, or FDA counts. "
        "Do NOT claim causation or rates. "
        f"Drug: {drug_brand} ({drug_ingredient}). Condition: {condition}. "
        f"Graph already has {node_count} nodes and {edge_count} edges. "
        "Core motif: drug_targets_gene / gene_associated_with_disease / "
        "drug_indicated_for_disease (Open Targets + ChEMBL), with ClinicalTrials, "
        "PubMed, openFDA, DailyMed, Europe PMC as corroboration. "
        "Write 2-3 sentences on how an LLM should use this graph for retrieval-only "
        "(not as training text dumps of abstracts)."
    )
    try:
        from google import genai

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = getattr(resp, "text", None) or str(resp)
        return {"used": True, "model": GEMINI_MODEL, "summary": text.strip()[:2000]}
    except Exception as e:  # noqa: BLE001
        return {"used": False, "error": str(e), "model": GEMINI_MODEL}


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
