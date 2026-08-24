"""Open Targets Platform GraphQL client (public API).

Attribution: Open Targets Platform data — CC0 1.0 (public domain dedication).
Not an endorsement by Open Targets / EMBL-EBI / partners.
Structured KB associations only — not causation claims.
"""

from __future__ import annotations

from typing import Any

import httpx

from living_evidence_graph.config import HTTP_TIMEOUT, OPENTARGETS_GRAPHQL, USER_AGENT

_ATTRIBUTION = (
    "Open Targets Platform (https://platform.opentargets.org/) — data available under "
    "CC0 1.0 Universal. This project does not claim endorsement by Open Targets or partners."
)

_SEARCH = """
query SearchDrug($q: String!) {
  search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 1}) {
    hits { id name entity }
  }
}
"""

_DRUG = """
query DrugSpine($id: String!) {
  drug(chemblId: $id) {
    id
    name
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targetName
        targets { id approvedSymbol }
      }
    }
    indications {
      rows {
        disease { id name }
      }
    }
  }
}
"""

_TARGET_DISEASES = """
query TargetDiseases($id: String!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 8}) {
      count
      rows {
        score
        disease { id name }
      }
    }
  }
}
"""


def fetch_opentargets(
    drug: str,
    *,
    condition_hint: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch drug→gene, gene→disease, and drug→disease associations.

    Never invents ChEMBL / Ensembl / MONDO IDs — only returns API hits.
    """
    t = timeout if timeout is not None else HTTP_TIMEOUT
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=t, headers=headers) as client:
            sr = client.post(
                OPENTARGETS_GRAPHQL,
                json={"query": _SEARCH, "variables": {"q": drug}},
            )
            sr.raise_for_status()
            sdata = sr.json()
            if sdata.get("errors"):
                return {
                    "ok": False,
                    "source": "opentargets",
                    "attribution": _ATTRIBUTION,
                    "license": "CC0-1.0",
                    "drug": drug,
                    "error": str(sdata["errors"]),
                    "targets": [],
                    "indications": [],
                    "gene_disease": [],
                    "fixture": False,
                }
            hits = (((sdata.get("data") or {}).get("search") or {}).get("hits")) or []
            if not hits:
                return {
                    "ok": True,
                    "source": "opentargets",
                    "attribution": _ATTRIBUTION,
                    "license": "CC0-1.0",
                    "drug": drug,
                    "chembl_id": None,
                    "targets": [],
                    "indications": [],
                    "gene_disease": [],
                    "fixture": False,
                    "empty": True,
                }
            chembl_id = hits[0].get("id")
            drug_name = hits[0].get("name") or drug
            if not chembl_id or not str(chembl_id).startswith("CHEMBL"):
                return {
                    "ok": False,
                    "source": "opentargets",
                    "attribution": _ATTRIBUTION,
                    "license": "CC0-1.0",
                    "drug": drug,
                    "error": "search hit missing CHEMBL id",
                    "targets": [],
                    "indications": [],
                    "gene_disease": [],
                    "fixture": False,
                }

            dr = client.post(
                OPENTARGETS_GRAPHQL,
                json={"query": _DRUG, "variables": {"id": chembl_id}},
            )
            dr.raise_for_status()
            djson = dr.json()
            drug_obj = ((djson.get("data") or {}).get("drug")) or {}

            targets: list[dict[str, Any]] = []
            for row in ((drug_obj.get("mechanismsOfAction") or {}).get("rows")) or []:
                for tgt in row.get("targets") or []:
                    ensembl = tgt.get("id")
                    symbol = tgt.get("approvedSymbol")
                    if not ensembl or not str(ensembl).startswith("ENSG"):
                        continue
                    targets.append(
                        {
                            "ensembl_id": ensembl,
                            "symbol": symbol,
                            "target_name": row.get("targetName"),
                            "mechanism": row.get("mechanismOfAction"),
                        }
                    )

            indications: list[dict[str, Any]] = []
            hint = (condition_hint or "").lower()
            for row in ((drug_obj.get("indications") or {}).get("rows")) or []:
                disease = row.get("disease") or {}
                did, dname = disease.get("id"), disease.get("name")
                if not did or not dname:
                    continue
                item = {"disease_id": did, "disease_name": dname}
                if hint and hint in str(dname).lower():
                    indications.insert(0, item)
                else:
                    indications.append(item)
            indications = indications[:12]

            gene_disease: list[dict[str, Any]] = []
            for tgt in targets[:3]:
                tr = client.post(
                    OPENTARGETS_GRAPHQL,
                    json={
                        "query": _TARGET_DISEASES,
                        "variables": {"id": tgt["ensembl_id"]},
                    },
                )
                if tr.status_code != 200:
                    continue
                tjson = tr.json()
                tobj = ((tjson.get("data") or {}).get("target")) or {}
                for row in ((tobj.get("associatedDiseases") or {}).get("rows")) or []:
                    disease = row.get("disease") or {}
                    did, dname = disease.get("id"), disease.get("name")
                    score = row.get("score")
                    if not did or not dname:
                        continue
                    gene_disease.append(
                        {
                            "ensembl_id": tgt["ensembl_id"],
                            "symbol": tgt.get("symbol") or tobj.get("approvedSymbol"),
                            "disease_id": did,
                            "disease_name": dname,
                            "score": score,
                        }
                    )
            # Prefer NSCLC / lung / demo condition near the front
            if hint:
                tokens = [t for t in hint.replace("-", " ").split() if len(t) > 3]

                def _rank(x: dict) -> tuple:
                    name = str(x.get("disease_name") or "").lower()
                    if hint in name:
                        return (0, -(x.get("score") or 0))
                    if "non-small cell lung" in name or "nsclc" in name:
                        return (0, -(x.get("score") or 0))
                    hits = sum(1 for t in tokens if t in name)
                    return (1 if hits == 0 else 0, -hits, -(x.get("score") or 0))

                gene_disease.sort(key=_rank)
            gene_disease = gene_disease[:12]
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "source": "opentargets",
            "attribution": _ATTRIBUTION,
            "license": "CC0-1.0",
            "drug": drug,
            "error": str(e),
            "targets": [],
            "indications": [],
            "gene_disease": [],
            "fixture": False,
        }

    return {
        "ok": True,
        "source": "opentargets",
        "attribution": _ATTRIBUTION,
        "license": "CC0-1.0",
        "drug": drug,
        "drug_name": drug_name,
        "chembl_id": chembl_id,
        "targets": targets,
        "indications": indications,
        "gene_disease": gene_disease,
        "url": f"https://platform.opentargets.org/drug/{chembl_id}",
        "fixture": False,
        "empty": not (targets or indications or gene_disease),
        "note": (
            "Structured associations for LLM retrieval over the graph spine — "
            "not causation, not treatment advice."
        ),
    }
