"""ChEMBL REST client for molecule / mechanism links.

Attribution: ChEMBL data — CC BY-SA 3.0.
Cite: Mendez et al., Nucleic Acids Res. 2019 (doi:10.1093/nar/gky1075)
and the ChEMBL release/version returned by the API when available.
ShareAlike applies to ChEMBL-derived graph subsets. No endorsement claimed.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from living_evidence_graph.config import (
    CHEMBL_MECHANISM,
    CHEMBL_MOLECULE_SEARCH,
    CHEMBL_TARGET,
    HTTP_TIMEOUT,
    USER_AGENT,
)

_ATTRIBUTION = (
    "ChEMBL (https://www.ebi.ac.uk/chembl/) — CC BY-SA 3.0. "
    "Cite: Mendez D et al. Nucleic Acids Res. 2019;47(D1):D930–D940. "
    "doi:10.1093/nar/gky1075. ChEMBL-derived graph subsets are ShareAlike. "
    "This project does not claim endorsement by EMBL-EBI / ChEMBL."
)


def _gene_symbol_from_target(target: dict[str, Any]) -> str | None:
    for comp in target.get("target_components") or []:
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") == "GENE_SYMBOL" and syn.get("component_synonym"):
                return str(syn["component_synonym"])
    return None


def fetch_chembl(
    drug: str,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Search molecule + mechanism of action. Never invents ChEMBL IDs."""
    t = timeout if timeout is not None else HTTP_TIMEOUT
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        with httpx.Client(timeout=t, headers=headers) as client:
            sr = client.get(
                CHEMBL_MOLECULE_SEARCH,
                params={"q": drug, "limit": 3},
            )
            sr.raise_for_status()
            sdata = sr.json()
            molecules = sdata.get("molecules") or []
            # Prefer exact synonym / pref_name match when present
            chosen = None
            low = drug.lower()
            for m in molecules:
                pref = str(m.get("pref_name") or "").lower()
                if pref == low or low in pref:
                    chosen = m
                    break
            if chosen is None and molecules:
                chosen = molecules[0]
            if not chosen:
                return {
                    "ok": True,
                    "source": "chembl",
                    "attribution": _ATTRIBUTION,
                    "license": "CC-BY-SA-3.0",
                    "citation": "Mendez et al., Nucleic Acids Res. 2019; doi:10.1093/nar/gky1075",
                    "drug": drug,
                    "molecule_chembl_id": None,
                    "mechanisms": [],
                    "fixture": False,
                    "empty": True,
                }

            mol_id = chosen.get("molecule_chembl_id")
            if not mol_id or not re.match(r"^CHEMBL\d+$", str(mol_id)):
                return {
                    "ok": False,
                    "source": "chembl",
                    "attribution": _ATTRIBUTION,
                    "license": "CC-BY-SA-3.0",
                    "drug": drug,
                    "error": "molecule response missing CHEMBL id",
                    "mechanisms": [],
                    "fixture": False,
                }

            mr = client.get(
                CHEMBL_MECHANISM,
                params={"molecule_chembl_id": mol_id, "limit": 10},
            )
            mr.raise_for_status()
            mechs_raw = (mr.json() or {}).get("mechanisms") or []

            mechanisms: list[dict[str, Any]] = []
            for mech in mechs_raw:
                tid = mech.get("target_chembl_id")
                if tid and not re.match(r"^CHEMBL\d+$", str(tid)):
                    continue
                gene_symbol = None
                target_name = None
                if tid:
                    try:
                        tr = client.get(CHEMBL_TARGET.format(chembl_id=tid))
                        if tr.status_code == 200:
                            tjson = tr.json() or {}
                            target_name = tjson.get("pref_name")
                            gene_symbol = _gene_symbol_from_target(tjson)
                    except Exception:  # noqa: BLE001
                        pass
                mechanisms.append(
                    {
                        "molecule_chembl_id": mol_id,
                        "target_chembl_id": tid,
                        "target_name": target_name or mech.get("mechanism_of_action"),
                        "gene_symbol": gene_symbol,
                        "action_type": mech.get("action_type"),
                        "mechanism_of_action": mech.get("mechanism_of_action"),
                        "max_phase": mech.get("max_phase"),
                    }
                )

            # Optional status for version string
            chembl_version = None
            try:
                st = client.get("https://www.ebi.ac.uk/chembl/api/data/status.json")
                if st.status_code == 200:
                    chembl_version = (st.json() or {}).get("chembl_db_version")
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "source": "chembl",
            "attribution": _ATTRIBUTION,
            "license": "CC-BY-SA-3.0",
            "citation": "Mendez et al., Nucleic Acids Res. 2019; doi:10.1093/nar/gky1075",
            "drug": drug,
            "error": str(e),
            "mechanisms": [],
            "fixture": False,
        }

    return {
        "ok": True,
        "source": "chembl",
        "attribution": _ATTRIBUTION,
        "license": "CC-BY-SA-3.0",
        "citation": "Mendez et al., Nucleic Acids Res. 2019; doi:10.1093/nar/gky1075",
        "chembl_db_version": chembl_version,
        "drug": drug,
        "molecule_chembl_id": mol_id,
        "pref_name": chosen.get("pref_name"),
        "mechanisms": mechanisms,
        "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{mol_id}/",
        "fixture": False,
        "empty": not mechanisms,
        "sharealike_note": (
            "Downstream graph subsets derived from ChEMBL mechanism rows are "
            "subject to CC BY-SA 3.0 ShareAlike."
        ),
    }
