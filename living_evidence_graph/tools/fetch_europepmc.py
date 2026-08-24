"""Europe PMC client — retraction / erratum / correction signals for known PMIDs.

Never invents PMIDs. Feeds retraction_penalty when flags are present.
Attribution: Europe PMC (EMBL-EBI).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import httpx

from living_evidence_graph.config import EUROPEPMC_SEARCH, HTTP_TIMEOUT, USER_AGENT

_ATTRIBUTION = "Europe PMC (EMBL-EBI). Status signals only — not clinical advice."


def _norm_pmid(pmid: str | int) -> str | None:
    s = str(pmid).strip()
    return s if s.isdigit() else None


def _flags_from_result(rec: dict[str, Any]) -> dict[str, bool]:
    pub_types: list[str] = []
    ptl = rec.get("pubTypeList") or {}
    raw = ptl.get("pubType") if isinstance(ptl, dict) else None
    if isinstance(raw, list):
        pub_types = [str(x) for x in raw]
    elif isinstance(raw, str):
        pub_types = [raw]

    joined = " | ".join(pub_types).lower()
    retracted = bool(rec.get("isRetracted")) or "retraction" in joined
    erratum = "erratum" in joined or "errata" in joined
    correction = "correction" in joined or "corrigendum" in joined

    ccl = rec.get("commentCorrectionList") or {}
    comments = ccl.get("commentCorrection") if isinstance(ccl, dict) else None
    if isinstance(comments, dict):
        comments = [comments]
    if isinstance(comments, list):
        for c in comments:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type") or c.get("commentType") or "").lower()
            if "retract" in ctype:
                retracted = True
            if "errat" in ctype:
                erratum = True
            if "correct" in ctype or "corrig" in ctype:
                correction = True

    return {
        "retracted": retracted,
        "erratum": erratum,
        "correction": correction,
    }


def fetch_europepmc_status(
    pmids: Sequence[str | int],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Check Europe PMC for retraction/erratum/correction on existing PMIDs only."""
    t = timeout if timeout is not None else HTTP_TIMEOUT
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    clean = [p for p in (_norm_pmid(x) for x in pmids) if p]
    # De-dupe, preserve order, cap for demo latency
    seen: set[str] = set()
    ordered: list[str] = []
    for p in clean:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    ordered = ordered[:10]

    if not ordered:
        return {
            "ok": True,
            "source": "europepmc",
            "attribution": _ATTRIBUTION,
            "publications": [],
            "fixture": False,
            "empty": True,
            "note": "No PMIDs supplied — nothing to check.",
        }

    publications: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=t, headers=headers) as client:
            for pmid in ordered:
                q = f"EXT_ID:{pmid} AND SRC:MED"
                params = {
                    "query": q,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": "1",
                }
                resp = client.get(EUROPEPMC_SEARCH, params=params)
                resp.raise_for_status()
                data = resp.json()
                results = (data.get("resultList") or {}).get("result") or []
                rec = results[0] if results else {}

                flags = _flags_from_result(rec) if rec else {
                    "retracted": False,
                    "erratum": False,
                    "correction": False,
                }

                # Secondary: any retraction notice referencing this PMID?
                if not flags["retracted"]:
                    rq = (
                        f'(PUB_TYPE:"Retraction of Publication") AND '
                        f'(REF:{pmid} OR EXT_ID:{pmid})'
                    )
                    r2 = client.get(
                        EUROPEPMC_SEARCH,
                        params={
                            "query": rq,
                            "format": "json",
                            "pageSize": "1",
                        },
                    )
                    if r2.status_code == 200:
                        hit = int((r2.json() or {}).get("hitCount") or 0)
                        if hit > 0 and re.fullmatch(r"\d+", pmid):
                            # Only flag if API returned a positive hitCount
                            flags["retracted"] = True
                            flags["retraction_notice_hits"] = hit  # type: ignore[index]

                title = (rec.get("title") if rec else None) or f"PMID {pmid}"
                publications.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "url": f"https://europepmc.org/article/MED/{pmid}",
                        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "pub_types": (
                            ((rec.get("pubTypeList") or {}).get("pubType"))
                            if rec
                            else None
                        ),
                        **flags,
                        "found_in_europepmc": bool(rec),
                    }
                )
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "source": "europepmc",
            "attribution": _ATTRIBUTION,
            "error": str(e),
            "publications": [],
            "fixture": False,
            "pmids_requested": ordered,
        }

    return {
        "ok": True,
        "source": "europepmc",
        "attribution": _ATTRIBUTION,
        "publications": publications,
        "retracted_pmids": [p["pmid"] for p in publications if p.get("retracted")],
        "erratum_or_correction_pmids": [
            p["pmid"]
            for p in publications
            if p.get("erratum") or p.get("correction")
        ],
        "fixture": False,
        "empty": False,
    }
