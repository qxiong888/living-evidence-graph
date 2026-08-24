"""ClinicalTrials.gov API v2 client. Never invent NCT IDs."""

from __future__ import annotations

from typing import Any

import httpx

from living_evidence_graph.config import CLINICALTRIALS_API, HTTP_TIMEOUT, USER_AGENT


def fetch_clinicaltrials(
    query: str,
    *,
    page_size: int = 5,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Search studies. Returns live payload or structured error (no invented NCTs)."""
    t = timeout if timeout is not None else HTTP_TIMEOUT
    params = {
        "query.term": query,
        "pageSize": str(page_size),
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = CLINICALTRIALS_API
    try:
        with httpx.Client(timeout=t, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001 — surface to caller; fixtures handle offline
        return {
            "ok": False,
            "source": "clinicaltrials.gov",
            "query": query,
            "error": str(e),
            "studies": [],
            "fixture": False,
        }

    studies: list[dict[str, Any]] = []
    for item in data.get("studies") or []:
        proto = item.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        status = proto.get("statusModule") or {}
        cond = proto.get("conditionsModule") or {}
        design = proto.get("designModule") or {}
        nct = ident.get("nctId")
        if not nct:
            continue
        studies.append(
            {
                "nct_id": nct,
                "title": ident.get("briefTitle") or ident.get("officialTitle") or nct,
                "overall_status": status.get("overallStatus"),
                "conditions": cond.get("conditions") or [],
                "phases": design.get("phases") or [],
                "url": f"https://clinicaltrials.gov/study/{nct}",
            }
        )

    return {
        "ok": True,
        "source": "clinicaltrials.gov",
        "query": query,
        "total_count": (data.get("totalCount") if isinstance(data.get("totalCount"), int) else len(studies)),
        "studies": studies,
        "fixture": False,
        "request_url": str(resp.url),
    }
