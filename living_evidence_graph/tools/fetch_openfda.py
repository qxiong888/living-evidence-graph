"""openFDA drug/event client. Counts are reports, not rates. Never invent totals."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import httpx

from living_evidence_graph.config import (
    HTTP_TIMEOUT,
    OPENFDA_API_KEY,
    OPENFDA_EVENT_URL,
    USER_AGENT,
)


def fetch_openfda_events(
    drug: str,
    *,
    limit: int = 5,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch FAERS adverse-event *reports* for a drug brand/ingredient.

    Label clearly: these are voluntary reports FDA received, not incidence rates
    and not proof of causation.
    """
    t = timeout if timeout is not None else HTTP_TIMEOUT
    # Prefer brand_name OR generic_name search; caller passes display string.
    search = f'patient.drug.medicinalproduct:"{drug}"'
    params: dict[str, str] = {
        "search": search,
        "limit": str(limit),
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = f"{OPENFDA_EVENT_URL}?{urlencode(params)}"

    try:
        with httpx.Client(timeout=t, headers=headers) as client:
            resp = client.get(OPENFDA_EVENT_URL, params=params)
            # openFDA returns 404 for zero results — treat as empty, not fatal
            if resp.status_code == 404:
                meta = {}
                try:
                    meta = (resp.json() or {}).get("meta") or {}
                except Exception:  # noqa: BLE001
                    pass
                return {
                    "ok": True,
                    "source": "openfda_faers",
                    "drug": drug,
                    "label": (
                        "openFDA FAERS voluntary reports — not rates, not causation, "
                        "not a safety signal."
                    ),
                    "total_reports": 0,
                    "last_updated": meta.get("last_updated"),
                    "reactions": [],
                    "sample_safetyreports": [],
                    "fixture": False,
                    "request_url": url,
                    "empty": True,
                }
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "source": "openfda_faers",
            "drug": drug,
            "error": str(e),
            "reactions": [],
            "sample_safetyreports": [],
            "fixture": False,
            "label": (
                "openFDA FAERS voluntary reports — not rates, not causation, "
                "not a safety signal."
            ),
        }

    meta = data.get("meta") or {}
    results_meta = meta.get("results") or {}
    total = results_meta.get("total")
    results = data.get("results") or []

    # Aggregate reaction terms from the returned sample only (not inventing)
    reaction_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for row in results:
        srid = row.get("safetyreportid")
        patient = row.get("patient") or {}
        reactions = patient.get("reaction") or []
        terms = []
        for r in reactions:
            term = r.get("reactionmeddrapt")
            if term:
                terms.append(term)
                reaction_counts[term] = reaction_counts.get(term, 0) + 1
        if srid:
            samples.append({"safetyreportid": srid, "reactions": terms[:5]})

    top_reactions = sorted(reaction_counts.items(), key=lambda x: (-x[1], x[0]))[:10]

    return {
        "ok": True,
        "source": "openfda_faers",
        "drug": drug,
        "label": (
            "openFDA FAERS voluntary reports — not rates, not causation, "
            "not a safety signal. Totals reflect reports FDA received."
        ),
        "total_reports": total if isinstance(total, int) else None,
        "last_updated": meta.get("last_updated"),
        "reactions": [{"term": t, "count_in_sample": c} for t, c in top_reactions],
        "sample_safetyreports": samples,
        "fixture": False,
        "request_url": url,
        "empty": False,
    }
