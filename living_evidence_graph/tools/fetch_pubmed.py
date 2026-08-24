"""PubMed / NCBI E-utilities client. Never invent PMIDs."""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree as ET

import httpx

from living_evidence_graph.config import (
    HTTP_TIMEOUT,
    NCBI_API_KEY,
    NCBI_EMAIL,
    PUBMED_ESEARCH,
    PUBMED_ESUMMARY,
    USER_AGENT,
)


def fetch_pubmed(
    query: str,
    *,
    retmax: int = 5,
    timeout: float | None = None,
) -> dict[str, Any]:
    """ESearch + ESummary. Returns live PMIDs only from the API response."""
    t = timeout if timeout is not None else HTTP_TIMEOUT
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params: dict[str, str] = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "email": NCBI_EMAIL,
        "tool": "living_evidence_graph",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    try:
        with httpx.Client(timeout=t, headers=headers) as client:
            es = client.get(PUBMED_ESEARCH, params=params)
            es.raise_for_status()
            es_data = es.json()
            idlist = (es_data.get("esearchresult") or {}).get("idlist") or []
            # Guard: only digits
            pmids = [pid for pid in idlist if str(pid).isdigit()]
            publications: list[dict[str, Any]] = []
            if pmids:
                sum_params = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "json",
                    "email": NCBI_EMAIL,
                    "tool": "living_evidence_graph",
                }
                if NCBI_API_KEY:
                    sum_params["api_key"] = NCBI_API_KEY
                sm = client.get(PUBMED_ESUMMARY, params=sum_params)
                sm.raise_for_status()
                sm_data = sm.json()
                result = sm_data.get("result") or {}
                for pmid in pmids:
                    rec = result.get(pmid) or {}
                    if not rec or rec.get("error"):
                        publications.append(
                            {
                                "pmid": pmid,
                                "title": f"PMID {pmid}",
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            }
                        )
                        continue
                    publications.append(
                        {
                            "pmid": pmid,
                            "title": rec.get("title") or f"PMID {pmid}",
                            "pubdate": rec.get("pubdate"),
                            "source": rec.get("source"),
                            "authors": [
                                a.get("name")
                                for a in (rec.get("authors") or [])[:5]
                                if isinstance(a, dict) and a.get("name")
                            ],
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        }
                    )
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "source": "pubmed",
            "query": query,
            "error": str(e),
            "publications": [],
            "fixture": False,
        }

    return {
        "ok": True,
        "source": "pubmed",
        "query": query,
        "count": int((es_data.get("esearchresult") or {}).get("count") or len(pmids)),
        "publications": publications,
        "fixture": False,
    }


def parse_medline_xml_title(xml_text: str) -> str | None:
    """Optional helper for XML mode; unused in default path."""
    try:
        root = ET.fromstring(xml_text)
        el = root.find(".//ArticleTitle")
        return el.text if el is not None else None
    except ET.ParseError:
        return None
