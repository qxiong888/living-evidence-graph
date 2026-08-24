"""DailyMed (FDA SPL via NLM) client. Never invent setids.

Attribution: FDA Structured Product Labels hosted by NLM DailyMed.
Labeled indication / warning text → high source_tier (dailymed_label).
Not causation; not incidence rates.
"""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from living_evidence_graph.config import (
    DAILYMED_SPL_XML,
    DAILYMED_SPLS,
    HTTP_TIMEOUT,
    USER_AGENT,
)

_ATTRIBUTION = "FDA SPL via NLM DailyMed (public). Not causation; not rates."


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _local_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _parse_spl_sections(xml_text: str) -> dict[str, Any]:
    """Extract indication snippets and warning-related AE-ish phrases from SPL XML."""
    out: dict[str, Any] = {
        "indications_snippet": None,
        "warnings_snippet": None,
        "warning_terms": [],
    }
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    sections: list[tuple[str, str]] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "section":
            continue
        title = ""
        text_bits: list[str] = []
        for child in el:
            cn = _strip_ns(child.tag)
            if cn == "title":
                title = _local_text(child)
            elif cn in {"text", "excerpt"}:
                text_bits.append(_local_text(child))
        body = " ".join(t for t in text_bits if t)[:4000]
        if title or body:
            sections.append((title.strip(), body))

    indications = []
    warnings = []
    for title, body in sections:
        low = f"{title} {body}".lower()
        if "indication" in low:
            indications.append((title, body[:800]))
        if any(
            w in low
            for w in (
                "boxed warning",
                "warning and precaution",
                "warnings and precautions",
                "immune-mediated",
                "adverse reaction",
            )
        ):
            warnings.append((title, body[:800]))

    if indications:
        out["indications_snippet"] = indications[0][1] or indications[0][0]
    if warnings:
        out["warnings_snippet"] = warnings[0][1] or warnings[0][0]

    blob = " ".join(w[1] for w in warnings).lower()
    candidates = [
        "pneumonitis",
        "colitis",
        "hepatitis",
        "endocrinopathy",
        "nephritis",
        "dermatitis",
        "myocarditis",
        "infusion-related reaction",
        "embryo-fetal toxicity",
    ]
    out["warning_terms"] = [t for t in candidates if t in blob][:8]
    return out


def fetch_dailymed(
    drug: str,
    *,
    pagesize: int = 5,
    parse_label_xml: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Search DailyMed SPLs by drug name; optionally parse one label XML."""
    t = timeout if timeout is not None else HTTP_TIMEOUT
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {"drug_name": drug, "pagesize": str(pagesize)}

    try:
        with httpx.Client(timeout=t, headers=headers, follow_redirects=True) as client:
            resp = client.get(DAILYMED_SPLS, params=params)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data") or []
            labels: list[dict[str, Any]] = []
            for row in rows:
                setid = row.get("setid")
                if not setid or not re.match(r"^[0-9a-fA-F-]{36}$", str(setid)):
                    continue
                title = row.get("title") or setid
                url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
                labels.append(
                    {
                        "setid": setid,
                        "title": title,
                        "published_date": row.get("published_date"),
                        "spl_version": row.get("spl_version"),
                        "url": url,
                    }
                )

            preferred = None
            for lab in labels:
                title_u = (lab.get("title") or "").upper()
                if "KEYTRUDA" in title_u and "QLEX" not in title_u:
                    preferred = lab
                    break
            if preferred is None and labels:
                preferred = labels[0]

            if preferred and parse_label_xml:
                xml_url = DAILYMED_SPL_XML.format(setid=preferred["setid"])
                try:
                    xr = client.get(xml_url, timeout=min(max(t, 25.0), 45.0))
                    if xr.status_code == 200 and xr.text:
                        parsed = _parse_spl_sections(xr.text)
                        preferred = {
                            **preferred,
                            "indications_snippet": parsed.get("indications_snippet"),
                            "warnings_snippet": parsed.get("warnings_snippet"),
                            "warning_terms": parsed.get("warning_terms") or [],
                        }
                except Exception:  # noqa: BLE001
                    pass
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "source": "dailymed",
            "attribution": _ATTRIBUTION,
            "drug": drug,
            "error": str(e),
            "labels": [],
            "fixture": False,
        }

    return {
        "ok": True,
        "source": "dailymed",
        "attribution": _ATTRIBUTION,
        "drug": drug,
        "labels": labels,
        "preferred": preferred,
        "count": len(labels),
        "fixture": False,
        "request_url": str(resp.url),
        "empty": not labels,
    }
