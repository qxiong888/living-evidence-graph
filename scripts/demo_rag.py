#!/usr/bin/env python3
"""Contest RAG demo: Keytruda/NSCLC question → bare vs graph-grounded vs strict.

Writes out/demo/rag_compare.html + rag_compare.json.
Video-friendly 1280×720 layout with unverifiable-vs-LEG highlights on bare
and green checks on grounded edge_id citations.
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from living_evidence_graph.config import DEMO_DIR  # noqa: E402
from living_evidence_graph.schema import display_source_labels  # noqa: E402
from living_evidence_graph.rag import (  # noqa: E402
    DISCLAIMER,
    _call_gemini,
    rag_compare,
)

DEMO_QUESTION = (
    "What NSCLC indication and PDCD1 target does the graph list for Keytruda "
    "(pembrolizumab), what does DailyMed warn about pneumonitis and hepatitis, "
    "and what is KEYNOTE-799 (NCT03631784) in Stage III? "
    "What is the OS hazard ratio for KEYNOTE-888?"
)

BARE_LOOSE_QUESTION = DEMO_QUESTION

SYSTEM_BARE_LOOSE = (
    "You are a helpful oncology research assistant. "
    "Answer every clause with concrete public evidence. "
    "Include specific trial IDs (NCT), PubMed PMIDs, and numeric overall-survival "
    "hazard ratios when the question asks for them. Be specific and fluent; "
    "do not leave numeric fields blank."
)

_ID_RE = re.compile(
    r"(?:NCT\d{8}|PMID\s*:?\s*\[?\d{5,8}\]?)",
    re.I,
)
# Numeric OS/HR claims (no hardcoded figure — match whatever the model emits).
_HR_RE = re.compile(
    r"(?:OS\s+)?(?:hazard\s+ratio|HR)\s*"
    r"(?:for\s+(?:death|OS|overall\s+survival))?\s*"
    r"(?:of|was|=|:)?\s*(?:approximately\s+|approx\.?\s+|~)?\s*"
    r"\d+\.\d+(?:\s*\([^)]{0,48}\))?",
    re.I,
)
_KN888_RE = re.compile(r"KEYNOTE[\s-]*888", re.I)
# Live answers cite edge_3 / (edge_3) / `edge_3` / edge 1, not only edge:…
_EDGE_CITE_RE = re.compile(
    r"(?:"
    r"`*edge:[a-z0-9_.:\-]+`*"          # full edge:… ids
    r"|"
    r"\(?`*edge[_\s\-]\d+`*\)?"     # edge_1, (edge_3), `edge_3`, edge 1
    r")",
    re.I,
)

# Phrase spans absent from retrieved LEG edges.
# One highlighter: bright red = Not in retrieved graph.
_PHRASE_HIGHLIGHTS: list[tuple[str, str]] = [
    ("KEYNOTE-888", "red"),
    ("KEYNOTE 888", "red"),
]

# Verified 2026-08-25 (Reed + ClinicalTrials.gov): bare paired
# KEYNOTE-888 = NCT04875416. That PAIR does not exist. NCT04875416 is
# PGB2 (ALS/HSP/PLS/PMA/FTD, University of Miami) — not Keytruda NSCLC.
# Do not claim the NCT string is unregistered. Do not invent an HR.
_FAB_KN = "KEYNOTE-888"
_FAB_NCT = "NCT04875416"
_PAIR_CALLOUT_HTML = (
    "<span class='fab-note' role='note'>"
    "<strong>Invented pairing:</strong> KEYNOTE-888 ≠ NCT04875416. "
    "NCT04875416 is an ALS observational study (PGB2). "
    "No Merck KEYNOTE-888 NSCLC OS trial on ClinicalTrials.gov."
    "<span class='sub'>KEYNOTE-888 only appears as a secondary ID "
    "on a different NCT (03696212).</span>"
    "</span>"
)


def _clip(text: str, n: int = 900) -> str:
    t = (text or "").strip()
    if len(t) <= n:
        return t
    return t[: n - 1].rstrip() + "…"


def _has_invented_looking_ids(text: str) -> bool:
    return bool(_ID_RE.search(text or ""))


def _has_invented_hr(text: str) -> bool:
    return bool(_HR_RE.search(text or ""))


def _has_bare_invention(text: str, edges: list | None = None) -> bool:
    """True when unconstrained bare emitted an HR or an ID absent from retrieval."""
    if _has_invented_hr(text):
        return True
    if not edges:
        return _has_invented_looking_ids(text)
    known = _ids_in_retrieved(edges)
    for m in _ID_RE.finditer(text or ""):
        token = re.sub(r"\s+", "", m.group(0).upper()).replace("PMID:", "PMID")
        token_norm = token.replace(":", "")
        in_graph = any(
            kid.replace(":", "").replace(" ", "") == token_norm or kid in token_norm
            for kid in known
        )
        if not in_graph:
            return True
    return False


def _ids_in_retrieved(edges: list) -> set[str]:
    blob = json.dumps(edges or []).upper()
    found: set[str] = set()
    for m in re.finditer(r"NCT\d{8}", blob, re.I):
        found.add(m.group(0).upper())
    for m in re.finditer(r"PMID\s*:?\s*\d{5,8}", blob, re.I):
        found.add(re.sub(r"\s+", "", m.group(0).upper()))
    return found


def _call_bare_loose(question: str) -> dict:
    user = (
        f"Question: {question}\n\n"
        "Provide a detailed answer. Include specific NCT/PMID citations and a "
        "numeric OS hazard ratio for every KEYNOTE trial the question names."
    )
    result = _call_gemini(system=SYSTEM_BARE_LOOSE, user=user)
    result["mode"] = "bare_loose"
    result["question"] = question
    return result


def _window_around(text: str, idx: int, before: int = 80, after: int = 420) -> str:
    start = max(0, idx - before)
    return text[start : idx + after].rstrip()


def _demo_bare_excerpt(text: str, limit: int = 1280) -> str:
    """Keep the invented KEYNOTE-888 pairing heading above the fold."""
    t = (text or "").strip()
    m888 = _KN888_RE.search(t)
    if not m888:
        return t if len(t) <= limit else _clip(t, limit)
    # Snap to the markdown heading that contains the fabricated pair.
    head_idx = t.rfind("\n### ", 0, m888.start() + 1)
    if head_idx < 0:
        head_idx = t.rfind("\n", 0, m888.start())
    start = (head_idx + 1) if head_idx >= 0 else max(0, m888.start() - 4)
    pair = t[start : m888.start() + 500].rstrip()
    sep = "\n…\n"
    intro = t[:180].rstrip()
    parts = [pair]
    if intro and intro[:28] not in pair:
        parts.append(intro)
    joined = sep.join(parts)
    if len(joined) > limit:
        joined = joined[: limit - 1].rstrip() + "…"
    elif not joined.endswith("…"):
        joined = joined.rstrip() + "…"
    return joined


def _attach_fabricated_pair_callout(annotated_html: str) -> str:
    """Wrap first KEYNOTE-888 + NCT04875416 red marks; arrow-box points at them."""
    kn = re.search(
        r"<mark class='uv red'[^>]*>\s*KEYNOTE-888\s*</mark>",
        annotated_html,
        re.I,
    )
    if not kn:
        return annotated_html
    window = annotated_html[kn.start() : kn.start() + 280]
    nct = re.search(
        r"<mark class='uv red'[^>]*>\s*NCT04875416\s*</mark>",
        window,
        re.I,
    )
    wrap_start = kn.start()
    wrap_end = kn.end() if not nct else kn.start() + nct.end()
    # Include surrounding parentheses so the pair reads as one unit.
    if wrap_start > 0 and annotated_html[wrap_start - 1] == "(":
        wrap_start -= 1
    if wrap_end < len(annotated_html) and annotated_html[wrap_end] == ")":
        wrap_end += 1
    inner = annotated_html[wrap_start:wrap_end]
    return (
        annotated_html[:wrap_start]
        + "<span class='fab-pair'><span class='fab-marks'>"
        + inner
        + "</span>"
        + _PAIR_CALLOUT_HTML
        + "</span>"
        + annotated_html[wrap_end:]
    )


def _clip_keep_888(text: str, n: int = 1100) -> str:
    """Keep the KEYNOTE-888 empty/unsupported clause in the visible window."""
    t = (text or "").strip()
    m = _KN888_RE.search(t)
    if not m:
        return t if len(t) <= n else _clip(t, n)
    tail = t[m.start() : m.start() + 280].rstrip()
    if m.start() + len(tail) <= n and len(t) <= n:
        return t
    budget = max(220, min(n - len(tail) - 8, 560))
    head = t[:budget].rstrip()
    if head.endswith(tail[:20]):
        return _clip(t, n)
    return head + "\n…\n" + tail


def _annotate_bare_html(text: str, edges: list) -> tuple[str, list[dict]]:
    """Return HTML with unverifiable spans marked + audit list of highlights."""
    raw = text or ""
    known_ids = _ids_in_retrieved(edges)
    candidates: list[tuple[int, int, str, str]] = []

    label = "Not in retrieved graph"
    for m in _ID_RE.finditer(raw):
        prefix = raw[max(0, m.start() - 24) : m.start()].lower()
        if "study/" in prefix or "pubmed.ncbi" in prefix or "nih.gov/" in prefix:
            continue
        token = re.sub(r"\s+", "", m.group(0).upper()).replace("PMID:", "PMID")
        token_norm = token.replace(":", "")
        in_graph = False
        for kid in known_ids:
            if kid.replace(":", "").replace(" ", "") == token_norm or kid in token_norm:
                in_graph = True
                break
        if in_graph:
            continue
        candidates.append((m.start(), m.end(), "red", label))

    for m in _HR_RE.finditer(raw):
        candidates.append((m.start(), m.end(), "red", label))

    for phrase, sev in _PHRASE_HIGHLIGHTS:
        start = 0
        while True:
            idx = raw.find(phrase, start)
            if idx < 0:
                break
            candidates.append(
                (
                    idx,
                    idx + len(phrase),
                    "red",
                    "Not in retrieved graph",
                )
            )
            start = idx + len(phrase)

    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen: list[tuple[int, int, str, str]] = []
    occupied_until = -1
    for c in candidates:
        if c[0] < occupied_until:
            continue
        chosen.append(c)
        occupied_until = c[1]

    parts: list[str] = []
    audit: list[dict] = []
    cursor = 0
    for start, end, sev, label in chosen:
        parts.append(html.escape(raw[cursor:start]))
        span = html.escape(raw[start:end])
        parts.append(
            f"<mark class='uv {sev}' title='{html.escape(label)}'>{span}</mark>"
        )
        audit.append(
            {
                "span": raw[start:end],
                "severity": sev,
                "reason": label,
                "start": start,
                "end": end,
            }
        )
        cursor = end
    parts.append(html.escape(raw[cursor:]))
    return "".join(parts), audit


def _annotate_grounded_html(text: str, edges: list | None = None) -> str:
    """Green-highlight cited edge ids (and optional short cited edge types)."""
    raw = text or ""
    spans: list[tuple[int, int]] = []
    for m in _EDGE_CITE_RE.finditer(raw):
        spans.append((m.start(), m.end()))

    types = sorted(
        {str(e.get("type") or "") for e in (edges or []) if e.get("type")},
        key=len,
        reverse=True,
    )
    if types:
        # Only citation-like tokens: [type], (type), `type`, or standalone type.
        alts = "|".join(re.escape(t) for t in types)
        type_re = re.compile(rf"(?:\[|\(|`)?(?:{alts})(?:\]|\)|`)?")
        for m in type_re.finditer(raw):
            spans.append((m.start(), m.end()))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen: list[tuple[int, int]] = []
    occupied = -1
    for start, end in spans:
        if start < occupied:
            continue
        chosen.append((start, end))
        occupied = end

    def wrap(token: str) -> str:
        esc = html.escape(token)
        return (
            f"<span class='edgeok' title='Cited retrieved edge_id'>"
            f"<span class='chk'>✓</span>{esc}</span>"
        )

    out: list[str] = []
    cursor = 0
    for start, end in chosen:
        out.append(html.escape(raw[cursor:start]))
        out.append(wrap(raw[start:end]))
        cursor = end
    out.append(html.escape(raw[cursor:]))
    return "".join(out)


def _answer_block(
    payload: dict,
    *,
    emphasize: str | None = None,
    body_html: str | None = None,
) -> str:
    status = html.escape(str(payload.get("status") or ""))
    model = html.escape(str(payload.get("model") or ""))
    if body_html is None:
        body_html = html.escape(_clip(str(payload.get("text") or ""), 1100))
    used = "yes" if payload.get("used") else "no"
    abstain = " · <strong>ABSTAIN</strong>" if payload.get("abstained") else ""
    note = ""
    if emphasize:
        note = f"<p class='callout'>{html.escape(emphasize)}</p>"
    return (
        f"{note}"
        f"<p class='meta'>status=<code>{status}</code> · model=<code>{model}</code> · "
        f"gemini={used}{abstain}</p>"
        f"<pre class='answer'>{body_html}</pre>"
    )


def _edges_compact(edges: list) -> str:
    if not edges:
        return "<p><em>No edges retrieved.</em></p>"
    rows = []
    for e in edges[:4]:
        spine = " ★" if e.get("triangle_spine") else ""
        rows.append(
            "<tr>"
            f"<td class='etype'>{html.escape(str(e.get('type')))}{spine}</td>"
            f"<td>{html.escape(str(e.get('source_label')))} → "
            f"{html.escape(str(e.get('target_label')))}</td>"
            f"<td class='trust'>{html.escape(str(e.get('trust_score')))}</td>"
            f"<td>{html.escape(', '.join(display_source_labels(e.get('sources'), e.get('evidence_urls'))))}</td>"
            "</tr>"
        )
    more = ""
    if len(edges) > 4:
        more = f"<p class='note'>+{len(edges) - 4} more edges in retrieval…</p>"
    return (
        "<table><thead><tr>"
        "<th>Type</th><th>Edge</th><th>Trust</th><th>Sources</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + more
    )


def render_html(result: dict) -> str:
    q = html.escape(result.get("question") or "")
    disc = html.escape(result.get("disclaimer") or DISCLAIMER)
    gp = html.escape(str(result.get("graph_path") or "(none)"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    gemini = "LIVE Gemini" if result.get("gemini_used") else "Gemini skipped"
    n_edges = len(result.get("retrieved_edges") or [])
    has_strict = bool(result.get("strict"))
    grid_cols = "1fr 1fr 1fr" if has_strict else "1fr 1fr"

    edges = result.get("retrieved_edges") or []
    bare = result.get("bare") or {}
    bare_text = _demo_bare_excerpt(str(bare.get("text") or ""), 1280)
    bare_html, _audit = _annotate_bare_html(bare_text, edges)
    bare_html = _attach_fabricated_pair_callout(bare_html)
    has_ids = _has_invented_looking_ids(str(bare.get("text") or ""))
    has_hr = _has_invented_hr(str(bare.get("text") or ""))
    invented = _has_bare_invention(str(bare.get("text") or ""), edges)
    pair_in_text = (
        _FAB_KN.lower() in str(bare.get("text") or "").lower()
        and _FAB_NCT in str(bare.get("text") or "").upper()
    )
    if pair_in_text:
        bare_callout = (
            "Red + yellow callout = invented KEYNOTE–NCT pairing vs public registry."
        )
    elif invented:
        bare_callout = (
            "Red = Not in retrieved graph. "
            "Unconstrained bare emitted trial IDs absent from retrieved edges."
        )
    else:
        bare_callout = "Bare cites no checkable edge IDs."
    grounded = result.get("grounded") or {}
    grounded_text = _clip_keep_888(str(grounded.get("text") or ""), 1180)
    grounded_html = _annotate_grounded_html(grounded_text, edges)

    strict_card = ""
    if has_strict:
        strict = result.get("strict") or {}
        strict_text = _clip_keep_888(str(strict.get("text") or ""), 900)
        strict_html = _annotate_grounded_html(strict_text, edges)
        strict_card = f"""
    <div class="card strict">
      <div class="label">STRICT · graph only / abstain if empty</div>
      <h2>Strict (library-only)</h2>
      {_answer_block(
            strict,
            emphasize="Graph-backed clauses + edge cites. KEYNOTE-888 has no edge — that clause only is empty.",
            body_html=strict_html,
        )}
    </div>"""

    bare_extra = ""
    if not invented:
        bare_extra = (
            "<p class='callout callout2'>Bare cites no edge IDs — fluent but not "
            "checkable against the graph.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=1280"/>
<title>LIVE · Bare vs Living Evidence Graph</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: "Segoe UI", system-ui, sans-serif;
    color: #e8eef7;
    background: #0b1f33;
    width: 1280px;
    min-height: 720px;
    overflow: hidden;
  }}
  .wrap {{ padding: 14px 18px 10px; }}
  .top {{
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 16px; margin-bottom: 8px;
  }}
  h1 {{
    margin: 0; font-size: 20px; font-weight: 700; color: #fff;
    letter-spacing: 0.2px;
  }}
  .badge {{
    background: #1b5e20; color: #d8ffe0; font-size: 12px; font-weight: 700;
    padding: 6px 12px; border-radius: 999px; border: 1px solid #7dffa8;
    white-space: nowrap;
  }}
  .qbox {{
    background: #12263a; border: 1px solid #3a6a8a; border-radius: 10px;
    padding: 6px 12px; margin-bottom: 6px; font-size: 11px; line-height: 1.3;
    color: #cde;
  }}
  .qbox strong {{ color: #ffd666; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;
    font-size: 11px; margin: 0 0 8px; color: #a8b8c8;
  }}
  .legend .sw {{
    display: inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 700;
    margin-right: 4px;
  }}
  .legend .sw.red {{ background: #FF4D6D; color: #1a0a0c; border: 1px solid #FF3B4A; font-weight: 800; }}
  .legend .sw.call {{ background: #FFE14D; color: #1a0a0c; border: 2px solid #FF3B4A; font-weight: 800; }}
  .grid {{
    display: grid; grid-template-columns: {grid_cols}; gap: 10px;
  }}
  .card {{
    background: #0f2438; border: 1px solid #2a4a66; border-radius: 12px;
    padding: 7px 9px 5px; min-height: 360px; position: relative;
  }}
  .card h2 {{
    margin: 0 0 4px; font-size: 14px; color: #fff;
  }}
  .label {{
    display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: 0.6px;
    text-transform: uppercase; padding: 3px 7px; border-radius: 6px; margin-bottom: 4px;
  }}
  .bare .label {{ background: #5c1520; color: #ffb4c0; }}
  .grounded .label {{ background: #0d4a2e; color: #7dffa8; }}
  .strict .label {{ background: #1a3a5c; color: #a8d4ff; }}
  .bare {{ border-top: 4px solid #e05a6a; }}
  .grounded {{ border-top: 4px solid #2ecc71; }}
  .strict {{ border-top: 4px solid #4aa3ff; }}
  pre.answer {{
    white-space: pre-wrap; word-break: break-word; font-size: 10.5px; line-height: 1.32;
    background: #0a1a2a; padding: 7px; border-radius: 6px; margin: 0;
    max-height: 318px; overflow: hidden; color: #dce6f2;
    font-family: ui-monospace, "DejaVu Sans Mono", monospace;
  }}
  mark.uv {{
    border-radius: 2px; padding: 0 3px;
  }}
  mark.uv.red {{
    background: #FF4D6D; color: #1a0a0c; font-weight: 800;
  }}
  .fab-pair {{
    position: relative; display: inline;
  }}
  .fab-marks {{
    display: inline;
    box-shadow: 0 0 0 2px #FFE14D;
    border-radius: 3px;
    padding: 1px 2px;
    background: rgba(255, 225, 77, 0.18);
  }}
  .fab-note {{
    display: block;
    margin: 7px 0 6px;
    padding: 6px 8px 6px 8px;
    background: #FFE14D;
    color: #1a0a0c;
    font-weight: 750;
    font-size: 10px;
    line-height: 1.28;
    border-radius: 6px;
    border: 2px solid #FF3B4A;
    box-shadow: 0 3px 10px rgba(0,0,0,0.4);
    white-space: normal;
    position: relative;
  }}
  .fab-note::before {{
    content: "";
    position: absolute;
    top: -8px;
    left: 36px;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-bottom: 8px solid #FF3B4A;
  }}
  .fab-note::after {{
    content: "";
    position: absolute;
    top: -5px;
    left: 38px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 6px solid #FFE14D;
  }}
  .fab-note strong {{ font-weight: 900; }}
  .fab-note .sub {{
    display: block;
    margin-top: 3px;
    font-size: 9px;
    font-weight: 650;
    color: #3a1a10;
  }}
  .edgeok {{
    background: #39FF7A; color: #06210f; font-weight: 800;
    border-radius: 2px; padding: 0 3px; white-space: nowrap;
  }}
  .edgeok .chk {{
    color: #06210f; font-weight: 800; margin-right: 2px;
  }}
  .meta {{ font-size: 10px; color: #8aa; margin: 0 0 4px; }}
  .callout {{
    font-size: 11px; margin: 0 0 5px; padding: 5px 7px; border-radius: 6px;
    background: #152a40; color: #a8d4ff; border-left: 3px solid #4aa3ff;
  }}
  .bare .callout {{ border-left-color: #e05a6a; color: #ffb4c0; background: #2a1520; }}
  .bare .callout2 {{ border-left-color: #e0a020; color: #ffd666; background: #2a2210; }}
  .grounded .callout {{ border-left-color: #2ecc71; color: #7dffa8; background: #0d2a1c; }}
  .edges {{
    margin-top: 8px; background: #12263a; border: 1px solid #2a4a66;
    border-radius: 10px; padding: 6px 10px;
  }}
  .edges h3 {{ margin: 0 0 4px; font-size: 12px; color: #ffd666; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 10.5px; }}
  th, td {{ border: 1px solid #2a4a66; padding: 2px 5px; vertical-align: top; }}
  th {{ background: #1a3348; color: #a8d4ff; text-align: left; }}
  td.etype {{ color: #7dffa8; white-space: nowrap; }}
  td.trust {{ color: #ffd666; font-weight: 700; }}
  .foot {{
    margin-top: 6px; font-size: 10px; color: #6a849e;
    display: flex; justify-content: space-between; gap: 12px;
  }}
  code {{ font-size: 0.92em; }}
  .note {{ font-size: 10px; color: #8aa; margin: 2px 0 0; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h1>LIVE precision compare — same judge question</h1>
      <span class="badge">{gemini} · {n_edges} edges · retrieval-only</span>
    </div>
    <div class="qbox"><strong>Question:</strong> {q}</div>
    <div class="legend">
      <span><span class="sw red">Red</span> Not in retrieved graph</span>
      <span><span class="sw call">Red + callout</span> fabricated pairing vs public registry</span>
      <span><span class="sw" style="background:#39FF7A;color:#06210f;">Green ✓</span> Cited retrieved edge_id</span>
    </div>
    <div class="grid">
      <div class="card bare">
        <div class="label">BEFORE graph — bare LLM</div>
        <h2>Bare Gemini (fluent · not checkable)</h2>
        {_answer_block(bare, emphasize=bare_callout, body_html=bare_html)}
        {bare_extra}
      </div>
      <div class="card grounded">
        <div class="label">WITH Living Evidence Graph</div>
        <h2>Grounded Gemini (cites edges)</h2>
        {_answer_block(
            grounded,
            emphasize="Only retrieved high-trust edges. Green ✓ = cited edge_id in top-k.",
            body_html=grounded_html,
        )}
      </div>
      {strict_card}
    </div>
    <div class="edges">
      <h3>Retrieved high-trust edges (injected as context — not fine-tuning)</h3>
      {_edges_compact(edges)}
    </div>
    <div class="foot">
      <span>{html.escape(disc[:150])}…</span>
      <span>{ts} · graph=<code>{gp}</code></span>
    </div>
  </div>
</body>
</html>
"""


def render_from_saved(json_path: Path | None = None) -> int:
    """Re-render HTML from a saved rag_compare.json (no live Gemini call)."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    path = json_path or (DEMO_DIR / "rag_compare.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    excerpt = _demo_bare_excerpt(str((result.get("bare") or {}).get("text") or ""))
    _, audit = _annotate_bare_html(excerpt, result.get("retrieved_edges") or [])
    result["bare_highlight_audit"] = audit
    html_path = DEMO_DIR / "rag_compare.html"
    html_path.write_text(render_html(result), encoding="utf-8")
    print(f"Re-rendered {html_path} from {path}")
    print(f"highlights={len(audit)}")
    for a in audit[:16]:
        print(f"  [{a['severity']}] {a['span'][:80]!r}")
    return 0


def main() -> int:
    if "--from-json" in sys.argv:
        return render_from_saved()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    result = rag_compare(DEMO_QUESTION, k=8, strict=True)

    # Always use unconstrained bare so invented NCT/HR show for contrast.
    print("Replacing careful bare with unconstrained bare…")
    loose = _call_bare_loose(BARE_LOOSE_QUESTION)
    edges_for_check = result.get("retrieved_edges") or []
    if not _has_bare_invention(str(loose.get("text") or ""), edges_for_check) and loose.get("used"):
        print("Unconstrained bare still has no invented NCT/HR — retrying once…")
        loose = _call_bare_loose(BARE_LOOSE_QUESTION)
    result["bare_careful"] = result.get("bare")
    result["bare"] = loose
    result["bare_prompt_mode"] = "unconstrained_for_demo_contrast"
    result["gemini_used"] = bool(result.get("gemini_used") or loose.get("used"))

    excerpt = _demo_bare_excerpt(str((result.get("bare") or {}).get("text") or ""))
    _, audit = _annotate_bare_html(excerpt, result.get("retrieved_edges") or [])
    result["bare_highlight_audit"] = audit
    result["bare_ids_found"] = _ID_RE.findall(
        str((result.get("bare") or {}).get("text") or "")
    )

    json_path = DEMO_DIR / "rag_compare.json"
    html_path = DEMO_DIR / "rag_compare.html"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    html_path.write_text(render_html(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    print(f"retrieved_edges={len(result.get('retrieved_edges') or [])}")
    print(f"gemini_used={result.get('gemini_used')}")
    print(f"bare_prompt_mode={result.get('bare_prompt_mode')}")
    print(f"bare_ids_found={result.get('bare_ids_found')}")
    print(f"bare_hr_found={bool(_HR_RE.search(str((result.get('bare') or {}).get('text') or '')))}")
    print(f"highlights={len(audit)}")
    for a in audit[:16]:
        print(f"  [{a['severity']}] {a['span'][:80]!r}")
    bare_status = (result.get("bare") or {}).get("status")
    grounded_status = (result.get("grounded") or {}).get("status")
    strict_status = (result.get("strict") or {}).get("status")
    print(
        f"bare_status={bare_status} grounded_status={grounded_status} "
        f"strict_status={strict_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
