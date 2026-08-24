#!/usr/bin/env python3
"""Contest RAG demo: Keytruda/NSCLC question → bare vs graph-grounded Gemini.

Writes out/demo/rag_compare.html + rag_compare.json.
Works without a Gemini key (answers labeled gemini_skipped; retrieval still runs).
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from living_evidence_graph.config import DEMO_DIR  # noqa: E402
from living_evidence_graph.rag import DISCLAIMER, rag_compare  # noqa: E402

# Fixed judge-facing question (Keytruda / NSCLC / evidence graph).
DEMO_QUESTION = (
    "What high-trust public evidence links Keytruda (pembrolizumab) to NSCLC, "
    "including drug–target–disease spine edges and labeled warnings or FAERS reports? "
    "Do not invent IDs or claim causation or incidence rates."
)


def _answer_block(payload: dict) -> str:
    status = html.escape(str(payload.get("status") or ""))
    model = html.escape(str(payload.get("model") or ""))
    text = html.escape(str(payload.get("text") or ""))
    used = "yes" if payload.get("used") else "no"
    return (
        f"<p class='meta'>status=<code>{status}</code> · model=<code>{model}</code> · "
        f"gemini_used={used}</p>"
        f"<pre class='answer'>{text}</pre>"
    )


def _edges_table(edges: list) -> str:
    if not edges:
        return "<p><em>No edges retrieved.</em></p>"
    rows = []
    for e in edges:
        spine = "★" if e.get("triangle_spine") else ""
        urls = e.get("evidence_urls") or []
        url_html = "<br/>".join(
            f'<a href="{html.escape(u)}" rel="noopener noreferrer">{html.escape(u)}</a>'
            for u in urls[:3]
        ) or "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(e.get('type')))} {spine}</td>"
            f"<td>{html.escape(str(e.get('source_label')))} → "
            f"{html.escape(str(e.get('target_label')))}</td>"
            f"<td>{html.escape(str(e.get('trust_score')))}</td>"
            f"<td>{html.escape(', '.join(e.get('sources') or []))}</td>"
            f"<td>{url_html}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Edge type</th><th>Endpoints</th><th>Trust</th><th>Sources</th><th>Evidence</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<p class='note'>★ = triangle spine (drug–target–disease). "
        "openFDA / reports_ae = voluntary reports ≠ rates.</p>"
    )


def render_html(result: dict) -> str:
    q = html.escape(result.get("question") or "")
    disc = html.escape(result.get("disclaimer") or DISCLAIMER)
    gp = html.escape(str(result.get("graph_path") or "(none)"))
    ctx = html.escape(result.get("context") or "")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    gemini = "used" if result.get("gemini_used") else "skipped / unavailable"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>RAG compare — Living Evidence Graph</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 1100px;
         color: #122; background: #fafbfc; }}
  h1 {{ font-size: 1.35rem; }}
  .banner {{ background: #fff3cd; border: 1px solid #e0c36a; padding: 0.75rem 1rem;
             border-radius: 6px; margin-bottom: 1rem; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: #fff; border: 1px solid #ccd; border-radius: 8px; padding: 1rem; }}
  .card h2 {{ margin-top: 0; font-size: 1.05rem; }}
  .bare {{ border-top: 4px solid #c45; }}
  .grounded {{ border-top: 4px solid #2a7; }}
  pre.answer, pre.ctx {{ white-space: pre-wrap; font-size: 0.85rem;
                         background: #f4f6f8; padding: 0.75rem; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
  th, td {{ border: 1px solid #ccd; padding: 0.35rem 0.5rem; vertical-align: top; }}
  th {{ background: #eef2f6; }}
  .meta {{ font-size: 0.8rem; color: #456; }}
  .note {{ font-size: 0.8rem; color: #456; }}
  code {{ font-size: 0.85em; }}
</style>
</head>
<body>
  <div class="banner"><strong>Disclaimer:</strong> {disc}</div>
  <h1>Living Evidence Graph — RAG retrieval demo</h1>
  <p class="meta">Generated {ts} · graph=<code>{gp}</code> · Gemini {gemini} ·
  retrieval-only (no fine-tuning)</p>
  <div class="card">
    <h2>Judge question</h2>
    <p>{q}</p>
  </div>
  <h2>Retrieved high-trust edges</h2>
  {_edges_table(result.get("retrieved_edges") or [])}
  <details>
    <summary>Compact context block sent to grounded Gemini</summary>
    <pre class="ctx">{ctx}</pre>
  </details>
  <div class="grid" style="margin-top:1rem">
    <div class="card bare">
      <h2>Bare Gemini (no graph)</h2>
      {_answer_block(result.get("bare") or {})}
    </div>
    <div class="card grounded">
      <h2>Grounded Gemini (graph edges only)</h2>
      {_answer_block(result.get("grounded") or {})}
    </div>
  </div>
  <p class="note">Attribution: Open Targets (CC0); ChEMBL (CC BY-SA 3.0, Mendez et al. NAR 2019);
  ClinicalTrials.gov / DailyMed / PubMed courtesy of NLM — not endorsed by NLM/NIH/NCBI/FDA.
  Reports ≠ rates. Not medical advice.</p>
</body>
</html>
"""


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    result = rag_compare(DEMO_QUESTION, k=8)
    json_path = DEMO_DIR / "rag_compare.json"
    html_path = DEMO_DIR / "rag_compare.html"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    html_path.write_text(render_html(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    print(f"retrieved_edges={len(result.get('retrieved_edges') or [])}")
    print(f"gemini_used={result.get('gemini_used')}")
    bare_status = (result.get("bare") or {}).get("status")
    grounded_status = (result.get("grounded") or {}).get("status")
    print(f"bare_status={bare_status} grounded_status={grounded_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
