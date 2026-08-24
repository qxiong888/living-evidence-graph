#!/usr/bin/env python3
"""One-pass local demo: pembrolizumab / Keytruda NSCLC evidence graph.

Attempts live fetches for all seven public APIs with timeout.
Existing three (CT / PubMed / openFDA) may fall back to labeled fixtures.
New four (DailyMed / Europe PMC / Open Targets / ChEMBL): on failure → skip
and label source_mode (never invent IDs).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from living_evidence_graph.config import (  # noqa: E402
    DEMO_CONDITION,
    DEMO_DIR,
    DEMO_DRUG_BRAND,
    DEMO_DRUG_INGREDIENT,
    DEMO_GOAL,
    FIXTURES_DIR,
    HTTP_TIMEOUT,
)
from living_evidence_graph.credibility import recompute_edges  # noqa: E402
from living_evidence_graph.extract import extract_from_sources  # noqa: E402
from living_evidence_graph.graph_store import upsert_graph  # noqa: E402
from living_evidence_graph.schema import TRIANGLE_EDGE_TYPES  # noqa: E402
from living_evidence_graph.tools.fetch_chembl import fetch_chembl  # noqa: E402
from living_evidence_graph.tools.fetch_clinicaltrials import fetch_clinicaltrials  # noqa: E402
from living_evidence_graph.tools.fetch_dailymed import fetch_dailymed  # noqa: E402
from living_evidence_graph.tools.fetch_europepmc import fetch_europepmc_status  # noqa: E402
from living_evidence_graph.tools.fetch_openfda import fetch_openfda_events  # noqa: E402
from living_evidence_graph.tools.fetch_opentargets import fetch_opentargets  # noqa: E402
from living_evidence_graph.tools.fetch_pubmed import fetch_pubmed  # noqa: E402


ATTRIBUTION_BLOCK = {
    "open_targets": (
        "Open Targets Platform — CC0 1.0. No endorsement by Open Targets / partners."
    ),
    "chembl": (
        "ChEMBL — CC BY-SA 3.0. Cite: Mendez et al., Nucleic Acids Res. 2019; "
        "doi:10.1093/nar/gky1075. ChEMBL-derived graph subsets are ShareAlike. "
        "No EMBL-EBI / ChEMBL endorsement."
    ),
    "openfda": (
        "openFDA FAERS voluntary reports (FDA) — not rates, not causation; "
        "not an FDA endorsement."
    ),
    "nlm_courtesy": (
        "ClinicalTrials.gov, DailyMed, and PubMed: Courtesy of the "
        "U.S. National Library of Medicine. Not endorsed by NLM/NIH/NCBI."
    ),
    "europepmc": (
        "Europe PMC: per-article licenses apply to full text; this demo stores "
        "status/metadata signals only (no non-OA full-text dump)."
    ),
    "ncbi": (
        "NCBI disclaimer: use of NCBI information does not imply endorsement "
        "(https://www.ncbi.nlm.nih.gov/home/about/policies/)."
    ),
    "llm_path": (
        "LLM path is retrieval-only over the graph. Do not dump PubMed abstracts "
        "or non-OA Europe PMC full text into training corpora."
    ),
}


def _load_fixtures() -> dict:
    path = FIXTURES_DIR / "keytruda_nsclc.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _pick(live: dict, fixture_block: dict, key_list: str) -> tuple[dict, str]:
    """Prefer live non-empty; else fixture. Only for CT/PubMed/openFDA."""
    live_ok = bool(live.get("ok"))
    if key_list == "studies":
        items = live.get("studies") or []
    elif key_list == "publications":
        items = live.get("publications") or []
    elif key_list == "reactions":
        items = live.get("reactions") or []
    else:
        items = []

    if live_ok and items:
        return live, "live"
    merged = dict(fixture_block)
    merged["live_attempt"] = {
        "ok": live.get("ok"),
        "error": live.get("error"),
        "empty": live.get("empty"),
    }
    return merged, "fixture_fallback"


def _live_or_skip(live: dict, nonempty_keys: list[str]) -> tuple[dict, str]:
    """New sources: live | skipped | error (never invent IDs on failure)."""
    if not live.get("ok"):
        return {
            "ok": False,
            "error": live.get("error"),
            "source": live.get("source"),
            "attribution": live.get("attribution"),
        }, "error"
    for k in nonempty_keys:
        if live.get(k):
            return live, "live"
    if live.get("empty"):
        return {**live, "skipped": True}, "skipped"
    return live, "live"


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    goal = DEMO_GOAL
    brand, ingredient, condition = DEMO_DRUG_BRAND, DEMO_DRUG_INGREDIENT, DEMO_CONDITION
    fixtures = _load_fixtures()

    print(f"[demo] goal={goal!r}")
    print(f"[demo] attempting live APIs (timeout={HTTP_TIMEOUT}s)…")

    ct_live = fetch_clinicaltrials(
        f"{ingredient} OR {brand} {condition}", page_size=5, timeout=HTTP_TIMEOUT
    )
    pm_live = fetch_pubmed(f"{ingredient} AND {condition}", retmax=5, timeout=HTTP_TIMEOUT)
    fda_live = fetch_openfda_events(brand, limit=5, timeout=HTTP_TIMEOUT)
    if not fda_live.get("ok") or fda_live.get("empty"):
        fda_ing = fetch_openfda_events(ingredient, limit=5, timeout=HTTP_TIMEOUT)
        if fda_ing.get("ok") and not fda_ing.get("empty"):
            fda_live = fda_ing

    dm_live = fetch_dailymed(brand, pagesize=5, timeout=HTTP_TIMEOUT)
    if not dm_live.get("ok") or dm_live.get("empty"):
        dm_ing = fetch_dailymed(ingredient, pagesize=5, timeout=HTTP_TIMEOUT)
        if dm_ing.get("ok") and not dm_ing.get("empty"):
            dm_live = dm_ing

    ct, ct_src = _pick(ct_live, fixtures["clinicaltrials"], "studies")
    pm, pm_src = _pick(pm_live, fixtures["pubmed"], "publications")
    fda, fda_src = _pick(fda_live, fixtures["openfda"], "reactions")

    pmids = [p.get("pmid") for p in (pm.get("publications") or []) if p.get("pmid")]
    epmc_live = fetch_europepmc_status(pmids, timeout=HTTP_TIMEOUT)
    ot_live = fetch_opentargets(ingredient, condition_hint=condition, timeout=HTTP_TIMEOUT)
    ch_live = fetch_chembl(ingredient, timeout=HTTP_TIMEOUT)

    dm, dm_src = _live_or_skip(dm_live, ["labels", "preferred"])
    epmc, epmc_src = _live_or_skip(epmc_live, ["publications"])
    ot, ot_src = _live_or_skip(ot_live, ["targets", "indications", "gene_disease"])
    ch, ch_src = _live_or_skip(ch_live, ["mechanisms", "molecule_chembl_id"])

    print(f"[demo] clinicaltrials={ct_src} studies={len(ct.get('studies') or [])}")
    print(f"[demo] pubmed={pm_src} pubs={len(pm.get('publications') or [])}")
    print(
        f"[demo] openfda={fda_src} reactions={len(fda.get('reactions') or [])} "
        f"total_reports={fda.get('total_reports')!r}"
    )
    def _err(d: dict) -> str:
        return f" err={d.get('error')!r}" if d.get("error") else ""

    print(f"[demo] dailymed={dm_src} labels={len(dm.get('labels') or [])}{_err(dm)}")
    print(f"[demo] europepmc={epmc_src} pubs={len(epmc.get('publications') or [])}{_err(epmc)}")
    print(
        f"[demo] opentargets={ot_src} targets={len(ot.get('targets') or [])} "
        f"indications={len(ot.get('indications') or [])}{_err(ot)}"
    )
    print(f"[demo] chembl={ch_src} mechanisms={len(ch.get('mechanisms') or [])}{_err(ch)}")

    extracted = extract_from_sources(
        drug_brand=brand,
        drug_ingredient=ingredient,
        condition=condition,
        clinicaltrials=ct,
        pubmed=pm,
        openfda=fda,
        dailymed=dm if not dm.get("skipped") else {},
        europepmc=epmc if not epmc.get("skipped") else {},
        opentargets=ot if not ot.get("skipped") else {},
        chembl=ch if not ch.get("skipped") else {},
        use_gemini=True,
    )
    edges = recompute_edges(extracted.get("edges") or [])
    slug = "pembrolizumab_non_small_cell_lung_cancer"
    source_mode = {
        "clinicaltrials": ct_src,
        "pubmed": pm_src,
        "openfda": fda_src,
        "dailymed": dm_src,
        "europepmc": epmc_src,
        "opentargets": ot_src,
        "chembl": ch_src,
    }
    doc = upsert_graph(
        goal=goal,
        nodes=extracted.get("nodes") or [],
        edges=edges,
        goal_slug=slug,
        meta={
            "demo": True,
            "source_mode": source_mode,
            "gemini": extracted.get("gemini"),
            "disclaimer": extracted.get("disclaimer"),
            "attribution_block": extracted.get("attribution_block") or ATTRIBUTION_BLOCK,
        },
    )

    triangle = [e for e in doc["edges"] if e.get("type") in TRIANGLE_EDGE_TYPES]
    other = [e for e in doc["edges"] if e.get("type") not in TRIANGLE_EDGE_TYPES]

    def _edge_view(e: dict) -> dict:
        return {
            "id": e.get("id"),
            "type": e.get("type"),
            "source": e.get("source"),
            "target": e.get("target"),
            "trust_score": e.get("trust_score"),
            "trust_breakdown": e.get("trust_breakdown"),
            "sources": e.get("sources"),
            "evidence_urls": e.get("evidence_urls"),
            "retracted": e.get("retracted"),
            "props": e.get("props"),
        }

    card = {
        "title": "Living Evidence Graph — demo card",
        "goal": goal,
        "drug": {"brand": brand, "ingredient": ingredient},
        "condition": condition,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "core_motif": {
            "name": "drug–target–disease triangle",
            "edge_types": list(TRIANGLE_EDGE_TYPES),
            "spine_sources": ["opentargets", "chembl"],
            "corroboration_sources": [
                "clinicaltrials",
                "pubmed",
                "openfda",
                "dailymed",
                "europepmc",
            ],
            "edges": [_edge_view(e) for e in triangle],
        },
        "disclaimer": (
            "Public data only. No PHI. No causation claims. "
            "openFDA FAERS = voluntary reports, not rates. "
            "No FDA/NLM/Open Targets/ChEMBL endorsement. "
            "LLM = retrieval-only over the graph (no abstract/full-text training dumps). "
            "Fixture blocks are clearly labeled when used; new sources skip on failure."
        ),
        "attribution": ATTRIBUTION_BLOCK,
        "node_count": len(doc["nodes"]),
        "edge_count": len(doc["edges"]),
        "triangle_edge_count": len(triangle),
        "nodes": doc["nodes"],
        "edges": [_edge_view(e) for e in [*triangle, *other]],
        "gemini": extracted.get("gemini"),
        "graph_path": (doc.get("meta") or {}).get("path"),
    }

    json_path = DEMO_DIR / "demo_card.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)

    def _rows(edgelist: list[dict]) -> str:
        return "".join(
            f"<tr><td>{e['type']}</td><td>{e['source']} → {e['target']}</td>"
            f"<td>{e.get('trust_score')}</td>"
            f"<td><code>{', '.join(e.get('sources') or [])}</code></td></tr>"
            for e in edgelist
        )

    node_lis = "".join(
        f"<li><strong>{n.get('type')}</strong>: {n.get('label')} "
        f"<code>{n.get('id')}</code></li>"
        for n in card["nodes"]
    )
    attr_lis = "".join(f"<li><strong>{k}</strong>: {v}</li>" for k, v in ATTRIBUTION_BLOCK.items())
    mode_str = ", ".join(f"{k}={v}" for k, v in source_mode.items())

    html_path = DEMO_DIR / "demo_card.html"
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Living Evidence Graph — Keytruda / NSCLC</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:32px auto;color:#122;padding:0 16px}}
.banner{{background:#fff3cd;border:1px solid #e6c200;padding:12px;border-radius:8px;margin:12px 0}}
.spine{{background:#e8f5e9;border:1px solid #81c784;padding:12px;border-radius:8px;margin:12px 0}}
.attr{{background:#fffde7;border:1px solid #fbc02d;padding:12px;border-radius:8px;margin:12px 0;font-size:13px}} /* Elena yellow-light */
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:6px 8px;font-size:14px}}
th{{background:#f4f6f8;text-align:left}} code{{font-size:12px}}
.meta{{color:#456}} h2.spine-h{{color:#1b5e20}}
</style></head><body>
<h1>Living Evidence Graph</h1>
<p class="meta">Demo vertical: <strong>{brand}</strong> ({ingredient}) · {condition}</p>
<div class="banner"><strong>Disclaimer:</strong> {card['disclaimer']}</div>
<div class="attr"><strong>Attribution / licenses</strong> (see LICENSES.md — no FDA/NLM/Open Targets/ChEMBL endorsement)<ul>{attr_lis}</ul></div>
<p>Generated (UTC): {card['generated_at']}<br/>
Source mode: {mode_str}<br/>
Nodes: {card['node_count']} · Edges: {card['edge_count']} ·
Triangle spine edges: {card['triangle_edge_count']}</p>
<div class="spine">
<h2 class="spine-h">Core motif — drug · target · disease</h2>
<p>First-class edges for LLM retrieval:
<code>drug_targets_gene</code> ·
<code>gene_associated_with_disease</code> ·
<code>drug_indicated_for_disease</code>
(Open Targets + ChEMBL). ClinicalTrials / PubMed / openFDA / DailyMed / Europe PMC
are corroboration layers around this spine.</p>
<table><thead><tr><th>Type</th><th>Link</th><th>Trust</th><th>Sources</th></tr></thead>
<tbody>{_rows([_edge_view(e) for e in triangle]) or "<tr><td colspan=4><em>No spine edges this run (sources skipped).</em></td></tr>"}</tbody></table>
</div>
<h2>All nodes</h2><ul>{node_lis}</ul>
<h2>Other edges (corroboration)</h2>
<table><thead><tr><th>Type</th><th>Link</th><th>Trust</th><th>Sources</th></tr></thead>
<tbody>{_rows([_edge_view(e) for e in other])}</tbody></table>
<h2>Gemini note</h2>
<pre>{json.dumps(card.get('gemini'), indent=2)}</pre>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")

    print(f"[demo] wrote {json_path}")
    print(f"[demo] wrote {html_path}")
    print(f"[demo] graph {(doc.get('meta') or {}).get('path')}")
    print(f"[demo] triangle_edges={len(triangle)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
