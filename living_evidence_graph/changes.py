"""Human-readable change digest between two graph snapshots (pure functions).

Public demo only — no PHI, no causation claims. Provenance comes only from
edge/node payloads (never invent NCT/PMID/setid/ChEMBL/Open Targets IDs).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypedDict

from living_evidence_graph.config import DEMO_DIR, GRAPH_DIR
from living_evidence_graph.schema import SOURCE_FAMILY

ChangeWhat = Literal["added", "updated", "retracted_or_downgraded", "trust_shift"]
ChangeKind = Literal["edge", "node"]

TRUST_SHIFT_EPS = 0.02


class ChangeEvent(TypedDict, total=False):
    id: str
    kind: ChangeKind
    what: ChangeWhat
    why: str
    edge_or_node_ref: str
    sources: list[str]
    evidence_urls: list[str]
    trust_before: float | None
    trust_after: float | None
    at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def snapshot_path(goal_slug: str = "default") -> Path:
    _ensure_dir(GRAPH_DIR)
    return GRAPH_DIR / f"{goal_slug}.prev.json"


def digest_path(goal_slug: str = "default") -> Path:
    _ensure_dir(GRAPH_DIR)
    return GRAPH_DIR / f"{goal_slug}.changes.json"


def demo_digest_path() -> Path:
    _ensure_dir(DEMO_DIR)
    return DEMO_DIR / "change_digest.json"


def load_snapshot(goal_slug: str = "default") -> dict[str, Any] | None:
    path = snapshot_path(goal_slug)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(doc: Mapping[str, Any], goal_slug: str = "default") -> Path:
    """Persist a previous-graph snapshot for the next refresh diff."""
    path = snapshot_path(goal_slug)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(doc), f, indent=2, ensure_ascii=False)
    return path


def _index_by_id(items: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items or []:
        eid = item.get("id")
        if eid:
            out[str(eid)] = dict(item)
    return out


def _family_tags(sources: Sequence[str] | None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for s in sources or []:
        fam = SOURCE_FAMILY.get(s, s)
        if fam and fam not in seen:
            seen.add(fam)
            tags.append(fam)
    return tags


def _collect_urls(item: Mapping[str, Any]) -> list[str]:
    """Collect evidence URLs already present on the payload — never invent."""
    urls: list[str] = []
    seen: set[str] = set()

    def _add(u: Any) -> None:
        if isinstance(u, str):
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)

    for u in item.get("evidence_urls") or []:
        _add(u)
    props = item.get("props") or {}
    if isinstance(props, Mapping):
        _add(props.get("url"))
        for key in ("evidence_urls", "urls"):
            for u in props.get(key) or []:
                _add(u)
    return urls


_ID_PROP_KEYS = (
    "nct_id",
    "pmid",
    "pmcid",
    "setid",
    "chembl_id",
    "molecule_chembl_id",
    "target_chembl_id",
    "ensembl_id",
    "ot_id",
    "open_targets_id",
    "drug_id",
    "disease_id",
    "target_id",
)


def _collect_id_refs(item: Mapping[str, Any]) -> list[str]:
    """Surface provenance IDs already on the payload (NCT, PMID, setid, …)."""
    refs: list[str] = []
    seen: set[str] = set()

    def _add(v: Any) -> None:
        if v is None:
            return
        s = str(v).strip()
        if not s or s in seen:
            return
        # Normalize pubmed URL fragments accidentally captured as IDs
        m_pm = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", s, re.I)
        if m_pm:
            s = m_pm.group(1)
        m_set = re.search(r"setid=([a-f0-9-]+)", s, re.I)
        if m_set:
            s = m_set.group(1)
        if s in seen:
            return
        seen.add(s)
        refs.append(s)

    props = item.get("props") or {}
    if isinstance(props, Mapping):
        for k in _ID_PROP_KEYS:
            _add(props.get(k))
    eid = str(item.get("id") or "")
    for m in re.finditer(r"(NCT\d+|PMID:?\d+|CHEMBL\d+|ENSG\d+)", eid, re.I):
        _add(m.group(1))
    # Numeric PMID often embedded in cites/supports edge ids: edge:cites:27216199:...
    for m in re.finditer(r"edge:(?:cites|supports):(\d+)", eid, re.I):
        _add(m.group(1))
    # Evidence URLs already listed separately — do not duplicate every ID into sources.
    return refs


def _sources_payload(item: Mapping[str, Any]) -> list[str]:
    """Source family tags (+ any raw tags) from the edge/node payload."""
    raw = list(item.get("sources") or [])
    families = _family_tags(raw)
    # Prefer family tags for contest copy; keep raw if no mapping
    if families:
        return families
    return [str(s) for s in raw]


def _event_id(kind: str, what: str, ref: str, at: str) -> str:
    h = hashlib.sha1(f"{kind}|{what}|{ref}|{at}".encode()).hexdigest()[:12]
    return f"chg:{what}:{h}"


def _why_added(item: Mapping[str, Any], kind: ChangeKind) -> str:
    etype = str(item.get("type") or "")
    sources = _sources_payload(item)
    if kind == "node":
        if etype == "Trial":
            return "new trial registered"
        if etype == "Publication":
            return "new publication indexed"
        if etype == "Gene":
            return "new target gene linked"
        return f"new {etype or 'node'} added"
    if etype == "studied_in":
        return "new trial registered"
    if etype == "supports":
        return "new supporting evidence edge"
    if etype == "contradicts":
        return "contradict edge"
    if etype == "warns_ae":
        return "label update"
    if etype == "treats_indication":
        if "dailymed" in sources:
            return "label update"
        return "new indication corroboration"
    if etype in ("drug_targets_gene", "gene_associated_with_disease", "drug_indicated_for_disease"):
        return "new drug–target–disease spine edge"
    if etype == "reports_ae":
        return "new adverse-event report term (reports, not rates)"
    if "clinicaltrials" in sources:
        return "new trial registered"
    if "dailymed" in sources:
        return "label update"
    if "europepmc" in sources or "pubmed" in sources:
        return "new literature evidence"
    if "opentargets" in sources or "chembl" in sources:
        return "new knowledge-base link"
    return f"new {etype or 'edge'} added"


def _why_updated(prev: Mapping[str, Any], nxt: Mapping[str, Any]) -> str:
    prev_src = set(prev.get("sources") or [])
    next_src = set(nxt.get("sources") or [])
    if next_src - prev_src:
        return "corroboration increased"
    if prev_src - next_src:
        return "source set narrowed"
    if (nxt.get("type") or "") == "contradicts" or (prev.get("type") or "") == "contradicts":
        return "contradict edge"
    prev_urls = set(prev.get("evidence_urls") or [])
    next_urls = set(nxt.get("evidence_urls") or [])
    if next_urls - prev_urls:
        return "evidence URLs updated"
    if (nxt.get("label") or "") != (prev.get("label") or ""):
        return "label update"
    props_p = prev.get("props") or {}
    props_n = nxt.get("props") or {}
    if isinstance(props_p, Mapping) and isinstance(props_n, Mapping):
        if props_p.get("overall_status") != props_n.get("overall_status"):
            return "trial status update"
    return "edge or node fields updated"


def _why_retracted(item: Mapping[str, Any]) -> str:
    sources = _sources_payload(item)
    if "europepmc" in sources or "pubmed" in sources:
        return "literature retraction"
    if item.get("type") == "contradicts":
        return "contradict edge"
    return "retracted or trust downgraded"


def _why_trust_shift(before: float | None, after: float | None, item: Mapping[str, Any]) -> str:
    b = before if before is not None else 0.0
    a = after if after is not None else 0.0
    delta = a - b
    sources = _sources_payload(item)
    if delta > 0 and len(sources) >= 2:
        return "corroboration increased"
    if delta > 0:
        return "trust increased"
    if item.get("retracted"):
        return "literature retraction"
    if "contradict" in str(item.get("type") or "").lower():
        return "contradict edge"
    if delta < 0:
        return "trust decreased"
    return "trust shift"


def _trust(item: Mapping[str, Any] | None) -> float | None:
    if not item:
        return None
    v = item.get("trust_score")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _significant_trust_shift(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return abs(after - before) >= TRUST_SHIFT_EPS


def _payload_changed(prev: Mapping[str, Any], nxt: Mapping[str, Any]) -> bool:
    """Compare substantive fields (ignore volatile timestamps / trust recomputes)."""
    keys = ("type", "source", "target", "label", "retracted", "sources", "evidence_urls", "props")
    for k in keys:
        if prev.get(k) != nxt.get(k):
            return True
    return False


def _make_event(
    *,
    kind: ChangeKind,
    what: ChangeWhat,
    why: str,
    item: Mapping[str, Any],
    trust_before: float | None = None,
    trust_after: float | None = None,
    at: str | None = None,
) -> ChangeEvent:
    ts = at or _now_iso()
    ref = str(item.get("id") or "unknown")
    sources = _sources_payload(item)
    urls = _collect_urls(item)
    # Attach compact id: tags from props / entity id only (URLs already carry IDs).
    id_refs = _collect_id_refs(item)
    provenance_sources = list(sources)
    for rid in id_refs[:6]:
        tag = f"id:{rid}"
        if tag not in provenance_sources:
            provenance_sources.append(tag)
    return ChangeEvent(
        id=_event_id(kind, what, ref, ts),
        kind=kind,
        what=what,
        why=why,
        edge_or_node_ref=ref,
        sources=provenance_sources,
        evidence_urls=urls,
        trust_before=trust_before,
        trust_after=trust_after,
        at=ts,
    )


def diff(
    prev_graph: Mapping[str, Any] | None,
    next_graph: Mapping[str, Any] | None,
    *,
    at: str | None = None,
) -> list[ChangeEvent]:
    """Compare two graph docs and return a list of ChangeEvent records.

    Categories:
      - added: present only in next
      - retracted_or_downgraded: removed, newly retracted, or large trust drop
      - trust_shift: trust moved without other substantive payload change
      - updated: other substantive field changes
    """
    ts = at or _now_iso()
    prev = prev_graph or {}
    nxt = next_graph or {}
    events: list[ChangeEvent] = []

    for kind, key in (("node", "nodes"), ("edge", "edges")):
        prev_map = _index_by_id(prev.get(key))
        next_map = _index_by_id(nxt.get(key))
        prev_ids = set(prev_map)
        next_ids = set(next_map)

        for eid in sorted(next_ids - prev_ids):
            item = next_map[eid]
            events.append(
                _make_event(
                    kind=kind,  # type: ignore[arg-type]
                    what="added",
                    why=_why_added(item, kind),  # type: ignore[arg-type]
                    item=item,
                    trust_before=None,
                    trust_after=_trust(item),
                    at=ts,
                )
            )

        for eid in sorted(prev_ids - next_ids):
            item = prev_map[eid]
            events.append(
                _make_event(
                    kind=kind,  # type: ignore[arg-type]
                    what="retracted_or_downgraded",
                    why=_why_retracted(item),
                    item=item,
                    trust_before=_trust(item),
                    trust_after=None,
                    at=ts,
                )
            )

        for eid in sorted(prev_ids & next_ids):
            p, n = prev_map[eid], next_map[eid]
            tb, ta = _trust(p), _trust(n)
            newly_retracted = bool(n.get("retracted")) and not bool(p.get("retracted"))
            large_drop = (
                tb is not None
                and ta is not None
                and (tb - ta) >= 0.15
            )
            if newly_retracted or large_drop:
                events.append(
                    _make_event(
                        kind=kind,  # type: ignore[arg-type]
                        what="retracted_or_downgraded",
                        why=_why_retracted(n if newly_retracted else n),
                        item=n,
                        trust_before=tb,
                        trust_after=ta,
                        at=ts,
                    )
                )
                continue
            if _payload_changed(p, n):
                why = _why_updated(p, n)
                # If sources grew, prefer corroboration wording even with trust shift
                events.append(
                    _make_event(
                        kind=kind,  # type: ignore[arg-type]
                        what="updated",
                        why=why,
                        item=n,
                        trust_before=tb,
                        trust_after=ta,
                        at=ts,
                    )
                )
                continue
            if _significant_trust_shift(tb, ta):
                events.append(
                    _make_event(
                        kind=kind,  # type: ignore[arg-type]
                        what="trust_shift",
                        why=_why_trust_shift(tb, ta, n),
                        item=n,
                        trust_before=tb,
                        trust_after=ta,
                        at=ts,
                    )
                )

    return events


def digest_document(
    events: Sequence[ChangeEvent],
    *,
    goal: str | None = None,
    goal_slug: str | None = None,
    prev_path: str | None = None,
    next_path: str | None = None,
) -> dict[str, Any]:
    """Wrap events into a contest-friendly digest artifact."""
    by_what: dict[str, int] = {}
    for e in events:
        w = str(e.get("what") or "unknown")
        by_what[w] = by_what.get(w, 0) + 1
    return {
        "title": "Living Evidence Graph — change digest",
        "goal": goal,
        "goal_slug": goal_slug,
        "generated_at": _now_iso(),
        "disclaimer": (
            "Public data only. No PHI. No causation claims. "
            "openFDA FAERS = voluntary reports, not rates. "
            "Provenance IDs/URLs are copied from graph payloads only — never invented."
        ),
        "prev_snapshot": prev_path,
        "next_graph": next_path,
        "change_count": len(events),
        "by_what": by_what,
        "changes": list(events),
    }


def write_digest(
    doc: Mapping[str, Any],
    *,
    goal_slug: str = "default",
    also_demo: bool = True,
) -> dict[str, Path]:
    """Persist digest under out/graph/ and optionally out/demo/change_digest.json."""
    paths: dict[str, Path] = {}
    gpath = digest_path(goal_slug)
    _ensure_dir(gpath.parent)
    with gpath.open("w", encoding="utf-8") as f:
        json.dump(dict(doc), f, indent=2, ensure_ascii=False)
    paths["graph"] = gpath
    if also_demo:
        dpath = demo_digest_path()
        with dpath.open("w", encoding="utf-8") as f:
            json.dump(dict(doc), f, indent=2, ensure_ascii=False)
        paths["demo"] = dpath
        md_path = DEMO_DIR / "change_digest.md"
        md_path.write_text(_to_markdown(doc), encoding="utf-8")
        paths["demo_md"] = md_path
    return paths


def _to_markdown(doc: Mapping[str, Any]) -> str:
    lines = [
        "# Change digest",
        "",
        f"**Goal:** {doc.get('goal') or '—'}",
        f"**Generated (UTC):** {doc.get('generated_at')}",
        f"**Changes:** {doc.get('change_count', 0)}",
        "",
        str(doc.get("disclaimer") or ""),
        "",
        "| what | why | ref | sources |",
        "|---|---|---|---|",
    ]
    for e in doc.get("changes") or []:
        sources = ", ".join(e.get("sources") or [])
        lines.append(
            f"| {e.get('what')} | {e.get('why')} | `{e.get('edge_or_node_ref')}` | {sources} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_and_persist_digest(
    prev_graph: Mapping[str, Any] | None,
    next_graph: Mapping[str, Any],
    *,
    goal_slug: str = "default",
    also_demo: bool = True,
) -> dict[str, Any]:
    """Diff, wrap, persist snapshot+digest. Returns the digest document."""
    events = diff(prev_graph, next_graph)
    # After a successful next graph, previous snapshot should be the *prior* next
    # (caller typically saves prev before overwrite). We still refresh snapshot here
    # only when prev was provided and next is the new truth — caller owns ordering.
    digest = digest_document(
        events,
        goal=str(next_graph.get("goal") or goal_slug),
        goal_slug=goal_slug,
        prev_path=str(snapshot_path(goal_slug)) if prev_graph is not None else None,
        next_path=str(GRAPH_DIR / f"{goal_slug}.json"),
    )
    write_digest(digest, goal_slug=goal_slug, also_demo=also_demo)
    return digest


def load_digest(goal_slug: str = "default") -> dict[str, Any] | None:
    path = digest_path(goal_slug)
    demo = demo_digest_path()
    for p in (path, demo):
        if p.exists():
            with p.open(encoding="utf-8") as f:
                return json.load(f)
    return None
