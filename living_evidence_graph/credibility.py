"""Edge credibility / trust scoring (pure functions).

Formula (see docs/CREDIBILITY.md):
  trust = clip(0, 1,
    0.35*source_tier + 0.30*corroboration + 0.20*recency + 0.15*consistency
    - retraction_penalty)
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from living_evidence_graph.schema import SOURCE_FAMILY, SOURCE_TIERS, sources_from_evidence_urls

WEIGHT_SOURCE = 0.35
WEIGHT_CORROBORATION = 0.30
WEIGHT_RECENCY = 0.20
WEIGHT_CONSISTENCY = 0.15
RETRACTION_PENALTY = 0.5


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def source_tier(sources: Sequence[str] | None) -> float:
    """Max known tier among source family tags; unknown → 0.3."""
    if not sources:
        return 0.3
    scores = [SOURCE_TIERS.get(s, 0.3) for s in sources]
    return max(scores)


def corroboration(sources: Sequence[str] | None) -> float:
    """min(1, distinct_source_families / 3)."""
    if not sources:
        return 0.0
    families = {SOURCE_FAMILY.get(s, s) for s in sources}
    return min(1.0, len(families) / 3.0)


def recency(age_days: float | None) -> float:
    """exp(-age_days / 365). Missing age → 0.5 (neutral)."""
    if age_days is None:
        return 0.5
    age = max(0.0, float(age_days))
    return math.exp(-age / 365.0)


def consistency(has_contradict: bool) -> float:
    return 0.4 if has_contradict else 1.0


def retraction_penalty(retracted: bool) -> float:
    return RETRACTION_PENALTY if retracted else 0.0


def score_edge(
    *,
    sources: Sequence[str] | None = None,
    age_days: float | None = None,
    has_contradict: bool = False,
    retracted: bool = False,
) -> dict[str, Any]:
    """Compute trust_score and trust_breakdown for one edge."""
    st = source_tier(sources)
    corr = corroboration(sources)
    rec = recency(age_days)
    cons = consistency(has_contradict)
    pen = retraction_penalty(retracted)
    raw = (
        WEIGHT_SOURCE * st
        + WEIGHT_CORROBORATION * corr
        + WEIGHT_RECENCY * rec
        + WEIGHT_CONSISTENCY * cons
        - pen
    )
    trust = clip01(raw)
    return {
        "trust_score": round(trust, 4),
        "trust_breakdown": {
            "source_tier": round(st, 4),
            "corroboration": round(corr, 4),
            "recency": round(rec, 4),
            "consistency": round(cons, 4),
            "retraction_penalty": round(pen, 4),
            "weights": {
                "source_tier": WEIGHT_SOURCE,
                "corroboration": WEIGHT_CORROBORATION,
                "recency": WEIGHT_RECENCY,
                "consistency": WEIGHT_CONSISTENCY,
            },
            "raw_before_clip": round(raw, 4),
        },
    }


def recompute_edges(
    edges: Iterable[Mapping[str, Any]],
    *,
    contradict_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return new edge dicts with updated trust fields.

    contradict_pairs: optional set of (source, target) that have a contradicts edge.
    If omitted, inferred from edges of type 'contradicts'.
    """
    edge_list = [dict(e) for e in edges]
    pairs = contradict_pairs
    if pairs is None:
        pairs = {
            (str(e.get("source")), str(e.get("target")))
            for e in edge_list
            if e.get("type") == "contradicts"
        }
        # also reverse so either direction flags inconsistency
        pairs |= {(b, a) for a, b in list(pairs)}

    out: list[dict[str, Any]] = []
    for e in edge_list:
        src, tgt = str(e.get("source")), str(e.get("target"))
        has_c = (src, tgt) in pairs and e.get("type") != "contradicts"
        # For the contradicts edge itself, consistency is still reduced
        if e.get("type") == "contradicts":
            has_c = True
        srcs = [str(s) for s in (e.get("sources") or []) if s]
        if not srcs:
            srcs = sources_from_evidence_urls(list(e.get("evidence_urls") or []))
            if srcs:
                e["sources"] = srcs
        scored = score_edge(
            sources=srcs,
            age_days=e.get("age_days"),
            has_contradict=has_c,
            retracted=bool(e.get("retracted")),
        )
        e["trust_score"] = scored["trust_score"]
        e["trust_breakdown"] = scored["trust_breakdown"]
        out.append(e)
    return out


def _demo() -> None:
    """CLI sanity check: python -m living_evidence_graph.credibility"""
    example = score_edge(
        sources=["clinicaltrials_registry", "pubmed_peer_reviewed"],
        age_days=120,
        has_contradict=False,
        retracted=False,
    )
    print("example_edge_trust=", example)


if __name__ == "__main__":
    _demo()
