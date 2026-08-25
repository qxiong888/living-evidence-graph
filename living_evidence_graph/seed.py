"""Seed the baked Keytruda/NSCLC demo graph into GRAPH_DIR on cold start.

Cloud Run sets LEG_OUT_DIR=/tmp/leg-out, which is empty on a new instance.
The 14-node / 10-edge public demo lives under fixtures/demo_graph/ (copied
into the image). If GRAPH_DIR is missing that slug file, copy the baked
files there. GRAPH_DIR stays writable so the scheduler can persist updates.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from living_evidence_graph.config import (
    DEMO_GRAPH_FIXTURES_DIR,
    DEMO_GRAPH_SLUG,
    GRAPH_DIR,
)


def seed_demo_graph_if_missing(*, graph_dir: Path | None = None) -> dict[str, Any]:
    """Copy fixtures/demo_graph/* into graph_dir when the demo slug file is absent."""
    dest_dir = Path(graph_dir) if graph_dir is not None else GRAPH_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{DEMO_GRAPH_SLUG}.json"
    if dest.is_file():
        return {
            "seeded": False,
            "reason": "already_present",
            "path": str(dest),
            "slug": DEMO_GRAPH_SLUG,
            "copied": [],
        }
    src_dir = DEMO_GRAPH_FIXTURES_DIR
    if not src_dir.is_dir():
        return {
            "seeded": False,
            "reason": "no_baked_fixtures",
            "path": None,
            "slug": DEMO_GRAPH_SLUG,
            "copied": [],
        }
    copied: list[str] = []
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        target = dest_dir / src.name
        if target.exists():
            continue
        shutil.copy2(src, target)
        copied.append(src.name)
    return {
        "seeded": bool(copied) and dest.is_file(),
        "reason": "copied" if dest.is_file() else "copy_incomplete",
        "path": str(dest) if dest.is_file() else None,
        "slug": DEMO_GRAPH_SLUG,
        "copied": copied,
    }
