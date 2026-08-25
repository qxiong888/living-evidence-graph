"""Auto-refresh private folder graphs when files change (no manual re-ingest)."""

from __future__ import annotations

import json
import time

import living_evidence_graph.changes as changes_mod
import living_evidence_graph.graph_store as store_mod
import living_evidence_graph.private_ingest as pi
from living_evidence_graph.config import DEMO_GRAPH_FIXTURES_DIR, DEMO_GRAPH_SLUG


def _patch_graph_dirs(tmp_path, monkeypatch):
    gdir = tmp_path / "graph"
    ddir = tmp_path / "demo"
    gdir.mkdir()
    ddir.mkdir()
    monkeypatch.setattr(pi, "GRAPH_DIR", gdir)
    monkeypatch.setattr(store_mod, "GRAPH_DIR", gdir)
    monkeypatch.setattr(changes_mod, "GRAPH_DIR", gdir)
    monkeypatch.setattr(changes_mod, "DEMO_DIR", ddir)
    return gdir


def test_refresh_if_stale_add_edit_delete_updates_graph_without_manual_reingest(
    tmp_path, monkeypatch
):
    """Ingest once; add/edit/delete; refresh_if_stale updates graph/digest."""
    gdir = _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib-auto"
    lib.mkdir()
    alpha = lib / "alpha.md"
    beta = lib / "beta.md"
    alpha.write_text("# Alpha Protocol\n\nNotes about pembrolizumab.\n", encoding="utf-8")
    beta.write_text("# Beta Findings\n\nSee alpha.md. Mentions PDCD1.\n", encoding="utf-8")

    first = pi.ingest_directory(lib, slug="auto-lib", mode="personal")
    assert first["file_count"] == 2
    nodes_before = first["node_count"]
    edges_before = first["edge_count"]
    graph_before = store_mod.load_graph("private_auto_lib")
    labels_before = {n.get("label") for n in graph_before["nodes"]}

    # Add / edit / delete — do NOT call ingest_directory again as the user action.
    (lib / "gamma.md").write_text(
        "# Gamma Notes\n\nNew file mentioning osimertinib and EGFR.\n",
        encoding="utf-8",
    )
    alpha.write_text(
        "# Alpha Protocol\n\nUpdated notes about pembrolizumab and extra context.\n",
        encoding="utf-8",
    )
    beta.unlink()

    assert pi.library_needs_refresh("auto-lib") is True
    result = pi.refresh_if_stale("auto-lib")
    assert result["status"] == "refreshed"
    assert result["refreshed"] is True
    assert result["file_count"] == 2
    assert result["library_slug"] == "private_auto_lib"

    graph = store_mod.load_graph("private_auto_lib")
    labels = {n.get("label") for n in graph["nodes"]}
    assert any("Gamma" in (lab or "") for lab in labels)
    digest_path = gdir / "private_auto_lib.changes.json"
    assert digest_path.exists()
    change_meta = result.get("change_digest") or {}
    assert (
        result["node_count"] != nodes_before
        or result["edge_count"] != edges_before
        or change_meta.get("change_count", 0) >= 1
        or labels != labels_before
    )
    assert (graph.get("meta") or {}).get("public_demo_mixed") is False
    assert not (tmp_path / "demo" / "change_digest.json").exists()


def test_refresh_if_stale_unchanged_does_not_rewrite(tmp_path, monkeypatch):
    gdir = _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib-stable"
    lib.mkdir()
    (lib / "keep.md").write_text("# Keep\n\nStable pembrolizumab note.\n", encoding="utf-8")
    pi.ingest_directory(lib, slug="stable-lib", mode="personal")

    graph_path = gdir / "private_stable_lib.json"
    manifest_path = gdir / "private_stable_lib.manifest.json"
    graph_bytes = graph_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    saved_at = json.loads(graph_bytes)["meta"]["saved_at"]

    assert pi.library_needs_refresh("stable-lib") is False
    result = pi.refresh_if_stale("stable-lib")
    assert result["status"] == "unchanged"
    assert result["refreshed"] is False
    assert graph_path.read_bytes() == graph_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert json.loads(graph_path.read_text(encoding="utf-8"))["meta"]["saved_at"] == saved_at


def test_refresh_if_stale_missing_watched_path_skips(tmp_path, monkeypatch):
    _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib-gone"
    lib.mkdir()
    (lib / "only.md").write_text("# Only\n\nTemp.\n", encoding="utf-8")
    pi.ingest_directory(lib, slug="gone-lib", mode="personal")
    for child in lib.iterdir():
        child.unlink()
    lib.rmdir()

    assert pi.library_needs_refresh("gone-lib") is False
    result = pi.refresh_if_stale("gone-lib")
    assert result["status"] == "skipped"
    assert result["refreshed"] is False
    assert result["reason"] == "watched_path_missing"


def test_public_demo_files_untouched_by_private_auto_refresh(tmp_path, monkeypatch):
    fixtures = (
        [p for p in DEMO_GRAPH_FIXTURES_DIR.iterdir() if p.is_file()]
        if DEMO_GRAPH_FIXTURES_DIR.is_dir()
        else []
    )
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in fixtures}

    gdir = _patch_graph_dirs(tmp_path, monkeypatch)
    public = gdir / f"{DEMO_GRAPH_SLUG}.json"
    public.write_text(
        '{"goal":"public demo","nodes":[],"edges":[],"meta":{}}',
        encoding="utf-8",
    )
    public_bytes = public.read_bytes()

    lib = tmp_path / "lib-sep"
    lib.mkdir()
    (lib / "note.md").write_text("# Private Note\n\npembrolizumab memo.\n", encoding="utf-8")
    pi.ingest_directory(lib, slug="sep-lib", mode="enterprise")
    (lib / "extra.md").write_text("# Extra\n\nEGFR mention.\n", encoding="utf-8")
    pi.refresh_if_stale("sep-lib")

    assert public.read_bytes() == public_bytes
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in fixtures}
    assert after == before
    assert not (tmp_path / "demo" / "change_digest.json").exists()


def test_library_status_reports_watching_when_watcher_live(tmp_path, monkeypatch):
    from living_evidence_graph.library_watch import (
        is_watching,
        start_watcher,
        start_watchers_from_manifests,
        stop_all_watchers,
    )

    _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib-watch"
    lib.mkdir()
    (lib / "a.md").write_text("# A\n\npembrolizumab.\n", encoding="utf-8")
    pi.ingest_directory(lib, slug="watch-lib", mode="personal")

    status = pi.library_status("watch-lib")
    assert status is not None
    assert status["auto_refresh"] is False
    assert status["watching"] is False

    try:
        assert start_watcher("private_watch_lib", lib) is True
        assert is_watching("watch-lib") is True
        status = pi.library_status("watch-lib")
        assert status["auto_refresh"] is True
        assert status["watching"] is True
    finally:
        stop_all_watchers()

    assert start_watcher("private_watch_lib", tmp_path / "does-not-exist") is False

    try:
        started = start_watchers_from_manifests()
        assert "private_watch_lib" in started
        assert pi.library_status("watch-lib")["watching"] is True
    finally:
        stop_all_watchers()


def test_watch_tick_add_file_refreshes_without_manual_ingest(tmp_path, monkeypatch):
    from living_evidence_graph.library_watch import (
        start_watcher,
        stop_all_watchers,
        watch_tick,
    )

    _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib-tick"
    lib.mkdir()
    (lib / "seed.md").write_text("# Seed\n\npembrolizumab.\n", encoding="utf-8")
    first = pi.ingest_directory(lib, slug="tick-lib", mode="personal")
    nodes_before = first["node_count"]

    try:
        assert start_watcher("private_tick_lib", lib) is True
        (lib / "new.md").write_text(
            "# New Topic\n\nMentions osimertinib and EGFR.\n", encoding="utf-8"
        )
        result = watch_tick("tick-lib")
        assert result["status"] == "refreshed"
        graph = store_mod.load_graph("private_tick_lib")
        assert len(graph["nodes"]) >= nodes_before
        labels = " ".join(n.get("label") or "" for n in graph["nodes"])
        assert "New Topic" in labels or result["file_count"] == 2

        (lib / "later.md").write_text("# Later Doc\n\nAnother EGFR note.\n", encoding="utf-8")
        deadline = time.time() + 6
        saw = False
        while time.time() < deadline:
            g = store_mod.load_graph("private_tick_lib")
            labs = " ".join(n.get("label") or "" for n in g["nodes"])
            meta_count = (g.get("meta") or {}).get("file_count")
            if "Later Doc" in labs or meta_count == 3:
                saw = True
                break
            time.sleep(0.2)
        assert saw, "background watcher did not auto-refresh after file add"
    finally:
        stop_all_watchers()


def test_http_ingest_starts_watcher_and_library_get_reports(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from living_evidence_graph.library_watch import stop_all_watchers
    from living_evidence_graph.server import app

    _patch_graph_dirs(tmp_path, monkeypatch)
    lib = tmp_path / "lib-http"
    lib.mkdir()
    (lib / "http.md").write_text("# HTTP Lib\n\npembrolizumab.\n", encoding="utf-8")

    try:
        with TestClient(app) as client:
            r = client.post(
                "/library/ingest",
                json={"path": str(lib), "slug": "http-lib", "mode": "personal"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["library_slug"] == "private_http_lib"
            assert body.get("watching") is True
            assert body.get("auto_refresh") is True
            st = client.get("/library/http-lib")
            assert st.status_code == 200
            js = st.json()
            assert js["auto_refresh"] is True
            assert js["watching"] is True
            assert js["public_demo_mixed"] is False
    finally:
        stop_all_watchers()
