"""Background auto-refresh for private personal/enterprise folder graphs.

First ingest registers watched_path. While the server (or CLI --watch) is
running, add/edit/delete of supported files rebuilds that private graph.
Public Keytruda demo is never mixed in. Cloud Run has no user folders —
watchers no-op when watched_path is missing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from living_evidence_graph.config import DEMO_GRAPH_SLUG
from living_evidence_graph.private_ingest import (
    library_needs_refresh,
    load_manifest,
    refresh_if_stale,
    resolve_private_slug,
)

log = logging.getLogger(__name__)

WATCH_DEBOUNCE_MS = 1000
WATCH_POLL_SECONDS = 1.0

_registry_lock = threading.Lock()
_watchers: dict[str, "_LibraryWatcher"] = {}


class _LibraryWatcher:
    def __init__(self, slug: str, watched_path: Path) -> None:
        self.slug = slug
        self.watched_path = Path(watched_path)
        self.stop_event = threading.Event()
        self.refresh_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._loop,
            name=f"leg-watch-{slug}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout=timeout)

    def alive(self) -> bool:
        return self.thread.is_alive()

    def _safe_refresh(self) -> None:
        # One refresh at a time per slug; a burst collapses via debounce + lock.
        with self.refresh_lock:
            try:
                refresh_if_stale(self.slug)
                # Writes during ingest: one more pass if still stale.
                if library_needs_refresh(self.slug):
                    refresh_if_stale(self.slug)
            except Exception:  # noqa: BLE001 — watcher must not die
                log.exception("private library refresh failed slug=%s", self.slug)

    def _loop(self) -> None:
        if not self.watched_path.is_dir():
            return
        try:
            from watchfiles import watch
        except ImportError:
            self._poll_loop()
            return
        try:
            for _changes in watch(
                str(self.watched_path),
                stop_event=self.stop_event,
                debounce=WATCH_DEBOUNCE_MS,
                rust_timeout=1000,
                recursive=True,
            ):
                if self.stop_event.is_set():
                    break
                self._safe_refresh()
        except Exception:  # noqa: BLE001
            log.exception("watchfiles loop failed slug=%s; falling back to poll", self.slug)
            self._poll_loop()

    def _poll_loop(self) -> None:
        while not self.stop_event.wait(WATCH_POLL_SECONDS):
            if not self.watched_path.is_dir():
                return
            self._safe_refresh()


def _graph_dir() -> Path:
    # Read through private_ingest so tests that patch GRAPH_DIR are honored.
    from living_evidence_graph import private_ingest as pi

    return pi.GRAPH_DIR


def is_watching(library_slug: str) -> bool:
    slug = resolve_private_slug(library_slug) or (library_slug or "").strip()
    with _registry_lock:
        watcher = _watchers.get(slug)
        if watcher is None and slug and not slug.startswith("private_"):
            watcher = _watchers.get(f"private_{slug}")
        return bool(watcher and watcher.alive())


def start_watcher(library_slug: str, watched_path: str | Path | None = None) -> bool:
    """Start or replace a folder watcher. False if path is missing (Cloud Run)."""
    raw = (library_slug or "").strip()
    # Public Keytruda/NSCLC graph is API-only — never attach a local-dir watcher.
    if not raw or raw == DEMO_GRAPH_SLUG or raw.startswith(f"{DEMO_GRAPH_SLUG}."):
        return False
    slug = resolve_private_slug(raw)
    if slug is None:
        slug = raw if raw.startswith("private_") else None
    if not slug or slug == DEMO_GRAPH_SLUG:
        return False
    path: Path | None = None
    if watched_path:
        path = Path(str(watched_path)).expanduser()
    else:
        manifest = load_manifest(slug) or {}
        wp = manifest.get("watched_path")
        if wp:
            path = Path(str(wp)).expanduser()
    if path is None or not path.is_dir():
        return False
    watcher = _LibraryWatcher(slug, path.resolve())
    with _registry_lock:
        old = _watchers.pop(slug, None)
        _watchers[slug] = watcher
    if old is not None:
        old.stop()
    watcher.start()
    return watcher.alive()


def stop_watcher(library_slug: str) -> None:
    slug = resolve_private_slug(library_slug) or (library_slug or "").strip()
    with _registry_lock:
        watcher = _watchers.pop(slug, None)
        if watcher is None and slug and not slug.startswith("private_"):
            watcher = _watchers.pop(f"private_{slug}", None)
    if watcher is not None:
        watcher.stop()


def stop_all_watchers() -> None:
    with _registry_lock:
        items = list(_watchers.values())
        _watchers.clear()
    for watcher in items:
        watcher.stop()


def start_watchers_from_manifests() -> list[str]:
    """Seed watchers for every private_*.manifest.json whose path is a real dir."""
    started: list[str] = []
    gdir = _graph_dir()
    if not gdir.is_dir():
        return started
    for path in sorted(gdir.glob("private_*.manifest.json")):
        name = path.name
        if not name.endswith(".manifest.json"):
            continue
        slug = name[: -len(".manifest.json")]
        if slug == DEMO_GRAPH_SLUG or not slug.startswith("private_"):
            continue
        if start_watcher(slug):
            started.append(slug)
    return started


def watch_tick(library_slug: str) -> dict[str, Any]:
    """One refresh-if-stale pass (poll loop / tests)."""
    return refresh_if_stale(library_slug)
