"""Personal / enterprise private library → living evidence graph.

Point at a local directory of documents; scan → extract text → build a private
graph (separate from the public Keytruda demo). On refresh, snapshot + diff →
change digest (what / why / sources = file paths).

Local absolute path only for MVP — no cloud storage required. Not a medical
product; private edges cite file paths only (never invent NCT/PMID IDs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, Mapping

from living_evidence_graph.changes import build_and_persist_digest, save_snapshot
from living_evidence_graph.config import GRAPH_DIR
from living_evidence_graph.credibility import recompute_edges
from living_evidence_graph.graph_store import load_graph, save_graph
from living_evidence_graph.schema import Edge, Node

LibraryMode = Literal["personal", "enterprise"]

SUPPORTED_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json", ".pdf"}
)
SKIP_DIR_NAMES = frozenset(
    {".git", ".svn", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
)

_ingest_locks_guard = threading.Lock()
_ingest_locks: dict[str, threading.RLock] = {}


def _ingest_lock(library_slug: str) -> threading.RLock:
    """One ingest/refresh at a time per private library slug."""
    with _ingest_locks_guard:
        lock = _ingest_locks.get(library_slug)
        if lock is None:
            lock = threading.RLock()
            _ingest_locks[library_slug] = lock
        return lock

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.M)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]{1,60}")
# Lightweight drug-ish / gene-ish heuristics (no invented registry IDs).
_DRUG_SUFFIX_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\-]{2,}(?:mab|nib|tinib|ciclib|parin|olol|pril|sartan|statin|azole))\b",
    re.I,
)
_GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,8})\b")
_COMMON_GENE_BLOCKLIST = frozenset(
    {
        "THE",
        "AND",
        "FOR",
        "WITH",
        "FROM",
        "THIS",
        "THAT",
        "PDF",
        "HTML",
        "JSON",
        "CSV",
        "HTTP",
        "HTTPS",
        "API",
        "FDA",
        "NLM",
        "NIH",
        "PMID",
        "NCT",
        "USA",
        "COVID",
        "DNA",
        "RNA",
        "SOP",
        "TODO",
        "NOTE",
        "NOTES",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(s: str, *, max_len: int = 80) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:max_len] or "item"


def _path_hash(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]


def normalize_library_slug(slug: str | None, watched_path: Path) -> str:
    """User slug or private_<hash(path)>. Always private_* to avoid public demo clash."""
    raw = (slug or "").strip()
    if raw:
        cleaned = _slug(raw, max_len=60)
        if cleaned.startswith("private_"):
            return cleaned
        return f"private_{cleaned}"
    return f"private_{_path_hash(str(watched_path.resolve()))}"


def manifest_path(library_slug: str) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPH_DIR / f"{library_slug}.manifest.json"


def load_manifest(library_slug: str) -> dict[str, Any] | None:
    path = manifest_path(library_slug)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_manifest(doc: Mapping[str, Any], library_slug: str) -> Path:
    path = manifest_path(library_slug)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(doc), f, indent=2, ensure_ascii=False)
    return path


def file_stable_id(path: Path, *, root: Path) -> str:
    """Stable id from relative path + mtime + size (content-aware via size/mtime)."""
    try:
        st = path.stat()
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except OSError:
        mtime, size = 0, 0
    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(path)
    raw = f"{rel}|{mtime}|{size}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _rel_under(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def evidence_ref(root: Path, path: Path) -> str:
    """Prefer relative path under watched dir; also expose file:// absolute."""
    return _rel_under(root, path)


def file_url(path: Path) -> str:
    return path.resolve().as_uri()


def scan_directory(root: Path) -> list[dict[str, Any]]:
    """Recursive inventory of supported text-ish files (skip binaries / junk dirs)."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        suf = path.suffix.lower()
        if suf not in SUPPORTED_SUFFIXES:
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        rel = _rel_under(root, path)
        inventory.append(
            {
                "path": str(path.resolve()),
                "rel_path": rel,
                "suffix": suf,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "file_id": file_stable_id(path, root=root),
            }
        )
    return inventory


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = False
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data:
            self._chunks.append(data)

    def text(self) -> str:
        return html.unescape(re.sub(r"[ \t]+\n", "\n", "".join(self._chunks)))


def _extract_pdf(path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return None
    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                t = ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts) if parts else ""
    except Exception:  # noqa: BLE001
        return None


def extract_text(path: Path) -> str | None:
    """Extract plain text from a supported file. None = skip (binary/unreadable)."""
    suf = path.suffix.lower()
    try:
        if suf in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suf in {".html", ".htm"}:
            raw = path.read_text(encoding="utf-8", errors="replace")
            parser = _HTMLTextExtractor()
            parser.feed(raw)
            return parser.text()
        if suf == ".csv":
            rows: list[str] = []
            with path.open(encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 500:
                        break
                    rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
            return "\n".join(rows)
        if suf == ".json":
            raw = path.read_text(encoding="utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            return json.dumps(data, indent=2, ensure_ascii=False)[:200_000]
        if suf == ".pdf":
            return _extract_pdf(path)
    except OSError:
        return None
    return None


def _source_tags(mode: LibraryMode) -> list[str]:
    # private_library is the family; mode is the product boundary flag.
    return ["private_library", mode]


def _ensure_node(nodes: dict[str, Node], node: Node) -> None:
    nid = node.get("id")
    if not nid:
        return
    if nid in nodes:
        prev = nodes[nid]
        # Prefer longer label if empty-ish
        if not (prev.get("label") or "").strip() and node.get("label"):
            prev["label"] = node["label"]
        props = dict(prev.get("props") or {})
        props.update(node.get("props") or {})
        prev["props"] = props
    else:
        nodes[nid] = node


def _heading_topics(text: str) -> list[str]:
    topics: list[str] = []
    for m in _HEADING_RE.finditer(text or ""):
        title = m.group(2).strip()
        if 2 <= len(title) <= 120:
            topics.append(title)
    if not topics:
        # First non-empty line as topic
        for line in (text or "").splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) >= 3:
                topics.append(line[:120])
                break
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:12]


def _guess_entities(text: str) -> list[tuple[str, str, str]]:
    """Return list of (entity_type, id_suffix, label) from lightweight heuristics."""
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for m in _DRUG_SUFFIX_RE.finditer(text or ""):
        label = m.group(1)
        key = f"drug:{_slug(label)}"
        if key not in seen:
            seen.add(key)
            found.append(("Drug", _slug(label), label))
    for m in _GENE_RE.finditer(text or ""):
        sym = m.group(1)
        if sym in _COMMON_GENE_BLOCKLIST:
            continue
        if not re.search(r"[A-Z]", sym):
            continue
        key = f"gene:{sym}"
        if key not in seen:
            seen.add(key)
            found.append(("Gene", sym, sym))
    return found[:20]


def build_graph_from_texts(
    items: list[dict[str, Any]],
    *,
    mode: LibraryMode,
    watched_path: Path,
    goal: str,
) -> dict[str, Any]:
    """Rules-based private graph: SourceDoc per file + topic/entity edges.

    evidence_urls / sources cite file paths only — never invent NCT/PMID.
    """
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    now = _now_iso()
    sources = _source_tags(mode)
    root = watched_path.resolve()

    # Index for cross-file cites by stem / rel path
    by_stem: dict[str, str] = {}  # stem_lower -> sourcedoc id

    file_payloads: list[dict[str, Any]] = []
    for item in items:
        path = Path(item["path"])
        text = item.get("text") or ""
        rel = item.get("rel_path") or _rel_under(root, path)
        fid = item.get("file_id") or file_stable_id(path, root=root)
        doc_id = f"sourcedoc:{fid}"
        label = path.stem.replace("_", " ").replace("-", " ")
        topics = _heading_topics(text)
        if topics:
            label = topics[0][:120]
        ev = [rel, file_url(path)]
        _ensure_node(
            nodes,
            {
                "id": doc_id,
                "type": "SourceDoc",
                "label": label,
                "props": {
                    "rel_path": rel,
                    "path": str(path.resolve()),
                    "suffix": path.suffix.lower(),
                    "file_id": fid,
                    "mode": mode,
                    "library": "private",
                },
            },
        )
        by_stem[path.stem.lower()] = doc_id
        by_stem[rel.lower()] = doc_id
        file_payloads.append(
            {
                "doc_id": doc_id,
                "path": path,
                "rel": rel,
                "text": text,
                "topics": topics,
                "ev": ev,
            }
        )

        # Topic nodes (Publication = claim/section label from the file; not a PMID)
        for topic in topics[:8]:
            tid = f"pub:private:{_slug(topic)}"
            _ensure_node(
                nodes,
                {
                    "id": tid,
                    "type": "Publication",
                    "label": topic,
                    "props": {
                        "from_file": rel,
                        "kind": "private_topic",
                        "note": "Topic/heading extracted from private library file — not a PMID",
                    },
                },
            )
            edges.append(
                {
                    "id": f"edge:supports:{fid}:{_slug(topic)}",
                    "type": "supports",
                    "source": doc_id,
                    "target": tid,
                    "evidence_urls": ev,
                    "sources": list(sources),
                    "first_seen": now,
                    "last_seen": now,
                    "props": {"rel_path": rel, "topic": topic, "mode": mode},
                }
            )

        for etype, suffix, elabel in _guess_entities(text):
            if etype == "Drug":
                eid = f"drug:{suffix}"
            else:
                eid = f"gene:{suffix}"
            _ensure_node(
                nodes,
                {
                    "id": eid,
                    "type": etype,  # type: ignore[typeddict-item]
                    "label": elabel,
                    "props": {"from_private_library": True, "mode": mode},
                },
            )
            edges.append(
                {
                    "id": f"edge:cites:{fid}:{_slug(eid)}",
                    "type": "cites",
                    "source": doc_id,
                    "target": eid,
                    "evidence_urls": ev,
                    "sources": list(sources),
                    "first_seen": now,
                    "last_seen": now,
                    "props": {"rel_path": rel, "mention": elabel, "mode": mode},
                }
            )

    # Cross-file cites when one document's text mentions another's filename stem
    for fp in file_payloads:
        text_low = (fp["text"] or "").lower()
        for other in file_payloads:
            if other["doc_id"] == fp["doc_id"]:
                continue
            stem = other["path"].stem.lower()
            if len(stem) < 3:
                continue
            if stem in text_low or other["rel"].lower() in text_low:
                edges.append(
                    {
                        "id": f"edge:cites:{_slug(fp['doc_id'])}:{_slug(other['doc_id'])}",
                        "type": "cites",
                        "source": fp["doc_id"],
                        "target": other["doc_id"],
                        "evidence_urls": fp["ev"],
                        "sources": list(sources),
                        "first_seen": now,
                        "last_seen": now,
                        "props": {
                            "rel_path": fp["rel"],
                            "cites_file": other["rel"],
                            "mode": mode,
                        },
                    }
                )

    # Dedupe edges by id
    edge_map = {e["id"]: e for e in edges if e.get("id")}
    scored = recompute_edges(edge_map.values())

    return {
        "goal": goal,
        "nodes": list(nodes.values()),
        "edges": scored,
        "meta": {
            "library": True,
            "mode": mode,
            "watched_path": str(root),
            "source_boundary": "private",
            "public_demo_mixed": False,
            "file_count": len(file_payloads),
            "disclaimer": (
                "Private library graph — not a medical product. "
                "Edges cite local file paths only. No invented NCT/PMID IDs. "
                "Not mixed with the public Keytruda demo graph."
            ),
        },
    }


def replace_private_graph(
    doc: dict[str, Any],
    *,
    library_slug: str,
    emit_change_digest: bool = True,
) -> dict[str, Any]:
    """Full replace (not merge) so deleted files drop their nodes/edges."""
    existing = load_graph(library_slug)
    has_prior = bool(existing.get("nodes") or existing.get("edges"))
    if has_prior:
        save_snapshot(existing, goal_slug=library_slug)

    path = save_graph(doc, goal_slug=library_slug)
    doc = dict(doc)
    meta = dict(doc.get("meta") or {})
    meta["path"] = str(path)
    doc["meta"] = meta

    if emit_change_digest:
        prev_for_diff = existing if has_prior else None
        digest = build_and_persist_digest(
            prev_for_diff,
            doc,
            goal_slug=library_slug,
            also_demo=False,  # never overwrite public Keytruda demo digest
        )
        meta["change_digest"] = {
            "change_count": digest.get("change_count", 0),
            "by_what": digest.get("by_what") or {},
            "digest_path": str(GRAPH_DIR / f"{library_slug}.changes.json"),
        }
        doc["meta"] = meta
        # Persist digest pointer into saved graph meta
        save_graph(doc, goal_slug=library_slug)

    return doc


def ingest_directory(
    directory: str | Path,
    *,
    slug: str | None = None,
    mode: LibraryMode = "personal",
    goal: str | None = None,
) -> dict[str, Any]:
    """Scan directory → extract → replace private graph + manifest + change digest."""
    if mode not in ("personal", "enterprise"):
        raise ValueError("mode must be 'personal' or 'enterprise'")
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")

    library_slug = normalize_library_slug(slug, root)
    with _ingest_lock(library_slug):
        return _ingest_directory_locked(
            root, library_slug=library_slug, mode=mode, goal=goal
        )


def _ingest_directory_locked(
    root: Path,
    *,
    library_slug: str,
    mode: LibraryMode,
    goal: str | None,
) -> dict[str, Any]:
    inventory = scan_directory(root)

    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in inventory:
        text = extract_text(Path(entry["path"]))
        if text is None:
            skipped.append({**entry, "reason": "unreadable_or_unsupported"})
            continue
        items.append({**entry, "text": text})

    goal_str = goal or f"private {mode} library: {root.name}"
    graph = build_graph_from_texts(
        items,
        mode=mode,
        watched_path=root,
        goal=goal_str,
    )
    graph["meta"]["library_slug"] = library_slug
    graph["meta"]["ingested_at"] = _now_iso()
    graph["meta"]["skipped_files"] = len(skipped)

    saved = replace_private_graph(graph, library_slug=library_slug)

    manifest = {
        "library_slug": library_slug,
        "mode": mode,
        "watched_path": str(root),
        "last_scan": _now_iso(),
        "file_count": len(items),
        "skipped_count": len(skipped),
        "node_count": len(saved.get("nodes") or []),
        "edge_count": len(saved.get("edges") or []),
        "files": [
            {
                "rel_path": it["rel_path"],
                "file_id": it["file_id"],
                "size": it["size"],
                "mtime": it["mtime"],
                "suffix": it["suffix"],
            }
            for it in items
        ],
        "skipped": skipped,
        "graph_path": str(GRAPH_DIR / f"{library_slug}.json"),
        "public_demo_mixed": False,
    }
    save_manifest(manifest, library_slug)

    change_meta = (saved.get("meta") or {}).get("change_digest") or {}
    return {
        "status": "success",
        "library_slug": library_slug,
        "mode": mode,
        "watched_path": str(root),
        "file_count": len(items),
        "skipped_count": len(skipped),
        "node_count": len(saved.get("nodes") or []),
        "edge_count": len(saved.get("edges") or []),
        "graph_path": str(GRAPH_DIR / f"{library_slug}.json"),
        "manifest_path": str(manifest_path(library_slug)),
        "change_digest": change_meta,
        "goal": goal_str,
    }



def normalize_existing_slug(library_slug: str) -> str | None:
    """Resolve a user-facing slug to the on-disk slug if graph or manifest exists."""
    slug = (library_slug or "").strip()
    if not slug:
        return None
    if not slug.startswith("private_"):
        alt = f"private_{_slug(slug)}"
        if (GRAPH_DIR / f"{alt}.json").exists() or manifest_path(alt).exists():
            return alt
        if not (GRAPH_DIR / f"{slug}.json").exists() and not manifest_path(slug).exists():
            return None
    return slug


def resolve_private_slug(library_slug: str) -> str | None:
    """Like normalize_existing_slug but never returns the public demo slug."""
    slug = normalize_existing_slug(library_slug)
    if slug is None or not slug.startswith("private_"):
        return None
    return slug


def _file_sig(entry: dict[str, Any]) -> tuple[str, int, int]:
    rel = str(entry.get("rel_path") or entry.get("path") or "").replace("\\", "/")
    return (rel, int(entry.get("size") or 0), int(entry.get("mtime") or 0))


def library_needs_refresh(library_slug: str) -> bool:
    """True when watched folder files (path + size + mtime) differ from the manifest.

    Skip (False) when there is no private library, or watched_path is missing
    (Cloud Run public demo has no user folders).
    """
    slug = resolve_private_slug(library_slug)
    if slug is None:
        return False
    manifest = load_manifest(slug)
    if not manifest:
        return False
    watched = manifest.get("watched_path")
    if not watched:
        return False
    root = Path(str(watched)).expanduser()
    if not root.is_dir():
        return False
    try:
        current = scan_directory(root)
    except FileNotFoundError:
        return False
    now_set = {_file_sig(e) for e in current}
    old_set = {_file_sig(f) for f in (manifest.get("files") or [])}
    return now_set != old_set


def refresh_if_stale(library_slug: str) -> dict[str, Any]:
    """Re-ingest when the watched folder changed; no-op when unchanged or path missing."""
    slug = resolve_private_slug(library_slug)
    if slug is None:
        return {
            "status": "skipped",
            "reason": "unknown_library",
            "refreshed": False,
            "library_slug": library_slug,
        }
    with _ingest_lock(slug):
        manifest = load_manifest(slug) or {}
        watched = manifest.get("watched_path")
        if not watched:
            return {
                "status": "skipped",
                "reason": "no_watched_path",
                "refreshed": False,
                "library_slug": slug,
            }
        root = Path(str(watched)).expanduser()
        if not root.is_dir():
            return {
                "status": "skipped",
                "reason": "watched_path_missing",
                "refreshed": False,
                "library_slug": slug,
                "watched_path": str(root),
            }
        if not library_needs_refresh(slug):
            return {
                "status": "unchanged",
                "refreshed": False,
                "library_slug": slug,
                "watched_path": str(root.resolve()),
                "file_count": manifest.get("file_count", 0),
                "node_count": manifest.get("node_count", 0),
                "edge_count": manifest.get("edge_count", 0),
            }
        mode = manifest.get("mode") or "personal"
        if mode not in ("personal", "enterprise"):
            mode = "personal"
        result = ingest_directory(root, slug=slug, mode=mode)
        result["status"] = "refreshed"
        result["refreshed"] = True
        return result


def library_status(library_slug: str) -> dict[str, Any] | None:
    """Status from manifest + graph counts."""
    slug = normalize_existing_slug(library_slug)
    if slug is None:
        return None

    manifest = load_manifest(slug) or {}
    graph = load_graph(slug)
    has_graph = bool(graph.get("nodes") or graph.get("edges") or (GRAPH_DIR / f"{slug}.json").exists())
    if not has_graph and not manifest:
        return None
    watching = False
    try:
        from living_evidence_graph.library_watch import is_watching

        watching = is_watching(slug)
    except Exception:  # noqa: BLE001
        watching = False
    return {
        "library_slug": slug,
        "mode": manifest.get("mode") or (graph.get("meta") or {}).get("mode") or "personal",
        "path": manifest.get("watched_path")
        or (graph.get("meta") or {}).get("watched_path"),
        "watched_path": manifest.get("watched_path")
        or (graph.get("meta") or {}).get("watched_path"),
        "file_count": manifest.get("file_count", 0),
        "node_count": len(graph.get("nodes") or []),
        "edge_count": len(graph.get("edges") or []),
        "last_updated": (graph.get("meta") or {}).get("saved_at")
        or manifest.get("last_scan"),
        "last_scan": manifest.get("last_scan"),
        "graph_path": str(GRAPH_DIR / f"{slug}.json"),
        "manifest_path": str(manifest_path(slug)),
        "public_demo_mixed": False,
        "change_digest": (graph.get("meta") or {}).get("change_digest"),
        "auto_refresh": watching,
        "watching": watching,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m living_evidence_graph.private_ingest",
        description=(
            "Ingest a local directory into a personal/enterprise private living graph "
            "(separate from the public Keytruda demo). "
            "Without --watch this is a one-shot ingest; leave the server or --watch "
            "running so later folder changes auto-refresh the private graph."
        ),
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Absolute or relative path to the document directory",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Library slug (stored as private_<slug>); default hash of path",
    )
    parser.add_argument(
        "--mode",
        choices=("personal", "enterprise"),
        default="personal",
        help="Product boundary flag (personal | enterprise)",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Optional goal string for the graph document",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "After the first ingest, stay running and auto-refresh the private graph "
            "when files change (Ctrl-C to stop). Without --watch, one-shot ingest only."
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = ingest_directory(
            args.dir,
            slug=args.slug,
            mode=args.mode,  # type: ignore[arg-type]
            goal=args.goal,
        )
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if not args.watch:
        return 0

    from living_evidence_graph.library_watch import start_watcher, stop_watcher

    slug = result["library_slug"]
    started = start_watcher(slug, result.get("watched_path"))
    result["auto_refresh"] = started
    result["watching"] = started
    if not started:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "could not start watcher (watched_path missing?)",
                    "library_slug": slug,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        f"Watching {result.get('watched_path')} for {slug}. "
        "File changes auto-refresh the private graph. Ctrl-C to stop.",
        file=sys.stderr,
    )
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_watcher(slug)
        print(json.dumps({"status": "stopped", "library_slug": slug}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
