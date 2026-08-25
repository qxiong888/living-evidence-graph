"""In-memory demo session store.

Binds a browser session to a graph slug + RAG mode (Grounded or Strict).
Cloud Run min-instances 0 may recycle memory; cookie + query session_id
are the client-facing keys. No secrets. Contest demo only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

SessionMode = Literal["grounded", "strict"]

_LOCK = Lock()
_SESSIONS: dict[str, "DemoSession"] = {}


@dataclass(frozen=True)
class DemoSession:
    session_id: str
    graph_slug: str
    mode: str
    created_at: str
    node_count: int = 0
    edge_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reset_sessions() -> None:
    """Test helper: drop all in-memory sessions."""
    with _LOCK:
        _SESSIONS.clear()


def create_session(
    *,
    graph_slug: str,
    mode: str,
    node_count: int = 0,
    edge_count: int = 0,
) -> DemoSession:
    sid = str(uuid4())
    sess = DemoSession(
        session_id=sid,
        graph_slug=graph_slug,
        mode=mode,
        created_at=datetime.now(timezone.utc).isoformat(),
        node_count=int(node_count or 0),
        edge_count=int(edge_count or 0),
    )
    with _LOCK:
        _SESSIONS[sid] = sess
    return sess


def get_session(session_id: str | None) -> DemoSession | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with _LOCK:
        return _SESSIONS.get(sid)


def session_count() -> int:
    with _LOCK:
        return len(_SESSIONS)
