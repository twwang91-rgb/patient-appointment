"""Structured JSON-lines tracing with PHI redaction.

Every event is one JSON object: timestamp, session id, turn number, event name,
and arbitrary fields. Phone numbers and dates of birth are never written in
plaintext; use `fingerprint()` for a stable, non-reversible correlation token.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

_PHONE_RE = re.compile(r"(?<!\d)(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b")
_ZIP_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")


def fingerprint(value: Optional[str]) -> Optional[str]:
    """Short one-way token so two log lines can be correlated without exposing the value."""
    if value is None:
        return None
    return "h:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def redact_text(text: str) -> str:
    """Mask phone numbers, dates, and zip codes inside free text before logging."""
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _DATE_RE.sub("<DATE>", text)
    text = _ZIP_RE.sub("<ZIP>", text)
    return text


class Tracer:
    """Collects events in memory and optionally appends them to a JSONL file / stderr."""

    def __init__(self, path: Optional[str] = None, echo: bool = False, session_id: Optional[str] = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.turn = 0
        self.events: List[Dict[str, Any]] = []
        self.echo = echo
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def event(self, name: str, **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": self.session_id,
            "turn": self.turn,
            "event": name,
        }
        record.update(fields)
        self.events.append(record)
        line = json.dumps(record, default=str)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.echo:
            print("  [trace] " + line, file=sys.stderr, flush=True)
        return record

    @contextmanager
    def timed(self, name: str, **fields: Any) -> Iterator[Dict[str, Any]]:
        """Time a block; the yielded dict can be filled with extra fields before it is logged."""
        extra: Dict[str, Any] = {}
        started = time.perf_counter()
        try:
            yield extra
        finally:
            ms = round((time.perf_counter() - started) * 1000, 1)
            self.event(name, latency_ms=ms, **fields, **extra)

    def find(self, name: str, **match: Any) -> List[Dict[str, Any]]:
        """Return recorded events with the given name whose fields match (used by evals)."""
        out = []
        for ev in self.events:
            if ev["event"] != name:
                continue
            if all(ev.get(k) == v for k, v in match.items()):
                out.append(ev)
        return out

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
