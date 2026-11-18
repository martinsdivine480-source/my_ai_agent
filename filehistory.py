"""Version history for the files the agent writes.

Before the agent creates or replaces a file, the old content is saved here (outside the workspace, so
the agent's own file tools cannot touch it). Any earlier version can be restored later. Restoring
saves the current content first, so a restore can be undone too.

Layout (next to this file):  history/index.json  and  history/blobs/<sha256 of the content>"""
import difflib
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "history"
BLOB_DIR = HISTORY_DIR / "blobs"
INDEX_FILE = HISTORY_DIR / "index.json"
LOCK = threading.RLock()

MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024   # a bigger file is not kept (its history entry says so)
MAX_VERSIONS_PER_FILE = 30
MAX_ENTRIES = 2000
MAX_TOTAL_BYTES = 400 * 1024 * 1024
PREVIEW_CHARS = 20000
DIFF_MAX_CHARS = 20000


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: the file was being read at that instant
            if attempt == 4:
                raise
            time.sleep(0.05)


def _load() -> dict:
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                top = max((int(e.get("id", 0)) for e in data["entries"] if isinstance(e, dict)), default=0)
                data["next_id"] = max(int(data.get("next_id", 1)), top + 1)
                return data
        except (OSError, ValueError):
            try:
                INDEX_FILE.replace(INDEX_FILE.with_name("index.corrupt.json"))  # keep the damaged file
            except OSError:
                pass
    return {"next_id": 1, "entries": []}


def _save(index: dict) -> None:
    _atomic_write(INDEX_FILE, json.dumps(index, ensure_ascii=False).encode("utf-8"))


def _total_bytes(entries: list) -> int:
    seen, total = set(), 0
    for e in entries:
        blob = e.get("blob")
        if blob and blob not in seen:
            seen.add(blob)
            try:
                total += (BLOB_DIR / blob).stat().st_size
            except OSError:
                pass
    return total


def _prune(index: dict) -> None:
    """Keep the newest versions of every file, and stay under the size limits."""
    counts, keep = {}, []
    for e in reversed(index["entries"]):  # newest first
        n = counts.get(e["path"], 0)
        if n < MAX_VERSIONS_PER_FILE:
            keep.append(e)
            counts[e["path"]] = n + 1
    keep.reverse()
    if len(keep) > MAX_ENTRIES:
        keep = keep[-MAX_ENTRIES:]
    while keep and _total_bytes(keep) > MAX_TOTAL_BYTES:
        keep.pop(0)
    index["entries"] = keep
    referenced = {e["blob"] for e in keep if e.get("blob")}
    try:
        for p in BLOB_DIR.iterdir():
            if p.name not in referenced and not p.name.endswith(".tmp"):
                p.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------
# Saving versions (called by the agent around every write)
# ---------------------------------------------------------------
def capture(target: Path) -> dict:
    """Read what is in the file now, BEFORE it is changed. Never raises."""
    try:
        if not target.is_file():
            return {"existed": False, "data": None, "size": 0}
        size = target.stat().st_size
        if size > MAX_SNAPSHOT_BYTES:
            return {"existed": True, "data": None, "size": size}  # too big to keep
        return {"existed": True, "data": target.read_bytes(), "size": size}
    except OSError:
        return {"existed": False, "data": None, "size": 0}


def record(relative: str, tool: str, snapshot: dict, size_after: int | None = None):
    """Save the captured old content as a version of `relative`. Returns the entry id, or None.
    Never raises: a history problem must not turn a successful write into an error."""
    try:
        with LOCK:
            blob = None
            data = snapshot.get("data")
            if data is not None:
                blob = hashlib.sha256(data).hexdigest()
                path = BLOB_DIR / blob
                if not path.exists():
                    _atomic_write(path, data)
            index = _load()
            entry_id = index["next_id"]
            index["next_id"] = entry_id + 1
            existed = bool(snapshot.get("existed"))
            index["entries"].append({
                "id": entry_id,
                "path": relative,
                "time": _now(),
                "tool": tool,
                "existed": existed,
                "blob": blob,
                "size": int(snapshot.get("size") or 0),
                "size_after": size_after,
                "kept": (blob is not None) or (not existed),
            })
            _prune(index)
            _save(index)
            return entry_id
    except Exception:
        return None


# ---------------------------------------------------------------
# Reading the history
# ---------------------------------------------------------------
def _public(e: dict) -> dict:
    return {
        "id": e["id"],
        "path": e["path"],
        "name": e["path"].rsplit("/", 1)[-1],
        "time": e["time"],
        "tool": e.get("tool", ""),
        "existed": bool(e.get("existed")),
        "size": int(e.get("size") or 0),
        "size_after": e.get("size_after"),
        "kept": bool(e.get("kept")),
    }


def _find(entry_id: int):
    with LOCK:
        for e in _load()["entries"]:
            if e["id"] == int(entry_id):
                return e
    return None


def list_entries(path: str | None = None, limit: int = 200) -> list:
    """Saved versions, newest first. path: only the versions of that file."""
    with LOCK:
        entries = list(_load()["entries"])
    if path:
        entries = [e for e in entries if e["path"] == path]
    entries.sort(key=lambda e: e["id"], reverse=True)
    return [_public(e) for e in entries[:max(1, limit)]]


def list_files(limit: int = 200) -> list:
    """The files that have saved versions: [{path, name, count, last}], most recently changed first."""
    with LOCK:
        entries = list(_load()["entries"])
    grouped: dict = {}
    for e in entries:
        item = grouped.setdefault(e["path"], {"path": e["path"], "name": e["path"].rsplit("/", 1)[-1],
                                              "count": 0, "last": "", "last_id": 0})
        item["count"] += 1
        if e["id"] > item["last_id"]:
            item["last_id"] = e["id"]
            item["last"] = e["time"]
    out = sorted(grouped.values(), key=lambda g: g["last_id"], reverse=True)
    for item in out:
        item.pop("last_id", None)
    return out[:max(1, limit)]


def stats() -> dict:
    with LOCK:
        entries = list(_load()["entries"])
        return {"versions": len(entries), "files": len({e["path"] for e in entries}),
                "bytes": _total_bytes(entries)}


def _read_blob(entry: dict):
    blob = entry.get("blob")
    if not blob:
        return None
    try:
        return (BLOB_DIR / blob).read_bytes()
    except OSError:
        return None


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def preview(entry_id: int) -> dict:
    """What a saved version contains (text files only)."""
    entry = _find(entry_id)
    if entry is None:
        return {"error": "No such version."}
    if not entry.get("existed"):
        return {"kind": "none", "note": "The file did not exist before this change. Restoring this "
                                        "version would delete the file."}
    data = _read_blob(entry)
    if data is None:
        return {"kind": "none", "note": "This version was too large to keep, or its data is gone."}
    if b"\x00" in data[:8192]:
        return {"kind": "binary", "note": "This is not a text file, so there is no preview.",
                "size": len(data)}
    text = _decode(data)
    return {"kind": "text", "text": text[:PREVIEW_CHARS], "truncated": len(text) > PREVIEW_CHARS}


def diff_to_current(entry_id: int, resolve) -> dict:
    """What restoring this version would change in the file as it is now.
    Lines starting with + would be added, lines starting with - would be removed.
    resolve: turns a workspace-relative path into a real path (agent._safe_path)."""
    entry = _find(entry_id)
    if entry is None:
        return {"error": "No such version."}
    if not entry.get("existed"):
        return {"kind": "none", "note": "The file did not exist before this change, so restoring it "
                                        "deletes the file."}
    old = _read_blob(entry)
    if old is None:
        return {"kind": "none", "note": "This version was too large to keep, or its data is gone."}
    try:
        target = resolve(entry["path"])
    except ValueError:
        return {"error": "That path is not allowed."}
    if not target.is_file():
        return {"kind": "none", "note": "The file no longer exists. Restoring brings it back."}
    try:
        now = target.read_bytes()
    except OSError as e:
        return {"error": f"Could not read the file: {e}"}
    if b"\x00" in old[:8192] or b"\x00" in now[:8192]:
        return {"kind": "binary", "note": "This is not a text file, so there is no line-by-line comparison."}
    now_lines = _decode(now).replace("\r\n", "\n").split("\n")
    old_lines = _decode(old).replace("\r\n", "\n").split("\n")
    diff = list(difflib.unified_diff(now_lines, old_lines, lineterm="", n=2))
    if not diff:
        return {"kind": "same", "note": "This version is identical to the file as it is now."}
    body = diff[2:]
    added = sum(1 for line in body if line.startswith("+"))
    removed = sum(1 for line in body if line.startswith("-"))
    lines, used = [], 0
    for index, line in enumerate(body):
        used += len(line) + 1
        if used > DIFF_MAX_CHARS:
            lines.append(f"... ({len(body) - index} more changed lines not shown)")
            break
        lines.append(line)
    return {"kind": "diff", "lines": lines, "added": added, "removed": removed}


# ---------------------------------------------------------------
# Restoring
# ---------------------------------------------------------------
def restore(entry_id: int, resolve) -> dict:
    """Put a saved version back. The file as it is now is saved first, so the restore can be undone.
    A version of 'the file did not exist' is restored by deleting the file."""
    entry = _find(entry_id)
    if entry is None:
        return {"error": "No such version."}
    if not entry.get("kept"):
        return {"error": "That version was too large to keep, so it cannot be restored."}
    try:
        target = resolve(entry["path"])
    except ValueError:
        return {"error": "That path is not allowed."}
    snapshot = capture(target)
    try:
        if entry.get("existed"):
            data = _read_blob(entry)
            if data is None:
                return {"error": "The saved data for that version is missing."}
            _atomic_write(target, data)
            status = "restored"
        else:
            if target.is_file():
                target.unlink()
            status = "deleted"
    except OSError as e:
        return {"error": f"Could not restore it: {e}"}
    record(entry["path"], "restore", snapshot, target.stat().st_size if target.is_file() else 0)
    return {"status": status, "path": entry["path"]}