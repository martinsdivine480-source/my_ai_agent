"""Starts and stops a project's dev server (npm start, npm run dev or python app.py) and keeps its
live log, so you do not have to copy commands into Command Prompt.

The server only listens on this computer. Starting a project runs that project's own code, so the
browser asks you first."""
import atexit
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_RUNNING = 3
MAX_LOG_LINES = 600
MAX_LINE_CHARS = 500
URL_RE = re.compile(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]):\d{2,5}[^\s'\"]*")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LOCK = threading.RLock()
RUNS: dict = {}  # project folder (relative to the workspace) -> run record


def detect(folder: Path) -> list:
    """The ways this project can be started: [{"label": ..., "cmd": [...], "kind": ...}]."""
    options = []
    package = folder / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        npm = shutil.which("npm") or "npm"
        if isinstance(scripts, dict):
            if "start" in scripts:
                options.append({"label": "npm start", "cmd": [npm, "start"], "kind": "node"})
            for name in ("dev", "preview"):
                if name in scripts:
                    options.append({"label": f"npm run {name}", "cmd": [npm, "run", name], "kind": "node"})
    for name in ("app.py", "main.py", "run.py"):
        if (folder / name).is_file():
            options.append({"label": f"python {name}", "cmd": [sys.executable, name], "kind": "python"})
    return options


def _kill_tree(proc) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _reader(run: dict) -> None:
    proc = run["proc"]
    try:
        for raw in iter(proc.stdout.readline, b""):
            text = ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
            if not text:
                continue
            with LOCK:
                run["seq"] += 1
                run["lines"].append((run["seq"], text[:MAX_LINE_CHARS]))
                if run["url"] is None:
                    found = URL_RE.search(text)
                    if found:
                        run["url"] = found.group(0).replace("0.0.0.0", "localhost").rstrip("/.,;")
    except Exception:
        pass
    finally:
        run["exit_code"] = proc.wait()
        with LOCK:
            run["seq"] += 1
            run["lines"].append((run["seq"], f"[process ended with exit code {run['exit_code']}]"))


def start(key: str, folder: Path, index: int = 0, base_env: dict | None = None) -> dict:
    options = detect(folder)
    if not options:
        return {"error": "No way to start this project was found (it needs a package.json with a start "
                         "or dev script, or an app.py)."}
    option = options[index] if 0 <= index < len(options) else options[0]
    if option["kind"] == "node":
        if shutil.which("npm") is None:
            return {"error": "Node.js (npm) was not found on this computer."}
        if not (folder / "node_modules").is_dir():
            return {"error": "node_modules is missing. Run npm install in this project folder first."}

    with LOCK:
        stop(key)
        running = sum(1 for r in RUNS.values() if r["proc"].poll() is None)
        if running >= MAX_RUNNING:
            return {"error": f"At most {MAX_RUNNING} projects can run at once. Stop one first."}
        env = dict(base_env if base_env is not None else os.environ)
        env.update({"PYTHONUNBUFFERED": "1", "BROWSER": "none", "FORCE_COLOR": "0", "NO_COLOR": "1",
                    "HOST": "127.0.0.1"})
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(option["cmd"], cwd=str(folder), env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
        except OSError as e:
            return {"error": f"Could not start it: {e}"}
        run = {"key": key, "proc": proc, "command": option["label"], "started": time.time(),
               "lines": collections.deque(maxlen=MAX_LOG_LINES), "seq": 0, "url": None, "exit_code": None}
        RUNS[key] = run
    threading.Thread(target=_reader, args=(run,), daemon=True).start()
    return {"status": "started", "command": option["label"]}


def stop(key: str) -> dict:
    with LOCK:
        run = RUNS.get(key)
    if run is None or run["proc"].poll() is not None:
        return {"status": "not running"}
    _kill_tree(run["proc"])
    try:
        run["proc"].wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return {"status": "stopped"}


def status(key: str, after: int = 0) -> dict:
    with LOCK:
        run = RUNS.get(key)
        if run is None:
            return {"running": False, "known": False, "lines": [], "next": 0}
        lines = [{"n": n, "text": t} for (n, t) in run["lines"] if n > after]
        return {
            "known": True,
            "running": run["proc"].poll() is None,
            "command": run["command"],
            "url": run["url"],
            "started": run["started"],
            "exit_code": run["exit_code"],
            "lines": lines,
            "next": run["seq"],
        }


def all_status() -> dict:
    with LOCK:
        return {k: {"running": r["proc"].poll() is None, "command": r["command"], "url": r["url"]}
                for k, r in RUNS.items()}


def stop_all() -> None:
    for key in list(RUNS):
        stop(key)


atexit.register(stop_all)