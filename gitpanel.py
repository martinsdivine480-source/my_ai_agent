"""Git helper for the Git panel: list changed files, suggest a commit message, commit and push.

Needs Git installed (https://git-scm.com). It only works in a project folder that is itself a Git
repository. It never forces a push, never asks for a password (Git Credential Manager or a saved
login must already work), and refuses to commit secrets such as .env files."""
import os
import re
import shutil
import subprocess
from pathlib import Path

GIT_TIMEOUT = 30
PUSH_TIMEOUT = 120
MAX_OUTPUT = 4000
MAX_FILES_LISTED = 300
MAX_MESSAGE_CHARS = 2000
# Folders that must be in .gitignore before everything is staged with "git add -A".
HEAVY_FOLDERS = ("node_modules", ".venv", "venv")
# Files that look like secrets. (.env.example and similar template files are fine.)
SENSITIVE_RE = re.compile(
    r"(^|/)(\.env(\.[\w.-]+)?|id_rsa[\w.-]*|id_ed25519[\w.-]*|[\w.-]+\.(pem|key|p12|pfx)|"
    r"credentials\.json|secrets?\.(json|ya?ml|toml))$", re.I)
TEMPLATE_ENDINGS = (".example", ".sample", ".template", ".dist")


class GitError(Exception):
    """A problem to show to the user as it is."""


def git_installed() -> bool:
    return shutil.which("git") is not None


def is_sensitive(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    if lowered.endswith(TEMPLATE_ENDINGS):
        return False
    return bool(SENSITIVE_RE.search(lowered))


def _git(folder: Path, args: list, timeout: int = GIT_TIMEOUT, env: dict | None = None):
    """Run one git command. Returns (exit code, stdout, stderr). Raises GitError."""
    exe = shutil.which("git")
    if exe is None:
        raise GitError("Git is not installed on this computer. Install it from https://git-scm.com "
                       "and restart the agent.")
    run_env = dict(env if env is not None else os.environ)
    run_env["GIT_TERMINAL_PROMPT"] = "0"  # never wait for a typed password
    run_env["LC_ALL"] = "C"               # English messages, so the hints below can recognise them
    options = {}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run([exe, "-c", "core.quotepath=off", *args], cwd=str(folder), env=run_env,
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, **options)
    except subprocess.TimeoutExpired:
        raise GitError(f"git {args[0]} took longer than {timeout} seconds and was stopped.")
    except OSError as e:
        raise GitError(f"Could not run git: {e}")
    return (done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace"))


def _is_own_repo(folder: Path, env) -> bool:
    """True when the folder itself is the top of a Git repository (not a folder inside a bigger one)."""
    code, out, _ = _git(folder, ["rev-parse", "--show-toplevel"], env=env)
    if code != 0 or not out.strip():
        return False
    try:
        return os.path.samefile(out.strip(), str(folder))
    except OSError:
        return False


def _label(code: str) -> str:
    if code == "??":
        return "new"
    if "U" in code or code in ("AA", "DD"):
        return "conflict"
    if "R" in code:
        return "renamed"
    if "D" in code:
        return "deleted"
    if "A" in code:
        return "added"
    if "M" in code:
        return "modified"
    return "changed"


def status(folder: Path, env: dict | None = None) -> dict:
    """The state of the project's repository: branch, changed files, remotes."""
    if not git_installed():
        return {"git_installed": False, "is_repo": False,
                "note": "Git is not installed on this computer. Install it from https://git-scm.com."}
    try:
        if not _is_own_repo(folder, env):
            return {"git_installed": True, "is_repo": False,
                    "note": "This project folder is not a Git repository. Open Command Prompt in the "
                            "folder and run: git init"}
        code, out, err = _git(folder, ["status", "--porcelain=v1", "-z", "--branch"], env=env)
        if code != 0:
            return {"git_installed": True, "is_repo": False, "note": (err or out).strip()[:300]}
        parts = out.split("\0")
        head = parts[0][3:] if parts and parts[0].startswith("## ") else ""
        ahead = behind = 0
        track = re.search(r" \[(.*)\]$", head)
        if track:
            found = re.search(r"ahead (\d+)", track.group(1))
            ahead = int(found.group(1)) if found else 0
            found = re.search(r"behind (\d+)", track.group(1))
            behind = int(found.group(1)) if found else 0
            head = head[:track.start()]
        upstream = None
        if "..." in head:
            branch, upstream = head.split("...", 1)
        else:
            branch = head
        branch = re.sub(r"^(No commits yet on|Initial commit on) ", "", branch).strip()
        detached = branch.startswith("HEAD (no branch)")

        files, i = [], 1
        while i < len(parts):
            entry = parts[i]
            i += 1
            if len(entry) < 4:
                continue
            xy, path = entry[:2], entry[3:]
            if xy[0] in "RC":  # a rename lists the old name in the next record
                i += 1
            files.append({"path": path, "code": xy.strip() or xy, "label": _label(xy),
                          "sensitive": is_sensitive(path)})
        _, remotes_out, _ = _git(folder, ["remote"], env=env)
        has_head, _, _ = _git(folder, ["rev-parse", "--verify", "-q", "HEAD"], env=env)
        return {
            "git_installed": True,
            "is_repo": True,
            "branch": None if detached else branch,
            "detached": detached,
            "upstream": upstream,
            "ahead": ahead,
            "behind": behind,
            "files": files[:MAX_FILES_LISTED],
            "total_files": len(files),
            "clean": not files,
            "remotes": remotes_out.split(),
            "has_commits": has_head == 0,
            "sensitive": [f["path"] for f in files if f["sensitive"]],
        }
    except GitError as e:
        return {"git_installed": True, "is_repo": False, "note": str(e)}


def log(folder: Path, count: int = 8, env: dict | None = None) -> list:
    """The latest commits: [{hash, message, author, when}]."""
    try:
        code, out, _ = _git(folder, ["log", f"-n{max(1, min(count, 30))}",
                                     "--pretty=format:%h%x1f%s%x1f%an%x1f%ar"], env=env)
    except GitError:
        return []
    if code != 0:
        return []
    commits = []
    for line in out.splitlines():
        pieces = line.split("\x1f")
        if len(pieces) == 4:
            commits.append({"hash": pieces[0], "message": pieces[1], "author": pieces[2], "when": pieces[3]})
    return commits


def _names(paths: list) -> str:
    shown = [Path(p).name for p in paths[:3]]
    text = ", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else shown[0]
    if len(paths) > 3:
        text = ", ".join(shown) + f" and {len(paths) - 3} more"
    return text


def suggest_message(files: list) -> str:
    """A short commit message made from the names of the changed files (no AI needed)."""
    added = [f["path"] for f in files if f["label"] in ("new", "added")]
    changed = [f["path"] for f in files if f["label"] in ("modified", "renamed", "changed", "conflict")]
    removed = [f["path"] for f in files if f["label"] == "deleted"]
    parts = []
    if changed:
        parts.append("update " + _names(changed))
    if added:
        parts.append("add " + _names(added))
    if removed:
        parts.append("remove " + _names(removed))
    if not parts:
        return ""
    message = "; ".join(parts)
    if len(message) > 72:  # too long for a subject line: say it with counts instead
        counts = []
        if changed:
            counts.append(f"update {len(changed)} file" + ("s" if len(changed) != 1 else ""))
        if added:
            counts.append(f"add {len(added)} file" + ("s" if len(added) != 1 else ""))
        if removed:
            counts.append(f"remove {len(removed)} file" + ("s" if len(removed) != 1 else ""))
        message = "; ".join(counts)
    return message[0].upper() + message[1:]


def diff_text(folder: Path, max_chars: int = 6000, env: dict | None = None) -> str:
    """The changes in the working folder as text (a summary plus the start of the diff), for writing a
    better commit message. New files are listed by name."""
    try:
        code, stat, _ = _git(folder, ["diff", "HEAD", "--stat"], env=env)
        _, body, _ = _git(folder, ["diff", "HEAD", "--no-color", "-U1"], env=env)
        _, new_files, _ = _git(folder, ["ls-files", "--others", "--exclude-standard"], env=env)
    except GitError:
        return ""
    text = (stat if code == 0 else "") + "\n" + (body if code == 0 else "")
    if new_files.strip():
        text += "\nNew files:\n" + new_files.strip()
    return text.strip()[:max_chars]


def _hint(text: str) -> str:
    lowered = text.lower()
    if "author identity unknown" in lowered or "please tell me who you are" in lowered:
        return ('Git does not know who you are yet. Run these once in Command Prompt: '
                'git config --global user.name "Your Name"   and   '
                'git config --global user.email "you@example.com"')
    if "authentication failed" in lowered or "could not read username" in lowered \
            or "terminal prompts disabled" in lowered or "permission denied" in lowered:
        return ("Git could not sign in to the remote. Sign in once outside this panel (run git push in "
                "Command Prompt, or use Git Credential Manager), then try again.")
    if "non-fast-forward" in lowered or "rejected" in lowered or "fetch first" in lowered:
        return ("The remote has changes you do not have. Pull them first (git pull) and resolve any "
                "conflicts. This panel never forces a push.")
    return ""


def commit(folder: Path, message: str, env: dict | None = None) -> dict:
    """Stage every change (git add -A, which respects .gitignore) and commit it."""
    message = (message or "").strip()
    if not message:
        return {"error": "The commit message is empty."}
    if len(message) > MAX_MESSAGE_CHARS:
        return {"error": f"The commit message is too long (limit is {MAX_MESSAGE_CHARS} characters)."}
    st = status(folder, env)
    if not st.get("is_repo"):
        return {"error": st.get("note") or "This folder is not a Git repository."}
    if st["clean"]:
        return {"error": "There is nothing to commit."}
    if st["sensitive"]:
        return {"error": "Not committed: these files look like secrets: " + ", ".join(st["sensitive"][:6]) +
                         ". Add them to .gitignore (or remove them), then try again."}
    try:
        for name in HEAVY_FOLDERS:
            if (folder / name).is_dir():
                code, _, _ = _git(folder, ["check-ignore", "-q", name], env=env)
                if code != 0:
                    return {"error": f"Not committed: the folder '{name}' is not in .gitignore, so it "
                                     "would be added to the repository. Add a line with its name to "
                                     ".gitignore first."}
        code, out, err = _git(folder, ["add", "-A"], env=env)
        if code != 0:
            return {"error": "git add failed: " + (err or out).strip()[:MAX_OUTPUT]}
        code, out, err = _git(folder, ["commit", "-m", message], env=env)
    except GitError as e:
        return {"error": str(e)}
    text = (out + "\n" + err).strip()
    if code != 0:
        return {"error": "git commit failed: " + text[-MAX_OUTPUT:], "hint": _hint(text)}
    return {"status": "committed", "output": text[-MAX_OUTPUT:]}


def push(folder: Path, env: dict | None = None) -> dict:
    """Push the current branch. Never uses --force."""
    st = status(folder, env)
    if not st.get("is_repo"):
        return {"error": st.get("note") or "This folder is not a Git repository."}
    if st["detached"] or not st["branch"]:
        return {"error": "You are not on a branch (detached HEAD), so there is nothing to push."}
    if not st["has_commits"]:
        return {"error": "There are no commits yet. Commit first."}
    if not st["remotes"]:
        return {"error": "This repository has no remote. Add one first, for example: "
                         "git remote add origin https://github.com/you/project.git"}
    if st["upstream"]:
        args = ["push"]
    else:
        remote = "origin" if "origin" in st["remotes"] else st["remotes"][0]
        args = ["push", "-u", remote, st["branch"]]
    try:
        code, out, err = _git(folder, args, timeout=PUSH_TIMEOUT, env=env)
    except GitError as e:
        return {"error": str(e)}
    text = (out + "\n" + err).strip()
    if code != 0:
        return {"error": "git push failed: " + text[-MAX_OUTPUT:], "hint": _hint(text)}
    return {"status": "pushed", "output": text[-MAX_OUTPUT:] or "Pushed."}