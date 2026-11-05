"""Shows your overdue and due-today tasks as a Windows notification.

  python reminder.py --install 08:00   run it every morning at 08:00 (Windows Task Scheduler)
  python reminder.py --uninstall       stop the morning reminder
  python reminder.py --now             show it right now (also says so when nothing is due)
  python reminder.py                   what Task Scheduler runs: shows a notification only if
                                       something is overdue or due today

It only reads tasks.json. It does not need the agent or the internet."""
import base64
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_FILE = HERE / "tasks.json"
TASK_NAME = "MyAIAgentMorningReminder"


def load_open_tasks() -> list:
    try:
        data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict) and t.get("status") == "open"]


def build_message():
    """Returns (title, text), or None when nothing is overdue or due today."""
    today = date.today().isoformat()
    tasks = load_open_tasks()
    overdue = [t for t in tasks if t.get("due") and t["due"] < today]
    due_today = [t for t in tasks if t.get("due") == today]
    if not overdue and not due_today:
        return None

    def names(items):
        shown = "; ".join(str(t.get("title", "?")) for t in items[:4])
        return shown + (f" (+{len(items) - 4} more)" if len(items) > 4 else "")

    parts = []
    if overdue:
        parts.append(f"Overdue ({len(overdue)}): {names(overdue)}")
    if due_today:
        parts.append(f"Due today ({len(due_today)}): {names(due_today)}")
    return "Your tasks for today", "\n".join(parts)


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def notify(title: str, text: str) -> None:
    if sys.platform != "win32":
        print(f"{title}\n{text}")
        return
    script = "\n".join([
        "Add-Type -AssemblyName System.Windows.Forms",
        "Add-Type -AssemblyName System.Drawing",
        "$n = New-Object System.Windows.Forms.NotifyIcon",
        "$n.Icon = [System.Drawing.SystemIcons]::Information",
        f"$n.BalloonTipTitle = {_ps_quote(title[:60])}",
        f"$n.BalloonTipText = {_ps_quote(text[:250])}",
        "$n.Visible = $true",
        "$n.ShowBalloonTip(15000)",
        "Start-Sleep -Seconds 16",
        "$n.Dispose()",
    ])
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.Popen(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-EncodedCommand", encoded],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def install(time_text: str) -> None:
    if sys.platform != "win32":
        print("Installing the reminder only works on Windows.")
        return
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time_text):
        print("Give the time as HH:MM, for example 08:00")
        return
    exe = Path(sys.executable)
    runner = exe.with_name("pythonw.exe")
    if not runner.exists():
        runner = exe
    command = f'"{runner}" "{Path(__file__).resolve()}"'
    result = subprocess.run(
        ["schtasks", "/Create", "/TN", TASK_NAME, "/TR", command, "/SC", "DAILY", "/ST", time_text, "/F"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Done. Windows will show your tasks every day at {time_text}.")
        print("If the computer is off or asleep at that time, the reminder is skipped that day.")
    else:
        print("Could not create the scheduled task:")
        print((result.stderr or result.stdout).strip())


def uninstall() -> None:
    if sys.platform != "win32":
        print("Nothing to remove on this system.")
        return
    result = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True, text=True)
    print("The morning reminder was removed." if result.returncode == 0
          else "Could not remove it (it may not be installed).")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--install":
        install(args[1] if len(args) > 1 else "08:00")
    elif args and args[0] == "--uninstall":
        uninstall()
    elif args and args[0] == "--now":
        message = build_message()
        notify(*(message or ("Your tasks for today", "Nothing is overdue or due today.")))
    else:
        message = build_message()
        if message:
            notify(*message)


if __name__ == "__main__":
    main()