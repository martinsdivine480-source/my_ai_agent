"""Live app tester: opens a web page in a headless browser at phone size, takes a screenshot and
reports what went wrong (console errors, crashes, failed requests, a blank page).

Needs:  pip install playwright
It uses the Microsoft Edge or Google Chrome already installed on this computer, so
"python -m playwright install chromium" is not required.

By default only your own computer's addresses (localhost / 127.0.0.1) are opened without approval;
the agent asks you first for any other address."""
import urllib.parse
from datetime import datetime
from pathlib import Path

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAX_ITEMS = 15
MAX_ITEM_CHARS = 400
NAVIGATION_TIMEOUT_MS = 30000
PHONE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")

# Runs inside the page after it has loaded: what is on the screen?
PAGE_PROBE = """() => {
  const body = document.body;
  const text = ((body && body.innerText) || "").trim();
  const root = document.querySelector("#root, #app, #__next");
  const all = body ? Array.from(body.querySelectorAll("*")).slice(0, 2000) : [];
  const visible = all.filter(function (el) {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
  }).length;
  return {
    title: document.title || "",
    text: text.slice(0, 1200),
    textLength: text.length,
    rootFound: !!root,
    rootChildren: root ? root.children.length : null,
    visibleElements: visible,
    images: document.images.length,
    overflowX: document.documentElement.scrollWidth > window.innerWidth + 1,
    viewport: [window.innerWidth, window.innerHeight]
  };
}"""


def is_local_url(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit((url or "").strip())
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (parts.hostname or "").lower() in LOCAL_HOSTS


def _clip(text) -> str:
    text = str(text).strip()
    return text if len(text) <= MAX_ITEM_CHARS else text[:MAX_ITEM_CHARS] + "..."


def _unique(items: list) -> list:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:MAX_ITEMS]


def _is_favicon(url: str) -> bool:
    return (url or "").split("?")[0].lower().endswith("favicon.ico")


def _launch_browser(p, playwright_error, local):
    """Edge or Chrome from this computer first, then Playwright's own Chromium if it is installed.
    Returns (browser, last_error)."""
    last_error = None
    args = ["--no-proxy-server"] if local else []  # a local page never needs a proxy
    for options in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, args=args, **options), None
        except playwright_error as e:
            last_error = e
    return None, last_error


def check_page(url: str, shot_dir: Path, workspace: Path, width: int = 390, height: int = 844,
               wait_ms: int = 3000, full_page: bool = False) -> dict:
    url = (url or "").strip()
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return {"error": "That is not a valid web address."}
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return {"error": "Give a full address such as http://localhost:3000"}

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"error": "Playwright is not installed. Run: pip install playwright"}

    width = max(320, min(int(width), 1920))
    height = max(480, min(int(height), 2400))
    wait_ms = max(500, min(int(wait_ms), 10000))
    console_errors, page_errors, failed = [], [], []
    status = None

    with sync_playwright() as p:
        browser, launch_error = _launch_browser(p, PlaywrightError, is_local_url(url))
        if browser is None:
            return {"error": "No browser could be started. Install Microsoft Edge or Google Chrome "
                             f"({_clip(launch_error)})"}
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height}, device_scale_factor=2,
                is_mobile=width < 768, has_touch=width < 768,
                user_agent=PHONE_UA if width < 768 else None,
            )
            page = context.new_page()

            def on_console(message):
                if message.type not in ("error", "warning"):
                    return
                text = message.text
                if "Failed to load resource" in text:
                    try:
                        resource = (message.location or {}).get("url", "")
                    except Exception:
                        resource = ""
                    if _is_favicon(resource):
                        return  # browsers ask for /favicon.ico by themselves: not a page problem
                    if resource:
                        text = f"{text} ({resource})"  # say WHICH file failed
                console_errors.append(f"[{message.type}] {_clip(text)}")

            def on_page_error(error):
                page_errors.append(_clip(error))

            def on_request_failed(request):
                if _is_favicon(request.url):
                    return
                try:
                    reason = request.failure
                except Exception:
                    reason = "failed"
                failed.append(f"{_clip(request.url)} ({reason})")

            def on_response(response):
                if response.status >= 400 and not _is_favicon(response.url):
                    failed.append(f"{_clip(response.url)} (HTTP {response.status})")

            page.on("console", on_console)
            page.on("pageerror", on_page_error)
            page.on("requestfailed", on_request_failed)
            page.on("response", on_response)

            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
                status = response.status if response else None
            except PlaywrightError as e:
                return {"error": f"Could not open {url}. Is the server running? ({_clip(e)})"}
            page.wait_for_timeout(wait_ms)  # let the app draw itself and log its errors

            probe = page.evaluate(PAGE_PROBE)
            shot_dir.mkdir(parents=True, exist_ok=True)
            name = datetime.now().strftime("screenshot-%Y%m%d-%H%M%S.png")
            shot = shot_dir / name
            page.screenshot(path=str(shot), full_page=bool(full_page))
        finally:
            browser.close()

    blank = (probe["textLength"] < 3 and probe["images"] == 0 and probe["visibleElements"] <= 3) or \
            (probe["rootFound"] and probe["rootChildren"] == 0)
    console_errors, page_errors, failed = _unique(console_errors), _unique(page_errors), _unique(failed)
    ok = not blank and not page_errors and not any(c.startswith("[error]") for c in console_errors) \
        and (status is None or status < 400)

    result = {
        "url": url,
        "status": status,
        "ok": ok,
        "blank_page": blank,
        "title": probe["title"],
        "viewport": probe["viewport"],
        "page_errors": page_errors,
        "console_messages": console_errors,
        "failed_requests": failed,
        "horizontal_scroll": probe["overflowX"],
        "visible_text": probe["text"][:600],
        "screenshot": shot.relative_to(workspace).as_posix(),
    }
    if blank:
        result["hint"] = ("The page is blank. The errors above usually name the cause, for example a "
                          "missing import such as useEffect or an undefined component.")
    if probe["overflowX"]:
        result["layout_note"] = "The page is wider than the screen, so it scrolls sideways on a phone."
    return result