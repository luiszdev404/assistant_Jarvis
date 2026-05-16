"""
skills/browser_control.py — Playwright-based browser automation for Linux GNOME.

Supported browsers: firefox (default), chrome, brave, edge, opera, vivaldi
Each browser runs in its own _BrowserSession with a real user profile.

Actions:
  go_to, search, click, scroll, fill_form, get_text, get_url,
  press, new_tab, close_tab, screenshot, back, forward, reload,
  smart_click, smart_type, close, close_all, list_browsers, switch
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    async_playwright,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeout,
)

from skills.base import Skill


# ── URL normalizer ─────────────────────────────────────────────────────────────
def _normalize_url(url: str) -> str:
    """
    'instagram'     → 'https://instagram.com'
    'instagram.com' → 'https://instagram.com'
    Full URLs pass through unchanged.
    """
    url = url.strip()
    if not url:
        return "about:blank"
    if "://" in url:
        return url
    if "." not in url:
        url = url + ".com"
    return "https://" + url


# ── Linux browser specs ────────────────────────────────────────────────────────
_BROWSER_SPECS: dict[str, dict] = {
    "firefox": {
        "engine": "firefox",
        "bins":   ["firefox"],
    },
    "chrome": {
        "engine": "chromium",
        "bins":   ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
    },
    "brave": {
        "engine": "chromium",
        "bins":   ["brave-browser", "brave"],
    },
    "edge": {
        "engine": "chromium",
        "bins":   ["microsoft-edge", "microsoft-edge-stable"],
    },
    "opera": {
        "engine": "chromium",
        "bins":   ["opera", "opera-stable"],
    },
    "vivaldi": {
        "engine": "chromium",
        "bins":   ["vivaldi-stable", "vivaldi"],
    },
}

_ALIASES: dict[str, str] = {
    "google chrome":  "chrome",
    "google-chrome":  "chrome",
    "chromium":       "chrome",
    "microsoft edge": "edge",
    "ms edge":        "edge",
    "msedge":         "edge",
    "mozilla firefox":"firefox",
    "brave browser":  "brave",
}


def _resolve_browser(name: str) -> dict | None:
    name = _ALIASES.get(name.lower().strip(), name.lower().strip())
    spec = _BROWSER_SPECS.get(name)
    if spec is None:
        return None

    exe = None
    for b in spec.get("bins", []):
        found = shutil.which(b)
        if found:
            exe = found
            break

    return {"name": name, "engine": spec["engine"], "exe": exe}


def _detect_default_browser() -> str:
    """Return the preferred browser. Brave is always preferred when installed."""
    for b in ("brave-browser", "brave"):
        if shutil.which(b):
            return "brave"
    try:
        out = subprocess.run(
            ["xdg-settings", "get", "default-web-browser"],
            capture_output=True, text=True, timeout=5,
        ).stdout.lower()
        for kw in ("firefox", "brave", "opera", "vivaldi", "chrome", "edge"):
            if kw in out:
                return kw
    except Exception:
        pass
    return "firefox"


def _firefox_profile_dir() -> str | None:
    """Find the default Firefox profile directory on Linux."""
    base = Path.home() / ".mozilla" / "firefox"
    ini  = base / "profiles.ini"
    if not ini.exists():
        return None

    current: dict[str, str] = {}
    default_path: str | None = None

    for line in ini.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("["):
            p = current.get("Path", "")
            if p and current.get("Default") == "1":
                is_rel = current.get("IsRelative", "1") == "1"
                default_path = str(base / p) if is_rel else p
            current = {}
        elif "=" in line:
            k, _, v = line.partition("=")
            current[k.strip()] = v.strip()

    p = current.get("Path", "")
    if p and current.get("Default") == "1":
        is_rel = current.get("IsRelative", "1") == "1"
        default_path = str(base / p) if is_rel else p

    if default_path and Path(default_path).exists():
        return default_path
    return None


def _chrome_profile_dir(browser: str) -> str:
    """Return the real Chromium-based browser profile directory."""
    home = Path.home()
    cfg  = home / ".config"
    candidates = {
        "chrome":  [cfg / "google-chrome", cfg / "chromium"],
        "brave":   [cfg / "BraveSoftware" / "Brave-Browser"],
        "edge":    [cfg / "microsoft-edge"],
        "opera":   [cfg / "opera"],
        "vivaldi": [cfg / "vivaldi"],
    }
    for p in candidates.get(browser, []):
        if p.exists():
            return str(p)
    fallback = home / ".jarvis2_profiles" / browser
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


# ── Single browser session ─────────────────────────────────────────────────────

class _BrowserSession:
    """Manages a single browser instance in a dedicated asyncio thread."""

    def __init__(self, browser_name: str) -> None:
        self.browser_name = browser_name
        self._spec         = _resolve_browser(browser_name)
        self._loop:    asyncio.AbstractEventLoop | None = None
        self._thread:  threading.Thread | None          = None
        self._ready    = threading.Event()
        self._pw:      Playwright     | None = None
        self._context: BrowserContext | None = None
        self._page:    Page           | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"BrowserThread-{self.browser_name}",
        )
        self._thread.start()
        self._ready.wait(timeout=20)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_init())
        self._ready.set()
        self._loop.run_forever()

    async def _async_init(self) -> None:
        self._pw = await async_playwright().start()

    def run(self, coro, timeout: int = 60) -> str:
        if not self._loop:
            raise RuntimeError(f"Session '{self.browser_name}' not started.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._async_close(), self._loop).result(10)

    async def _async_close(self) -> None:
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
        self._context = self._page = None

    # ── Launch ─────────────────────────────────────────────────────────────────

    async def _launch(self) -> None:
        if self._context is not None:
            return
        if self._spec is None:
            raise RuntimeError(f"'{self.browser_name}' is not supported on this system.")

        engine_name = self._spec["engine"]
        exe         = self._spec.get("exe")
        engine_obj  = getattr(self._pw, engine_name)

        # Firefox
        if engine_name == "firefox":
            profile = _firefox_profile_dir() or str(
                Path.home() / ".jarvis2_profiles" / "firefox"
            )
            kwargs: dict = {
                "headless":    False,
                "slow_mo":     0,
                "viewport":    None,
                "no_viewport": True,
            }
            if exe:
                kwargs["executable_path"] = exe
            try:
                self._context = await engine_obj.launch_persistent_context(profile, **kwargs)
            except Exception as e:
                print(f"[Browser] Firefox real profile failed ({e}), using fallback")
                jarvis = str(Path.home() / ".jarvis2_profiles" / "firefox_jarvis")
                Path(jarvis).mkdir(parents=True, exist_ok=True)
                self._context = await engine_obj.launch_persistent_context(jarvis, **kwargs)
            await asyncio.sleep(0.5)
            self._page = await self._context.new_page()
            print(f"[Browser] ✅ Firefox launched")
            return

        # Chromium-based
        profile = _chrome_profile_dir(self.browser_name)
        kwargs = {
            "headless":    False,
            "slow_mo":     0,
            "viewport":    None,
            "no_viewport": True,
            "args": [
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--disable-default-apps",
                "--no-default-browser-check",
            ],
        }
        if exe:
            kwargs["executable_path"] = exe

        try:
            self._context = await engine_obj.launch_persistent_context(profile, **kwargs)
            await asyncio.sleep(0.5)
            self._page = await self._context.new_page()
            print(f"[Browser] ✅ {self.browser_name} launched with profile: {profile}")
        except Exception as e:
            jarvis_profile = str(Path.home() / ".jarvis2_profiles" / self.browser_name)
            Path(jarvis_profile).mkdir(parents=True, exist_ok=True)
            try:
                self._context = await engine_obj.launch_persistent_context(
                    jarvis_profile, **kwargs
                )
                await asyncio.sleep(0.5)
                self._page = await self._context.new_page()
                print(f"[Browser] ✅ {self.browser_name} launched with Jarvis2 profile")
            except Exception as e2:
                raise RuntimeError(f"Could not launch {self.browser_name}: {e2}") from e2

    async def _get_page(self) -> Page:
        await self._launch()
        if self._page is None or self._page.is_closed():
            self._page = await self._context.new_page()
            await asyncio.sleep(0.2)
        return self._page

    # ── Actions ────────────────────────────────────────────────────────────────

    async def go_to(self, url: str) -> str:
        url  = _normalize_url(url)
        page = await self._get_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(0.3)
        except PlaywrightTimeout:
            pass
        except Exception as e:
            print(f"[Browser] goto exception (non-fatal): {e}")
        current = page.url
        return f"Opened: {current}" if current not in ("about:blank", "", None) else f"Could not open: {url}"

    async def search(self, query: str, engine: str = "google") -> str:
        _engines = {
            "google":     "https://www.google.com/search?q=",
            "bing":       "https://www.bing.com/search?q=",
            "duckduckgo": "https://duckduckgo.com/?q=",
            "yandex":     "https://yandex.com/search/?text=",
        }
        base = _engines.get(engine.lower(), _engines["google"])
        return await self.go_to(base + query.replace(" ", "+"))

    async def click(self, selector: str | None = None, text: str | None = None) -> str:
        page = await self._get_page()
        try:
            if text:
                await page.get_by_text(text, exact=False).first.click(timeout=8_000)
                return f"Clicked: '{text}'"
            if selector:
                await page.click(selector, timeout=8_000)
                return f"Clicked selector: {selector}"
            return "Provide selector or text."
        except PlaywrightTimeout:
            return "Element not found (timeout)."
        except Exception as e:
            return f"Click error: {e}"

    async def smart_click(self, description: str) -> str:
        page = await self._get_page()
        for role in ("button", "link", "searchbox", "textbox", "menuitem", "tab"):
            try:
                loc = page.get_by_role(role, name=description)
                if await loc.count() > 0:
                    await loc.first.click(timeout=5_000)
                    return f"Clicked ({role}): '{description}'"
            except Exception:
                pass
        for attempt in (
            lambda: page.get_by_text(description, exact=False).first.click(timeout=5_000),
            lambda: page.get_by_placeholder(description, exact=False).first.click(timeout=5_000),
            lambda: page.locator(
                f'[alt*="{description}" i],[title*="{description}" i],'
                f'[aria-label*="{description}" i]'
            ).first.click(timeout=5_000),
        ):
            try:
                await attempt()
                return f"Clicked: '{description}'"
            except Exception:
                pass
        return f"Could not find element: '{description}'"

    async def smart_type(self, description: str, text: str) -> str:
        page = await self._get_page()
        candidates = [
            ("placeholder", page.get_by_placeholder(description, exact=False)),
            ("label",       page.get_by_label(description, exact=False)),
            ("role",        page.get_by_role("textbox", name=description)),
            ("searchbox",   page.get_by_role("searchbox")),
        ]
        for method, loc in candidates:
            try:
                el = loc.first
                if await el.count() == 0:
                    continue
                await el.clear()
                await el.type(text, delay=50)
                return f"Typed into ({method}): '{description}'"
            except Exception:
                continue
        return f"Could not find input: '{description}'"

    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> str:
        page = await self._get_page()
        try:
            el = page.locator(selector).first
            if clear_first:
                await el.clear()
            await el.type(text, delay=50)
            return "Text typed."
        except Exception as e:
            return f"Type error: {e}"

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        page = await self._get_page()
        try:
            y = amount if direction == "down" else -amount
            await page.mouse.wheel(0, y)
            return f"Scrolled {direction}."
        except Exception as e:
            return f"Scroll error: {e}"

    async def press(self, key: str) -> str:
        page = await self._get_page()
        try:
            await page.keyboard.press(key)
            return f"Pressed: {key}"
        except Exception as e:
            return f"Key error: {e}"

    async def get_text(self) -> str:
        page = await self._get_page()
        try:
            text = await page.inner_text("body")
            return text[:4_000]
        except Exception as e:
            return f"Could not get page text: {e}"

    async def get_url(self) -> str:
        page = await self._get_page()
        return page.url

    async def fill_form(self, fields: dict) -> str:
        page    = await self._get_page()
        results = []
        for selector, value in fields.items():
            try:
                el = page.locator(selector).first
                await el.clear()
                await el.type(str(value), delay=40)
                results.append(f"✓ {selector}")
            except Exception as e:
                results.append(f"✗ {selector}: {e}")
        return "Form filled: " + ", ".join(results)

    async def new_tab(self, url: str = "") -> str:
        page = await self._get_page()
        new  = await page.context.new_page()
        self._page = new
        if url:
            return await self.go_to(url)
        return "New tab opened."

    async def close_tab(self) -> str:
        page = self._page
        if page and not page.is_closed():
            ctx   = page.context
            await page.close()
            pages = ctx.pages
            self._page = pages[-1] if pages else None
            return "Tab closed."
        return "No active tab to close."

    async def screenshot(self, path: str | None = None) -> str:
        page = await self._get_page()
        try:
            save_path = path or str(Path.home() / "Desktop" / "jarvis2_screenshot.png")
            await page.screenshot(path=save_path, full_page=False)
            return f"Screenshot saved: {save_path}"
        except Exception as e:
            return f"Screenshot error: {e}"

    async def back(self) -> str:
        page = await self._get_page()
        try:
            await page.go_back(timeout=10_000)
            return f"Navigated back: {page.url}"
        except Exception as e:
            return f"Back error: {e}"

    async def forward(self) -> str:
        page = await self._get_page()
        try:
            await page.go_forward(timeout=10_000)
            return f"Navigated forward: {page.url}"
        except Exception as e:
            return f"Forward error: {e}"

    async def reload(self) -> str:
        page = await self._get_page()
        try:
            await page.reload(timeout=15_000)
            return f"Page reloaded: {page.url}"
        except Exception as e:
            return f"Reload error: {e}"

    async def close_browser(self) -> str:
        await self._async_close()
        return f"{self.browser_name} closed."


# ── Session registry ───────────────────────────────────────────────────────────

class _SessionRegistry:
    """Manages all active browser sessions."""

    def __init__(self) -> None:
        self._sessions:       dict[str, _BrowserSession] = {}
        self._active_browser: str                         = ""
        self._lock = threading.Lock()

    def _get_or_create(self, browser_name: str) -> _BrowserSession:
        with self._lock:
            if browser_name not in self._sessions:
                session = _BrowserSession(browser_name)
                session.start()
                self._sessions[browser_name] = session
            self._active_browser = browser_name
            return self._sessions[browser_name]

    def active(self) -> _BrowserSession | None:
        if not self._active_browser:
            default = _detect_default_browser()
            return self._get_or_create(default)
        return self._sessions.get(self._active_browser)

    def get(self, name: str) -> _BrowserSession:
        return self._get_or_create(name)

    def switch(self, name: str) -> str:
        with self._lock:
            if name not in self._sessions:
                return f"No active session for '{name}'. Open it first."
            self._active_browser = name
            return f"Switched to {name}."

    def list_browsers(self) -> list[str]:
        return list(self._sessions.keys())

    def close_all(self) -> str:
        for session in self._sessions.values():
            try:
                session.close()
            except Exception:
                pass
        self._sessions.clear()
        self._active_browser = ""
        return "All browsers closed."


_REGISTRY = _SessionRegistry()


# ── Skill ──────────────────────────────────────────────────────────────────────

class BrowserControlSkill(Skill):
    """Control any web browser using Playwright on Linux GNOME."""

    TOOL_DECLARATION = {
        "name": "browser_control",
        "description": (
            "Controls a web browser. Use for: opening websites, searching, "
            "clicking elements, filling forms, scrolling, screenshots, navigation. "
            "Specify 'browser' to target a specific browser (firefox, chrome, brave, edge, opera, vivaldi)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "go_to | search | click | smart_click | type | smart_type | "
                        "scroll | fill_form | get_text | get_url | press | new_tab | "
                        "close_tab | screenshot | back | forward | reload | "
                        "switch | list_browsers | close | close_all"
                    ),
                },
                "browser": {
                    "type": "STRING",
                    "description": "Target browser: firefox | chrome | brave | edge | opera | vivaldi",
                },
                "url":         {"type": "STRING",   "description": "URL for go_to / new_tab"},
                "query":       {"type": "STRING",   "description": "Search query"},
                "engine":      {"type": "STRING",   "description": "google | bing | duckduckgo | yandex"},
                "selector":    {"type": "STRING",   "description": "CSS selector for click/type"},
                "text":        {"type": "STRING",   "description": "Text to click or type"},
                "description": {"type": "STRING",   "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING",   "description": "up | down"},
                "amount":      {"type": "INTEGER",  "description": "Scroll amount in pixels (default: 500)"},
                "key":         {"type": "STRING",   "description": "Key name for press (e.g. Enter, Escape)"},
                "path":        {"type": "STRING",   "description": "Save path for screenshot"},
                "clear_first": {"type": "BOOLEAN",  "description": "Clear field before typing (default: true)"},
            },
            "required": ["action"],
        },
    }

    def execute(self, params: dict) -> str:
        action  = params.get("action", "").lower().strip()
        browser = params.get("browser", "").lower().strip()

        if action == "list_browsers":
            browsers = _REGISTRY.list_browsers()
            return "Active browsers: " + (", ".join(browsers) or "none")

        if action == "close_all":
            return _REGISTRY.close_all()

        if action == "switch":
            name = browser or params.get("text", "")
            return _REGISTRY.switch(name)

        # Get or create the target session
        session = _REGISTRY.get(browser) if browser else _REGISTRY.active()

        if action == "close":
            return session.run(session.close_browser())

        dispatch: dict[str, object] = {
            "go_to":       lambda: session.go_to(params.get("url", "")),
            "search":      lambda: session.search(
                params.get("query", ""), params.get("engine", "google")
            ),
            "click":       lambda: session.click(
                selector=params.get("selector"), text=params.get("text")
            ),
            "smart_click": lambda: session.smart_click(params.get("description", "")),
            "type":        lambda: session.type_text(
                params.get("selector", ""),
                params.get("text", ""),
                params.get("clear_first", True),
            ),
            "smart_type":  lambda: session.smart_type(
                params.get("description", ""), params.get("text", "")
            ),
            "scroll":      lambda: session.scroll(
                params.get("direction", "down"), params.get("amount", 500)
            ),
            "fill_form":   lambda: session.fill_form(params.get("fields", {})),
            "get_text":    lambda: session.get_text(),
            "get_url":     lambda: session.get_url(),
            "press":       lambda: session.press(params.get("key", "Enter")),
            "new_tab":     lambda: session.new_tab(params.get("url", "")),
            "close_tab":   lambda: session.close_tab(),
            "screenshot":  lambda: session.screenshot(params.get("path")),
            "back":        lambda: session.back(),
            "forward":     lambda: session.forward(),
            "reload":      lambda: session.reload(),
        }

        fn = dispatch.get(action)
        if fn is None:
            return f"Unknown browser action: '{action}'"

        try:
            return session.run(fn())
        except Exception as e:
            return f"Browser action '{action}' failed: {e}"
