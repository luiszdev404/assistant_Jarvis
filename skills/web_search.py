"""
skills/web_search.py — Web search via Gemini (Google Search grounding)
with DuckDuckGo fallback + Gemini synthesis.

Modes:
  - search (default): single-query search
  - news:             recent news/events
  - compare:          side-by-side comparison of items
"""
from __future__ import annotations

import concurrent.futures
import time

from google import genai

from core.settings import LITE_MODEL
from skills.base import Skill, _get_ddgs


class WebSearchSkill(Skill):
    """Search the web using Gemini grounding or DuckDuckGo + Gemini synthesis."""

    TOOL_DECLARATION = {
        "name": "web_search",
        "description": (
            "Search the web for any information: current events, facts, prices, "
            "news, or real-time data. Can also compare items or search for recent news."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "The search query",
                },
                "mode": {
                    "type": "STRING",
                    "description": "search (default) | news | compare",
                },
                "items": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Items to compare (for compare mode)",
                },
                "aspect": {
                    "type": "STRING",
                    "description": "Aspect to compare: price | specs | reviews | performance | general",
                },
            },
            "required": ["query"],
        },
    }

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key)
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

    # ── Gemini grounded search ────────────────────────────────────────────────

    def _gemini_search(self, query: str, news: bool = False) -> tuple[str, list[str]]:
        """Returns (answer_text, list_of_source_urls)."""
        prompt = query
        if news:
            prompt = f"Latest news and recent developments about: {query}"

        response = self._client.models.generate_content(
            model=LITE_MODEL,
            contents=prompt,
            config={"tools": [{"google_search": {}}]},
        )

        text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        text = text.strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")

        sources = self._extract_sources(response)
        return text, sources

    def _extract_sources(self, response) -> list[str]:
        """Extract grounding source URLs from a Gemini response."""
        sources: list[str] = []
        try:
            candidates = response.candidates
            if not candidates:
                return sources
            metadata = getattr(candidates[0], "grounding_metadata", None)
            if not metadata:
                return sources
            chunks = getattr(metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    sources.append(web.uri)
        except Exception:
            pass
        return sources

    # ── DuckDuckGo with retry ─────────────────────────────────────────────────

    def _ddg_search(self, query: str, max_results: int = 6) -> list[dict]:
        DDGS = _get_ddgs()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "title":   r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url":     r.get("href", ""),
                    }
                    for r in results
                ]
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"DuckDuckGo failed after 3 attempts: {last_exc}")

    # ── Synthesize DDG results with Gemini ────────────────────────────────────

    def _synthesize_with_gemini(self, query: str, results: list[dict]) -> str:
        """Feed DDG snippets to Gemini for a clean, synthesized answer."""
        if not results:
            return f"No results found for: {query}"

        snippets = "\n\n".join(
            f"[{i+1}] {r['title']}\n{r['snippet']}\n{r['url']}"
            for i, r in enumerate(results)
            if r.get("snippet")
        )

        prompt = (
            f"Based on the following search results, answer this query concisely and accurately:\n"
            f"Query: {query}\n\n"
            f"Search results:\n{snippets}\n\n"
            f"Provide a clear, direct answer. Mention sources by number when relevant."
        )

        response = self._client.models.generate_content(
            model=LITE_MODEL,
            contents=prompt,
        )
        text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        text = text.strip()

        if text:
            sources_block = "\n".join(
                f"[{i+1}] {r['url']}" for i, r in enumerate(results) if r.get("url")
            )
            return f"{text}\n\nSources:\n{sources_block}"

        return self._format_ddg_raw(query, results)

    def _format_ddg_raw(self, query: str, results: list[dict]) -> str:
        """Plain formatting fallback if synthesis also fails."""
        if not results:
            return f"No results found for: {query}"
        lines = [f"Results for: {query}\n"]
        for i, r in enumerate(results, 1):
            if r.get("title"):   lines.append(f"{i}. {r['title']}")
            if r.get("snippet"): lines.append(f"   {r['snippet']}")
            if r.get("url"):     lines.append(f"   {r['url']}")
            lines.append("")
        return "\n".join(lines).strip()

    def _format_with_sources(self, text: str, sources: list[str]) -> str:
        if not sources:
            return text
        unique = list(dict.fromkeys(sources))[:5]
        src_block = "\n".join(f"  • {u}" for u in unique)
        return f"{text}\n\nSources:\n{src_block}"

    # ── Compare mode (parallel) ───────────────────────────────────────────────

    def _compare(self, items: list[str], aspect: str, base_query: str) -> str:
        query = (
            f"Compare {', '.join(items)} in terms of {aspect}. "
            "Give specific facts, data, and a clear recommendation."
        )
        try:
            text, sources = self._gemini_search(query)
            return self._format_with_sources(text, sources)
        except Exception as e:
            self.log(f"Gemini compare failed: {e} — falling back to DDG parallel")

        def fetch(item: str) -> tuple[str, list[dict]]:
            try:
                return item, self._ddg_search(f"{item} {aspect}", max_results=3)
            except Exception:
                return item, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as ex:
            item_results = dict(ex.map(lambda i: fetch(i), items))

        snippets = "\n\n".join(
            f"### {item}\n" + "\n".join(
                f"- {r['snippet']}" for r in item_results.get(item, []) if r.get("snippet")
            )
            for item in items
        )
        prompt = (
            f"Compare {', '.join(items)} in terms of {aspect} using these search results:\n\n"
            f"{snippets}\n\n"
            f"Give a structured comparison and a recommendation."
        )
        try:
            response = self._client.models.generate_content(
                model=LITE_MODEL,
                contents=prompt,
            )
            text = "".join(
                part.text for part in response.candidates[0].content.parts
                if hasattr(part, "text") and part.text
            ).strip()
            if text:
                return text
        except Exception:
            pass

        lines = [f"Comparison — {aspect.upper()}", "─" * 40]
        for item in items:
            lines.append(f"\n▸ {item}")
            for r in item_results.get(item, [])[:2]:
                if r.get("snippet"):
                    lines.append(f"  • {r['snippet']}")
        return "\n".join(lines)

    # ── Main execute ──────────────────────────────────────────────────────────

    def execute(self, params: dict) -> str:
        query  = params.get("query", "").strip()
        mode   = params.get("mode", "search").lower().strip()
        items  = params.get("items", [])
        aspect = params.get("aspect", "general").strip() or "general"

        if not query and not items:
            return "Please provide a search query."

        if items and mode != "compare":
            mode = "compare"

        self.log(f"Query: {query!r}  Mode: {mode}")

        if mode == "compare" and items:
            self.log(f"Comparing: {items}")
            return self._compare(items, aspect, query)

        is_news = (mode == "news")

        # 1. Try Gemini grounded search
        self.log(f"Trying Gemini grounded search (news={is_news})...")
        try:
            text, sources = self._gemini_search(query, news=is_news)
            self.log(f"Gemini OK — {len(sources)} source(s).")
            return self._format_with_sources(text, sources)
        except Exception as e:
            self.log(f"Gemini grounding failed ({e}) — falling back to DuckDuckGo...")

        # 2. Try DuckDuckGo
        try:
            results = self._ddg_search(query)
        except Exception as ddg_e:
            self.log(f"DuckDuckGo also failed: {ddg_e}")
            return f"Search failed: no available search backend."

        if not results:
            return f"No results found for: {query}"

        self.log(f"DDG: {len(results)} result(s). Synthesizing...")

        # 3. Try Gemini synthesis over DDG results; fall back to raw format
        try:
            return self._synthesize_with_gemini(query, results)
        except Exception as syn_e:
            self.log(f"Gemini synthesis failed ({syn_e}) — returning raw DDG results.")
            return self._format_ddg_raw(query, results)
