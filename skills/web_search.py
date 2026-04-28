"""
skills/web_search.py — Web search via Gemini (with Google Search grounding)
and DuckDuckGo as a fallback.

Actions:
  - search (default): single-query search
  - compare: compare multiple items on a specific aspect
"""
from __future__ import annotations

from skills.base import Skill


class WebSearchSkill(Skill):
    """Search the web using Gemini grounding or DuckDuckGo fallback."""

    TOOL_DECLARATION = {
        "name": "web_search",
        "description": (
            "Search the web for any information. Use this for current events, "
            "facts, prices, news, or any real-time information. "
            "Can also compare items (e.g., two products, two technologies)."
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
                    "description": "search (default) or compare",
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

    # ── Gemini search ─────────────────────────────────────────────────────────

    def _gemini_search(self, query: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=query,
            config={"tools": [{"google_search": {}}]},
        )
        text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        text = text.strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text

    # ── DuckDuckGo fallback ───────────────────────────────────────────────────

    def _ddg_search(self, query: str, max_results: int = 6) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        return results

    def _format_ddg(self, query: str, results: list[dict]) -> str:
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            if r.get("title"):   lines.append(f"{i}. {r['title']}")
            if r.get("snippet"): lines.append(f"   {r['snippet']}")
            if r.get("url"):     lines.append(f"   {r['url']}")
            lines.append("")
        return "\n".join(lines).strip()

    # ── Compare mode ──────────────────────────────────────────────────────────

    def _compare(self, items: list[str], aspect: str) -> str:
        query = (
            f"Compare {', '.join(items)} in terms of {aspect}. "
            "Give specific facts and data."
        )
        try:
            return self._gemini_search(query)
        except Exception as e:
            self.log(f"Gemini compare failed: {e} — falling back to DDG")

        all_results: dict[str, list] = {}
        for item in items:
            try:
                all_results[item] = self._ddg_search(f"{item} {aspect}", max_results=3)
            except Exception:
                all_results[item] = []

        lines = [f"Comparison — {aspect.upper()}", "─" * 40]
        for item in items:
            lines.append(f"\n▸ {item}")
            for r in all_results.get(item, [])[:2]:
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

        try:
            if mode == "compare" and items:
                self.log(f"Comparing: {items}")
                return self._compare(items, aspect)

            self.log("Trying Gemini grounded search...")
            try:
                result = self._gemini_search(query)
                self.log("Gemini OK.")
                return result
            except Exception as e:
                self.log(f"Gemini failed ({e}) — trying DuckDuckGo...")
                results = self._ddg_search(query)
                result  = self._format_ddg(query, results)
                self.log(f"DDG: {len(results)} result(s).")
                return result

        except Exception as e:
            self.log(f"All backends failed: {e}")
            return f"Search failed: {e}"
