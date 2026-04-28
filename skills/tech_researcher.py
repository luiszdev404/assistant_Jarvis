"""
skills/tech_researcher.py — Automated tech research with Obsidian Markdown output.

This skill:
  1. Searches multiple sources (Gemini grounding + DuckDuckGo)
  2. Synthesizes the information into a structured research document
  3. Saves it as Obsidian-compatible Markdown to ~/Documents/Obsidian/Jarvis-Research/
  4. Optionally opens the file in Obsidian or the default text editor

Obsidian features used:
  - YAML frontmatter (tags, date, sources)
  - Callouts (> [!note], > [!tip], > [!important])
  - WikiLinks for related topics [[topic]]
  - Sections: Introduction, Key Concepts, Deep Dive, Trends, References
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

from core.settings import OBSIDIAN_VAULT
from skills.base import Skill


def _sanitize_filename(title: str) -> str:
    """Convert a topic title to a safe filename."""
    safe = re.sub(r'[^\w\s-]', '', title.lower())
    safe = re.sub(r'[\s_]+', '-', safe).strip('-')
    return safe[:80]


def _open_file(filepath: Path) -> None:
    """Open file with Obsidian URI scheme if available, else xdg-open."""
    try:
        import shutil
        if shutil.which("obsidian"):
            # Obsidian URI: obsidian://open?path=<absolute_path>
            uri = f"obsidian://open?path={filepath}"
            subprocess.Popen(
                ["obsidian", uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
    except Exception:
        pass
    try:
        subprocess.Popen(
            ["xdg-open", str(filepath)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[tech_researcher] Could not open file: {e}")


class TechResearcherSkill(Skill):
    """Research a technology topic and generate an Obsidian-compatible Markdown note."""

    TOOL_DECLARATION = {
        "name": "tech_researcher",
        "description": (
            "Research any technology topic in depth and generate a structured Markdown note "
            "for Obsidian. Covers: concepts, trends, use cases, pros/cons, references. "
            "Also generates structured content like YouTube scripts, blog posts, or guides. "
            "Use when the user asks to research, investigate, or create content about a tech topic."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic": {
                    "type": "STRING",
                    "description": "The technology topic to research (e.g. 'Rust programming language', 'WebAssembly')",
                },
                "output_type": {
                    "type": "STRING",
                    "description": (
                        "Type of output: "
                        "note (default — Obsidian research note) | "
                        "youtube_script (structured script) | "
                        "blog_post | guide"
                    ),
                },
                "depth": {
                    "type": "STRING",
                    "description": "brief | standard (default) | deep — controls research depth",
                },
                "open_file": {
                    "type": "BOOLEAN",
                    "description": "Open the generated file after saving (default: true)",
                },
                "language": {
                    "type": "STRING",
                    "description": "Language for the document content: en (default) | es | fr | de | pt",
                },
            },
            "required": ["topic"],
        },
    }

    # ── Research pipeline ──────────────────────────────────────────────────────

    def _search_web(self, query: str) -> str:
        """Search with Gemini grounding; fall back to DuckDuckGo."""
        from google import genai
        try:
            client   = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=query,
                config={"tools": [{"google_search": {}}]},
            )
            text = "".join(
                part.text for part in response.candidates[0].content.parts
                if hasattr(part, "text") and part.text
            ).strip()
            if text:
                return text
        except Exception as e:
            self.log(f"Gemini search failed: {e}")

        # DuckDuckGo fallback
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=8):
                    results.append(f"- **{r.get('title','')}**: {r.get('body','')} ({r.get('href','')})")
            return "\n".join(results)
        except Exception as e:
            return f"Search failed: {e}"

    def _generate_document(
        self,
        topic: str,
        output_type: str,
        depth: str,
        language: str,
        web_context: str,
    ) -> str:
        """Use Gemini to synthesize research into structured Markdown."""
        from google import genai

        depth_instructions = {
            "brief":    "Create a concise overview (500-800 words).",
            "standard": "Create a comprehensive note (1000-1500 words).",
            "deep":     "Create an in-depth technical deep-dive (2000-3000 words).",
        }

        type_instructions = {
            "note": (
                "Generate a research note in Obsidian Markdown format with:\n"
                "1. YAML frontmatter (tags, date, aliases)\n"
                "2. # Introduction\n"
                "3. ## Key Concepts (with > [!note] callouts for important points)\n"
                "4. ## Deep Dive / How It Works\n"
                "5. ## Use Cases & Applications\n"
                "6. ## Pros & Cons (use a table)\n"
                "7. ## Current Trends (with > [!tip] for emerging developments)\n"
                "8. ## Related Topics (use [[WikiLink]] format for related concepts)\n"
                "9. ## References & Resources (with clickable links)\n"
            ),
            "youtube_script": (
                "Generate a YouTube video script with:\n"
                "1. YAML frontmatter\n"
                "2. ## Hook (first 30 seconds — attention-grabbing opener)\n"
                "3. ## Introduction (who this video is for)\n"
                "4. ## Main Content (broken into segments with timestamps)\n"
                "5. ## Key Takeaways\n"
                "6. ## Call to Action\n"
                "7. ## Resources Mentioned\n"
            ),
            "blog_post": (
                "Generate a blog post with:\n"
                "1. YAML frontmatter (title, description, tags, date)\n"
                "2. ## Introduction (hook + thesis)\n"
                "3. ## Background (context the reader needs)\n"
                "4. ## Main Sections (3-5 well-developed sections)\n"
                "5. ## Conclusion\n"
                "6. ## Further Reading\n"
            ),
            "guide": (
                "Generate a practical how-to guide with:\n"
                "1. YAML frontmatter\n"
                "2. ## Prerequisites\n"
                "3. ## Overview\n"
                "4. ## Step-by-Step Instructions (numbered, with code blocks where relevant)\n"
                "5. ## Common Issues & Solutions\n"
                "6. ## Next Steps\n"
            ),
        }

        lang_note = f"Write the entire document in {language} language." if language != "es" else ""

        prompt = f"""You are a technical writer and researcher AI assistant (JARVIS).
{depth_instructions.get(depth, depth_instructions['standard'])}
{type_instructions.get(output_type, type_instructions['note'])}
{lang_note}

Topic: {topic}

Web research context (use this as your primary source of facts):
{web_context[:6000]}

Requirements:
- Use proper Markdown formatting with headers, bold, italics, code blocks
- Include specific facts, numbers, and dates where available
- Use Obsidian callout syntax: > [!note], > [!tip], > [!important], > [!warning]
- Use [[WikiLink]] format for related topics in the "Related Topics" section
- Make the content educational, accurate, and engaging
- Include real URLs in the References section
"""
        client   = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()

    def _add_frontmatter_if_missing(self, content: str, topic: str, output_type: str) -> str:
        """Ensure the document has YAML frontmatter."""
        if content.startswith("---"):
            return content

        now  = datetime.now()
        tags = [
            output_type,
            "jarvis-research",
            "technology",
            topic.lower().replace(" ", "-")[:30],
        ]
        frontmatter = (
            f"---\n"
            f"title: {topic}\n"
            f"date: {now.strftime('%Y-%m-%d')}\n"
            f"tags: [{', '.join(tags)}]\n"
            f"created_by: Jarvis2\n"
            f"type: {output_type}\n"
            f"---\n\n"
        )
        return frontmatter + content

    # ── Main execute ───────────────────────────────────────────────────────────

    def execute(self, params: dict) -> str:
        topic       = params.get("topic", "").strip()
        output_type = params.get("output_type", "note").lower().strip()
        depth       = params.get("depth", "standard").lower().strip()
        should_open = params.get("open_file", True)
        language    = params.get("language", "en").lower().strip()

        if not topic:
            return "Please provide a topic to research."

        if output_type not in ("note", "youtube_script", "blog_post", "guide"):
            output_type = "note"
        if depth not in ("brief", "standard", "deep"):
            depth = "standard"

        self.log(f"Researching: '{topic}' | type={output_type} | depth={depth}")

        # Step 1: Gather web context
        self.log("Gathering web context...")
        search_queries = [
            topic,
            f"{topic} overview explained",
            f"{topic} use cases 2024 2025",
            f"{topic} pros cons comparison",
        ]
        web_context_parts = []
        for q in search_queries[:2 if depth == "brief" else 4]:
            result = self._search_web(q)
            if result:
                web_context_parts.append(f"Query: {q}\n{result}")
        web_context = "\n\n---\n\n".join(web_context_parts)

        # Step 2: Generate document
        self.log("Generating document with Gemini...")
        try:
            content = self._generate_document(
                topic, output_type, depth, language, web_context
            )
        except Exception as e:
            return f"Document generation failed: {e}"

        # Step 3: Add frontmatter if needed
        content = self._add_frontmatter_if_missing(content, topic, output_type)

        # Step 4: Save to Obsidian vault
        OBSIDIAN_VAULT.mkdir(parents=True, exist_ok=True)
        filename  = _sanitize_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filepath  = OBSIDIAN_VAULT / f"{filename}-{timestamp}.md"
        filepath.write_text(content, encoding="utf-8")

        self.log(f"Saved to: {filepath}")

        # Step 5: Open the file
        if should_open:
            _open_file(filepath)

        word_count = len(content.split())
        return (
            f"Research complete. Generated {output_type} on '{topic}' "
            f"({word_count} words, {depth} depth). "
            f"Saved to: {filepath}"
        )
