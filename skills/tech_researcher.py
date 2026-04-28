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

MEJORAS v2:
  - Guiones de YouTube SIEMPRE en español, sin excepciones
  - Hooks virales diseñados para máxima retención (primeros 3-5 segundos)
  - Cada sección incluye 📽️ NOTA DE PRODUCCIÓN con recomendaciones visuales/de edición

MEJORAS v3:
  - Usa gemini_api_skill (API key dedicada para skills, separada de la sesión Live)
  - execute() lanza la investigación en un daemon thread y retorna inmediatamente,
    de modo que Jarvis permanece completamente reactivo durante el proceso.
"""
from __future__ import annotations

import re
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path

from core.settings import OBSIDIAN_VAULT, get_skill_api_key
from skills.base import Skill

# ── Active background tasks tracker ───────────────────────────────────────────
# Maps task_id -> {"topic": str, "status": "running"|"done"|"error", "result": str}
_background_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


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
    """Research a technology topic and generate an Obsidian-compatible Markdown note.

    The heavy Gemini calls run in a background daemon thread so Jarvis stays
    fully responsive during the research process. Uses the dedicated
    ``gemini_api_skill`` API key from config/api_keys.json.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Always use the dedicated skill key; ignore the Live session key passed in
        super().__init__(api_key=get_skill_api_key())

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
            "brief":    "Crea una descripción concisa (500-800 palabras)." if output_type == "youtube_script" else "Create a concise overview (500-800 words).",
            "standard": "Crea un guión completo (1000-1500 palabras)." if output_type == "youtube_script" else "Create a comprehensive note (1000-1500 words).",
            "deep":     "Crea un guión extenso y detallado (2000-3000 palabras)." if output_type == "youtube_script" else "Create an in-depth technical deep-dive (2000-3000 words).",
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
                "Genera un guión de YouTube COMPLETAMENTE EN ESPAÑOL. NUNCA uses inglés en el contenido del guión.\n\n"
                "ESTILO OBLIGATORIO — sigue estas reglas sin excepción:\n"
                "- CERO emojis en todo el guión. Ni uno solo.\n"
                "- Ve directo al grano. Define el concepto en las primeras frases, sin construir suspenso largo.\n"
                "- Lenguaje conversacional y coloquial, como si hablaras con un amigo programador.\n"
                "- Frases cortas. Ritmo rápido. Sin relleno.\n"
                "- Usa analogías del mundo real, simples y cotidianas (transporte, comida, objetos físicos).\n"
                "- Nunca uses lenguaje de marketing ni frases vacías como 'increíble', 'revolucionario', 'no te lo puedes perder'.\n"
                "- El tono es directo, honesto y cercano. Como alguien que sabe mucho pero no alardea.\n\n"
                "Estructura OBLIGATORIA. Cada sección tiene su texto del guión seguido de una NOTA DE PRODUCCION:\n\n"
                "## HOOK\n"
                "Una o dos frases máximo. Sin saludo. Sin presentación. "
                "El hook debe ser directo: di de qué va el video y por qué importa en una sola idea. "
                "No construyas misterio, ve al punto pero hazlo con gancho. "
                "Ejemplos del estilo correcto: "
                "'Docker es la herramienta que resuelve el problema más molesto del desarrollo: que tu código funcione en todos lados igual.' "
                "'Git guarda el historial completo de tu código. Cada cambio, quién lo hizo y por qué. Eso es todo, pero cambia todo.'\n\n"
                "> NOTA DE PRODUCCION: [Solo recursos de pantalla: qué texto aparece en pantalla, "
                "animación, captura de terminal, o B-roll de código. Sin mencionar cara ni cámara. "
                "Sé específico y breve.]\n\n"
                "## DESARROLLO\n"
                "Explica el tema dividido en bloques naturales. Cada bloque sigue este patrón:\n"
                "1. Enuncia el concepto o problema en una frase corta.\n"
                "2. Explícalo con una analogía simple y concreta.\n"
                "3. Añade el detalle técnico real que lo hace útil o que la mayoría no sabe.\n\n"
                "Usa preguntas para avanzar entre bloques: 'De acuerdo, pero ¿qué diferencia hay con X?' "
                "Cada bloque va así:\n\n"
                "### [Título del bloque sin emoji]\n"
                "[Texto del guión]\n\n"
                "> NOTA DE PRODUCCION: [Solo recursos de pantalla para este bloque: "
                "terminal con comandos reales, animación de diagrama, texto superpuesto clave, "
                "captura de código, o pantalla de herramienta. Sin mencionar cara ni cámara.]\n\n"
                "## CONCLUSION\n"
                "Resume en 2-3 frases qué es el tema y por qué importa. "
                "Termina con una pregunta directa al espectador que invite al comentario.\n\n"
                "> NOTA DE PRODUCCION: [Solo recursos de pantalla: resumen de puntos en texto, "
                "animación de cierre, o captura final relevante. Sin mencionar cara ni cámara.]\n\n"
                "## CTA\n"
                "Llamada a la acción conversacional y específica. Menciona algo concreto del canal o del próximo video.\n\n"
                "> NOTA DE PRODUCCION: [Pantalla final con botón de suscripción animado, "
                "clip sugerido del próximo video, o pantalla de recursos. Sin mencionar cara ni cámara.]\n\n"
                "## RECURSOS\n"
                "Lista simple de links, herramientas o referencias del guión.\n\n"
                "> NOTA DE PRODUCCION: [Los recursos van en la descripción del video "
                "y como texto en pantalla al final.]\n\n"
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

        # Para guiones de YouTube: forzar español siempre, ignorar el parámetro `language`
        if output_type == "youtube_script":
            lang_note = (
                "IMPORTANTE: El guión COMPLETO debe estar escrito en ESPAÑOL. "
                "Esto incluye el hook, el desarrollo, la conclusión y la CTA. "
                "NO escribas ninguna parte en inglés. "
                "PROHIBIDO usar emojis en cualquier parte del guión, ni en los títulos de sección. "
                "El texto debe ser 100% español natural y coloquial, apropiado para YouTube en español. "
                "Habla como una persona real, no como un redactor de marketing."
            )
        else:
            lang_note = f"Write the entire document in {language} language." if language != "en" else ""

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
{"- EVERY section must end with a NOTA DE PRODUCCION block (plain text, no emoji) with specific production/editing recommendations" if output_type == "youtube_script" else ""}
{"- NO emojis anywhere in the script, not even in section titles or production notes" if output_type == "youtube_script" else ""}
{"- Use short sentences, fast rhythm, real-world analogies. Avoid marketing language." if output_type == "youtube_script" else ""}
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

        if output_type == "youtube_script":
            tags = [
                "youtube-script",
                "jarvis-research",
                "tecnologia",
                topic.lower().replace(" ", "-")[:30],
            ]
            frontmatter = (
                f"---\n"
                f"title: Guión YouTube — {topic}\n"
                f"date: {now.strftime('%Y-%m-%d')}\n"
                f"tags: [{', '.join(tags)}]\n"
                f"created_by: Jarvis2\n"
                f"type: youtube_script\n"
                f"idioma: español\n"
                f"---\n\n"
            )
        else:
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
        """Launch research in the background and return immediately.

        Jarvis stays fully responsive while the research compiles in a
        daemon thread using the dedicated ``gemini_api_skill`` key.
        """
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

        # Guiones de YouTube siempre en español, sin importar el parámetro `language`
        effective_language = "es" if output_type == "youtube_script" else language

        # Unique ID to track this background job
        task_id = str(uuid.uuid4())[:8]
        with _tasks_lock:
            _background_tasks[task_id] = {
                "topic":  topic,
                "status": "running",
                "result": "",
            }

        self.log(f"[{task_id}] Launching background research: '{topic}' | type={output_type} | depth={depth} | lang={effective_language}")

        # ── Background worker ──────────────────────────────────────────────────
        def _run() -> None:
            try:
                self.log(f"[{task_id}] Gathering web context...")
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

                self.log(f"[{task_id}] Generating document with Gemini...")
                content = self._generate_document(
                    topic, output_type, depth, effective_language, web_context
                )

                content = self._add_frontmatter_if_missing(content, topic, output_type)

                OBSIDIAN_VAULT.mkdir(parents=True, exist_ok=True)
                filename  = _sanitize_filename(topic)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filepath  = OBSIDIAN_VAULT / f"{filename}-{timestamp}.md"
                filepath.write_text(content, encoding="utf-8")

                self.log(f"[{task_id}] Saved to: {filepath}")

                if should_open:
                    _open_file(filepath)

                word_count = len(content.split())
                summary = (
                    f"Research complete. Generated {output_type} on '{topic}' "
                    f"({word_count} words, {depth} depth). "
                    f"Saved to: {filepath}"
                )
                with _tasks_lock:
                    _background_tasks[task_id]["status"] = "done"
                    _background_tasks[task_id]["result"] = summary
                self.log(f"[{task_id}] Done — {summary}")

            except Exception as e:  # noqa: BLE001
                error_msg = f"Research failed for '{topic}': {e}"
                with _tasks_lock:
                    _background_tasks[task_id]["status"] = "error"
                    _background_tasks[task_id]["result"] = error_msg
                self.log(f"[{task_id}] ERROR: {e}")

        thread = threading.Thread(target=_run, daemon=True, name=f"TechResearch-{task_id}")
        thread.start()

        # Return immediately — Jarvis stays responsive
        type_label = {
            "note":           "research note",
            "youtube_script": "YouTube script",
            "blog_post":      "blog post",
            "guide":          "guide",
        }.get(output_type, output_type)

        return (
            f"Got it! I've started researching '{topic}' in the background (task {task_id}). "
            f"I'll generate a {type_label} ({depth} depth) and save it to your Obsidian vault "
            f"when it's ready. You can keep talking to me in the meantime."
        )