"""
skills/news_journalist.py — Noticias con análisis periodístico, sin cambiar la voz de Jarvis.

Reutiliza WebSearchSkill para la búsqueda (Gemini grounding + fallback DDG).
Un segundo call a gemini-flash-lite enriquece cada noticia con contexto, análisis
e implicaciones. Corre en hilo daemon para no bloquear a Jarvis.

El usuario puede decir "detente" o "para" en cualquier momento para cancelar
la búsqueda en curso antes de que lleguen los resultados.
"""
from __future__ import annotations

import threading
import uuid
from typing import Callable

from google import genai

from core.settings import LITE_MODEL
from skills.base import Skill, _get_ddgs
from skills.web_search import WebSearchSkill


_CATEGORY_QUERIES = {
    "tecnologia":      "últimas noticias tecnología hoy",
    "mundo":           "noticias internacionales más importantes hoy",
    "deportes":        "noticias deportivas más importantes hoy fútbol",
    "entretenimiento": "noticias entretenimiento cine música hoy",
    "politica":        "noticias políticas más importantes de colombia hoy",
    "general":         "noticias más importantes del día hoy",
}


class NewsJournalistSkill(Skill):
    """Busca noticias actuales y las analiza con criterio periodístico."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key)
        self.notify_fn: Callable[[str], None] | None = None
        self._web = WebSearchSkill(api_key=self.api_key)
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None
        self._cancel_event: threading.Event | None = None
        self._active_task_id: str | None = None
        self._lock = threading.Lock()

    TOOL_DECLARATION = {
        "name": "news_journalist",
        "description": (
            "Busca las últimas noticias y las explica con contexto, análisis e implicaciones. "
            "Úsala cuando el usuario pida noticias, titulares, qué pasó hoy, "
            "novedades de tecnología, deportes, política, entretenimiento o mundo, "
            "o cuando pida noticias sobre un tema específico. "
            "Usa action='stop' cuando el usuario diga 'detente', 'para', 'cancela' o 'olvídalo'."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "get_news (default) — busca y analiza noticias. "
                        "stop — cancela la búsqueda en curso."
                    ),
                },
                "topic": {
                    "type": "STRING",
                    "description": (
                        "Tema específico sobre el que buscar noticias. "
                        "Si no se especifica, se usan las noticias más importantes del día."
                    ),
                },
                "category": {
                    "type": "STRING",
                    "description": (
                        "Categoría: general (default) | tecnologia | mundo | "
                        "deportes | entretenimiento | politica"
                    ),
                },
                "num_news": {
                    "type": "INTEGER",
                    "description": "Número de noticias: 5 (default) | 3 | 7",
                },
                "detail_topic": {
                    "type": "STRING",
                    "description": (
                        "Cuando el usuario pide profundizar en una noticia específica, "
                        "pasa aquí el tema o titular para buscar más información."
                    ),
                },
            },
            "required": [],
        },
    }

    def _stop_active(self) -> str:
        with self._lock:
            if self._cancel_event is None or self._cancel_event.is_set():
                return "No hay ninguna búsqueda de noticias en curso."
            self._cancel_event.set()
            tid = self._active_task_id or "?"
        self.log(f"[{tid}] Cancelado por el usuario.")
        return "Búsqueda de noticias cancelada."

    def _analyze(self, raw: str, num_news: int, topic: str) -> str:
        """Produce un boletín de noticias: titular + una frase por noticia.

        Si Gemini falla, genera un boletín mínimo limpio desde el texto crudo.
        """
        topic_hint = f" sobre '{topic}'" if topic else ""
        try:
            prompt = (
                f"Responde ÚNICAMENTE en español. Ni una sola palabra en inglés.\n\n"
                f"Eres un locutor de radio que lee el boletín de noticias{topic_hint}. "
                f"Aquí hay texto crudo de fuentes de noticias:\n\n{raw[:5000]}\n\n"
                f"Selecciona las {num_news} noticias más relevantes y reales "
                f"(descarta publicidad, listas de 'lo más visto', y contenido sin impacto). "
                f"Para cada noticia escribe exactamente esto:\n\n"
                f"Un titular directo de máximo 10 palabras que diga el hecho central. "
                f"Seguido de un punto. Luego una sola frase que explique por qué importa o qué pasó. "
                f"Nada más.\n\n"
                f"FORMATO OBLIGATORIO — cada noticia es DOS frases:\n"
                f"Primera: el titular del hecho, máximo 12 palabras.\n"
                f"Segunda: por qué importa o qué ocurrió exactamente, en una sola frase.\n"
                f"Separa cada noticia con una línea en blanco. Sin numeración, sin etiquetas.\n\n"
                f"Ejemplo:\n"
                f"Apple lanza el chip M4 Ultra para servidores de inteligencia artificial.\n"
                f"Promete reducir el costo de inferencia a la mitad frente a las GPU actuales.\n\n"
                f"Reglas:\n"
                f"- Texto plano. Sin markdown, asteriscos, guiones ni símbolos.\n"
                f"- Sin introducción ni cierre. Empieza directo con el primer titular.\n"
                f"- Sin adjetivos sensacionalistas. Sin clickbait.\n"
                f"- Si no hay {num_news} noticias reales, incluye solo las verificables.\n"
            )

            response = self._client.models.generate_content(
                model=LITE_MODEL,
                contents=prompt,
            )
            result = "".join(
                part.text for part in response.candidates[0].content.parts
                if hasattr(part, "text") and part.text
            ).strip()
            if result:
                return result
        except Exception as e:
            self.log(f"Análisis falló ({e}), construyendo boletín desde texto crudo.")

        return self._clean_raw_fallback(raw, num_news)

    def _clean_raw_fallback(self, raw: str, num_news: int) -> str:
        """Extrae titulares limpios del texto crudo cuando Gemini no está disponible.

        Los chunks de DDG son líneas largas del tipo:
          "Título completo. Texto del snippet largo... (Fuente on MSN)"
        Extraemos solo la primera frase de cada chunk como titular.
        """
        import re

        _SKIP = ("http://", "https://", "www.", "©", "Results for:",
                 "| Noticias", "Univision", "Lo que debes saber")

        headlines: list[str] = []

        # Split by double newline (items separados en el raw de DDG)
        chunks = [c.strip() for c in re.split(r'\n{2,}', raw) if c.strip()]

        for chunk in chunks:
            if len(headlines) >= num_news:
                break

            # Descartar bloques de navegación o con URLs
            if any(bad in chunk for bad in _SKIP):
                continue

            # Tomar solo la primera oración del chunk
            first = re.split(r'\.\s+(?=[A-ZÁÉÍÓÚÜÑ"«])', chunk)[0].strip()

            # Quitar atribución de fuente al final: "(Xataka on MSN)"
            first = re.sub(r'\s*\([^)]{2,60}\)\s*$', '', first).strip()

            # Quitar pipes de navegación: "Título | Sección | Sitio"
            if '|' in first:
                first = first[:first.index('|')].strip()

            # Descartar si quedó demasiado corto o es solo ruido
            if len(first) < 25:
                continue

            # Normalizar punto final
            if first[-1] not in '.!?':
                first += '.'

            if first not in headlines:
                headlines.append(first)

        if headlines:
            return "\n\n".join(headlines)

        # Último recurso: primera línea no vacía del raw
        for line in raw.splitlines():
            line = line.strip()
            if len(line) > 30 and not any(b in line for b in _SKIP):
                return line
        return "No se pudieron obtener noticias en este momento."

    def _ddg_news_fallback(self, query: str, num_news: int) -> str:
        """Usa DDGS().news() con reintentos; si falla, cae a _ddg_search de WebSearchSkill."""
        import time
        DDGS = _get_ddgs()
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.news(query, max_results=num_news * 2))
                if results:
                    lines = []
                    for r in results[:num_news * 2]:
                        title  = r.get("title", "")
                        body   = r.get("body", "") or r.get("excerpt", "")
                        source = r.get("source", "") or r.get("url", "")
                        if title:
                            lines.append(f"{title}. {body} ({source})")
                    if lines:
                        return "\n\n".join(lines)
            except Exception as e:
                last_exc = e
                self.log(f"DDG news intento {attempt+1} falló: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

        # Último recurso: búsqueda de texto con retry (desde WebSearchSkill)
        self.log(f"DDG news agotado ({last_exc}), usando _ddg_search de texto...")
        text_results = self._web._ddg_search(query, max_results=num_news * 2)
        if text_results:
            return "\n\n".join(
                f"{r['title']}. {r['snippet']} ({r['url']})"
                for r in text_results if r.get("title")
            )

        raise RuntimeError(f"Todas las fuentes fallaron para: {query} — {last_exc}")

    def execute(self, params: dict) -> str:
        action = params.get("action", "get_news").lower().strip()

        if action == "stop":
            return self._stop_active()

        detail_topic = params.get("detail_topic", "").strip()
        if detail_topic:
            # Delegate deep-dive to web_search directly
            return self._web.execute({"query": detail_topic, "mode": "search"})

        topic    = params.get("topic", "").strip()
        category = params.get("category", "general").lower().strip()
        num_news = int(params.get("num_news", 5))

        if category not in _CATEGORY_QUERIES:
            category = "general"
        num_news = max(1, min(num_news, 7))

        query = (
            f"últimas noticias {topic}"
            if topic
            else _CATEGORY_QUERIES[category]
        )

        task_id = str(uuid.uuid4())[:8]
        cancel  = threading.Event()

        with self._lock:
            if self._cancel_event and not self._cancel_event.is_set():
                self._cancel_event.set()
            self._cancel_event   = cancel
            self._active_task_id = task_id

        self.log(f"[{task_id}] Buscando noticias: '{query}'")

        _NO_RESULT_PREFIXES = ("No results found", "Search failed", "Results for:")

        def _run() -> None:
            import time
            try:
                # 1. Búsqueda vía Gemini grounding / DDG (con reintento)
                if cancel.is_set():
                    return

                raw = ""
                for attempt in range(2):
                    try:
                        raw = self._web.execute({"query": query, "mode": "news"})
                        if raw and not any(raw.startswith(p) for p in _NO_RESULT_PREFIXES):
                            break
                    except Exception as e:
                        self.log(f"[{task_id}] web_search intento {attempt+1} falló: {e}")
                    if attempt == 0:
                        time.sleep(2)

                if cancel.is_set():
                    self.log(f"[{task_id}] Cancelado tras búsqueda.")
                    return

                if not raw or any(raw.startswith(p) for p in _NO_RESULT_PREFIXES):
                    self.log(f"[{task_id}] web_search sin resultado, probando DDGS.news()...")
                    raw = self._ddg_news_fallback(query, num_news)

                if cancel.is_set():
                    self.log(f"[{task_id}] Cancelado antes del análisis.")
                    return

                # 2. Análisis periodístico
                analyzed = self._analyze(raw, num_news, topic)

                if cancel.is_set():
                    self.log(f"[{task_id}] Cancelado tras análisis.")
                    return

                self.log(f"[{task_id}] Listo — {len(analyzed.split())} palabras.")

                if self.notify_fn:
                    self.notify_fn(
                        f"[BACKGROUND TASK COMPLETE] Lee estas noticias palabra por palabra, "
                        f"sin omitir ninguna, sin resumir. NO añadas introducción. "
                        f"Al terminar la última, di solo: "
                        f"'Eso es todo, señor. ¿Desea que profundice en alguna?'\n\n{analyzed}"
                    )

            except Exception as e:
                self.log(f"[{task_id}] ERROR: {e}")
                if not cancel.is_set() and self.notify_fn:
                    self.notify_fn(
                        f"[BACKGROUND TASK FAILED] No pude obtener las noticias: {e}. "
                        f"Informa al usuario brevemente."
                    )

        threading.Thread(target=_run, daemon=True, name=f"News-{task_id}").start()

        subject = f"sobre '{topic}'" if topic else f"de {category}"
        return (
            f"[BUSCANDO EN SEGUNDO PLANO] Buscando noticias {subject}. "
            f"Los resultados llegarán en unos segundos. "
            f"NO generes ni resumas noticias desde tu memoria — espera los resultados reales."
        )
