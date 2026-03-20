"""
Motore Gemini: invia il testo alla API e restituisce contenuto strutturato
per la dispensa (titolo, introduzione, indice, capitoli). Usa il System Prompt da prompts.txt.
"""

import os
from pathlib import Path

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.txt"


def _load_system_prompt() -> str:
    """Carica il System Prompt dal file prompts.txt (ignora righe che iniziano con #)."""
    if not PROMPTS_PATH.exists():
        return ""
    lines = PROMPTS_PATH.read_text(encoding="utf-8").strip().splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#")).strip()


def process_with_gemini(raw_text: str) -> dict | None:
    """
    Invia il testo trascritto a Gemini e restituisce un dizionario con:
    - title: str
    - introduction: str
    - toc: list[str]  (titoli dei capitoli per l'indice)
    - chapters: list[dict]  con "title" e "content" per ogni capitolo
    """
    from google.genai import Client, types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Imposta GEMINI_API_KEY o GOOGLE_API_KEY (variabile d'ambiente o .env)"
        )

    system_prompt = _load_system_prompt()
    if not system_prompt:
        raise RuntimeError("Il file prompts.txt è vuoto o mancante. Inserisci il System Prompt.")

    client = Client(api_key=api_key)

    user_message = (
        "Trasforma il seguente testo della lezione in una dispensa strutturata. "
        "Rispetta il system prompt: ignora ripetizioni e raggruppa logicamente gli argomenti "
        "trattati in momenti diversi della lezione. "
        "Rispondi SOLO con un oggetto JSON valido (nessun testo prima o dopo) con questa struttura:\n"
        '{"title": "Titolo dispensa", "introduction": "Testo introduzione", '
        '"toc": ["Capitolo 1", "Capitolo 2", ...], '
        '"chapters": [{"title": "Capitolo 1", "content": "Testo capitolo"}, ...]}\n\n'
        "Vincoli di formato (obbligatori): "
        "- JSON compatto (nessuna intestazione testuale esterna). "
        "- introduction max 1500 caratteri. "
        "- toc max 7 elementi. "
        "- chapters max 7 elementi. "
        "- ogni chapters[i].content max 2600 caratteri. "
        "Testo della lezione:\n\n"
        f"{raw_text}"
    )

    # Usiamo un solo budget alto per evitare ripetizioni con JSON invalido.
    max_tokens_candidates = [16384]
    last_error: str | None = None
    for attempt_idx, max_output_tokens in enumerate(max_tokens_candidates, start=1):
        print(
            f"  Attendo risposta da Gemini (max_output_tokens={max_output_tokens})...",
            flush=True,
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # Forza un output JSON (evita prefissi tipo "Introduzione" + JSON).
                response_mime_type="application/json",
                # Mantieni l'output più deterministico e consistente.
                temperature=0.2,
                max_output_tokens=max_output_tokens,
            ),
        )
        if not response or not response.text:
            last_error = "Risposta Gemini vuota."
            continue

        text = response.text.strip()
        parsed = _parse_structured_response(text)
        if parsed:
            return parsed

        last_error = "Parsing Gemini non riuscito (risposta non strutturata/JSON invalido)."
        print(f"  Avviso: {last_error}", flush=True)

    return None


def _parse_structured_response(text: str) -> dict | None:
    """
    Estrae titolo, introduzione, indice e capitoli dalla risposta di Gemini.
    Supporta risposta in JSON o con marcatori (## Titolo, ## Introduzione, ## Capitolo 1, ecc.).
    """
    import json
    import re

    def _strip_code_fences(s: str) -> str:
        # Rimuove eventuali blocchi markdown tipo ```json ... ```
        s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```", "", s)
        return s.strip()

    def _coerce_structure(data: dict) -> dict:
        # Normalizza le chiavi più comuni
        if "chapters" not in data and "chapter" in data:
            data["chapters"] = data["chapter"]
        if "toc" not in data and "chapters" in data and isinstance(data["chapters"], list):
            data["toc"] = [
                (ch.get("title") or "").strip()
                for ch in data["chapters"]
                if isinstance(ch, dict) and (ch.get("title") or "").strip()
            ]
        data.setdefault("title", "Dispensa")
        data.setdefault("introduction", "")
        data.setdefault("toc", [])
        data.setdefault("chapters", [])
        return data

    def _try_parse_json_candidate(candidate: str) -> dict | None:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            return None
        decoder = json.JSONDecoder()
        try:
            obj, _end = decoder.raw_decode(candidate)
            if isinstance(obj, dict) and ("title" in obj) and ("chapters" in obj or "chapter" in obj):
                return _coerce_structure(obj)
        except json.JSONDecodeError:
            # Heuristica: rimuove trailing commas prima di } o ]
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                obj, _end = decoder.raw_decode(cleaned)
                if isinstance(obj, dict) and ("title" in obj) and ("chapters" in obj or "chapter" in obj):
                    return _coerce_structure(obj)
            except json.JSONDecodeError:
                return None
        return None

    # Prova prima come JSON: approccio robusto basato su decoder (gestisce brace annidate).
    s = _strip_code_fences(text)
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = s[first_brace : last_brace + 1].strip()
        data = _try_parse_json_candidate(json_candidate)
        if data:
            return data

    # Fallback: parsing con sezioni
    sections = re.split(r"\n\s*##\s+", text, flags=re.IGNORECASE)
    result = {
        "title": "Dispensa",
        "introduction": "",
        "toc": [],
        "chapters": [],
    }
    for i, block in enumerate(sections):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        first_line = lines[0].strip() if lines else ""
        rest = "\n".join(lines[1:]).strip()

        if i == 0 and not first_line.lower().startswith("capitolo"):
            # Se il modello mette "Introduzione" prima del JSON, proviamo a decodificarlo.
            if rest:
                candidate = rest.strip()
                candidate_looks_json = (
                    candidate.startswith("{")
                    or candidate.startswith('"title"')
                    or candidate.startswith('"chapters"')
                    or candidate.startswith('"toc"')
                    or candidate.startswith('"introduction"')
                )
                if candidate_looks_json and ('"chapters"' in candidate or '"toc"' in candidate or '"title"' in candidate):
                    data = _try_parse_json_candidate(candidate)
                    if data:
                        return data
                    # Se sembra JSON ma è non decodificabile (tronco), non usarlo
                    # come "introduzione" perché finirebbe nel Word come testo grezzo.
                    return None
            result["title"] = first_line or result["title"]
            if rest:
                result["introduction"] = rest
            continue
        if "introduzione" in first_line.lower():
            result["introduction"] = rest
            continue
        if "indice" in first_line.lower() or "contenuti" in first_line.lower():
            result["toc"] = [l.strip().lstrip("-* ") for l in rest.split("\n") if l.strip()]
            continue
        result["chapters"].append({"title": first_line, "content": rest})
        result["toc"].append(first_line)

    if not result["chapters"] and result["introduction"]:
        result["chapters"].append({"title": "Contenuto", "content": result["introduction"]})
        result["toc"] = ["Contenuto"]

    # Guard-rail finale: se "introduction" sembra JSON, prova a decodificarlo,
    # ma NON fallire la parse globale in caso di JSON non valido/troncato.
    # Altrimenti `main.py` usa un fallback e finisci per inserire la
    # trascrizione grezza nel Word.
    try_intro = (result.get("introduction") or "").lstrip()
    looks_json = (
        try_intro.startswith("{")
        or try_intro.startswith('"title"')
        or try_intro.startswith('"chapters"')
        or try_intro.startswith('"toc"')
        or try_intro.startswith('"introduction"')
    )
    if looks_json and ('"chapters"' in try_intro or '"toc"' in try_intro or '"title"' in try_intro):
        data = _try_parse_json_candidate(try_intro)
        if data:
            return data
        # Se sembra JSON ma non è decodificabile, segnaliamo errore così
        # `main.py` usa fallback (evita JSON "grezzo" nel Word).
        return None
    return result
