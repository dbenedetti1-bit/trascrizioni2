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
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Imposta GEMINI_API_KEY o GOOGLE_API_KEY (variabile d'ambiente o .env)"
        )
    genai.configure(api_key=api_key)

    system_prompt = _load_system_prompt()
    if not system_prompt:
        raise RuntimeError("Il file prompts.txt è vuoto o mancante. Inserisci il System Prompt.")

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_prompt,
    )

    user_message = (
        "Trasforma il seguente testo della lezione in una dispensa strutturata. "
        "Rispetta il system prompt: ignora ripetizioni e raggruppa logicamente gli argomenti "
        "trattati in momenti diversi della lezione. "
        "Rispondi SOLO con un oggetto JSON valido (nessun testo prima o dopo) con questa struttura:\n"
        '{"title": "Titolo dispensa", "introduction": "Testo introduzione", '
        '"toc": ["Capitolo 1", "Capitolo 2", ...], '
        '"chapters": [{"title": "Capitolo 1", "content": "Testo capitolo"}, ...]}\n\n'
        "Testo della lezione:\n\n"
        f"{raw_text}"
    )

    print("  Attendo risposta da Gemini (può richiedere alcuni secondi)...")
    response = model.generate_content(user_message)
    if not response or not response.text:
        return None

    # Parsing semplificato: si assume che Gemini restituisca JSON o un formato parsabile.
    # Per robustezza si può usare un prompt che chieda esplicitamente JSON.
    text = response.text.strip()
    return _parse_structured_response(text)


def _parse_structured_response(text: str) -> dict | None:
    """
    Estrae titolo, introduzione, indice e capitoli dalla risposta di Gemini.
    Supporta risposta in JSON o con marcatori (## Titolo, ## Introduzione, ## Capitolo 1, ecc.).
    """
    import json
    import re

    # Prova prima come JSON
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "title" in data and "chapters" in data:
                return data
        except json.JSONDecodeError:
            pass

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
    return result
