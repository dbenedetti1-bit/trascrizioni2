"""
Motore Gemini: invia il testo alla API e restituisce contenuto strutturato
per la dispensa (titolo, introduzione, indice, capitoli). Usa il System Prompt da prompts.txt.
"""

import os
from pathlib import Path

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.txt"
_DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug-316c26.log"


def _ndjson_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    run_id: str = "pre-fix",
) -> None:
    """Scrive una riga NDJSON nel file di log debug (no contenuti testuali sensibili)."""
    import json as _json
    import time as _time

    payload = {
        "sessionId": "316c26",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(_time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


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
        # #region debug_log_response_shape
        _ndjson_debug_log(
            hypothesis_id="H1",
            location="process_with_gemini:response_text",
            message="shape flags",
            data={
                "text_len": len(text),
                "starts_with_brace": text.lstrip().startswith("{"),
                "has_title_key": '"title"' in text,
                "has_introduction_key": '"introduction"' in text,
                "has_toc_key": '"toc"' in text,
                "has_chapters_key": '"chapters"' in text,
            },
        )
        # #endregion
        parsed = _parse_structured_response(text)
        if parsed:
            # #region debug_log_parsed_result
            _ndjson_debug_log(
                hypothesis_id="H3",
                location="process_with_gemini:parsed_ok",
                message="parsed dict summary",
                data={
                    "title_len": len(str(parsed.get("title", "") or "")),
                    "intro_len": len(str(parsed.get("introduction", "") or "")),
                    "intro_starts_with_brace": str(parsed.get("introduction", "") or "").lstrip().startswith("{"),
                    "intro_contains_title_key": '"title"' in str(parsed.get("introduction", "") or ""),
                    "toc_is_list": isinstance(parsed.get("toc"), list),
                    "chapters_is_list": isinstance(parsed.get("chapters"), list),
                    "chapters_count": len(parsed.get("chapters") or []),
                },
            )
            # #endregion
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

    # #region debug_log_parser_start
    _ndjson_debug_log(
        hypothesis_id="H2",
        location="_parse_structured_response:start",
        message="parser start",
        data={
            "text_len": len(text),
            "starts_with_brace": text.lstrip().startswith("{"),
            "has_title_key": '"title"' in text,
            "has_introduction_key": '"introduction"' in text,
            "has_toc_key": '"toc"' in text,
            "has_chapters_key": '"chapters"' in text,
        },
    )
    # #endregion

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
        # #region debug_log_json_candidate_attempt
        _ndjson_debug_log(
            hypothesis_id="H1",
            location="_parse_structured_response:json_candidate_extract",
            message="json candidate extraction",
            data={
                "first_brace_idx": first_brace,
                "last_brace_idx": last_brace,
                "candidate_len": len(json_candidate),
            },
        )
        # #endregion
        data = _try_parse_json_candidate(json_candidate)
        if data:
            # #region debug_log_json_candidate_success
            _ndjson_debug_log(
                hypothesis_id="H1",
                location="_parse_structured_response:json_candidate_success",
                message="json decoded",
                data={
                    "intro_len": len(str(data.get("introduction", "") or "")),
                    "intro_contains_title_key": '"title"' in str(data.get("introduction", "") or ""),
                    "chapters_is_list": isinstance(data.get("chapters", None), list),
                },
            )
            # #endregion
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
                # #region debug_log_intro_assigned_sections_block
                _ndjson_debug_log(
                    hypothesis_id="H2",
                    location="_parse_structured_response:intro_from_sections_i0",
                    message="introduction assigned from sections",
                    data={
                        "intro_len": len(result.get("introduction", "") or ""),
                        "intro_starts_with_brace": (result.get("introduction", "") or "").lstrip().startswith("{"),
                        "intro_contains_title_key": '"title"' in (result.get("introduction", "") or ""),
                        "intro_contains_chapters_key": '"chapters"' in (result.get("introduction", "") or ""),
                        "intro_contains_toc_key": '"toc"' in (result.get("introduction", "") or ""),
                    },
                )
                # #endregion
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
        # #region debug_log_intro_json_not_decodable
        _ndjson_debug_log(
            hypothesis_id="H2",
            location="_parse_structured_response:final_guard_return_none",
            message="intro looks-json but not decodable",
            data={
                "intro_len": len(try_intro),
                "intro_starts_with_brace": try_intro.startswith("{"),
                "intro_has_title_key": '"title"' in try_intro,
                "intro_has_chapters_key": '"chapters"' in try_intro,
                "intro_has_toc_key": '"toc"' in try_intro,
            },
        )
        # #endregion
        return None
    return result
