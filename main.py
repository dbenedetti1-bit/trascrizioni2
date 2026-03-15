"""
Orchestratore principale: lezioni universitarie → dispense Word.
Chiede il file di partenza e coordina trascrizione, elaborazione LLM e generazione Word.
"""

import os
import sys
from pathlib import Path

# Carica variabili da .env (GEMINI_API_KEY) prima di importare gli altri moduli
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from transcriber import transcribe_media
from gemini_engine import process_with_gemini
from doc_builder import build_document


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mp3", ".wav", ".txt", ".pdf"}


def get_input_file() -> Path:
    """Chiede all'utente il percorso del file di partenza (video, audio, testo o PDF)."""
    print("=" * 60)
    print("  Lezioni → Dispense Word")
    print("=" * 60)
    print("\nFormati supportati: video (.mp4, .mkv), audio (.mp3, .wav), testo (.txt), PDF (.pdf)\n")

    while True:
        path_str = input("Inserisci il percorso del file di partenza: ").strip().strip('"')
        if not path_str:
            print("Percorso vuoto. Riprova.")
            continue

        path = Path(path_str)
        if not path.exists():
            print(f"File non trovato: {path}")
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Estensione non supportata: {path.suffix}. Usa: {', '.join(SUPPORTED_EXTENSIONS)}")
            continue

        return path.resolve()


def get_text_from_file(path: Path) -> str:
    """Legge il testo da file .txt o .pdf (senza trascrizione)."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print("Per i PDF installa: pip install pypdf")
            sys.exit(1)
    return ""


def run_workflow(input_path: Path) -> None:
    """Esegue il flusso: trascrizione (se media) → Gemini → Word."""
    transcript_path = input_path.with_name(f"{input_path.stem}_trascrizione.txt")
    output_path = input_path.with_name(f"{input_path.stem}_dispensa.docx")

    print("\n" + "=" * 50)
    print("  Step 1/4: Analisi file di partenza")
    print("=" * 50)
    print(f"  File: {input_path.name}")

    # Testo: da file; Media: da trascrizione
    if input_path.suffix.lower() in {".txt", ".pdf"}:
        print("  Lettura testo da file...")
        raw_text = get_text_from_file(input_path)
        if not raw_text.strip():
            print("  Errore: file vuoto o testo non estraibile.")
            return
        print(f"  Testo estratto ({len(raw_text)} caratteri).")
    else:
        print("  Trascrizione audio/video in corso...")
        raw_text = transcribe_media(input_path)
        if not raw_text or not raw_text.strip():
            print("  Errore: trascrizione vuota. Verifica il file.")
            return
        print(f"  Trascrizione completata ({len(raw_text)} caratteri).")
        print("  Creazione file trascrizione...")
        transcript_path.write_text(raw_text, encoding="utf-8")
        print(f"  Trascrizione salvata: {transcript_path}")

    print("\n" + "=" * 50)
    print("  Step 2/4: Elaborazione con Gemini")
    print("=" * 50)
    print("  Invio testo a Gemini per strutturazione e riordino...")
    structured_content = process_with_gemini(raw_text)
    if not structured_content:
        print("  Errore: nessun contenuto restituito da Gemini.")
        return
    print("  Risposta ricevuta. Struttura pronta per il documento.")

    print("\n" + "=" * 50)
    print("  Step 3/4: Generazione documento Word")
    print("=" * 50)
    print("  Creazione dispensa (titolo, introduzione, indice, capitoli)...")
    build_document(structured_content, output_path)
    print(f"  Dispensa salvata: {output_path}")

    print("\n" + "=" * 50)
    print("  Step 4/4: Completato")
    print("=" * 50)
    if input_path.suffix.lower() not in {".txt", ".pdf"}:
        print(f"  File creati:")
        print(f"    - Trascrizione: {transcript_path}")
    print(f"    - Dispensa Word: {output_path}")
    print()


def main() -> None:
    input_path = get_input_file()
    run_workflow(input_path)


if __name__ == "__main__":
    main()
