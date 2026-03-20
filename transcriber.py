"""
Trascrittore: estrae audio da video (se necessario) e trascrive con faster-whisper.
Supporta .mp4, .mkv (video) e .mp3, .wav (audio). Configurazione GPU.
"""

import tempfile
from pathlib import Path
# Questo aiuta Python a trovare le DLL appena installate nel venv
import os
import sys
import time
os.environ["PATH"] += os.pathsep + os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin')
os.environ["PATH"] += os.pathsep + os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')

# Estensioni considerate video vs audio
VIDEO_EXTENSIONS = {".mp4", ".mkv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}

# Incremento per il progresso a schermo (secondi)
PROGRESS_INTERVAL_SEC = 30


def _format_duration(seconds: float) -> str:
    """Formatta secondi in MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def get_audio_duration_seconds(audio_path: Path) -> float:
    """Restituisce la durata in secondi del file audio (per progress e stima rimanenti)."""
    path = Path(audio_path)
    suffix = path.suffix.lower()

    duration: float = 0.0
    if suffix == ".wav":
        import wave
        with wave.open(str(path), "rb") as w:
            duration = w.getnframes() / float(w.getframerate())
            return duration
    if suffix == ".mp3":
        try:
            from mutagen.mp3 import MP3
            duration = MP3(str(path)).info.length
            if duration and duration > 0:
                return float(duration)
        except Exception:
            pass
    if suffix == ".m4a":
        # Durata non essenziale: se non disponibile, il progresso va su "?".
        # Mutagen a volte supporta m4a/mp4, ma può fallire in base al codec.
        try:
            from mutagen import File as MutagenFile
            f = MutagenFile(str(path))
            if f and getattr(f, "info", None) and getattr(f.info, "length", None):
                duration = float(f.info.length)
                if duration and duration > 0:
                    return float(duration)
        except Exception:
            pass

    # Fallback: moviepy (usa ffmpeg internamente quando necessario).
    # È più affidabile di mutagen su alcuni codec/container e nel tuo ambiente
    # evita di dipendere dalla presenza di `ffprobe` in PATH.
    try:
        from moviepy.editor import AudioFileClip

        clip = AudioFileClip(str(path))
        try:
            duration = float(getattr(clip, "duration", 0.0) or 0.0)
            if duration and duration > 0:
                return duration
        finally:
            try:
                clip.close()
            except Exception:
                pass
    except Exception:
        pass

    return 0.0


def extract_audio_from_video(video_path: Path) -> Path:
    """Estrae l'audio da un file video usando moviepy. Restituisce il path del file audio temporaneo."""
    from moviepy.editor import VideoFileClip

    print("  Estrazione audio dal video in corso...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(tmp_path), verbose=False, logger=None)
        clip.close()
        print("  Audio estratto.")
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def transcribe_media(media_path: Path, transcript_save_path: Path | None = None) -> str:
    """
    Trascrive un file audio o video con faster-whisper.
    - Se è video: estrae prima l'audio con moviepy.
    - Usa device="cuda" e compute_type="float16" quando disponibile.
    - Se transcript_save_path è fornito, salva il testo su file prima di restituire (così non si perde in caso di errore dopo).
    """
    media_path = Path(media_path).resolve()
    suffix = media_path.suffix.lower()

    if suffix not in VIDEO_EXTENSIONS and suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Formato non supportato per trascrizione: {suffix}")

    audio_path = media_path
    temp_audio_path = None

    if suffix in VIDEO_EXTENSIONS:
        audio_path = extract_audio_from_video(media_path)
        temp_audio_path = audio_path

    try:
        from faster_whisper import WhisperModel

        duration_sec = get_audio_duration_seconds(audio_path)
        total_str = _format_duration(duration_sec) if duration_sec > 0 else "?"
        print(f"  Durata audio: {total_str}. Avvio trascrizione (aggiornamento ogni {PROGRESS_INTERVAL_SEC} sec)...")

        # Se la trascrizione risulta tronca (es. dopo lunghi silenzi),
        # rilanciamo automaticamente senza VAD per recuperare il contenuto.
        vad_filter_primary = os.environ.get("WHISPER_VAD_FILTER", "true").strip().lower()
        vad_filter_primary = vad_filter_primary in {"1", "true", "yes", "y", "on"}
        vad_filter_retry = False

        def run_transcription(model, vad_filter: bool) -> tuple[str, float]:
            segments, _ = model.transcribe(
                str(audio_path),
                language="it",
                vad_filter=vad_filter,
            )
            last_milestone = 0
            parts = []
            start_ts = time.monotonic()
            last_segment_end = 0.0
            for s in segments:
                if s.text:
                    parts.append(s.text)
                try:
                    last_segment_end = max(last_segment_end, float(s.end or 0.0))
                except Exception:
                    pass
                # Progresso:
                # - se sappiamo la durata audio: usiamo il timestamp dei segmenti (`s.end`)
                # - se non sappiamo la durata: usiamo il tempo reale trascorso
                if duration_sec > 0:
                    while s.end >= last_milestone + PROGRESS_INTERVAL_SEC:
                        last_milestone += PROGRESS_INTERVAL_SEC
                        remaining = max(0, duration_sec - last_milestone)
                        print(
                            f"  Trascrizione: {_format_duration(last_milestone)} / {total_str} — rimanenti ~{_format_duration(remaining)}",
                            flush=True,
                        )
                else:
                    while (time.monotonic() - start_ts) >= last_milestone + PROGRESS_INTERVAL_SEC:
                        last_milestone += PROGRESS_INTERVAL_SEC
                        print(
                            f"  Trascrizione: {_format_duration(last_milestone)} trascorsi / ? — avvio...",
                            flush=True,
                        )
            return " ".join(parts).strip(), last_segment_end

        # Prova prima con GPU; se fallisce (es. cublas64_12.dll mancante) usa CPU
        try:
            model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float16",
                download_root=None,
            )
            text, last_segment_end = run_transcription(model, vad_filter_primary)
            if (
                vad_filter_primary
                and duration_sec > 0
                and last_segment_end > 0
                and last_segment_end < duration_sec * 0.85
            ):
                print(
                    "  Avviso: possibile trascrizione tronca (fine stimata prima della durata attesa). Riprovo senza VAD..."
                )
                text, last_segment_end = run_transcription(model, vad_filter_retry)
        except (RuntimeError, Exception) as e:
            if "cuda" in str(e).lower() or "cublas" in str(e).lower() or "dll" in str(e).lower():
                print("  GPU non disponibile, uso trascrizione in CPU (può essere più lenta).")
            model = WhisperModel("base", device="cpu", compute_type="int8")
            text, last_segment_end = run_transcription(model, vad_filter_primary)
            if (
                vad_filter_primary
                and duration_sec > 0
                and last_segment_end > 0
                and last_segment_end < duration_sec * 0.85
            ):
                print(
                    "  Avviso: possibile trascrizione tronca (fine stimata prima della durata attesa). Riprovo senza VAD..."
                )
                text, last_segment_end = run_transcription(model, vad_filter_retry)
        if duration_sec > 0:
            print(f"  Trascrizione completata: {total_str} di audio.", flush=True)
        # Salva subito la trascrizione su file (prima del return) così non si perde mai
        if transcript_save_path and text:
            try:
                transcript_save_path.write_text(text, encoding="utf-8")
                print(f"  Trascrizione salvata: {transcript_save_path}", flush=True)
            except Exception as e:
                print(f"  Attenzione: impossibile salvare trascrizione su file: {e}", flush=True)
        return text
    finally:
        # Non far mai propagare errori dalla pulizia: altrimenti si perde il valore di return
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink(missing_ok=True)
            except Exception:
                pass
