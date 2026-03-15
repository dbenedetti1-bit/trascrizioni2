"""
Trascrittore: estrae audio da video (se necessario) e trascrive con faster-whisper.
Supporta .mp4, .mkv (video) e .mp3, .wav (audio). Configurazione GPU.
"""

import tempfile
from pathlib import Path
# Questo aiuta Python a trovare le DLL appena installate nel venv
import os
import sys
os.environ["PATH"] += os.pathsep + os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin')
os.environ["PATH"] += os.pathsep + os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')

# Estensioni considerate video vs audio
VIDEO_EXTENSIONS = {".mp4", ".mkv"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}

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
    if suffix == ".wav":
        import wave
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    if suffix == ".mp3":
        try:
            from mutagen.mp3 import MP3
            return MP3(str(path)).info.length
        except Exception:
            return 0.0
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


def transcribe_media(media_path: Path) -> str:
    """
    Trascrive un file audio o video con faster-whisper.
    - Se è video: estrae prima l'audio con moviepy.
    - Usa device="cuda" e compute_type="float16" quando disponibile.
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

        def run_transcription(model):
            segments, _ = model.transcribe(str(audio_path), language="it", vad_filter=True)
            last_milestone = 0
            parts = []
            for s in segments:
                if s.text:
                    parts.append(s.text)
                # Progress ogni 15 sec
                while duration_sec > 0 and s.end >= last_milestone + PROGRESS_INTERVAL_SEC:
                    last_milestone += PROGRESS_INTERVAL_SEC
                    remaining = max(0, duration_sec - last_milestone)
                    print(f"  Trascrizione: {_format_duration(last_milestone)} / {total_str} — rimanenti ~{_format_duration(remaining)}")
            return " ".join(parts).strip()

        # Prova prima con GPU; se fallisce (es. cublas64_12.dll mancante) usa CPU
        try:
            model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float16",
                download_root=None,
            )
            text = run_transcription(model)
        except (RuntimeError, Exception) as e:
            if "cuda" in str(e).lower() or "cublas" in str(e).lower() or "dll" in str(e).lower():
                print("  GPU non disponibile, uso trascrizione in CPU (può essere più lenta).")
            model = WhisperModel("base", device="cpu", compute_type="int8")
            text = run_transcription(model)
        if duration_sec > 0:
            print(f"  Trascrizione completata: {total_str} di audio.")
        return text
    finally:
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
