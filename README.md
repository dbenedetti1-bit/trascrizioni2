# Lezioni → Dispense Word

Tool modulare in Python per trasformare lezioni universitarie (video, audio o testo) in dispense Word con struttura accademica: titolo, introduzione, indice e capitoli riordinati logicamente.

## Requisiti

- **Python 3.10+**
- **CUDA** (opzionale): per usare la GPU con faster-whisper (trascrizione più veloce)
- **Chiave API Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) (o Google Cloud)

## Installazione

1. Clona il repository:
   ```bash
   git clone https://github.com/TUO_USERNAME/Trascrizioni2.git
   cd Trascrizioni2
   ```

2. Crea un ambiente virtuale (consigliato):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

4. Configura la chiave API Gemini :
   - **Variabile d’ambiente** (Windows PowerShell):
     ```powershell
     $env:GEMINI_API_KEY = "la-tua-chiave-api"
     ```
   - **File `.env`** nella cartella del progetto: `copy .env.example .env` poi modifica `.env` (non committare `.env`):
     ```
     GEMINI_API_KEY=la-tua-chiave-api
     ```
     Poi apri `.env` e inserisci la chiave presa da [Google AI Studio](https://aistudio.google.com/apikey). Il file `.env` non viene caricato su Git.

## Utilizzo

1. Avvia il programma:
   ```bash
   python main.py
   ```

2. Quando richiesto, inserisci il **percorso completo** del file di partenza:
   - **Video**: `.mp4`, `.mkv` → viene estratto l’audio e poi trascritto
   - **Audio**: `.mp3`, `.wav` → trascrizione diretta
   - **Testo/PDF/Word**: `.txt`, `.pdf`, `.docx` → nessuna trascrizione, il testo viene inviato a Gemini

3. Il flusso esegue in automatico:
   - Trascrizione (se media) con faster-whisper
   - Elaborazione con Gemini (riordino, raggruppamento argomenti, struttura)
   - Generazione del documento Word

4. La dispensa viene salvata nella **stessa cartella** del file di partenza con nome `[nome_file]_dispensa.docx`.

## Configurazione

### GPU per la trascrizione

Il trascrittore prova prima la **GPU** (CUDA); se le librerie mancano (es. `cublas64_12.dll` su Windows) o si verifica un errore, passa automaticamente alla **CPU** con modello `base` e avvisa in console. Per usare la GPU servono CUDA 12 e driver NVIDIA aggiornati.

### System Prompt (Gemini)

Il comportamento di Gemini (stile, riordino, formato) si configura in **`prompts.txt`**:
- Le righe che iniziano con `#` sono commenti e non vengono inviate al modello.
- Sotto i commenti inserisci il **System Prompt** che definisce come trasformare la lezione in dispensa (ignorare ripetizioni, raggruppare argomenti, tono accademico, ecc.).

## Struttura del progetto

| File | Descrizione |
|------|-------------|
| `main.py` | Punto di ingresso: chiede il file, orchestra trascrizione → Gemini → Word |
| `transcriber.py` | Estrazione audio da video (moviepy) e trascrizione (faster-whisper) |
| `gemini_engine.py` | Chiamate a Gemini e parsing della risposta strutturata |
| `doc_builder.py` | Creazione del documento Word (python-docx) |
| `prompts.txt` | System Prompt per Gemini (logica LLM) |
| `requirements.txt` | Dipendenze Python |
| `.env.example` | Template per il file `.env` (copia in `.env` e inserisci `GEMINI_API_KEY`) |

## Licenza

Progetto personale/didattico; adatta la licenza in base alle tue esigenze.
