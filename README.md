# Voice Meeting Transcriber

Simple FastAPI service that turns iPhone voice memos into text files using the OpenAI transcription API.

## Prerequisites

- Python 3.10+
- An OpenAI API key with access to the Whisper (or compatible) transcription model
- `ffmpeg` available on your PATH (used to compress large voice notes automatically)

## Setup

1. Create a virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export your API key:
   ```bash
   export OPENAI_API_KEY="sk-your-key"
   ```
4. Put `.m4a` files inside `src/voicememos/`. The app will also accept `.mp3`, `.wav`, or `.mp4` files.

## Running the API

Start the FastAPI server with Uvicorn:

```bash
# Using uv (project runner) if available
uv run --env-file .env uvicorn src.main:app --reload

# Or directly with uvicorn
uvicorn src.main:app --reload
```

Once the server is running, open http://127.0.0.1:8000/docs to explore the interactive API.

> The service automatically compresses audio files larger than 25 MB before sending them to the OpenAI API. If compression fails, ensure `ffmpeg` is installed (e.g., `brew install ffmpeg` on macOS).

## Available Endpoints

- `GET /health` – Lightweight health check.
- `GET /audio` – List audio files waiting for transcription in `src/voicememos/`.
- `GET /transcripts` – List generated `.txt` transcripts in `src/transcripts/`.
- `POST /transcribe` – Trigger transcription. Payload example:
  ```json
  {
    "filename": "my_audio_1.m4a",
    "model": "whisper-1"
  }
  ```
  If `filename` is omitted, the newest audio file in the folder is used.

- `GET /models` – List available transcription models (e.g. `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`).

Transcriptions are saved to `src/transcripts/<audio-name>.txt`.

Note: very short audio chunks (OpenAI "audio_too_short") are handled gracefully — the server will log a warning and produce an empty transcript for that chunk instead of returning an error.

## Model selection & advanced options

This service lets you pick which OpenAI transcription model to use by setting the `model` field in the `/transcribe` request (see `GET /models` for available options). Brief notes:

- Available models include `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`, and `gpt-4o-transcribe-diarize`.
- Response formats differ by model:
  - `whisper-1` supports `json`, `text`, `srt`, `verbose_json`, and `vtt`.
  - `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` support `json` and `text`.
  - `gpt-4o-transcribe-diarize` supports `json`, `text`, and `diarized_json` (speaker segments).
- Diarization: to get speaker-aware output use `gpt-4o-transcribe-diarize` and request `response_format=diarized_json`. For inputs longer than ~30s set `chunking_strategy="auto"`. You can optionally supply short reference clips (2–10s) encoded as data URLs together with `known_speaker_names` and `known_speaker_references` to map segments to known speakers.
- Translations: the Audio API also offers a translations endpoint to translate audio into English. This service currently uses the transcriptions endpoint; to translate you can call the Audio API's translations endpoint directly or extend this service to forward translation parameters.

Note: the OpenAI Audio API supports additional parameters (for example `response_format`, `chunking_strategy`, and speaker reference fields). This project currently forwards only `model` and the audio file — if you'd like, I can add support to pass extra transcription parameters through the HTTP payload and into the OpenAI client.
