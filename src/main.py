"""FastAPI service that transcribes voice memos with the OpenAI API."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
VOICEMEMO_DIR = BASE_DIR / "voicememos"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
SUPPORTED_EXTENSIONS = (".m4a", ".mp3", ".wav", ".mp4")
MAX_CONTENT_SIZE = 25 * 1024 * 1024  # OpenAI Whisper limit is 25 MB
MIN_SEGMENT_DURATION = 15.0  # seconds

# Models supported by the transcriptions endpoint (per OpenAI Audio API)
AVAILABLE_MODELS = [
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-transcribe-diarize",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Memo Transcriber",
    description="Transcribe audio files from the voicememos folder with OpenAI",
    version="0.1.0",
)


class TranscriptionRequest(BaseModel):
    """Request payload for the transcription endpoint."""

    filename: Optional[str] = Field(
        default=None,
        description="Optional audio filename inside the voicememos folder. Default is the newest file.",
    )
    model: str = Field(
        default="whisper-1",
        description="OpenAI model to use for transcription.",
    )


class TranscriptionResponse(BaseModel):
    """Response payload describing the generated transcript."""

    audio_filename: str
    transcript_filename: str
    model: str
    text_preview: str


class FileListResponse(BaseModel):
    """Response payload for listing audio files or transcripts."""

    items: List[str]


def _ensure_directories() -> None:
    """Make sure the expected folders exist before we interact with them."""

    if not VOICEMEMO_DIR.exists():
        raise HTTPException(status_code=500, detail="voicememos directory is missing")
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def _select_audio_file(target_name: Optional[str]) -> Path:
    """Resolve the desired audio file, defaulting to the newest entry."""

    audio_files = sorted(
        [
            path
            for path in VOICEMEMO_DIR.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not audio_files:
        raise HTTPException(
            status_code=404, detail="No audio files found in voicememos"
        )

    if target_name is None:
        return audio_files[0]

    candidate = VOICEMEMO_DIR / target_name
    if not candidate.exists():
        raise HTTPException(
            status_code=404, detail=f"Audio file {target_name} does not exist"
        )
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file extension")

    return candidate


def _transcribe_audio(audio_path: Path, model: str) -> str:
    """Transcribe the provided audio file using the OpenAI API."""

    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError as err:  # pragma: no cover - executed only when key is missing
        logger.error("OPENAI_API_KEY is not set")
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY environment variable is required"
        ) from err

    client = OpenAI(api_key=api_key)

    try:
        with _audio_processing_context(audio_path) as prepared_paths:
            transcripts: List[str] = []
            total_parts = len(prepared_paths)
            for index, prepared_path in enumerate(prepared_paths, start=1):
                logger.info(
                    "Uploading chunk %s/%s (%s bytes)",
                    index,
                    total_parts,
                    prepared_path.stat().st_size,
                )
                with prepared_path.open("rb") as audio_file:
                    try:
                        response = client.audio.transcriptions.create(
                            model=model,
                            file=audio_file,
                        )
                    except Exception as exc:
                        # Handle short-audio errors from OpenAI gracefully by returning
                        # an empty transcript and logging a warning instead of
                        # propagating an error to the client.
                        err_msg = getattr(exc, "args", [None])[0]
                        # openai.BadRequestError from the library carries a .response
                        # with JSON detail; we check for the known 'audio_too_short' code
                        code = None
                        try:
                            # Some exception types include a 'response' attribute
                            # with a .json() method or a .text payload.
                            resp = getattr(exc, "response", None)
                            if resp is not None:
                                # resp may be an httpx.Response or similar
                                try:
                                    j = resp.json()
                                except Exception:
                                    j = None
                                if isinstance(j, dict):
                                    code = j.get("error", {}).get("code")
                        except Exception:
                            code = None

                        if code == "audio_too_short" or (
                            isinstance(err_msg, dict)
                            and err_msg.get("error", {}).get("code")
                            == "audio_too_short"
                        ):
                            logger.warning(
                                "Audio chunk %s was too short for transcription; skipping and returning empty transcript",
                                prepared_path.name,
                            )
                            # Append empty string for this chunk and continue; ultimately
                            # we'll join transcripts which will ignore empty entries.
                            transcripts.append("")
                            continue
                        # Not a short-audio error: re-raise to be handled by outer
                        # exception handler which converts to a 502 HTTP response.
                        raise

                text = getattr(response, "text", None)
                if not text:
                    # If the API returned an empty text (but no exception), treat
                    # it as a warning and append an empty string rather than
                    # failing the whole request.
                    logger.warning(
                        "OpenAI returned empty transcription for %s", prepared_path.name
                    )
                    transcripts.append("")
                else:
                    transcripts.append(text)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Audio file not found: {audio_path.name}"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surface API issues to the client
        logger.exception("OpenAI transcription failed")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    return "\n\n".join(transcripts)


def _write_transcript(audio_path: Path, transcript_text: str) -> Path:
    """Persist the transcript to a text file next to the audio."""

    transcript_path = TRANSCRIPTS_DIR / f"{audio_path.stem}.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    return transcript_path


@contextlib.contextmanager
def _audio_processing_context(audio_path: Path) -> Iterator[List[Path]]:
    """Prepare an audio file for upload, compressing and chunking when needed."""

    temp_files: List[Path] = []
    temp_dirs: List[Path] = []
    try:
        candidate = audio_path
        if candidate.stat().st_size > MAX_CONTENT_SIZE:
            logger.info("Compressing %s to fit upload size limits", candidate.name)
            compressed = _transcode_audio(candidate)
            temp_files.append(compressed)
            candidate = compressed

        if candidate.stat().st_size <= MAX_CONTENT_SIZE:
            yield [candidate]
            return

        logger.info("Splitting %s into multiple chunks for upload", candidate.name)
        chunks, chunk_dir = _chunk_audio(candidate)
        temp_files.extend(chunks)
        temp_dirs.append(chunk_dir)
        yield chunks
    finally:
        for file_path in temp_files:
            with contextlib.suppress(FileNotFoundError):
                file_path.unlink()
        for dir_path in temp_dirs:
            shutil.rmtree(dir_path, ignore_errors=True)


def _transcode_audio(audio_path: Path) -> Path:
    """Convert the audio to a compressed mono MP3 using ffmpeg."""

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    except OSError as exc:  # pragma: no cover - temp failures are rare
        raise HTTPException(
            status_code=500, detail="Unable to allocate temp file for conversion"
        ) from exc

    temp_path = Path(temp_file.name)
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-b:a",
        "64k",
        str(temp_path),
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail="ffmpeg is required to compress large audio files"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail="ffmpeg failed to compress the audio"
        ) from exc

    return temp_path


def _chunk_audio(audio_path: Path) -> tuple[List[Path], Path]:
    """Split an audio file into <=25MB segments using ffmpeg."""

    duration = _probe_duration(audio_path)
    if duration <= 0:
        raise HTTPException(
            status_code=500, detail="Unable to determine audio duration for chunking"
        )

    total_size = audio_path.stat().st_size
    chunk_count = max(2, math.ceil(total_size / MAX_CONTENT_SIZE))
    segment_time = max(duration / chunk_count, MIN_SEGMENT_DURATION)

    temp_dir = Path(tempfile.mkdtemp(prefix="transcriber_chunks_"))
    output_pattern = temp_dir / "chunk_%03d.mp3"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        f"{segment_time:.2f}",
        "-c",
        "copy",
        str(output_pattern),
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail="ffmpeg is required to split large audio files"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail="ffmpeg failed to split the audio"
        ) from exc

    chunks = sorted(temp_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise HTTPException(
            status_code=500, detail="ffmpeg did not produce any audio chunks"
        )

    too_large = [chunk for chunk in chunks if chunk.stat().st_size > MAX_CONTENT_SIZE]
    if too_large:
        raise HTTPException(
            status_code=400, detail="Audio chunks remain too large after splitting"
        )

    return chunks, temp_dir


def _probe_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail="ffprobe is required to split large audio files"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail="ffprobe failed to inspect the audio file"
        ) from exc

    try:
        return float(completed.stdout.strip())
    except ValueError as exc:  # pragma: no cover - depends on ffprobe output
        raise HTTPException(
            status_code=500, detail="Unable to parse audio duration"
        ) from exc


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Simple health endpoint for monitoring."""

    return {"status": "ok"}


@app.get("/")
def root() -> RedirectResponse:
    """Redirect root to the interactive docs."""

    return RedirectResponse(url="/docs")


@app.get("/models", response_model=FileListResponse)
def list_models() -> FileListResponse:
    """List available transcription models."""

    items = AVAILABLE_MODELS.copy()
    return FileListResponse(items=items)


@app.get("/audio", response_model=FileListResponse)
def list_audio_files() -> FileListResponse:
    """List all supported audio files waiting for transcription."""

    _ensure_directories()
    items = sorted(
        path.name
        for path in VOICEMEMO_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return FileListResponse(items=items)


@app.get("/transcripts", response_model=FileListResponse)
def list_transcripts() -> FileListResponse:
    """List saved transcript files."""

    _ensure_directories()
    items = sorted(path.name for path in TRANSCRIPTS_DIR.glob("*.txt"))
    return FileListResponse(items=items)


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_audio(payload: TranscriptionRequest) -> TranscriptionResponse:
    """Transcribe a specific audio file or the newest one in the folder."""

    _ensure_directories()
    audio_path = _select_audio_file(payload.filename)
    logger.info("Transcribing %s with model %s", audio_path.name, payload.model)

    if payload.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model {payload.model} is not supported. See /models for available models",
        )

    transcript_text = _transcribe_audio(audio_path, payload.model)
    transcript_path = _write_transcript(audio_path, transcript_text)

    preview = transcript_text[:120].replace("\n", " ") + (
        "…" if len(transcript_text) > 120 else ""
    )

    return TranscriptionResponse(
        audio_filename=audio_path.name,
        transcript_filename=transcript_path.name,
        model=payload.model,
        text_preview=preview,
    )
