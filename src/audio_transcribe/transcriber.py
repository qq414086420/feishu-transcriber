"""Transcribe audio using SenseVoice via FunASR."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from audio_transcribe.aligner import align, build_verbatim
from audio_transcribe.diarizer import diarize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable result of an audio transcription."""

    text: str
    segments: list[dict]
    language: str
    duration: float
    speakers: tuple[str, ...] = ()
    verbatim: str = ""

    def to_json(self, path: Path) -> None:
        """Write the transcription result to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "text": self.text,
            "segments": self.segments,
            "speakers": list(self.speakers),
            "language": self.language,
            "duration": self.duration,
            "verbatim": self.verbatim,
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _create_model(device: str):
    """Create a FunASR AutoModel with SenseVoice.

    This helper isolates the FunASR import so tests can mock it.
    """
    from funasr import AutoModel  # type: ignore[import-untyped]

    return AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", device=device)


def _clean_sensevoice_text(text: str) -> str:
    """Remove SenseVoice special tokens like <|zh|><|NEUTRAL|><|Speech|><|withitn|>."""
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def _parse_segments(timestamp_str: str | None, text: str) -> list[dict]:
    """Parse FunASR timestamp string into segment dicts.

    timestamp format: "start1,end1;start2,end2;..." in milliseconds.
    Text is distributed evenly across segments.
    """
    if not timestamp_str:
        if not text:
            return []
        return [{"start": None, "end": None, "text": text}]

    pairs = timestamp_str.split(";")
    num_segments = len(pairs)

    # Distribute text evenly across segments
    if num_segments == 0 or not text:
        char_per_seg = 0
    else:
        char_per_seg = len(text) // num_segments

    segments: list[dict] = []
    for i, pair in enumerate(pairs):
        parts = pair.split(",")
        if len(parts) != 2:
            continue
        start_ms, end_ms = int(parts[0]), int(parts[1])
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        if i < num_segments - 1:
            seg_text = text[i * char_per_seg : (i + 1) * char_per_seg]
        else:
            # Last segment gets the remainder
            seg_text = text[i * char_per_seg :]

        segments.append({"start": start_sec, "end": end_sec, "text": seg_text})

    return segments


def transcribe(
    input_path: Path,
    output_path: Path,
    language: str = "zh",
    device: str = "cpu",
) -> TranscriptionResult | None:
    """Transcribe an audio file using SenseVoice via FunASR.

    Returns TranscriptionResult on success, None on failure.
    """
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return None

    try:
        model = _create_model(device)
        results = model.generate(input=str(input_path), language=language, use_itn=True)

        if not results:
            logger.warning("Model returned empty results for %s", input_path)
            return None

        first = results[0]
        raw_text = first.get("text", "")
        text = _clean_sensevoice_text(raw_text)
        timestamp_str = first.get("timestamp")

        segments = _parse_segments(timestamp_str, text)

        # Calculate total duration from segments if available
        if segments and segments[-1].get("end") is not None:
            duration = segments[-1]["end"]
        else:
            duration = 0.0

        result = TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            duration=duration,
        )

        result.to_json(output_path)
        logger.info("Transcription saved to %s", output_path)
        return result

    except Exception:
        logger.exception("Transcription failed for %s", input_path)
        return None


def transcribe_with_speakers(
    input_path: Path,
    output_path: Path,
    language: str = "zh",
    device: str = "cpu",
) -> TranscriptionResult | None:
    """Transcribe audio with speaker diarization.

    Runs SenseVoice ASR + 3D-Speaker diarization, aligns results,
    and outputs a verbatim transcript with speaker labels.

    Falls back to UNKNOWN speakers if diarization fails.
    """
    asr_result = transcribe(input_path, output_path, language=language, device=device)
    if asr_result is None:
        return None

    diarization_segments = diarize(input_path, device=device)
    aligned = align(asr_result.segments, diarization_segments)

    speakers = tuple(sorted({seg["speaker"] for seg in aligned}))
    verbatim = build_verbatim(aligned)
    duration = aligned[-1]["end"] if aligned and aligned[-1].get("end") is not None else asr_result.duration

    enriched = TranscriptionResult(
        text=asr_result.text,
        segments=aligned,
        language=asr_result.language,
        duration=duration,
        speakers=speakers,
        verbatim=verbatim,
    )

    enriched.to_json(output_path)
    logger.info("Transcription with speakers saved to %s (%d speakers)", output_path, len(speakers))
    return enriched
