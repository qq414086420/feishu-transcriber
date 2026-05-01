"""Speaker diarization using 3D-Speaker via ModelScope."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiarizationSegment:
    """Immutable segment representing a speaker's turn."""

    start: float
    end: float
    speaker: str


def _create_diarization_pipeline(device: str):
    """Create a ModelScope speaker diarization pipeline.

    This helper isolates the ModelScope import so tests can mock it.
    """
    from modelscope.pipelines import pipeline  # type: ignore[import-untyped]
    from modelscope.utils.constant import Tasks  # type: ignore[import-untyped]

    return pipeline(
        task=Tasks.speaker_diarization,
        model="iic/speech_campplus_speaker-diarization_common",
        device=device,
    )


def _parse_rttm(rttm_text: str) -> list[DiarizationSegment]:
    """Parse RTTM format text into DiarizationSegment objects.

    RTTM format per line:
        SPEAKER <name> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    segments: list[DiarizationSegment] = []
    for line in rttm_text.strip().splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker_id = parts[7] if len(parts) > 7 else "unknown"
        segments.append(
            DiarizationSegment(
                start=start,
                end=start + duration,
                speaker=speaker_id,
            )
        )
    return segments


def diarize(input_path: Path, device: str = "cuda") -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Returns a list of DiarizationSegment objects, or an empty list on failure.
    """
    try:
        pipeline = _create_diarization_pipeline(device)
        result = pipeline(input=str(input_path))
        rttm_text = result.get("text", "")
        if not rttm_text:
            return []
        return _parse_rttm(rttm_text)
    except Exception:
        logger.exception("Diarization failed for %s", input_path)
        return []
