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


def _parse_diarization_output(raw_segments: list) -> list[DiarizationSegment]:
    """Parse diarization output into DiarizationSegment objects.

    Output format: [[start, end, speaker_id], ...] where times are in seconds.
    """
    segments: list[DiarizationSegment] = []
    for item in raw_segments:
        if len(item) < 3:
            continue
        start = float(item[0])
        end = float(item[1])
        speaker = f"SPEAKER_{item[2]:02d}"
        segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))
    return segments


def diarize(input_path: Path, device: str = "cuda") -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Returns a list of DiarizationSegment objects, or an empty list on failure.
    """
    try:
        pipeline = _create_diarization_pipeline(device)
        result = pipeline(audio=str(input_path))
        raw_segments = result.get("text", [])
        if not raw_segments:
            return []
        return _parse_diarization_output(raw_segments)
    except Exception:
        logger.exception("Diarization failed for %s", input_path)
        return []
