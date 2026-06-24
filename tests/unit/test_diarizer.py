"""Unit tests for the speaker diarization module."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests for DiarizationSegment
# ---------------------------------------------------------------------------
class TestDiarizationSegment:
    """Tests for the DiarizationSegment frozen dataclass."""

    def test_frozen_dataclass(self) -> None:
        """DiarizationSegment is frozen and immutable."""
        from audio_transcribe.diarizer import DiarizationSegment

        seg = DiarizationSegment(start=0.0, end=1.5, speaker="spk1")
        with pytest.raises(FrozenInstanceError):
            seg.start = 2.0  # type: ignore[misc]

    def test_hashable(self) -> None:
        """DiarizationSegment is hashable (can be used in sets/dicts)."""
        from audio_transcribe.diarizer import DiarizationSegment

        seg1 = DiarizationSegment(start=0.0, end=1.5, speaker="spk1")
        seg2 = DiarizationSegment(start=0.0, end=1.5, speaker="spk1")
        assert seg1 == seg2
        assert hash(seg1) == hash(seg2)
        assert len({seg1, seg2}) == 1


# ---------------------------------------------------------------------------
# Tests for _parse_diarization_output
# ---------------------------------------------------------------------------
class TestParseDiarizationOutput:
    """Tests for parsing the diarization output format."""

    def test_parse_list_format(self) -> None:
        """Parses [[start, end, speaker_id], ...] format."""
        from audio_transcribe.diarizer import _parse_diarization_output

        raw = [[0.07, 1.63, 0], [1.63, 26.24, 1], [26.24, 111.40, 0]]
        segments = _parse_diarization_output(raw)

        assert len(segments) == 3
        assert segments[0].start == 0.07
        assert segments[0].end == 1.63
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"
        assert segments[2].speaker == "SPEAKER_00"

    def test_empty_input(self) -> None:
        """Empty list returns empty segments."""
        from audio_transcribe.diarizer import _parse_diarization_output

        assert _parse_diarization_output([]) == []

    def test_short_items_skipped(self) -> None:
        """Items with fewer than 3 elements are skipped."""
        from audio_transcribe.diarizer import _parse_diarization_output

        raw = [[0.0, 1.0], [1.0, 2.0, 0]]
        segments = _parse_diarization_output(raw)
        assert len(segments) == 1
        assert segments[0].speaker == "SPEAKER_00"


# ---------------------------------------------------------------------------
# Tests for diarize
# ---------------------------------------------------------------------------
class TestDiarize:
    """Tests for the main diarize function."""

    def test_returns_segments(self, tmp_path: Path) -> None:
        """Successful diarization returns parsed DiarizationSegment list."""
        from audio_transcribe.diarizer import DiarizationSegment, diarize

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"\x00" * 100)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"text": [[0.0, 1.5, 0], [1.5, 3.5, 1]]}

        with patch(
            "audio_transcribe.diarizer._create_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            segments = diarize(audio_path)

        assert len(segments) == 2
        assert isinstance(segments[0], DiarizationSegment)
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].start == 1.5
        assert segments[1].end == 3.5
        assert segments[1].speaker == "SPEAKER_01"

    def test_empty_result_on_failure(self, tmp_path: Path) -> None:
        """When pipeline raises RuntimeError, returns empty list."""
        from audio_transcribe.diarizer import diarize

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"\x00" * 100)

        with patch(
            "audio_transcribe.diarizer._create_diarization_pipeline",
            side_effect=RuntimeError("GPU OOM"),
        ):
            segments = diarize(audio_path)

        assert segments == []

    def test_graceful_degradation(self, tmp_path: Path) -> None:
        """When pipeline returns empty dict, returns empty list."""
        from audio_transcribe.diarizer import diarize

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"\x00" * 100)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {}

        with patch(
            "audio_transcribe.diarizer._create_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            segments = diarize(audio_path)

        assert segments == []
