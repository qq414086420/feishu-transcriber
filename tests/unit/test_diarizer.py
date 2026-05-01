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
# Tests for diarize
# ---------------------------------------------------------------------------
class TestDiarize:
    """Tests for the main diarize function."""

    def test_returns_segments(self, tmp_path: Path) -> None:
        """Successful diarization returns parsed DiarizationSegment list."""
        from audio_transcribe.diarizer import DiarizationSegment, diarize

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"\x00" * 100)

        rttm_text = (
            "SPEAKER audio 1 0.00 1.50 <NA> <NA> spk1 <NA> <NA>\n"
            "SPEAKER audio 1 1.50 2.00 <NA> <NA> spk2 <NA> <NA>\n"
        )

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"text": rttm_text}

        with patch(
            "audio_transcribe.diarizer._create_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            segments = diarize(audio_path)

        assert len(segments) == 2
        assert isinstance(segments[0], DiarizationSegment)
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[0].speaker == "spk1"
        assert segments[1].start == 1.5
        assert segments[1].end == 3.5
        assert segments[1].speaker == "spk2"

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
