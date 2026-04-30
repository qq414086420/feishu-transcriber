"""Unit tests for audio_transcribe tool."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests for TranscriptionResult
# ---------------------------------------------------------------------------
class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass and to_json method."""

    def test_to_json_creates_valid_output(self, tmp_path: Path) -> None:
        """to_json writes a JSON file with correct structure."""
        from audio_transcribe.transcriber import TranscriptionResult

        result = TranscriptionResult(
            text="hello world",
            segments=[{"start": 0.0, "end": 1.5, "text": "hello world"}],
            language="en",
            duration=3.0,
        )
        output = tmp_path / "out.json"
        result.to_json(output)

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["text"] == "hello world"
        assert len(data["segments"]) == 1
        assert data["segments"][0]["start"] == 0.0
        assert data["segments"][0]["end"] == 1.5
        assert data["segments"][0]["text"] == "hello world"
        assert data["language"] == "en"
        assert data["duration"] == 3.0

    def test_to_json_unicode(self, tmp_path: Path) -> None:
        """Chinese text is preserved in JSON output (ensure_ascii=False)."""
        from audio_transcribe.transcriber import TranscriptionResult

        result = TranscriptionResult(
            text="你好世界",
            segments=[],
            language="zh",
            duration=2.0,
        )
        output = tmp_path / "unicode.json"
        result.to_json(output)

        raw = output.read_text(encoding="utf-8")
        assert "你好世界" in raw
        data = json.loads(raw)
        assert data["text"] == "你好世界"


# ---------------------------------------------------------------------------
# Tests for _parse_segments
# ---------------------------------------------------------------------------
class TestParseSegments:
    """Tests for the _parse_segments helper."""

    def test_no_timestamp_returns_single_segment(self) -> None:
        """When no timestamp, returns single segment with full text."""
        from audio_transcribe.transcriber import _parse_segments

        segments = _parse_segments(None, "hello world")
        assert len(segments) == 1
        assert segments[0]["text"] == "hello world"
        assert segments[0]["start"] is None
        assert segments[0]["end"] is None

    def test_empty_text_returns_empty_list(self) -> None:
        """When text is empty and no timestamps, returns empty list."""
        from audio_transcribe.transcriber import _parse_segments

        segments = _parse_segments(None, "")
        assert segments == []

    def test_parse_timestamp_segments(self) -> None:
        """Parses 's1,e1;s2,e2;...' into segment dicts with seconds."""
        from audio_transcribe.transcriber import _parse_segments

        segments = _parse_segments("0,1500;1500,3000", "hello world foo")
        assert len(segments) == 2
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.5
        assert segments[1]["start"] == 1.5
        assert segments[1]["end"] == 3.0

    def test_parse_timestamp_single_segment(self) -> None:
        """Single timestamp pair produces one segment."""
        from audio_transcribe.transcriber import _parse_segments

        segments = _parse_segments("100,2000", "hi")
        assert len(segments) == 1
        assert segments[0]["start"] == 0.1
        assert segments[0]["end"] == 2.0
        assert segments[0]["text"] == "hi"


# ---------------------------------------------------------------------------
# Tests for transcribe function
# ---------------------------------------------------------------------------
class TestTranscribe:
    """Tests for the main transcribe function."""

    def test_transcribe_returns_result(self, tmp_path: Path) -> None:
        """Successful transcription returns TranscriptionResult and writes JSON."""
        from audio_transcribe.transcriber import transcribe

        input_path = tmp_path / "audio.wav"
        input_path.write_bytes(b"\x00" * 100)

        output_path = tmp_path / "output" / "audio.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()
        mock_model.generate.return_value = [
            {"text": "你好世界", "timestamp": "0,1500;1500,3000"}
        ]

        with patch("audio_transcribe.transcriber._create_model", return_value=mock_model):
            result = transcribe(input_path, output_path)

        assert result is not None
        assert result.text == "你好世界"
        assert result.language == "zh"
        assert len(result.segments) == 2
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["text"] == "你好世界"
        assert "segments" in data
        assert data["language"] == "zh"

    def test_transcribe_empty_audio(self, tmp_path: Path) -> None:
        """Transcription with empty text returns result with empty segments."""
        from audio_transcribe.transcriber import transcribe

        input_path = tmp_path / "silence.wav"
        input_path.write_bytes(b"\x00" * 100)

        output_path = tmp_path / "output" / "silence.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": ""}]

        with patch("audio_transcribe.transcriber._create_model", return_value=mock_model):
            result = transcribe(input_path, output_path)

        assert result is not None
        assert result.text == ""
        assert result.segments == []

    def test_transcribe_missing_input_returns_none(self, tmp_path: Path) -> None:
        """Non-existent input file returns None."""
        from audio_transcribe.transcriber import transcribe

        missing = tmp_path / "does_not_exist.wav"
        output_path = tmp_path / "output.json"

        result = transcribe(missing, output_path)
        assert result is None

    def test_transcribe_model_error_returns_none(self, tmp_path: Path) -> None:
        """When model raises an exception, returns None."""
        from audio_transcribe.transcriber import transcribe

        input_path = tmp_path / "audio.wav"
        input_path.write_bytes(b"\x00" * 100)

        output_path = tmp_path / "output" / "audio.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("audio_transcribe.transcriber._create_model", side_effect=RuntimeError("GPU OOM")):
            result = transcribe(input_path, output_path)

        assert result is None
