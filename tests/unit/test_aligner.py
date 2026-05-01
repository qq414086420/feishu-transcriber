"""Unit tests for the aligner module."""

from audio_transcribe.diarizer import DiarizationSegment
from audio_transcribe.aligner import align, build_verbatim


class TestAlign:
    """Tests for the align function."""

    def test_perfect_overlap(self):
        """One ASR segment perfectly matches one diarization segment."""
        asr = [{"start": 1.0, "end": 3.0, "text": "hello"}]
        dia = [DiarizationSegment(start=1.0, end=3.0, speaker="spk0")]
        result = align(asr, dia)
        assert len(result) == 1
        assert result[0]["speaker"] == "spk0"
        assert result[0]["text"] == "hello"
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 3.0

    def test_best_overlap_selected(self):
        """One ASR seg overlaps two dia segs — picks the one with more overlap."""
        asr = [{"start": 0.0, "end": 4.0, "text": "hello world"}]
        dia = [
            DiarizationSegment(start=0.0, end=1.0, speaker="spk0"),  # 1s overlap
            DiarizationSegment(start=1.0, end=4.0, speaker="spk1"),  # 3s overlap
        ]
        result = align(asr, dia)
        assert result[0]["speaker"] == "spk1"

    def test_no_diarization_returns_unknown(self):
        """Empty diarization list assigns UNKNOWN to all segments."""
        asr = [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 2.0, "end": 4.0, "text": "world"},
        ]
        result = align(asr, [])
        assert len(result) == 2
        assert all(seg["speaker"] == "UNKNOWN" for seg in result)

    def test_empty_asr_returns_empty(self):
        """Empty ASR list returns empty result."""
        dia = [DiarizationSegment(start=0.0, end=5.0, speaker="spk0")]
        result = align([], dia)
        assert result == []

    def test_multiple_segments(self):
        """Three ASR + three dia segments — each gets the correct speaker."""
        asr = [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 2.0, "end": 4.0, "text": "world"},
            {"start": 4.0, "end": 6.0, "text": "foo"},
        ]
        dia = [
            DiarizationSegment(start=0.0, end=2.0, speaker="spk0"),
            DiarizationSegment(start=2.0, end=4.0, speaker="spk1"),
            DiarizationSegment(start=4.0, end=6.0, speaker="spk2"),
        ]
        result = align(asr, dia)
        assert len(result) == 3
        assert result[0]["speaker"] == "spk0"
        assert result[1]["speaker"] == "spk1"
        assert result[2]["speaker"] == "spk2"

    def test_none_timestamps_in_asr(self):
        """ASR segments with None start/end get UNKNOWN speaker."""
        asr = [
            {"start": None, "end": 2.0, "text": "hello"},
            {"start": 1.0, "end": None, "text": "world"},
            {"start": None, "end": None, "text": "both none"},
        ]
        dia = [DiarizationSegment(start=0.0, end=5.0, speaker="spk0")]
        result = align(asr, dia)
        assert len(result) == 3
        assert all(seg["speaker"] == "UNKNOWN" for seg in result)

    def test_does_not_mutate_input(self):
        """align must not mutate the original ASR dicts."""
        original = {"start": 1.0, "end": 3.0, "text": "hello"}
        asr = [original]
        dia = [DiarizationSegment(start=1.0, end=3.0, speaker="spk0")]
        result = align(asr, dia)
        assert "speaker" not in original
        assert result[0] is not original

    def test_preserves_extra_keys(self):
        """Extra keys in ASR dicts are preserved in the output."""
        asr = [{"start": 1.0, "end": 3.0, "text": "hello", "confidence": 0.95}]
        dia = [DiarizationSegment(start=1.0, end=3.0, speaker="spk0")]
        result = align(asr, dia)
        assert result[0]["confidence"] == 0.95
        assert result[0]["speaker"] == "spk0"


class TestBuildVerbatim:
    """Tests for the build_verbatim function."""

    def test_format(self):
        """Verify output format with multiple segments."""
        segments = [
            {"speaker": "spk0", "text": "hello"},
            {"speaker": "spk1", "text": "world"},
        ]
        result = build_verbatim(segments)
        assert result == "[spk0]: hello\n[spk1]: world\n"

    def test_empty(self):
        """Empty input returns empty string."""
        assert build_verbatim([]) == ""
