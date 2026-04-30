"""Unit tests for media_to_audio tool."""

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_wav(path: Path, sample_rate: int, channels: int, num_samples: int) -> None:
    """Write a valid WAV file with the given parameters."""
    with wave.open(str(path), "w") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(
            struct.pack(
                f"<{num_samples * channels}h",
                *([0] * num_samples * channels),
            )
        )


# ---------------------------------------------------------------------------
# Tests for converter.is_target_wav
# ---------------------------------------------------------------------------
class TestIsTargetWav:
    """Tests for is_target_wav: detecting 16kHz mono WAV files."""

    def test_correct_16khz_mono_wav_returns_true(self, tmp_path: Path) -> None:
        """A valid 16kHz mono WAV file returns True."""
        from media_to_audio.converter import is_target_wav

        wav_path = tmp_path / "correct.wav"
        _write_wav(wav_path, sample_rate=16000, channels=1, num_samples=100)
        assert is_target_wav(wav_path) is True

    def test_wrong_sample_rate_returns_false(self, tmp_path: Path) -> None:
        """A 44100Hz WAV file returns False."""
        from media_to_audio.converter import is_target_wav

        wav_path = tmp_path / "44100.wav"
        _write_wav(wav_path, sample_rate=44100, channels=1, num_samples=100)
        assert is_target_wav(wav_path) is False

    def test_stereo_returns_false(self, tmp_path: Path) -> None:
        """A stereo (2-channel) 16kHz WAV file returns False."""
        from media_to_audio.converter import is_target_wav

        wav_path = tmp_path / "stereo.wav"
        _write_wav(wav_path, sample_rate=16000, channels=2, num_samples=100)
        assert is_target_wav(wav_path) is False

    def test_non_wav_extension_returns_false(self, tmp_path: Path) -> None:
        """A non-.wav file extension returns False even if content exists."""
        from media_to_audio.converter import is_target_wav

        mp3_path = tmp_path / "audio.mp3"
        mp3_path.write_bytes(b"\x00" * 100)
        assert is_target_wav(mp3_path) is False

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        """A nonexistent file path returns False."""
        from media_to_audio.converter import is_target_wav

        missing = tmp_path / "does_not_exist.wav"
        assert is_target_wav(missing) is False


# ---------------------------------------------------------------------------
# Tests for converter.convert_to_wav
# ---------------------------------------------------------------------------
class TestConvertToWav:
    """Tests for convert_to_wav: converting media files to 16kHz mono WAV."""

    def test_already_target_format_creates_symlink(self, tmp_path: Path) -> None:
        """When input is already 16kHz mono WAV, a symlink is created."""
        from media_to_audio.converter import convert_to_wav

        input_path = tmp_path / "input"
        input_path.mkdir()
        wav_file = input_path / "meeting.wav"
        _write_wav(wav_file, sample_rate=16000, channels=1, num_samples=100)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = convert_to_wav(wav_file, output_dir)

        assert result is not None
        assert result.exists()
        assert result.is_symlink()
        assert result.resolve() == wav_file.resolve()

    def test_mp3_conversion_calls_ffmpeg(self, tmp_path: Path) -> None:
        """An mp3 file triggers ffmpeg via subprocess.run with correct args."""
        from media_to_audio.converter import convert_to_wav

        input_path = tmp_path / "input"
        input_path.mkdir()
        mp3_file = input_path / "audio.mp3"
        mp3_file.write_bytes(b"\xff\xfb" + b"\x00" * 100)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("media_to_audio.converter.subprocess.run", return_value=mock_result) as mock_run:
            result = convert_to_wav(mp3_file, output_dir)

        assert result is not None
        assert result == output_dir / "audio.wav"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-i" in cmd
        assert str(mp3_file) in cmd
        assert "-ar" in cmd
        assert "16000" in cmd
        assert "-ac" in cmd
        assert "1" in cmd
        assert "-c:a" in cmd
        assert "pcm_s16le" in cmd
        assert str(output_dir / "audio.wav") in cmd
        assert call_args[1]["timeout"] == 600

    def test_ffmpeg_failure_returns_none(self, tmp_path: Path) -> None:
        """When ffmpeg exits with non-zero code, returns None."""
        from subprocess import CalledProcessError

        from media_to_audio.converter import convert_to_wav

        input_path = tmp_path / "input"
        input_path.mkdir()
        mp3_file = input_path / "bad.mp3"
        mp3_file.write_bytes(b"\x00" * 10)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch(
            "media_to_audio.converter.subprocess.run",
            side_effect=CalledProcessError(returncode=1, cmd="ffmpeg"),
        ):
            result = convert_to_wav(mp3_file, output_dir)

        assert result is None

    def test_ffmpeg_not_found_returns_none(self, tmp_path: Path) -> None:
        """When ffmpeg is not installed (FileNotFoundError), returns None."""
        from media_to_audio.converter import convert_to_wav

        input_path = tmp_path / "input"
        input_path.mkdir()
        mp3_file = input_path / "audio.mp3"
        mp3_file.write_bytes(b"\x00" * 10)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("media_to_audio.converter.subprocess.run", side_effect=FileNotFoundError):
            result = convert_to_wav(mp3_file, output_dir)

        assert result is None

    def test_ffmpeg_timeout_returns_none(self, tmp_path: Path) -> None:
        """When ffmpeg times out, returns None."""
        from subprocess import TimeoutExpired
        from media_to_audio.converter import convert_to_wav

        input_path = tmp_path / "input"
        input_path.mkdir()
        mp3_file = input_path / "long.mp3"
        mp3_file.write_bytes(b"\x00" * 10)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("media_to_audio.converter.subprocess.run", side_effect=TimeoutExpired(cmd="ffmpeg", timeout=600)):
            result = convert_to_wav(mp3_file, output_dir)

        assert result is None
