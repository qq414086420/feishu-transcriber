"""Unit tests for pipeline_run orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests for PipelineStep
# ---------------------------------------------------------------------------
class TestPipelineStep:
    """Tests for the PipelineStep dataclass and execute method."""

    def test_successful_step(self) -> None:
        """Successful subprocess returns output_path from stdout."""
        from pipeline_run.runner import PipelineStep

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/tmp/work/audio/meeting.wav\n"

        step = PipelineStep(
            name="media_to_audio",
            cmd=["python", "-m", "media_to_audio", "--input", "/tmp/work/inbox/meeting.mp3"],
        )

        with patch("pipeline_run.runner.subprocess.run", return_value=mock_result) as mock_run:
            result = step.execute()

        mock_run.assert_called_once_with(
            step.cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.success is True
        assert result.output_path == "/tmp/work/audio/meeting.wav"
        assert result.error == ""

    def test_failed_step(self) -> None:
        """Non-zero returncode yields success=False with stderr as error."""
        from pipeline_run.runner import PipelineStep

        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stderr = "File not found"

        step = PipelineStep(
            name="media_to_audio",
            cmd=["python", "-m", "media_to_audio", "--input", "bad.mp3"],
        )

        with patch("pipeline_run.runner.subprocess.run", return_value=mock_result):
            result = step.execute()

        assert result.success is False
        assert "File not found" in result.error

    def test_timeout_step(self) -> None:
        """SubprocessTimeoutExpired yields a timed-out error."""
        from subprocess import TimeoutExpired

        from pipeline_run.runner import PipelineStep

        step = PipelineStep(name="audio_transcribe", cmd=["python", "-m", "audio_transcribe"])

        with patch("pipeline_run.runner.subprocess.run", side_effect=TimeoutExpired(cmd="python", timeout=1800)):
            result = step.execute()

        assert result.success is False
        assert "timed out" in result.error

    def test_command_not_found_step(self) -> None:
        """FileNotFoundError yields a command-not-found error."""
        from pipeline_run.runner import PipelineStep

        step = PipelineStep(name="bad_tool", cmd=["nonexistent_binary"])

        with patch("pipeline_run.runner.subprocess.run", side_effect=FileNotFoundError("not found")):
            result = step.execute()

        assert result.success is False
        assert "command not found" in result.error


# ---------------------------------------------------------------------------
# Tests for run_pipeline
# ---------------------------------------------------------------------------
class TestRunPipeline:
    """Tests for the run_pipeline orchestrator function."""

    def test_full_pipeline_success(self, tmp_path: Path) -> None:
        """All steps succeed returns status ok with paths."""
        from pipeline_run.runner import run_pipeline

        # Create a fake input file
        file_path = tmp_path / "meeting.mp3"
        file_path.write_bytes(b"fake audio data")

        work_dir = tmp_path / "work"

        # Mock three subprocess calls: media_to_audio, audio_transcribe, text_summarize
        mock_step1 = MagicMock()
        mock_step1.returncode = 0
        mock_step1.stdout = str(work_dir / "audio" / "meeting.wav")

        mock_step2 = MagicMock()
        mock_step2.returncode = 0
        mock_step2.stdout = str(work_dir / "transcripts" / "meeting.txt")

        mock_step3 = MagicMock()
        mock_step3.returncode = 0
        mock_step3.stdout = str(work_dir / "summaries" / "meeting_summary.md")

        with patch("pipeline_run.runner.subprocess.run", side_effect=[mock_step1, mock_step2, mock_step3]):
            result = run_pipeline(file_path=str(file_path), work_dir=str(work_dir))

        assert result["status"] == "ok"
        assert result["transcript_path"] == str(work_dir / "transcripts" / "meeting.txt")
        assert result["summary_path"] == str(work_dir / "summaries" / "meeting_summary.md")

    def test_pipeline_stops_on_step_failure(self, tmp_path: Path) -> None:
        """Pipeline stops and returns error when a step fails."""
        from pipeline_run.runner import run_pipeline

        file_path = tmp_path / "meeting.mp3"
        file_path.write_bytes(b"fake audio data")

        work_dir = tmp_path / "work"

        # Step 1 succeeds, step 2 fails
        mock_step1 = MagicMock()
        mock_step1.returncode = 0
        mock_step1.stdout = str(work_dir / "audio" / "meeting.wav")

        mock_step2 = MagicMock()
        mock_step2.returncode = 2
        mock_step2.stderr = "Transcription failed"

        with patch("pipeline_run.runner.subprocess.run", side_effect=[mock_step1, mock_step2]):
            result = run_pipeline(file_path=str(file_path), work_dir=str(work_dir))

        assert result["status"] == "error"
        assert result["step"] == "audio_transcribe"
        assert "Transcription failed" in result["error"]

    def test_pipeline_no_summarize(self, tmp_path: Path) -> None:
        """With no_summarize=True, only 2 subprocess calls are made."""
        from pipeline_run.runner import run_pipeline

        file_path = tmp_path / "meeting.mp3"
        file_path.write_bytes(b"fake audio data")

        work_dir = tmp_path / "work"

        mock_step1 = MagicMock()
        mock_step1.returncode = 0
        mock_step1.stdout = str(work_dir / "audio" / "meeting.wav")

        mock_step2 = MagicMock()
        mock_step2.returncode = 0
        mock_step2.stdout = str(work_dir / "transcripts" / "meeting.txt")

        with patch("pipeline_run.runner.subprocess.run", side_effect=[mock_step1, mock_step2]) as mock_run:
            result = run_pipeline(file_path=str(file_path), work_dir=str(work_dir), no_summarize=True)

        assert result["status"] == "ok"
        assert result["transcript_path"] == str(work_dir / "transcripts" / "meeting.txt")
        assert "summary_path" not in result
        # Only 2 subprocess calls: media_to_audio + audio_transcribe
        assert mock_run.call_count == 2

    def test_pipeline_cleanup(self, tmp_path: Path) -> None:
        """With cleanup=True, intermediate dirs are removed."""
        from pipeline_run.runner import run_pipeline

        file_path = tmp_path / "meeting.mp3"
        file_path.write_bytes(b"fake audio data")

        work_dir = tmp_path / "work"

        mock_step1 = MagicMock()
        mock_step1.returncode = 0
        mock_step1.stdout = str(work_dir / "audio" / "meeting.wav")

        mock_step2 = MagicMock()
        mock_step2.returncode = 0
        mock_step2.stdout = str(work_dir / "transcripts" / "meeting.txt")

        mock_step3 = MagicMock()
        mock_step3.returncode = 0
        mock_step3.stdout = str(work_dir / "summaries" / "meeting_summary.md")

        with patch("pipeline_run.runner.subprocess.run", side_effect=[mock_step1, mock_step2, mock_step3]):
            result = run_pipeline(file_path=str(file_path), work_dir=str(work_dir), cleanup=True)

        assert result["status"] == "ok"
        # Intermediate dirs should be removed
        assert not (work_dir / "inbox").exists()
        assert not (work_dir / "audio").exists()
        assert not (work_dir / "transcripts").exists()
        # Summaries dir should remain
        assert (work_dir / "summaries").exists()

    def test_pipeline_copies_file_to_inbox(self, tmp_path: Path) -> None:
        """File not in inbox is copied there before processing."""
        from pipeline_run.runner import run_pipeline

        file_path = tmp_path / "original" / "meeting.mp3"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"fake audio data")

        work_dir = tmp_path / "work"

        mock_step1 = MagicMock()
        mock_step1.returncode = 0
        mock_step1.stdout = str(work_dir / "audio" / "meeting.wav")

        mock_step2 = MagicMock()
        mock_step2.returncode = 0
        mock_step2.stdout = str(work_dir / "transcripts" / "meeting.txt")

        mock_step3 = MagicMock()
        mock_step3.returncode = 0
        mock_step3.stdout = str(work_dir / "summaries" / "meeting_summary.md")

        with patch("pipeline_run.runner.subprocess.run", side_effect=[mock_step1, mock_step2, mock_step3]):
            result = run_pipeline(file_path=str(file_path), work_dir=str(work_dir))

        # File should have been copied to inbox
        assert (work_dir / "inbox" / "meeting.mp3").exists()
        assert result["status"] == "ok"
