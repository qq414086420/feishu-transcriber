"""Pipeline orchestrator: chains media-to-audio, audio-transcribe, text-summarize."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single pipeline step."""

    success: bool
    output_path: str = ""
    error: str = ""


def _last_line(text: str) -> str:
    """Return the last non-empty line from multi-line stdout."""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass(frozen=True)
class PipelineStep:
    """A single step in the processing pipeline."""

    name: str
    cmd: list[str]

    def execute(self) -> StepResult:
        """Run the step as a subprocess and return the result."""
        try:
            result = subprocess.run(
                self.cmd,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return StepResult(success=False, error=f"{self.name} timed out")
        except FileNotFoundError:
            return StepResult(success=False, error=f"{self.name} command not found")

        if result.returncode == 0:
            output_path = _last_line(result.stdout)
            return StepResult(success=True, output_path=output_path)

        error_msg = result.stderr.strip() or f"{self.name} exited with code {result.returncode}"
        return StepResult(success=False, error=error_msg)


def run_pipeline(
    file_path: str,
    work_dir: str,
    python_cmd: str = "python",
    no_summarize: bool = False,
    cleanup: bool = False,
    style: str = "verbatim_summary",
) -> dict:
    """Run the full transcription pipeline.

    Returns a dict with status and paths or error details.
    """
    work = Path(work_dir)
    inbox_dir = work / "inbox"
    audio_dir = work / "audio"
    transcripts_dir = work / "transcripts"
    summaries_dir = work / "summaries"

    # Ensure subdirectories exist
    for d in [inbox_dir, audio_dir, transcripts_dir, summaries_dir]:
        d.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)
    inbox_path = inbox_dir / src.name

    # Copy file to inbox if not already there
    if src.resolve() != inbox_path.resolve():
        shutil.copy2(str(src), str(inbox_path))

    # Step 1: media-to-audio
    step1 = PipelineStep(
        "media_to_audio",
        [python_cmd, "-m", "media_to_audio", "--input", str(inbox_path)],
    )
    r1 = step1.execute()
    if not r1.success:
        return {"status": "error", "step": "media_to_audio", "error": r1.error}

    audio_path = r1.output_path

    # Step 2: audio-transcribe
    step2 = PipelineStep(
        "audio_transcribe",
        [python_cmd, "-m", "audio_transcribe", "--input", audio_path],
    )
    r2 = step2.execute()
    if not r2.success:
        return {"status": "error", "step": "audio_transcribe", "error": r2.error}

    transcript_path = r2.output_path

    # Step 3: text-summarize (optional)
    if not no_summarize:
        step3 = PipelineStep(
            "text_summarize",
            [python_cmd, "-m", "text_summarize", "--input", transcript_path, "--style", style],
        )
        r3 = step3.execute()
        if not r3.success:
            return {"status": "error", "step": "text_summarize", "error": r3.error}

        summary_path = r3.output_path
    else:
        summary_path = None

    # Cleanup intermediate directories if requested
    if cleanup:
        shutil.rmtree(str(inbox_dir), ignore_errors=True)
        shutil.rmtree(str(audio_dir), ignore_errors=True)
        shutil.rmtree(str(transcripts_dir), ignore_errors=True)

    result: dict = {
        "status": "ok",
        "transcript_path": transcript_path,
    }
    if summary_path is not None:
        result["summary_path"] = summary_path

    return result
