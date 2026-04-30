"""Convert audio/video files to 16kHz mono WAV for transcription."""

import logging
import subprocess
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# Target format constants
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
FFMPEG_TIMEOUT = 600


def is_target_wav(path: Path) -> bool:
    """Check if a file is already a 16kHz mono WAV.

    Returns False for non-existent files, non-WAV extensions,
    or WAV files that don't match the target format.
    """
    if not path.exists():
        return False
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "r") as w:
            return (
                w.getframerate() == TARGET_SAMPLE_RATE
                and w.getnchannels() == TARGET_CHANNELS
                and w.getsampwidth() == TARGET_SAMPLE_WIDTH
            )
    except (wave.Error, OSError):
        return False


def convert_to_wav(input_path: Path, output_dir: Path) -> Path | None:
    """Convert an audio/video file to 16kHz mono WAV.

    If the input is already in target format, creates a symlink
    in output_dir instead of converting. Otherwise runs ffmpeg.

    Returns the output Path on success, or None on failure.
    """
    output_path = output_dir / (input_path.stem + ".wav")

    if is_target_wav(input_path):
        try:
            # Create a relative symlink for portability
            target = input_path.resolve()
            if output_path.exists() or output_path.is_symlink():
                output_path.unlink()
            output_path.symlink_to(target)
            logger.info("Symlinked %s -> %s (already target format)", output_path, target)
            return output_path
        except OSError as exc:
            logger.error("Failed to create symlink: %s", exc)
            return None

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", str(TARGET_CHANNELS),
                "-c:a", "pcm_s16le",
                str(output_path),
            ],
            check=True,
            timeout=FFMPEG_TIMEOUT,
        )
        logger.info("Converted %s -> %s", input_path, output_path)
        return output_path
    except FileNotFoundError:
        logger.error("ffmpeg not found — is it installed?")
        return None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out after %ds for %s", FFMPEG_TIMEOUT, input_path)
        return None
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed with return code %d: %s", exc.returncode, exc)
        return None
