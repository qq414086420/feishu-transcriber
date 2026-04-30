"""CLI entry point for the media-to-audio tool."""

import argparse
import sys
from pathlib import Path

from shared.config import Config
from shared.exit_codes import EXIT_ARG_ERROR, EXIT_FAILURE, EXIT_OK
from shared.logging import setup_tool_logging

from media_to_audio.converter import convert_to_wav


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media-to-audio",
        description="Convert audio/video files to 16kHz mono WAV for transcription",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input audio/video file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the WAV file (default: config.audio_dir)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = Config.from_env()
    config.ensure_dirs()
    setup_tool_logging("media_to_audio", config.logs_dir)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(EXIT_ARG_ERROR)

    output_dir = args.output or config.audio_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = convert_to_wav(input_path, output_dir)

    if result_path is None:
        sys.exit(EXIT_FAILURE)

    print(result_path)
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
