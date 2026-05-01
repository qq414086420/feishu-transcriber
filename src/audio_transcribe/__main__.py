"""CLI entry point for the audio-transcribe tool."""

import argparse
import sys
from pathlib import Path

from shared.config import Config
from shared.exit_codes import EXIT_ARG_ERROR, EXIT_FAILURE, EXIT_OK
from shared.logging import setup_tool_logging

from audio_transcribe.transcriber import transcribe_with_speakers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-transcribe",
        description="Transcribe audio using SenseVoice via FunASR",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: config.transcripts_dir / <stem>.json)",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="Language code for transcription (default: zh)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model inference: cpu or cuda (default: cpu)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = Config.from_env()
    config.ensure_dirs()
    setup_tool_logging("audio_transcribe", config.logs_dir)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(EXIT_ARG_ERROR)

    output_path = args.output or config.transcripts_dir / f"{input_path.stem}.json"

    result = transcribe_with_speakers(
        input_path=input_path,
        output_path=output_path,
        language=args.language,
        device=args.device,
    )

    if result is None:
        sys.exit(EXIT_FAILURE)

    print(str(output_path))
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
