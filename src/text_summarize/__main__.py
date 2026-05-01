"""CLI entry point for the text-summarize tool."""

import argparse
import sys
from pathlib import Path

from shared.config import Config
from shared.exit_codes import EXIT_ARG_ERROR, EXIT_FAILURE, EXIT_OK
from shared.logging import setup_tool_logging

from text_summarize.summarizer import summarize


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text-summarize",
        description="Summarize meeting transcripts using Claude API",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input transcript JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: config.summaries_dir / <stem>_summary.md)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt template with {transcript} placeholder",
    )
    parser.add_argument(
        "--style",
        default="verbatim_summary",
        help="Summary style template name (default: verbatim_summary)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = Config.from_env()
    config.ensure_dirs()
    setup_tool_logging("text_summarize", config.logs_dir)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(EXIT_ARG_ERROR)

    output_path = args.output or config.summaries_dir / f"{input_path.stem}_summary.md"

    result = summarize(
        transcript_path=input_path,
        output_path=output_path,
        api_key=config.anthropic_api_key,
        custom_prompt=args.prompt,
        style=args.style,
    )

    if result is None:
        sys.exit(EXIT_FAILURE)

    print(str(output_path))
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
