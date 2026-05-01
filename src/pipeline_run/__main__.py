"""CLI entry point for pipeline-run: orchestrates all 4 tools."""

import argparse
import json
import sys
from pathlib import Path

from shared.config import Config
from shared.exit_codes import EXIT_ARG_ERROR, EXIT_FAILURE, EXIT_OK, exit_arg_error, exit_failure
from shared.logging import setup_tool_logging

from pipeline_run.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for pipeline-run."""
    parser = argparse.ArgumentParser(
        prog="pipeline_run",
        description="Orchestrate feishu-download, media-to-audio, audio-transcribe, text-summarize.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--message-id",
        help="Feishu message ID for online mode (requires --file-key)",
    )
    mode.add_argument(
        "--file-path",
        help="Local file path for offline mode",
    )

    parser.add_argument(
        "--file-key",
        help="Feishu file key (required with --message-id)",
    )
    parser.add_argument(
        "--type",
        choices=["audio", "video", "file"],
        default="file",
        help="Media type (default: file)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate directories after success",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip the text-summarize step",
    )
    parser.add_argument(
        "--style",
        default="verbatim_summary",
        help="Summary style template name (default: verbatim_summary)",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    config = Config.from_env()
    logger = setup_tool_logging("pipeline_run", config.logs_dir)

    # Validate arguments
    if args.message_id and not args.file_key:
        exit_arg_error("--file-key is required when using --message-id")

    python_cmd = sys.executable

    # Online mode: download from Feishu first
    if args.message_id:
        try:
            from feishu_download.client import download_from_feishu

            logger.info("Downloading from Feishu: message_id=%s file_key=%s", args.message_id, args.file_key)
            file_path = download_from_feishu(
                message_id=args.message_id,
                file_key=args.file_key,
                file_type=args.type,
                config=config,
            )
            if file_path is None:
                exit_failure("Failed to download file from Feishu")
        except Exception as exc:
            exit_failure(f"Feishu download error: {exc}")
    else:
        file_path = args.file_path
        if not Path(file_path).exists():
            exit_arg_error(f"File not found: {file_path}")

    # Run the pipeline
    work_dir = str(config.data_dir)
    logger.info("Starting pipeline: file=%s work_dir=%s", file_path, work_dir)

    result = run_pipeline(
        file_path=str(file_path),
        work_dir=work_dir,
        python_cmd=python_cmd,
        no_summarize=args.no_summarize,
        cleanup=args.cleanup,
        style=args.style,
    )

    print(json.dumps(result, ensure_ascii=False))

    if result["status"] == "ok":
        logger.info("Pipeline completed: %s", result)
        sys.exit(EXIT_OK)
    else:
        logger.error("Pipeline failed at step %s: %s", result.get("step"), result.get("error"))
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()
