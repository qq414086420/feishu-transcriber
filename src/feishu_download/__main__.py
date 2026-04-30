"""CLI entry point for the feishu-download tool."""

import argparse
import sys
from pathlib import Path

from shared.config import Config
from shared.exit_codes import EXIT_ARG_ERROR, EXIT_FAILURE, EXIT_OK
from shared.logging import setup_tool_logging

from feishu_download.client import download_from_feishu
from feishu_download.local import copy_local_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feishu-download",
        description="Download files from Feishu or copy local files to inbox",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file-path", type=Path, default=None, help="Local file path (offline mode)")
    mode.add_argument("--message-id", default=None, help="Feishu message ID (online mode)")
    parser.add_argument("--file-key", default=None, help="Feishu file key (required in online mode)")
    parser.add_argument("--type", choices=["audio", "video", "file"], default="file", help="Resource type (default: file)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = Config.from_env()
    config.ensure_dirs()
    setup_tool_logging("feishu_download", config.logs_dir)

    output_dir = args.output_dir or config.inbox_dir

    result_path: Path | None = None

    if args.file_path:
        result_path = copy_local_file(source=args.file_path, output_dir=output_dir)
    else:
        if not args.message_id or not args.file_key:
            print("Online mode requires both --message-id and --file-key", file=sys.stderr)
            sys.exit(EXIT_ARG_ERROR)
        result_path = download_from_feishu(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            message_id=args.message_id,
            file_key=args.file_key,
            file_type=args.type,
            output_dir=output_dir,
        )

    if result_path is None:
        sys.exit(EXIT_FAILURE)

    print(result_path)
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
