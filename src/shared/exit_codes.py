"""Unified exit codes for all CLI tools."""

import sys

EXIT_OK = 0
EXIT_ARG_ERROR = 1
EXIT_FAILURE = 2


def exit_ok() -> None:
    sys.exit(EXIT_OK)


def exit_arg_error(msg: str) -> None:
    print(f"Argument error: {msg}", file=sys.stderr)
    sys.exit(EXIT_ARG_ERROR)


def exit_failure(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(EXIT_FAILURE)
