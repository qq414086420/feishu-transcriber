"""Shared logging setup for CLI tools."""

import logging
from pathlib import Path


def setup_tool_logging(tool_name: str, logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger(tool_name)
    if logger.handlers:
        return logger
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(logs_dir / f"{tool_name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
