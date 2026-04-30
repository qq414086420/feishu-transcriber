"""Copy local files into the inbox directory (offline mode)."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("feishu_download")


def copy_local_file(source: Path, output_dir: Path) -> Path | None:
    """Copy *source* into *output_dir* preserving the file name.

    Returns the destination path on success, or ``None`` if *source* does not
    exist.
    """
    if not source.exists():
        logger.warning("Source file does not exist: %s", source)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = shutil.copy2(source, output_dir / source.name)
    logger.info("Copied %s -> %s", source, dest)
    return Path(dest)
