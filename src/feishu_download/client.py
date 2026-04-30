"""Download files from Feishu via the lark_oapi SDK.

The lark_oapi import is deferred to function level because the SDK may
hang on import in certain environments.  Tests mock the ``_get_lark``
and ``_get_request_cls`` helpers instead.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("feishu_download")

_EXTENSION_MAP: dict[str, str] = {
    "audio": ".aud",
    "video": ".vid",
    "file": ".bin",
    "image": ".img",
}


def _get_lark() -> Any:
    """Return the ``lark_oapi`` module (imported lazily)."""
    import lark_oapi as lark  # noqa: I001

    return lark


def _get_request_cls() -> Any:
    """Return ``DownloadMessageResourceRequest`` (imported lazily)."""
    from lark_oapi.api.im.v1 import DownloadMessageResourceRequest  # noqa: I001

    return DownloadMessageResourceRequest


def download_from_feishu(
    app_id: str,
    app_secret: str,
    message_id: str,
    file_key: str,
    file_type: str,
    output_dir: Path,
) -> Path | None:
    """Download a file resource from Feishu and write it to *output_dir*.

    Returns the path to the written file on success, or ``None`` on failure.
    """
    if not app_id or not app_secret:
        logger.warning("Missing Feishu app_id or app_secret; skipping download")
        return None

    lark = _get_lark()
    DownloadMessageResourceRequest = _get_request_cls()

    client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()

    request = (
        DownloadMessageResourceRequest.builder()
        .message_id(message_id)
        .file_key(file_key)
        .type(file_type)
        .build()
    )

    response = client.im.v1.message_resource.download(request)

    if not response.success():
        logger.error(
            "Feishu download failed: code=%s msg=%s",
            response.code,
            response.msg,
        )
        return None

    ext = _EXTENSION_MAP.get(file_type, ".bin")
    dest = output_dir / f"{file_key}{ext}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        for chunk in response.file.iter_bytes():
            f.write(chunk)

    logger.info("Downloaded %s -> %s", file_key, dest)
    return dest
