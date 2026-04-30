"""Unit tests for feishu_download tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_request_builder() -> MagicMock:
    """Build a mock DownloadMessageResourceRequest builder chain."""
    builder = MagicMock(name="RequestBuilder")
    builder.message_id.return_value = builder
    builder.file_key.return_value = builder
    builder.type.return_value = builder
    builder.build.return_value = MagicMock(name="BuiltRequest")
    return builder


def _setup_download_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create and return (mock_lark, mock_req_cls, mock_client)."""
    mock_client = MagicMock()
    mock_lark = MagicMock()
    mock_lark.Client.builder.return_value.app_id.return_value.app_secret.return_value.build.return_value = mock_client
    mock_req_builder = _make_mock_request_builder()
    mock_req_cls = MagicMock()
    mock_req_cls.builder.return_value = mock_req_builder
    return mock_lark, mock_req_cls, mock_client


# ---------------------------------------------------------------------------
# Tests for client.download_from_feishu
# ---------------------------------------------------------------------------
class TestDownloadFromFeishu:
    """Tests for online download via Feishu API."""

    def test_download_writes_to_inbox(self, tmp_path: Path) -> None:
        """Successful download writes file to output directory."""
        from feishu_download.client import download_from_feishu

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        mock_lark, mock_req_cls, mock_client = _setup_download_mocks()

        mock_response = MagicMock()
        mock_response.success.return_value = True
        mock_response.file.iter_bytes.return_value = [b"fake audio data chunk1", b"fake audio data chunk2"]
        mock_client.im.v1.message_resource.download.return_value = mock_response

        with patch("feishu_download.client._get_lark", return_value=mock_lark), \
             patch("feishu_download.client._get_request_cls", return_value=mock_req_cls):
            result = download_from_feishu(
                app_id="cli_test",
                app_secret="secret_test",
                message_id="msg_001",
                file_key="file_abc",
                file_type="audio",
                output_dir=output_dir,
            )

        assert result is not None
        assert result.exists()
        assert result.name == "file_abc.aud"
        assert result.read_bytes() == b"fake audio data chunk1fake audio data chunk2"

    def test_download_failure_returns_none(self, tmp_path: Path) -> None:
        """Failed API response returns None without writing file."""
        from feishu_download.client import download_from_feishu

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        mock_lark, mock_req_cls, mock_client = _setup_download_mocks()

        mock_response = MagicMock()
        mock_response.success.return_value = False
        mock_client.im.v1.message_resource.download.return_value = mock_response

        with patch("feishu_download.client._get_lark", return_value=mock_lark), \
             patch("feishu_download.client._get_request_cls", return_value=mock_req_cls):
            result = download_from_feishu(
                app_id="cli_test",
                app_secret="secret_test",
                message_id="msg_001",
                file_key="file_abc",
                file_type="file",
                output_dir=output_dir,
            )

        assert result is None

    def test_download_missing_app_id_returns_none(self, tmp_path: Path) -> None:
        """Empty app_id returns None without calling the API."""
        from feishu_download.client import download_from_feishu

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        mock_lark, mock_req_cls, _ = _setup_download_mocks()

        with patch("feishu_download.client._get_lark", return_value=mock_lark) as mock_get_lark, \
             patch("feishu_download.client._get_request_cls", return_value=mock_req_cls) as mock_get_req:
            result = download_from_feishu(
                app_id="",
                app_secret="secret_test",
                message_id="msg_001",
                file_key="file_abc",
                file_type="file",
                output_dir=output_dir,
            )

        # _get_lark should not be called when credentials are empty
        mock_get_lark.assert_not_called()
        assert result is None

    def test_download_missing_app_secret_returns_none(self, tmp_path: Path) -> None:
        """Empty app_secret returns None without calling the API."""
        from feishu_download.client import download_from_feishu

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        mock_lark, mock_req_cls, _ = _setup_download_mocks()

        with patch("feishu_download.client._get_lark", return_value=mock_lark) as mock_get_lark, \
             patch("feishu_download.client._get_request_cls", return_value=mock_req_cls):
            result = download_from_feishu(
                app_id="cli_test",
                app_secret="",
                message_id="msg_001",
                file_key="file_abc",
                file_type="file",
                output_dir=output_dir,
            )

        mock_get_lark.assert_not_called()
        assert result is None

    def test_download_video_uses_vid_extension(self, tmp_path: Path) -> None:
        """Video file type uses .vid extension."""
        from feishu_download.client import download_from_feishu

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        mock_lark, mock_req_cls, mock_client = _setup_download_mocks()

        mock_response = MagicMock()
        mock_response.success.return_value = True
        mock_response.file.iter_bytes.return_value = [b"video data"]
        mock_client.im.v1.message_resource.download.return_value = mock_response

        with patch("feishu_download.client._get_lark", return_value=mock_lark), \
             patch("feishu_download.client._get_request_cls", return_value=mock_req_cls):
            result = download_from_feishu(
                app_id="cli_test",
                app_secret="secret_test",
                message_id="msg_002",
                file_key="file_vid",
                file_type="video",
                output_dir=output_dir,
            )

        assert result is not None
        assert result.name == "file_vid.vid"

    def test_download_image_uses_img_extension(self, tmp_path: Path) -> None:
        """Image file type uses .img extension."""
        from feishu_download.client import download_from_feishu

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        mock_lark, mock_req_cls, mock_client = _setup_download_mocks()

        mock_response = MagicMock()
        mock_response.success.return_value = True
        mock_response.file.iter_bytes.return_value = [b"image data"]
        mock_client.im.v1.message_resource.download.return_value = mock_response

        with patch("feishu_download.client._get_lark", return_value=mock_lark), \
             patch("feishu_download.client._get_request_cls", return_value=mock_req_cls):
            result = download_from_feishu(
                app_id="cli_test",
                app_secret="secret_test",
                message_id="msg_003",
                file_key="file_img",
                file_type="image",
                output_dir=output_dir,
            )

        assert result is not None
        assert result.name == "file_img.img"


# ---------------------------------------------------------------------------
# Tests for local.copy_local_file
# ---------------------------------------------------------------------------
class TestCopyLocalFile:
    """Tests for offline local file copy."""

    def test_copy_existing_file_to_inbox(self, tmp_path: Path) -> None:
        """Existing source file is copied to output directory."""
        from feishu_download.local import copy_local_file

        source = tmp_path / "source"
        source.mkdir()
        src_file = source / "meeting_recording.aud"
        src_file.write_bytes(b"recording data")

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        result = copy_local_file(source=src_file, output_dir=output_dir)

        assert result is not None
        assert result.exists()
        assert result.name == "meeting_recording.aud"
        assert result.read_bytes() == b"recording data"

    def test_copy_nonexistent_file_returns_none(self, tmp_path: Path) -> None:
        """Missing source file returns None."""
        from feishu_download.local import copy_local_file

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        result = copy_local_file(
            source=tmp_path / "nonexistent.m4a",
            output_dir=output_dir,
        )

        assert result is None

    def test_copy_preserves_extension(self, tmp_path: Path) -> None:
        """File extension is preserved in the copy."""
        from feishu_download.local import copy_local_file

        source = tmp_path / "source"
        source.mkdir()
        src_file = source / "voice_note.m4a"
        src_file.write_bytes(b"m4a audio data")

        output_dir = tmp_path / "inbox"
        output_dir.mkdir()

        result = copy_local_file(source=src_file, output_dir=output_dir)

        assert result is not None
        assert result.suffix == ".m4a"
