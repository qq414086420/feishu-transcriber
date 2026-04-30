"""Shared test fixtures."""

from pathlib import Path

import pytest

from shared.config import Config


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """Config with temporary directories for isolated testing."""
    return Config(
        data_dir=tmp_path / "data",
        inbox_dir=tmp_path / "data" / "inbox",
        audio_dir=tmp_path / "data" / "audio",
        transcripts_dir=tmp_path / "data" / "transcripts",
        summaries_dir=tmp_path / "data" / "summaries",
        logs_dir=tmp_path / "logs",
        feishu_app_id="cli_test",
        feishu_app_secret="secret_test",
        anthropic_api_key="sk-test",
    )
