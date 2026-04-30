"""Unit tests for text_summarize tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests for build_prompt
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    """Tests for the build_prompt helper."""

    def test_default_prompt_contains_transcript(self) -> None:
        """Default prompt includes the transcript text."""
        from text_summarize.prompts import build_prompt

        result = build_prompt("这是会议内容")
        assert "这是会议内容" in result
        assert "转写文本" in result or "会议纪要" in result

    def test_custom_prompt_overrides_default(self) -> None:
        """Custom template with {transcript} placeholder is used."""
        from text_summarize.prompts import build_prompt

        custom = "Summarize: {transcript}"
        result = build_prompt("hello meeting", custom_prompt=custom)
        assert result == "Summarize: hello meeting"
        assert "会议纪要" not in result


# ---------------------------------------------------------------------------
# Tests for summarize
# ---------------------------------------------------------------------------
class TestSummarize:
    """Tests for the summarize function."""

    def test_summarize_writes_markdown(self, tmp_path: Path) -> None:
        """Successful API call writes a .md file with the response text."""
        from text_summarize.summarizer import summarize

        transcript_path = tmp_path / "meeting.json"
        transcript_path.write_text('{"text": "讨论了项目进度"}', encoding="utf-8")

        output_path = tmp_path / "output" / "meeting_summary.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="# 会议纪要\n\n## 主题\n项目进度讨论")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("text_summarize.summarizer.anthropic.Anthropic", return_value=mock_client):
            result = summarize(
                transcript_path=transcript_path,
                output_path=output_path,
                api_key="sk-test",
            )

        assert result == output_path
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "会议纪要" in content

    def test_summarize_api_failure_returns_none(self, tmp_path: Path) -> None:
        """When the API raises an exception, returns None."""
        from text_summarize.summarizer import summarize

        transcript_path = tmp_path / "meeting.json"
        transcript_path.write_text('{"text": "讨论了项目进度"}', encoding="utf-8")

        output_path = tmp_path / "output" / "meeting_summary.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("text_summarize.summarizer.anthropic.Anthropic", side_effect=Exception("API error")):
            result = summarize(
                transcript_path=transcript_path,
                output_path=output_path,
                api_key="sk-test",
            )

        assert result is None

    def test_summarize_missing_transcript_returns_none(self, tmp_path: Path) -> None:
        """Non-existent transcript file returns None."""
        from text_summarize.summarizer import summarize

        missing = tmp_path / "does_not_exist.json"
        output_path = tmp_path / "output" / "summary.md"

        result = summarize(
            transcript_path=missing,
            output_path=output_path,
            api_key="sk-test",
        )

        assert result is None
