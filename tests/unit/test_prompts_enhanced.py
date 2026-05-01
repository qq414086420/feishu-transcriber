"""Tests for enhanced prompt loading with YAML styles."""

from text_summarize.prompts import load_style, build_prompt, DEFAULT_SUMMARY_PROMPT


class TestLoadStyle:
    def test_load_existing_style(self):
        result = load_style("verbatim_summary")
        assert "{transcript}" in result
        assert "逐字稿" in result

    def test_load_missing_style_returns_default(self):
        result = load_style("nonexistent_style")
        assert result == DEFAULT_SUMMARY_PROMPT


class TestBuildPromptEnhanced:
    def test_build_with_style(self):
        transcript = '{"text": "hello", "segments": []}'
        result = build_prompt(transcript, style="verbatim_summary")
        assert "hello" in result
        assert "逐字稿" in result

    def test_build_with_missing_style_uses_default(self):
        transcript = '{"text": "test", "segments": []}'
        result = build_prompt(transcript, style="nonexistent")
        assert "test" in result
        assert result == DEFAULT_SUMMARY_PROMPT.replace("{transcript}", transcript)

    def test_build_with_custom_prompt_overrides_style(self):
        transcript = '{"text": "test", "segments": []}'
        custom = "Summarize: {transcript}"
        result = build_prompt(transcript, custom_prompt=custom)
        assert result == "Summarize: " + transcript
