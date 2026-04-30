"""Summarize meeting transcripts using Claude API."""

import logging
from pathlib import Path

import anthropic

from text_summarize.prompts import build_prompt

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3


def summarize(
    transcript_path: Path,
    output_path: Path,
    api_key: str,
    custom_prompt: str | None = None,
) -> Path | None:
    """Summarize a transcript JSON file using Claude API.

    Returns the output Path on success, None on failure.
    """
    if not transcript_path.exists():
        logger.error("Transcript file not found: %s", transcript_path)
        return None

    try:
        transcript_text = transcript_path.read_text(encoding="utf-8")
        prompt = build_prompt(transcript_text, custom_prompt)

        client = anthropic.Anthropic(api_key=api_key, max_retries=MAX_RETRIES)
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        summary_text = message.content[0].text

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary_text, encoding="utf-8")

        logger.info("Summary saved to %s", output_path)
        return output_path

    except Exception:
        logger.exception("Summarization failed for %s", transcript_path)
        return None
