"""Prompt templates for meeting summary generation."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

STYLES_DIR = Path(__file__).parent.parent.parent / "config" / "summary_styles"

DEFAULT_SUMMARY_PROMPT = """你是一个会议纪要助手。根据以下转写文本，生成结构化的会议纪要。

转写文本：
{transcript}

请按以下格式输出：

# 会议纪要

## 主题
（一句话概括会议主题）

## 要点
- （列出 3-8 个关键讨论要点）

## 待办事项
- [ ] （列出明确的行动项，如果有的话）

## 关键结论
- （列出主要结论或决策）

注意：
- 如果转写文本太短或没有实质内容，简单说明即可
- 用中文输出"""


def load_style(style_name: str) -> str:
    """Load a prompt template from a YAML style file.

    Returns the prompt string with {transcript} placeholder.
    Falls back to DEFAULT_SUMMARY_PROMPT if file not found.
    """
    style_path = STYLES_DIR / f"{style_name}.yaml"
    if not style_path.exists():
        logger.warning("Style file not found: %s, using default prompt", style_path)
        return DEFAULT_SUMMARY_PROMPT

    with open(style_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompt = data.get("prompt", "")
    if not prompt:
        logger.warning("Empty prompt in style file: %s", style_path)
        return DEFAULT_SUMMARY_PROMPT

    return prompt


def build_prompt(
    transcript_json: str,
    custom_prompt: str | None = None,
    style: str = "verbatim_summary",
) -> str:
    """Build the prompt for Claude API.

    Priority: custom_prompt > style template > default.
    """
    if custom_prompt:
        template = custom_prompt
    else:
        template = load_style(style)

    return template.replace("{transcript}", transcript_json)
