"""Prompt templates for meeting summary generation."""

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


def build_prompt(transcript_json: str, custom_prompt: str | None = None) -> str:
    """Build the prompt for Claude API by inserting transcript text into the template."""
    template = custom_prompt or DEFAULT_SUMMARY_PROMPT
    return template.replace("{transcript}", transcript_json)
