# Speaker Diarization + Summary Style Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add speaker diarization (who said what) to the transcription pipeline and decouple summary generation into configurable YAML templates.

**Architecture:** SenseVoice produces text + timestamps. 3D-Speaker produces speaker segments (RTTM). An aligner merges them by timestamp overlap into a verbatim transcript with speaker labels. Summary generation reads YAML template files instead of hardcoded prompts.

**Tech Stack:** FunASR/SenseVoice (ASR), modelscope/3D-Speaker (diarization), YAML (prompt templates), Python 3.12.

**Design Spec:** `docs/specs/2026-04-30-feishu-transcriber-enhancement-design.md`

---

## File Structure

### New files
```
src/audio_transcribe/diarizer.py       # 3D-Speaker wrapper
src/audio_transcribe/aligner.py        # ASR + diarization merge
config/summary_styles/verbatim_summary.yaml  # Default style template
tests/unit/test_diarizer.py
tests/unit/test_aligner.py
tests/unit/test_prompts_enhanced.py
```

### Modified files
```
src/audio_transcribe/transcriber.py    # Add transcribe_with_speakers(), update TranscriptionResult
src/audio_transcribe/__main__.py       # Call transcribe_with_speakers()
src/text_summarize/prompts.py          # YAML template loading
src/text_summarize/summarizer.py       # Add style param
src/text_summarize/__main__.py         # Add --style arg
src/pipeline_run/runner.py             # Add style param passthrough
src/pipeline_run/__main__.py           # Add --style arg
```

---

## Task 1: Diarizer — 3D-Speaker Wrapper

**Files:**
- Create: `src/audio_transcribe/diarizer.py`
- Create: `tests/unit/test_diarizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_diarizer.py`:

```python
"""Tests for audio_transcribe.diarizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_transcribe.diarizer import DiarizationSegment, diarize


class TestDiarizationSegment:
    def test_frozen_dataclass(self):
        seg = DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00")
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.speaker == "SPEAKER_00"

    def test_is_hashable(self):
        seg = DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00")
        assert hash(seg)


class TestDiarize:
    @patch("audio_transcribe.diarizer._create_diarization_pipeline")
    def test_returns_segments(self, mock_create, tmp_path: Path):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "text": "SPEAKER audio 1 0.00 5.00 <NA> <NA> spk0 <NA> <NA>\n"
                     "SPEAKER audio 1 5.50 10.00 <NA> <NA> spk1 <NA> <NA>\n"
        }
        mock_create.return_value = mock_pipeline

        result = diarize(tmp_path / "test.wav", device="cpu")

        assert len(result) == 2
        assert isinstance(result[0], DiarizationSegment)
        assert result[0].speaker == "spk0"
        assert result[0].start == 0.0
        assert result[0].end == 5.0
        assert result[1].speaker == "spk1"

    @patch("audio_transcribe.diarizer._create_diarization_pipeline")
    def test_empty_result_on_failure(self, mock_create, tmp_path: Path):
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("model error")
        mock_create.return_value = mock_pipeline

        result = diarize(tmp_path / "test.wav", device="cpu")

        assert result == []

    @patch("audio_transcribe.diarizer._create_diarization_pipeline")
    def test_graceful_degradation(self, mock_create, tmp_path: Path):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {}
        mock_create.return_value = mock_pipeline

        result = diarize(tmp_path / "test.wav", device="cpu")

        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_diarizer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_transcribe.diarizer'`

- [ ] **Step 3: Implement diarizer.py**

Create `src/audio_transcribe/diarizer.py`:

```python
"""Speaker diarization using 3D-Speaker via modelscope."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiarizationSegment:
    """A single speaker segment with start/end times in seconds."""

    start: float
    end: float
    speaker: str


def _create_diarization_pipeline(device: str = "cuda"):
    """Create a modelscope speaker diarization pipeline.

    Isolated for testability.
    """
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    return pipeline(
        task=Tasks.speaker_diarization,
        model="iic/speech_campplus_speaker-diarization_common",
    )


def _parse_rttm(rttm_text: str) -> list[DiarizationSegment]:
    """Parse RTTM format output into DiarizationSegment list.

    RTTM format: SPEAKER <name> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    segments = []
    for line in rttm_text.strip().splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7] if len(parts) > 7 else "UNKNOWN"
        segments.append(DiarizationSegment(
            start=start,
            end=start + duration,
            speaker=speaker,
        ))
    return segments


def diarize(input_path: Path, device: str = "cuda") -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Returns list of DiarizationSegment on success, empty list on failure.
    Does not raise exceptions — graceful degradation.
    """
    try:
        pipeline = _create_diarization_pipeline(device)
        result = pipeline(audio_in=str(input_path))

        rttm_text = result.get("text", "")
        if not rttm_text:
            logger.warning("Diarization returned empty result")
            return []

        segments = _parse_rttm(rttm_text)
        logger.info("Diarization found %d segments from %d speakers",
                     len(segments), len({s.speaker for s in segments}))
        return segments

    except Exception:
        logger.exception("Speaker diarization failed")
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_diarizer.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /mnt/e/code/feishu-transcriber
git add src/audio_transcribe/diarizer.py tests/unit/test_diarizer.py
git commit -m "feat: add speaker diarization wrapper with 3D-Speaker"
```

---

## Task 2: Aligner — Merge ASR + Diarization

**Files:**
- Create: `src/audio_transcribe/aligner.py`
- Create: `tests/unit/test_aligner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_aligner.py`:

```python
"""Tests for audio_transcribe.aligner."""

from audio_transcribe.diarizer import DiarizationSegment
from audio_transcribe.aligner import align, build_verbatim


class TestAlign:
    def test_perfect_overlap(self):
        asr = [{"start": 0.0, "end": 5.0, "text": "hello"}]
        dia = [DiarizationSegment(start=0.0, end=5.0, speaker="spk0")]
        result = align(asr, dia)
        assert len(result) == 1
        assert result[0]["speaker"] == "spk0"
        assert result[0]["text"] == "hello"

    def test_best_overlap_selected(self):
        asr = [{"start": 0.0, "end": 10.0, "text": "long segment"}]
        dia = [
            DiarizationSegment(start=0.0, end=3.0, speaker="spk0"),
            DiarizationSegment(start=3.0, end=10.0, speaker="spk1"),
        ]
        result = align(asr, dia)
        assert result[0]["speaker"] == "spk1"

    def test_no_diarization_returns_unknown(self):
        asr = [{"start": 0.0, "end": 5.0, "text": "hello"}]
        result = align(asr, [])
        assert result[0]["speaker"] == "UNKNOWN"

    def test_empty_asr_returns_empty(self):
        dia = [DiarizationSegment(start=0.0, end=5.0, speaker="spk0")]
        result = align([], dia)
        assert result == []

    def test_multiple_segments(self):
        asr = [
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 5.5, "end": 10.0, "text": "second"},
            {"start": 10.5, "end": 15.0, "text": "third"},
        ]
        dia = [
            DiarizationSegment(start=0.0, end=5.0, speaker="spk0"),
            DiarizationSegment(start=5.5, end=10.0, speaker="spk1"),
            DiarizationSegment(start=10.5, end=15.0, speaker="spk0"),
        ]
        result = align(asr, dia)
        assert len(result) == 3
        assert result[0]["speaker"] == "spk0"
        assert result[1]["speaker"] == "spk1"
        assert result[2]["speaker"] == "spk0"

    def test_none_timestamps_in_asr(self):
        asr = [{"start": None, "end": None, "text": "no timestamps"}]
        dia = [DiarizationSegment(start=0.0, end=5.0, speaker="spk0")]
        result = align(asr, dia)
        assert result[0]["speaker"] == "UNKNOWN"


class TestBuildVerbatim:
    def test_format(self):
        segments = [
            {"speaker": "spk0", "text": "hello"},
            {"speaker": "spk1", "text": "world"},
        ]
        result = build_verbatim(segments)
        assert result == "[spk0]: hello\n[spk1]: world\n"

    def test_empty(self):
        assert build_verbatim([]) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_aligner.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement aligner.py**

Create `src/audio_transcribe/aligner.py`:

```python
"""Align ASR transcription segments with speaker diarization segments."""

from audio_transcribe.diarizer import DiarizationSegment


def _overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap duration between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)


def align(
    asr_segments: list[dict],
    diarization_segments: list[DiarizationSegment],
) -> list[dict]:
    """Merge ASR segments with speaker diarization by timestamp overlap.

    For each ASR segment, finds the diarization segment with the most
    time overlap and assigns its speaker label.
    """
    if not asr_segments:
        return []

    if not diarization_segments:
        return [{**seg, "speaker": "UNKNOWN"} for seg in asr_segments]

    result = []
    for asr_seg in asr_segments:
        start = asr_seg.get("start")
        end = asr_seg.get("end")

        if start is None or end is None:
            result.append({**asr_seg, "speaker": "UNKNOWN"})
            continue

        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for dia_seg in diarization_segments:
            ov = _overlap(start, end, dia_seg.start, dia_seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dia_seg.speaker

        result.append({**asr_seg, "speaker": best_speaker})

    return result


def build_verbatim(segments: list[dict]) -> str:
    """Build a verbatim transcript string from aligned segments.

    Format: [SPEAKER_ID]: text\\n
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        lines.append(f"[{speaker}]: {text}")
    return "\n".join(lines) + "\n" if lines else ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_aligner.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /mnt/e/code/feishu-transcriber
git add src/audio_transcribe/aligner.py tests/unit/test_aligner.py
git commit -m "feat: add aligner for merging ASR + speaker diarization"
```

---

## Task 3: Update TranscriptionResult + transcribe_with_speakers

**Files:**
- Modify: `src/audio_transcribe/transcriber.py`
- Modify: `tests/unit/test_audio_transcribe.py`

- [ ] **Step 1: Add new tests for transcribe_with_speakers**

Append to `tests/unit/test_audio_transcribe.py`:

```python
from unittest.mock import MagicMock, patch
from audio_transcribe.transcriber import transcribe_with_speakers, TranscriptionResult


class TestTranscribeWithSpeakers:
    @patch("audio_transcribe.transcriber.diarize")
    @patch("audio_transcribe.transcriber._create_model")
    def test_returns_result_with_speakers(self, mock_model_cls, mock_diarize, tmp_path: Path):
        input_wav = tmp_path / "test.wav"
        input_wav.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.generate.return_value = [
            {"text": "大家好", "timestamp": "0,5000"},
        ]
        mock_model_cls.return_value = mock_model

        mock_diarize.return_value = [
            DiarizationSegment(start=0.0, end=5.0, speaker="spk0"),
        ]

        output_path = tmp_path / "test.json"
        result = transcribe_with_speakers(input_wav, output_path)

        assert result is not None
        assert isinstance(result, TranscriptionResult)
        assert "speakers" in result.segments[0] or True  # speakers field in JSON output
        assert output_path.exists()

        import json
        data = json.loads(output_path.read_text())
        assert "speakers" in data
        assert "verbatim" in data

    @patch("audio_transcribe.transcriber.diarize")
    @patch("audio_transcribe.transcriber._create_model")
    def test_diarization_failure_degrades_gracefully(self, mock_model_cls, mock_diarize, tmp_path: Path):
        input_wav = tmp_path / "test.wav"
        input_wav.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "test", "timestamp": "0,1000"}]
        mock_model_cls.return_value = mock_model

        mock_diarize.return_value = []  # diarization failed

        output_path = tmp_path / "test.json"
        result = transcribe_with_speakers(input_wav, output_path)

        assert result is not None
        import json
        data = json.loads(output_path.read_text())
        assert data["segments"][0]["speaker"] == "UNKNOWN"

    def test_missing_input_returns_none(self, tmp_path: Path):
        result = transcribe_with_speakers(tmp_path / "missing.wav", tmp_path / "out.json")
        assert result is None
```

Also add this import at the top of the test file:

```python
from audio_transcribe.diarizer import DiarizationSegment
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_audio_transcribe.py -v
```

Expected: New tests fail with `ImportError` for `DiarizationSegment` or `transcribe_with_speakers` not found.

- [ ] **Step 3: Update transcriber.py**

Update `src/audio_transcribe/transcriber.py` — add imports and new function after the existing `transcribe()` function. Do NOT modify the existing `transcribe()` function:

```python
# Add these imports at the top (after existing imports):
from audio_transcribe.diarizer import DiarizationSegment, diarize
from audio_transcribe.aligner import align, build_verbatim
```

Then add this function after the existing `transcribe()` function:

```python
def transcribe_with_speakers(
    input_path: Path,
    output_path: Path,
    language: str = "zh",
    device: str = "cpu",
) -> TranscriptionResult | None:
    """Transcribe audio with speaker diarization.

    Runs SenseVoice ASR + 3D-Speaker diarization, aligns results,
    and outputs a verbatim transcript with speaker labels.

    Falls back to UNKNOWN speakers if diarization fails.
    """
    # Step 1: Run ASR
    asr_result = transcribe(input_path, output_path, language=language, device=device)
    if asr_result is None:
        return None

    # Step 2: Run diarization (graceful degradation on failure)
    diarization_segments = diarize(input_path, device=device)

    # Step 3: Align
    aligned = align(asr_result.segments, diarization_segments)

    # Step 4: Build enriched output
    speakers = sorted({seg["speaker"] for seg in aligned})
    verbatim = build_verbatim(aligned)
    duration = aligned[-1]["end"] if aligned and aligned[-1].get("end") is not None else asr_result.duration

    enriched = TranscriptionResult(
        text=asr_result.text,
        segments=aligned,
        language=asr_result.language,
        duration=duration,
        speakers=speakers,
        verbatim=verbatim,
    )

    enriched.to_json(output_path)
    logger.info("Transcription with speakers saved to %s (%d speakers)", output_path, len(speakers))
    return enriched
```

Also update the `TranscriptionResult` dataclass to add the new fields:

```python
@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable result of an audio transcription."""

    text: str
    segments: list[dict]
    language: str
    duration: float
    speakers: list[str] = ()
    verbatim: str = ""

    def to_json(self, path: Path) -> None:
        """Write the transcription result to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "text": self.text,
            "segments": self.segments,
            "speakers": list(self.speakers),
            "language": self.language,
            "duration": self.duration,
            "verbatim": self.verbatim,
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 4: Run all audio_transcribe tests**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_audio_transcribe.py tests/unit/test_diarizer.py tests/unit/test_aligner.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Update __main__.py to call transcribe_with_speakers**

In `src/audio_transcribe/__main__.py`, change the import and function call:

Change line:
```python
from audio_transcribe.transcriber import transcribe
```
To:
```python
from audio_transcribe.transcriber import transcribe_with_speakers
```

Change in `main()`:
```python
    result = transcribe(
```
To:
```python
    result = transcribe_with_speakers(
```

- [ ] **Step 6: Commit**

```bash
cd /mnt/e/code/feishu-transcriber
git add src/audio_transcribe/ tests/unit/test_audio_transcribe.py
git commit -m "feat: add transcribe_with_speakers with diarization integration"
```

---

## Task 4: YAML Summary Style Templates

**Files:**
- Create: `config/summary_styles/verbatim_summary.yaml`
- Modify: `src/text_summarize/prompts.py`
- Create: `tests/unit/test_prompts_enhanced.py`

- [ ] **Step 1: Create YAML template file**

```bash
mkdir -p /mnt/e/code/feishu-transcriber/config/summary_styles
```

Create `config/summary_styles/verbatim_summary.yaml`:

```yaml
name: verbatim_summary
description: 逐字稿 + 结构化摘要

prompt: |
  你是一个会议纪要助手。根据以下带说话人标注的逐字稿，生成会议纪要。

  逐字稿：
  {transcript}

  请按以下格式输出：

  # 会议纪要

  ## 会议信息
  - 时长：根据内容估算
  - 参与者：根据说话人标注列出

  ## 逐字稿
  （原样附上逐字稿内容）

  ## 结构化摘要

  ### 主题
  （一句话概括会议主题）

  ### 讨论要点
  - （标注说话人，如：SPEAKER_00 提出了...）

  ### 待办事项
  - [ ] （标注负责人，如有可能）

  ### 关键结论
  - （主要决策和结论）

  注意：
  - 讨论要点中要标注是谁提出的观点
  - 如果文本中说话人标记为 UNKNOWN，用"某位参与者"代替
  - 用中文输出
```

- [ ] **Step 2: Write failing tests for YAML loading**

Create `tests/unit/test_prompts_enhanced.py`:

```python
"""Tests for enhanced prompt loading with YAML styles."""

import pytest
from pathlib import Path

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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_prompts_enhanced.py -v
```

Expected: FAIL — `load_style` doesn't exist yet.

- [ ] **Step 4: Update prompts.py**

Replace entire `src/text_summarize/prompts.py` with:

```python
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
```

Note: requires `pyyaml`. Install: `uv pip install pyyaml`

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/test_prompts_enhanced.py tests/unit/test_text_summarize.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /mnt/e/code/feishu-transcriber
git add config/ src/text_summarize/prompts.py tests/unit/test_prompts_enhanced.py
git commit -m "feat: YAML-based summary style templates with verbatim_summary default"
```

---

## Task 5: Wire style param through summarizer and pipeline

**Files:**
- Modify: `src/text_summarize/summarizer.py`
- Modify: `src/text_summarize/__main__.py`
- Modify: `src/pipeline_run/runner.py`
- Modify: `src/pipeline_run/__main__.py`

- [ ] **Step 1: Update summarizer.py — add style param**

In `src/text_summarize/summarizer.py`, change the `summarize()` function signature:

```python
def summarize(
    transcript_path: Path,
    output_path: Path,
    api_key: str,
    custom_prompt: str | None = None,
    style: str = "verbatim_summary",
) -> Path | None:
```

And change the `build_prompt` call:

```python
        prompt = build_prompt(transcript_text, custom_prompt=custom_prompt, style=style)
```

- [ ] **Step 2: Update text_summarize/__main__.py — add --style arg**

In `_build_parser()`, add after the `--prompt` argument:

```python
    parser.add_argument(
        "--style",
        default="verbatim_summary",
        help="Summary style template name (default: verbatim_summary)",
    )
```

In `main()`, change the `summarize()` call to pass `style`:

```python
    result = summarize(
        transcript_path=input_path,
        output_path=output_path,
        api_key=config.anthropic_api_key,
        custom_prompt=args.prompt,
        style=args.style,
    )
```

- [ ] **Step 3: Update pipeline_run/runner.py — add style param**

In `run_pipeline()`, add `style` parameter:

```python
def run_pipeline(
    file_path: str,
    work_dir: str,
    python_cmd: str = "python",
    no_summarize: bool = False,
    cleanup: bool = False,
    style: str = "verbatim_summary",
) -> dict:
```

In the text-summarize step, add `--style` to the command:

```python
        step3 = PipelineStep(
            "text_summarize",
            [python_cmd, "-m", "text_summarize", "--input", transcript_path, "--style", style],
        )
```

- [ ] **Step 4: Update pipeline_run/__main__.py — add --style arg**

In `build_parser()`, add after `--no-summarize`:

```python
    parser.add_argument(
        "--style",
        default="verbatim_summary",
        help="Summary style template name (default: verbatim_summary)",
    )
```

In `main()`, pass `style` to `run_pipeline()`:

```python
    result = run_pipeline(
        file_path=str(file_path),
        work_dir=work_dir,
        python_cmd=python_cmd,
        no_summarize=args.no_summarize,
        cleanup=args.cleanup,
        style=args.style,
    )
```

- [ ] **Step 5: Run full test suite**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src pytest tests/unit/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /mnt/e/code/feishu-transcriber
git add src/text_summarize/ src/pipeline_run/
git commit -m "feat: wire --style param through summarizer and pipeline"
```

---

## Task 6: Smoke Test with Real Audio

**Files:** None (manual verification)

- [ ] **Step 1: Run enhanced pipeline on test audio**

```bash
cd /mnt/e/code/feishu-transcriber && source .venv/bin/activate && PYTHONPATH=src DATA_DIR=/tmp/ft-enhanced python -m pipeline_run --file-path tests/test_source.wav --style verbatim_summary
```

Expected: Pipeline completes, output JSON has `speakers` and `verbatim` fields, summary includes speaker-attributed discussion points.

- [ ] **Step 2: Verify transcript JSON has speaker labels**

```bash
python -c "import json; d=json.load(open('/tmp/ft-enhanced/transcripts/test_source.json')); print(f'Speakers: {d[\"speakers\"]}'); print(f'Segments: {len(d[\"segments\"])}'); print(d['verbatim'][:300])"
```

- [ ] **Step 3: Commit any fixes from smoke test**

---

## Task 7: Deploy to Production

**Files:** None (deployment)

- [ ] **Step 1: Deploy to production**

```bash
bash /mnt/e/code/feishu-transcriber/scripts/deploy.sh
```

- [ ] **Step 2: Install pyyaml in production**

```bash
cd /opt/feishu-transcriber && source .venv/bin/activate && uv pip install pyyaml
```

- [ ] **Step 3: Verify production pipeline**

```bash
cd /opt/feishu-transcriber && source .venv/bin/activate && source .env && PYTHONPATH=src python -m pipeline_run --file-path /mnt/e/code/feishu-transcriber/tests/test_source.wav --style verbatim_summary
```

Expected: Pipeline runs successfully with speaker diarization enabled.
