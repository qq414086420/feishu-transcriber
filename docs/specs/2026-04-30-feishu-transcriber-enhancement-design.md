# 转写增强（说话人识别）+ 纪要风格解耦设计

## 概述

增强现有飞书转写管线的两个维度：1) 加入说话人识别（Speaker Diarization），生成带说话人标注的逐字稿；2) 将纪要生成模板解耦为可配置的 YAML 文件，支持多种纪要风格。

## 技术选型

- **ASR**：保留 SenseVoice（iic/SenseVoiceSmall），中文识别最优
- **说话人识别**：3D-Speaker 独立 pipeline（iic/speech_campplus_speaker-diarization_common）
- **合并策略**：按时间戳重叠度对齐 ASR 片段与说话人段落
- **原因**：SenseVoice + cam++ 存在已知不兼容 bug（GitHub Issue #2706 未修复），无法使用集成方案

## 架构

```
音频 ──→ SenseVoice（文本 + 句级时间戳）
      ──→ 3D-Speaker（说话人时间段 RTTM）
      ──→ aligner（按时间戳合并）
      ──→ 带说话人的逐字稿 JSON
                    ↓
           Claude（按 YAML 模板生成纪要）
```

## 模块改动

### 新增：`src/audio_transcribe/diarizer.py`

3D-Speaker 说话人分离封装。

```python
def diarize(input_path: Path, device: str = "cuda") -> list[DiarizationSegment]:
    """运行说话人分离，返回说话人时间段列表。"""
```

- 使用 modelscope pipeline：`pipeline(task=Tasks.speaker_diarization, model='iic/speech_campplus_speaker-diarization_common')`
- 输入：WAV 音频路径
- 输出：`list[DiarizationSegment]`，每个包含 `start`(秒), `end`(秒), `speaker`(str)
- DiarizationSegment 为 frozen dataclass
- 异常时返回空列表（不阻断管线，降级为无说话人模式）

### 新增：`src/audio_transcribe/aligner.py`

ASR 结果与 diarization 结果按时间戳对齐。

```python
def align(asr_segments: list[dict], diarization_segments: list[DiarizationSegment]) -> list[dict]:
    """将 ASR 文本片段与说话人段落按时间重叠度对齐。"""
```

- 对每个 ASR 片段，找到时间重叠最大的 diarization 段落，取其 speaker
- 无匹配时 speaker 设为 "UNKNOWN"
- 输出与原 ASR segments 相同结构，增加 `speaker` 字段

### 重构：`src/audio_transcribe/transcriber.py`

当前 SenseVoice 返回整块文本无有效时间戳。需要改进：

1. `transcribe()` 函数改为返回句级时间戳的 segments（而非单个整块文本）
2. SenseVoice 的 timestamp 输出格式为 `"start1,end1;start2,end2;..."`（毫秒），需正确解析
3. 如果 SenseVoice 不提供有效 timestamp，用 VAD 切分产生的段时间作为 fallback
4. 新增 `transcribe_with_speakers()` 函数，串联 transcribe → diarize → align

```python
def transcribe_with_speakers(
    input_path: Path,
    output_path: Path,
    language: str = "zh",
    device: str = "cpu",
) -> TranscriptionResult | None:
    """转写 + 说话人识别 + 对齐，输出带说话人的完整结果。"""
```

### 转写输出格式（增强后）

```json
{
  "text": "完整文本",
  "segments": [
    {"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00", "text": "大家好..."},
    {"start": 5.5, "end": 12.0, "speaker": "SPEAKER_01", "text": "好的我先介绍一下..."}
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "language": "zh",
  "duration": 6850.0,
  "verbatim": "[SPEAKER_00]: 大家好...\n[SPEAKER_01]: 好的我先介绍一下...\n..."
}
```

字段说明：
- `segments`：带说话人的逐段记录
- `speakers`：去重后的说话人列表
- `verbatim`：完整逐字稿，格式为 `[说话人]: 文本\n`
- 当 diarization 失败时，speaker 全部为 "UNKNOWN"，verbatim 仍然生成

### 重构：`src/text_summarize/prompts.py`

从硬编码字符串改为加载 YAML 模板文件：

```python
def load_style(style_name: str) -> str:
    """从 config/summary_styles/<name>.yaml 加载 prompt 模板。"""

def build_prompt(transcript_json: str, style: str = "verbatim_summary") -> str:
    """使用指定风格的 prompt 模板。"""
```

- 模板文件位置：`config/summary_styles/<style_name>.yaml`
- YAML 格式：`name`, `description`, `prompt`（含 `{transcript}` 占位符）
- 找不到模板文件时 fallback 到内置 DEFAULT_SUMMARY_PROMPT

### 新增：`config/summary_styles/verbatim_summary.yaml`

默认纪要风格模板：

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
  - 时长：{duration}
  - 参与者：{speakers}

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

### 小改：`src/text_summarize/summarizer.py`

- `summarize()` 函数新增 `style` 参数，默认 `"verbatim_summary"`
- 调用 `build_prompt(transcript_json, style=style)` 获取 prompt

### 小改：`src/text_summarize/__main__.py`

- 新增 `--style` 参数，透传给 `summarize()`

### 小改：`src/pipeline_run/runner.py` + `__main__.py`

- `run_pipeline()` 新增 `style` 参数
- 透传到 text-summarize 步骤的 `--style` 参数

### 小改：`src/audio_transcribe/__main__.py`

- `transcribe_with_speakers()` 替代原来的 `transcribe()` 作为默认调用
- 保持 `--device` 等参数不变

## 依赖变更

```
# pyproject.toml 新增
modelscope >= 1.0.0    # 已有，用于 3D-Speaker pipeline
```

无需新增依赖，modelscope 已在项目中。

## 错误处理

- diarization 失败时降级为无说话人模式（speaker="UNKNOWN"），不阻断管线
- aligner 对空输入直接返回空列表
- 模板文件缺失时 fallback 到内置 prompt
- 所有异常记录日志

## 测试

- `test_diarizer.py`：mock modelscope pipeline，验证输出结构
- `test_aligner.py`：用构造的 ASR/diarization 数据测试对齐逻辑
- `test_transcriber.py`：更新已有测试，新增 `test_transcribe_with_speakers`
- `test_prompts.py`：测试 YAML 模板加载和 fallback
- 集成测试：端到端带说话人的管线

## 后续扩展（不在本期）

- 更多纪要风格模板（纯摘要型、议题型等）
- 说话人命名（从第一次提到的名字推断 SPEAKER_00 → "张三"）
- 指定说话人数量参数
- GPU 内存优化（同时加载 SenseVoice + 3D-Speaker）
