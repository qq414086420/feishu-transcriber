# Feishu Transcriber — Agent 使用指南

## 概述

飞书音视频转文字管线。将飞书传来的音频/视频文件自动转写为文字并生成结构化会议纪要。

## 调用方式

### 方式一：传入已下载的文件路径（推荐）

当已获得本地文件时使用：

```bash
cd /opt/feishu-transcriber && source .venv/bin/activate && source .env && \
PYTHONPATH=src python -m pipeline_run --file-path <文件路径>
```

### 方式二：传入飞书消息标识

当收到飞书文件消息，需要自动下载时：

```bash
cd /opt/feishu-transcriber && source .venv/bin/activate && source .env && \
PYTHONPATH=src python -m pipeline_run --message-id <消息ID> --file-key <文件key> --type <audio|video|file>
```

### 可选参数

| 参数 | 说明 |
|------|------|
| `--cleanup` | 成功后删除中间文件（inbox/audio/transcripts），只保留纪要 |
| `--no-summarize` | 只转写不生成纪要 |

## 输出格式

成功时 stdout 输出一行 JSON：

```json
{
  "status": "ok",
  "transcript_path": "/opt/feishu-transcriber/data/transcripts/meeting.json",
  "summary_path": "/opt/feishu-transcriber/data/summaries/meeting_summary.md"
}
```

失败时：

```json
{
  "status": "error",
  "step": "audio_transcribe",
  "error": "错误描述"
}

```

## 输出文件

| 文件 | 路径 | 内容 |
|------|------|------|
| 转写文本 | `data/transcripts/<文件名>.json` | 含完整文本、带时间戳段落、语种、时长 |
| 会议纪要 | `data/summaries/<文件名>_summary.md` | 结构化 Markdown：主题、要点、待办、结论 |

## 退出码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 参数错误 |
| 2 | 处理失败 |

## 支持的文件格式

音频：mp3, wav, m4a, ogg, flac, aac
视频：mp4, avi, mkv, mov, webm

## 处理性能

- 2 小时音频 → 约 22 秒（GPU 模式）
- 纯中文语音识别
- 纪要生成约 5-15 秒

## 典型调用示例

### OpenClaw 收到飞书音频消息后触发

```
用户在飞书发送了一个音频文件
  → OpenClaw 解析消息，提取 file_key 和 message_id
  → 调用 pipeline_run --message-id xxx --file-key yyy --type audio
  → 读取 stdout JSON 中的 summary_path
  → 读取纪要内容，通过飞书回复用户
```

### 直接处理本地文件

```bash
cd /opt/feishu-transcriber && source .venv/bin/activate && source .env && \
PYTHONPATH=src python -m pipeline_run --file-path /tmp/recording.mp3 --cleanup
```

## 单独调用子工具

如需只执行某一步骤：

```bash
# 仅下载飞书文件
PYTHONPATH=src python -m feishu_download --message-id xxx --file-key yyy --type audio

# 仅转换格式
PYTHONPATH=src python -m media_to_audio --input /path/to/file.mp4

# 仅转写
PYTHONPATH=src python -m audio_transcribe --input /path/to/audio.wav --device cuda

# 仅生成纪要
PYTHONPATH=src python -m text_summarize --input /path/to/transcript.json
```

## 环境要求

- `/opt/feishu-transcriber/.env` 需配置：
  - `FEISHU_APP_ID` / `FEISHU_APP_SECRET`（在线下载模式必需）
  - `ANTHROPIC_API_KEY`（纪要生成必需）
- ffmpeg 已安装在系统中
- GPU 可用时自动使用 CUDA，否则回退 CPU
