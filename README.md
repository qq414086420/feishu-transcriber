# Feishu Transcriber · 飞书语音转纪要

> Atomic CLI tools for transcribing Feishu audio/video files and generating structured meeting minutes.
>
> 把飞书里的音频/视频文件自动转写成文字，并生成结构化的会议纪要。

端到端管线：**下载/导入 → 转码 → 转写（含说话人分离）→ 生成纪要**。每个环节都是独立的命令行工具，既可以一键串起来跑，也可以单独调用，方便集成进 agent 或自动化脚本。

---

## ✨ 特性

- **端到端管线**：一条命令完成「媒体文件 → 会议纪要 Markdown」。
- **中文 ASR**：基于 [FunASR](https://github.com/modelscope/FunASR) 的 SenseVoice（`iic/SenseVoiceSmall`），中文识别效果优秀。
- **说话人分离**：3D-Speaker（CAM++）做 diarization，再按时间戳与 ASR 结果对齐，输出**带说话人标注的逐字稿**。
- **VAD 分块转写**：用 fsmn-vad 切分语音段落，得到准确的句级时间戳，提升说话人对齐质量。
- **纪要风格可配置**：纪要 prompt 抽成 `config/summary_styles/*.yaml` 模板，`--style` 切换，可自定义。
- **机器友好**：成功/失败都输出**单行 JSON**，便于脚本和 agent 解析。
- **优雅降级**：说话人分离失败时自动退化为 `UNKNOWN` 说话人，不阻断管线。

---

## 🧩 工作流程

```
音频/视频文件
   │
   ├─→ media_to_audio      转 16kHz 单声道 WAV（ffmpeg）
   │
   ├─→ audio_transcribe    SenseVoice ASR（VAD 分块）+ 3D-Speaker 说话人分离 + 对齐
   │                          ↓
   │                   带说话人标注的逐字稿 JSON
   │
   └─→ text_summarize      Claude 按 YAML 模板生成纪要
                              ↓
                       会议纪要 Markdown
```

---

## 📋 环境要求

| 依赖 | 说明 |
|------|------|
| **Python ≥ 3.11** | |
| **ffmpeg** | 系统级安装，用于音频转码与切片（`ffmpeg -version` 可验证） |
| **NVIDIA GPU + CUDA**（可选） | 显著加速转写；无 GPU 则 CPU 运行 |
| **飞书应用凭证** | 仅「在线下载模式」需要 |
| **Anthropic API Key** | 生成纪要需要 |

---

## 🚀 安装

```bash
git clone https://github.com/qq414086420/feishu-transcriber.git
cd feishu-transcriber

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # 或：uv pip install -e .

cp .env.example .env             # 然后填入真实凭证（见下表）
```

> 安装后所有模块都可通过 `python -m <module>` 直接运行；不想安装也可用 `PYTHONPATH=src python -m <module>`。

---

## ⚙️ 配置（`.env`）

| 变量 | 必需 | 说明 |
|------|:----:|------|
| `FEISHU_APP_ID` | 在线模式 | 飞书应用 App ID |
| `FEISHU_APP_SECRET` | 在线模式 | 飞书应用 App Secret |
| `ANTHROPIC_API_KEY` | 生成纪要 | Claude API Key |
| `DATA_DIR` | 否 | 数据目录，默认 `./data` |
| `LOGS_DIR` | 否 | 日志目录，默认 `./logs` |

---

## 📖 使用

### 一键管线（推荐）

两种互斥模式 —— 本地文件（离线）或飞书消息（在线）：

```bash
# 离线模式：直接处理本地文件
python -m pipeline_run --file-path /tmp/recording.mp3

# 在线模式：从飞书下载后再处理
python -m pipeline_run \
    --message-id <消息ID> \
    --file-key  <文件key> \
    --type      audio          # audio | video | file

# 处理完只保留纪要，删除中间产物
python -m pipeline_run --file-path /tmp/recording.mp3 --cleanup

# 只转写，不生成纪要
python -m pipeline_run --file-path /tmp/recording.mp3 --no-summarize

# 指定纪要风格
python -m pipeline_run --file-path /tmp/recording.mp3 --style verbatim_summary
```

#### 完整参数

| 参数 | 说明 |
|------|------|
| `--file-path <路径>` | 离线模式：本地文件路径（与 `--message-id` 二选一） |
| `--message-id <ID>` | 在线模式：飞书消息 ID（需配合 `--file-key`） |
| `--file-key <key>` | 在线模式：飞书文件 key |
| `--type <type>` | 媒体类型：`audio` / `video` / `file`（默认 `file`） |
| `--cleanup` | 成功后删除中间目录（inbox/audio/transcripts），仅保留纪要 |
| `--no-summarize` | 跳过纪要生成步骤 |
| `--style <name>` | 纪要风格模板名（默认 `verbatim_summary`） |

### 单独使用子工具

```bash
# 1) 下载飞书文件 / 导入本地文件到 inbox
python -m feishu_download --file-path /tmp/a.mp3
python -m feishu_download --message-id <ID> --file-key <key> --type audio

# 2) 媒体转 16kHz 单声道 WAV
python -m media_to_audio --input /tmp/a.mp4

# 3) 转写（支持 --device cuda 走 GPU）
python -m audio_transcribe --input data/audio/a.wav --device cuda

# 4) 生成纪要
python -m text_summarize --input data/transcripts/a.json --style verbatim_summary
```

> 💡 **GPU 加速**：一键管线目前默认用 **CPU** 转写。如需 GPU，请单独运行 `audio_transcribe --device cuda` 完成转写，再跑 `text_summarize` 生成纪要。

---

## 📤 输出

**标准输出**为单行 JSON，便于程序解析：

```jsonc
// 成功
{ "status": "ok", "transcript_path": ".../data/transcripts/a.json", "summary_path": ".../data/summaries/a_summary.md" }

// 失败
{ "status": "error", "step": "audio_transcribe", "error": "错误描述" }
```

**生成的文件**：

| 文件 | 路径 | 内容 |
|------|------|------|
| 逐字稿 | `data/transcripts/<name>.json` | 完整文本、带时间戳+说话人的段落、语种、时长 |
| 会议纪要 | `data/summaries/<name>_summary.md` | 结构化 Markdown：主题 / 讨论要点 / 待办 / 结论 |

**数据目录**（`DATA_DIR`，默认 `./data`）：

```
data/
├── inbox/        # 原始导入文件
├── audio/        # 转码后的 WAV
├── transcripts/  # 转写结果 JSON
└── summaries/    # 会议纪要 Markdown
```

**退出码**：`0` 成功 ｜ `1` 参数错误 ｜ `2` 处理失败

---

## 🎨 纪要风格模板

纪要的 prompt 由 YAML 模板驱动，位于 `config/summary_styles/`：

- 内置 `verbatim_summary.yaml`（逐字稿 + 结构化摘要）。
- 通过 `--style <name>` 选择 `config/summary_styles/<name>.yaml`。
- **自定义新风格**：在 `config/summary_styles/` 下新建 `my_style.yaml`，包含 `name`、`description`、`prompt`（用 `{transcript}` 占位符），然后 `--style my_style` 即可。模板缺失时自动回退到内置默认 prompt。

---

## ⚡ 性能与格式

- **支持格式**：音频 `mp3 / wav / m4a / ogg / flac / aac`；视频 `mp4 / avi / mkv / mov / webm`。
- **参考速度**（来自实测，仅供参考）：2 小时音频，GPU 模式转写约 22 秒；纪要生成约 5–15 秒。
- 转写以**中文**为主（`--language zh`）。

---

## 🧪 测试

```bash
pytest                  # 运行全部单测 + 集成测试
pytest --cov            # 覆盖率
ruff check .            # lint
```

---

## 📦 部署

`scripts/deploy.sh` 会把项目 rsync 到生产目录（默认 `/opt/feishu-transcriber`），跳过 `.venv/.git/data/logs/.env`，并在目标侧 `pip install -e .`。按需修改脚本里的 `SRC` / `DST` 路径后执行：

```bash
bash scripts/deploy.sh
```

---

## 🗂️ 项目结构

```
feishu-transcriber/
├── pyproject.toml
├── .env.example
├── scripts/
│   └── deploy.sh
├── config/
│   └── summary_styles/
│       └── verbatim_summary.yaml     # 纪要风格模板
├── src/
│   ├── feishu_download/              # 飞书下载 / 本地导入
│   ├── media_to_audio/               # 音视频 → WAV
│   ├── audio_transcribe/             # ASR + 说话人分离
│   │   ├── transcriber.py            #   SenseVoice + VAD 分块转写
│   │   ├── diarizer.py               #   3D-Speaker 说话人分离
│   │   └── aligner.py                #   ASR ↔ diarization 时间戳对齐
│   ├── text_summarize/               # Claude 生成纪要
│   ├── pipeline_run/                 # 管线编排
│   └── shared/                       # 配置 / 日志 / 退出码
├── tests/
│   ├── unit/
│   └── integration/
└── docs/
    ├── transcribe-skill.md           # 详细调用指南
    └── specs/                        # 设计文档
```

---

## 📝 说明

- 凭证通过环境变量注入，**不会**进入代码或仓库（`.env` 已被 `.gitignore` 忽略）。
- 本项目尚未指定开源协议（默认保留所有权利）。如希望允许他人使用/修改，建议添加 `LICENSE` 文件（如 MIT）。
