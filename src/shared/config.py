"""Shared configuration: paths, env vars."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    inbox_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    summaries_dir: Path
    logs_dir: Path
    feishu_app_id: str
    feishu_app_secret: str
    anthropic_api_key: str

    @classmethod
    def from_env(cls) -> "Config":
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        return cls(
            data_dir=data_dir,
            inbox_dir=data_dir / "inbox",
            audio_dir=data_dir / "audio",
            transcripts_dir=data_dir / "transcripts",
            summaries_dir=data_dir / "summaries",
            logs_dir=Path(os.getenv("LOGS_DIR", "./logs")),
            feishu_app_id=os.getenv("FEISHU_APP_ID", ""),
            feishu_app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )

    def ensure_dirs(self) -> None:
        for d in [self.inbox_dir, self.audio_dir, self.transcripts_dir, self.summaries_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
