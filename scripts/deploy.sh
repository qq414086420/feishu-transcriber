#!/usr/bin/env bash
set -euo pipefail

SRC="/mnt/e/code/feishu-transcriber"
DST="/opt/feishu-transcriber"

echo "Deploying feishu-transcriber to production..."
rsync -av --delete \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='data' \
    --exclude='logs' \
    --exclude='.env' \
    "$SRC/src/" "$DST/src/"

cp "$SRC/pyproject.toml" "$DST/"

mkdir -p "$DST"/{data/{inbox,audio,transcripts,summaries},logs}

cd "$DST"
source .venv/bin/activate
uv pip install -e . --quiet 2>/dev/null || pip install -e . --quiet

echo "Deployment complete."
echo "Remember to update .env if credentials changed."
