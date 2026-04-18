#!/usr/bin/env bash
set -euo pipefail

SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --skip-download|-s) SKIP_DOWNLOAD=1 ;;
    -h|--help)
      echo "Usage: $0 [--skip-download|-s]"
      echo "  --skip-download, -s   Reuse existing data.zip instead of re-downloading"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

source .venv/bin/activate
uv sync --all-extras

if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
  if [ ! -f data.zip ]; then
    echo "--skip-download set but data.zip is missing" >&2
    exit 1
  fi
  echo "Skipping download, reusing existing data.zip"
else
  curl -L "https://temp.kotol.cloud/api/download/TRAIN_SET.zip?code=7IDU" --output data.zip
  unzip -o data.zip -d data_raw
fi



python tearcls/data_split.py
python tearcls/augmentation.py
