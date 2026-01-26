#!/usr/bin/env bash
set -euo pipefail

DEST="data/raw/retailrocket"
mkdir -p "$DEST"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Missing 'kaggle' CLI."
  echo "Install:  .venv/bin/pip install kaggle"
  echo "Auth:     https://www.kaggle.com/docs/api  (download kaggle.json to ~/.kaggle/kaggle.json)"
  exit 1
fi

echo "Downloading RetailRocket dataset (events.csv) into $DEST ..."
kaggle datasets download -d retailrocket/ecommerce-dataset -p "$DEST" --unzip

echo "Done. Expected file:"
echo "  $DEST/events.csv"
ls -la "$DEST" | sed -n '1,200p'

