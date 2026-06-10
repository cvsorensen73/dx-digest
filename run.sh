#!/usr/bin/env bash
# Manual trigger — runs the digest locally and prints the output path.
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env file found. Copy .env.example to .env and fill in your API keys."
  exit 1
fi

python3 digest.py
echo ""
echo "Open docs/index.html in your browser, or push to GitHub to update the live page."
