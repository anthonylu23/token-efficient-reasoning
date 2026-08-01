#!/usr/bin/env bash
set -euo pipefail

uv run ter-m3 evaluate \
  --phase bridge \
  --condition c0 \
  --split validation \
  --decoding-seed 1729 \
  --progress-every 100 \
  --output-dir artifacts/m3/bridge/evaluation/c0

uv run ter-m3 evaluate \
  --phase bridge \
  --condition c1 \
  --split validation \
  --decoding-seed 1729 \
  --progress-every 100 \
  --output-dir artifacts/m3/bridge/evaluation/c1

uv run ter-m3 evaluate \
  --phase bridge \
  --condition c2 \
  --split validation \
  --decoding-seed 1729 \
  --progress-every 100 \
  --adapter artifacts/m3/bridge/training/c2-resume100/adapter \
  --output-dir artifacts/m3/bridge/evaluation/c2

uv run ter-m3 evaluate \
  --phase bridge \
  --condition c3 \
  --split validation \
  --decoding-seed 1729 \
  --progress-every 100 \
  --adapter artifacts/m3/bridge/training/c3/adapter \
  --output-dir artifacts/m3/bridge/evaluation/c3

uv run ter-m3 evaluate \
  --phase bridge \
  --condition c4 \
  --split validation \
  --decoding-seed 1729 \
  --progress-every 100 \
  --adapter artifacts/m3/bridge/training/c4/adapter \
  --output-dir artifacts/m3/bridge/evaluation/c4

uv run ter-m3 bridge-report \
  --evaluation-runs \
  artifacts/m3/bridge/evaluation/c0 \
  artifacts/m3/bridge/evaluation/c1 \
  artifacts/m3/bridge/evaluation/c2 \
  artifacts/m3/bridge/evaluation/c3 \
  artifacts/m3/bridge/evaluation/c4 \
  --training-runs \
  artifacts/m3/bridge/training/c2-resume100 \
  artifacts/m3/bridge/training/c3 \
  artifacts/m3/bridge/training/c4 \
  --output-dir artifacts/m3/bridge/report
