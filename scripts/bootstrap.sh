#!/usr/bin/env bash

set -euo pipefail

# Bootstrap development environment:
# - Create uv-managed virtualenv if missing
# - Install project in editable mode with dev deps
# - Run test suite

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

echo "==> segimage bootstrap starting"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is required but not found on PATH." >&2
  echo "Install uv: see https://docs.astral.sh/uv/getting-started/" >&2
  exit 1
fi

# Create venv if it doesn't exist
if [ -d ".venv" ]; then
  echo "==> Using existing virtual environment: .venv"
else
  echo "==> Creating virtual environment with uv"
  uv venv
fi

echo "==> Installing editable project with dev dependencies"
uv pip install -e '.[dev]'

echo "==> Running tests"
uv run -m pytest -q

echo "==> Bootstrap complete"


