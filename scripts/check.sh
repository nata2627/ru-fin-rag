#!/usr/bin/env bash
# Проверка перед тем, как считать задачу сделанной. Падает на первой ошибке,
# чтобы нельзя было отчитаться о готовности при красном тесте.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if [ ! -x "$PY" ]; then
  echo "Нет окружения $PY. Собрать: make venv" >&2
  exit 1
fi

echo "== ruff"
"$PY" -m ruff check src tests
"$PY" -m ruff format --check src tests

echo "== pytest"
"$PY" -m pytest

echo "== секреты в рабочем дереве"
scripts/check_secrets.sh

echo "check: зеленый"
