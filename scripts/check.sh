#!/usr/bin/env bash
# Проверка перед тем, как считать задачу сделанной. Падает на первой ошибке,
# чтобы нельзя было отчитаться о готовности при красном тесте.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

# Проверка запуском, а не по признаку файла: в CI PY это имя команды из PATH,
# а не путь к venv, и `-x python` там ложно срабатывает как отсутствие окружения.
if ! "$PY" -c "import rufinrag" >/dev/null 2>&1; then
  echo "Нет окружения $PY с установленным пакетом. Собрать: make venv" >&2
  exit 1
fi

echo "== ruff"
"$PY" -m ruff check src tests dags
"$PY" -m ruff format --check src tests dags

echo "== pytest"
"$PY" -m pytest

echo "== числа публичных текстов"
"$PY" scripts/verify_numbers.py

echo "== секреты в рабочем дереве"
scripts/check_secrets.sh

echo "check: зеленый"
