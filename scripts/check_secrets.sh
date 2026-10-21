#!/usr/bin/env bash
# Шаблоны секретов в том, что уходит наружу. Ищем в файлах под контролем git,
# потому что именно они попадают в публичный репозиторий.
set -euo pipefail

cd "$(dirname "$0")/.."

patterns='hf_[A-Za-z0-9]{34}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{70,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'

found=$(git ls-files -z | xargs -0 grep -InE "$patterns" 2>/dev/null || true)
if [ -n "$found" ]; then
  echo "Найдено похожее на секрет:" >&2
  echo "$found" >&2
  exit 1
fi
echo "секретов не найдено"
