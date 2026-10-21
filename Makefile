PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: help venv check lint test fmt serve clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 20

venv: ## собрать окружение на python3.11 и поставить зависимости
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[dev]'

venv-dense: ## добавить в окружение плотный поиск и переранжирование (тянет torch)
	$(PIP) install -e '.[dense,dev]'

check: ## линтер, тесты, проверка на секреты
	scripts/check.sh

lint: ## только линтер
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt: ## отформатировать код
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

test: ## только тесты
	$(PY) -m pytest

serve: ## поднять сервис локально
	$(PY) -m uvicorn rufinrag.api:app --app-dir src --host 127.0.0.1 --port 8000

clean: ## убрать кэши инструментов
	rm -rf .pytest_cache .ruff_cache
	find src tests -name __pycache__ -type d -exec rm -rf {} +
