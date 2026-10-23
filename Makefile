PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: help venv venv-dense venv-dbt check lint test fmt serve warehouse mart drain logs-up logs-down clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 20

venv: ## собрать окружение на python3.11 и поставить зависимости
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	# analytics это движок ClickHouse внутри процесса, 354 МБ на диске: без
	# него сверка режимов витрины пропускается, а она часть проверки.
	$(PIP) install -e '.[analytics,dev]'

venv-dense: ## добавить в окружение плотный поиск и переранжирование (тянет torch)
	$(PIP) install -e '.[dense,dev]'

venv-dbt: ## добавить в окружение dbt с адаптером ClickHouse (нужен сервер, см. logs-up)
	$(PIP) install -e '.[dbt]'

check: ## линтер, тесты, числа README, проверка на секреты
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

warehouse: ## сверить режимы витрины: файлы через chdb против ClickHouse из compose
	$(PY) scripts/compare_warehouses.py

mart: ## собрать витрину dbt в ClickHouse и сверить ее вывод с бесплатным режимом
	$(PY) scripts/build_mart.py

drain: ## репетиция выгрузки журнала: сервис процессом, выгрузка по сети, файлы в репозитории
	$(PY) scripts/rehearse_drain.py

logs-up: ## поднять ClickHouse для стройки (профиль logs)
	docker compose --profile logs up -d clickhouse

logs-down: ## остановить ClickHouse
	docker compose --profile logs down

clean: ## убрать кэши инструментов
	rm -rf .pytest_cache .ruff_cache
	find src tests -name __pycache__ -type d -exec rm -rf {} +
