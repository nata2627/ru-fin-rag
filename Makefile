PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: help venv venv-dense venv-dbt check lint test fmt serve vectors warehouse mart dashboard dag dag-drill dag-report alert sample-size drain logs-up logs-down clean

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
	$(PY) -m ruff check src tests dags
	$(PY) -m ruff format --check src tests dags

fmt: ## отформатировать код
	$(PY) -m ruff format src tests dags
	$(PY) -m ruff check --fix src tests dags

test: ## только тесты
	$(PY) -m pytest

serve: ## поднять сервис локально
	$(PY) -m uvicorn rufinrag.api:app --app-dir src --host 127.0.0.1 --port 8000

vectors: ## догнать плотные векторы до корпуса после ночной докачки (нужны venv-dense и веса)
	$(PY) scripts/extend_vectors.py

warehouse: ## сверить режимы витрины: файлы через chdb против ClickHouse из compose
	$(PY) scripts/compare_warehouses.py

mart: ## собрать витрину dbt в ClickHouse и сверить ее вывод с бесплатным режимом
	$(PY) scripts/build_mart.py

dashboard: ## собрать страницу дашборда по журналу репозитория в site/
	$(PY) scripts/build_dashboard.py

dag: ## прогнать DAG регламента дважды в контейнере Airflow (нужен поднятый докер)
	scripts/run_dag.sh

dag-drill: ## то же плюс учебное падение шага в середине цепочки
	scripts/run_dag.sh --drill

dag-report: ## собрать results/dag.md из отчетов прогонов DAG
	$(PY) scripts/report_dag.py

alert: ## учебная деградация метрики: алерт обязан сработать и обязан молчать на шуме
	$(PY) scripts/check_alert.py

sample-size: ## пересчитать правило остановки и размеры выборки для сравнения конфигураций
	$(PY) scripts/plan_sample_size.py

drain: ## репетиция выгрузки журнала: сервис процессом, выгрузка по сети, файлы в репозитории
	$(PY) scripts/rehearse_drain.py

logs-up: ## поднять ClickHouse для стройки (профиль logs)
	docker compose --profile logs up -d clickhouse

logs-down: ## остановить ClickHouse
	docker compose --profile logs down

clean: ## убрать кэши инструментов
	rm -rf .pytest_cache .ruff_cache
	find src tests -name __pycache__ -type d -exec rm -rf {} +
