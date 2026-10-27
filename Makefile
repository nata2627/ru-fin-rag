PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: help venv venv-dense venv-dbt check check-retrieval check-all lint test fmt serve answer-threshold vectors graph graph-expansion warehouse mart dashboard dag dag-drill dag-report alert monitoring monitoring-thresholds monitoring-drill sample-size drain logs-up logs-down clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 20

# На чистой Ubuntu 24.04 стоит python3.12, и жесткий вызов python3.11 обрывал
# README на первой же команде сообщением «No such file or directory». Берем то,
# что есть, и проверяем версию сами: ошибка должна называть причину.
PYBOOT := $(shell command -v python3.11 || command -v python3.12 || command -v python3)

venv: ## собрать окружение (python 3.11 или 3.12) и поставить зависимости
	@test -n "$(PYBOOT)" || { echo "python3 не найден: нужен python 3.11 или 3.12"; exit 1; }
	@$(PYBOOT) -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info < (3,13) else 1)' \
		|| { echo "нужен python 3.11 или 3.12, найден $$($(PYBOOT) -V)"; exit 1; }
	$(PYBOOT) -m venv .venv
	$(PIP) install --upgrade pip
	# analytics это движок ClickHouse внутри процесса, 354 МБ на диске: без
	# него сверка режимов витрины пропускается, а она часть проверки.
	$(PIP) install -e '.[analytics,dev]'

venv-dense: ## добавить в окружение плотный поиск и переранжирование (тянет torch)
	$(PIP) install -e '.[dense,dev]'

venv-dbt: ## добавить в окружение dbt с адаптером ClickHouse (нужен сервер, см. logs-up)
	$(PIP) install -e '.[dbt]'

venv-tracking: ## добавить в окружение трекинг прогонов MLflow (клиент без сервера, 66 МБ)
	$(PIP) install -e '.[tracking]'

registry: ## пересобрать реестр версий results/registry.json
	$(PY) scripts/register_versions.py

track: ## завести прогоны из results/*.json в MLflow (нужен venv-tracking)
	$(PY) scripts/track_runs.py

check: ## линтер, тесты, числа README, проверка на секреты
	scripts/check.sh

# Часть А живет папкой `retrieval` со своим окружением: ей нужны torch и
# sentence-transformers, части Б fastapi и langgraph, и одно окружение тянуло
# бы тяжелое в сервис, который держится на 512 МБ бесплатного тарифа.
check-retrieval: ## тесты и стиль части А (папка retrieval, свое окружение)
	cd retrieval && $(MAKE) lint test

check-all: ## обе проверки одной командой: сервис и часть А
	$(MAKE) check
	$(MAKE) check-retrieval

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

answer-threshold: ## замерить порог отказа слоя ответа, поля цитаты и цену опоры
	$(PY) scripts/measure_answer.py

answer-quality: ## замерить долю ответов с корректной ссылкой на источник (шесть минут)
	$(PY) scripts/measure_answer_quality.py

router: ## замерить роутер против прямого поиска, качество и задержка (десять минут)
	$(PY) scripts/measure_router.py

venv-router: ## добавить в окружение роутер на инструментах (langgraph, 70 МБ)
	$(PIP) install -e '.[router]'

vectors: ## догнать плотные векторы до корпуса после ночной докачки (нужны venv-dense и веса)
	$(PY) scripts/extend_vectors.py

graph: ## собрать граф ссылок между актами корпуса и напечатать его числа
	$(PY) scripts/build_graph.py

graph-expansion: ## замерить расширение выдачи по цитированию: настройка на dev, числа на подвыборках
	$(PY) scripts/measure_graph_expansion.py

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

monitoring: ## мониторинг по журналу запросов: дрейф, доля без хорошего ответа, регрессия
	$(PY) scripts/check_monitoring.py

monitoring-thresholds: ## пересчитать пороги мониторинга из разброса на эталонном наборе
	$(PY) scripts/measure_monitoring_thresholds.py

monitoring-drill: ## учебная лестница: алерты мониторинга обязаны сработать и обязаны молчать
	$(PY) scripts/drill_monitoring.py

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
