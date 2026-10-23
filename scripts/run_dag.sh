#!/usr/bin/env bash
# Прогон DAG регламента в контейнере Airflow: один и тот же путь на маке и в CI.
#
# Внутри образа Airflow поднимается самым дешевым способом: база SQLite и
# `airflow dags test`, то есть задачи исполняются по порядку одним процессом,
# без планировщика и веб-сервера. Планировщик с Postgres здесь не нужен ни для
# чего: прогон одноразовый, и расписание проекту дает GitHub Actions, а не
# Airflow (README, раздел «Регламент по расписанию»).
#
# Зависимости сервиса ставятся в отдельное окружение внутри контейнера, а не
# рядом с зависимостями Airflow: у Airflow свое замыкание версий, и сводить их
# с пинами сервиса незачем. Отсюда и граница ответственности: задачи DAG
# запускают скрипты подпроцессом через `RUFINRAG_PYTHON`.
#
# Запуск: scripts/run_dag.sh          (мак, нужен поднятый докер)
#         scripts/run_dag.sh --inside (внутри контейнера, так его зовет строка выше)
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=${AIRFLOW_IMAGE:-apache/airflow:2.10.5-python3.11}
HOME_DIR=logs/airflow-home

if [ "${1:-}" != "--inside" ]; then
  mkdir -p "$HOME_DIR"
  # `--user` с группой 0 это способ, которым образ Airflow рассчитан ходить под
  # чужим идентификатором: иначе контейнер пишет в смонтированный репозиторий
  # от uid 50000, а в прогоне CI дерево принадлежит раннеру.
  docker run --rm \
    --memory 2g \
    --user "$(id -u):0" \
    -e HOME=/tmp \
    -e AIRFLOW_HOME=/repo/$HOME_DIR \
    -e AIRFLOW__CORE__DAGS_FOLDER=/repo/dags \
    -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
    -e AIRFLOW_IMAGE="$IMAGE" \
    -e RUFINRAG_ROOT=/repo \
    -e RUFINRAG_PYTHON=/repo/$HOME_DIR/venv/bin/python \
    -e RUFINRAG_EVENT_DIR=/tmp/rufinrag-events \
    -v "$PWD:/repo" \
    -w /repo \
    "$IMAGE" bash scripts/run_dag.sh --inside
  echo
  echo "репетиция переписала рабочие файлы прогона: results/index.json,"
  echo "results/regression.json, results/regulation.json и data/index."
  echo "Вернуть записи плановых прогонов: git checkout -- results"
  exit 0
fi

VENV=$HOME_DIR/venv
PY=$VENV/bin/python

if [ ! -x "$PY" ]; then
  echo "== окружение сервиса внутри контейнера"
  python -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet .
fi

echo "== корпус до прогонов"
"$PY" scripts/check_dag_rerun.py --stage before --state "$HOME_DIR/corpus-before.json"

echo "== база Airflow"
airflow db migrate > "$HOME_DIR/db-migrate.log" 2>&1
airflow dags list

# Две даты в прошлом: один и тот же DAG прогоняется дважды, и второй прогон
# обязан увидеть, что входы не менялись.
RUN1=$(date -u -d '-2 hours' +%Y-%m-%dT%H:00:00+00:00)
RUN2=$(date -u -d '-1 hours' +%Y-%m-%dT%H:00:00+00:00)

number=0
for run in "$RUN1" "$RUN2"; do
  number=$((number + 1))
  echo "== прогон DAG на $run"
  # Лог каждого прогона отдельным файлом: `airflow dags test` пишет задачи в
  # свой вывод, а не в папку логов, и в артефакт прогона CI уезжает именно он.
  airflow dags test corpus_refresh "$run" 2>&1 | tee "$HOME_DIR/dag-test-$number.log"
done

airflow tasks states-for-dag-run corpus_refresh "$RUN1" --output json > "$HOME_DIR/states-1.json"
airflow tasks states-for-dag-run corpus_refresh "$RUN2" --output json > "$HOME_DIR/states-2.json"

# Пик памяти самого контейнера, а не оценка на глаз: cgroup v2 держит его сам.
PEAK=/sys/fs/cgroup/memory.peak
if [ -r "$PEAK" ]; then
  RUFINRAG_DAG_PEAK_MB=$(( $(cat "$PEAK") / 1048576 ))
  export RUFINRAG_DAG_PEAK_MB
  echo "== пик памяти контейнера ${RUFINRAG_DAG_PEAK_MB} МБ"
fi

echo "== повторный прогон не плодит дубли"
"$PY" scripts/check_dag_rerun.py --stage after \
  --state "$HOME_DIR/corpus-before.json" \
  --states "$HOME_DIR/states-1.json" "$HOME_DIR/states-2.json"
