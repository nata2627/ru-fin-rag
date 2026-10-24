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
#         scripts/run_dag.sh --drill   (то же плюс учебное падение шага в середине)
#         scripts/run_dag.sh --inside  (внутри контейнера, так его зовет строка выше)
#
# Учебное падение стоит за флагом, а не в основном пути: прогон CI
# (`.github/workflows/dag.yml`) зовет скрипт без флага, и пока замер падения не
# снят руками, в плановый прогон он не попадает.
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=${AIRFLOW_IMAGE:-apache/airflow:2.10.5-python3.11}
HOME_DIR=logs/airflow-home
DRILL=no
for arg in "$@"; do
  [ "$arg" = "--drill" ] && DRILL=yes
done

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
    -e GITHUB_RUN_ID="${GITHUB_RUN_ID:-}" \
    -e RUFINRAG_ROOT=/repo \
    -e RUFINRAG_PYTHON=/repo/$HOME_DIR/venv/bin/python \
    -e RUFINRAG_EVENT_DIR=/tmp/rufinrag-events \
    -v "$PWD:/repo" \
    -w /repo \
    -e RUFINRAG_DAG_DRILL="$DRILL" \
    "$IMAGE" bash scripts/run_dag.sh --inside
  echo
  echo "репетиция переписала рабочие файлы прогона: results/index.json,"
  echo "results/regression.json, results/regulation.json, results/dag-rerun.json"
  echo "и data/index. Числа README сняты прогоном CI, поэтому после репетиции"
  echo "записи плановых прогонов возвращаются: git checkout -- results"
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

# Прогон CI клонирует репозиторий, и отметок шагов в клоне нет: первый прогон
# обязан сделать работу. Повторная репетиция на той же машине начинается с
# отметками прошлой, и тогда первый прогон пропускает все шаги, а проверка
# повторного прогона сверяет не то, что собиралась. Поэтому репетиция стартует
# оттуда же, откуда стартует CI.
rm -f data/stamps/*.json

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
SECONDS_LIST=()
for run in "$RUN1" "$RUN2"; do
  number=$((number + 1))
  echo "== прогон DAG на $run"
  started=$(date +%s)
  # Лог каждого прогона отдельным файлом: `airflow dags test` пишет задачи в
  # свой вывод, а не в папку логов, и в артефакт прогона CI уезжает именно он.
  airflow dags test corpus_refresh "$run" 2>&1 | tee "$HOME_DIR/dag-test-$number.log"
  # Время прохода снимается здесь, а не внутри задач: нужно время всего
  # регламента, включая разбор DAG и работу базы, а не сумма шагов.
  SECONDS_LIST+=("$(( $(date +%s) - started ))")
  echo "== прогон $number занял ${SECONDS_LIST[-1]} с"
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
  --states "$HOME_DIR/states-1.json" "$HOME_DIR/states-2.json" \
  --seconds "${SECONDS_LIST[@]}"

if [ "${RUFINRAG_DAG_DRILL:-no}" = "yes" ]; then
  echo "== учебное падение шага в середине цепочки"
  # Отметки шагов убираются нарочно: после двух прогонов входы «не менялись», и
  # датчик свежести пропустил бы переиндексацию, то есть падать было бы нечему.
  # Так же выглядит состояние после настоящей доливки корпуса.
  rm -f data/stamps/index.json data/stamps/regression.json

  # Интерпретатор-обманка: один шаг регламента он запускать отказывается, а
  # остальные уходят настоящему интерпретатору. Падает ровно так, как упал бы
  # настоящий шаг: подпроцесс вернул ненулевой код, и DAG об этом узнает тем же
  # способом, что в плановом прогоне.
  FAIL_PY=$HOME_DIR/fail-python
  cat > "$FAIL_PY" <<'WRAP'
#!/usr/bin/env bash
set -u
for arg in "$@"; do
  case "$arg" in
    *"$RUFINRAG_FAIL_SCRIPT")
      echo "учебная тревога: $RUFINRAG_FAIL_SCRIPT остановлен" >&2
      exit 9
      ;;
  esac
done
exec "$RUFINRAG_REAL_PYTHON" "$@"
WRAP
  chmod +x "$FAIL_PY"

  RUN3=$(date -u -d '-40 minutes' +%Y-%m-%dT%H:%M:00+00:00)
  RUN4=$(date -u -d '-20 minutes' +%Y-%m-%dT%H:%M:00+00:00)

  "$PY" scripts/check_dag_recovery.py --stage before --state "$HOME_DIR/recovery.json"

  echo "== прогон DAG на $RUN3, шаг build_index обязан упасть"
  started=$(date +%s)
  # Пауза между попытками на время тревоги секундная: `airflow dags test` спит
  # ее честно, и две попытки по пять минут были бы десятью минутами ожидания.
  RUFINRAG_PYTHON="/repo/$FAIL_PY" \
  RUFINRAG_REAL_PYTHON="/repo/$PY" \
  RUFINRAG_FAIL_SCRIPT=scripts/build_index.py \
  RUFINRAG_RETRY_MINUTES=0.05 \
    airflow dags test corpus_refresh "$RUN3" 2>&1 | tee "$HOME_DIR/dag-test-3.log" || true
  SECONDS_LIST+=("$(( $(date +%s) - started ))")
  airflow tasks states-for-dag-run corpus_refresh "$RUN3" --output json > "$HOME_DIR/states-3.json"
  "$PY" scripts/check_dag_recovery.py --stage fail --state "$HOME_DIR/recovery.json"

  echo "== прогон DAG на $RUN4, тот же регламент без обманки"
  started=$(date +%s)
  airflow dags test corpus_refresh "$RUN4" 2>&1 | tee "$HOME_DIR/dag-test-4.log"
  SECONDS_LIST+=("$(( $(date +%s) - started ))")
  airflow tasks states-for-dag-run corpus_refresh "$RUN4" --output json > "$HOME_DIR/states-4.json"

  echo "== что осталось от работы после падения и доделал ли ее следующий прогон"
  "$PY" scripts/check_dag_recovery.py --stage after \
    --state "$HOME_DIR/recovery.json" \
    --states "$HOME_DIR/states-3.json" "$HOME_DIR/states-4.json" \
    --log "$HOME_DIR/dag-test-3.log" \
    --seconds "${SECONDS_LIST[@]}"
fi
