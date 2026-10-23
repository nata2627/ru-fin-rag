"""Регламент обновления корпуса: докачка, переиндексация, регрессия, отметка живости.

Зачем этот DAG, если расписание проекта работает на GitHub Actions. Регламент
по расписанию живет в `.github/workflows/nightly.yml`, потому что бесплатный
хостинг не держит cron, а Airflow нельзя держать бесплатно и без рук (SPEC,
раздел 2). Airflow здесь показан не на словах: тот же порядок шагов описан
DAG'ом, и DAG исполняется в контейнере в прогоне CI
(`.github/workflows/dag.yml`), а лог задач уезжает в артефакты прогона. Если бы
он просто лежал в репозитории, это был бы мертвый код, и так и пришлось бы
написать в README.

Что делает DAG сам и чего не делает. Работу делают скрипты, те же, что в ночном
прогоне, и вызываются они подпроцессом, а не переписаны здесь функциями. DAG
отвечает за три вещи: порядок, ретраи и датчики свежести. Отсюда и граница
окружений: зависимости сервиса стоят отдельным окружением
(`RUFINRAG_PYTHON`), и к зависимостям Airflow они не прикасаются.

Идемпотентность собрана из трех частей.
1. Докачка сверяет акты перечня с корпусом по ключу «номер плюс дата» и с
   памятью о неудачах (`data/corpus/skipped.json`), а дописывает корпус
   подменой файла целиком. Повторный прогон на том же перечне не добавляет ни
   одного фрагмента.
2. Сборка индекса и регрессия имеют датчики свежести: отпечаток входа сверяется
   с отпечатком, записанным при создании вывода, и совпадение означает пропуск
   шага. Вывод при этом перезаписывается, а не дописывается, то есть и
   принудительный повторный прогон дубля не создает.
3. `max_active_runs=1` и `catchup=False`: пропущенные дни не поднимают пачку
   прогонов, которые дрались бы за один файл корпуса.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowSkipException
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(os.environ.get("RUFINRAG_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import pipeline  # noqa: E402

# Интерпретатор с зависимостями сервиса. По умолчанию окружение проекта на
# машине автора (`make venv`), в контейнере прогона это отдельный venv внутри
# образа Airflow.
PYTHON = os.environ.get("RUFINRAG_PYTHON", str(ROOT / ".venv/bin/python"))

# Пауза между попытками упавшей задачи. Плановое значение пять минут, а проверка
# восстановления (`scripts/run_dag.sh`, третий прогон) ставит секунды: `airflow
# dags test` честно спит эту паузу, и две попытки съедали бы десять минут
# прогона CI на ожидание в логе.
RETRY_MINUTES = float(os.environ.get("RUFINRAG_RETRY_MINUTES", "5"))

log = logging.getLogger(__name__)


def run_script(argv: list[str], extra_env: dict[str, str] | None = None) -> int:
    """Запустить шаг подпроцессом и переложить его вывод в лог задачи.

    Вывод именно переливается, а не теряется: лог задач уезжает в артефакты
    прогона CI, и без вывода скриптов там не на что смотреть.
    """
    command = [PYTHON, *argv]
    log.info("запускаю: %s", " ".join(command))
    done = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(extra_env or {})},
        capture_output=True,
        text=True,
        check=False,
    )
    if done.stdout:
        log.info("вывод шага:\n%s", done.stdout.strip())
    if done.stderr:
        log.warning("ошибки шага:\n%s", done.stderr.strip())
    log.info("код выхода %s", done.returncode)
    return done.returncode


def fetch_acts() -> None:
    sensor = pipeline.source_freshness(ROOT)
    live = pipeline.live_run()
    log.info("датчик свежести источника: %s", sensor.reason)
    log.info("боевой режим: %s", "включен" if live else "выключен, RUFINRAG_LIVE не равен 1")
    if not live:
        raise AirflowSkipException("боевой режим выключен, cbr.ru не трогаем")
    if not sensor.stale:
        raise AirflowSkipException(sensor.reason)

    code = run_script(["scripts/update_corpus.py"])
    if code == pipeline.EXIT_SOURCE_UNAVAILABLE:
        # Отказ сайта это не поломка регламента: корпус на диске цел, и
        # остальные шаги идут как обычно. Ровно так же устроен ночной прогон.
        raise AirflowSkipException("cbr.ru отказал, докачка отложена до следующего прогона")
    if code != 0:
        raise RuntimeError(f"докачка корпуса упала кодом {code}")


def run_with_sensor(step_name: str, argv: list[str], extra_env: dict[str, str] | None = None):
    def task() -> None:
        step = pipeline.STEPS[step_name]
        sensor = pipeline.freshness(ROOT, step)
        log.info("датчик свежести шага %s: %s", step.name, sensor.reason)
        if not sensor.stale:
            raise AirflowSkipException(sensor.reason)
        code = run_script(argv, extra_env)
        if code != 0:
            raise RuntimeError(f"шаг {step.name} упал кодом {code}")
        stamp = pipeline.write_stamp(ROOT, step)
        log.info("отметка шага: %s", json.dumps(stamp, ensure_ascii=False))

    return task


def record_heartbeat() -> None:
    code = run_script(["scripts/record_heartbeat.py"])
    if code != 0:
        raise RuntimeError(f"отметка живости упала кодом {code}")


with DAG(
    dag_id="corpus_refresh",
    description="докачка актов с cbr.ru, переиндексация и регрессия на эталонном наборе",
    schedule="17 2,14 * * *",
    start_date=datetime(2026, 10, 22, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "nata2627",
        # Ретрай ставится там, где повтор может помочь, и безопасен он ровно
        # потому, что шаги идемпотентны: повторная докачка не добавит акт
        # дважды, повторная сборка перепишет индекс, а не допишет.
        "retries": 2,
        "retry_delay": timedelta(minutes=RETRY_MINUTES),
    },
    tags=["rufinrag", "corpus"],
) as dag:
    fetch = PythonOperator(task_id="fetch_acts", python_callable=fetch_acts)

    build_index = PythonOperator(
        task_id="build_index",
        python_callable=run_with_sensor("index", ["scripts/build_index.py"]),
        # Правило `none_failed`, а не `all_success`: пропуск докачки не отменяет
        # переиндексацию. В ночном прогоне один отказ cbr.ru уже уронил этим
        # все остальные шаги, и часы приемки автономности обнулились.
        trigger_rule="none_failed",
    )

    regression = PythonOperator(
        task_id="run_regression",
        python_callable=run_with_sensor(
            "regression",
            ["scripts/run_regression.py", "--config", "bm25"],
            {"RUFINRAG_DISABLE_DENSE": "1"},
        ),
        trigger_rule="none_failed",
        # Регрессия падает тогда, когда метрика просела против эталона. Это
        # детерминированный исход, и три попытки дадут тот же ответ втрое
        # дороже.
        retries=0,
    )

    heartbeat = PythonOperator(
        task_id="record_heartbeat",
        python_callable=record_heartbeat,
        trigger_rule="none_failed",
        retries=1,
    )

    fetch >> build_index >> regression >> heartbeat
