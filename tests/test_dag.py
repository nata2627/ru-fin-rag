"""Разбор DAG регламента: состав задач, порядок и правила пропуска.

Тест идет только там, где Airflow стоит, то есть в контейнере прогона
`.github/workflows/dag.yml`. В окружении сервиса Airflow нет и не будет: он
тянет свое замыкание зависимостей, а образу хостинга оно не нужно ни строкой.

Проверяется не состав ради состава. Правило `none_failed` на задачах после
докачки это то место, где проект уже один раз заплатил: в ночном прогоне отказ
cbr.ru уронил вместе с докачкой переиндексацию, регрессию и отметку живости, и
цепочка приемки автономности обнулилась. У правила `all_success` пропуск
докачки означал бы то же самое.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="Airflow стоит только в контейнере прогона DAG")

from airflow.models.dagbag import DagBag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ["fetch_acts", "build_index", "run_regression", "record_heartbeat"]


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(ROOT / "dags"), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    return bag.get_dag("corpus_refresh")


def test_dag_holds_the_four_steps_of_the_regulation(dag):
    assert sorted(task.task_id for task in dag.tasks) == sorted(CHAIN)


def test_steps_go_in_one_chain(dag):
    for upstream, downstream in pairwise(CHAIN):
        assert dag.get_task(downstream).upstream_task_ids == {upstream}


def test_a_skipped_fetch_does_not_cancel_the_rest(dag):
    for task_id in CHAIN[1:]:
        assert dag.get_task(task_id).trigger_rule == "none_failed"


def test_retries_stand_where_a_repeat_can_help(dag):
    assert dag.get_task("fetch_acts").retries == 2
    assert dag.get_task("run_regression").retries == 0


def test_runs_do_not_pile_up(dag):
    """Два прогона над одним файлом корпуса одновременно это гонка."""
    assert dag.max_active_runs == 1
    assert dag.catchup is False
