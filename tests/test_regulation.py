"""Счет регламента: отметка живости и приемка автономности.

Оба числа тут такие, которые проверяющий будет пересчитывать за нами, поэтому
они считаются функциями, а не на глаз по журналу прогонов. Проверяется ровно то
место, где легко соврать в свою пользу: зеленая цепочка после красного прогона
и зеленый прогон, до cbr.ru не дошедший.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heartbeat = load("record_heartbeat")
acceptance = load("record_acceptance")

REGRESSION = {"corpus_version": "chunks62594-acts1554", "metrics": {"NDCG@10": 0.4839}}


def run(created: str, conclusion: str, event: str = "schedule") -> dict:
    return {
        "createdAt": created,
        "conclusion": conclusion,
        "event": event,
        "url": f"https://github.com/nata2627/ru-fin-rag/actions/runs/{created[8:10]}",
    }


def test_streak_restarts_after_a_red_scheduled_run():
    """Правило требует четырнадцать дней подряд, а не четырнадцать зеленых дней.

    Зеленые прогоны до красного в цепочку не входят, сколько бы их ни было, и
    это то толкование, которое было записано до того, как пошли числа.
    """
    scheduled = [
        run("2026-10-22T18:33:43Z", "success"),
        run("2026-10-23T07:34:08Z", "failure"),
        run("2026-10-24T02:17:00Z", "success"),
        run("2026-10-24T14:17:00Z", "success"),
    ]
    streak = acceptance.streak_of(scheduled)
    assert [r["createdAt"] for r in streak] == [
        "2026-10-24T02:17:00Z",
        "2026-10-24T14:17:00Z",
    ]


def test_streak_is_empty_while_the_last_scheduled_run_is_red():
    scheduled = [
        run("2026-10-22T18:33:43Z", "success"),
        run("2026-10-23T07:34:08Z", "failure"),
    ]
    assert acceptance.streak_of(scheduled) == []


def test_cancelled_run_breaks_the_streak_too():
    """Прогон, снятый по таймауту, это не прошедший прогон."""
    scheduled = [run("2026-10-22T18:33:43Z", "success"), run("2026-10-23T07:34:08Z", "cancelled")]
    assert acceptance.streak_of(scheduled) == []


def test_manual_and_running_runs_stay_out_of_the_count():
    """Кнопка проверяет код, расписание проверяет автономность.

    Прогон без вывода идет прямо сейчас, и его исход неизвестен: засчитывать
    его зеленым нельзя, красным тоже.
    """
    runs = [
        run("2026-10-22T18:33:43Z", "success"),
        run("2026-10-22T11:40:16Z", "cancelled", event="workflow_dispatch"),
        run("2026-10-23T07:34:08Z", "", event="schedule"),
    ]
    assert [r["createdAt"] for r in acceptance.decided_scheduled(runs)] == ["2026-10-22T18:33:43Z"]


def test_heartbeat_separates_the_run_from_the_source():
    """Прогон дошел до конца, а до cbr.ru не дошел: это два разных факта.

    Без этого разделения цепочка зеленых прогонов читалась бы как «корпус
    обновлялся», хотя означала бы «сайт нас не пускал».
    """
    update = {"status": "source_unavailable", "source_error": "403", "new_acts_added": 0}
    state = heartbeat.next_state({}, update, REGRESSION, "2026-10-23T18:00:00Z")

    assert state["last_success"] == "2026-10-23T18:00:00Z"
    assert state["source_status"] == "source_unavailable"
    assert state["source_unavailable_runs"] == 1
    assert state["last_source_contact"] is None


def test_heartbeat_counts_refusals_in_a_row_and_forgets_them_after_contact():
    update = {"status": "source_unavailable", "new_acts_added": 0}
    state = heartbeat.next_state({}, update, REGRESSION, "2026-10-23T18:00:00Z")
    state = heartbeat.next_state(state, update, REGRESSION, "2026-10-24T02:20:00Z")
    assert state["source_unavailable_runs"] == 2
    assert state["runs"] == 2

    state = heartbeat.next_state(
        state, {"status": "ok", "new_acts_added": 0}, REGRESSION, "2026-10-24T14:20:00Z"
    )
    assert state["source_unavailable_runs"] == 0
    assert state["last_source_contact"] == "2026-10-24T14:20:00Z"
    assert state["runs"] == 3
