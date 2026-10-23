"""Отметка живости регламента: след каждого успешного прогона.

Зачем. У GitHub расписание в публичном репозитории тихо выключается после 60
дней без активности репозитория. В спокойный период ночной прогон не меняет ни
корпус, ни метрики, то есть не коммитит ничего, и активности нет. Эта отметка
меняется каждый прогон, поэтому коммит бота есть всегда, а заодно по файлу
видно, когда регламент последний раз доходил до конца.

Отметка говорит о прогоне, а не о cbr.ru. Прогон доходит до конца и когда сайт
отказал: корпус на диске цел, индекс пересобран, регрессия посчитана. Поэтому
рядом с `last_success` стоит состояние источника: `source_status`,
`last_source_contact` и сколько прогонов подряд до сайта не дошли. Без этих
полей цепочка зеленых прогонов означала бы «корпус обновлялся», а на деле
могла бы означать «cbr.ru две недели нас не пускал».

Числа отсюда в публичные тексты не берутся: файл меняется каждую ночь, а текст
с таким числом ломал бы проверку чисел на следующем же прогоне.

Запуск: python scripts/record_heartbeat.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def next_state(state: dict, update: dict, regression: dict, now: str) -> dict:
    # Состояние источника берется из отчета докачки, а не из аргумента командной
    # строки: так оно не может разойтись с тем, что записал сам шаг сбора.
    status = update.get("status", "ok")
    reached = status == "ok"
    return {
        **state,
        "last_success": now,
        "runs": int(state.get("runs", 0)) + 1,
        "corpus_version": regression["corpus_version"],
        "new_acts_last_run": update["new_acts_added"],
        "watched_metric": regression["metrics"]["NDCG@10"],
        "source_status": status,
        "last_source_contact": now if reached else state.get("last_source_contact"),
        "source_unavailable_runs": (
            0 if reached else int(state.get("source_unavailable_runs", 0)) + 1
        ),
        "note": (
            "отметка живости регламента: прогон дошел до конца. "
            "Дошел ли он до cbr.ru, сказано в source_status"
        ),
    }


def main() -> int:
    path = RESULTS / "regulation.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    update = json.loads((RESULTS / "corpus-update.json").read_text(encoding="utf-8"))
    regression = json.loads((RESULTS / "regression.json").read_text(encoding="utf-8"))

    state = next_state(
        state, update, regression, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
