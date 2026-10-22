"""Отметка живости регламента: след каждого успешного прогона.

Зачем. У GitHub расписание в публичном репозитории тихо выключается после 60
дней без активности репозитория. В спокойный период ночной прогон не меняет ни
корпус, ни метрики, то есть не коммитит ничего, и активности нет. Эта отметка
меняется каждый прогон, поэтому коммит бота есть всегда, а заодно по файлу
видно, когда регламент последний раз доходил до конца.

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


def main() -> int:
    path = RESULTS / "regulation.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    update = json.loads((RESULTS / "corpus-update.json").read_text(encoding="utf-8"))
    regression = json.loads((RESULTS / "regression.json").read_text(encoding="utf-8"))

    state.update(
        {
            "last_success": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runs": int(state.get("runs", 0)) + 1,
            "corpus_version": regression["corpus_version"],
            "new_acts_last_run": update["new_acts_added"],
            "watched_metric": regression["metrics"]["NDCG@10"],
            "note": "отметка живости регламента, пишется каждым успешным прогоном",
        }
    )
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
