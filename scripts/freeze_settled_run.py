"""Заморозка установившегося прогона регламента.

README сравнивает первый прогон с установившимся: первый качал 80 выпусков и
стоил минуты, установившийся не качает ничего и стоит десятки секунд. Первое
число заморожено в `results/corpus-update-first.json`, а второе до 28.10.2026
стояло в README вписанным руками, и ни в одном файле его не было. Такое число
устаревает молча: `results/corpus-update.json` перезаписывается каждой ночью,
и к моменту проверки README рассказывал про 25,0 с, когда прогон стоил 28,9.

Поэтому установившийся прогон замораживается так же, как первый, отдельным
файлом с датой. Условия заморозки проверяются, а не предполагаются: прогон
должен быть успешным, не качать выпусков и не добавлять актов, иначе он не
установившийся и сравнивать его с первым нельзя.

Запуск: python scripts/freeze_settled_run.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ИСТОЧНИК = ROOT / "results/corpus-update.json"
ЦЕЛЬ = ROOT / "results/corpus-update-settled.json"


def проверить(отчет: dict) -> None:
    if отчет.get("status") != "ok":
        raise SystemExit(f"прогон не успешный: status={отчет.get('status')!r}")
    for поле in ("issues_downloaded", "new_acts_added"):
        if отчет.get(поле):
            raise SystemExit(f"прогон не установившийся: {поле}={отчет[поле]}")


def main() -> int:
    отчет = json.loads(ИСТОЧНИК.read_text(encoding="utf-8"))
    проверить(отчет)
    отчет["note"] = (
        "установившийся прогон регламента, заморожен из results/corpus-update.json. "
        "Файл заморожен специально: README ссылается на это число, а живой отчет "
        "перезаписывается каждой ночью"
    )
    ЦЕЛЬ.write_text(json.dumps(отчет, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"заморожен прогон {отчет['checked_at']}: {отчет['seconds']} с, "
        f"загрузок {отчет['issues_downloaded']}, новых актов {отчет['new_acts_added']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
