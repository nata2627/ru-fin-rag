#!/usr/bin/env python3
"""Строка состояния приемки в README, написанная прогоном, а не рукой.

29.10.2026 плановый прогон покраснел, а за ним и прогон приемки, оба на сверке
чисел: `record_acceptance.py` записал в `results/acceptance.json` новый прогон,
порвавший цепочку, а README называл прежний. Это тот же дефект, что и с таблицей
навыков: число, которое прогон меняет сам, стояло в тексте руками.

Поэтому состояние приемки живет одной строкой, и переписывает ее прогон перед
сверкой. Рассказ про разрывы 23.10 рядом остается рукописным нарочно: это
история, ее ссылки не меняются никогда.

Запуск: python scripts/acceptance_line.py [--write]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
НАЧАЛО = "**Состояние приемки"


def ru(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def дата(метка: str) -> str:
    return f"{метка[8:10]}.{метка[5:7]}.{метка[:4]}"


def строка() -> str:
    д = json.loads((ROOT / "results" / "acceptance.json").read_text(encoding="utf-8"))
    части = [
        f"**Состояние приемки на {дата(д['checked_at'])}: "
        f"плановых прогонов {д['scheduled_runs_total']}, "
        f"зеленых {д['scheduled_runs_green']}, "
        f"цепочка {д['streak_green_runs']} прогонов и {ru(д['streak_days'])} дня "
        f"из {д['days_required']}.**"
    ]
    разрыв = д.get("broken_by")
    if разрыв:
        номер = разрыв["run"].rsplit("/", 1)[-1]
        части.append(
            f" Последний разрыв цепочки: [прогон {номер}]({разрыв['run']}), "
            f"{дата(разрыв['at'])}."
        )
    return "".join(части)


def вписать(текст: str, новая: str) -> str:
    начало = текст.index(НАЧАЛО)
    конец = текст.index("\n", начало)
    return текст[:начало] + новая + текст[конец:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="вписать строку в README")
    args = ap.parse_args()
    if not args.write:
        print(строка())
        return
    main_bez_argumentov()


def main_bez_argumentov() -> None:
    новая = строка()
    текст = README.read_text(encoding="utf-8")
    if НАЧАЛО not in текст:
        raise SystemExit(
            "в README нет строки состояния приемки: ее переписали руками, "
            "и генератор больше не держит эти числа"
        )
    обновлен = вписать(текст, новая)
    if обновлен == текст:
        print("строка состояния приемки уже совпадает с замером")
        return
    README.write_text(обновлен, encoding="utf-8")
    print("строка состояния приемки переписана по замеру")


if __name__ == "__main__":
    main()
