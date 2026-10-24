#!/usr/bin/env python3
"""Приёмка всех судей против ручной разметки: один файл чисел на весь порог.

Правило приёмки объявлено до данных (`orchestration/EXPERIMENTS.md` части Б,
S17): каппа судьи с человеком не ниже 0,40, согласие двух прогонов одной модели
не ниже 0,70, стек вывода записан. `scripts/kappa.py` считает это по одному
судье и печатает разбор, а здесь то же самое считается по всем сразу и ложится
одним файлом, чтобы числа публичных текстов брались из него, а не переносились
из четырёх текстовых отчётов руками.

Ручная разметка 150 пар сделана человеком 24.10.2026 (`human_sample_new.tsv`).
До неё человеческого якоря в проекте не было вовсе: прежняя каппа 0,408
измерена между двумя моделями, и в карточке набора это написано.

Запуск: python scripts/judge_acceptance.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rufin.annotation import cohen_kappa, kappa_interval, kappa_verdict  # noqa: E402

spec = importlib.util.spec_from_file_location("kappa", os.path.join(ROOT, "scripts", "kappa.py"))
kappa_mod = importlib.util.module_from_spec(spec)
sys.modules["kappa"] = kappa_mod
spec.loader.exec_module(kappa_mod)

ЧЕЛОВЕК = "data/queries/human_sample_new.tsv"
ВЫХОД = "data/results/judge_acceptance.json"
ОТЧЁТ = "docs/raw/judge_acceptance.txt"
ПОРОГ_ЧЕЛОВЕК = kappa_mod.ПОРОГ_ЧЕЛОВЕК
ПОРОГ_ПОВТОР = kappa_mod.ПОРОГ_ПОВТОР

СУДЬИ = (
    {
        "судья": "Qwen2.5-14B-Instruct-AWQ",
        "лист": "data/kaggle_out/ru-fin-judge-proba-14b/queries/judge_live.tsv",
        "повтор": "data/kaggle_out/ru-fin-judge-14b-repeat/queries/judge_live_repeat.tsv",
        "машина": "Tesla T4",
    },
    {
        "судья": "Qwen2.5-32B-Instruct-AWQ",
        "лист": "data/kaggle_out/ru-fin-judge-32b-fix/queries/judge_live_32b.tsv",
        "повтор": "",
        "машина": "Tesla T4 x2",
    },
    {
        "судья": "Qwen2.5-7B-Instruct-AWQ",
        "лист": "data/kaggle_out/ru-fin-judge-7b/queries/judge_live_7b.tsv",
        "повтор": "",
        "машина": "Tesla T4",
    },
    {
        "судья": "Qwen3-4B-AWQ",
        "лист": "data/kaggle_out/local-3070/judge_local_4b.tsv",
        "повтор": "data/kaggle_out/local-3070/judge_local_4b_repeat.tsv",
        "машина": "RTX 3070",
    },
)


def согласие(первый: dict, второй: dict) -> dict | None:
    общие = sorted(set(первый) & set(второй))
    if len(общие) < 2:
        return None
    a = [первый[k] for k in общие]
    b = [второй[k] for k in общие]
    низ, верх = kappa_interval(a, b)
    простая = cohen_kappa(a, b)
    return {
        "пар": len(общие),
        "каппа": round(простая, 3),
        "интервал": [round(низ, 3), round(верх, 3)],
        "каппа взвешенная": round(cohen_kappa(a, b, weighted=True), 3),
        "словами": kappa_verdict(простая),
        "совпало дословно": sum(1 for x, y in zip(a, b) if x == y),
        "грубых расхождений": sum(1 for x, y in zip(a, b) if abs(x - y) == 2),
    }


def main() -> int:
    человек = kappa_mod.read_marks(os.path.join(ROOT, ЧЕЛОВЕК))
    if not человек:
        return print(f"{ЧЕЛОВЕК}: разметки нет, считать нечего", file=sys.stderr) or 2

    записи = []
    for судья in СУДЬИ:
        путь = os.path.join(ROOT, судья["лист"])
        if not os.path.exists(путь):
            continue
        оценки = kappa_mod.read_marks(путь)
        с_человеком = согласие(оценки, человек)
        строка = {
            "судья": судья["судья"],
            "машина": судья["машина"],
            "лист": судья["лист"],
            "размечено судьёй": len(оценки),
            "с человеком": с_человеком,
        }
        мета = os.path.splitext(путь)[0] + "_meta.json"
        строка["стек вывода записан"] = os.path.exists(мета) and all(
            json.load(open(мета, encoding="utf-8")).get(k) is not None
            for k in ("модель", "семя", "температура", "стек вывода")
        )
        if судья["повтор"] and os.path.exists(os.path.join(ROOT, судья["повтор"])):
            строка["с собой"] = согласие(
                оценки, kappa_mod.read_marks(os.path.join(ROOT, судья["повтор"]))
            )
        взято = [
            bool(с_человеком and с_человеком["каппа"] >= ПОРОГ_ЧЕЛОВЕК),
            bool(строка.get("с собой") and строка["с собой"]["каппа"] >= ПОРОГ_ПОВТОР),
            bool(строка["стек вывода записан"]),
        ]
        строка["чисел взято"] = sum(взято)
        строка["приёмка"] = all(взято)
        записи.append(строка)

    лучший = max(записи, key=lambda r: (r["с человеком"] or {}).get("каппа", -1))
    итог = {
        "дата": "2026-10-24",
        "правило": {
            "порог каппы с человеком": ПОРОГ_ЧЕЛОВЕК,
            "порог согласия двух прогонов": ПОРОГ_ПОВТОР,
            "записано": "до данных, orchestration/EXPERIMENTS.md части Б, S17",
        },
        "ручная разметка": {
            "файл": ЧЕЛОВЕК,
            "пар": len(человек),
            "кто разметил": "человек (автор проекта), 24.10.2026",
            "оценки": {
                str(оценка): sum(1 for v in человек.values() if v == оценка) for оценка in (0, 1, 2)
            },
        },
        "судьи": записи,
        "лучший судья": лучший["судья"],
        "приёмка взята": any(r["приёмка"] for r in записи),
        "следствие": (
            "разметка судьи не применяется: пул глубины 50 не размечается, "
            "метрики остаются на прежней разметке"
        ),
    }
    with open(os.path.join(ROOT, ВЫХОД), "w", encoding="utf-8") as f:
        json.dump(итог, f, ensure_ascii=False, indent=2)
        f.write("\n")

    строки = [
        "Приёмка судей против ручной разметки 150 пар",
        f"Порог каппы с человеком {ПОРОГ_ЧЕЛОВЕК}, объявлен до данных (S17).",
        "",
        f"{'судья':28} {'пар':>5} {'каппа':>7} {'интервал':>18} {'совпало':>9} {'приёмка':>8}",
    ]
    for r in записи:
        ч = r["с человеком"] or {}
        строки.append(
            f"{r['судья']:28} {ч.get('пар', 0):>5} {ч.get('каппа', 0):>7.3f} "
            f"{str(ч.get('интервал')):>18} "
            f"{ч.get('совпало дословно', 0):>4} из {ч.get('пар', 0):<4} "
            f"{'взята' if r['приёмка'] else 'НЕ ВЗЯТА':>8}"
        )
    строки += ["", итог["следствие"]]
    with open(os.path.join(ROOT, ОТЧЁТ), "w", encoding="utf-8") as f:
        f.write("\n".join(строки) + "\n")

    print("\n".join(строки))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
