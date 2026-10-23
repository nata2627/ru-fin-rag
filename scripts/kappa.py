#!/usr/bin/env python3
"""Приёмка разметки судьи: три числа, а не одно.

Прежде печаталось одно число, каппа Коэна с человеком, и этого оказалось
мало. 01.10.2026 судья `Qwen2.5-14B-Instruct-AWQ` дал на выборке 150 пар
каппу 0,408, 23.10.2026 он же на тех же парах дал 0,307, и расхождение
оказалось односторонним: ни одну пару второй прогон не повысил. Значит
каппа с человеком меряет не только судью, но и сборку, в которой он
запущен, а по одному числу различить это нельзя.

Поэтому приёмка это три числа, и порог у каждого объявлен до прогона:

1. каппа с человеком на выборке, не ниже 0,40;
2. согласие двух прогонов одной и той же модели между собой, не ниже 0,70;
3. стек вывода записан: версия движка, образ, семя и температура лежат
   в метаданных рядом с разметкой.

Не взято хотя бы одно, разметка не применяется. Это отрицательный результат,
а не повод подбирать конфигурацию: порог записан раньше, чем увидены числа.

Каппы считаются две: простая и с квадратичными весами. Оценки упорядочены,
и «2 против 0» хуже, чем «2 против 1»; простая каппа считает эти промахи
одинаковыми. Рядом с каждой печатается интервал бутстрэпом по парам, иначе
сравнение 0,408 с 0,307 выглядит разницей, которой на 150 парах может
и не быть.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import cohen_kappa, kappa_interval, kappa_verdict, scale_of  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "kappa.txt")

ПОРОГ_ЧЕЛОВЕК = 0.40
ПОРОГ_ПОВТОР = 0.70
НУЖНЫ_В_МЕТА = ("стек вывода", "семя", "температура", "модель")


def read_marks(path: str) -> dict[tuple[str, str], int]:
    """Оценки из листа. Где их нет, они собираются из двух ответов.

    У судьи обе колонки заполнены кодом, у человека заполнены только ответы:
    оценку он не ставит, её собирает правило из инструкции.
    """
    out: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["query_id"], row["chunk_id"])
            mark = (row.get("otsenka_0_1_2") or "").strip()
            if mark in ("0", "1", "2"):
                out[key] = int(mark)
                continue
            first = (row.get("otvet1_celikom") or "").strip().lower()
            second = (row.get("otvet2_vmeste") or "").strip().lower()
            if first == "да":
                out[key] = scale_of(True, False)
            elif first == "нет" and second in ("да", "нет"):
                out[key] = scale_of(False, second == "да")
    return out


def read_types(path: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            kind = (row.get("tip") or "").strip().lower()
            if kind in ("общая", "часть", "иное"):
                out[(row["query_id"], row["chunk_id"])] = kind
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(QDIR, "judge_live.tsv"))
    ap.add_argument("--human", default=os.path.join(QDIR, "human_sample_new.tsv"))
    ap.add_argument("--repeat", default="",
                    help="второй прогон той же модели теми же настройками. "
                         "Без него второе число приёмки не считается, и это "
                         "не мелочь: именно оно отличает свойство модели "
                         "от свойства сборки")
    ap.add_argument("--meta", default="",
                    help="метаданные прогона; по умолчанию рядом с --judge")
    ap.add_argument("--sizes", default="",
                    help="размеры выборки через запятую: печатает, насколько "
                         "шире станет интервал каппы на меньшей выборке")
    ap.add_argument("--out", default=REPORT,
                    help="куда писать отчёт. Своё имя нужно всякому замеру, "
                         "кроме опубликованного: иначе проверка другого судьи "
                         "затирает число, которое стоит в карточке набора")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    if not os.path.exists(args.judge):
        raise SystemExit(f"нет {os.path.relpath(args.judge, ROOT)}")
    судья = read_marks(args.judge)
    say(f"разметка судьи: {os.path.relpath(args.judge, ROOT)}, пар {len(судья)}")

    # ---- число 1: согласие с человеком ----
    взято = {}
    if os.path.exists(args.human):
        человек = read_marks(args.human)
        общие = sorted(set(судья) & set(человек))
    else:
        человек, общие = {}, []
        say(f"листа человека нет ({os.path.relpath(args.human, ROOT)}): "
            f"первое число приёмки не считается")

    if общие:
        a = [судья[k] for k in общие]
        b = [человек[k] for k in общие]
        простая = cohen_kappa(a, b)
        взвешенная = cohen_kappa(a, b, weighted=True)
        нижняя, верхняя = kappa_interval(a, b)
        совпало = sum(1 for x, y in zip(a, b) if x == y)
        say(f"пар размечено человеком: {len(человек)}, из них есть у судьи: "
            f"{len(общие)}")
        say(f"совпало дословно: {совпало} из {len(общие)} "
            f"({100 * совпало / len(общие):.0f}%)")
        say()
        say(f"каппа Коэна простая:     {простая:.3f} [{нижняя:.3f}; {верхняя:.3f}]"
            f"  — {kappa_verdict(простая)}")
        say(f"каппа Коэна взвешенная:  {взвешенная:.3f}"
            f"  — {kappa_verdict(взвешенная)}")
        say()
        say("матрица расхождений (строки судья, столбцы человек):")
        матрица: collections.Counter = collections.Counter(zip(a, b))
        say("        " + "".join(f"{j:>8}" for j in (0, 1, 2)))
        for i in (0, 1, 2):
            say(f"   {i:<4}" + "".join(f"{матрица[(i, j)]:>8}" for j in (0, 1, 2)))
        грубые = [(k, x, y) for k, x, y in zip(общие, a, b) if abs(x - y) >= 2]
        say()
        say(f"грубых расхождений (на две ступени): {len(грубые)} из {len(общие)}")
        for k, x, y in грубые[:10]:
            say(f"   {k[0]} / {k[1]}: судья {x}, человек {y}")
        # Тип в оценку не входит, поэтому согласие по нему считается отдельно:
        # им мерится, одинаково ли понята новая часть инструкции.
        тип_с, тип_ч = read_types(args.judge), read_types(args.human)
        оба = sorted(set(тип_с) & set(тип_ч))
        if оба:
            сошлись = sum(1 for k in оба if тип_с[k] == тип_ч[k])
            say()
            say(f"пометка типа: оба поставили на {len(оба)} парах, совпала "
                f"{сошлись} ({100 * сошлись / len(оба):.0f}%)")
            say("   по типам у судьи: "
                + ", ".join(f"{k} {v}" for k, v in
                            sorted(collections.Counter(тип_с.values()).items())))
            say("   по типам у человека: "
                + ", ".join(f"{k} {v}" for k, v in
                            sorted(collections.Counter(тип_ч.values()).items())))
        взято["согласие с человеком"] = (простая, ПОРОГ_ЧЕЛОВЕК,
                                         простая >= ПОРОГ_ЧЕЛОВЕК)

        if args.sizes:
            say()
            say("цена сокращения выборки (бутстрэп по этим же парам):")
            эталон = верхняя - нижняя
            for размер in [int(x) for x in args.sizes.split(",")]:
                lo, hi = kappa_interval(a, b, size=размер)
                say(f"   {размер:>4} пар: интервал [{lo:.3f}; {hi:.3f}], "
                    f"ширина {hi - lo:.3f}, это "
                    f"{(hi - lo) / эталон:.2f} от ширины на {len(общие)}")

    # ---- число 2: согласие двух прогонов одной модели ----
    if args.repeat:
        if not os.path.exists(args.repeat):
            raise SystemExit(f"нет {os.path.relpath(args.repeat, ROOT)}")
        повтор = read_marks(args.repeat)
        пары = sorted(set(судья) & set(повтор))
        if not пары:
            raise SystemExit("два прогона не пересекаются ни на одной паре")
        x = [судья[k] for k in пары]
        y = [повтор[k] for k in пары]
        k2 = cohen_kappa(x, y)
        lo2, hi2 = kappa_interval(x, y)
        одинаково = sum(1 for p, q in zip(x, y) if p == q)
        ниже = sum(1 for p, q in zip(x, y) if q < p)
        выше = sum(1 for p, q in zip(x, y) if q > p)
        say()
        say(f"повтор того же судьи: {os.path.relpath(args.repeat, ROOT)}")
        say(f"каппа двух прогонов между собой: {k2:.3f} [{lo2:.3f}; {hi2:.3f}], "
            f"взвешенная {cohen_kappa(x, y, weighted=True):.3f}")
        say(f"дословно совпало {одинаково} из {len(пары)}; "
            f"повтор понизил {ниже}, повысил {выше}")
        if ниже and not выше or выше and not ниже:
            say("расхождение одностороннее: это сдвиг строгости, а не шум. "
                "Так было 23.10.2026, когда каппа с человеком упала с 0,408 "
                "до 0,307 на той же модели")
        взято["согласие двух прогонов"] = (k2, ПОРОГ_ПОВТОР, k2 >= ПОРОГ_ПОВТОР)
    else:
        say()
        say("второго прогона не передано (--repeat): второе число приёмки "
            "не считается")

    # ---- число 3: стек вывода записан ----
    meta_path = args.meta or os.path.splitext(args.judge)[0] + "_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        нет = [k for k in НУЖНЫ_В_МЕТА if not meta.get(k)]
        стек = meta.get("стек вывода") or {}
        say()
        say(f"метаданные прогона: {os.path.relpath(meta_path, ROOT)}")
        say(f"   модель {meta.get('модель', '?')}, температура "
            f"{meta.get('температура', '?')}, семя {meta.get('семя', '?')}")
        say(f"   vllm {стек.get('vllm', '?')}, torch {стек.get('torch', '?')}, "
            f"питон {стек.get('питон', '?')}")
        say(f"   образ {стек.get('KAGGLE_DOCKER_IMAGE', 'не Kaggle')}, "
            f"карты {стек.get('карты', '?')}")
        say(f"   рубрика sha256 {стек.get('рубрика sha256', '?')}")
        взято["стек вывода записан"] = (0.0 if нет else 1.0, 1.0, not нет)
        if нет:
            say(f"   в метаданных нет: {', '.join(нет)}")
    else:
        say()
        say(f"метаданных нет ({os.path.relpath(meta_path, ROOT)}): третье "
            f"число приёмки не взято")
        взято["стек вывода записан"] = (0.0, 1.0, False)

    # ---- итог ----
    say()
    say("приёмка судьи:")
    for имя, (значение, порог, ок) in взято.items():
        say(f"   {имя:<24} {значение:.3f} при пороге {порог:.2f}: "
            f"{'взято' if ок else 'НЕ ВЗЯТО'}")
    не_считано = [n for n in ("согласие с человеком", "согласие двух прогонов",
                              "стек вывода записан") if n not in взято]
    for имя in не_считано:
        say(f"   {имя:<24} не считано")
    if не_считано:
        say("приёмка неполна: пока все три не посчитаны, разметка не применяется")
    elif all(ок for _, _, ок in взято.values()):
        say("все три взяты: разметку можно применять")
    else:
        say("приёмка не взята: разметка не применяется. Это отрицательный "
            "результат, а не повод подбирать конфигурацию")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
