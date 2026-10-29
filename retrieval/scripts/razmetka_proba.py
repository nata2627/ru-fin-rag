#!/usr/bin/env python3
"""Проба разметки языковой моделью: 150 пар против контрольной разметки человека.

Пары те же, что в `human_sample_new.tsv` (их разметил автор проекта 24.10.2026),
все они из готового пула `pool_live.tsv`. Разметчик видит запрос, акт, раздел,
пункты и фрагмент, то есть ровно то, что видел человек, и инструкцию `RUBRIC`
из `rufin.annotation`, ту же, что у человека и у прежних судей. Ответов человека
в партиях нет: `prepare` берет из листа только колонки без ответов.

Порог записан до данных (24.10.2026, задание этапа 11): простая каппа с
человеком ниже 0,40 означает стоп и доклад, а не другую модель или другой
промпт.

    python3 scripts/razmetka_proba.py prepare --dir <папка партий>
    (каждая партия размечается, ответ кладется рядом: batch_NN.answers.tsv)
    python3 scripts/razmetka_proba.py apply --dir <папка партий> --usage <usage.json>
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import RUBRIC, cohen_kappa, kappa_interval, kappa_verdict  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from kappa import read_marks, read_types  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
HUMAN = os.path.join(QDIR, "human_sample_new.tsv")
EARLIER = os.path.join(QDIR, "model_sample_new.tsv")
OUT_SHEET = os.path.join(QDIR, "razmetka_proba.tsv")
REPEAT_SHEET = os.path.join(QDIR, "razmetka_proba_povtor.tsv")
DOC = os.path.join(ROOT, "docs", "18-nabor-do-i-posle.md")
BLOCK_START = "<!-- tablica: razmetka_proba.py -->"
BLOCK_END = "<!-- /tablica proba -->"
RESULT = os.path.join(ROOT, "data", "results", "razmetka-proba.json")
REPORT = os.path.join(ROOT, "docs", "raw", "razmetka_proba.txt")
POOL = os.path.join(QDIR, "pool_live.tsv")

THRESHOLD = 0.40
# Второе число приемки судьи (S17): согласие двух прогонов одной модели.
REPEAT_THRESHOLD = 0.70
BLIND = ("nomer", "zapros", "akt", "razdel", "punkty", "fragment", "query_id", "chunk_id")
SHEET = ("nomer", "otvet1_celikom", "otvet2_vmeste", "tip") + BLIND[1:]
ANSWERS = {"да", "нет"}
SECOND = {"да", "нет", "-"}
TYPES = {"общая", "часть", "иное", "-"}

TASK = """Ниже инструкция разметчику и пронумерованные пары «запрос, фрагмент». Разметь каждую
пару строго по инструкции. Ответ: по одной строке на пару, четыре поля через табуляцию, без
заголовка и без пояснений:

номер<TAB>ответ на вопрос 1 (да или нет)<TAB>ответ на вопрос 2 (да, нет или -, если на
вопрос 1 «да»)<TAB>пометка типа (общая, часть, иное, или -, если пометка не ставится)

"""


def read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def render(row: dict) -> str:
    return (f"=== Пара {row['nomer']} ===\nЗапрос: {row['zapros']}\nАкт: {row['akt']}\n"
            f"Раздел: {row['razdel']}\nПункты: {row['punkty']}\nФрагмент:\n{row['fragment']}\n")


def prepare(args) -> None:
    rows = [{k: r[k] for k in BLIND} for r in read_tsv(HUMAN)]
    pool = {(r["query_id"], r["chunk_id"]) for r in read_tsv(POOL)}
    outside = [r["nomer"] for r in rows if (r["query_id"], r["chunk_id"]) not in pool]
    if outside:
        raise SystemExit(f"пары вне пула: {outside[:5]}")
    os.makedirs(args.dir, exist_ok=True)
    head = TASK + "ИНСТРУКЦИЯ\n\n" + RUBRIC + "\n\nПАРЫ\n\n"
    batches = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    for n, batch in enumerate(batches, start=1):
        with open(os.path.join(args.dir, f"batch_{n:02d}.prompt.txt"), "w", encoding="utf-8") as f:
            f.write(head + "\n".join(render(r) for r in batch))
    manifest = {"пар": len(rows), "партий": len(batches), "в партии": args.batch,
                "sha256 инструкции": hashlib.sha256(RUBRIC.encode()).hexdigest()[:16],
                "sha256 задания": hashlib.sha256(TASK.encode()).hexdigest()[:16]}
    with open(os.path.join(args.dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False))


def parse(path: str) -> dict[str, tuple[str, str, str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().lower() for p in line.rstrip("\n").split("\t")]
            if len(parts) != 4 or not parts[0].isdigit():
                continue
            nomer, first, second, kind = parts
            if first not in ANSWERS or second not in SECOND or kind not in TYPES:
                raise SystemExit(f"{os.path.basename(path)}: ответ вне шкалы в строке {line!r}")
            if first == "да":
                second, kind = "-", "-"
            elif second == "нет":
                kind = "-"
            out[nomer] = (first, second, kind)
    return out


def agreement(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    x, y = [a[k] for k in common], [b[k] for k in common]
    k, (lo, hi) = cohen_kappa(x, y), kappa_interval(x, y)
    return {"пар": len(common), "каппа": round(k, 3), "интервал": [round(lo, 3), round(hi, 3)],
            "каппа взвешенная": round(cohen_kappa(x, y, weighted=True), 3), "словами": kappa_verdict(k),
            "совпало дословно": sum(1 for i, j in zip(x, y) if i == j),
            "грубых расхождений": sum(1 for i, j in zip(x, y) if abs(i - j) == 2),
            "матрица": {f"{i}-{j}": n for (i, j), n in sorted(collections.Counter(zip(x, y)).items())}}


def read_answers(folder: str) -> dict[str, tuple[str, str, str]]:
    answers: dict[str, tuple[str, str, str]] = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".answers.tsv"):
            answers.update(parse(os.path.join(folder, name)))
    return answers


def write_sheet(path: str, answers: dict[str, tuple[str, str, str]], human: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(SHEET)
        for r in human:
            first, second, kind = answers[r["nomer"]]
            w.writerow([r["nomer"], first, second, kind] + [r[k] for k in BLIND[1:]])


def num(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def doc_block(result: dict) -> str:
    h, r = result["с человеком"], result["повтор"]
    rows = [("первый прогон с человеком", h), ("повтор с человеком", r["с человеком"]),
            ("повтор с первым прогоном", r["с первым прогоном"]),
            ("первый прогон с прежней внешней моделью", result["с прежней внешней моделью"])]
    out = ["| Сверка | Пар | Каппа [95%] | Взвешенная | Дословно | Грубых |", "|---|---|---|---|---|---|"]
    for name, a in rows:
        out.append(f"| {name} | {a['пар']} | {num(a['каппа'])} [{num(a['интервал'][0])}; "
                   f"{num(a['интервал'][1])}] | {num(a['каппа взвешенная'])} | {a['совпало дословно']} | "
                   f"{a['грубых расхождений']} |")
    t = result["токены"]
    total = f"{t['всего']:,}".replace(",", " ")
    m, p = result["оценки модели"], result["оценки человека"]
    out += ["", f"Оценки 0, 1, 2 у модели {m['0']}, {m['1']}, {m['2']}, у человека {p['0']}, {p['1']}, "
                f"{p['2']}. Пороги {num(result['порог простой каппы'])} с человеком и "
                f"{num(r['порог'])} между прогонами {'взяты' if result['порог взят'] else 'не взяты'}. "
                f"Токенов на пару за один прогон {t['на пару за один прогон']} по отчету запуска, "
                f"за оба прогона {total}."]
    return "\n".join(out)


def apply(args) -> None:
    human = read_tsv(HUMAN)
    answers = read_answers(args.dir)
    missing = [r["nomer"] for r in human if r["nomer"] not in answers]
    if missing:
        raise SystemExit(f"нет ответов на пары: {missing[:10]} (всего {len(missing)})")
    write_sheet(OUT_SHEET, answers, human)

    model, person = read_marks(OUT_SHEET), read_marks(HUMAN)
    with_human = agreement(model, person)
    with_earlier = agreement(model, read_marks(EARLIER))
    types_m, types_h = read_types(OUT_SHEET), read_types(HUMAN)
    both = sorted(set(types_m) & set(types_h))
    usage = json.load(open(args.usage, encoding="utf-8"))
    manifest = json.load(open(os.path.join(args.dir, "manifest.json"), encoding="utf-8"))
    repeat = None
    if args.repeat:
        again = read_answers(args.repeat)
        lost = [r["nomer"] for r in human if r["nomer"] not in again]
        if lost:
            raise SystemExit(f"в повторе нет ответов на пары: {lost[:10]}")
        write_sheet(REPEAT_SHEET, again, human)
        second = read_marks(REPEAT_SHEET)
        repeat = {"с первым прогоном": agreement(second, model), "с человеком": agreement(second, person),
                  "порог": REPEAT_THRESHOLD}
        repeat["порог взят"] = repeat["с первым прогоном"]["каппа"] >= REPEAT_THRESHOLD
    passed = with_human["каппа"] >= THRESHOLD and (repeat is None or repeat["порог взят"])
    result = {"дата": args.date, "разметчик": usage["модель"], "как запущен": usage["как"],
              "пар": len(human), "партий": manifest["партий"],
              "инструкция": {"sha256": manifest["sha256 инструкции"], "задание sha256": manifest["sha256 задания"]},
              "порог простой каппы": THRESHOLD, "порог записан": "24.10.2026, задание этапа 11, задача 11.4",
              "с человеком": with_human, "с прежней внешней моделью": with_earlier,
              "оценки модели": {str(k): v for k, v in sorted(collections.Counter(model.values()).items())},
              "оценки человека": {str(k): v for k, v in sorted(collections.Counter(person.values()).items())},
              "пометка типа совпала": {"пар с пометкой у обоих": len(both),
                                       "совпало": sum(1 for k in both if types_m[k] == types_h[k])},
              "повтор": repeat, "токены": usage["токены"], "порог взят": passed,
              "решение": "идти в полную разметку ядра можно" if passed else "стоп и доклад"}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    lines = [f"разметчик {result['разметчик']}, пар {result['пар']}, партий {result['партий']}",
             f"с человеком: каппа {with_human['каппа']} {with_human['интервал']}, взвешенная "
             f"{with_human['каппа взвешенная']}, дословно {with_human['совпало дословно']} из "
             f"{with_human['пар']}, грубых {with_human['грубых расхождений']}",
             f"матрица (модель-человек): {with_human['матрица']}",
             f"с прежней внешней моделью: каппа {with_earlier['каппа']} {with_earlier['интервал']}",
             f"оценки модели {result['оценки модели']}, человека {result['оценки человека']}",
             f"токены: {usage['токены']}",
             (f"повтор: с первым прогоном каппа {repeat['с первым прогоном']['каппа']} "
              f"{repeat['с первым прогоном']['интервал']}, дословно {repeat['с первым прогоном']['совпало дословно']}; "
              f"с человеком {repeat['с человеком']['каппа']} {repeat['с человеком']['интервал']}"
              if repeat else "повтора нет"),
             f"пороги {THRESHOLD} и {REPEAT_THRESHOLD}: {'взяты' if passed else 'не взяты'}, {result['решение']}"]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if repeat is not None:
        body = open(DOC, encoding="utf-8").read()
        pattern = re.compile(re.escape(BLOCK_START) + ".*?" + re.escape(BLOCK_END), re.S)
        if not pattern.search(body):
            raise SystemExit(f"в {DOC} нет меток {BLOCK_START} ... {BLOCK_END}")
        with open(DOC, "w", encoding="utf-8") as f:
            f.write(pattern.sub(lambda _: f"{BLOCK_START}\n{doc_block(result)}\n{BLOCK_END}", body))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dir", required=True)
    p.add_argument("--batch", type=int, default=25)
    a = sub.add_parser("apply")
    a.add_argument("--dir", required=True)
    a.add_argument("--usage", required=True, help="json: модель, как запущен, токены по отчету запуска")
    a.add_argument("--date", required=True)
    a.add_argument("--repeat", help="папка второго прогона теми же партиями")
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else apply(args)


if __name__ == "__main__":
    main()
