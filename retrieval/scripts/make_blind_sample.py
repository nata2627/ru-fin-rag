#!/usr/bin/env python3
"""Слепая подвыборка листа разметки: пары, которые человек размечает без подсказки.

Зачем. Первый проход по листу на 150 пар делает модель, а человек потом
просматривает ее ответы. Это экономит время, но меняет измеряемое: проверяющий,
который видит готовый ответ, соглашается с ним чаще, чем разметил бы сам, и
каппа судьи с такой разметкой выходит завышенной. Правило приемки судьи
(`orchestration/EXPERIMENTS.md` части Б, запись S17) объявлено до данных и
говорит «каппа с человеком», поэтому человеческий якорь нужен настоящий.

Здесь он и собирается: случайная подвыборка пар того же листа, с записанным
семенем, в которой нет ни ответов модели, ни оценок судьи. Человек размечает ее
первой, до того как увидит чужие ответы, и только потом идет просматривать
остальное. Разница его слепых и просмотренных ответов на этих же парах это цена
подсказки, измеренная, а не предположенная.

Запуск: python3 scripts/make_blind_sample.py [--size 40] [--seed 20261024]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import RUBRIC  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
FIELDS = ["nomer", "otvet1_celikom", "otvet2_vmeste", "tip",
          "zapros", "akt", "razdel", "punkty", "fragment", "query_id", "chunk_id"]

HEAD = """\
Слепой лист разметки: {size} пар из {total} большого листа, выбраны семенем {seed}.

Размечается ПЕРВЫМ, до того как смотреть ответы модели в
data/queries/model_sample_new.tsv. В этом листе нет ни ответов модели, ни
оценок судьи: он и есть человеческий якорь, по которому считается приемка
судьи. Посмотрите чужие ответы раньше, и число перестанет что-либо значить.

Заполняются три колонки таблицы {table}:
  otvet1_celikom   да или нет
  otvet2_vmeste    да или нет; при «да» в первой колонке ставится «-»
  tip              общая, часть или иное; при «да» в первой колонке
                   или «нет» во второй ставится «-»

Оценка 0/1/2 не ставится: она собирается из ответов кодом.

================================ ИНСТРУКЦИЯ ================================

{rubric}
============================== КОНЕЦ ИНСТРУКЦИИ =============================
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=os.path.join(QDIR, "human_sample_new.tsv"))
    ap.add_argument("--out", default=os.path.join(QDIR, "human_blind.tsv"))
    ap.add_argument("--readable", default=os.path.join(ROOT, "docs", "raw", "human_blind.txt"))
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    with open(args.human, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    filled = [r for r in rows if (r.get("otvet1_celikom") or "").strip()]
    if filled:
        # Лист уже размечен, и слепая подвыборка из него бессмысленна: ответы
        # человека в нем уже есть. Это не повод молча взять другие пары.
        raise SystemExit(
            f"в листе {os.path.relpath(args.human, ROOT)} уже заполнено "
            f"{len(filled)} строк из {len(rows)}: слепая подвыборка берется до разметки"
        )

    picked = sorted(random.Random(args.seed).sample(rows, min(args.size, len(rows))),
                    key=lambda r: int(r["nomer"]))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(picked)

    with open(args.readable, "w", encoding="utf-8") as f:
        f.write(HEAD.format(size=len(picked), total=len(rows), seed=args.seed,
                            table=os.path.relpath(args.out, ROOT), rubric=RUBRIC))
        for number, r in enumerate(picked, start=1):
            f.write(f"\n\n--- пара {number} из {len(picked)} "
                    f"(номер в большом листе {r['nomer']}) ---\n\n")
            f.write(f"ВОПРОС: {r['zapros']}\n\n")
            f.write(f"АКТ: {r['akt']}, раздел «{r['razdel']}», "
                    f"пункты {r['punkty'] or 'не названы'}\n\n")
            f.write(f"ФРАГМЕНТ:\n{r['fragment']}\n\n")
            f.write("ответ 1 (ответ целиком, да/нет): ___\n")
            f.write("ответ 2 (помогает вместе, да/нет): ___\n")
            f.write("тип (общая/часть/иное): ___\n")

    queries = len({r["query_id"] for r in picked})
    print(f"слепой лист: {len(picked)} пар из {len(rows)}, вопросов {queries}, "
          f"семя {args.seed}")
    print(f"таблица {os.path.relpath(args.out, ROOT)}, "
          f"читаемая копия {os.path.relpath(args.readable, ROOT)}")


if __name__ == "__main__":
    main()
