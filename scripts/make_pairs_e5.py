#!/usr/bin/env python3
"""Пары связанных фрагментов для E5: вопрос, ответ на который требует двух норм.

Пять прошлых прогонов обучали модель на вопросах вида «ответ содержится именно
в этом фрагменте»: один фрагмент, один готовый ответ. Живой вопрос практика
устроен иначе, норма собирается из частей, и это единственное, что меняет E5
(S26 части Б). Здесь готовится вход генератора: пары фрагментов, по которым
вопрос вообще можно задать так, чтобы нужны были оба.

Пара берется двумя способами:

1. `внутри акта`: два фрагмента одного акта из разных пунктов. Соседние по
   порядку не берутся: нарезка идет с перекрытием, и соседи это один кусок
   нормы, разрезанный пополам. Номера пунктов обязательны и не пересекаются.
2. `между актами`: фрагмент, в тексте которого стоит ссылка на другой акт
   корпуса, и фрагмент того акта.

Второй фрагмент пары выбирается лучшим по BM25 по словам первого, а не
случайным из подходящих, и это главное решение скрипта. Случайный далекий
фрагмент того же акта к первому отношения не имеет, вопрос, требующий обоих,
по такой паре не формулируется, и генератор молча вернется к однофрагментному
вопросу. Проверять это потом нечем: доля многофрагментных считается учителем
уже после генерации. BM25 дает тематическую связь и число рядом с парой.

**Оба фрагмента только из группы `train`.** Цитируемый акт, попавший в dev или
в тест, в обучение не идет ни одной парой, иначе проверять E5 было бы нечем.
Это ограничение и есть причина, по которой равные доли из постановки
недостижимы: ссылку на другой обучающий акт содержат 1358 пригодных
фрагментов из 14 905, и дуг у них 1928, а пар «внутри акта» корпус дает
в восемь раз больше. Числа печатаются прогоном, объем выборки держится
прежним (6 490 принятых вопросов), и перекос долей записан в S26
до генерации.

Запуск: python3 scripts/make_pairs_e5.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rufin import split as S  # noqa: E402
from rufin.queryfilter import is_good_source  # noqa: E402
from rufin.references import act_numbers  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402

WITHIN = "внутри акта"
CROSS = "между актами"

# Минимальная длина второго фрагмента. Без нее лучшим по BM25 иногда
# оказывается шапка акта: она состоит из названия, то есть из тех же слов,
# что ссылка в первом фрагменте, а нормы в ней нет.
MIN_SECOND_CHARS = 200
# Насколько далеко должны стоять фрагменты одного акта, чтобы считаться
# разными нормами. Нарезка идет с перекрытием: сосед это вторая половина
# того же куска.
MIN_POSITION_GAP = 2
# Доля принятых генерацией, замерена пробой на карте 26.10: 79 из 100 пар.
# Нужна одной строке вывода, которая пересчитывает пары в вопросы.
ACCEPT_SHARE = 0.79


def body(chunk: dict) -> str:
    """Текст без приписанной шапки акта: шапка одинакова у всех фрагментов."""
    text = chunk["text"]
    return text.split("\n\n", 1)[-1] if "\n\n" in text else text


def parse_units(units: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Диапазон пунктов фрагмента: «1.4–1.5.4» -> ((1,4), (1,5,4))."""
    if not units:
        return None
    parts = [p.strip() for p in units.replace("—", "–").split("–")]
    numbers = []
    for part in parts[:2]:
        try:
            numbers.append(tuple(int(x) for x in part.split(".") if x != ""))
        except ValueError:
            return None
    if not numbers or not numbers[0]:
        return None
    return numbers[0], numbers[-1]


def _nested(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Вложен ли один номер пункта в другой: 1.5 и 1.5.4 это одна норма."""
    return a[:len(b)] == b or b[:len(a)] == a


def units_overlap(first: dict, second: dict) -> bool:
    """Пересекаются ли диапазоны пунктов. Нечитаемый диапазон считается
    пересечением: пара, про которую неизвестно, разные ли это пункты,
    не берется.

    Вложенность проверяется отдельно от порядка. Лексикографически (1, 5)
    меньше (1, 5, 4), поэтому пункт 1.5 и его подпункт 1.5.4 сравнение
    диапазонов пропускало как разные нормы, а это один и тот же текст
    на двух уровнях: 230 таких пар из 12 076 в первом прогоне скрипта.
    """
    a, b = parse_units(first["units"]), parse_units(second["units"])
    if a is None or b is None:
        return True
    if a[0] <= b[1] and b[0] <= a[1]:
        return True
    return any(_nested(x, y) for x in a for y in b)


class ActIndex:
    """Индексы BM25 по актам, по одному на акт и не больше одного раза."""

    def __init__(self, chunks: list[dict]) -> None:
        self.by_act: dict[str, list[dict]] = {}
        self.texts: dict[str, dict] = {}
        for chunk in chunks:
            self.by_act.setdefault(chunk["act_id"], []).append(chunk)
            self.texts[chunk["chunk_id"]] = chunk
        self._cache: dict[str, BM25Index] = {}

    def ranked(self, act_id: str, query: str) -> list[tuple[str, float]]:
        chunks = self.by_act.get(act_id, [])
        if not chunks:
            return []
        index = self._cache.get(act_id)
        if index is None:
            index = BM25Index.build([c["chunk_id"] for c in chunks],
                                    [c["text"] for c in chunks])
            self._cache[act_id] = index
        return index.search(query, len(chunks))


def row(first: dict, second: dict, origin: str, score: float,
        reference: str = "") -> dict:
    return {
        "origin": origin,
        "chunk_id": first["chunk_id"], "chunk_id_2": second["chunk_id"],
        "act_id": first["act_id"], "act_id_2": second["act_id"],
        "number": first["number"], "number_2": second["number"],
        "units": first["units"], "units_2": second["units"],
        "position": first["position"], "position_2": second["position"],
        "act_title": first["title"], "section": first["section"],
        "date": first["date"],
        "bm25": round(score, 2),
        # по какому номеру построена дуга: иначе разбор пары требует
        # повторного разбора текста тем же выражением
        "reference": reference,
    }


def cross_pairs(good: list[dict], index: ActIndex, train_acts: set[str],
                number_to_acts: dict[str, list[str]], per_act: int) -> tuple[list[dict], dict]:
    counts = {"фрагментов со ссылкой внутрь": 0, "дуг": 0,
              "второй короткий": 0, "без общих слов": 0, "предел на акт": 0}
    per_source_act: dict[str, int] = {}
    pairs: list[dict] = []
    for first in good:
        own = first["number"]
        targets: list[tuple[str, str]] = []
        for number in act_numbers(first["text"]):
            if number == own:
                continue
            for act_id in number_to_acts.get(number, []):
                if act_id != first["act_id"] and act_id in train_acts:
                    targets.append((act_id, number))
        if not targets:
            continue
        counts["фрагментов со ссылкой внутрь"] += 1
        seen: set[str] = set()
        for act_id, number in targets:
            if act_id in seen:
                continue
            seen.add(act_id)
            counts["дуг"] += 1
            if per_source_act.get(first["act_id"], 0) >= per_act:
                counts["предел на акт"] += 1
                continue
            взято = None
            for chunk_id, score in index.ranked(act_id, body(first)):
                if score <= 0:
                    break
                second = index.texts[chunk_id]
                if len(body(second)) < MIN_SECOND_CHARS:
                    counts["второй короткий"] += 1
                    continue
                взято = (second, score)
                break
            if взято is None:
                counts["без общих слов"] += 1
                continue
            second, score = взято
            per_source_act[first["act_id"]] = per_source_act.get(first["act_id"], 0) + 1
            pairs.append(row(first, second, CROSS, score, reference=number))
    counts["актов источников"] = len(per_source_act)
    return pairs, counts


def within_pairs(good: list[dict], index: ActIndex, per_act: int,
                 rounds: int = 2) -> tuple[list[dict], dict]:
    """Пары внутри акта, лучшие по BM25 первыми.

    Проходов столько, сколько просит `--rounds` (по умолчанию четыре), и это
    не удвоение ради объема. Лучшее совпадение взаимно:
    для фрагмента A лучший это B, а для B лучший это A, и один проход по
    источникам выбрасывал бы вторую половину пар как повтор (1808 пар на
    первом прогоне). Поэтому во втором проходе источник отдает следующее
    по BM25 совпадение, и предел на акт держится общий.
    """
    counts = {"актов с парами": 0, "второй короткий": 0, "без общих слов": 0,
              "повтор пары": 0}
    by_act: dict[str, list[dict]] = {}
    for chunk in good:
        if chunk["units"]:
            by_act.setdefault(chunk["act_id"], []).append(chunk)
    pairs: list[dict] = []
    for act_id, chunks in by_act.items():
        if len(chunks) < 2:
            continue
        chunks = sorted(chunks, key=lambda c: c["position"])
        texts = {c["chunk_id"]: c for c in chunks}
        порядок = {c["chunk_id"]: index.ranked(act_id, body(c)) for c in chunks}
        место = {c["chunk_id"]: 0 for c in chunks}
        взято: set[frozenset[str]] = set()
        for _ in range(rounds):
            for first in chunks:
                if len(взято) >= per_act:
                    break
                ranked = порядок[first["chunk_id"]]
                while место[first["chunk_id"]] < len(ranked):
                    chunk_id, score = ranked[место[first["chunk_id"]]]
                    место[first["chunk_id"]] += 1
                    if score <= 0:
                        counts["без общих слов"] += 1
                        break
                    second = texts.get(chunk_id)
                    if second is None or second["chunk_id"] == first["chunk_id"]:
                        continue
                    if abs(second["position"] - first["position"]) < MIN_POSITION_GAP:
                        continue
                    if units_overlap(first, second):
                        continue
                    if len(body(second)) < MIN_SECOND_CHARS:
                        counts["второй короткий"] += 1
                        continue
                    ключ = frozenset((first["chunk_id"], second["chunk_id"]))
                    if ключ in взято:
                        counts["повтор пары"] += 1
                        continue
                    взято.add(ключ)
                    pairs.append(row(first, second, WITHIN, score))
                    break
        if взято:
            counts["актов с парами"] += 1
    return pairs, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.path.join(ROOT, "data", "chunks", "base.jsonl"))
    ap.add_argument("--split", default=os.path.join(ROOT, "data", "queries", "split.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "queries", "pairs_e5.jsonl"))
    ap.add_argument("--seed", type=int, default=13, help="то же семя, что у прошлых прогонов")
    ap.add_argument("--per-act", type=int, default=24,
                    help="пар на акт в пуле. Это попытки, а не принятые вопросы: "
                         "предел принятых на акт (12) держит сама генерация, "
                         "а фильтры отклоняют около половины попыток")
    ap.add_argument("--rounds", type=int, default=4,
                    help="сколько совпадений отдает один фрагмент-источник "
                         "в парах внутри акта. Пул должен быть больше цели "
                         "вдвое: фильтры отклоняют около половины попыток")
    ap.add_argument("--per-chunk", type=int, default=8,
                    help="в сколько пар максимум входит один фрагмент. "
                         "В прошлых выборках у фрагмента был один вопрос, "
                         "а здесь ссылочный хаб попадал в 30 пар, и его текст "
                         "стал бы позитивом тридцати вопросов")
    ap.add_argument("--min-bm25", type=float, default=20.0,
                    help="пары, у которых второй фрагмент почти не имеет общих "
                         "слов с первым, отбрасываются: это и есть случайный "
                         "кусок того же акта, по которому вопрос про две нормы "
                         "не задать")
    ap.add_argument("--show", type=int, default=20,
                    help="сколько пар выписать с текстами для проверки глазами")
    ap.add_argument("--show-out", default=os.path.join(ROOT, "logs", "pairs_e5_proba.txt"))
    args = ap.parse_args()

    with open(args.chunks, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    split = S.load(args.split)
    by_act = S.group_by_act(split)
    train_acts = {act_id for act_id, group in by_act.items() if group == S.TRAIN}
    train = [c for c in chunks if c["act_id"] in train_acts]
    good = [c for c in train if is_good_source(c)]
    print(f"фрагментов {len(chunks)}, в группе train {len(train)}, "
          f"пригодных как источник {len(good)}", flush=True)

    number_to_acts: dict[str, list[str]] = {}
    for chunk in chunks:
        acts = number_to_acts.setdefault(chunk["number"], [])
        if chunk["act_id"] not in acts:
            acts.append(chunk["act_id"])

    index = ActIndex(train)
    между, счет_между = cross_pairs(good, index, train_acts, number_to_acts, args.per_act)
    print(f"\n{CROSS}: пар {len(между)}", flush=True)
    for имя, число in счет_между.items():
        print(f"   {имя:<32} {число}", flush=True)

    внутри, счет_внутри = within_pairs(good, index, args.per_act, args.rounds)
    print(f"\n{WITHIN}: пар {len(внутри)}", flush=True)
    for имя, число in счет_внутри.items():
        print(f"   {имя:<32} {число}", flush=True)

    pairs = между + внутри
    # Дедуп по неупорядоченной паре идет по обоим способам вместе. Между
    # актами встречаются обратные ссылки (4464-У и 4466-У ссылаются друг
    # на друга), и такая пара приезжала дважды, с переставленными
    # фрагментами.
    видели: set[frozenset[str]] = set()
    без_повторов: list[dict] = []
    for pair in pairs:
        ключ = frozenset((pair["chunk_id"], pair["chunk_id_2"]))
        if ключ in видели:
            continue
        видели.add(ключ)
        без_повторов.append(pair)
    повторов = len(pairs) - len(без_повторов)
    pairs = без_повторов
    print(f"\nповторов по неупорядоченной паре отброшено {повторов}", flush=True)
    слабые = [pair for pair in pairs if pair["bm25"] < args.min_bm25]
    pairs = [pair for pair in pairs if pair["bm25"] >= args.min_bm25]
    между = [pair for pair in pairs if pair["origin"] == CROSS]
    внутри = [pair for pair in pairs if pair["origin"] == WITHIN]
    print(f"\nпо порогу BM25 {args.min_bm25:.0f} отброшено {len(слабые)} пар",
          flush=True)
    random.Random(args.seed).shuffle(pairs)
    if args.per_chunk:
        вхождений: dict[str, int] = {}
        отобранные: list[dict] = []
        for pair in pairs:
            свои = (pair["chunk_id"], pair["chunk_id_2"])
            if any(вхождений.get(c, 0) >= args.per_chunk for c in свои):
                continue
            for c in свои:
                вхождений[c] = вхождений.get(c, 0) + 1
            отобранные.append(pair)
        print(f"по пределу {args.per_chunk} пар на фрагмент отброшено "
              f"{len(pairs) - len(отобранные)} пар", flush=True)
        pairs = отобранные
        между = [pair for pair in pairs if pair["origin"] == CROSS]
        внутри = [pair for pair in pairs if pair["origin"] == WITHIN]
    for номер, pair in enumerate(pairs):
        pair["pair_id"] = f"pr{номер:05d}"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    доля = len(между) / max(1, len(pairs))
    print(f"\nвсего пар {len(pairs)}: {CROSS} {len(между)} "
          f"({100 * доля:.0f}%), {WITHIN} {len(внутри)}", flush=True)
    # Доля принятых взята из пробы на карте (79 из 100), а не назначена:
    # пересчет пар в вопросы без замера был бы догадкой в выводе прогона.
    print(f"равные доли дали бы {2 * len(между)} пар, то есть около "
          f"{int(2 * len(между) * ACCEPT_SHARE)} принятых вопросов против 6490 "
          f"у прошлых прогонов (доля принятых {ACCEPT_SHARE} замерена пробой): "
          f"объем выборки важнее равенства долей, перекос записан в S26",
          flush=True)
    for имя, часть in ((CROSS, между), (WITHIN, внутри)):
        if часть:
            print(f"   {имя}: BM25 медиана "
                  f"{statistics.median(p['bm25'] for p in часть):.1f}, "
                  f"актов {len({p['act_id'] for p in часть})}", flush=True)
    print(f"файл {os.path.relpath(args.out, ROOT)}", flush=True)

    if args.show:
        рнд = random.Random(args.seed)
        выборка = рнд.sample(pairs, min(args.show, len(pairs)))
        texts = {c["chunk_id"]: c for c in chunks}
        os.makedirs(os.path.dirname(args.show_out) or ".", exist_ok=True)
        with open(args.show_out, "w", encoding="utf-8") as f:
            for pair in выборка:
                f.write(f"==== {pair['pair_id']} [{pair['origin']}] "
                        f"BM25 {pair['bm25']}\n")
                for ключ, ключ_п in (("chunk_id", "units"), ("chunk_id_2", "units_2")):
                    chunk = texts[pair[ключ]]
                    f.write(f"-- {pair[ключ]} пункты {pair[ключ_п] or 'нет'}\n")
                    f.write(body(chunk)[:1200].strip() + "\n\n")
        print(f"двадцать пар с текстами: {os.path.relpath(args.show_out, ROOT)}",
              flush=True)


if __name__ == "__main__":
    main()
