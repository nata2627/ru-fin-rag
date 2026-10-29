"""Пункт акта как единица разметки вместо фрагмента нарезки.

Эталон «ответ во фрагменте `#0005`» зависит от нарезки, а не от вопроса. Два
случая, где это ломает метрику (`docs/16-razbor-provalov.md`):

* Банк России называет пункт «3.12», разметка по пунктам ставит эталоном
  фрагмент, где 3.12 начинается, а поиск приносит фрагмент с подпунктом
  3.12.2.8. Ответ найден, метрика пишет промах.
* Длинный пункт не помещается во фрагмент и режется на куски. Эталоном стоит
  один кусок, соседний кусок того же пункта считается промахом.

Поэтому единица релевантности здесь пункт. Названный пункт N засчитывается
фрагментом, где начинается N или любой его подпункт N.x, и куском
продолжения N. Метка, поставленная на фрагмент (синтетика, судья, ручные),
становится единицей «фрагмент», а если фрагмент это кусок длинного пункта,
единицей «все куски этого пункта».

Как считается метрика. Выдача фрагментов разворачивается в выдачу пунктов:
фрагмент заменяется пунктами эталона, которые он приносит впервые, фрагмент
без пунктов эталона занимает одну позицию как промах, а повтор уже
принесенного пункта позиции не занимает. На этой выдаче Recall@10, MRR@10 и
NDCG@10 считаются как обычно, поэтому NDCG не превышает единицы, а фрагмент с
двумя названными пунктами получает обе позиции, как и в идеальной выдаче. Для
запроса, где каждая метка стоит на отдельном фрагменте и кусков нет,
развернутая выдача совпадает с исходной, и все числа прежние, это проверяет
тест.

Прежние метки не удаляются: каждый прежний позитив остается внутри своей
единицы.
"""
from __future__ import annotations

import collections
import csv
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

ANNEX = "Приложение"


@dataclass
class Unit:
    unit_id: str
    score: int
    chunks: set[str] = field(default_factory=set)
    osnovanie: str = ""


def piece_groups(chunks: Iterable[dict]) -> dict[str, list[str]]:
    """Куски одного длинного пункта: фрагмент -> все куски его пункта по порядку.

    Куском считается фрагмент, у которого в поле `units` один номер, и соседний
    по позиции фрагмент того же раздела с тем же номером, причем их отрезки
    `char_start`/`char_end` перекрываются или смыкаются. Одинаковый номер в
    разных приложениях одного акта поэтому не склеивается.
    """
    rows = sorted(((c["act_id"], c["position"], c) for c in chunks), key=lambda r: r[:2])
    groups: dict[str, list[str]] = {}
    run: list[dict] = []

    def close() -> None:
        if len(run) > 1:
            ids = [c["chunk_id"] for c in run]
            for cid in ids:
                groups[cid] = ids

    for _, _, c in rows:
        single = bool(c["units"]) and "–" not in c["units"]
        prev = run[-1] if run else None
        if (single and prev is not None and prev["act_id"] == c["act_id"]
                and c["position"] == prev["position"] + 1 and c["units"] == prev["units"]
                and c["section"] == prev["section"] and c["char_start"] <= prev["char_end"] + 1):
            run.append(c)
            continue
        close()
        run = [c] if single else []
    close()
    return groups


def starts_clause_or_sub(body: str, clause: str) -> bool:
    """В фрагменте отдельной строкой начинается пункт или его подпункт."""
    return re.search(r"(?:^|\n)\s*" + re.escape(clause) + r"(?:\.\d+)*\.\s", body) is not None


def clause_chunks(act_chunks: Sequence[dict], clause: str, groups: Mapping[str, list[str]],
                  starts_exact) -> set[str]:
    """Фрагменты, которые засчитываются за названный пункт.

    Начало самого пункта ищется прежним правилом разметки по пунктам во всех
    разделах, чтобы ни один прежний эталон не выпал. Подпункты и куски
    продолжения ищутся только вне приложений: нумерация приложений своя, и
    «3.1» там не подпункт главы 3.
    """
    out = {c["chunk_id"] for c in act_chunks if starts_exact(c["body"], clause)}
    for c in act_chunks:
        if c["section"].startswith(ANNEX):
            continue
        if starts_clause_or_sub(c["body"], clause):
            out.add(c["chunk_id"])
        elif c["chunk_id"] in groups and (c["units"] == clause or c["units"].startswith(clause + ".")):
            out.add(c["chunk_id"])
    return out


def build_units(qrels: Mapping[str, Mapping[str, int]], named: Mapping[str, list[str]],
                act_chunks: Mapping[str, list[dict]], act_of_query: Mapping[str, str],
                groups: Mapping[str, list[str]], starts_exact) -> dict[str, dict[str, Unit]]:
    """Метки на пунктах для каждого запроса.

    named: запрос -> пункты, названные Банком России (маршрут «по пунктам»).
    Метка фрагмента, который засчитывается за один из названных пунктов,
    поглощается этим пунктом с оценкой 2. Остальные метки становятся
    единицей «фрагмент» или «куски одного пункта» с прежней оценкой.
    """
    out: dict[str, dict[str, Unit]] = {}
    for qid, rel in qrels.items():
        units: dict[str, Unit] = {}
        covered: set[str] = set()
        act = act_of_query.get(qid)
        for clause in named.get(qid, ()):
            hit = clause_chunks(act_chunks.get(act, ()), clause, groups, starts_exact)
            if hit:
                uid = f"{act}#п{clause}"
                units[uid] = Unit(uid, 2, hit, f"пункт {clause}, назван Банком России")
                covered |= hit
        for cid, score in rel.items():
            if score <= 0 or cid in covered:
                continue
            members = groups.get(cid, [cid])
            if len(members) > 1:
                uid = f"{cid.split('#')[0]}#куски{members[0].split('#')[1]}-{members[-1].split('#')[1]}"
                why = "куски одного длинного пункта"
            else:
                uid, why = cid, "фрагмент"
            unit = units.setdefault(uid, Unit(uid, score, set(members), why))
            unit.score = max(unit.score, score)
        out[qid] = units
    return out


def write_units(path: str, units: Mapping[str, Mapping[str, Unit]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("query-id\tpunkt-id\tscore\tchunk-ids\tosnovanie\n")
        for qid in sorted(units):
            for u in sorted(units[qid].values(), key=lambda u: u.unit_id):
                f.write(f"{qid}\t{u.unit_id}\t{u.score}\t{','.join(sorted(u.chunks))}\t{u.osnovanie}\n")


def read_units(path: str) -> dict[str, dict[str, Unit]]:
    out: dict[str, dict[str, Unit]] = collections.defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["query-id"]][row["punkt-id"]] = Unit(
                row["punkt-id"], int(row["score"]), set(row["chunk-ids"].split(",")), row["osnovanie"])
    return dict(out)


def expand(ranked: Sequence[str], units: Mapping[str, Unit], k: int) -> list[int]:
    """Первые k позиций выдачи пунктов: оценка пункта или 0 за промах."""
    by_chunk: dict[str, list[Unit]] = collections.defaultdict(list)
    for u in units.values():
        if u.score > 0:
            for cid in u.chunks:
                by_chunk[cid].append(u)
    seen: set[str] = set()
    out: list[int] = []
    for cid in ranked:
        hit = by_chunk.get(cid)
        if not hit:
            out.append(0)
        else:
            for u in sorted(hit, key=lambda u: (-u.score, u.unit_id)):
                if u.unit_id not in seen:
                    seen.add(u.unit_id)
                    out.append(u.score)
        if len(out) >= k:
            break
    return out[:k]


def metrics(ranked: Sequence[str], units: Mapping[str, Unit]) -> dict[str, float]:
    good = [u.score for u in units.values() if u.score > 0]
    if not good:
        return {"Recall@10": math.nan, "MRR@10": math.nan, "NDCG@10": math.nan}
    g = expand(ranked, units, 10)
    dcg = sum((2 ** s - 1) / math.log2(i + 1) for i, s in enumerate(g, start=1))
    ideal = sorted(good, reverse=True)[:10]
    idcg = sum((2 ** s - 1) / math.log2(i + 1) for i, s in enumerate(ideal, start=1))
    first = next((i for i, s in enumerate(g, start=1) if s > 0), None)
    return {"Recall@10": sum(1 for s in g if s > 0) / len(good),
            "MRR@10": 1 / first if first else 0.0, "NDCG@10": dcg / idcg}
