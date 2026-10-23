#!/usr/bin/env python3
"""Контрольный прогон: наш код на чужих данных.

Вопрос один. На нормативке Банка России наш поиск дает 0,484 NDCG@10 у BM25
и 0,421 у лучшей связки на живых вопросах, а в наборе того же жанра у соседа
по тем же метрикам выходит 0,670 и 0,859. Это разница предметной области или
ошибка в нашем коде? Проверяется прямо: тот же наш BM25, те же наши метрики,
но на его корпусе, его запросах и его разметке.

Набор `alekseevpavel04/ru-law-retrieval` в формате MTEB (`fetch_friend_set.py`).
Его протокол «chunk»: индексируются фрагменты, разметка стоит на статьях,
поэтому выдача фрагментов поднимается к статье и повторы снимаются. Это
ровно тот подъем к родителю, что в `rufin.parents`, только родитель здесь
назван в самом фрагменте полем `doc_id`, и считать пересечения не нужно.

Рядом считается протокол «article»: индексируются статьи целиком. Разница
двух протоколов на одних данных это цена нарезки у него, и она сравнима с
нашим замером нарезки (документ 13).

Его опубликованные числа берутся из карточки набора и помечены как чужие:
наш код их не считал.

Ступень задается ключом `--dense <модель>`: вместо BM25 берутся готовые
векторы его фрагментов, его статей и его запросов, посчитанные на видеокарте
(`scripts/embed_chunks.py`). Все остальное у двух половин общее построчно:
протоколы, подъем к статье, метрики, разбивка по типам вопросов и сравнение
с его таблицей. Плотная половина отвечает на отдельный вопрос: у плотной
ступени токенизации нет, поэтому если отставание нашего BM25 на его данных
объясняется отсутствием нормализации формы слова, то здесь его быть не должно.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.dense import DenseIndex  # noqa: E402
from rufin.retrieval.model_specs import MODELS  # noqa: E402
from rufin.retrieval.text import tokenize, truncate_tokens  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRIEND = os.path.join(ROOT, "data", "friend")
OURS = os.path.join(ROOT, "data")
EMB = os.path.join(FRIEND, "embeddings")
REPORT = os.path.join(ROOT, "docs", "raw", "friend_control.txt")
RESULT = os.path.join(ROOT, "data", "results", "friend-control.json")

DEEP = 200
TOP = 10
SPLITS = ("test", "golden", "tk_hard")

# Чужие числа из карточки набора huggingface.co/datasets/alekseevpavel04/
# ru-law-retrieval, протокол chunk, сплит test. Нашим кодом не считались и
# служат только точкой сравнения для строки bm25.
HIS = {
    "bm25": 0.670,
    "multilingual-e5-small-ru-law (дообученная им)": 0.859,
    "e5-large-instruct": 0.865,
    "FRIDA": 0.878,
}
# Его же BM25 в разбивке, из README репозитория ru-law-retrieval. Разбивка
# важнее уровня: она показывает, где именно наш BM25 отстает.
HIS_BY_TYPE = {"everyday": 0.265, "search": 0.822, "legal": 0.922}
HIS_TK_HARD = 0.480
# Какие наши модели есть в его таблице под тем же именем. Только для них
# сравнение идет напрямую: это одни и те же веса, запущенные двумя
# реализациями, и расхождение читается как расхождение кода.
SAME_MODEL = {"frida": "FRIDA"}

# Две токенизации. Первая рабочая, вторая диагностическая: она отвечает на
# один вопрос, в морфологии ли дело.
TOKENIZERS = {"как есть": tokenize, "обрезка до 6": truncate_tokens}


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for rec in read_jsonl(path):
        score = int(rec["score"])
        if score > 0:
            out[rec["query-id"]][rec["corpus-id"]] = score
    return dict(out)


def dense_hits(index: DenseIndex, folder: str, qids: list[str], batch: int = 200
               ) -> dict[str, list[str]]:
    """Выдача плотного поиска на глубину DEEP по готовым векторам запросов."""
    vectors = np.load(os.path.join(folder, "vectors.npy"))
    order = [l.rstrip("\n") for l in open(os.path.join(folder, "ids.txt"),
                                          encoding="utf-8")]
    place = {qid: i for i, qid in enumerate(order)}
    missing = [q for q in qids if q not in place]
    if missing:
        raise SystemExit(f"нет векторов для {len(missing)} запросов в {folder}, "
                         f"первый {missing[0]}")
    qvec = np.stack([vectors[place[q]] for q in qids]).astype("float32")
    out: dict[str, list[str]] = {}
    for start in range(0, len(qids), batch):
        part = qids[start:start + batch]
        for qid, hits in zip(part, index.search_vectors(qvec[start:start + batch],
                                                        k=DEEP)):
            out[qid] = [cid for cid, _ in hits]
    return out


def run_split(hits_of, parent_of: dict[str, str] | None,
              qrels: dict[str, dict[str, int]]):
    """Выдача по сплиту: поиск, подъем к статье, если он нужен, снятие повторов."""
    run = {}
    for qid in qrels:
        hits = hits_of(qid)
        if parent_of is None:
            run[qid] = hits[:TOP]
            continue
        out, seen = [], set()
        for cid in hits:
            doc = parent_of[cid]
            if doc in seen:
                continue
            seen.add(doc)
            out.append(doc)
            if len(out) == TOP:
                break
        run[qid] = out
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--dense", default=None, choices=sorted(MODELS),
                    help="плотная ступень на готовых векторах вместо BM25")
    ap.add_argument("--embeddings", default=EMB,
                    help="корень папок <часть>/<модель>/vectors.npy")
    args = ap.parse_args()

    stage = f"dense-{args.dense}" if args.dense else "bm25"
    report_path = REPORT if not args.dense else REPORT.replace(".txt", f"_{stage}.txt")
    result_path = RESULT if not args.dense else RESULT.replace(".json", f"-{stage}.json")

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    chunks = list(read_jsonl(os.path.join(FRIEND, "chunks", "chunks.jsonl")))
    corpus = list(read_jsonl(os.path.join(FRIEND, "corpus", "corpus.jsonl")))
    say("контрольный прогон нашего кода на наборе alekseevpavel04/ru-law-retrieval")
    say(f"его корпус: статей {len(corpus)}, фрагментов {len(chunks)}")

    parent_of = {c["_id"]: c["doc_id"] for c in chunks}
    say(f"ступень: {stage}")
    indexes = {}
    if args.dense:
        # Имена фрагментов берутся из его файлов, а не из файла векторов:
        # если порядок строк матрицы разошелся с его набором, вектор достанется
        # чужому фрагменту, и по метрикам это выглядело бы как плохое качество.
        for protocol, part, ids in (("chunk", "friend-chunks", [c["_id"] for c in chunks]),
                                    ("article", "friend-articles", [d["_id"] for d in corpus])):
            folder = os.path.join(args.embeddings, part, args.dense)
            index = DenseIndex.from_files(MODELS[args.dense],
                                          os.path.join(folder, "vectors.npy"),
                                          os.path.join(folder, "ids.txt"),
                                          backend="numpy")
            if index.ids != ids:
                raise SystemExit(f"{part}: имена в {folder}/ids.txt разошлись с "
                                 f"его набором ({len(index.ids)} против {len(ids)})")
            index.vectors = index.vectors.astype("float32")
            indexes[(protocol, args.dense)] = (index, parent_of if protocol == "chunk" else None)
        for (protocol, vname), (index, _) in indexes.items():
            say(f"векторы, протокол {protocol}, модель «{vname}»: "
                f"{index.size_mb:.0f} МБ, {index.vectors.shape[1]} измерений")
    else:
        for tname, tok in TOKENIZERS.items():
            indexes[("chunk", tname)] = (
                BM25Index.build([c["_id"] for c in chunks], [c["text"] for c in chunks],
                                tokenizer=tok), parent_of)
            indexes[("article", tname)] = (
                BM25Index.build([d["_id"] for d in corpus],
                                [f"{d['title']}\n\n{d['text']}" for d in corpus],
                                tokenizer=tok), None)
        for (protocol, tname), (index, _) in indexes.items():
            say(f"индекс BM25, протокол {protocol}, токенизация «{tname}»: "
                f"{index.size_mb:.0f} МБ, слов в словаре {len(index.vocab)}")

    results: dict[str, dict] = {}
    say()
    for split in args.splits:
        queries = {q["_id"]: q["text"]
                   for q in read_jsonl(os.path.join(FRIEND, "queries", f"{split}.jsonl"))}
        qrels = load_qrels(os.path.join(FRIEND, "qrels", f"{split}.jsonl"))
        missing = [q for q in qrels if q not in queries]
        if missing:
            raise SystemExit(f"{split}: нет текста запроса у {len(missing)}, первый {missing[0]}")
        golds = [len(v) for v in qrels.values()]
        say(f"сплит {split}: запросов с разметкой {len(qrels)}, "
            f"эталонов на запрос медиана {statistics.median(golds):.0f}, "
            f"максимум {max(golds)}")
        results[split] = {"запросов": len(qrels),
                          "эталонов на запрос, медиана": statistics.median(golds)}
        qtype = {q["_id"]: q.get("qtype", "")
                 for q in read_jsonl(os.path.join(FRIEND, "queries", f"{split}.jsonl"))}
        kept: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for (protocol, tname), (index, parents) in indexes.items():
            if args.dense:
                deep = dense_hits(index, os.path.join(
                    args.embeddings, f"friend-queries-{split}", args.dense),
                    sorted(qrels))
                hits_of = deep.__getitem__
            else:
                hits_of = lambda qid: [c for c, _ in index.search(queries[qid], DEEP)]  # noqa: E731
            run = run_split(hits_of, parents, qrels)
            per_q = M.per_query(run, qrels)
            kept[(protocol, tname)] = per_q
            qids = list(per_q["_qids"])
            mrr5 = np.array([M.mrr_at_k(run[q], qrels[q], 5) for q in qids])
            cis = {m: M.bootstrap_ci(per_q[m]) for m in ("Recall@10", "NDCG@10")}
            cis["MRR@5"] = M.bootstrap_ci(mrr5)
            say(f"    {protocol:<8} «{tname:<12}» "
                + "   ".join(f"{k} {v}" for k, v in cis.items()))
            results[split][f"{protocol}, {tname}"] = {
                k: {"mean": round(v.mean, 4), "lo": round(v.lo, 4), "hi": round(v.hi, 4)}
                for k, v in cis.items()}
            # разбивка по типам запроса: у него она опубликована, и именно
            # в ней видно, где расходятся реализации
            by_type = {}
            for t in sorted({qtype[q] for q in qids if qtype[q]}):
                part = [q for q in qids if qtype[q] == t]
                vals = np.array([M.ndcg_at_k(run[q], qrels[q], 10) for q in part])
                ci = M.bootstrap_ci(vals)
                by_type[t] = {"запросов": len(part), "mean": round(ci.mean, 4),
                              "lo": round(ci.lo, 4), "hi": round(ci.hi, 4)}
            if by_type:
                say("             по типам: " + "   ".join(
                    f"{t} {v['mean']:.3f} ({v['запросов']})" for t, v in by_type.items()))
                results[split][f"{protocol}, {tname}"]["NDCG@10 по типам"] = by_type

        # Его протокол chunk против протокола article на одних запросах это
        # тот же подъем к родителю, что меряет документ 13, только на его
        # данных и его нарезке. Сравнение парное: запросы те же самые.
        for vname in sorted({v for _, v in indexes}):
            pair = (("chunk", vname), ("article", vname))
            if not all(k in kept for k in pair):
                continue
            diff, pval = M.paired_diff_ci(kept[pair[0]]["NDCG@10"],
                                          kept[pair[1]]["NDCG@10"])
            mark = " *" if diff.lo * diff.hi > 0 else ""
            say(f"    подъем к статье против статей целиком, «{vname}»: "
                f"NDCG@10 {diff} p={pval:.3f}{mark}")
            results[split][f"chunk против article, {vname}"] = {
                "diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                "hi": round(diff.hi, 4), "p": round(pval, 4)}

    # Сравнение с его опубликованными числами идет только там, где он
    # публиковал ту же связку на том же сплите: BM25, протокол chunk, test.
    say()
    say("его опубликованные числа (карточка набора и README, протокол chunk, сплит test):")
    for name, value in HIS.items():
        say(f"    {name:<46} NDCG@10 {value:.3f}")
    say("    его BM25 по типам: " + "   ".join(
        f"{t} {v:.3f}" for t, v in HIS_BY_TYPE.items()) + f";  tk_hard {HIS_TK_HARD:.3f}")
    if args.dense:
        # Плотная ступень сравнивается с его числом только там, где это одна
        # и та же модель. Для остальных его плотных строк наш код считал
        # другие веса, и расхождение читалось бы как расхождение моделей.
        ours = results.get("test", {}).get(f"chunk, {args.dense}", {}
                                           ).get("NDCG@10", {}).get("mean")
        his_key = SAME_MODEL.get(args.dense)
        if ours is not None and his_key:
            say(f"наш {args.dense}: NDCG@10 {ours:.3f} против его "
                f"{HIS[his_key]:.3f} на той же модели, расхождение "
                f"{ours - HIS[his_key]:+.3f}; лексическое расхождение было "
                f"{0.602 - HIS['bm25']:+.3f}")
        elif ours is not None:
            say(f"наш {args.dense}: NDCG@10 {ours:.3f}. В его таблице этой "
                f"модели нет, прямое сравнение не считается")
    else:
        for tname in TOKENIZERS:
            key = f"chunk, {tname}"
            ours = results.get("test", {}).get(key, {}).get("NDCG@10", {}).get("mean")
            if ours is None:
                continue
            say(f"наш BM25, токенизация «{tname}»: NDCG@10 {ours:.3f} против его "
                f"{HIS['bm25']:.3f}, расхождение {ours - HIS['bm25']:+.3f}")

    # Различия корпусов. Считаются здесь же по обоим наборам, чтобы таблица
    # в документе не переписывалась руками.
    say()
    our_chunks = list(read_jsonl(os.path.join(OURS, "chunks", "base.jsonl")))
    our_qrels = read_qrels(os.path.join(OURS, "queries", "qrels.tsv"))
    our_golds = {q: sum(1 for g in rel.values() if g > 0) for q, rel in our_qrels.items()}
    diff = {
        "документов (акты против статей)": (len({c["act_id"] for c in our_chunks}), len(corpus)),
        "фрагментов": (len(our_chunks), len(chunks)),
        "символов во фрагменте, медиана": (
            int(statistics.median(len(c["text"]) for c in our_chunks)),
            int(statistics.median(len(c["text"]) for c in chunks))),
        "слов во фрагменте после токенизации, медиана": (
            int(statistics.median(len(tokenize(c["text"])) for c in our_chunks)),
            int(statistics.median(len(tokenize(c["text"])) for c in chunks))),
        "запросов с разметкой": (len(our_golds), results.get("test", {}).get("запросов", 0)),
        "эталонов на запрос, медиана": (
            statistics.median(our_golds.values()),
            results.get("test", {}).get("эталонов на запрос, медиана", 0)),
    }
    say(f"{'различие':<46} {'наш корпус':>12} {'его корпус':>12}")
    for name, (a, b) in diff.items():
        say(f"{name:<46} {a:>12} {b:>12}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"набор": "alekseevpavel04/ru-law-retrieval", "ступень": stage,
                   "глубина чтения": DEEP, "позиций в выдаче": TOP,
                   "его числа из карточки": HIS,
                   "наши числа": results,
                   "различия корпусов": {k: {"наш": v[0], "его": v[1]}
                                         for k, v in diff.items()}},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(report_path, ROOT)}, "
        f"{os.path.relpath(result_path, ROOT)}")


if __name__ == "__main__":
    main()
