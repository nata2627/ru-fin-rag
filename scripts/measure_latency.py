#!/usr/bin/env python3
"""Замеры задержки поиска — на маке, где система и работает.

Всё остальное тяжёлое считается на видеокарте Kaggle, но задержку там мерить
бессмысленно: пользователь работает не на T4. Поэтому замеры снимаются здесь,
и в отчёте указывается, на каком железе.

Методика: три прогона на прогрев, затем десять замеров на каждый запрос,
из которых берётся медиана; по запросам считаются медиана и 95-й перцентиль.
Прогрев нужен, чтобы не измерять загрузку весов вместо самого поиска.

Память. Конфигурации меряются по одной и модели выгружаются между ними:
на восьми гигабайтах два энкодера одновременно не помещаются. Перед каждым
шагом печатается пик памяти процесса — если он подбирается к четырём
гигабайтам, шаг лучше запустить отдельным вызовом через --only.

Про готовые векторы запросов (--query-vectors). Веса модели нужны только
чтобы закодировать запрос, и для моделей, которых нет на этой машине,
кодирование мерить нечем: качать два гигабайта незачем, а считать запросы
другой моделью, чем корпус, нельзя — это разные пространства. Зато поиск
по индексу от весов не зависит: с готовыми векторами, посчитанными там же,
где векторы корпуса, он мерится как есть. Кодирование запроса при этом
не входит в число и мерится отдельным шагом (--encoder), потому что
от размерности матрёшки оно не зависит вовсе.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.dense import truncate  # noqa: E402
from rufin.retrieval.hybrid import rrf  # noqa: E402
from rufin.retrieval.model_specs import MODELS, truncated_family  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
QDIR = os.path.join(ROOT, "data", "queries")
RAW = os.path.join(ROOT, "docs", "raw")

TOP_RETRIEVE = 50
TOP_REPORT = 10
STEPS = ("bm25", "dense", "hybrid", "rerank", "encode")


def peak_gb() -> float:
    """Пик памяти процесса. На macOS ru_maxrss возвращается в байтах."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1073741824 if sys.platform == "darwin" else raw / 1048576


def machine() -> str:
    return f"{platform.system()} {platform.machine()}, {platform.processor() or 'arm64'}"


def measure(fn, keys: list[str], warmup: int = 3, runs: int = 10, sample: int = 20,
            seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(sample, len(keys)), replace=False)]
    for _ in range(warmup):
        for k in pick:
            fn(k)
    per_query = []
    for k in pick:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(k)
            times.append((time.perf_counter() - t0) * 1000)
        per_query.append(statistics.median(times))
    per_query.sort()
    i95 = min(len(per_query) - 1, int(round(0.95 * (len(per_query) - 1))))
    return {"median_ms": round(statistics.median(per_query), 1),
            "p95_ms": round(per_query[i95], 1), "queries": len(pick),
            "runs": runs, "warmup": warmup}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--dense", default="bge-m3", help="модель для плотного поиска и гибрида")
    ap.add_argument("--family", action="store_true",
                    help="пройти ещё и урезанные размерности матрёшки той же "
                         "модели: те же веса, другое truncate_dim, и индекс "
                         "каждой собран `make matryoshka`")
    ap.add_argument("--model-path", default=None,
                    help="откуда брать веса вместо пути из model_specs. Нужно "
                         "дообученной модели до публикации на HuggingFace: "
                         "задержка меряется по весам, приехавшим с видеокарты, "
                         "а не по тому, чего ещё нет в сети")
    ap.add_argument("--query-vectors", default=None,
                    help="готовые векторы запросов (.npy) и рядом файл .ids.txt; "
                         "с ними модель не загружается и кодирование запроса "
                         "в задержку поиска не входит")
    ap.add_argument("--encoder", default=None, choices=sorted(MODELS),
                    help="чьи веса мерить на шаге «encode». Шаг отдельный, "
                         "потому что от размерности матрёшки кодирование "
                         "запроса не зависит")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--only", nargs="*", default=None, choices=STEPS,
                    help="что мерить: " + " ".join(STEPS))
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--backend", default=None, choices=("faiss", "numpy"),
                    help="чем искать по матрице. По умолчанию FAISS, а если "
                         "в замер входит шаг с моделью, то numpy: FAISS и torch "
                         "в одном процессе на маке роняют его по SIGSEGV")
    ap.add_argument("--tag", default="",
                    help="приписка к имени файлов отчёта. Нужна, когда по одной "
                         "модели снимаются разные наборы шагов: иначе второй "
                         "замер молча затирает первый")
    ap.add_argument("--threads", type=int, default=0,
                    help="сколько потоков отдать FAISS. Ноль это по умолчанию, "
                         "то есть все ядра машины. Один нужен, чтобы знать "
                         "задержку там, где ядро одно: бесплатный хостинг "
                         "и восьмиядерный мак это разные машины")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    ids = [c["chunk_id"] for c in chunks]
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    qtext = {q["query_id"]: q["text"] for q in queries}
    qids = list(qtext)

    def want(name: str) -> bool:
        return args.only is None or name in args.only

    report: list[str] = []
    result: dict = {"machine": machine(), "chunks": args.chunks, "chunk_count": len(ids),
                    "latency": {}, "index": {}}

    def say(s: str = "") -> None:
        print(s, flush=True)
        report.append(s)

    say(f"машина: {machine()}, ядер {os.cpu_count()}")
    if args.threads:
        import faiss
        faiss.omp_set_num_threads(args.threads)
        result["faiss_threads"] = args.threads
        say(f"FAISS ограничен {args.threads} потоками")
    say(f"нарезка: {args.chunks}, фрагментов {len(ids)}, запросов {len(qids)}")
    say()

    place: dict[str, int] = {}
    qvec_all = None
    if args.query_vectors:
        qvec_all = np.load(args.query_vectors).astype("float32")
        order = open(args.query_vectors.replace(".npy", ".ids.txt"),
                     encoding="utf-8").read().split()
        place = {qid: i for i, qid in enumerate(order)}
        qids = [q for q in qids if q in place]
        if not qids:
            raise SystemExit(f"ни для одного запроса из {args.queries} нет готового "
                             f"вектора в {args.query_vectors}")
        say(f"векторы запросов готовые: {os.path.relpath(args.query_vectors, ROOT)}, "
            f"{qvec_all.shape[0]} × {qvec_all.shape[1]}; модель не загружается, "
            f"кодирование запроса в задержку поиска не входит")
        say(f"запросов с готовым вектором: {len(qids)} из {len(qtext)}")
        result["query_vectors"] = os.path.relpath(args.query_vectors, ROOT)
        say()

    # FAISS и torch в одном процессе на маке падают по SIGSEGV без traceback,
    # две сборки libomp (dense.py). Проверено здесь же: шаг rerank с индексом
    # на FAISS уронил процесс с кодом 139. Поэтому любой замер, в который
    # входит модель, идёт на numpy, и бэкенд печатается в отчёт.
    с_моделью = want("rerank") or qvec_all is None
    backend = args.backend or ("numpy" if с_моделью else "faiss")
    result["backend"] = backend

    bm25 = None
    if want("bm25") or want("hybrid") or want("rerank"):
        bm25 = BM25Index.build(ids, [texts[i] for i in ids])
        result["index"]["bm25"] = {"build_seconds": round(bm25.build_seconds, 1),
                                   "size_mb": round(bm25.size_mb, 1),
                                   "vocab": len(bm25.vocab)}
        say(f"BM25: построен за {bm25.build_seconds:.0f} с, индекс {bm25.size_mb:.0f} МБ, "
            f"словарь {len(bm25.vocab)} слов, пик памяти {peak_gb():.2f} ГБ")
        if want("bm25"):
            result["latency"]["bm25"] = measure(
                lambda k: bm25.search(qtext[k], TOP_REPORT), qids, sample=args.sample)

    # Кодирование запроса. Шаг отдельный от поиска и считается один раз:
    # от размерности матрёшки он не зависит (обрезка идёт после модели),
    # а весов меряемой модели на этой машине может не быть вовсе.
    if want("encode") and args.encoder:
        from rufin.retrieval.dense import Encoder
        enc_spec = MODELS[args.encoder]
        if args.model_path:
            enc_spec = dataclasses.replace(enc_spec, path=args.model_path)
        enc = Encoder(enc_spec)
        say(f"кодирование запроса: {enc_spec.path} на {enc.device}, "
            f"половинная точность {enc.half}, пик памяти {peak_gb():.2f} ГБ")
        result["index"]["encoder"] = {"model": enc_spec.path, "device": enc.device,
                                      "half": enc.half, "dim": enc_spec.dim}
        result["latency"][f"encode:{args.encoder}"] = measure(
            lambda k: enc.encode([qtext[k]], is_query=True), qids, sample=args.sample)
        enc = None
        gc.collect()

    models = [args.dense]
    if args.family:
        models += [m for m in truncated_family(args.dense) if m != args.dense]
    for имя in models:
        if not (want("dense") or want("hybrid") or want("rerank")):
            break
        # Урезанные размерности матрёшки это та же модель с другим полем
        # truncate_dim, и путь к весам у них общий: подмена пути должна
        # распространяться на все записи сразу.
        spec = MODELS[имя]
        if args.model_path:
            spec = dataclasses.replace(spec, path=args.model_path)
            say(f"веса взяты из {args.model_path} вместо {MODELS[имя].path}")
        emb = os.path.join(EMBDIR, args.chunks, имя)
        if not os.path.exists(os.path.join(emb, "vectors.npy")):
            say(f"матрицы эмбеддингов нет ({args.chunks}/{имя}): плотный поиск "
                f"и гибрид по этой размерности не меряются")
            say("   урезанные размерности собирает `make matryoshka`, полную "
                "считает карта")
            continue
        from rufin.retrieval.dense import DenseIndex, Encoder
        путь = os.path.join(emb, "vectors.npy")
        на_диске = os.path.getsize(путь) / 1048576
        dtype_диска = str(np.load(путь, mmap_mode="r").dtype)
        index = DenseIndex.from_files(spec, путь, os.path.join(emb, "ids.txt"),
                                      backend=backend, float32=backend == "numpy")
        dim = int(index.vectors.shape[1])
        # Два разных числа, и в отчёт идут оба: на диске матрица лежит
        # в float16, а под поиск и FAISS, и numpy держат float32.
        запись = {"disk_mb": round(на_диске, 1), "dtype_disk": dtype_диска,
                  "memory_mb": round(index.vectors.shape[0] * dim * 4 / 1048576, 1),
                  "dim": dim, "backend": index.backend}
        enc = None
        if qvec_all is None:
            enc = Encoder(spec)
            запись["device"] = enc.device
        result["index"][f"dense:{имя}"] = запись
        say(f"плотный поиск {имя}: размерность {dim}, матрица на диске "
            f"{на_диске:.1f} МБ ({dtype_диска}), в памяти под поиск "
            f"{запись['memory_mb']:.1f} МБ, бэкенд {index.backend}, "
            f"пик памяти процесса {peak_gb():.2f} ГБ")

        def qvec(k: str, _spec=spec, _enc=enc) -> np.ndarray:
            if qvec_all is None:
                return _enc.encode([qtext[k]], is_query=True)
            return truncate(qvec_all[place[k]:place[k] + 1], _spec.truncate_dim)

        if want("dense"):
            result["latency"][f"dense:{имя}"] = measure(
                lambda k: index.search_vectors(qvec(k), TOP_REPORT),
                qids, sample=args.sample)
        if want("hybrid"):
            result["latency"][f"hybrid:{имя}"] = measure(
                lambda k: rrf([bm25.search(qtext[k], TOP_RETRIEVE),
                               index.search_vectors(qvec(k), TOP_RETRIEVE)[0]],
                              top=TOP_RETRIEVE),
                qids, sample=args.sample)
        if want("rerank"):
            from rufin.retrieval.rerank import Reranker
            rr = Reranker(model_path=args.rerank_model)
            say(f"реранкер {rr.model_path} на {rr.device}, "
                f"пик памяти {peak_gb():.2f} ГБ")

            def hybrid_then_rerank(k: str):
                cand = rrf([bm25.search(qtext[k], TOP_RETRIEVE),
                            index.search_vectors(qvec(k), TOP_RETRIEVE)[0]],
                           top=TOP_RETRIEVE)
                return rr.rerank(qtext[k], cand, texts, top=TOP_REPORT)

            # реранкер на порядок дороже поиска, поэтому выборка меньше
            result["latency"][f"hybrid+rerank:{имя}"] = measure(
                hybrid_then_rerank, qids, sample=min(args.sample, 10))
            result["index"]["reranker"] = {"model": rr.model_path, "device": rr.device,
                                           "candidates": TOP_RETRIEVE}
            rr = None
            gc.collect()
        index = enc = None
        gc.collect()

    say()
    say(f"{'конфигурация':<22} {'медиана, мс':>12} {'p95, мс':>10}   методика")
    for cfg, lat in result["latency"].items():
        say(f"{cfg:<22} {lat['median_ms']:>12.1f} {lat['p95_ms']:>10.1f}   "
            f"{lat['queries']} запросов, {lat['warmup']} прогрева + {lat['runs']} замеров")
    say()
    say("индексы:")
    for k, v in result["index"].items():
        say(f"   {k:<22} {v}")
    result["peak_memory_gb"] = round(peak_gb(), 2)
    say(f"\nпик памяти процесса: {result['peak_memory_gb']:.2f} ГБ")

    хвост = f"{args.chunks}_{args.dense}"
    if args.tag:
        хвост += f"_{args.tag}"
    if args.threads:
        хвост += f"_{args.threads}thr"
    with open(os.path.join(ROOT, "data", "results", f"latency_{хвост}.json"),
              "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RAW, f"latency_{хвост}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    say(f"сырой вывод: docs/raw/latency_{хвост}.txt")


if __name__ == "__main__":
    main()
