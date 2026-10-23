"""Пересчет чисел публичных текстов по файлам из `results/`.

Зачем. Числа в README набираются руками, и один раз в этом проекте в таблицу
уже попало значение, которого не было ни в одном замере. Этот скрипт берет
число из файла результата, форматирует его так, как оно должно стоять в тексте,
и падает, если такой строки в тексте нет. Правка замера без правки текста
ломает проверку, и наоборот.

Запуск: python scripts/verify_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ru(value: float, digits: int = 2) -> str:
    """Число так, как оно пишется в русском тексте: запятая вместо точки.

    Хвостовые нули не срезаются: в тексте стоит «215,0 МБ», и проверка должна
    требовать ровно эту запись, иначе она пропустит «215,04».
    """
    return f"{value:.{digits}f}".replace(".", ",")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def checks() -> list[tuple[str, str, str]]:
    """Список проверок: что проверяем, в каком файле лежит, что должно быть в тексте."""
    latency = load("latency-local.json")
    memory = load("memory.json")
    store = load("vector-store.json")
    store_container = load("vector-store-container.json")
    quant = load("quantization.json")
    repro = load("reproducibility.json")
    rehearsal = load("render-rehearsal.json")
    corpus = load("corpus.json")
    acceptance = load("acceptance.json")
    hosted = load("latency-hosted.json")
    rerank = load("rerank-quality.json")
    rerank_run = load("rerank-hybrid-rufin.json")
    first_run = load("corpus-update-first.json")
    load_report = load("load.json")
    warehouse = load("warehouse.json")
    runs = {(r["config"], r["users"]): r for r in load_report["runs"]}

    out: list[tuple[str, str, str]] = []
    for name, digits in (("bm25", 2), ("dense", 1), ("hybrid", 1)):
        row = latency["configs"][name]
        out.append(
            (f"задержка {name} p50", "latency-local.json", f"{ru(row['p50_ms'], digits)} мс")
        )
        out.append(
            (f"задержка {name} p95", "latency-local.json", f"{ru(row['p95_ms'], digits)} мс")
        )
    for name in ("bm25-rerank", "hybrid-rerank"):
        row = latency["configs"][name]
        out.append(
            (f"задержка {name} p50", "latency-local.json", f"{ru(row['p50_ms'] / 1000, 1)} с")
        )

    for name in ("bm25", "hybrid"):
        row = memory["configs"][name]
        out.append((f"пик памяти {name}", "memory.json", f"{ru(row['peak_mb'], 1)} МБ"))
        out.append((f"старт {name}", "memory.json", f"{ru(row['startup_seconds'], 2)} с"))

    for name in ("numpy", "faiss", "qdrant"):
        out.append(
            (
                f"задержка хранилища {name}",
                "vector-store.json",
                f"{ru(store['stores'][name]['p50_ms'], 1)} мс",
            )
        )
    out.append(
        ("память индекса", "vector-store.json", f"{ru(store['stores']['numpy']['index_mb'], 1)} МБ")
    )
    qdrant_server = store_container["stores"]["qdrant"]
    out.append(
        (
            "задержка Qdrant контейнером",
            "vector-store-container.json",
            f"{ru(qdrant_server['p50_ms'], 1)} мс",
        )
    )
    out.append(
        (
            "диск Qdrant контейнером",
            "vector-store-container.json",
            f"{ru(qdrant_server['index_mb'], 1)} МБ",
        )
    )
    out.append(
        (
            "память контейнера Qdrant",
            "vector-store-container.json",
            qdrant_server["container_memory"].replace("MiB", " МиБ").replace(".", ","),
        )
    )
    out.append(("память int8", "quantization.json", f"{ru(quant['memory']['int8_mb'], 1)} МБ"))
    out.append(
        (
            "потеря качества при квантизации",
            "quantization.json",
            f"{ru(quant['quality']['loss_mean'], 3)}",
        )
    )

    out.append(("актов в корпусе", "corpus.json", f"{corpus['acts']} акта"))
    out.append(("размер корпуса", "corpus.json", f"{ru(corpus['file_mib'], 1)} МиБ"))
    started = acceptance["started_at"]
    out.append(
        (
            "дата старта приемки",
            "acceptance.json",
            f"{started[8:10]}.{started[5:7]}.{started[:4]}",
        )
    )
    out.append(
        ("ссылка на первый плановый прогон", "acceptance.json", acceptance["first_scheduled_run"])
    )
    out.append(("задержка на хостинге p50", "latency-hosted.json", f"{ru(hosted['p50_ms'], 1)} мс"))
    out.append(("задержка на хостинге p95", "latency-hosted.json", f"{ru(hosted['p95_ms'], 1)} мс"))
    out.append(
        ("худший запрос на хостинге", "latency-hosted.json", f"{ru(hosted['max_ms'], 1)} мс")
    )
    out.append(
        (
            "старт на хостинге",
            "latency-hosted.json",
            f"{ru(hosted['health']['startup_seconds'], 1)} с",
        )
    )
    reranked = rerank["runs"]["hybrid-rerank, связка сервиса"]["metrics"]["NDCG@10"]
    gain = rerank["difference_ndcg10"]
    out.append(("качество связки с реранком", "rerank-quality.json", f"{ru(reranked, 3)}"))
    out.append(
        (
            "прибавка реранка",
            "rerank-quality.json",
            f"{ru(gain['mean'], 3)} NDCG@10 с интервалом [{ru(gain['lo'], 3)}; {ru(gain['hi'], 3)}]",
        )
    )
    out.append(
        (
            "пар в переранжировании",
            "rerank-hybrid-rufin.json",
            f"{rerank_run['pairs']:,}".replace(",", " "),
        )
    )
    out.append(
        (
            "время переранжирования",
            "rerank-hybrid-rufin.json",
            f"{ru(rerank_run['seconds'], 0)} с".replace(",", ""),
        )
    )
    out.append(
        (
            "скорость переранжирования",
            "rerank-hybrid-rufin.json",
            f"{ru(rerank_run['pairs_per_second'], 0)} парах".replace(",", ""),
        )
    )
    out.append(
        (
            "видеопамять переранжирования",
            "rerank-hybrid-rufin.json",
            f"{ru(rerank_run['peak_vram_mb'], 1)} МБ",
        )
    )
    # Числа первого прогона регламента: они описаны в README как замер на дату,
    # поэтому сверяются с файлом того прогона, а не с живым состоянием.
    out.append(
        ("публикаций в перечне cbr.ru", "corpus-update-first.json", f"{first_run['index_acts']}")
    )
    out.append(
        (
            "актов под состав корпуса",
            "corpus-update-first.json",
            f"{first_run['wanted_acts']} актов",
        )
    )
    out.append(
        ("время первого прогона", "corpus-update-first.json", f"{ru(first_run['seconds'], 1)} с")
    )
    out.append(
        ("фрагментов в корпусе", "corpus.json", f"{corpus['chunks']:,} фрагмента".replace(",", " "))
    )
    out.append(
        ("сборка в репетиции", "render-rehearsal.json", f"{ru(rehearsal['build_seconds'], 1)} с")
    )
    out.append(
        ("старт в репетиции", "render-rehearsal.json", f"{ru(rehearsal['startup_seconds'], 1)} с")
    )
    out.append(
        (
            "пик памяти в репетиции",
            "render-rehearsal.json",
            f"{ru(rehearsal['peak_memory_mb'], 1)} МБ из 512",
        )
    )
    out.append(
        (
            "запас памяти до потолка",
            "render-rehearsal.json",
            f"{ru(512 - rehearsal['peak_memory_mb'], 1)} МБ",
        )
    )

    for name in ("bm25", "hybrid"):
        row = repro["configs"][name]
        out.append(
            (
                f"воспроизведение {name}, верхний результат",
                "reproducibility.json",
                f"{row['top1']} из {row['queries']}",
            )
        )
        out.append(
            (
                f"воспроизведение {name}, состав десятки",
                "reproducibility.json",
                f"{row['composition']} из {row['queries']}",
            )
        )

    sample = warehouse["events"]
    out.append(("строк в наборе сверки", "warehouse.json", f"{sample['lines']} строк"))
    out.append(
        ("событий запроса в наборе", "warehouse.json", f"{sample['query_events']} событий запроса")
    )
    out.append(("оценок в наборе", "warehouse.json", f"{sample['feedback_events']} оценки"))
    out.append(
        (
            "оценок без своего запроса",
            "warehouse.json",
            f"{sample['orphan_feedback']} из {sample['feedback_events']}",
        )
    )
    out.append(("запросов витрины", "warehouse.json", f"витрины {len(warehouse['queries'])}"))
    out.append(
        (
            "движки в сверке режимов",
            "warehouse.json",
            f"chdb {warehouse['free_mode']['version']} против ClickHouse "
            f"{warehouse['build_mode']['version']}",
        )
    )

    for (config, users), row in runs.items():
        search = row["endpoints"]["/search"]
        out.append(
            (
                f"нагрузка {config} на {users} пользователях, RPS",
                "load.json",
                f"{ru(search['rps'], 1)}",
            )
        )
    return out


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for what, source, expected in checks():
        if expected not in readme:
            missing.append((what, source, expected))
    if missing:
        print("README расходится с замерами:", file=sys.stderr)
        for what, source, expected in missing:
            print(f"  {what}: в {source} стоит {expected}, в README такого нет", file=sys.stderr)
        return 1
    print(f"числа README сошлись с замерами, проверок {len(checks())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
