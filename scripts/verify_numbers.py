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
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import dashboard  # noqa: E402


def ru(value: float, digits: int = 2) -> str:
    """Число так, как оно пишется в русском тексте: запятая вместо точки.

    Хвостовые нули не срезаются: в тексте стоит «215,0 МБ», и проверка должна
    требовать ровно эту запись, иначе она пропустит «215,04».
    """
    return f"{value:.{digits}f}".replace(".", ",")


def ru_spaced(value: float, digits: int = 1) -> str:
    """То же, но с пробелом в разрядах от десяти тысяч.

    Порог ровно там, где он стоит в текстах проекта: «1554 акта» без пробела,
    «62 594 фрагмента» с пробелом.
    """
    text = f"{value:,.{digits}f}" if abs(value) >= 10_000 else f"{value:.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def gain_text(diff: dict) -> str:
    """Разница со своим интервалом так, как она пишется в тексте проекта.

    Одной записью на все прибавки нарочно: разных записей было бы столько же,
    сколько строк в README, и разойтись они могли бы незаметно.
    """
    return f"+{ru(diff['mean'], 3)} [{ru(diff['lo'], 3)}; {ru(diff['hi'], 3)}]"


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
    rerank_bm25 = load("rerank-quality-bm25.json")
    first_stage = load("rerank-first-stage.json")
    rerank_run = load("rerank-hybrid-rufin.json")
    vectors = load("vectors.json")
    first_run = load("corpus-update-first.json")
    load_report = load("load.json")
    warehouse = load("warehouse.json")
    mart = load("dbt.json")
    drain = load("events-rehearsal.json")
    logging_report = load("logging.json")
    page = load("dashboard.json")
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
    if acceptance["broken_by"]:
        # Прогон, который порвал цепочку, назван в README по ссылке. Проверка
        # держит README и файл вместе: иначе рассказ про приемку легко оставить
        # прежним, когда счет уже другой.
        out.append(("прогон, порвавший цепочку", "acceptance.json", acceptance["broken_by"]["run"]))
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
            "первая ступень под реранком",
            "rerank-quality.json",
            f"{ru(rerank['runs']['hybrid, связка сервиса']['metrics']['NDCG@10'], 3)}",
        )
    )
    out.append(("прибавка реранка", "rerank-quality.json", gain_text(gain)))
    # Замененный замер стоит в README рядом с новым, и берется он из того же
    # файла полем `replaced`, а не из памяти автора.
    was = rerank["replaced"]
    out.append(
        (
            "прежнее качество связки с реранком",
            "rerank-quality.json",
            f"{ru(was['runs']['hybrid-rerank, связка сервиса'], 3)}",
        )
    )
    out.append(
        (
            "прежняя первая ступень под реранком",
            "rerank-quality.json",
            f"{ru(was['runs']['hybrid, связка сервиса'], 3)}",
        )
    )
    out.append(
        ("прежняя прибавка реранка", "rerank-quality.json", gain_text(was["difference_ndcg10"]))
    )
    out.append(
        (
            "качество реранка поверх BM25",
            "rerank-quality-bm25.json",
            f"{ru(rerank_bm25['runs']['bm25-rerank, связка сервиса']['metrics']['NDCG@10'], 3)}",
        )
    )
    out.append(
        (
            "прибавка реранка поверх BM25",
            "rerank-quality-bm25.json",
            gain_text(rerank_bm25["difference_ndcg10"]),
        )
    )
    out.append(
        (
            "цена гибридной первой ступени после реранка",
            "rerank-first-stage.json",
            gain_text(first_stage["difference_ndcg10"]),
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
        out.append(
            (
                f"воспроизведение {name}, порядок десятки",
                "reproducibility.json",
                f"{row['order']} из {row['queries']}",
            )
        )

    # Догон векторов до выросшего корпуса. Проверяется и согласие с матрицей
    # части А: без него дописанный хвост может оказаться из другого
    # пространства, и по метрикам поиска на двенадцати фрагментах это не видно.
    agreement = vectors["agreement_with_part_a"]
    out.append(
        ("дописано векторов", "vectors.json", f"недостающие {vectors['added']} это один акт")
    )
    out.append(
        ("выборка сверки векторов", "vectors.json", f"{agreement['sample']} уже посчитанных")
    )
    out.append(
        (
            "согласие с матрицей части А",
            "vectors.json",
            f"косинуса {ru(agreement['cosine_median'], 6)}, "
            f"минимум {ru(agreement['cosine_min'], 6)}",
        )
    )
    out.append(
        ("время счета векторов", "vectors.json", f"за {ru(vectors['seconds'], 1)} с на процессоре")
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
    out.append(("моделей витрины", "warehouse.json", f"моделей витрины {len(warehouse['models'])}"))

    # Числа сборки витрины dbt. Моделей и тестов считает сам dbt в своем
    # отчете о прогоне (`run_results.json`), а не разбор его вывода.
    counts = mart["build"]["counts"]
    out.append(
        (
            "моделей dbt",
            "dbt.json",
            f"{counts['models']}: {counts['staging_models']} в `staging` "
            f"и {counts['mart_models']} в `marts`",
        )
    )
    out.append(
        (
            "тестов dbt",
            "dbt.json",
            f"{counts['tests']}: {counts['generic_tests']} типовой "
            f"и {counts['singular_tests']} своих",
        )
    )
    out.append(
        (
            "падений в сборке витрины",
            "dbt.json",
            f"{counts['failed']} из {counts['models'] + counts['tests']}",
        )
    )
    agreed = mart["agreed"]
    out.append(
        (
            "согласие dbt с бесплатным режимом",
            "dbt.json",
            f"на {sum(agreed.values())} моделях из {len(agreed)}",
        )
    )
    out.append(
        (
            "вес замыкания dbt",
            "dbt.json",
            f"{mart['dbt']['packages']} пакетов и {ru(mart['dbt']['disk_mb'], 1)} МБ",
        )
    )
    out.append(
        (
            "версии dbt",
            "dbt.json",
            f"dbt-core {mart['dbt']['versions']['dbt-core']} с адаптером "
            f"dbt-clickhouse {mart['dbt']['versions']['dbt-clickhouse']}",
        )
    )

    # Числа репетиции выгрузки. Живой файл `results/events.json` растет каждым
    # прогоном, поэтому в текст идут числа репетиции, а не его.
    first, repeat, after = drain["drains"]
    out.append(
        (
            "трафик в репетиции выгрузки",
            "events-rehearsal.json",
            f"{drain['service']['queries_sent']} запросов и "
            f"{drain['service']['feedback_sent']} оценок",
        )
    )
    out.append(
        (
            "первая выгрузка",
            "events-rehearsal.json",
            f"{first['added']} строк из {first['offered_lines']} предложенных",
        )
    )
    out.append(
        (
            "повторная выгрузка",
            "events-rehearsal.json",
            f"добавила {repeat['added']} из {repeat['offered_lines']}",
        )
    )
    out.append(
        (
            "дубли повторной выгрузки",
            "events-rehearsal.json",
            f"опознала {repeat['duplicates']} дублей",
        )
    )
    out.append(
        (
            "выгрузка после новых запросов",
            "events-rehearsal.json",
            f"третья добавила {after['added']} из {after['offered_lines']} предложенных",
        )
    )
    out.append(
        (
            "строк после третьей выгрузки",
            "events-rehearsal.json",
            f"стало {after['lines_in_repo']} строк",
        )
    )
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
    writer = {row["threads"]: row for row in logging_report["writer"]["runs"]}
    out.append(
        (
            "запись журнала в один поток",
            "logging.json",
            f"{ru_spaced(writer[1]['median_events_per_second'])} события в секунду, "
            f"{ru(writer[1]['median_us_per_event'], 2)} мкс",
        )
    )
    out.append(
        (
            "запись журнала в восемь потоков",
            "logging.json",
            f"{ru_spaced(writer[8]['median_events_per_second'])} события в секунду",
        )
    )
    gap = logging_report["request"]["difference_p50"]
    out.append(
        (
            "цена журнала в задержке запроса",
            "logging.json",
            f"{'минус' if gap['point_ms'] < 0 else 'плюс'} {ru(abs(gap['point_ms']), 3)} мс "
            f"с интервалом [{ru(gap['lo_ms'], 3)}; {ru(gap['hi_ms'], 3)}] мс",
        )
    )
    shapes = logging_report["size"]["groups"]
    out.append(
        (
            "обычное событие журнала",
            "logging.json",
            f"{ru_spaced(shapes['answered-top10']['mean_bytes'])} байта",
        )
    )
    out.append(
        (
            "худшее событие журнала",
            "logging.json",
            f"{ru_spaced(shapes['answered-top50']['mean_bytes'])} байта",
        )
    )
    month = logging_report["month"]["levels"]
    base = month[str(logging_report["month"]["base_per_day"])]
    out.append(("месяц журнала сырыми", "logging.json", f"{ru(base['raw_mb'], 2)} МБ"))
    out.append(("месяц журнала после gzip", "logging.json", f"{ru(base['gzip_mb'], 2)} МБ"))
    out.append(
        (
            "месяц журнала при десяти тысячах",
            "logging.json",
            f"{ru(month['10000']['raw_mb'], 2)} МБ",
        )
    )
    out.append(
        ("сжатие журнала", "logging.json", f"{ru(logging_report['month']['gzip_ratio'], 2)} раза")
    )
    failure = logging_report["failure"]
    out.append(
        (
            "сервис при отказе журнала",
            "logging.json",
            f"{failure['answered_200']} из {failure['requests']} запросов с кодом 200",
        )
    )
    out.append(
        (
            "отказов записи при недоступном каталоге",
            "logging.json",
            f"{failure['event_log']['failed']} из {failure['write_attempts']} попыток",
        )
    )
    out.append(
        (
            "задержка запроса в замере журнала",
            "logging.json",
            f"{ru(logging_report['request']['on']['p50_ms'], 3)} мс",
        )
    )

    # Числа страницы дашборда. Сами числа витрины на страницу печатает
    # `rufinrag/dashboard.py`, а здесь проверяются числа про саму страницу:
    # сколько их на ней, сколько она весит и сколько стоит сборка.
    sample = page["builds"]["sample"]
    empty = page["builds"]["empty"]
    out.append(("чисел на странице дашборда", "dashboard.json", f"{sample['numbers']} чисел"))
    out.append(("графиков на странице", "dashboard.json", f"{sample['charts']} графика"))
    out.append(("вес страницы", "dashboard.json", f"{ru(sample['page_kb'], 1)} КБ"))
    out.append(("сборка страницы", "dashboard.json", f"{ru(sample['seconds'], 2)} с"))
    out.append(("страница на пустом журнале", "dashboard.json", f"{ru(empty['page_kb'], 1)} КБ"))
    out.append(("чисел на пустом журнале", "dashboard.json", f"{empty['numbers']} число"))
    out.append(
        (
            "внешние ссылки страницы",
            "dashboard.json",
            f"{sample['external_resources'] + empty['external_resources']} ссылок на чужие машины",
        )
    )
    out.append(("суток на странице", "dashboard.json", f"последние {page['days_shown']} суток"))

    # Лексическая ступень после решения 16: уровень качества, прежний уровень
    # рядом с ним, цена стеммера на запросе и словарь собранного индекса.
    # Прежние числа берутся из строк «replaced» тех же файлов, поэтому
    # оборот «было столько, стало столько» тоже под проверкой.
    baseline = load("regression-baseline.json")
    index = load("index.json")
    out.append(
        (
            "базовая линия лексической связки",
            "regression-baseline.json",
            ru(baseline["metrics"]["NDCG@10"], 3),
        )
    )
    out.append(
        (
            "рост базовой линии",
            "regression-baseline.json",
            f"с {ru(baseline['replaced']['NDCG@10'], 3)} до "
            f"{ru(baseline['metrics']['NDCG@10'], 3)}",
        )
    )
    lexical = latency["configs"]["bm25"]
    out.append(
        (
            "рост задержки лексической связки",
            "latency-local.json",
            f"с {ru(lexical['replaced']['p50_ms'])} мс до {ru(lexical['p50_ms'])} мс",
        )
    )
    tokens = latency["lexical_tokenization"]
    out.append(("токенов в запросе", "latency-local.json", f"из {tokens['tokens_p50']} токенов"))
    for label, key in (
        ("стеммер на пустом кэше", "stemmed_cold_us_p50"),
        ("стеммер на теплом кэше", "stemmed_warm_us_p50"),
        ("прежнее разбиение", "forms_us_p50"),
    ):
        out.append((label, "latency-local.json", f"{ru(tokens[key], 1)} мкс"))
    # Без слова после числа нарочно: падеж зависит от последней цифры, а
    # словарь растет каждой докачкой корпуса.
    out.append(("словарь индекса", "index.json", f"словарь индекса {ru_spaced(index['vocab'], 0)}"))
    # Размер кэша основ сверяется не с замером, а с самим кодом: это константа
    # токенизации, и README называет ее числом.
    from rufinrag.retrieval.text import stem

    out.append(
        (
            "кэш основ",
            "rufinrag/retrieval/text.py",
            f"{ru_spaced(stem.cache_info().maxsize, 0)} записями",
        )
    )

    # Прогон DAG в контейнере: чем поднят Airflow, сколько он съел памяти и что
    # показали два прогона подряд. Числа идут из той же проверки повторного
    # прогона, которая роняет работу CI при расхождении.
    dag = load("dag-rerun.json")
    out.append(("образ Airflow", "dag-rerun.json", dag["image"]))
    out.append(("пик памяти контейнера DAG", "dag-rerun.json", f"{dag['peak_memory_mb']} МБ"))
    out.append(
        (
            "пропуск задач на повторном прогоне",
            "dag-rerun.json",
            f"пропускает {len(dag['skipped_on_repeat'])} задачи из {dag['tasks']}",
        )
    )
    out.append(
        (
            "добавлено фрагментов на повторе",
            "dag-rerun.json",
            f"добавляет {dag['added_chunks']} фрагментов",
        )
    )
    out.append(
        (
            "повторы идентификаторов",
            "dag-rerun.json",
            f"{dag['duplicate_chunk_ids']} повторов идентификаторов на "
            f"{ru_spaced(dag['corpus_after']['chunks'], 0)} фрагментов",
        )
    )
    dag_ci = load("dag-rerun-ci.json")
    out.append(
        (
            "пик памяти контейнера DAG в прогоне CI",
            "dag-rerun-ci.json",
            f"{dag_ci['peak_memory_mb']} МБ в прогоне CI",
        )
    )

    # Падение шага в середине цепочки: учебная тревога в контейнере.
    recovery = load("dag-recovery.json")
    out.append(
        (
            "попыток упавшего шага",
            "dag-recovery.json",
            f"делает {recovery['attempts']} попытки",
        )
    )
    out.append(
        (
            "задач после упавшего шага",
            "dag-recovery.json",
            f"{len(recovery['tasks_not_started'])} задачи из {recovery['tasks']} после него "
            "не идут вовсе",
        )
    )
    out.append(
        (
            "файлов прошлого прогона тронуто падением",
            "dag-recovery.json",
            f"{len(recovery['files_touched_by_failure'])} файлов прошлого прогона",
        )
    )

    # Порог алерта по деградации метрики и учебная деградация, на которой он
    # проверен. Числа порога идут из замера разброса, числа ступеней из прогона.
    threshold = load("alert-threshold.json")
    drill = load("alert-drill.json")
    out.append(
        (
            "порог алерта",
            "alert-threshold.json",
            f"больше {ru(threshold['threshold'], 3)} останавливает прогон",
        )
    )
    out.append(
        (
            "разброс метрики в истории прогонов",
            "alert-threshold.json",
            f"{threshold['history']['repeated_runs']} прогонов из "
            f"{threshold['history']['runs']} в истории репозитория дали одно и то же значение "
            f"на {len(threshold['history']['repeated_machines'])} разных",
        )
    )
    out.append(
        (
            "прогонов доливки",
            "alert-threshold.json",
            f"доливка повторена кодом {threshold['delivery_noise']['runs']} раз",
        )
    )
    out.append(
        (
            "наибольший сдвиг доливки",
            "alert-threshold.json",
            f"наибольший сдвиг там {ru(threshold['delivery_noise']['ceiling'], 4)}",
        )
    )
    out.append(
        (
            "потолок порога",
            "alert-threshold.json",
            f"меньше {ru(threshold['cap'], 4)}",
        )
    )
    out.append(
        (
            "реальный сдвиг качества от стеммера",
            "alert-threshold.json",
            f"на {ru(threshold['real_shift_same_corpus'], 4)}",
        )
    )
    fired = next(rung for rung in drill["rungs"] if rung["alert_fired"])
    silent = max(
        (rung for rung in drill["rungs"] if not rung["alert_fired"]), key=lambda r: r["drop"]
    )
    out.append(
        (
            "ступень, на которой алерт сработал",
            "alert-drill.json",
            f"потере {fired['relevant_removed']} фрагментов из {fired['relevant_total']} "
            f"падение {ru(fired['drop'], 4)}",
        )
    )
    out.append(
        (
            "ступень, на которой алерт молчит",
            "alert-drill.json",
            f"потере {silent['relevant_removed']} из {silent['relevant_total']} падение "
            f"{ru(silent['drop'], 4)}",
        )
    )
    out.append(
        (
            "сколько ступеней алерт промолчал",
            "alert-drill.json",
            f"на {len(drill['silent_steps'])} ступенях из {drill['steps']}",
        )
    )
    return out


def dashboard_values() -> list[str]:
    """Пересчет чисел самой страницы: страница собирается заново и сверяется.

    Вход сборки лежит в двух файлах результата: вывод моделей витрины на
    собранном наборе в `warehouse.json`, остальное состояние в
    `dashboard.json`. Движок витрины для этого не нужен, поэтому проверка
    работает и там, где его не ставили. Расходится она, если страница начала
    печатать не то, что считает витрина: потеряла знаменатель, округлила
    число или подставила значение мимо модели.
    """
    page = load("dashboard.json")
    snapshot = dashboard.Snapshot(
        mart=load("warehouse.json")["mart"], live=False, regulation=None, **page["snapshot"]
    )
    rendered = dashboard.render(snapshot)
    rebuilt = [
        {"label": value.label, "text": value.text, "source": value.source}
        for value in rendered.values
    ]
    known = page["values"]
    if rebuilt == known:
        return []
    problems = [
        f"  {old['label']}: в витрине {new['text']}, на странице было {old['text']}"
        for old, new in zip(known, rebuilt, strict=False)
        if old != new
    ]
    if len(rebuilt) != len(known):
        problems.append(f"  чисел на странице {len(rebuilt)}, в замере {len(known)}")
    return problems


def main() -> int:
    # Переносы строк убираются: ожидаемая запись длиннее строки README попадала
    # бы в проверку разорванной, и проверка зависела бы от того, где встал
    # перенос, а не от числа.
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    missing = []
    for what, source, expected in checks():
        if expected not in readme:
            missing.append((what, source, expected))
    if missing:
        print("README расходится с замерами:", file=sys.stderr)
        for what, source, expected in missing:
            print(f"  {what}: в {source} стоит {expected}, в README такого нет", file=sys.stderr)
        return 1
    problems = dashboard_values()
    if problems:
        print("страница дашборда разошлась с витриной:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        f"числа README сошлись с замерами, проверок {len(checks())}; "
        f"страница дашборда пересобрана, чисел {len(load('dashboard.json')['values'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
