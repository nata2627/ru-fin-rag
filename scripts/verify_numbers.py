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
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import skills_table  # noqa: E402

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


def ru_plural(count: int, one: str, few: str, many: str) -> str:
    """«41 прогон», «2 прогона», «5 прогонов»: число идет в русский текст.

    Нужно потому, что проверка требует ровно той строки, что стоит в README, и
    согласование по числу иначе пришлось бы писать в тексте руками, то есть
    ломать его при каждом новом прогоне.
    """
    last_two, last = count % 100, count % 10
    if 11 <= last_two <= 14 or last == 0 or last >= 5:
        return f"{count} {many}"
    return f"{count} {one}" if last == 1 else f"{count} {few}"


def стоит_в_тексте(expected: str, text: str) -> bool:
    """Есть ли в тексте ровно это число, а не число, которое с него начинается.

    Простая проверка подстрокой пропускала бы «0,27» внутри «0,274» и «174»
    внутри «1174»: ошибка в README выглядела бы как сошедшееся число. Поэтому
    по краям требуется не цифра и не продолжение дробной части.
    """
    шаблон = rf"(?<![\d,]){re.escape(expected)}(?!\d|,\d)"
    return re.search(шаблон, text) is not None


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def diff_text(diff: dict) -> str:
    """Разница со знаком и интервалом, четыре знака.

    Отдельно от `gain_text`: та печатает прибавку и ставит плюс сама, а здесь
    разница бывает отрицательной, и «+-0,002» это не запись числа.
    """
    return f"{diff['mean']:+.4f} [{diff['lo']:+.4f}; {diff['hi']:+.4f}]".replace(".", ",")


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
            f"доливка повторена кодом {threshold['delivery_noise']['runs']} раз: "
            f"{threshold['delivery_noise']['single_act_runs']} прогонов убирают по одному "
            f"случайному акту без размеченных фрагментов, "
            f"{threshold['delivery_noise']['group_act_runs']} прогонов убирают по "
            f"{threshold['delivery_noise']['acts_in_group']} таких актов сразу",
        )
    )
    out.append(
        (
            "наибольший сдвиг доливки",
            "alert-threshold.json",
            f"Наибольшее по модулю изменение {ru(threshold['delivery_noise']['ceiling'], 4)}",
        )
    )
    out.append(
        (
            "наибольшее падение среди прогонов доливки",
            "alert-threshold.json",
            f"наибольшее падение среди тех же {threshold['delivery_noise']['runs']} прогонов "
            f"{ru(threshold['delivery_noise']['max_drop'], 4)}",
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
    # Вырожденная лестница (нет сработавшей ступени или нет молчащей) это
    # поломка замера, и сказать о ней надо словами: прежде она прилетала в
    # ночной прогон трейсбеком StopIteration.
    fired = next((rung for rung in drill["rungs"] if rung["alert_fired"]), None)
    silent = max(
        (rung for rung in drill["rungs"] if not rung["alert_fired"]),
        key=lambda r: r["drop"],
        default=None,
    )
    if fired is None or silent is None:
        raise SystemExit(
            "alert-drill.json вырожден: сработавших ступеней "
            f"{sum(1 for r in drill['rungs'] if r['alert_fired'])}, молчавших "
            f"{sum(1 for r in drill['rungs'] if not r['alert_fired'])}. "
            "Пересчитать: make alert"
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
            f"потере {silent['relevant_removed']} фрагментов из {silent['relevant_total']} "
            f"падение {ru(silent['drop'], 4)}",
        )
    )
    # Слой ответа: порог отказа, его цена и то, чего отказ не ловит.
    answer = load("answer.json")
    chosen = answer["chosen"]
    citations = answer["citations"]
    out.append(("порог опоры", "answer.json", f"порог {ru(chosen['threshold'])}"))
    out.append(
        (
            "ложные отказы на размеченных",
            "answer.json",
            f"{chosen['false_refusals_labeled']['refused']} из "
            f"{chosen['false_refusals_labeled']['of']}",
        )
    )
    out.append(
        (
            "отказы на живых без разметки",
            "answer.json",
            f"{chosen['live_unlabeled']['refused']} из {chosen['live_unlabeled']['of']}",
        )
    )
    out.append(
        (
            "цена шага опоры",
            "answer.json",
            f"медиана {ru(answer['support_cost_ms']['median'], 1)} мс, "
            f"p95 {ru(answer['support_cost_ms']['p95'], 1)} мс",
        )
    )
    out.append(
        (
            "зависимость отвергнутого признака от длины",
            "answer.json",
            f"корреляция {ru(answer['candidates']['top_score']['length_dependence'], 3)}",
        )
    )
    out.append(
        (
            "раздел у фрагмента",
            "answer.json",
            f"есть у {ru_spaced(citations['with_section'], 0)} фрагментов из "
            f"{ru_spaced(citations['chunks'], 0)}",
        )
    )
    out.append(
        (
            "пункты найдены в тексте фрагмента",
            "answer.json",
            f"нашлись у {ru_spaced(citations['with_clauses'], 0)} из "
            f"{ru_spaced(citations['chunks'], 0)}",
        )
    )
    out.append(
        (
            "порядок пунктов внутри акта",
            "answer.json",
            f"{ru_spaced(citations['clause_pairs_non_decreasing'], 0)} неубывающих пар из "
            f"{ru_spaced(citations['clause_pairs_in_act'], 0)}",
        )
    )

    # Качество ответа: куда ведет ссылка. Числа обеих связок, потому что порог
    # отказа выведен на лексической выдаче, а отвечает сервис связкой по
    # умолчанию, и разницу между ними текст называет прямо.
    quality = load("answer-quality.json")
    lexical = quality["configs"]["bm25"]
    default_config = next(name for name in quality["configs"] if name != "bm25")
    online = quality["configs"][default_config]
    for label, key in (
        ("ссылка на релевантный фрагмент", "ссылка на релевантный фрагмент"),
        ("ссылка на фрагмент с полным ответом", "ссылка на фрагмент с полным ответом"),
        ("ссылка на релевантный акт", "ссылка на релевантный акт"),
        ("первая ссылка релевантна", "первая ссылка релевантна"),
    ):
        out.append(
            (
                f"{label}, обе связки",
                "answer-quality.json",
                f"| {lexical[key]['hit']} из {lexical[key]['of']} "
                f"| {online[key]['hit']} из {online[key]['of']} |",
            )
        )
    out.append(
        (
            "отказы связки по умолчанию",
            "answer-quality.json",
            f"{online['refused']} раз из {online['queries']}",
        )
    )
    out.append(
        (
            "отказы связки по умолчанию на живых",
            "answer-quality.json",
            f"{online['by_subsample']['живые']['refused']} из "
            f"{online['by_subsample']['живые']['queries']} против "
            f"{lexical['by_subsample']['живые']['refused']} из "
            f"{lexical['by_subsample']['живые']['queries']}",
        )
    )
    live = online["by_subsample"]["живые"]["ссылка на релевантный фрагмент"]
    synthetic = online["by_subsample"]["синтетика"]["ссылка на релевантный фрагмент"]
    out.append(
        (
            "живые против синтетики по ссылке",
            "answer-quality.json",
            f"на {ru(synthetic['share'] - live['share'], 3)} ({live['hit']} из {live['of']} "
            f"против {synthetic['hit']} из {synthetic['of']})",
        )
    )
    strict_gap = (
        online["ссылка на релевантный фрагмент"]["share"]
        - lexical["ссылка на релевантный фрагмент"]["share"]
    )
    out.append(
        (
            "разрыв связок по ссылке",
            "answer-quality.json",
            f"мере {ru(strict_gap, 3)}",
        )
    )
    cross = quality["cross_support"]
    pinned = cross["с опорой последней цитатой"]
    out.append(
        (
            "отказы при лексической опоре",
            "answer-quality.json",
            f"отказов становится {cross['refused']} из {cross['queries']} и "
            f"{cross['by_subsample']['живые']['refused']} из "
            f"{cross['by_subsample']['живые']['queries']} на живых вопросах",
        )
    )
    out.append(
        (
            "опора не в цитатах, сейчас и при лексической опоре",
            "answer-quality.json",
            f"растет с {online['опора не попала в цитаты']['hit']} из "
            f"{online['опора не попала в цитаты']['of']} до "
            f"{cross['опора не попала в цитаты']['hit']} из "
            f"{cross['опора не попала в цитаты']['of']}",
        )
    )
    out.append(
        (
            "поправка с опорой в цитатах",
            "answer-quality.json",
            f"полностью ({pinned['опора не попала в цитаты']['hit']} из "
            f"{pinned['опора не попала в цитаты']['of']})",
        )
    )
    for label, key in (
        ("ссылка на релевантный фрагмент", "ссылка на релевантный фрагмент"),
        ("на релевантный акт", "ссылка на релевантный акт"),
        ("первая ссылка релевантна", "первая ссылка релевантна"),
    ):
        out.append(
            (
                f"поправка против нынешнего поведения, {label}",
                "answer-quality.json",
                f"{pinned[key]['hit']} из {pinned[key]['of']} против "
                f"{online[key]['hit']} из {online[key]['of']}",
            )
        )
    human = quality["human"]
    out.append(
        (
            "эталон щедрее человека",
            "answer-quality.json",
            f"{human['эталон выше человека']['hit']} пара из {human['эталон выше человека']['of']}",
        )
    )
    out.append(
        (
            "человек щедрее эталона",
            "answer-quality.json",
            f"против {human['человек выше эталона']['hit']} из "
            f"{human['человек выше эталона']['of']} в обратную сторону",
        )
    )
    full = human["полный ответ по эталону"]
    out.append(
        (
            "полный ответ эталона против человека",
            "answer-quality.json",
            f"({full['пар']} пар), человек согласился полностью на "
            f"{full['человек согласился полностью']} и увидел хотя бы часть ответа на "
            f"{full['человек увидел хотя бы часть']}",
        )
    )
    out.append(
        (
            "приемка судьи по ответам",
            "answer-quality.json",
            f"согласие с человеком {ru(human['судья']['каппа с человеком'], 3)} при интервале "
            f"[{ru(human['судья']['интервал'][0], 3)}; {ru(human['судья']['интервал'][1], 3)}] "
            f"на {human['судья']['пар']} парах и объявленном до прогона пороге "
            f"{ru(human['судья']['порог'], 2)}",
        )
    )

    # Роутер на инструментах: качество, задержка и распределение маршрутов.
    router = load("router.json")
    quality = router["quality"]
    latency = router["latency_ms"]
    tool = router["graph_tool"]
    out.append(
        (
            "качество роутера против прямого поиска",
            "router.json",
            f"| {quality['router']['correct']} из {quality['router']['of']} "
            f"| {quality['direct']['correct']} из {quality['direct']['of']} |",
        )
    )
    out.append(
        (
            "разница роутера с прямым поиском",
            "router.json",
            f"{ru(quality['difference']['mean'], 1)} [{ru(quality['difference']['lo'], 1)}; "
            f"{ru(quality['difference']['hi'], 1)}]",
        )
    )
    out.append(
        (
            "задержка роутера",
            "router.json",
            f"| {ru(latency['router']['median'], 1)} мс | {ru(latency['direct']['median'], 1)} мс |",
        )
    )
    out.append(
        (
            "цена роутера в медиане",
            "router.json",
            f"стоит {ru(latency['difference_median'], 1)} мс в медиане",
        )
    )
    out.append(
        (
            "маршруты роутера",
            "router.json",
            f"прямой поиск {router['routes']['акты']}, граф {router['routes']['граф']}, "
            f"отказ {router['routes']['отказ']}, уточнение {router['routes']['уточнение']} "
            f"из {router['queries']}",
        )
    )
    out.append(
        (
            "инструмент графа на тех же запросах",
            "router.json",
            f"прямой поиск {tool['base']} из {tool['queries']}, граф с расширением на "
            f"последнем месте {tool['reserve_deep']} из {tool['queries']}, граф на глубине, "
            f"равной числу цитат, где расширение попадает в пятерку, "
            f"{tool['reserve_in_citations']} из {tool['queries']}",
        )
    )
    out.append(
        (
            "первое обращение к графу в роутере",
            "router.json",
            f"{ru(latency['graph_first_query'], 1)} мс на первое обращение",
        )
    )

    # Граф ссылок между актами и расширение выдачи по цитированию. Числа графа
    # и обе разницы с интервалами: отрицательный результат публикуется так же
    # подробно, как прибавка.
    graph = load("graph.json")["stats"]
    expansion = load("graph-expansion.json")
    reserve = expansion["variants"]["reserve"]["difference_ndcg10"]["hybrid"]
    fusion = expansion["variants"]["fusion"]["difference_ndcg10"]["hybrid"]
    out.append(
        (
            "разных ссылок в корпусе",
            "graph.json",
            f"нашлось {ru_spaced(graph['references'], 0)} разных ссылок",
        )
    )
    out.append(
        (
            "ссылки внутрь корпуса и дуги",
            "graph.json",
            f"{ru_spaced(graph['references_inside'], 0)} указывают внутрь корпуса и дают "
            f"{ru_spaced(graph['edges'], 0)} дуг между актами",
        )
    )
    out.append(
        (
            "ссылки за пределы корпуса",
            "graph.json",
            f"{ru_spaced(graph['references_outside'], 0)} из "
            f"{ru_spaced(graph['references'], 0)} указывают за его пределы",
        )
    )
    out.append(
        (
            "актов со ссылкой внутрь корпуса",
            "graph.json",
            f"есть у {ru_spaced(graph['acts_with_references'], 0)} актов из {graph['acts']}",
        )
    )
    out.append(
        (
            "упоминаний номеров актов",
            "graph.json",
            f"упоминаний всего {ru_spaced(graph['mentions'], 0)}",
        )
    )
    for label, block in (("резерв места", reserve), ("слияние с весом", fusion)):
        for where, key in (("все 991", "all"), ("живые", "live"), ("связанные акты", "linked")):
            out.append(
                (
                    f"расширение по графу, {label}, {where}",
                    "graph-expansion.json",
                    diff_text(block[key]),
                )
            )
    out.append(
        (
            "потолок прибавки по графу",
            "graph-expansion.json",
            f"потолок прибавки это "
            f"{expansion['variants']['reserve']['ceiling']['all']['reachable']} запросов из 991",
        )
    )

    # Мониторинг: пороги дрейфа и доли без ответа, их обоснование и учебная
    # лестница, на которой они проверены.
    mon = load("monitoring-thresholds.json")
    drill_mon = load("monitoring-drill.json")
    rule = mon["rule_applied"]
    first = mon["ladder"][0]
    support = mon["features"]["support_candidates"]["top_score"]
    answerless = mon["answerless_on_reference"]
    out.append(
        (
            "порог дрейфа запросов",
            "monitoring-thresholds.json",
            f"{ru(rule['drift_threshold'])} на окне {rule['window']} запросов",
        )
    )
    out.append(
        (
            "потолок шума дрейфа",
            "monitoring-thresholds.json",
            f"потолок PSI по {first['draws']} окнам из {rule['reference_queries']} "
            f"замороженных вопросов {ru(first['noise_psi_max'], 4)}",
        )
    )
    out.append(
        (
            "порог доли без хорошего ответа",
            "monitoring-thresholds.json",
            f"оценка лучшего кандидата ниже {ru(support['floor'], 4)}",
        )
    )
    out.append(
        (
            "наибольшая доля без ответа в окнах",
            "monitoring-thresholds.json",
            f"наибольшая доля в тех же {first['draws']} окнах {ru(first['noise_share_max'])} "
            f"при базовой доле {answerless['answerless']} из {answerless['answered']}",
        )
    )
    out.append(
        (
            "признак опоры, J Юдена",
            "monitoring-thresholds.json",
            f"J Юдена {ru(support['j'], 4)}, он называет безответными "
            f"{support['good_below_floor']} вопросов из {support['positives']}",
        )
    )
    out.append(
        (
            "пропуски признака опоры",
            "monitoring-thresholds.json",
            f"пропускает {support['bad_above_floor']} из {support['negatives']} без ответа",
        )
    )
    out.append(
        (
            "ступени учебной лестницы мониторинга",
            "monitoring-drill.json",
            f"лестницей из {drill_mon['steps']} ступеней",
        )
    )
    out.append(
        (
            "дрейф, на котором алерт сработал",
            "monitoring-drill.json",
            f"PSI {ru(drill_mon['first_drift_alert'], 4)}",
        )
    )
    out.append(
        (
            "дрейф, на котором алерт промолчал",
            "monitoring-drill.json",
            f"промолчал на 25 живых из 50 ({ru(drill_mon['largest_silent_drift'], 4)})",
        )
    )
    out.append(
        (
            "доля без ответа, на которой алерт промолчал",
            "monitoring-drill.json",
            f"промолчал на {ru(drill_mon['largest_silent_share'])}",
        )
    )

    out.append(
        (
            "сколько ступеней алерт промолчал",
            "alert-drill.json",
            f"на {len(drill['silent_steps'])} ступенях из {drill['steps']}",
        )
    )

    # Реестр версий и трекинг прогонов (задача 6.3). Длина версии выпуска
    # проверяется тоже: укоротится отпечаток, и README начнет обещать не то.
    tracking = load("tracking.json")
    registered = load("registry.json")
    out.append(
        (
            "прогонов в трекере",
            "tracking.json",
            f"{ru_plural(tracking['runs_total'], 'прогон', 'прогона', 'прогонов')} "
            f"из {ru_plural(tracking['reports_total'], 'отчета', 'отчетов', 'отчетов')}",
        )
    )
    out.append(
        (
            "файлов, которые не прогон",
            "tracking.json",
            "прогоном не считаются "
            + ru_plural(len(tracking["reports_not_a_run"]), "файл", "файла", "файлов"),
        )
    )
    out.append(("хранилище трекера", "tracking.json", f"{ru(tracking['store_mb'], 1)} МБ"))
    out.append(("пакет трекинга", "tracking.json", tracking["package"]))
    out.append(
        (
            "артефактов в реестре",
            "registry.json",
            "отпечаток "
            + ru_plural(len(registered["artifacts"]), "артефакта", "артефактов", "артефактов"),
        )
    )
    out.append(
        (
            "связок в реестре",
            "registry.json",
            f"каждой из {len(registered['config_versions'])} связок",
        )
    )
    out.append(
        (
            "длина версии выпуска",
            "registry.json",
            f"{len(str(registered['version']))} знаков отпечатка",
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


def skills_rows(readme: str) -> list[str]:
    """Строки таблицы навыков обязаны стоять в README ровно так, как их
    печатает `skills_table.py`.

    Проверяется строка целиком, а не одно число: в таблице рядом с числом
    стоит файл, где оно посчитано, и разъехаться они не должны тоже.
    """
    беда = []
    for навык, где, число, файл in skills_table.строки():
        строка = " ".join(f"| {навык} | {где} | {число} | {файл} |".split())
        if строка not in readme:
            беда.append(f"  {навык}: ожидается строка «{строка}»")
    return беда


def dag_report_fresh() -> list[str]:
    """Пересборка `results/dag.md`: текст обязан совпасть с отчетами прогонов.

    Файл собирается кодом, и в его заголовке так и написано. Без этой проверки
    утверждение держалось на честном слове: правка руками или разошедшийся с
    отчетами текст проходили бы зелеными.
    """
    path = RESULTS / "dag.md"
    before = path.read_text(encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts/report_dag.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        return [f"  пересборка dag.md упала: {done.stderr.strip()[-300:]}"]
    after = path.read_text(encoding="utf-8")
    if after == before:
        return []
    path.write_text(before, encoding="utf-8")
    return ["  results/dag.md разошелся с отчетами прогонов: пересобрать `make dag-report`"]


def main() -> int:
    # Переносы строк убираются: ожидаемая запись длиннее строки README попадала
    # бы в проверку разорванной, и проверка зависела бы от того, где встал
    # перенос, а не от числа.
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    missing = []
    for what, source, expected in checks():
        if not стоит_в_тексте(expected, readme):
            missing.append((what, source, expected))
    if missing:
        print("README расходится с замерами:", file=sys.stderr)
        for what, source, expected in missing:
            print(f"  {what}: в {source} стоит {expected}, в README такого нет", file=sys.stderr)
        return 1
    problems = skills_rows(readme)
    if problems:
        print("таблица навыков разошлась с замерами:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    problems = dag_report_fresh()
    if problems:
        print("отчет регламента разошелся с замерами:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    problems = dashboard_values()
    if problems:
        print("страница дашборда разошлась с витриной:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        f"числа README сошлись с замерами, проверок {len(checks())}, "
        f"строк таблицы навыков {len(skills_table.строки())}; "
        f"страница дашборда пересобрана, чисел {len(load('dashboard.json')['values'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
