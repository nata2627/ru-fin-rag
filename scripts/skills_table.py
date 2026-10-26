#!/usr/bin/env python3
"""Таблица «навык, где он в коде, каким числом подтвержден» для README.

Постановка (раздел 7) требует именно такую таблицу и требование к ней одно:
каждая строка проверяема по ссылке за минуту. Поэтому таблица не набирается
руками, а печатается этим скриптом из файлов `results/`, а `verify_numbers.py`
сверяет каждую ее строку с тем, что стоит в README. Руками в README правится
текст строки, но не число: разойдись они, проверка упадет.

Ключ `--write` вписывает таблицу в README на место прежней. Это нужно потому,
что два числа таблицы меняются сами: ночной прогон увеличивает счетчик своих
прогонов и запись приемки. Без перезаписи README следующий же прогон упал бы на
сверке чисел, поэтому ночной прогон вписывает таблицу перед сверкой, а бот
коммитит README вместе с результатами.

Строки части А помечены отдельно. Их числа лежат в отчетах `retrieval/docs/`,
а не в `results/`: сырые выводы прогонов части А закрыты ее `.gitignore`, и это
незакрытый вопрос к сдаче (решение за Наталией: либо класть в репозиторий
компактные итоговые таблицы, либо открыть часть `data/results/`).

Запуск: python scripts/skills_table.py [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def ru(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def ru_spaced(value: float, digits: int = 0) -> str:
    text = f"{value:,.{digits}f}" if abs(value) >= 10_000 else f"{value:.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def строки() -> list[tuple[str, str, str, str]]:
    """Навык, где он в коде, число, где это число посчитано."""
    latency = load("latency-local.json")
    hosted = load("latency-hosted.json")
    rerank = load("rerank-quality.json")
    store = load("vector-store.json")
    quant = load("quantization.json")
    house = load("warehouse.json")
    dbt = load("dbt.json")
    online = load("compare-configs.json")["все"]
    size = load("sample-size.json")
    threshold = load("alert-threshold.json")
    drill = load("alert-drill.json")
    rules = load("regulation.json")
    answer = load("answer-quality.json")["configs"]["hybrid"]
    router = load("router.json")
    graph = load("graph-expansion.json")
    repro = load("reproducibility.json")
    accept = load("acceptance.json")
    corpus = load("corpus.json")

    bm25 = latency["configs"]["bm25"]
    hybrid = latency["configs"]["hybrid"]
    ссылка = answer["ссылка на релевантный фрагмент"]
    return [
        ("Лексический поиск BM25 со снятием формы слова",
         "`src/rufinrag/retrieval/bm25.py`, `text.py`",
         f"p50 {ru(bm25['p50_ms'])} мс, p95 {ru(bm25['p95_ms'])} мс на маке",
         "`results/latency-local.json`"),
        ("Плотный поиск и гибрид через RRF",
         "`src/rufinrag/retrieval/dense.py`, `fusion.py`",
         f"гибрид p50 {ru(hybrid['p50_ms'])} мс, p95 {ru(hybrid['p95_ms'])} мс",
         "`results/latency-local.json`"),
        ("Переранжирование кросс-энкодером",
         "`src/rufinrag/retrieval/ranking.py`",
         f"прибавка NDCG@10 {ru(rerank['difference_ndcg10']['mean'], 3)} "
         f"[{ru(rerank['difference_ndcg10']['lo'], 4)}; "
         f"{ru(rerank['difference_ndcg10']['hi'], 3)}]",
         "`results/rerank-quality.json`"),
        ("Выбор векторного хранилища по замеру",
         "`scripts/compare_vector_stores.py`, `src/rufinrag/retrieval/dense.py`",
         f"faiss p50 {ru(store['stores']['faiss']['p50_ms'])} мс против "
         f"numpy {ru(store['stores']['numpy']['p50_ms'])} мс, "
         f"индекс {ru(store['stores']['numpy']['index_mb'], 1)} МБ",
         "`results/vector-store.json`"),
        ("Квантование векторов под бесплатный тариф",
         "`scripts/measure_quantization.py`, `src/rufinrag/search.py`",
         f"память {ru(quant['memory']['int8_mb'], 1)} МБ против "
         f"{ru(quant['memory']['float32_mb'], 1)} МБ, перекрытие топ-10 "
         f"{ru(quant['quality']['mean_overlap_top10'], 3)}",
         "`results/quantization.json`"),
        ("Сервис на FastAPI на бесплатном хостинге",
         "`src/rufinrag/api.py`",
         f"p50 {ru(hosted['p50_ms'], 1)} мс, p95 {ru(hosted['p95_ms'], 1)} мс "
         f"по открытой ссылке",
         "`results/latency-hosted.json`"),
        ("Журнал запросов и витрина на ClickHouse в двух режимах",
         "`src/rufinrag/warehouse.py`, `events.py`",
         f"{house['events']['lines']} строк журнала: "
         f"{house['events']['query_events']} запросов и "
         f"{house['events']['feedback_events']} отзывов, оба режима сошлись",
         "`results/warehouse.json`"),
        ("Модели витрины на dbt",
         "`dbt/models/`",
         f"{dbt['build']['counts']['models']} моделей собираются за "
         f"{ru(dbt['build']['seconds'], 1)} с",
         "`results/dbt.json`"),
        ("Онлайн сравнение связок и перестановочный тест",
         "`src/rufinrag/interleaving.py`, `scripts/compare_configs.py`",
         f"{ru(online['ndcg']['right'], 4)} против "
         f"{ru(online['ndcg']['left'], 4)}, разница "
         f"{ru(online['ndcg']['diff_right_minus_left'], 4)}",
         "`results/compare-configs.json`"),
        ("Правило размера выборки и остановки, записанное до данных",
         "`scripts/plan_sample_size.py`, `src/rufinrag/interleaving.py`",
         f"альфа {ru(size['rule']['alpha'])}, мощность {ru(size['rule']['power'])}, "
         f"целевой эффект {ru(size['rule']['offline_target_effect'])}",
         "`results/sample-size.json`"),
        ("Порог алерта по истории прогонов и проверка деградацией",
         "`src/rufinrag/monitoring.py`, `scripts/measure_alert_threshold.py`",
         f"порог {ru(threshold['history']['repeated_value'], 4)} по "
         f"{threshold['history']['runs']} прогонам, учебная деградация поймана "
         f"на просадке {ru(drill['first_fired_drop'], 4)}, самая большая "
         f"незамеченная {ru(drill['largest_silent_drop'], 4)}",
         "`results/alert-threshold.json`, `results/alert-drill.json`"),
        ("Регламент по расписанию без ее машины",
         "`.github/workflows/nightly.yml`, `dags/`",
         f"{rules['runs']} прогонов, следимая метрика {ru(rules['watched_metric'], 4)}",
         "`results/regulation.json`"),
        ("Ответ с цитатами и отказ без опоры",
         "`src/rufinrag/answer.py`, `generation.py`",
         f"ссылка на релевантный фрагмент {ссылка['hit']} из {ссылка['of']}, "
         f"отказов {answer['refused']} из {answer['queries']}",
         "`results/answer-quality.json`"),
        ("Роутер на инструментах, отрицательный результат",
         "`src/rufinrag/router.py`",
         f"маршруты: акты {router['routes']['акты']}, граф "
         f"{router['routes']['граф']}, отказ {router['routes']['отказ']}; "
         f"качество не изменилось",
         "`results/router.json`"),
        ("Граф ссылок между актами, отрицательный результат",
         "`src/rufinrag/graph.py`",
         f"{ru_spaced(graph['graph']['edges'])} дуг, ссылки внутрь у "
         f"{ru_spaced(graph['graph']['acts_with_references'])} актов из "
         f"{ru_spaced(corpus['acts'])}, прибавки нет",
         "`results/graph-expansion.json`"),
        ("Воспроизводимость выдачи от прогона к прогону",
         "`scripts/check_reproducibility.py`",
         f"состав топ-10 совпал у {repro['configs']['bm25']['composition']} из "
         f"{repro['configs']['bm25']['queries']} запросов",
         "`results/reproducibility.json`"),
        ("Приемка автономности по факту, а не по обещанию",
         "`scripts/record_acceptance.py`",
         f"прогонов {accept['scheduled_runs_total']}, зеленых "
         f"{accept['scheduled_runs_green']}, нужно {accept['days_required']} дней",
         "`results/acceptance.json`"),
    ]


def таблица() -> str:
    шапка = ("| Навык | Где в коде | Чем подтвержден | Где посчитано |\n"
             "|---|---|---|---|\n")
    тело = "".join(f"| {навык} | {где} | {число} | {файл} |\n"
                   for навык, где, число, файл in строки())
    return шапка + тело


def вписать(readme: str, table: str) -> str:
    """Заменить таблицу в README. Границы это шапка и последняя строка таблицы."""
    шапка = "| Навык | Где в коде | Чем подтвержден | Где посчитано |"
    начало = readme.index(шапка)
    конец = начало
    for строка in readme[начало:].split("\n"):
        if строка.startswith("|"):
            конец += len(строка) + 1
        else:
            break
    return readme[:начало] + table + readme[конец:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="вписать таблицу в README вместо печати")
    args = ap.parse_args()
    if not args.write:
        print(таблица(), end="")
        return
    readme = ROOT / "README.md"
    текст = readme.read_text(encoding="utf-8")
    новый = вписать(текст, таблица())
    if новый == текст:
        print("таблица навыков в README уже совпадает с замерами")
        return
    readme.write_text(новый, encoding="utf-8")
    print("таблица навыков в README перезаписана по замерам")


if __name__ == "__main__":
    main()
