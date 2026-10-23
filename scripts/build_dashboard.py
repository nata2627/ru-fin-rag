"""Сборка статической страницы дашборда по журналу запросов.

Одна команда собирает страницу, которую ночной прогон кладет на GitHub Pages:

    python scripts/build_dashboard.py                 # по журналу репозитория
    python scripts/build_dashboard.py --sample        # по собранному набору событий

Что откуда берется. Числа трафика и качества считают модели витрины из `dbt/`,
их по файлам журнала исполняет `chdb` (src/rufinrag/warehouse.py), и на
страницу уходит их вывод без пересчета. Версия корпуса берется из отметки
последнего прогона регламента, версия связки из файла конфигурации, то есть
оба числа приходят из того же места, откуда их берет сам сервис.

Пустой журнал это штатное состояние, а не отказ: открытый сервис мог не
получить ни одного запроса, а до первой выгрузки с хостинга журнал в
репозитории пуст вообще. В этом случае страница собирается и честно говорит,
что данных нет. Прогон от этого не краснеет: красный `main` остановил бы часы
приемки автономности.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import dashboard, warehouse  # noqa: E402
from rufinrag.config import load_catalog  # noqa: E402

JOURNAL = ROOT / "data/events"
REGULATION = ROOT / "results/regulation.json"
CORPUS = ROOT / "results/corpus.json"

JOURNAL_SOURCE = "журнал сервиса, data/events"
SAMPLE_SOURCE = "собранный набор, scripts/compare_warehouses.py"


def collected_events():
    """Модуль сверки режимов витрины: там живет собранный набор событий.

    Набор не копируется сюда, потому что числа страницы на нем должны
    совпадать с числами сверки до знака, иначе они описывают разные наборы.
    """
    spec = importlib.util.spec_from_file_location(
        "compare_warehouses", ROOT / "scripts/compare_warehouses.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_warehouses"] = module
    spec.loader.exec_module(module)
    return module


def corpus_version() -> tuple[str, str]:
    """Версия корпуса и файл, из которого она взята.

    Сначала отметка регламента: ее пишет каждый успешный ночной прогон, то
    есть она свежее любого замера. Корпус целиком здесь не открывается: это
    62 594 фрагмента ради одной строки.
    """
    for path, key in ((REGULATION, "corpus_version"), (CORPUS, "version")):
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
            if value:
                return value, f"results/{path.name}"
    return "неизвестна", "замера корпуса в results нет"


def regulation() -> dict | None:
    if not REGULATION.exists():
        return None
    return json.loads(REGULATION.read_text(encoding="utf-8"))


def mart(directory: Path) -> dict[str, dict]:
    """Вывод моделей витрины по файлам журнала."""
    free = warehouse.FilesWarehouse(directory)
    return warehouse.report({name: free.run(name) for name in sorted(warehouse.MART)})


def snapshot(directory: Path, *, live: bool, source: str) -> dashboard.Snapshot:
    journal = dashboard.journal_summary(directory)
    rows: dict[str, dict] = {}
    if journal["lines"]:
        try:
            rows = mart(directory)
        except ImportError as error:
            # Журнал есть, а движка нет: это поломка окружения сборки, а не
            # пустой журнал, и молча отдавать страницу «данных нет» нельзя.
            raise SystemExit(
                f"нет движка витрины ({error}), поставить: pip install -e '.[analytics]'"
            ) from error
    version, version_source = corpus_version()
    return dashboard.Snapshot(
        mart=rows,
        journal=journal,
        corpus_version=version,
        corpus_source=version_source,
        config_version=load_catalog().get().version,
        source=source,
        built_at=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        live=live,
        regulation=regulation(),
    )


def build(directory: Path, out: Path, *, live: bool, source: str) -> dict:
    started = time.monotonic()
    state = snapshot(directory, live=live, source=source)
    rendered = dashboard.render(state)
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    page.write_text(rendered.page, encoding="utf-8")
    (out / "mart.json").write_text(
        json.dumps(state.mart, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "page": str(page.relative_to(ROOT)) if page.is_relative_to(ROOT) else str(page),
        "source": source,
        "live": live,
        "empty": state.empty,
        "journal": state.journal,
        "corpus_version": state.corpus_version,
        "config_version": state.config_version,
        "built_at": state.built_at,
        "mart_models": sorted(state.mart),
        "numbers": rendered.numbers,
        "charts": rendered.charts,
        "page_bytes": len(rendered.page.encode("utf-8")),
        "seconds": round(time.monotonic() - started, 2),
        "values": [
            {"label": value.label, "text": value.text, "source": value.source}
            for value in rendered.values
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=JOURNAL)
    parser.add_argument(
        "--sample", action="store_true", help="собранный набор событий вместо журнала"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args()

    if args.sample:
        with tempfile.TemporaryDirectory(prefix="rufinrag-dashboard-") as temporary:
            directory = Path(temporary)
            sample = collected_events()
            sample.write_sample(directory, sample.sample_events())
            report = build(directory, args.out, live=False, source=SAMPLE_SOURCE)
    else:
        report = build(args.events, args.out, live=True, source=JOURNAL_SOURCE)

    print(
        json.dumps({k: v for k, v in report.items() if k != "values"}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
