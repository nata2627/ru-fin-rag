"""Сверка двух режимов хранения журнала запросов на одном наборе событий.

Режим стройки это ClickHouse сервером из compose (профиль `logs`), бесплатный
режим это `chdb` поверх файлов JSONL, без сервера. Модели витрины у них одни
(проект `dbt`, разворачивает их в запрос `rufinrag/warehouse.py`), и этот
прогон печатает, совпали ли выходы. Сборку тех же моделей самим dbt проверяет
`scripts/build_mart.py`: ему нужен сервер, которого в бесплатном режиме нет.

События здесь собранные, а не живые: открытый сервис еще не работал, и брать
числа трафика из воздуха нельзя. Набор строится теми же функциями, которыми
пишет события сервис (`rufinrag/events.py`), поэтому схема в сверке настоящая,
а числа в отчете это числа собранного набора, что в нем и написано.

Запуск: python scripts/compare_warehouses.py [--server http://127.0.0.1:8123]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import events, warehouse  # noqa: E402
from rufinrag.search import Hit, SearchResult  # noqa: E402

QUERIES = (
    "норматив достаточности базового капитала",
    "порядок формирования резервов по ссудам",
    "требования к отчетности микрофинансовой организации",
    "правила ведения бухгалтерского учета в кредитных организациях",
    "идентификация клиента при дистанционном обслуживании",
)
# Связки с разным составом ступеней: у bm25 плотной ступени нет, у
# hybrid-rerank есть переранжирование. Витрина обязана показать ступени такими,
# какие пришли, а не теми, что заложены в колонки.
STAGES = {
    "bm25": ("bm25",),
    "hybrid": ("bm25", "dense", "fusion"),
    "hybrid-rerank": ("bm25", "dense", "fusion", "rerank"),
}


def _result(rng: random.Random, query: str, config: str) -> SearchResult:
    timings = {stage: round(rng.uniform(1.0, 40.0), 3) for stage in STAGES[config]}
    timings["total"] = round(sum(timings.values()) + rng.uniform(0.5, 3.0), 3)
    hits = [
        Hit(
            chunk_id=f"chunk-{rng.randrange(62594)}",
            score=round(rng.uniform(0.1, 9.9), 6),
            rank=rank,
            act_id=f"act-{rng.randrange(1554)}",
            act="Указание Банка России",
            title="Акт",
            section="1.1",
            url="https://www.cbr.ru/act",
            text="фрагмент",
        )
        for rank in range(1, 4)
    ]
    return SearchResult(
        query=query,
        config=config,
        config_version=f"{config}+cand50",
        corpus_version="chunks62594-acts1554",
        hits=hits,
        timings_ms=timings,
        degraded=[],
    )


def sample_events(
    seed: int = 17,
    days: tuple[str, ...] = ("2026-10-21", "2026-10-22"),
    per_day: int = 100,
    rated_every: int = 4,
    orphans: int = 5,
) -> list[dict]:
    """Набор событий с теми же изъянами, какие бывают у живого журнала.

    В нем есть отвергнутые запросы, отказ сервиса, оценки без оценок и оценки
    на идентификатор, которого в журнале нет: именно на сиротах расходятся
    реализации соединения, поэтому они в наборе есть всегда.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for day in days:
        for index in range(per_day):
            received_at = f"{day}T{index // 60 + 8:02d}:{index % 60:02d}:{index % 60:02d}.000Z"
            request_id = events.new_request_id()
            query = QUERIES[index % len(QUERIES)]
            if index % 50 == 49:
                out.append(
                    events.rejected(
                        request_id, received_at, query, 10, 503, "сервис не готов", "0.1.0"
                    )
                )
                continue
            if index % 25 == 24:
                out.append(
                    events.rejected(
                        request_id, received_at, "   ", 10, 422, "запрос из пробелов", "0.1.0"
                    )
                )
                continue
            config = list(STAGES)[index % len(STAGES)]
            out.append(
                events.answered(request_id, received_at, 10, _result(rng, query, config), "0.1.0")
            )
            if index % rated_every == 0:
                rated_at = f"{day}T{index // 60 + 8:02d}:{index % 60:02d}:{index % 60:02d}.900Z"
                out.append(
                    events.feedback(request_id, "up" if index % 8 == 0 else "down", rated_at)
                )
    for number in range(orphans):
        out.append(
            events.feedback(
                f"{number:032x}", "down", f"{days[-1]}T23:5{number % 10}:00.000Z"
            )
        )
    return out


def write_sample(directory: Path, batch: list[dict]) -> dict:
    log = events.EventLog(directory)
    for event in batch:
        log.append(event)
    log.close()
    if log.failed:
        raise SystemExit(f"не записано {log.failed} событий: {log.last_error}")
    known = {e["request_id"] for e in batch if e["kind"] == "query"}
    ratings = [e for e in batch if e["kind"] == "feedback"]
    return {
        "source": "собранный набор, scripts/compare_warehouses.py",
        "files": sorted(p.name for p in directory.glob("queries-*.jsonl")),
        "lines": log.written,
        "query_events": len(known),
        "feedback_events": len(ratings),
        "orphan_feedback": sum(1 for e in ratings if e["request_id"] not in known),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8123")
    parser.add_argument("--events", type=Path, help="готовый каталог журнала вместо собранного")
    parser.add_argument("--out", type=Path, default=ROOT / "results/warehouse.json")
    args = parser.parse_args()

    temporary = None
    if args.events:
        directory = args.events
        sample = {"source": str(directory)}
    else:
        # Набор живет в каталоге процесса, а не в рабочем дереве: события
        # прогонов в репозиторий не едут, туда едут события сервиса.
        temporary = tempfile.TemporaryDirectory(prefix="rufinrag-warehouse-")
        directory = Path(temporary.name)
        sample = write_sample(directory, sample_events())

    files = warehouse.FilesWarehouse(directory)
    free = warehouse.mart(files)

    server = warehouse.ServerWarehouse(args.server)
    built = None
    if server.available():
        server.load(directory)
        built = warehouse.mart(server)
        agreed = {name: built[name] == free[name] for name in warehouse.MART}
    else:
        agreed = {}

    report = {
        "events": sample,
        "free_mode": {"engine": "chdb", "version": _chdb_version()},
        "build_mode": {
            "engine": "clickhouse-server",
            "url": args.server,
            "version": server.scalar("SELECT version()") if built else None,
            "checked": built is not None,
        },
        "models": sorted(warehouse.MART),
        "agreed": agreed,
        "all_agreed": bool(agreed) and all(agreed.values()),
        "mart": warehouse.report(free),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if temporary is not None:
        temporary.cleanup()

    print(json.dumps({k: v for k, v in report.items() if k != "mart"}, ensure_ascii=False))
    if not built:
        print(
            f"ClickHouse на {args.server} не отвечает, сверены только файлы: "
            "docker compose --profile logs up -d clickhouse",
            file=sys.stderr,
        )
        return 1
    if not report["all_agreed"]:
        differing = [name for name, same in agreed.items() if not same]
        print(f"режимы разошлись: {', '.join(differing)}", file=sys.stderr)
        return 1
    return 0


def _chdb_version() -> str:
    import chdb

    return str(chdb.query("SELECT version()", "TSV")).strip()


if __name__ == "__main__":
    raise SystemExit(main())
