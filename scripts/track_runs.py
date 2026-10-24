"""Трекинг прогонов в MLflow: каждый замер проекта становится прогоном.

Задача 6.3. Прогоны обучения и оценки в этом проекте это не ноутбук с циклом, а
скрипты, которые пишут отчет в `results/*.json`. Трекинг поэтому устроен так:
отчет это прогон, его числа это метрики, его обстановка (машина, версия корпуса,
связка) это параметры, а отпечаток файла это метка, по которой прогон узнается
второй раз и не заводится заново.

Почему `mlflow-skinny`, а не mlflow. Нужен клиент и файловое хранилище, а не
сервер с вебом: сервер стоит денег и рук, а проект обязан работать без того и
другого. Skinny ставится в 66 МБ зависимостей против полного пакета с pandas и
sqlalchemy, и в образ сервиса не идет вовсе: сервис читает свою версию из
`results/registry.json` и MLflow ему не нужен (`src/rufinrag/registry.py`).

Хранилище лежит в `logs/mlruns` и в git не едет: это служебная папка трекера, а
проверяемый итог это `results/tracking.json` и реестр версий. Так числа остаются
пересчитываемыми без установки MLflow.

Запуск: python scripts/track_runs.py [--store logs/mlruns] [--experiment ru-fin-rag]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.registry import DIGEST_CHARS, file_digest, load  # noqa: E402

# Отчеты, которые прогонами не являются: эталон регрессии (это запись о
# прошлом прогоне), сам реестр и итог этого трекинга.
NOT_A_RUN = {"registry.json", "tracking.json", "regression-baseline.json"}
PARAM_CHARS = 250
# Метрикой считается число, параметром строка. Поля времени в метрики не идут:
# «дата замера» это не метрика, и в графике MLflow она врет.
NOT_A_METRIC = {"measured_at", "built_at", "recomputed_at", "written_at"}


ALLOWED = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. :/"


def safe_name(key: str) -> str:
    """Имя метрики, которое принимает MLflow.

    Он разрешает только латиницу, цифры и несколько знаков, поэтому `NDCG@10`
    становится `NDCG_at_10`, а русские ключи отчетов (`живые`, `все`)
    переводятся в подчеркивания. Правило печатается в итог трекинга, чтобы по
    имени метрики можно было вернуться к полю отчета.
    """
    swapped = key.replace("@", "_at_")
    return "".join(ch if ch in ALLOWED else "_" for ch in swapped)


def flatten(value: object, prefix: str = "") -> tuple[dict[str, float], dict[str, str]]:
    """Числа и строки отчета по плоским ключам, рекурсивно.

    Списки разворачиваются по индексу только если это числа: список словарей
    (например лестница учебной деградации) уходит в параметр целиком, потому что
    метрикой он не является, а потерять его нельзя.
    """
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}
    key = prefix.rstrip(".")
    if isinstance(value, bool):
        metrics[key] = float(value)
    elif isinstance(value, int | float):
        if key.rsplit(".", 1)[-1] not in NOT_A_METRIC:
            metrics[key] = float(value)
    elif isinstance(value, str):
        params[key] = value[:PARAM_CHARS]
    elif isinstance(value, dict):
        for name, inner in value.items():
            m, p = flatten(inner, f"{prefix}{name}.")
            metrics.update(m)
            params.update(p)
    elif isinstance(value, list):
        numbers = all(
            isinstance(item, int | float) and not isinstance(item, bool) for item in value
        )
        if value and numbers:
            for number, item in enumerate(value):
                metrics[f"{key}.{number}"] = float(item)
        elif value:
            params[key] = json.dumps(value, ensure_ascii=False)[:PARAM_CHARS]
    return metrics, params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "logs/mlruns")
    parser.add_argument("--experiment", default="ru-fin-rag")
    parser.add_argument("--out", type=Path, default=ROOT / "results/tracking.json")
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError:
        print(
            "нет mlflow. Поставить: pip install -e '.[tracking]'. Реестр версий "
            "(scripts/register_versions.py) от MLflow не зависит и считается без него",
            file=sys.stderr,
        )
        return 2

    started = time.monotonic()
    args.store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.store.resolve().as_uri())
    mlflow.set_experiment(args.experiment)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)

    registry = load(ROOT) or {}
    known = {
        run.data.tags.get("report_digest")
        for run in client.search_runs([experiment.experiment_id], max_results=50000)
    }

    reports = sorted(p for p in (ROOT / "results").glob("*.json") if p.name not in NOT_A_RUN)
    logged: list[dict[str, object]] = []
    skipped = 0
    collisions = 0
    for path in reports:
        digest = file_digest(path)
        if digest in known:
            skipped += 1
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics, params = flatten(report)
        with mlflow.start_run(run_name=path.stem) as run:
            mlflow.set_tags(
                {
                    "report": path.name,
                    "report_digest": digest,
                    # Прогон привязан к выпуску: по метке видно, на каком
                    # индексе и какой связке он снят, даже когда сам отчет
                    # этого не пишет.
                    "registry_version": registry.get("version", "нет реестра"),
                    "measured_at": str(report.get("measured_at", "")),
                }
            )
            if params:
                mlflow.log_params(
                    {safe_name(key): value for key, value in sorted(params.items())[:100]}
                )
            if metrics:
                named = {safe_name(key): value for key, value in sorted(metrics.items())}
                collisions += len(metrics) - len(named)
                mlflow.log_metrics(named)
            logged.append(
                {
                    "report": path.name,
                    "run_id": run.info.run_id,
                    "metrics": len(metrics),
                    "params": len(params),
                    "digest": digest,
                }
            )

    size_mb = sum(p.stat().st_size for p in args.store.rglob("*") if p.is_file()) / 1048576
    out = {
        "tracked_at": time.strftime("%Y-%m-%d"),
        "mlflow": mlflow.__version__,
        "package": "mlflow-skinny",
        "store": str(args.store.relative_to(ROOT)),
        "store_in_git": False,
        "experiment": args.experiment,
        "reports_total": len(reports),
        "reports_not_a_run": sorted(NOT_A_RUN),
        "runs_logged_now": len(logged),
        "runs_skipped_as_known": skipped,
        "runs_total": len(client.search_runs([experiment.experiment_id], max_results=50000)),
        "store_mb": round(size_mb, 1),
        "digest_chars": DIGEST_CHARS,
        "metric_name_rule": "@ переходит в _at_, остальные недопустимые знаки "
                            "(в том числе русские буквы) в подчеркивание",
        "metric_name_collisions": collisions,
        "registry_version": registry.get("version"),
        "logged": logged,
        "seconds": round(time.monotonic() - started, 1),
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"прогонов в MLflow {out['runs_total']}: завели сейчас {out['runs_logged_now']}, "
        f"уже были {out['runs_skipped_as_known']}, отчетов {out['reports_total']}, "
        f"хранилище {out['store_mb']} МБ в {out['store']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
