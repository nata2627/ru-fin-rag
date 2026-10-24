"""Реестр версий и трекинг прогонов: что в них ломается тихо.

Главное здесь одно: версия, которую сервис называет в ответе, обязана быть той,
что записана в реестре. Разойдутся они молча, и числа README окажутся про одну
сборку, а отвечать будет другая.

Второе: сам файл реестра обязан отвечать репозиторию. Смените связку по
умолчанию или порог отказа и забудьте пересобрать реестр, и сервис продолжит
называть прежнюю версию выпуска. Это ровно тот отказ, который видно только
проверкой.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rufinrag import __version__, registry
from rufinrag.api import app

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("track_runs", ROOT / "scripts/track_runs.py")
track = importlib.util.module_from_spec(spec)
sys.modules["track_runs"] = track
spec.loader.exec_module(track)


@pytest.fixture(scope="module")
def registered() -> dict:
    entry = registry.load(ROOT)
    assert entry is not None, "нет results/registry.json, собрать: make registry"
    return entry


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_version_is_a_digest_of_the_entry(registered):
    assert registered["version"] == registry.version_of(registered)


def test_any_change_moves_the_version(registered):
    changed = dict(registered)
    changed["default_config"] = "bm25" if registered["default_config"] != "bm25" else "hybrid"
    assert registry.version_of(changed) != registered["version"]


def test_registry_file_matches_the_repository(registered):
    """Файл реестра пересобирается и сверяется, а не живет своей жизнью."""
    fresh = registry.build(ROOT, corpus_version=registered["corpus_version"])
    missing_now = [name for name, value in fresh["artifacts"].items() if value is None]
    missing_then = [name for name, value in registered["artifacts"].items() if value is None]
    if missing_now != missing_then:
        pytest.skip(f"на этой машине нет артефактов: {', '.join(sorted(set(missing_now)))}")
    for field in (
        "service_version",
        "catalog_version",
        "default_config",
        "config_versions",
        "answer_version",
        "artifacts",
        "version",
    ):
        assert fresh[field] == registered[field], f"реестр отстал по полю {field}"


def test_health_names_the_registry(client, registered):
    body = client.get("/health").json()
    assert body["registry"]["version"] == registered["version"]
    assert body["registry"]["matches_running"], body["registry"]["mismatch"]
    assert body["version"] == __version__ == registered["service_version"]


def test_answer_carries_the_registered_version(client, registered):
    search = client.post("/search", json={"query": "резерв по ссуде", "top": 3}).json()
    assert search["registry_version"] == registered["version"]
    assert search["corpus_version"] == registered["corpus_version"]
    assert search["config_version"] == registered["config_versions"][registered["default_config"]]


def test_mismatch_is_named_not_hidden(registered):
    running = {
        "service_version": registered["service_version"],
        "corpus_version": "chunks1-acts1",
        "config_version": registered["config_versions"][registered["default_config"]],
    }
    found = registry.compare(registered, running)
    assert [m.field for m in found] == ["corpus_version"]
    assert found[0].registered == registered["corpus_version"]
    assert registry.compare(None, running)[0].field == "registry"


def test_metric_names_follow_mlflow_rules():
    assert track.safe_name("baseline.NDCG@10") == "baseline.NDCG_at_10"
    assert track.safe_name("живые.queries") == "_" * len("живые") + ".queries"
    assert set(track.safe_name("a@b.живые/c:d 1-2")) <= set(track.ALLOWED)


def test_flatten_splits_numbers_from_words():
    metrics, params = track.flatten(
        {
            "measured_at": 2026,
            "passed": True,
            "config": "hybrid",
            "metrics": {"NDCG@10": 0.6126},
            "seconds": [1.5, 2.5],
            "rungs": [{"share": 0.04}],
        }
    )
    # Дата замера метрикой не считается: на графике она врет.
    assert "measured_at" not in metrics
    assert metrics["metrics.NDCG@10"] == 0.6126
    assert metrics["passed"] == 1.0
    assert metrics["seconds.0"] == 1.5 and metrics["seconds.1"] == 2.5
    assert params["config"] == "hybrid"
    # Список словарей уходит параметром целиком: метрикой он не является, а
    # терять его нельзя.
    assert json.loads(params["rungs"]) == [{"share": 0.04}]


def test_tracking_report_counts_every_report():
    path = ROOT / "results/tracking.json"
    if not path.exists():
        pytest.skip("трекинг не прогонялся: make track")
    report = json.loads(path.read_text(encoding="utf-8"))
    reports = [p.name for p in (ROOT / "results").glob("*.json") if p.name not in track.NOT_A_RUN]
    assert report["reports_total"] == len(reports)
    assert report["runs_total"] >= report["reports_total"]
    # Хранилище трекера в git не едет, и это сказано в самом отчете: иначе
    # проверяющий ищет папку, которой в репозитории нет.
    assert report["store_in_git"] is False
    assert report["package"] == "mlflow-skinny"
