"""Датчики свежести и отметки шагов регламента.

Проверяется то, на чем держится идемпотентность повторного прогона: шаг, у
которого вход не менялся, обязан быть пропущен, а шаг, у которого вывод
потерялся или вход поменялся, обязан пойти заново. Отдельно проверяется срок
опроса cbr.ru: без него повторный запуск регламента руками стоил бы госсайту
еще шести запросов.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from rufinrag import pipeline

STEP = pipeline.INDEX


def prepare(root, corpus: bytes = b"corpus", outputs: bool = True) -> None:
    for name in STEP.sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(corpus)
    if outputs:
        for name in STEP.outputs:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact")


def test_run_is_needed_while_the_output_is_missing(tmp_path):
    prepare(tmp_path, outputs=False)
    sensor = pipeline.freshness(tmp_path, STEP)
    assert sensor.stale
    assert "нет вывода" in sensor.reason


def test_run_is_needed_without_a_stamp_even_when_the_output_is_there(tmp_path):
    """Вывод на диске есть, а чем он собран, неизвестно: пересобираем."""
    prepare(tmp_path)
    assert pipeline.freshness(tmp_path, STEP).stale


def test_second_pass_over_the_same_input_is_skipped(tmp_path):
    prepare(tmp_path)
    pipeline.write_stamp(tmp_path, STEP)
    sensor = pipeline.freshness(tmp_path, STEP)
    assert not sensor.stale
    assert "не менялся" in sensor.reason


def test_changed_input_asks_for_a_run(tmp_path):
    prepare(tmp_path)
    pipeline.write_stamp(tmp_path, STEP)
    (tmp_path / pipeline.CORPUS_FILE).write_bytes(b"corpus with a new act")
    sensor = pipeline.freshness(tmp_path, STEP)
    assert sensor.stale
    assert pipeline.CORPUS_FILE in sensor.reason


def test_lost_output_asks_for_a_run_despite_the_stamp(tmp_path):
    """Отметка есть, а индекса нет: так выглядит свежий клон репозитория."""
    prepare(tmp_path)
    pipeline.write_stamp(tmp_path, STEP)
    (tmp_path / STEP.outputs[0]).unlink()
    assert pipeline.freshness(tmp_path, STEP).stale


def test_stamp_holds_the_digest_of_every_input(tmp_path):
    prepare(tmp_path)
    stamp = pipeline.write_stamp(tmp_path, STEP)
    assert set(stamp["sources"]) == set(STEP.sources)
    assert stamp["sources"][pipeline.CORPUS_FILE] == pipeline.digest(
        tmp_path / pipeline.CORPUS_FILE
    )


def write_report(root, checked_at: datetime | None) -> None:
    path = root / pipeline.UPDATE_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "ok"}
    if checked_at is not None:
        report["checked_at"] = checked_at.strftime(pipeline.CHECKED_AT)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_source_is_not_asked_twice_within_the_cooldown(tmp_path):
    now = datetime(2026, 10, 23, 18, 0, 0)
    write_report(tmp_path, now - timedelta(hours=2))
    sensor = pipeline.source_freshness(tmp_path, hours=6, now=now)
    assert not sensor.stale
    assert "2,0 ч назад" in sensor.reason.replace(".", ",")


def test_source_is_asked_again_after_the_cooldown(tmp_path):
    now = datetime(2026, 10, 23, 18, 0, 0)
    write_report(tmp_path, now - timedelta(hours=12))
    assert pipeline.source_freshness(tmp_path, hours=6, now=now).stale


def test_first_run_asks_the_source(tmp_path):
    assert pipeline.source_freshness(tmp_path, now=datetime(2026, 10, 23, 18, 0, 0)).stale


def test_report_without_a_time_asks_the_source(tmp_path):
    write_report(tmp_path, None)
    assert pipeline.source_freshness(tmp_path, now=datetime(2026, 10, 23, 18, 0, 0)).stale


def test_a_poll_stamped_in_the_future_counts_as_just_now(tmp_path):
    """Отчет пишет прогон CI по UTC, а читает его мак по своему поясу.

    Из двух толкований расхождения часов берется осторожное: сайт не трогаем.
    """
    now = datetime(2026, 10, 23, 18, 0, 0)
    write_report(tmp_path, now + timedelta(hours=3))
    assert not pipeline.source_freshness(tmp_path, now=now).stale
