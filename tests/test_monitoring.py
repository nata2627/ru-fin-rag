"""Мониторинг трафика: мера, пороги и правила учебной лестницы.

Сам прогон лестницы идет отдельной командой и стоит 35 с
(`scripts/drill_monitoring.py`). Здесь проверяется то, что ломается тихо: мера
дрейфа на сдвигах, которые видно глазами, порог, взятый не из замера, признак
опоры и правила лестницы, по которым исход считается сошедшимся. Последнее
важнее всего: правила, которые сходятся всегда, ничего не доказывают.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rufinrag.monitoring import (
    ACT,
    OOV,
    TOKENS,
    MonitoringError,
    QueryShape,
    answered_window,
    answerless,
    answerless_share,
    bin_of,
    counts_of,
    drift,
    event_totals,
    monitoring_rule,
    psi,
    read_events,
    shape_of_event,
    shape_of_text,
    token_edges,
    youden,
)

ROOT = Path(__file__).resolve().parents[1]
# Словарь индекса хранит основы слов, а не слова: «кредит» лежит в нем как «кред».
VOCAB = {"кред", "норматив", "590-п", "резерв"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drill = load_script("drill_monitoring")
check = load_script("check_monitoring")


def shape(tokens: int, found: int | None = None, act: bool = False, score: float | None = 50.0):
    return QueryShape(
        tokens=tokens,
        found=tokens if found is None else found,
        act_number=act,
        hits=0 if score is None else 10,
        top_score=score,
    )


def write_rule(root: Path, body: dict | None, passed: bool = True) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results/monitoring-thresholds.json").write_text(
        json.dumps({"passed": passed, "rule_applied": body}), encoding="utf-8"
    )


def test_same_distribution_gives_zero_psi():
    assert psi([10, 20, 30], [10, 20, 30]) == 0.0


def test_same_shape_on_a_smaller_window_gives_a_small_floor():
    """Сглаживание не инвариантно к размеру окна, и это и есть измеренный шум.

    Окно меньше эталона дает PSI не ноль даже при тех же долях по бинам.
    Величина этого дна входит в потолок шума, из которого выведен порог
    (`results/monitoring-thresholds.json`), а не прячется.
    """
    assert 0 < psi([10, 20, 30], [100, 200, 300]) < 0.05


def test_psi_grows_with_the_shift():
    far = psi([90, 10], [10, 90])
    near = psi([60, 40], [40, 60])
    assert far > near > 0


def test_empty_bin_does_not_give_infinity():
    """Сглаживание: один необычный запрос не обязан поднимать алерт в одиночку."""
    value = psi([50, 0], [25, 25])
    assert 0 < value < 10


def test_bins_mismatch_is_an_error():
    with pytest.raises(MonitoringError):
        psi([1, 2], [1, 2, 3])


def test_token_edges_are_quartiles_of_the_reference():
    assert token_edges([shape(n) for n in range(1, 101)]) == [25, 50, 75]


def test_token_bin_counts_the_edge_as_the_lower_bin():
    edges = [10, 15, 28]
    assert bin_of(TOKENS, shape(10), edges) == 0
    assert bin_of(TOKENS, shape(11), edges) == 1
    assert bin_of(TOKENS, shape(28), edges) == 2
    assert bin_of(TOKENS, shape(29), edges) == 3


def test_zero_oov_is_its_own_bin():
    assert bin_of(OOV, shape(10, found=10), []) == 0
    assert bin_of(OOV, shape(10, found=9), []) == 1
    assert bin_of(OOV, shape(10, found=5), []) == 2
    assert bin_of(OOV, shape(10, found=0), []) == 3


def test_act_number_is_recognised_by_the_tokenizer_pattern():
    assert shape_of_text("что в 590-П про резерв", VOCAB).act_number
    assert shape_of_text("письмо ИН-03-23/109", VOCAB).act_number
    assert not shape_of_text("кредитно-денежный норматив", VOCAB).act_number


def test_unknown_words_are_counted_against_the_index_vocabulary():
    found = shape_of_text("кредит и резерв по неизвестномуслову", VOCAB)
    # «и» и «по» это стоп-слова токенизации индекса, их в запросе нет вовсе.
    assert (found.tokens, found.found) == (3, 2)
    assert round(found.oov_share, 4) == 0.3333


def test_drift_reports_the_worst_feature():
    reference = [shape(5, act=False) for _ in range(100)]
    window = [shape(5, act=True) for _ in range(50)]
    found = drift(window, reference, [10, 15, 28])
    assert found["worst_feature"] == ACT
    assert found["psi"] == found["by_feature"][ACT] > 0
    assert found["by_feature"][TOKENS] < 0.05


def test_counts_cover_every_query():
    shapes = [shape(5), shape(40, act=True), shape(12, found=6)]
    counts = counts_of(shapes, [10, 15, 28])
    assert all(sum(values) == len(shapes) for values in counts.values())


def test_event_shape_is_read_from_what_the_search_saw():
    """Токены берутся из `query_normalized`, а оценка из первой строки выдачи."""
    event = {
        "kind": "query",
        "outcome": "ok",
        "query": "что там про резервы",
        "query_normalized": "кред норматив неизвестн",
        "hits": 2,
        "results": [{"rank": 1, "score": 12.5}, {"rank": 2, "score": 3.0}],
    }
    found = shape_of_event(event, VOCAB)
    assert (found.tokens, found.found, found.hits, found.top_score) == (3, 2, 2, 12.5)
    assert round(found.top_score_per_token, 4) == 6.25


def test_query_without_hits_has_no_good_answer():
    assert answerless(shape(5, score=None), "top_score", 32.5)


def test_support_below_the_floor_has_no_good_answer():
    assert answerless(shape(5, score=30.0), "top_score", 32.5)
    assert not answerless(shape(5, score=33.0), "top_score", 32.5)


def test_answerless_share_publishes_the_denominator():
    shapes = [shape(5, score=10.0), shape(5, score=40.0), shape(5, score=None)]
    found = answerless_share(shapes, "top_score", 32.5)
    assert (found["answerless"], found["answered"], found["share"]) == (2, 3, 0.6667)
    assert found["no_hits"] == 1


def test_unknown_support_feature_is_an_error():
    with pytest.raises(MonitoringError):
        shape(5).support("ндцг")


def test_youden_finds_the_separating_floor():
    values = [1.0, 2.0, 10.0, 11.0]
    good = [False, False, True, True]
    found = youden(values, good)
    assert found["floor"] == 10.0
    assert found["j"] == 1.0
    assert (found["good_below_floor"], found["bad_above_floor"]) == (0, 0)


def test_youden_counts_both_errors():
    values = [1.0, 9.0, 2.0, 10.0]
    good = [False, False, True, True]
    found = youden(values, good)
    assert found["good_below_floor"] + found["bad_above_floor"] > 0


def test_youden_needs_both_classes():
    with pytest.raises(MonitoringError):
        youden([1.0, 2.0], [True, True])


def test_window_takes_the_last_answered_queries():
    events = [
        {"kind": "query", "outcome": "ok", "received_at": "2026-10-01T00:00:00.000Z"},
        {"kind": "query", "outcome": "rejected", "received_at": "2026-10-02T00:00:00.000Z"},
        {"kind": "feedback", "received_at": "2026-10-03T00:00:00.000Z"},
        {"kind": "query", "outcome": "ok", "received_at": "2026-10-04T00:00:00.000Z"},
    ]
    window = answered_window(events, 1)
    assert [row["received_at"] for row in window] == ["2026-10-04T00:00:00.000Z"]
    assert answered_window(events, 10) == [events[0], events[3]]


def test_totals_separate_rejected_from_answered():
    events = [
        {"kind": "query", "outcome": "ok", "received_at": "2026-10-01T00:00:00.000Z"},
        {"kind": "query", "outcome": "rejected", "received_at": "2026-10-01T00:00:01.000Z"},
        {"kind": "query", "outcome": "unavailable", "received_at": "2026-10-02T00:00:00.000Z"},
        {"kind": "feedback", "received_at": "2026-10-02T00:00:01.000Z"},
    ]
    totals = event_totals(events)
    assert totals["queries"] == 3
    assert (totals["answered"], totals["rejected"], totals["unavailable"]) == (1, 1, 1)
    assert (totals["feedback"], totals["days"]) == (1, 2)


def test_events_are_read_in_order_of_the_event_time(tmp_path):
    (tmp_path / "queries-2026-10-02.jsonl").write_text(
        '{"kind":"query","received_at":"2026-10-02T00:00:00.000Z"}\n\n', encoding="utf-8"
    )
    (tmp_path / "queries-2026-10-01.jsonl").write_text(
        '{"kind":"query","received_at":"2026-10-01T00:00:00.000Z"}\n', encoding="utf-8"
    )
    rows = read_events(tmp_path)
    assert [row["received_at"][:10] for row in rows] == ["2026-10-01", "2026-10-02"]


def test_broken_line_names_the_file_and_the_line(tmp_path):
    (tmp_path / "queries-2026-10-01.jsonl").write_text("{нет\n", encoding="utf-8")
    with pytest.raises(MonitoringError, match="строка 1"):
        read_events(tmp_path)


def test_thresholds_come_from_the_measurement(tmp_path):
    write_rule(tmp_path, {"window": 50, "drift_threshold": 0.7})
    assert monitoring_rule(tmp_path)["drift_threshold"] == 0.7


def test_failed_measurement_is_not_a_threshold(tmp_path):
    """Замер, который не отличил живые вопросы от эталона, числом по умолчанию не заменяется."""
    write_rule(tmp_path, {"window": 50, "drift_threshold": 0.7}, passed=False)
    with pytest.raises(MonitoringError):
        monitoring_rule(tmp_path)


def test_missing_measurement_is_an_error(tmp_path):
    with pytest.raises(MonitoringError, match="monitoring-thresholds"):
        monitoring_rule(tmp_path)


def test_regression_signal_reads_the_alert_block(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/regression.json").write_text(
        json.dumps(
            {
                "queries_scored": 991,
                "measured_at": "2026-10-24",
                "passed": False,
                "alert": {"fired": True, "drop": 0.0216, "threshold": 0.02, "text": "упал"},
            }
        ),
        encoding="utf-8",
    )
    signal = check.regression_signal(tmp_path)
    assert signal["fired"] and signal["text"] == "упал"
    assert signal["report_has_alert_block"]


def test_regression_signal_survives_a_report_without_the_alert_block(tmp_path):
    """Отчет прогона постарше блока `alert` не должен делать мониторинг слепым."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results/regression.json").write_text(
        json.dumps(
            {
                "queries_scored": 991,
                "measured_at": "2026-10-23",
                "passed": False,
                "baseline": {"measured_at": "2026-10-22", "drop": 0.03, "tolerance": 0.02},
            }
        ),
        encoding="utf-8",
    )
    signal = check.regression_signal(tmp_path)
    assert signal["fired"] and not signal["report_has_alert_block"]
    assert "0.03" in signal["text"] and "991" in signal["text"]


def test_missing_regression_report_is_not_an_alert(tmp_path):
    signal = check.regression_signal(tmp_path)
    assert not signal["fired"] and not signal["measured"]


RULE = {"window": 50, "drift_threshold": 0.7, "share_threshold": 0.4}


def rung(
    name: str,
    kind: str,
    psi_value: float | None,
    share: float | None,
    drift_fired: bool = False,
    share_fired: bool = False,
    status: str = "watching",
    text: str | None = None,
    events: int = 50,
) -> dict:
    fired = drift_fired or share_fired
    parts = []
    if drift_fired:
        parts.append(f"дрейф {psi_value} при пороге {RULE['drift_threshold']} на окне {events}")
    if share_fired:
        parts.append(f"доля {share} при пороге {RULE['share_threshold']} из {events}")
    return {
        "name": name,
        "kind": kind,
        "events": events,
        "status": status,
        "exit_code": 1 if fired else 0,
        "alert_fired": fired,
        "alert_text": text if text is not None else ". ".join(parts),
        "signals": {
            "query_drift": {
                "fired": drift_fired,
                "value": psi_value,
                "threshold": RULE["drift_threshold"],
            },
            "answerless_share": {
                "fired": share_fired,
                "value": share,
                "threshold": RULE["share_threshold"],
            },
        },
    }


def passing_ladder() -> list[dict]:
    return [
        rung("эталонная смесь", "control", 0.0578, 0.18),
        rung("короткое окно", "short", None, None, status="dormant", events=49),
        rung("живых 50 из 50", "drift", 1.1828, 0.02, drift_fired=True),
        rung("без опоры 25 из 50", "answerless", 0.4724, 0.5, share_fired=True),
    ]


def test_passing_ladder_passes():
    assert drill.problems(passing_ladder(), RULE) == []


def test_ladder_without_a_control_step_is_a_failure():
    rungs = [row for row in passing_ladder() if row["kind"] != "control"]
    assert any("эталонной смеси" in line for line in drill.problems(rungs, RULE))


def test_alert_on_the_reference_mix_is_a_failure():
    """Алерт, который кричит на окне из той же совокупности, бесполезен как мертвый."""
    rungs = passing_ladder()
    rungs[0] = rung("эталонная смесь", "control", 0.0578, 0.18, drift_fired=True)
    assert any("эталонной смеси подняла" in line for line in drill.problems(rungs, RULE))


def test_short_window_that_woke_the_rule_is_a_failure():
    rungs = passing_ladder()
    rungs[1] = rung("короткое окно", "short", 0.9, 0.5, status="watching", events=49)
    problems = drill.problems(rungs, RULE)
    assert any("не усыпило правило" in line for line in problems)


def test_silent_signal_above_the_threshold_is_a_failure():
    rungs = passing_ladder()
    rungs[2] = rung("живых 50 из 50", "drift", 1.1828, 0.02)
    assert any("а алерта нет" in line for line in drill.problems(rungs, RULE))


def test_signal_fired_inside_the_threshold_is_a_failure():
    rungs = passing_ladder()
    rungs[0] = rung("эталонная смесь", "control", 0.0578, 0.18, share_fired=True)
    assert any("а алерт сработал" in line for line in drill.problems(rungs, RULE))


def test_ladder_without_a_single_drift_alert_is_a_failure():
    rungs = [row for row in passing_ladder() if row["kind"] != "drift"]
    problems = drill.problems(rungs, RULE)
    assert any("ни одна ступень не подняла query_drift" in line for line in problems)


def test_alert_text_without_the_number_is_a_failure():
    rungs = passing_ladder()
    rungs[2] = rung("живых 50 из 50", "drift", 1.1828, 0.02, drift_fired=True, text="дрейф есть")
    assert any("в тексте алерта нет" in line for line in drill.problems(rungs, RULE))


def test_exit_code_must_follow_the_alert():
    rungs = passing_ladder()
    rungs[2]["exit_code"] = 0
    assert any("а проверка вернула ноль" in line for line in drill.problems(rungs, RULE))


def test_measured_ladder_is_the_one_that_passed():
    """Отчет прогона и правила проверки сверяются друг с другом, а не живут рядом."""
    report = json.loads((ROOT / "results/monitoring-drill.json").read_text(encoding="utf-8"))
    assert report["passed"]
    assert drill.problems(report["rungs"], report["rule"]) == []


def test_published_thresholds_passed_their_own_check():
    report = json.loads((ROOT / "results/monitoring-thresholds.json").read_text(encoding="utf-8"))
    assert report["passed"]
    rule = report["rule_applied"]
    chosen = next(row for row in report["ladder"] if row["window"] == rule["window"])
    assert chosen["drift_threshold"] < chosen["cap"]
    assert rule["drift_threshold"] > chosen["noise_psi_max"]
