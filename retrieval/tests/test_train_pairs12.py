"""Обучающие пары этапа 12: ни одной пары на запрещенном фрагменте, ни одного выброшенного вопроса."""
from __future__ import annotations

import json
import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_pairs12 as T  # noqa: E402

from rufin.leak import forbidden_chunks  # noqa: E402

LIVE12 = ROOT / "data" / "interim" / "live12"


def test_zapret_i_vopros_bez_pozitiva():
    act_of = {"A#1": "A", "A#2": "A", "A#3": "A", "B#1": "B", "B#2": "B", "C#1": "C", "C#2": "C"}
    labels = {("q1", "A#1"): 2, ("q1", "A#2"): 0, ("q1", "A#3"): 1, ("q2", "B#1"): 1}
    qs = [{"query_id": "q1", "text": "x", "act": "A"}, {"query_id": "q2", "text": "y", "act": "B"}]
    rows = T.build(qs, labels, act_of, frozenset({"A#3", "B#1", "C#1"}), "live", random.Random(0))
    assert [r["query_id"] for r in rows] == ["q1"]
    r = rows[0]
    assert r["pos"] == {"A#1": 2} and r["neg"] == ["A#2"]
    assert set(r["neg_random"]) <= {"B#2", "C#2"}


@pytest.mark.data
def test_nastoyashchie_pary():
    bad = forbidden_chunks()
    ids = lambda name: {json.loads(l)["query_id"] for l in open(LIVE12 / name, encoding="utf-8")}  # noqa: E731
    dropped = (ids("synth_q.jsonl") - ids("synth_q_clean.jsonl")) | (ids("train_live.jsonl") - ids("train_live_clean.jsonl"))
    for name in ("live.jsonl", "live_synth.jsonl"):
        for line in open(LIVE12 / "train" / name, encoding="utf-8"):
            r = json.loads(line)
            assert r["query_id"] not in dropped
            assert r["pos"]
            assert not (set(r["pos"]) | set(r["neg"]) | set(r["neg_random"])) & bad
