"""Таблицы документа 17 сходятся с файлами, которые их посчитали.

Числа в документ набираются руками, и это место, где текст тихо расходится
с данными: пересчитали метрики, поменялась выдача, переехала подвыборка.
Поэтому каждая клетка сверяется с источником: качество с
`data/results/base.json` (`make metrics`), задержка и размер индекса с
`data/results/latency_base_user-bge-m3.json` (`make latency-matryoshka`).

Сверки с этими двумя файлами помечены `data`: оба воспроизводимы, с 28.10
каталог `data/results` лежит в репозитории, а пометка осталась для случая,
когда его нет на диске. Без них остаются две
проверки, которым файлы не нужны: знаменатели подвыборок (их состав
в репозитории лежит) и размеры индексов, которые считаются арифметикой.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ДОК = ROOT / "docs" / "17-podvyborki-i-zaderzhka.md"
МЕТРИКИ = ROOT / "data" / "results" / "base.json"
ЗАДЕРЖКА = ROOT / "data" / "results" / "latency_base_user-bge-m3.json"
СОСТАВ = ROOT / "data" / "queries" / "podvyborki.json"
РЕРАНК_КАРТА = ROOT / "data" / "results" / "latency_base_user-bge-m3_rerank.json"
РЕРАНК_ПРОЦЕССОР = (ROOT / "data" / "results"
                    / "latency_base_user-bge-m3_rerank_cpu_proba.json")

ФРАГМЕНТОВ = 62594
# порядок колонок таблицы качества после подписи строки и числа запросов
СВЯЗКИ = ("bm25", "dense-bge-m3", "dense-user-bge-m3", "hybrid", "hybrid-rerank")
# её строки: подпись в документе -> подвыборка в base.json. None означает
# «по всему размеченному», там метрика лежит в другом ключе отчёта
СТРОКИ = {"все размеченные": None, "заголовочная": "заголовочная",
          "живые": "живые", "живые без 590-П": "живые без 590-П",
          "невиданные акты": "невиданные акты", "dev": "dev",
          "синтетика": "синтетика"}
# размерность матрёшки -> имя модели в отчётах
РАЗМЕРНОСТИ = {1024: "user-bge-m3", 512: "user-bge-m3-512",
               256: "user-bge-m3-256", 128: "user-bge-m3-128"}

ИНТЕРВАЛ = re.compile(r"^(\d,\d{3}) \[(\d,\d{3}); (\d,\d{3})\]$")


@pytest.fixture
def таблицы() -> list[list[list[str]]]:
    """Таблицы документа блоками: подряд идущие строки с вертикальной чертой.

    Блоками, а не одним списком строк, потому что подписи строк в документе
    повторяются: «живые» стоит и в составе подвыборок, и в таблице качества.
    """
    текст = ДОК.read_text(encoding="utf-8")
    assert "—" not in текст, "длинное тире в документе"
    блоки: list[list[list[str]]] = []
    текущий: list[list[str]] = []
    for строка in текст.splitlines():
        if строка.startswith("|"):
            текущий.append([c.strip() for c in строка.strip().strip("|").split("|")])
        elif текущий:
            блоки.append(текущий)
            текущий = []
    if текущий:
        блоки.append(текущий)
    return блоки


def таблица(блоки: list[list[list[str]]], по_шапке: str) -> dict[str, list[str]]:
    """Таблица, в шапке которой есть этот текст, в виде «подпись -> клетки»."""
    нашлись = [b for b in блоки if any(по_шапке in c for c in b[0])]
    assert len(нашлись) == 1, f"таблиц с «{по_шапке}» в документе {len(нашлись)}"
    return {строка[0].strip("*`"): строка[1:] for строка in нашлись[0][2:]}


def число(ячейка: str) -> float:
    return float(ячейка.replace(",", "."))


def интервал(ячейка: str) -> tuple[float, float, float]:
    m = ИНТЕРВАЛ.match(ячейка)
    assert m, f"клетка «{ячейка}» не в виде «0,571 [0,547; 0,595]»"
    низ, верх = число(m.group(2)), число(m.group(3))
    return число(m.group(1)), низ, верх


def test_znamenateli_iz_sostava_podvyborok(таблицы):
    """Сколько запросов в подвыборке, берётся из её состава, а не на глаз."""
    состав = json.loads(СОСТАВ.read_text(encoding="utf-8"))
    for подпись, клетки in таблица(таблицы, "что это").items():
        assert число(клетки[0]) == len(состав[подпись]), подпись
    качество = таблица(таблицы, "BM25")
    for подпись, имя in СТРОКИ.items():
        ждём = 991 if имя is None else len(состав[имя])
        assert число(качество[подпись][0]) == ждём, подпись


def test_razmer_indeksa_eto_arifmetika(таблицы):
    """На диске матрица лежит в float16, то есть ровно два байта
    на измерение на фрагмент, а под поиск вдвое больше."""
    строки = таблица(таблицы, "индекс на диске, МБ")
    for dim in РАЗМЕРНОСТИ:
        клетки = строки[str(dim)]
        assert число(клетки[0]) == round(ФРАГМЕНТОВ * dim * 2 / 1048576, 1)
        assert число(клетки[1]) == round(ФРАГМЕНТОВ * dim * 4 / 1048576, 1)


@pytest.mark.data
def test_kachestvo_po_podvyborkam(таблицы):
    if not МЕТРИКИ.exists():
        pytest.skip("метрики не посчитаны: нужен `make metrics`")
    отчёт = json.loads(МЕТРИКИ.read_text(encoding="utf-8"))
    строки = таблица(таблицы, "BM25")
    for подпись, имя in СТРОКИ.items():
        for i, связка in enumerate(СВЯЗКИ, start=1):
            ci = (отчёт["metrics"][связка]["NDCG@10"] if имя is None
                  else отчёт["podvyborki"][имя][связка])
            ждём = (round(ci["mean"], 3), round(ci["lo"], 3), round(ci["hi"], 3))
            assert интервал(строки[подпись][i]) == ждём, f"{подпись} / {связка}"


@pytest.mark.data
def test_kachestvo_urezannyh_razmernostey(таблицы):
    if not МЕТРИКИ.exists():
        pytest.skip("метрики не посчитаны: нужен `make metrics`")
    отчёт = json.loads(МЕТРИКИ.read_text(encoding="utf-8"))
    строки = таблица(таблицы, "по всем 991")
    for dim, модель in РАЗМЕРНОСТИ.items():
        клетки = строки[str(dim)]
        источники = [отчёт["metrics"][f"dense-{модель}"]["NDCG@10"],
                     отчёт["podvyborki"]["заголовочная"][f"dense-{модель}"],
                     отчёт["podvyborki"]["живые"][f"dense-{модель}"]]
        for клетка, ci in zip(клетки, источники):
            ждём = (round(ci["mean"], 3), round(ci["lo"], 3), round(ci["hi"], 3))
            assert интервал(клетка) == ждём, f"{dim} / {клетка}"


@pytest.mark.data
def test_zaderzhka_iz_zamera(таблицы):
    if not ЗАДЕРЖКА.exists():
        pytest.skip("задержка не замерена: нужен `make latency-matryoshka`")
    отчёт = json.loads(ЗАДЕРЖКА.read_text(encoding="utf-8"))
    assert отчёт["chunk_count"] == ФРАГМЕНТОВ
    строки = таблица(таблицы, "индекс на диске, МБ")
    for dim, модель in РАЗМЕРНОСТИ.items():
        клетки = строки[str(dim)]
        индекс = отчёт["index"][f"dense:{модель}"]
        assert индекс["dim"] == dim
        assert число(клетки[0]) == индекс["disk_mb"]
        assert число(клетки[1]) == индекс["memory_mb"]
        for i, ключ in enumerate((f"dense:{модель}", f"hybrid:{модель}")):
            лат = отчёт["latency"][ключ]
            assert число(клетки[2 + 2 * i]) == лат["median_ms"], f"{dim} {ключ}"
            assert число(клетки[3 + 2 * i]) == лат["p95_ms"], f"{dim} {ключ} p95"


@pytest.mark.data
def test_zaderzhka_reranka_iz_zamerov():
    """Числа переранжирования стоят в тексте, а не в таблице, и тоже из файлов.

    Прозу проверять так же важно, как таблицу: оба числа замерены на чужой
    машине, повторить их на маке нельзя, и разойтись с отчетом им негде только
    пока их кто-то сверяет.
    """
    if not РЕРАНК_КАРТА.exists() or not РЕРАНК_ПРОЦЕССОР.exists():
        pytest.skip("переранжирование не замерено на внешней машине")
    текст = ДОК.read_text(encoding="utf-8")
    карта = json.loads(РЕРАНК_КАРТА.read_text(encoding="utf-8"))
    процессор = json.loads(РЕРАНК_ПРОЦЕССОР.read_text(encoding="utf-8"))
    ключ = "hybrid+rerank:user-bge-m3"
    на_карте = карта["latency"][ключ]
    на_процессоре = процессор["latency"][ключ]
    assert карта["index"]["reranker"]["device"] == "cuda"
    assert процессор["index"]["reranker"]["device"] == "cpu"
    for значение in (на_карте["median_ms"], на_карте["p95_ms"], на_процессоре["median_ms"]):
        напечатано = f"{значение}".replace(".", ",")
        assert напечатано in текст, f"{напечатано} в документе нет"
    # Оговорка про один запрос обязана стоять рядом с числом: медиана и p95 у
    # замера на процессоре совпадают по построению, и без оговорки это читается
    # как устойчивость по запросам.
    assert на_процессоре["queries"] == 1
    assert "Запрос при этом **один**" in текст
    раз = на_процессоре["median_ms"] / на_карте["median_ms"]
    assert f"в {раз:.1f} раза".replace(".", ",") in текст
