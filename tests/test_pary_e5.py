"""E5: правило приемки пары и подбор самих пар.

Опыт меняет одно: обучающий вопрос пишется по двум связанным нормам.
Проверять это нечем, если пара фиктивная, то есть второй фрагмент
к вопросу отношения не имеет. Поэтому правило приемки пары записано
до прогона (S26, пункт 3) и проверяется здесь, а не на карте: первый
эталон в лучшей половине кандидатов, второй не ниже тридцатого места.

Это же правило считает долю многофрагментных вопросов, по которой проба
останавливает прогон до обучения. Ошибка в нем выглядела бы как успех
опыта, а не как падение теста.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

import gpu_gen as G  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "make_pairs_e5", ROOT / "scripts" / "make_pairs_e5.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


@pytest.mark.parametrize("место_1,место_2,принята", [
    (1, 1, True),
    (25, 30, True),
    (25, 31, False),
    (1, 40, False),
    (26, 1, False),
    (0, 1, False),
    (1, 0, False),
])
def test_pravilo_priemki_pary(место_1, место_2, принята):
    assert G.pair_ranks_ok(место_1, место_2, keep_top=25, second_keep=30) is принята


def test_porog_vtorogo_ne_menshe_glubiny_otkaz():
    """Оба эталона вставляются в кандидатов силой, поэтому их место не больше
    глубины. Порог второго, равный глубине, пропускал бы любую пару, а доля
    многофрагментных выходила бы единицей: опыт померил бы сам себя."""
    with pytest.raises(ValueError):
        G.filter_pairs_by_teacher([], [], depth=30, keep_top=15, second_keep=30)


def test_stiley_dva_i_oba_sprashivayut_vopros():
    """Короткий поисковый запрос из режима пары исключен: запросом в три
    слова ситуацию с двумя нормами не задашь."""
    assert len(G.PAIR_STYLES) == 2
    assert all(стиль.question_mark for стиль in G.PAIR_STYLES)


def test_podskazka_soderzhit_oba_fragmenta():
    первый = {"text": "шапка\n\n" + "первая норма про резервы. " * 8}
    второй = {"text": "шапка\n\n" + "вторая норма про отчетность. " * 8}
    пара = {"origin": "между актами", "chunk_id": "a#0", "chunk_id_2": "b#0"}
    текст = G.PAIR_HEAD.format(откуда=G.PAIR_ORIGINS[пара["origin"]],
                               first=G.body_of(первый), second=G.body_of(второй))
    assert "первая норма про резервы" in текст
    assert "вторая норма про отчетность" in текст
    assert "норма 1" in текст and "норма 2" in текст


def test_spisannyy_so_vtorogo_fragmenta_otklonen():
    """Фильтр «списан» применяется к обоим фрагментам пары. Вопрос, собранный
    из слов второго фрагмента, так же сводит задачу к поиску подстроки."""
    первый = ("Кредитная организация оценивает кредитный риск по ссуде "
              "на постоянной основе и выносит профессиональное суждение. ") * 3
    второй = ("Резервные копии электронных баз данных размещаются на одном "
              "или нескольких отчуждаемых машинных носителях информации. ") * 3
    # Корпус для обратной частоты слов разнообразный нарочно: на трех
    # документах редких слов не бывает вовсе, и вопрос отклонялся бы
    # как подходящий к слишком многим фрагментам, то есть не тем фильтром.
    корпус = [f"посторонний документ номер {i} про лицензии и надзор"
              for i in range(200)] + [первый, второй]
    idf = G.IdfTable(корпус)
    списан = ("Резервные копии электронных баз данных размещаются на одном "
              "или нескольких отчуждаемых машинных носителях?")
    ok, почему = G.judge_pair_query(списан, первый, второй, idf)
    assert not ok
    assert "списан" in почему


@pytest.mark.parametrize("units,ожидание", [
    ("1.1–1.4", ((1, 1), (1, 4))),
    ("2.1.13", ((2, 1, 13), (2, 1, 13))),
    ("", None),
    ("приложение", None),
])
def test_razbor_punktov(units, ожидание):
    assert P.parse_units(units) == ожидание


@pytest.mark.parametrize("первый,второй,пересекаются", [
    ("1.1–1.4", "1.4–1.5.4", True),
    ("1.1–1.4", "2.1–2.3", False),
    ("2.3", "2.3", True),
    ("", "2.3", True),
    # Вложенность: пункт и его подпункт это одна норма на двух уровнях,
    # а лексикографическое сравнение диапазонов их пропускало.
    ("1.5", "1.5.4", True),
    ("1.1", "1", True),
    ("5.2–5.3", "4–5", True),
    ("1.5.4", "1.6", False),
])
def test_peresechenie_punktov(первый, второй, пересекаются):
    assert P.units_overlap({"units": первый}, {"units": второй}) is пересекаются


def test_pary_sobrany_po_pravilam():
    """Сам файл пар: ни одной пары из dev и теста, соседние не взяты,
    пункты не пересекаются."""
    путь = ROOT / "data" / "queries" / "pairs_e5.jsonl"
    if not путь.exists():
        pytest.skip("пары не собраны: `python3 scripts/make_pairs_e5.py`")
    sys.path.insert(0, str(ROOT / "src"))
    from rufin import split as S

    split = S.load(str(ROOT / "data" / "queries" / "split.json"))
    by_act = S.group_by_act(split)
    with open(путь, encoding="utf-8") as f:
        пары = [json.loads(line) for line in f if line.strip()]
    assert пары
    for пара in пары:
        assert by_act.get(пара["act_id"]) == S.TRAIN
        assert by_act.get(пара["act_id_2"]) == S.TRAIN
        assert пара["chunk_id"] != пара["chunk_id_2"]
        assert пара["bm25"] >= 20
        if пара["origin"] == "внутри акта":
            assert abs(пара["position"] - пара["position_2"]) >= 2
            assert not P.units_overlap({"units": пара["units"]},
                                       {"units": пара["units_2"]})
        else:
            assert пара["act_id"] != пара["act_id_2"]
            assert пара["reference"] == пара["number_2"]
    способы = {пара["origin"] for пара in пары}
    assert способы == {"внутри акта", "между актами"}


class ПоддельныйТокенизатор:
    def apply_chat_template(self, messages, tokenize=False,
                           add_generation_prompt=True, **kwargs) -> str:
        return "\n".join(m["content"] for m in messages)


class ПоддельныйВывод:
    def __init__(self, text: str) -> None:
        self.outputs = [type("o", (), {"text": text})()]


class ПоддельныйДвижок:
    """Движок, который пишет вопрос по заданным текстам, не поднимая vLLM.

    Нужен затем, что ошибка в сборке подсказки или в полях строки обходится
    в прогон на карте: квота Kaggle 30 часов в неделю, и проект ее дважды
    терял. Здесь проверяется весь путь генерации по паре, кроме самой модели.
    """

    def __init__(self, ответы: list[str]) -> None:
        self.ответы = ответы
        self.подсказки: list[str] = []

    def get_tokenizer(self):
        return ПоддельныйТокенизатор()

    def generate(self, prompts, params, use_tqdm=False):
        self.подсказки += list(prompts)
        return [ПоддельныйВывод(self.ответы[i % len(self.ответы)])
                for i in range(len(prompts))]


def поддельный_vllm(monkeypatch) -> None:
    модуль = type(sys)("vllm")
    модуль.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", модуль)


def фрагмент(номер: int, текст: str) -> dict:
    return {"chunk_id": f"акт{номер // 2}#{номер:04d}", "act_id": f"акт{номер // 2}",
            "number": f"{номер}-П", "date": "01.01.2020", "title": "Положение",
            "section": "", "units": f"{номер}.1", "position": номер,
            "text": "шапка акта\n\n" + текст}


def test_generaciya_po_pare_na_poddelnom_dvizhke(monkeypatch, tmp_path):
    поддельный_vllm(monkeypatch)
    тексты = [
        ("Кредитная организация оценивает кредитный риск по ссуде на постоянной "
         "основе и выносит профессиональное суждение о категории качества. "),
        ("Резервные копии баз данных хранятся на отчуждаемых носителях и "
         "представляются в Банк России вместе с паспортом копий. "),
        ("Страховщик обеспечивает функционирование информационных систем для "
         "обмена сведениями о заключенных договорах страхования. "),
        ("Управляющая компания раскрывает оценочную стоимость активов фонда "
         "и состав имущества на своем официальном сайте. "),
    ]
    chunks = [фрагмент(i, текст * 4) for i, текст in enumerate(тексты)]
    пары = [{"pair_id": "pr00000", "origin": "внутри акта", "bm25": 80.0,
             "chunk_id": chunks[0]["chunk_id"], "chunk_id_2": chunks[1]["chunk_id"],
             "act_id": chunks[0]["act_id"], "act_id_2": chunks[1]["act_id"],
             "units": chunks[0]["units"], "units_2": chunks[1]["units"]},
            {"pair_id": "pr00001", "origin": "между актами", "bm25": 95.0,
             "chunk_id": chunks[2]["chunk_id"], "chunk_id_2": chunks[3]["chunk_id"],
             "act_id": chunks[2]["act_id"], "act_id_2": chunks[3]["act_id"],
             "units": chunks[2]["units"], "units_2": chunks[3]["units"]}]
    вопрос = ("Как часто пересматривается категория качества и какие документы "
              "прилагаются к передаваемым в надзорный орган копиям?")
    движок = ПоддельныйДвижок([вопрос])
    корпус = [f"посторонний документ номер {i} про лицензии и надзор"
              for i in range(200)] + [c["text"] for c in chunks]

    out = tmp_path / "synthetic_train_e5.jsonl"
    stats = G.make_pair_queries(движок, пары, chunks, str(out), target=2,
                               idf_texts=корпус, batch=1)

    assert stats["accepted"] == 2
    строки = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert {s["pair_origin"] for s in строки} == {"внутри акта", "между актами"}
    for строка in строки:
        assert строка["gold_chunk_id"] != строка["gold_chunk_id_2"]
        assert строка["style"] in {с.name for с in G.PAIR_STYLES}
    # В подсказке стоят оба фрагмента пары, иначе вопрос по двум нормам
    # взяться ниоткуда не может.
    assert "Кредитная организация оценивает" in движок.подсказки[0]
    assert "Резервные копии баз данных" in движок.подсказки[0]


def test_filtr_pary_beret_uchitelya_iz_vhodov_i_schitaet_meru(monkeypatch):
    """Два места разом: откуда берутся веса учителя и что считает мера.

    Веса: ядру, созданному командой, токен HuggingFace прицепить нельзя,
    и при выключенном интернете загрузка из хаба падает уже после того, как
    BM25 построен. На этом прогон фильтра пробы E5 и встал.

    Мера: доля многофрагментных считается по всем поданным вопросам, а рядом
    печатается доля при первом эталоне в лучшей половине. Порог остановки
    0,5 сравнивается с первой, и спутать их нельзя.
    """
    записано: dict = {}
    порядки = {
        # первый на 1, второй на 2: пара принимается
        "q1": ["c000", "c001"] + [f"c{i:03d}" for i in range(2, 50)],
        # первый на 1, второй на 40: второй к вопросу отношения не имеет
        "q2": ["c002"] + [f"c{i:03d}" for i in range(10, 48)] + ["c003"],
        # первый на 30: в лучшую половину не попал
        "q3": [f"c{i:03d}" for i in range(20, 49)] + ["c004", "c005"],
    }

    поиск = type(sys)("gpu_search")
    поиск.RERANKER = "BAAI/bge-reranker-v2-m3"

    def rerank_runs(queries, candidates, texts, model_path=None, batch_size=32,
                    device="cuda"):
        записано["путь"] = model_path
        записано["кандидатов"] = {q["query_id"]: len(candidates[q["query_id"]])
                                  for q in queries}
        записано["списки"] = {q["query_id"]: [c for c, _ in candidates[q["query_id"]]]
                             for q in queries}
        return {q["query_id"]: порядки[q["query_id"]] for q in queries}

    поиск.rerank_runs = rerank_runs
    общее = type(sys)("gpu_common")
    общее.resolve_model = lambda hf_id, *a, **k: "/kaggle/input/reranker/weights"
    monkeypatch.setitem(sys.modules, "gpu_search", поиск)
    monkeypatch.setitem(sys.modules, "gpu_common", общее)

    chunks = [{"chunk_id": f"c{i:03d}",
               "text": f"пункт {i}. Кредитная организация оценивает риск и "
                       f"ведет учет операций номер {i} по правилам главы {i}."}
              for i in range(60)]
    queries = [
        {"query_id": "q1", "text": "как оценивается риск и как ведется учет операций?",
         "gold_chunk_id": "c000", "gold_chunk_id_2": "c001"},
        {"query_id": "q2", "text": "какие правила главы применяются к учету операций?",
         "gold_chunk_id": "c002", "gold_chunk_id_2": "c003"},
        {"query_id": "q3", "text": "кто ведет учет операций по правилам главы?",
         "gold_chunk_id": "c004", "gold_chunk_id_2": "c005"},
    ]

    kept, stats = G.filter_pairs_by_teacher(queries, chunks, depth=50, keep_top=25,
                                           second_keep=30, device="cpu")

    assert записано["путь"] == "/kaggle/input/reranker/weights"
    assert [q["query_id"] for q in kept] == ["q1"]
    assert kept[0]["teacher_rank"] == 1 and kept[0]["teacher_rank_2"] == 2
    assert stats["доля многофрагментных"] == round(1 / 3, 3)
    assert stats["первый в лучшей половине"] == 2
    assert stats["доля многофрагментных при первом в половине"] == 0.5
    # Оба эталона обязаны оказаться среди кандидатов, и длина списка от этого
    # не растет: иначе место эталона считалось бы по списку другой глубины.
    assert set(записано["кандидатов"].values()) == {50}
    # Оба эталона обязаны быть в списке у каждого вопроса, в том числе когда
    # BM25 нашел один из них сам: прежний срез выбрасывал найденный эталон
    # из хвоста выдачи, место выходило нулем, и пара отклонялась по длине
    # списка, а не по сути.
    for q in queries:
        список = записано["списки"][q["query_id"]]
        assert q["gold_chunk_id"] in список
        assert q["gold_chunk_id_2"] in список
        assert len(список) == len(set(список))


def test_predel_na_akt_derzhitsya_vnutri_odnoy_pachki(monkeypatch, tmp_path):
    """Предел вопросов на акт считается в момент приемки.

    Счетчик проверялся только при наборе пачки, а пачка это 256 пар. Акт,
    у которого в одной пачке лежало больше предела пар, забирал их все,
    и выборка получала перекос в сторону самых «парных» актов. S26 говорит,
    что предел на акт не меняется, значит он обязан держаться.
    """
    поддельный_vllm(monkeypatch)
    тело = ("Кредитная организация оценивает кредитный риск по ссуде и ведет "
            "учет операций по правилам внутренних документов. ") * 4
    chunks = [{"chunk_id": f"акт0#{i:04d}", "act_id": "акт0", "number": "1-П",
               "date": "01.01.2020", "title": "Положение", "section": "",
               "units": f"{i}.1", "position": i, "text": "шапка\n\n" + тело}
              for i in range(8)]
    пары = [{"pair_id": f"pr{i:05d}", "origin": "внутри акта", "bm25": 80.0,
             "chunk_id": chunks[i]["chunk_id"], "chunk_id_2": chunks[i + 4]["chunk_id"],
             "act_id": "акт0", "act_id_2": "акт0",
             "units": chunks[i]["units"], "units_2": chunks[i + 4]["units"]}
            for i in range(4)]
    вопрос = ("Как часто пересматривается категория качества ссуды и в каком "
              "порядке сведения попадают во внутренние документы?")
    движок = ПоддельныйДвижок([вопрос])
    корпус = [f"посторонний документ номер {i} про лицензии и надзор"
              for i in range(200)] + [c["text"] for c in chunks]

    out = tmp_path / "train.jsonl"
    stats = G.make_pair_queries(движок, пары, chunks, str(out), target=4,
                               per_act=2, idf_texts=корпус, batch=8)

    assert stats["accepted"] == 2
    assert stats["пропущено по пределу на акт"] == 2


def test_kontrolnaya_vetka_pishet_vopros_po_odnomu_fragmentu(monkeypatch, tmp_path):
    """Контроль нужен мере многофрагментности.

    Само значение меры ни о чем не говорит: пара отобрана по BM25, фрагменты
    тематически близки, и однофрагментный вопрос по первому может вытянуть
    второй в тот же топ сам собой. Поэтому тем же фильтром меряется контроль
    по прежней инструкции на тех же парах, и смысл имеет разница.
    """
    поддельный_vllm(monkeypatch)
    тексты = [
        ("Кредитная организация оценивает кредитный риск по ссуде на постоянной "
         "основе и выносит профессиональное суждение о категории качества. "),
        ("Резервные копии баз данных хранятся на отчуждаемых носителях и "
         "представляются в Банк России вместе с паспортом копий. "),
    ]
    chunks = [фрагмент(i, текст * 4) for i, текст in enumerate(тексты)]
    пары = [{"pair_id": "pr00000", "origin": "внутри акта", "bm25": 80.0,
             "chunk_id": chunks[0]["chunk_id"], "chunk_id_2": chunks[1]["chunk_id"],
             "act_id": chunks[0]["act_id"], "act_id_2": chunks[1]["act_id"],
             "units": chunks[0]["units"], "units_2": chunks[1]["units"]}]
    вопрос = ("В каком порядке выносится профессиональное суждение о категории "
              "качества выданной ссуды?")
    движок = ПоддельныйДвижок([вопрос])
    корпус = [f"посторонний документ номер {i} про лицензии и надзор"
              for i in range(200)] + [c["text"] for c in chunks]

    out = tmp_path / "control.jsonl"
    stats = G.make_pair_queries(движок, пары, chunks, str(out), target=1,
                               idf_texts=корпус, batch=1, control=True)

    assert stats["accepted"] == 1
    assert "контроль" in stats["ветка"]
    # В подсказке контроля второго фрагмента нет, иначе это тот же опыт.
    assert "Кредитная организация оценивает" in движок.подсказки[0]
    assert "Резервные копии баз данных" not in движок.подсказки[0]
    # Поля пары при этом сохраняются: фильтр меряет место обоих эталонов
    # и у контроля тоже, иначе мере не с чем сравниваться.
    строка = json.loads(open(out, encoding="utf-8").read().strip())
    assert строка["gold_chunk_id_2"] == chunks[1]["chunk_id"]
    assert строка["style"] in {с.name for с in G.STYLES[:2]}
