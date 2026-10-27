"""Таск MTEB для набора ru-fin-retrieval.

Зачем он нужен. Свой бенчмарк, который прогоняется только своим кодом, —
это утверждение, а не измерение: проверить его снаружи нельзя. Набор лежит
в формате BeIR (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), и этот
таск отдаёт его библиотеке `mteb`, ничего в наборе не меняя.

Разделение одно — `test`: 691 запрос, всё размеченное кроме dev. Дев лежит
отдельным файлом `qrels/dev.tsv` и в таск не входит: он нужен обучению, а
не сравнению моделей. Одной цифрой по `test` пользоваться нельзя, это смесь
речевых режимов (живые вопросы, ручные запросы, невиданные акты, синтетика):
разбивку по подвыборкам печатает `make metrics` своим кодом.

Проверено на **mteb 2.26.27** (27.10.2026, RTX 3070): таск собирается,
читает опубликованный набор с HuggingFace и считает метрику. Чужой код дал
NDCG@10 **0,42255** у `multilingual-e5-small`, наш на тех же 691 запросах
**0,42064**, разница 0,0019. Прежняя версия таска не загружалась ни одной
настоящей версией mteb, и это выяснилось только прогоном: в метаданных стояли
значения, которых нет в словарях схемы (`category="s2p"`,
`annotations_creators="mixed"`). Поэтому таск держится на проверке делом,
а не на чтении документации.

Поле `title` в корпус не передается, и это не мелочь: в формате BeIR оно
лежит отдельным полем, а заголовок акта **уже стоит внутри текста каждого
фрагмента** (проверено на 3 000 фрагментах: 3 000 из 3 000). Склейка
`title` и `text`, которую делает стандартный загрузчик, добавляет к каждому
документу 9,3 процента дублирующего префикса (медианы 198 и 1 930 знаков) и
стоит 0,031 NDCG@10: тем же прогоном со склейкой выходит 0,39183 против
0,42255 без нее. Числа проекта считаны по тексту фрагмента, и таск повторяет
ровно это.

Откуда берутся файлы. Если задана переменная `RU_FIN_RETRIEVAL_PATH`, они
читаются из этой папки (так удобно проверять то, что лежит на диске). Иначе
скачиваются из опубликованного набора на HuggingFace. Структуры, которые
ждёт mteb 2, собираются здесь же: загрузчик самой библиотеки умеет читать
только её собственную раскладку на хабе, а набор опубликован в BeIR.

Запуск:

    import mteb
    from ru_fin_retrieval import RuFinRetrieval

    model = mteb.get_model("intfloat/multilingual-e5-small")
    mteb.MTEB(tasks=[RuFinRetrieval()]).run(model, output_folder="results")
"""
from __future__ import annotations

import json
import os

from datasets import Dataset
from mteb import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval

LOCAL = os.environ.get("RU_FIN_RETRIEVAL_PATH")
HF_NAME = "nata2627/ru-fin-retrieval"
ФАЙЛЫ = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")


class RuFinRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RuFinRetrieval",
        description=(
            "Поиск по нормативным актам Банка России: Положения, Инструкции, "
            "Указания, Методические рекомендации и Официальные разъяснения "
            "2013–2026 годов, нарезанные на фрагменты. Запросы — настоящие "
            "вопросы поднадзорных организаций из раздела «Разъяснения» Банка "
            "России, короткие поисковые запросы, написанные руками, "
            "и синтетические вопросы к актам, не участвовавшим в обучении."),
        reference="https://github.com/nata2627/ru-fin-rag/tree/main/retrieval",
        dataset={"path": HF_NAME, "revision": "main"},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2013-01-01", "2026-09-29"),
        domains=["Legal", "Financial", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="mit",
        # Разметку делала модель-судья, человек проверил пробу из 150 пар
        # (каппа 0,408, взвешенная 0,611). Это и есть «LM-generated and
        # reviewed» в словаре mteb.
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        # Живые вопросы найдены на сайте Банка России, синтетические написаны
        # моделью и проверены фильтрами: в словаре mteb такая смесь это
        # «multiple».
        sample_creation="multiple",
        bibtex_citation="""@misc{ru-fin-retrieval,
  title  = {ru-fin-retrieval: бенчмарк поиска по нормативным актам Банка России},
  author = {Гордеева, Наталия},
  year   = {2026},
  url    = {https://github.com/nata2627/ru-fin-rag/tree/main/retrieval}
}""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        корпус, запросы, разметка = _прочитать(LOCAL or _скачать())
        self.dataset = {
            "default": {
                "test": {
                    "corpus": корпус,
                    "queries": запросы,
                    "relevant_docs": разметка,
                    "top_ranked": None,
                }
            }
        }
        self.dataset_transform()
        self.data_loaded = True


def _скачать() -> str:
    """Забрать опубликованный набор с HuggingFace и вернуть папку с ним."""
    from huggingface_hub import hf_hub_download

    куда = None
    for имя in ФАЙЛЫ:
        путь = hf_hub_download(HF_NAME, имя, repo_type="dataset")
        if имя == "corpus.jsonl":
            куда = os.path.dirname(путь)
    assert куда is not None
    return куда


def _прочитать(корень: str) -> tuple[Dataset, Dataset, dict[str, dict[str, int]]]:
    """Собрать из трёх файлов BeIR то, что ждёт mteb 2: два Dataset и словарь."""
    ids: list[str] = []
    texts: list[str] = []
    with open(os.path.join(корень, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["_id"])
            texts.append(d["text"])
    # Без колонки title: иначе mteb склеит её с текстом, в котором заголовок
    # акта уже есть. Цена склейки измерена и равна 0,031 NDCG@10.
    корпус = Dataset.from_dict({"id": ids, "text": texts})

    qids: list[str] = []
    qtexts: list[str] = []
    with open(os.path.join(корень, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["_id"])
            qtexts.append(d["text"])
    запросы = Dataset.from_dict({"id": qids, "text": qtexts})

    разметка: dict[str, dict[str, int]] = {}
    with open(os.path.join(корень, "qrels", "test.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")[:3]
            разметка.setdefault(qid, {})[cid] = int(score)
    return корпус, запросы, разметка
