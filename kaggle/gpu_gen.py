"""Генерация вопросов на видеокарте: vLLM, три речевых режима, три фильтра.

Чем это отличается от `gpu_queries.py`, который тут уже был. Тот поднимал
Qwen2.5-14B в четырёх битах и генерировал пачками по восемь через
`model.generate`; так было получено 150 вопросов. На пятнадцати тысячах
это десяток часов видеокарты — треть недельной квоты на один прогон,
и двенадцатичасовая сессия может не успеть. Поэтому здесь vLLM и модель
поменьше: вопрос по фрагменту рассуждений не требует, а непрерывная пачка
и общий кэш внимания дают порядок разницы в скорости.

**Про T4 и типы данных.** T4 — это Turing, bf16 на нём нет. Запуск vLLM
с `dtype="bfloat16"` либо падает, либо тихо деградирует, поэтому тип
задаётся явно и по умолчанию `float16`. FlashAttention тоже требует Ampere;
vLLM сам откатится на другой бэкенд, и это нормально на длине 4096.

**Про объём.** Сначала замер на сотне фрагментов, потом назначение объёма
по замеру. Десять тысяч вопросов за три часа лучше пятнадцати, которые
не досчитаются.

**Три речевых режима.** Вопрос, написанный моделью по фрагменту, наследует
его словарь, и набор из одного такого стиля меряет, как поиск сопоставляет
канцелярит с канцеляритом. Поэтому стилей три, и они чередуются: письмо
практика, бытовая ситуация и короткий поисковый запрос — тот самый, который
набирают в строку и у которого нет ни глагола, ни вопросительного знака.

**Фильтры.** Два прежних (`rufin.queryfilter`): вопрос, списанный
с фрагмента, сводит задачу к поиску подстроки и дарит преимущество BM25;
вопрос, подходящий к половине корпуса, правильного ответа не имеет.
Третий добавлен здесь: если учитель-кросс-энкодер не ставит исходный
фрагмент в первые пятьдесят, вопрос выбрасывается. Он ловит то, чего
не ловят первые два, — вопрос, формально безупречный, но отвечает на него
не этот фрагмент.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass

from rufin.queryfilter import (
    MAX_COMMON_RUN,
    MAX_COPY_SHARE,
    IdfTable,
    copy_score,
    is_good_source,
    judge_query,
)

# Генератор обучающей выборки и генератор теста обязаны быть из разных
# семейств: иначе тест меряет, насколько ученик выучил стиль своего же
# генератора. Обе модели записываются в DATASET_CARD.md.
TRAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# Модель для dev и теста — обязательно из другого семейства, чем TRAIN_MODEL:
# иначе тест померит не качество поиска, а то, насколько ученик выучил стиль
# своего же генератора. Но семейство мало выбрать — оно должно ещё и держать
# русский. Mistral-7B-Instruct-v0.2 на пробе 30.09 выдал одиннадцать английских
# вопросов из двадцати в тесте, а среди русских попадались «заполнение графф»
# и «привести в информированное состояние». Vikhr-Nemo — Mistral-Nemo,
# дообученный на русском: семейство по-прежнему не Qwen, а язык родной.
# Запасная сборка, если эта не заведётся:
#   mandanya/Vikhr-Nemo-12B-Instruct-R-21-09-24-AWQ
TEST_MODEL = "NiGuLa/Vikhr-Nemo-12B-Instruct-R-21-09-24-awq-4bit"

# Требование писать по-русски стоит первым и повторяется в требованиях
# к каждому вопросу. Это не перестраховка: модель для теста берётся из
# другого семейства нарочно, а другое семейство хуже держит русский —
# Mistral-7B-Instruct на пробе выдал одиннадцать английских вопросов
# из двадцати. Фильтр такие отбрасывает, но каждый отброшенный оплачен
# временем видеокарты, и дешевле их не порождать.
SYSTEM = ("Ты помогаешь составить набор проверочных вопросов к нормативным актам "
          "Банка России. Пиши ТОЛЬКО по-русски. Отвечай только самим вопросом, "
          "без пояснений и без кавычек.")

HEAD = """Ниже фрагмент нормативного акта Банка России.

--- начало фрагмента ---
{passage}
--- конец фрагмента ---
"""

COMMON = """
Требования:
- вопрос по-русски, целиком; латиница допустима только в обозначениях
  вроде USD, IFRS, SWIFT;
- своими словами, без дословных кусков из фрагмента длиннее трёх слов подряд;
- конкретно: по запросу должен находиться именно этот фрагмент, а не любой
  документ Банка России;
- не упоминай номер и название акта.
"""


@dataclass(frozen=True)
class Style:
    name: str
    tail: str
    question_mark: bool = True
    max_tokens: int = 96
    # Показывать ли примеры живых вопросов (E4). Короткому поисковому запросу
    # они не показываются: живые вопросы это письма в Банк России, и примером
    # письма запрос из трех слов не объяснишь, а испортить его так можно.
    fewshot: bool = True


STYLES = (
    Style("вопрос профессионала", """
Сформулируй ОДИН вопрос, который задал бы риск-аналитик, методолог или
бухгалтер банка, и ответ на который содержится именно в этом фрагменте.
Одно предложение, заканчивается знаком вопроса.
""" + COMMON),
    Style("бытовая ситуация", """
Опиши ОДНУ рабочую ситуацию в двух-трёх предложениях и закончи вопросом,
ответ на который содержится именно в этом фрагменте. Пиши так, как пишут
в банке коллеге: без ссылок на пункты, с обычными словами.
""" + COMMON, max_tokens=160),
    Style("короткий поисковый запрос", """
Напиши ОДИН короткий поисковый запрос — так, как его набрали бы в строку
поиска: три-восемь слов, без глагола, без вопросительного знака, без
вежливых оборотов. Он должен находить именно этот фрагмент.
""" + COMMON, question_mark=False, max_tokens=32, fewshot=False),
)


FEWSHOT = """
Вот как пишут вопросы те, кто приходит с ними в Банк России. Это чужие
вопросы, не про этот фрагмент: смотри на манеру, а не на содержание.

{examples}
Пиши в такой же манере: своими словами, с обстоятельствами дела, как пишет
живой человек, а не как написано в самом акте.
"""


def load_fewshot(path: str) -> list[str]:
    """Примеры живых вопросов для подсказки генератора (E4).

    В файле лежат только те живые вопросы, которых нет в наборе запросов
    (`scripts/make_fewshot_examples.py`). Проверять это здесь нечем: набор
    на видеокарту не едет целиком, и условие держится тем, что файл собран
    локально одним скриптом вместе со счетом исключенных.
    """
    import json

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row["text"] for row in rows if row.get("text")]


def shots_for(examples: list[str], shots: int, seed: int, number: int) -> str:
    """Какие примеры показать этой подсказке.

    Примеры разные у разных фрагментов: один и тот же набор на всю выборку
    задал бы генератору один речевой режим, и выборка вышла бы однообразнее
    живых вопросов, то есть E4 проверял бы сам себя. Выбор детерминирован
    семенем и номером подсказки, иначе прогон не повторить.
    """
    rnd = random.Random(f"{seed}:{number}")
    picked = rnd.sample(examples, min(shots, len(examples)))
    return "".join(f"{i}. {text}\n" for i, text in enumerate(picked, start=1))


def body_of(chunk: dict) -> str:
    """Текст фрагмента без приписанной шапки: вопрос задаётся по содержанию."""
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def load_llm(model_path: str, dtype: str = "float16", quantization: str | None = None,
             tensor_parallel: int = 1, max_model_len: int = 4096,
             gpu_memory_utilization: float = 0.90, prefix_caching: bool = False,
             max_num_seqs: int = 0, seed: int = 0, enforce_eager: bool = False):
    """Поднять vLLM.

    Тип данных задаётся явно: на Turing нет bf16, и «auto» у модели,
    обученной в bfloat16, приводит либо к падению, либо к молчаливой
    деградации — а деградацию видно только по качеству вопросов, то есть
    когда квота уже потрачена.
    """
    # vLLM поднимает движок отдельным процессом и по умолчанию делает это
    # через fork. Форк не годится, если CUDA уже тронута в родителе, —
    # а она тронута всегда: `pick_device()` спрашивает `torch.cuda
    # .is_available()` первой же строкой, до всякой генерации. Падает это
    # не там, где причина: «Cannot re-initialize CUDA in forked subprocess»
    # прилетает из недр движка через полминуты после старта. Переменная
    # ставится до импорта vllm — после него её уже не читают.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM
    kwargs = dict(model=model_path, dtype=dtype, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization,
                  tensor_parallel_size=tensor_parallel, trust_remote_code=True)
    # Кэш общего префикса окупается там, где у всех подсказок одно начало:
    # у судьи это инструкция разметчику. Генерации он не нужен, там общего
    # начала почти нет, а память под кэш отнимается у ключей.
    if prefix_caching:
        kwargs["enable_prefix_caching"] = True
    if quantization:
        kwargs["quantization"] = quantization
    # Сколько подсказок движок держит в работе разом. На судье в тридцать два
    # миллиарда параметров и двух картах кэш ключей тесный, и значение по
    # умолчанию (256) движок урезал бы сам, но молча; здесь он урезан явно
    # и попадает в метаданные прогона.
    if max_num_seqs:
        kwargs["max_num_seqs"] = max_num_seqs
    # Графы CUDA на карте 8 ГБ съедают около гигабайта, которого там нет: на
    # RTX 3070 модель в четырех битах вместе с активациями оставляет под кэш
    # ключей меньше половины гигабайта. На T4 Kaggle это не нужно.
    if enforce_eager:
        kwargs["enforce_eager"] = True
    # Семя закрепляется затем, что воспроизводимость разметки это одно из трех
    # чисел ее приемки. При нулевой температуре оно почти ничего не меняет,
    # но «почти» в метаданные не напишешь.
    if seed:
        kwargs["seed"] = seed
    print(f"поднимаю vLLM: {model_path} ({dtype}"
          + (f", {quantization}" if quantization else "")
          + f", карт {tensor_parallel}, память {gpu_memory_utilization})", flush=True)
    return LLM(**kwargs)


def _clean(text: str) -> str:
    text = re.sub(r"^[\s\"'«»\-–—]*", "", text)
    return text.split("\n")[0].strip(" \"'«»")


def _chat(tok, задание: str, system: str = SYSTEM, thinking: bool = False) -> str:
    """Собрать запрос по шаблону модели, пережив отсутствие системной роли.

    Наставление передается параметром, а не берется из константы модуля.
    Пока его не было, судья вызывал `_chat` и получал системное сообщение
    генератора вопросов («помогаешь составить набор проверочных вопросов»),
    а своего собственного не получал вовсе: вся его инструкция доезжала
    только внутри задания. Разметка от этого не рассыпалась, но системное
    сообщение судьи в подсказке отсутствовало, и назвать прогон
    воспроизводимым по коду было нельзя.

    Шаблоны разных семейств несовместимы, и это не мелочь оформления.
    Qwen принимает системную роль, а Mistral-7B-Instruct требует строгого
    чередования user/assistant и на системное сообщение отвечает
    `TemplateError: Conversation roles must alternate`. Модель другого
    семейства для теста обязательна — иначе тест померит, насколько ученик
    выучил стиль своего же генератора, — поэтому подстраиваться приходится
    здесь, а не выбором модели.

    Запасной путь не выбрасывает наставление, а приклеивает его к вопросу:
    выбросить — значит менять задачу, и вопросы Mistral оказались бы
    несопоставимы с вопросами Qwen.
    """
    # Рассуждение вслух выключено. Это не вкус: Qwen3 по умолчанию пишет
    # <think>, и на карте 8 ГБ разметка шла 0,5 пары/с против 5,6 без него, а
    # ответ при этом чаще не разбирался вовсе (16 из 150 на пробе 22.10).
    # Шаблоны, которые про рассуждение не знают, этот ключ не принимают, и
    # тогда он просто не передается.
    for kwargs in ({"enable_thinking": thinking}, {}):
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": задание}],
                tokenize=False, add_generation_prompt=True, **kwargs)
        except TypeError:
            continue
        except Exception:
            break
    return tok.apply_chat_template(
        [{"role": "user", "content": system + "\n\n" + задание}],
        tokenize=False, add_generation_prompt=True)


def generate(llm, style: Style, passages: list[str], temperature: float = 0.7,
             examples: list[str] | None = None, shots: int = 3,
             seed: int = 13, offset: int = 0) -> list[str]:
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    if examples and style.fewshot and shots:
        prompts = [
            _chat(tok, HEAD.format(passage=p)
                  + FEWSHOT.format(examples=shots_for(examples, shots, seed, offset + i))
                  + style.tail)
            for i, p in enumerate(passages)
        ]
    else:
        prompts = [_chat(tok, HEAD.format(passage=p) + style.tail) for p in passages]
    params = SamplingParams(temperature=temperature, top_p=0.9,
                            max_tokens=style.max_tokens)
    out = llm.generate(prompts, params, use_tqdm=False)
    return [_clean(o.outputs[0].text) for o in out]


def probe_speed(llm, chunks: list[dict], sample: int = 100, seed: int = 13) -> dict:
    """Замер скорости: сколько вопросов в секунду и во что это выльется.

    Ставится до всякой генерации. Без него объём обучающей выборки
    назначается на глаз, и узнать, что он не помещается в сессию, можно
    только через несколько часов.
    """
    rnd = random.Random(seed)
    picked = rnd.sample(chunks, min(sample, len(chunks)))
    stats = {"sample": len(picked), "по стилям": {}}
    total = 0.0
    for style in STYLES:
        t0 = time.time()
        got = generate(llm, style, [body_of(c)[:4000] for c in picked])
        dt = time.time() - t0
        total += dt
        stats["по стилям"][style.name] = {
            "секунд": round(dt, 1),
            "вопросов в секунду": round(len(got) / dt, 2),
            "средняя длина": round(sum(len(g) for g in got) / max(1, len(got))),
        }
        print(f"   [{style.name}] {len(got)} за {dt:.0f} с "
              f"= {len(got) / dt:.2f} вопр./с", flush=True)
    rate = len(picked) * len(STYLES) / total
    stats["вопросов в секунду"] = round(rate, 2)
    stats["часов на 10000"] = round(10000 / rate / 3600, 2)
    stats["часов на 15000"] = round(15000 / rate / 3600, 2)
    print(f"\nитого {rate:.2f} вопр./с (до фильтров): "
          f"10 000 за {stats['часов на 10000']:.1f} ч, "
          f"15 000 за {stats['часов на 15000']:.1f} ч", flush=True)
    print("объём обучающей выборки назначается по этому числу, а не по плану",
          flush=True)
    return stats


def make_queries(llm, chunks: list[dict], out_path: str, target: int,
                 prefix: str = "syn", per_act: int = 12, seed: int = 13,
                 batch: int = 256, idf_texts: list[str] | None = None,
                 examples: list[str] | None = None, shots: int = 3) -> dict:
    """Сгенерировать и отфильтровать вопросы. Возвращает статистику.

    Обратная частота слов считается по всему корпусу, а не по обучающей
    части: фильтр «подходит к слишком многим фрагментам» должен мерить
    общность слова в языке актов, а не в отобранной половине.
    """
    idf = IdfTable(idf_texts if idf_texts is not None else [c["text"] for c in chunks])
    good = [c for c in chunks if is_good_source(c)]
    random.Random(seed).shuffle(good)
    print(f"фрагментов дано {len(chunks)}, пригодных как источник {len(good)} "
          f"({100 * len(good) / max(1, len(chunks)):.0f}%)", flush=True)

    out: list[dict] = []
    rejected: collections.Counter = collections.Counter()
    per_act_count: collections.Counter = collections.Counter()
    by_style: collections.Counter = collections.Counter()
    t0 = time.time()
    pos = 0
    while len(out) < target and pos < len(good):
        batch_chunks: list[dict] = []
        while len(batch_chunks) < batch and pos < len(good):
            ch = good[pos]
            pos += 1
            if per_act_count[ch["act_id"]] >= per_act:
                continue
            batch_chunks.append(ch)
        if not batch_chunks:
            break
        style = STYLES[next_style(by_style)]
        questions = generate(llm, style, [body_of(c)[:4000] for c in batch_chunks],
                             examples=examples, shots=shots, seed=seed,
                             offset=pos - len(batch_chunks))
        for ch, q in zip(batch_chunks, questions):
            passage = body_of(ch)
            ok, why = judge_query(q, passage, idf,
                                  require_question_mark=style.question_mark)
            if not ok:
                rejected[why.split("(")[0].strip()] += 1
                continue
            run, share = copy_score(q, passage)
            per_act_count[ch["act_id"]] += 1
            by_style[style.name] += 1
            out.append({
                "query_id": f"{prefix}{len(out):05d}", "origin": "синтетический",
                "style": style.name, "text": q,
                "gold_chunk_id": ch["chunk_id"], "act_id": ch["act_id"],
                "number": ch["number"], "date": ch["date"], "act_title": ch["title"],
                "section": ch["section"], "units": ch["units"],
                "mean_idf": round(idf.mean_idf(q), 2),
                "copy_run": run, "copy_share": round(share, 2),
            })
            if len(out) >= target:
                break
        print(f"   принято {len(out)}/{target}, отклонено {sum(rejected.values())}, "
              f"{time.time() - t0:.0f} с", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    attempts = len(out) + sum(rejected.values())
    stats = {"accepted": len(out), "rejected": sum(rejected.values()),
             "reject_share": round(sum(rejected.values()) / max(1, attempts), 2),
             "reasons": dict(rejected.most_common()), "by_style": dict(by_style),
             "acts_touched": len(per_act_count), "seconds": round(time.time() - t0, 1),
             # Чем этот прогон отличается от прежнего: с примерами живых
             # вопросов или без. Иначе две выборки в отчете не различить.
             "примеров в подсказке": shots if examples else 0,
             "примеров доступно": len(examples) if examples else 0}
    print(f"\nпринято {stats['accepted']}, отклонено {stats['rejected']} "
          f"({100 * stats['reject_share']:.0f}% попыток)", flush=True)
    print(f"причины отказа: {stats['reasons']}", flush=True)
    print(f"по стилям: {stats['by_style']}", flush=True)
    print(f"актов затронуто: {stats['acts_touched']}, "
          f"время {stats['seconds'] / 60:.0f} мин", flush=True)
    return stats


def next_style(by_style: collections.Counter,
               styles: tuple[Style, ...] = STYLES) -> int:
    """Следующий стиль — тот, которого принято меньше всего.

    Не по кругу: стили отсеиваются фильтрами по-разному (короткий запрос
    чаще ловится на «подходит к слишком многим»), и ровное чередование
    пачек дало бы перекос в принятом.
    """
    counts = [by_style[s.name] for s in styles]
    return counts.index(min(counts))


def filter_by_teacher(queries: list[dict], chunks: list[dict], depth: int = 100,
                      keep_top: int | None = None, batch_size: int = 64,
                      device: str = "cuda") -> tuple[list[dict], dict]:
    """Третий фильтр: учитель обязан видеть исходный фрагмент в первых
    `keep_top` из `depth` кандидатов BM25.

    **Порог считается от глубины, а не задан числом, и это не мелочь.**
    Эталон всегда среди кандидатов: если BM25 его не нашёл, его вставляют
    силой. Значит после переранжирования его место — число от 1 до `depth`.
    При жёстком пороге 50 и глубине 30 условие «место не хуже 50-го»
    выполняется всегда, фильтр не отбрасывает ничего и стоит при этом час
    с лишним видеокарты, а отчёт показывает бодрое «принято 12000 из 12000».
    Ровно это и было заложено в план урезания: глубину предлагалось снизить
    со 100 до 30, оставив порог 50, то есть выключить фильтр, думая,
    что он лишь подешевел.

    Поэтому по умолчанию порог — половина глубины, как было при исходных
    100 и 50. Отношение и есть смысл: эталон обязан попасть в лучшую
    половину того, что показали учителю.

    Зачем он нужен помимо двух прежних. Первые два смотрят на пару
    «вопрос и фрагмент» и ловят списывание и общие слова. Они пропускают
    вопрос, на который в корпусе лучше отвечает другой фрагмент: такой
    вопрос формально хорош, а как обучающий пример он вреден — модель
    учится притягивать к нему не тот текст.

    Цена честно считается заранее: `len(queries) * depth` проходов
    кросс-энкодера. На пятнадцати тысячах вопросов при глубине 100 это
    полтора миллиона пар, то есть часы. Глубину имеет смысл уменьшать,
    а не отказываться от фильтра.
    """
    import gpu_common as common
    import gpu_search as S

    from rufin.retrieval.bm25 import BM25Index

    if keep_top is None:
        keep_top = max(1, depth // 2)
    if keep_top >= depth:
        raise ValueError(
            f"порог {keep_top} не меньше глубины {depth}: эталон всегда "
            f"попадает в кандидатов, поэтому такой фильтр не отбрасывает "
            f"ничего и только тратит время. Уменьшите порог или уберите "
            f"--teacher-filter совсем.")
    print(f"третий фильтр: глубина {depth}, порог {keep_top} "
          f"(эталон обязан попасть в первые {keep_top} из {depth})", flush=True)

    # Веса учителя из входов, а не из хаба: см. тот же разбор в фильтре пары.
    учитель = common.resolve_model(S.RERANKER)

    # Вопросы, назвавшие нормы словами подсказки, выбрасываются здесь тоже,
    # а не только при генерации: выборка могла быть сделана прежней версией
    # кода, а переделывать пять часов карты из-за четырех вопросов незачем.
    каркас = [q for q in queries if упоминает_каркас(q["text"])]
    if каркас:
        queries = [q for q in queries if not упоминает_каркас(q["text"])]
        print(f"выброшено как назвавших нормы словами подсказки: {len(каркас)}",
              flush=True)

    texts = {c["chunk_id"]: c["text"] for c in chunks}
    t0 = time.time()
    index = BM25Index.build([c["chunk_id"] for c in chunks],
                            [c["text"] for c in chunks])
    print(f"BM25 для фильтра построен за {time.time() - t0:.0f} с", flush=True)
    print(f"проходов кросс-энкодера: {len(queries) * depth}", flush=True)

    candidates: dict[str, list[tuple[str, float]]] = {}
    for q in queries:
        found = [c for c, _ in index.search(q["text"], depth)]
        if q["gold_chunk_id"] not in found:
            found = found[:depth - 1] + [q["gold_chunk_id"]]
        candidates[q["query_id"]] = [(c, 0.0) for c in found]

    ranked = S.rerank_runs([{"query_id": q["query_id"], "text": q["text"]} for q in queries],
                           candidates, texts, model_path=учитель,
                           batch_size=batch_size, device=device)

    kept, positions = [], []
    for q in queries:
        order = ranked.get(q["query_id"], [])
        place = order.index(q["gold_chunk_id"]) + 1 if q["gold_chunk_id"] in order else 0
        if place and place <= keep_top:
            q = dict(q, teacher_rank=place)
            kept.append(q)
            positions.append(place)
    stats = {"на входе": len(queries), "принято": len(kept),
             "отброшено": len(queries) - len(kept),
             "глубина": depth, "порог": keep_top,
             "медиана места эталона": sorted(positions)[len(positions) // 2] if positions else 0}
    print(f"третий фильтр: принято {stats['принято']} из {stats['на входе']}, "
          f"медиана места эталона {stats['медиана места эталона']}", flush=True)
    return kept, stats


# ---- E5: вопрос по двум связанным нормам ----
#
# Все 6 112 вопросов прошлых прогонов собраны по одной инструкции: «ответ
# содержится именно в этом фрагменте». Живой вопрос требует собрать норму
# из частей, и E5 меняет ровно это (S26 части Б). Пары готовятся локально
# (`scripts/make_pairs_e5.py`), здесь по паре пишется вопрос.
#
# Стилей два, а не три: короткий поисковый запрос из режима пары исключен.
# Запросом в три слова ситуацию с двумя нормами не задашь, и такой вопрос
# неизбежно оказался бы однофрагментным, то есть мерил бы не то, что
# проверяет опыт. Примеры живых вопросов (E4) остаются: манера оказалась
# не причиной, но и не помехой.

PAIR_HEAD = """Ниже два фрагмента нормативных актов Банка России. Это две разные
нормы ({откуда}).

--- норма 1 ---
{first}
--- конец нормы 1 ---

--- норма 2 ---
{second}
--- конец нормы 2 ---
"""

PAIR_COMMON = """
Требования:
- чтобы ответить на вопрос, нужны ОБЕ нормы; вопрос, на который хватает
  одной из них, не годится;
- вопрос по-русски, целиком; латиница допустима только в обозначениях
  вроде USD, IFRS, SWIFT;
- своими словами, без дословных кусков из фрагментов длиннее трех слов
  подряд;
- не упоминай номера и названия актов, не упоминай слова «норма 1»
  и «норма 2».
"""

PAIR_STYLES = (
    Style("вопрос профессионала по двум нормам", """
Сформулируй ОДИН вопрос, который задал бы риск-аналитик, методолог или
бухгалтер банка и ответ на который собирается из обеих норм сразу.
Одно-два предложения, заканчивается знаком вопроса.
""" + PAIR_COMMON, max_tokens=160),
    Style("бытовая ситуация по двум нормам", """
Опиши ОДНУ рабочую ситуацию в двух-четырех предложениях и закончи
вопросом, ответ на который требует обеих норм. Пиши так, как пишут
в банке коллеге: без ссылок на пункты, с обычными словами.
""" + PAIR_COMMON, max_tokens=224),
)

PAIR_ORIGINS = {"внутри акта": "разные пункты одного акта",
                "между актами": "разные акты, второй назван в первом"}


def pair_prompt(tok, style: Style, pair: dict, first: dict, second: dict,
                examples: list[str] | None, shots: int, seed: int, number: int) -> str:
    head = PAIR_HEAD.format(откуда=PAIR_ORIGINS.get(pair["origin"], pair["origin"]),
                            first=body_of(first)[:2600],
                            second=body_of(second)[:2600])
    if examples and style.fewshot and shots:
        head += FEWSHOT.format(examples=shots_for(examples, shots, seed, number))
    return _chat(tok, head + style.tail)


def generate_pairs(llm, style: Style, pairs: list[dict], texts: dict[str, dict],
                   temperature: float = 0.7, examples: list[str] | None = None,
                   shots: int = 3, seed: int = 13, offset: int = 0,
                   control: bool = False) -> list[str]:
    """Вопросы по парам. `control` пишет вопрос по ОДНОМУ первому фрагменту.

    Контрольная ветка нужна одной мере. Доля многофрагментных считается так:
    учитель ставит второй фрагмент не ниже тридцатого места при первом
    в лучшей половине. Само это число ни о чем не говорит: пара отобрана
    по BM25, фрагменты тематически близки, и однофрагментный вопрос по
    первому часто вытянет второй в тот же топ сам собой. Поэтому тем же
    фильтром меряется контроль, собранный по прежней инструкции на тех же
    парах, и смысл имеет разница.
    """
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    if control:
        prompts = []
        for i, pair in enumerate(pairs):
            first = texts[pair["chunk_id"]]
            head = HEAD.format(passage=body_of(first)[:4000])
            if examples and style.fewshot and shots:
                head += FEWSHOT.format(
                    examples=shots_for(examples, shots, seed, offset + i))
            prompts.append(_chat(tok, head + style.tail))
    else:
        prompts = [pair_prompt(tok, style, pair, texts[pair["chunk_id"]],
                               texts[pair["chunk_id_2"]], examples, shots, seed,
                               offset + i)
                   for i, pair in enumerate(pairs)]
    params = SamplingParams(temperature=temperature, top_p=0.9,
                            max_tokens=style.max_tokens)
    out = llm.generate(prompts, params, use_tqdm=False)
    return [_clean(o.outputs[0].text) for o in out]


# Слова самой подсказки, протекшие в вопрос. На пробе так вышло 6 вопросов
# из 73, и 4 из них фильтры пропустили: «чтобы соответствовать требованиям
# нормы 1 и нормы 2», «требования обеих норм». `PAIR_COMMON` это запрещает
# отдельным пунктом, но запрет в подсказке держится на послушании модели,
# а проверять надо кодом. Живой человек так не спрашивает, и такой вопрос
# в обучении учит модель узнавать каркас задания, а не собирать норму.
КАРКАС = re.compile(
    r"норм[аыуе]\s*№?\s*[12]\b"
    r"|об[еи]их\s+норм|обоих\s+норм|об[ае]\s+норм|двух\s+норм"
    r"|перв(?:ая|ой|ую)\s+норм|втор(?:ая|ой|ую)\s+норм"
    r"|указанных\s+норм", re.I)


def упоминает_каркас(query: str) -> bool:
    """Назвал ли вопрос нормы словами подсказки («обеих норм», «нормы 1»)."""
    return КАРКАС.search(query) is not None


def judge_pair_query(query: str, first: str, second: str, idf: IdfTable) -> tuple[bool, str]:
    """Годится ли вопрос по паре. Фильтр «списан» применяется к обоим
    фрагментам (S26, пункт 3): списанный со второго так же сводит задачу
    к поиску подстроки, как списанный с первого."""
    ok, why = judge_query(query, first, idf, require_question_mark=True)
    if not ok:
        return False, why
    if упоминает_каркас(query):
        return False, "назвал нормы словами подсказки"
    run, share = copy_score(query, second)
    if run > MAX_COMMON_RUN or share > MAX_COPY_SHARE:
        return False, f"списан со второго фрагмента (цепочка {run} слов)"
    return True, ""


def make_pair_queries(llm, pairs: list[dict], chunks: list[dict], out_path: str,
                      target: int, prefix: str = "tr", per_act: int = 12,
                      seed: int = 13, batch: int = 256,
                      idf_texts: list[str] | None = None,
                      examples: list[str] | None = None, shots: int = 3,
                      control: bool = False) -> dict:
    """Вопросы по парам фрагментов. Возвращает статистику.

    Доля способов подбора пары печатается по принятым вопросам, а не по
    пулу пар: равные доли в постановке были, а корпус их не дает (ссылка
    на другой обучающий акт есть у 1358 пригодных фрагментов из 14 905),
    и перекос надо видеть там, где он попадает в обучение.
    """
    idf = IdfTable(idf_texts if idf_texts is not None else [c["text"] for c in chunks])
    texts = {c["chunk_id"]: c for c in chunks}
    годные = [p for p in pairs
              if p["chunk_id"] in texts and p["chunk_id_2"] in texts]
    if len(годные) != len(pairs):
        print(f"ВНИМАНИЕ: пар с неизвестными фрагментами {len(pairs) - len(годные)}: "
              f"нарезка пар и нарезка корпуса разошлись", flush=True)
    random.Random(seed).shuffle(годные)
    print(f"пар дано {len(pairs)}, пригодных {len(годные)}, цель {target}", flush=True)

    out: list[dict] = []
    rejected: collections.Counter = collections.Counter()
    per_act_count: collections.Counter = collections.Counter()
    by_style: collections.Counter = collections.Counter()
    by_origin: collections.Counter = collections.Counter()
    # Пропуск по пределу на акт считается отдельно от отказов фильтров:
    # доля отклоненных это мера качества генерации, и смешивать с ней
    # вопросы, которых фильтры не видели, значит портить обе величины.
    по_пределу = 0
    t0 = time.time()
    pos = 0
    while len(out) < target and pos < len(годные):
        batch_pairs: list[dict] = []
        while len(batch_pairs) < batch and pos < len(годные):
            pair = годные[pos]
            pos += 1
            if per_act_count[pair["act_id"]] >= per_act:
                continue
            batch_pairs.append(pair)
        if not batch_pairs:
            break
        # В контроле стили те же два, но инструкция прежняя, по одному
        # фрагменту: иначе сравнивались бы два изменения сразу, манера
        # и устройство задачи.
        стили = STYLES[:2] if control else PAIR_STYLES
        style = стили[next_style(by_style, стили)]
        questions = generate_pairs(llm, style, batch_pairs, texts, examples=examples,
                                  shots=shots, seed=seed, offset=pos - len(batch_pairs),
                                  control=control)
        for pair, q in zip(batch_pairs, questions):
            # Предел проверяется и здесь, а не только при наборе пачки. Пачка
            # это 256 пар, счетчик растет после приемки, и акт, у которого
            # в одной пачке лежало больше двенадцати пар, забирал их все.
            if per_act_count[pair["act_id"]] >= per_act:
                по_пределу += 1
                continue
            first, second = texts[pair["chunk_id"]], texts[pair["chunk_id_2"]]
            # В контроле фильтр «списан» применяется только к первому
            # фрагменту: вопрос писался по нему, и требовать того же
            # от второго значило бы отбирать контроль строже опыта.
            if control:
                ok, why = judge_query(q, body_of(first), idf,
                                      require_question_mark=style.question_mark)
            else:
                ok, why = judge_pair_query(q, body_of(first), body_of(second), idf)
            if not ok:
                rejected[why.split("(")[0].strip()] += 1
                continue
            run, share = copy_score(q, body_of(first))
            run2, _ = copy_score(q, body_of(second))
            per_act_count[pair["act_id"]] += 1
            by_style[style.name] += 1
            by_origin[pair["origin"]] += 1
            out.append({
                "query_id": f"{prefix}{len(out):05d}", "origin": "синтетический",
                "style": style.name, "text": q,
                "gold_chunk_id": pair["chunk_id"],
                "gold_chunk_id_2": pair["chunk_id_2"],
                "pair_origin": pair["origin"], "pair_id": pair["pair_id"],
                "pair_bm25": pair["bm25"],
                "act_id": pair["act_id"], "act_id_2": pair["act_id_2"],
                "number": first["number"], "date": first["date"],
                "act_title": first["title"], "section": first["section"],
                "units": pair["units"], "units_2": pair["units_2"],
                "mean_idf": round(idf.mean_idf(q), 2),
                "copy_run": run, "copy_share": round(share, 2),
                "copy_run_2": run2,
            })
            if len(out) >= target:
                break
        print(f"   принято {len(out)}/{target}, отклонено {sum(rejected.values())}, "
              f"{time.time() - t0:.0f} с", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    attempts = len(out) + sum(rejected.values())
    # Медиана, а не верхний средний элемент: на четном числе вопросов
    # `длины[len(длины) // 2]` давал 201 там, где медиана 200, и это уже
    # уехало в отчет опыта.
    длины = [len(q["text"]) for q in out]
    stats = {"accepted": len(out), "rejected": sum(rejected.values()),
             "reject_share": round(sum(rejected.values()) / max(1, attempts), 2),
             "reasons": dict(rejected.most_common()), "by_style": dict(by_style),
             "по способу пары": dict(by_origin),
             "пропущено по пределу на акт": по_пределу,
             "доля между актами": round(by_origin["между актами"] / max(1, len(out)), 3),
             "медиана длины вопроса": round(statistics.median(длины)) if длины else 0,
             "acts_touched": len(per_act_count), "seconds": round(time.time() - t0, 1),
             "пар израсходовано": pos,
             "ветка": "контроль, вопрос по одному фрагменту" if control
                      else "опыт, вопрос по двум нормам",
             "примеров в подсказке": shots if examples else 0,
             "примеров доступно": len(examples) if examples else 0}
    print(f"\nпринято {stats['accepted']}, отклонено {stats['rejected']} "
          f"({100 * stats['reject_share']:.0f}% попыток)", flush=True)
    print(f"причины отказа: {stats['reasons']}", flush=True)
    print(f"по стилям: {stats['by_style']}", flush=True)
    print(f"по способу пары: {stats['по способу пары']} "
          f"(между актами {100 * stats['доля между актами']:.0f}%)", flush=True)
    print(f"медиана длины вопроса {stats['медиана длины вопроса']} знаков, "
          f"актов затронуто {stats['acts_touched']}, "
          f"время {stats['seconds'] / 60:.0f} мин", flush=True)
    return stats


def pair_ranks_ok(место_1: int, место_2: int, keep_top: int, second_keep: int) -> bool:
    """Правило приемки пары, объявленное в S26 до прогона.

    Ноль означает, что эталона в выдаче учителя нет вовсе. Такая пара
    не принимается: место неизвестно, а не велико.
    """
    return bool(место_1 and место_2 and место_1 <= keep_top
                and место_2 <= second_keep)


def filter_pairs_by_teacher(queries: list[dict], chunks: list[dict], depth: int = 50,
                            keep_top: int | None = None, second_keep: int = 30,
                            batch_size: int = 64, device: str = "cuda",
                            ranks_path: str = "") -> tuple[list[dict], dict]:
    """Фильтр учителя для пары и мера многофрагментности разом (S26, пункты 3 и 5).

    Вопрос принимается, если первый фрагмент пары учитель ставит в лучшую
    половину кандидатов (как в прошлых прогонах), а второй не ниже
    тридцатого места. Иначе второй фрагмент к вопросу отношения не имеет,
    и пара фиктивная: вопрос однофрагментный, как во всех прошлых прогонах,
    а значит E5 мерил бы сам себя.

    Эта же проверка и есть объявленная до прогона мера доли
    многофрагментных вопросов. Ниже 0,5 на пробе прогон останавливается
    **до обучения**, а не после.

    Порог второго обязан быть строго меньше глубины. Оба эталона
    вставляются в кандидатов силой, поэтому их место всегда не больше
    глубины: при `second_keep >= depth` условие по второму фрагменту
    выполняется у любой пары, фильтр не отбрасывает ничего и показывает
    долю многофрагментных, равную единице.
    """
    import gpu_common as common
    import gpu_search as S

    from rufin.retrieval.bm25 import BM25Index

    if keep_top is None:
        keep_top = max(1, depth // 2)
    if keep_top >= depth or second_keep >= depth:
        raise ValueError(
            f"порог первого {keep_top} или второго {second_keep} не меньше "
            f"глубины {depth}: оба эталона попадают в кандидатов силой, "
            f"поэтому такой фильтр не отбрасывает ничего, а доля "
            f"многофрагментных выходит единицей. Увеличьте глубину.")
    print(f"фильтр пары: глубина {depth}, первый в первых {keep_top}, "
          f"второй в первых {second_keep}", flush=True)

    # Веса учителя берутся из входов, а не из хаба. Ядру, созданному
    # командой, токен HuggingFace прицепить нельзя, и без токена хаб не
    # падает, а молча режет скорость; при выключенном интернете он падает
    # уже на загрузке, после того как BM25 построен и кандидаты посчитаны.
    # На этом прогон фильтра пробы E5 и встал. Пороги проверяются раньше
    # весов: отказ по порогу должен стоить секунду, а не монтирование модели.
    учитель = common.resolve_model(S.RERANKER)

    # Вопросы, назвавшие нормы словами подсказки, выбрасываются здесь тоже,
    # а не только при генерации: выборка могла быть сделана прежней версией
    # кода, а переделывать пять часов карты из-за четырех вопросов незачем.
    каркас = [q for q in queries if упоминает_каркас(q["text"])]
    if каркас:
        queries = [q for q in queries if not упоминает_каркас(q["text"])]
        print(f"выброшено как назвавших нормы словами подсказки: {len(каркас)}",
              flush=True)

    texts = {c["chunk_id"]: c["text"] for c in chunks}
    t0 = time.time()
    index = BM25Index.build([c["chunk_id"] for c in chunks],
                            [c["text"] for c in chunks])
    print(f"BM25 для фильтра построен за {time.time() - t0:.0f} с", flush=True)
    print(f"проходов кросс-энкодера: {len(queries) * depth}", flush=True)

    candidates: dict[str, list[tuple[str, float]]] = {}
    for q in queries:
        found = [c for c, _ in index.search(q["text"], depth)]
        эталоны = [q["gold_chunk_id"], q["gold_chunk_id_2"]]
        # Эталоны сначала вынимаются из выдачи, потом дописываются в конец.
        # Срез `found[:depth - len(нехватка)]` выбрасывал тот эталон, который
        # BM25 нашел сам и поставил в хвост: место выходило нулем, и пара
        # отклонялась не по сути, а по длине списка. Длина остается ровно
        # `depth`, иначе места эталонов считались бы по спискам разной
        # глубины у разных вопросов.
        прочие = [c for c in found if c not in эталоны]
        found = прочие[:max(0, depth - len(эталоны))] + эталоны
        candidates[q["query_id"]] = [(c, 0.0) for c in found]

    ranked = S.rerank_runs([{"query_id": q["query_id"], "text": q["text"]} for q in queries],
                           candidates, texts, model_path=учитель,
                           batch_size=batch_size, device=device)

    kept: list[dict] = []
    места_1: list[int] = []
    места_2: list[int] = []
    первый_в_половине = 0
    for q in queries:
        order = ranked.get(q["query_id"], [])
        место_1 = order.index(q["gold_chunk_id"]) + 1 if q["gold_chunk_id"] in order else 0
        место_2 = order.index(q["gold_chunk_id_2"]) + 1 if q["gold_chunk_id_2"] in order else 0
        if место_1:
            места_1.append(место_1)
        if место_2:
            места_2.append(место_2)
        if место_1 and место_1 <= keep_top:
            первый_в_половине += 1
        if pair_ranks_ok(место_1, место_2, keep_top, second_keep):
            kept.append(dict(q, teacher_rank=место_1, teacher_rank_2=место_2))

    def медиана(значения: list[int]) -> int:
        return sorted(значения)[len(значения) // 2] if значения else 0

    # Места всех поданных вопросов, включая отклоненных, ложатся файлом рядом.
    # Без него в отчете остаются только медианы, а они по опубликованным
    # файлам не пересчитываются: у принятых место есть в самой выборке,
    # у отклоненных не остается нигде.
    if ranks_path:
        os.makedirs(os.path.dirname(ranks_path) or ".", exist_ok=True)
        with open(ranks_path, "w", encoding="utf-8") as f:
            for q in queries:
                order = ranked.get(q["query_id"], [])
                f.write(json.dumps({
                    "query_id": q["query_id"],
                    "место первого": order.index(q["gold_chunk_id"]) + 1
                                     if q["gold_chunk_id"] in order else 0,
                    "место второго": order.index(q["gold_chunk_id_2"]) + 1
                                      if q["gold_chunk_id_2"] in order else 0,
                }, ensure_ascii=False) + "\n")
        print(f"места эталонов всех поданных: {ranks_path}", flush=True)

    доля = len(kept) / max(1, len(queries))
    stats = {"на входе": len(queries), "принято": len(kept),
             "выброшено как каркас подсказки": len(каркас),
             "отброшено": len(queries) - len(kept),
             "глубина": depth, "порог первого": keep_top, "порог второго": second_keep,
             "первый в лучшей половине": первый_в_половине,
             "медиана места первого": медиана(места_1),
             "медиана места второго": медиана(места_2),
             # Мера объявлена в S26 до прогона: доля вопросов, у которых
             # второй фрагмент выше тридцатого места при первом в лучшей
             # половине. Знаменатель это все поданные вопросы.
             "доля многофрагментных": round(доля, 3),
             "доля многофрагментных при первом в половине":
                 round(len(kept) / max(1, первый_в_половине), 3)}
    print(f"фильтр пары: принято {len(kept)} из {len(queries)}, "
          f"места эталонов (медианы) {stats['медиана места первого']} "
          f"и {stats['медиана места второго']}", flush=True)
    print(f"доля многофрагментных {stats['доля многофрагментных']:.3f} "
          f"({len(kept)} из {len(queries)}), порог остановки 0,5", flush=True)
    if доля < 0.5:
        print("НИЖЕ 0,5: генератор инструкцию не выполняет, прогон "
              "останавливается до обучения (S26, пункт 5)", flush=True)
    return kept, stats
