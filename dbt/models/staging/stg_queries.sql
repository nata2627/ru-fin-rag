-- Событие запроса в том виде, в каком его читает витрина.
--
-- Сутки берутся из начала `received_at` строкой: файл журнала уже нарезан по
-- суткам UTC (events.py), и разбор в дату с часовым поясом добавил бы ошибку
-- там, где ее нет.
--
-- `timings_ms` остается картой: состав ступеней зависит от связки поиска (у
-- bm25 нет плотной ступени, у hybrid-rerank есть переранжирование), и колонка
-- под каждую ступень половину времени стояла бы пустой. Рядом вынесена
-- ступень `total`: по ней считается задержка ответа, и доставать ее из карты в
-- каждой модели ниже значило бы повторять одно выражение пять раз.

select
    request_id,
    received_at,
    substring(received_at, 1, 10) as day,
    outcome,
    status,
    config,
    config_version,
    corpus_version,
    query_normalized,
    query_chars,
    hits,
    timings_ms,
    timings_ms['total'] as total_ms
from {{ source('journal', 'query_events') }}
where kind = 'query'
