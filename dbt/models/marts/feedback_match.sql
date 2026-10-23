-- Соединение оценок с запросами по `request_id`. Оценка без своего запроса это
-- не мусор, который можно выбросить молча: процесс сервиса не проверяет
-- существование идентификатора (api.py), и такая строка приходит и от опечатки
-- клиента, и от оценки запроса, чьи сутки уже уехали в другой файл. Поэтому
-- сирота считается числом рядом с общим числом оценок.
--
-- `ifNull` вокруг признака совпадения стоит затем, чтобы ответ не зависел от
-- настройки `join_use_nulls`: при одном ее значении несовпавшая строка дает 0,
-- при другом NULL, и `known = 0` во втором случае тихо потерял бы всех сирот.

select
    count() as feedback_events,
    countIf(is_matched = 1) as matched,
    countIf(is_matched = 0) as orphans,
    countIf(rating = 'up') as up,
    countIf(rating = 'down') as down
from (
    select
        f.rating as rating,
        ifNull(q.known, 0) as is_matched
    from (
        select request_id, rating
        from {{ ref('stg_feedback') }}
    ) as f
    left join (
        select distinct request_id, 1 as known
        from {{ ref('stg_queries') }}
    ) as q using (request_id)
)
