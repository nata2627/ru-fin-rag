-- Оценка без своего запроса обязана быть посчитана, а не потеряна.
--
-- Ломалось бы молча: соединение оценок с запросами это `left join`, и при
-- одной настройке сервера несовпавшая строка дает признак 0, а при другой
-- NULL. Во втором случае сирота не попала бы ни в `matched`, ни в `orphans`,
-- сумма перестала бы сходиться с общим числом оценок, и доля оценок выглядела
-- бы лучше, чем она есть. Тест сверяет арифметику модели с прямым счетом
-- сирот, который соединения не использует вовсе.

with model as (
    select feedback_events, matched, orphans
    from {{ ref('feedback_match') }}
),

direct as (
    select
        count() as feedback_events,
        countIf(request_id in (select request_id from {{ ref('stg_queries') }})) as matched,
        countIf(request_id not in (select request_id from {{ ref('stg_queries') }})) as orphans
    from {{ ref('stg_feedback') }}
)

select
    model.feedback_events as model_feedback_events,
    model.matched as model_matched,
    model.orphans as model_orphans,
    direct.feedback_events as direct_feedback_events,
    direct.matched as direct_matched,
    direct.orphans as direct_orphans
from model
cross join direct
where model.matched + model.orphans != model.feedback_events
    or model.matched != direct.matched
    or model.orphans != direct.orphans
    or model.feedback_events != direct.feedback_events
