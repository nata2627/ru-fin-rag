-- Сколько отвеченных запросов получили хотя бы одну оценку. Числитель и
-- знаменатель стоят рядом числами (AGENTS.md, раздел 2), долю считает тот, кто
-- печатает текст.
--
-- Оценок на один запрос бывает несколько, поэтому оценки сначала сворачиваются
-- по `request_id`, и только потом считается запрос с оценкой: иначе соединение
-- размножило бы сам запрос и знаменатель вырос бы на повторных нажатиях.

select
    day,
    count() as answered,
    countIf(ratings > 0) as rated
from (
    select day, request_id
    from {{ ref('stg_queries') }}
    where outcome = 'ok'
) as q
left join (
    select request_id, count() as ratings
    from {{ ref('stg_feedback') }}
    group by request_id
) as r using (request_id)
group by day
order by day
