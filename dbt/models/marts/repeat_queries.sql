-- Повторы запросов по суткам: сколько текстов спросили впервые, сколько
-- спросили не впервые, и сколько проходит между двумя одинаковыми запросами.
--
-- Окно здесь единственный способ посчитать: место запроса в
-- ряду повторов своего текста (`occurrence`) и время с прошлого такого же
-- запроса считаются до любой группировки, по строке журнала. Группировка по
-- тексту сразу вернула бы одну строку на текст и сам запрос потеряла бы,
-- соединение таблицы с собой по условию «раньше по времени» стоило бы квадрат
-- числа строк, а знать «этот запрос новый или его уже задавали» нужно именно
-- про каждый запрос.
--
-- Порядок внутри текста задан парой `received_at, request_id`: на одинаковом
-- времени нумерация строк иначе зависела бы от порядка чтения, и два режима
-- хранения разошлись бы на ровном месте.
--
-- Сворачивается это до суток: мартам нужен ряд по суткам, а строка на запрос
-- живет слоем ниже, в представлении, поверх которого считается свертка.

with ranked as (
    select
        day,
        query_normalized,
        row_number() over text_in_time as occurrence,
        if(
            row_number() over text_in_time = 1,
            null,
            dateDiff(
                'second',
                parseDateTimeBestEffort(lagInFrame(received_at) over text_in_time),
                parseDateTimeBestEffort(received_at)
            )
        ) as seconds_since_previous
    from {{ ref('stg_queries') }}
    where outcome = 'ok'
    window text_in_time as (
        partition by query_normalized
        order by received_at asc, request_id asc
        rows between unbounded preceding and current row
    )
)
select
    day,
    count() as answered,
    countIf(occurrence = 1) as first_time,
    countIf(occurrence > 1) as repeated,
    uniqExact(query_normalized) as distinct_texts,
    round(quantileExactIf(0.5)(seconds_since_previous, occurrence > 1), 1) as median_gap_seconds
from ranked
group by day
order by day
