-- Оценка ответа. Своих суток у оценки две: сутки самой оценки и сутки
-- запроса, который оценили. Здесь лежат сутки оценки, а модели качества
-- считают по суткам запроса, иначе оценка, поставленная после полуночи,
-- уехала бы в чужой день.

select
    request_id,
    received_at,
    substring(received_at, 1, 10) as day,
    rating
from {{ source('journal', 'query_events') }}
where kind = 'feedback'
