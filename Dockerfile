# Образ без torch: связка по умолчанию в онлайне плотную часть считает моделью
# запроса, и чем ее заменить на бесплатном хостинге, решается в задаче 2.4.
# Пока образ поднимает лексический поиск, он же режим деградации.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY scripts ./scripts

ENV RUFINRAG_DISABLE_DENSE=1
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "rufinrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
