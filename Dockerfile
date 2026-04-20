FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY reports ./reports
COPY mlruns ./mlruns

RUN pip install --no-cache-dir -e .

FROM base AS train
CMD ["python", "-m", "demand_forecast.run_pipeline"]

FROM base AS serve
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000"]

