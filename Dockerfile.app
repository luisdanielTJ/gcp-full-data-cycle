FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY adapters/ adapters/
COPY app/ app/
COPY start.sh ./
RUN chmod +x start.sh

ENV PYTHONPATH=/app

CMD ["./start.sh"]
