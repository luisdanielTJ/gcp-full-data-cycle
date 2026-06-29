FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY adapters/ adapters/

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-c", "print('crypto-edge foundation ready')"]
