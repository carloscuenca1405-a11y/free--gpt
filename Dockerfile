FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend/main.py ./
COPY frontend/index.html /app/frontend/index.html

ENV PATH="/app/backend/.venv/bin:$PATH"
ENV PORT=7860
EXPOSE 7860
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
