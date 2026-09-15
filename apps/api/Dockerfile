FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY agentguard ./agentguard
COPY apps/api ./apps/api

RUN pip install --no-cache-dir . fastapi uvicorn[standard]

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
