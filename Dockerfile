FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY projects/ projects/

RUN mkdir -p /data/output

ENV STORAGE_PATH=/data/output
ENV PROJECTS_PATH=/app/projects

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
