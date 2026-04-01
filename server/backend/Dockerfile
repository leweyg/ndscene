FROM python:3.10-slim

WORKDIR /app

COPY server/backend/requirements.txt /app/server/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/server/backend/requirements.txt

COPY py /app/py
COPY json /app/json
COPY server/backend /app/server/backend

ENV PYTHONPATH=/app/py
ENV NDSCENE_DB_PATH=/app/py/examples/freed_go.sqlite

WORKDIR /app/server/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
