FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "gunicorn --preload -w 1 --threads 1 --timeout 120 -b 0.0.0.0:$PORT app:app"]
