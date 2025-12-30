FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 180 app:app"]
