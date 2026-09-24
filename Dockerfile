FROM python:3.13-slim

LABEL qfbench2.interface_version="2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
COPY LICENSE /app/LICENSE
COPY forecast /usr/local/bin/forecast
RUN chmod +x /usr/local/bin/forecast

CMD ["forecast", "--help"]
